import io
import os
import zipfile
import tempfile
import logging
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Query

from facerecserver.api.schemas import ApiResponse
from facerecserver.gallery.repository import GalleryRepository
from facerecserver.gallery.schemas import GalleryAddRequest
from facerecserver.face_recognition.utils import base64_to_image, load_image
from facerecserver.face_detection.detector import FaceNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gallery")


def _get_repo(request: Request) -> GalleryRepository:
    repo = getattr(request.app.state, "gallery_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "底库未初始化"})
    return repo


def _get_extractor(request: Request):
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "模型未加载"})
    return extractor


@router.post("", response_model=ApiResponse)
async def add_face(
    request: Request,
    file: UploadFile | None = File(None),
    body: GalleryAddRequest | None = None,
):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    try:
        if file is not None:
            contents = await file.read()
            image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
            name = os.path.splitext(file.filename or "unknown")[0]
        elif body and body.image:
            image = base64_to_image(body.image)
            name = body.name or "unknown"
        elif body and body.image_url:
            import requests as http_requests
            resp = http_requests.get(body.image_url, timeout=30)
            resp.raise_for_status()
            image = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
            name = body.name or "unknown"
        else:
            return ApiResponse(code=400, message="请提供图片 (file, image, 或 image_url)", data=None)

        embedding = extractor.extract(image)
        face_id = repo.add(embedding, name)
        return ApiResponse(code=0, message="success", data={"face_id": face_id, "name": name})

    except FaceNotFoundError as e:
        return ApiResponse(code=1001, message=str(e), data=None)
    except ValueError as e:
        return ApiResponse(code=1002, message=str(e), data=None)
    except Exception as e:
        logger.exception("注册人脸失败")
        return ApiResponse(code=-1, message=f"处理失败: {str(e)}", data=None)


@router.post("/batch", response_model=ApiResponse)
async def add_faces_batch(request: Request, file: UploadFile = File(...)):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    contents = await file.read()
    if not contents:
        return ApiResponse(code=400, message="ZIP 文件为空", data=None)

    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            zf.extractall(temp_dir)

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []
        for root, _dirs, files in os.walk(temp_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in image_exts:
                    image_paths.append(os.path.join(root, fname))

        embeddings = []
        names = []
        errors = []
        for img_path in image_paths:
            try:
                image = load_image(img_path)
                emb = extractor.extract(image)
                embeddings.append(emb)
                names.append(os.path.splitext(os.path.basename(img_path))[0])
            except FaceNotFoundError:
                errors.append({"file": os.path.basename(img_path), "reason": "未检测到人脸"})
            except Exception as e:
                errors.append({"file": os.path.basename(img_path), "reason": str(e)})

        stats = repo.add_batch(embeddings, names)
        for err in errors:
            stats.failed += 1
            stats.failures.append(err)

        return ApiResponse(code=0, message="success", data={
            "total": stats.total + len(errors),
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "failures": stats.failures,
        })
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("", response_model=ApiResponse)
async def list_faces(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query("", max_length=50),
):
    repo = _get_repo(request)
    items, total = repo.list_faces(page, page_size, search)
    return ApiResponse(code=0, message="success", data={
        "items": [{"face_id": f.face_id, "name": f.name, "created_at": f.created_at} for f in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    })


@router.delete("/{face_id}", response_model=ApiResponse)
async def delete_face(request: Request, face_id: str):
    repo = _get_repo(request)
    if not repo.delete(face_id):
        return ApiResponse(code=2002, message="人脸不存在", data=None)
    return ApiResponse(code=0, message="success", data=None)


@router.delete("", response_model=ApiResponse)
async def clear_gallery(request: Request):
    repo = _get_repo(request)
    repo.clear()
    return ApiResponse(code=0, message="success", data=None)
