import io
import os
import zipfile
import tempfile
import logging
import shutil
import requests
import numpy as np
from PIL import Image, ImageDraw
from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Query
from fastapi.responses import FileResponse, Response

from facerecserver.api.schemas import ApiResponse
from facerecserver.gallery.repository import GalleryRepository
from facerecserver.face_recognition.utils import base64_to_image, load_image
from facerecserver.face_detection.detector import FaceNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gallery")

_MAX_FILE_SIZE = 10 * 1024 * 1024
_MAX_ZIP_SIZE = 500 * 1024 * 1024
_MAX_IMAGE_DIM = 4096


def _validate_image(data: bytes) -> np.ndarray:
    if len(data) > _MAX_FILE_SIZE:
        raise ValueError(f"文件过大: {len(data) / 1024 / 1024:.1f}MB > {_MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
    img = Image.open(io.BytesIO(data))
    if max(img.size) > _MAX_IMAGE_DIM:
        raise ValueError(f"图片尺寸过大: {img.size}，最大允许 {_MAX_IMAGE_DIM}px")
    return np.array(img.convert("RGB"))


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


def _make_image_from_bytes(data: bytes) -> np.ndarray:
    return _validate_image(data)


async def _parse_image_from_request(request: Request) -> tuple[np.ndarray, str | None, str]:
    """Parse image, optional name, and employee_id from request (multipart or JSON)."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        up_file = form.get("file")
        if up_file is None or not hasattr(up_file, "read"):
            raise ValueError("请上传图片文件")
        contents = await up_file.read()
        basename = os.path.splitext(up_file.filename or "unknown")[0]
        employee_id = ""
        name = basename
        if "-" in basename:
            parts = basename.rsplit("-", 1)
            name = parts[0]
            employee_id = parts[1]
        return _make_image_from_bytes(contents), name, employee_id
    body = await request.json()
    name = body.get("name") or ""
    employee_id = body.get("employee_id", "")
    if not employee_id and "-" in name:
        parts = name.rsplit("-", 1)
        name = parts[0]
        employee_id = parts[1]
    if body.get("image"):
        return base64_to_image(body["image"]), name, employee_id
    if body.get("image_url"):
        resp = requests.get(body["image_url"], timeout=30)
        resp.raise_for_status()
        return _make_image_from_bytes(resp.content), name, employee_id
    raise ValueError("请提供图片 (file, image, 或 image_url)")


@router.post("", response_model=ApiResponse)
async def add_face(request: Request):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    try:
        image, name, employee_id = await _parse_image_from_request(request)
        name = name or "unknown"
        embedding, face_crop = extractor.extract(image, return_face=True)
        face_id = repo.add(embedding, name, employee_id=employee_id, image=face_crop)
        return ApiResponse(code=0, message="success", data={"face_id": face_id, "name": name, "employee_id": employee_id})

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
    if len(contents) > _MAX_ZIP_SIZE:
        return ApiResponse(code=400, message=f"ZIP 文件过大: {len(contents) / 1024 / 1024:.1f}MB > {_MAX_ZIP_SIZE / 1024 / 1024:.0f}MB", data=None)

    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            for name in zf.namelist():
                dest = os.path.abspath(os.path.join(temp_dir, name))
                if not dest.startswith(os.path.abspath(temp_dir)):
                    raise ValueError(f"非法的 ZIP 路径: {name}")
            zf.extractall(temp_dir)

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_paths = []
        for root, _dirs, files in os.walk(temp_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in image_exts:
                    image_paths.append(os.path.join(root, fname))

        embeddings = []
        names = []
        employee_ids = []
        images = []
        errors = []
        for img_path in image_paths:
            try:
                image = load_image(img_path)
                emb, face_crop = extractor.extract(image, return_face=True)
                embeddings.append(emb)
                basename = os.path.splitext(os.path.basename(img_path))[0]
                if "-" in basename:
                    parts = basename.rsplit("-", 1)
                    names.append(parts[0])
                    employee_ids.append(parts[1])
                else:
                    names.append(basename)
                    employee_ids.append("")
                images.append(face_crop)
            except FaceNotFoundError:
                errors.append({"file": os.path.basename(img_path), "reason": "未检测到人脸"})
            except Exception as e:
                errors.append({"file": os.path.basename(img_path), "reason": str(e)})

        stats = repo.add_batch(embeddings, names, employee_ids=employee_ids, images=images)
        for err in errors:
            stats.failed += 1
            stats.failures.append(err)

        total_input = len(embeddings) + len(errors)
        return ApiResponse(code=0, message="success", data={
            "total": total_input,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "failures": stats.failures,
        })
    finally:
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
        "items": [{
            "face_id": f.face_id,
            "name": f.name,
            "employee_id": f.employee_id,
            "created_at": f.created_at,
            "image_url": repo.get_image_url(f.face_id),
            "person_id": f.person_id,
        } for f in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    })


@router.get("/persons", response_model=ApiResponse)
async def list_persons(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query("", max_length=50),
):
    repo = _get_repo(request)
    items, total = repo.list_persons(page, page_size, search)
    return ApiResponse(code=0, message="success", data={
        "items": [{
            "person_id": p.person_id,
            "name": p.name,
            "employee_id": p.employee_id,
            "created_at": p.created_at,
            "face_count": p.face_count,
        } for p in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    })


def _make_placeholder() -> bytes:
    img = Image.new("RGB", (120, 120), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 20, 90, 80], outline=(150, 150, 150), width=2)
    draw.arc([40, 70, 80, 130], 0, 180, fill=(150, 150, 150), width=2)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


_PLACEHOLDER: bytes | None = None


@router.get("/{face_id}/image")
async def get_face_image(request: Request, face_id: str):
    global _PLACEHOLDER
    repo = _get_repo(request)
    image_path = repo.get_image_path(face_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail={"code": 2002, "message": "图片不存在"})
    abs_path = os.path.join(repo.gallery_dir, image_path)
    if os.path.exists(abs_path):
        return FileResponse(abs_path, media_type="image/jpeg")
    if _PLACEHOLDER is None:
        _PLACEHOLDER = _make_placeholder()
    return Response(content=_PLACEHOLDER, media_type="image/jpeg")


@router.delete("/{face_id}", response_model=ApiResponse)
async def delete_face(request: Request, face_id: str):
    repo = _get_repo(request)
    if not repo.delete(face_id):
        return ApiResponse(code=2002, message="人脸不存在", data=None)
    return ApiResponse(code=0, message="success", data=None)


@router.delete("/persons/{person_id}", response_model=ApiResponse)
async def delete_person(request: Request, person_id: int):
    repo = _get_repo(request)
    if not repo.delete_person(person_id):
        return ApiResponse(code=2002, message="人员不存在", data=None)
    return ApiResponse(code=0, message="success", data=None)


@router.delete("", response_model=ApiResponse)
async def clear_gallery(request: Request):
    repo = _get_repo(request)
    repo.clear()
    return ApiResponse(code=0, message="success", data=None)


@router.post("/recognize", response_model=ApiResponse)
async def recognize_face(request: Request, top_k: int = Query(5, ge=1, le=50)):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    try:
        image, _name, _emp_id = await _parse_image_from_request(request)
        embedding = extractor.extract(image)
        results = repo.search(embedding, top_k)
        return ApiResponse(code=0, message="success", data={"results": results})

    except FaceNotFoundError as e:
        return ApiResponse(code=1001, message=str(e), data=None)
    except ValueError as e:
        return ApiResponse(code=1002, message=str(e), data=None)
    except Exception as e:
        logger.exception("人脸识别失败")
        return ApiResponse(code=-1, message=f"处理失败: {str(e)}", data=None)
