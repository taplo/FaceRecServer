import logging
import os
from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
import numpy as np
from PIL import Image
import io

from facerecserver.api.schemas import EmbeddingRequest, CompareRequest, ApiResponse
from facerecserver.face_recognition.embedding import FaceEmbeddingExtractor
from facerecserver.face_recognition.utils import base64_to_image
from facerecserver.face_detection.detector import FaceNotFoundError
from facerecserver.gallery.repository import GalleryRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

_MAX_FILE_SIZE = 10 * 1024 * 1024
_MAX_IMAGE_DIM = 4096


def _validate_image_bytes(data: bytes):
    if len(data) > _MAX_FILE_SIZE:
        raise ValueError(f"文件过大: {len(data) / 1024 / 1024:.1f}MB > {_MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
    img = Image.open(io.BytesIO(data))
    if max(img.size) > _MAX_IMAGE_DIM:
        raise ValueError(f"图片尺寸过大: {img.size}，最大允许 {_MAX_IMAGE_DIM}px")
    return np.array(img.convert("RGB"))


def _get_gallery_repo(request: Request) -> GalleryRepository:
    repo = getattr(request.app.state, "gallery_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "底库未初始化"})
    return repo


def _get_extractor(request: Request) -> FaceEmbeddingExtractor:
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "模型未加载"})
    return extractor


@router.post("/compare", response_model=ApiResponse)
async def compare_faces(
    request: Request,
    file1: UploadFile | None = File(None),
    file2: UploadFile | None = File(None),
    body: CompareRequest | None = None,
):
    extractor = _get_extractor(request)

    try:
        if file1 is not None and file2 is not None:
            img1 = _validate_image_bytes(await file1.read())
            img2 = _validate_image_bytes(await file2.read())
        elif body and body.image1 and body.image2:
            img1 = base64_to_image(body.image1)
            img2 = base64_to_image(body.image2)
        elif body and body.image1_url and body.image2_url:
            import requests as http_requests
            resp1 = http_requests.get(body.image1_url, timeout=30)
            resp1.raise_for_status()
            resp2 = http_requests.get(body.image2_url, timeout=30)
            resp2.raise_for_status()
            img1 = _validate_image_bytes(resp1.content)
            img2 = _validate_image_bytes(resp2.content)
        else:
            return ApiResponse(code=400, message="请提供两张图片 (file1+file2, 或 image1+image2, 或 image1_url+image2_url)", data=None)

        emb1 = extractor.extract(img1)
        emb2 = extractor.extract(img2)
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return ApiResponse(code=0, message="success", data={"similarity": sim})

    except FaceNotFoundError as e:
        return ApiResponse(code=1001, message=str(e), data=None)
    except ValueError as e:
        return ApiResponse(code=1002, message=str(e), data=None)
    except Exception as e:
        logger.exception("人脸比对失败")
        return ApiResponse(code=-1, message=f"处理失败: {str(e)}", data=None)


@router.get("/health")
async def health_check(request: Request):
    import time
    extractor = getattr(request.app.state, "extractor", None)
    repo = getattr(request.app.state, "gallery_repo", None)
    status = "ok" if extractor else "degraded"
    return {
        "status": status,
        "model_loaded": extractor is not None,
        "gallery_ready": repo is not None,
        "device": request.app.state.config.device if hasattr(request.app.state, "config") else "unknown",
        "uptime_seconds": int(time.time() - request.app.state.start_time) if hasattr(request.app.state, "start_time") else 0,
    }


@router.get("/livez")
async def liveness(request: Request):
    return {"status": "alive"}


@router.get("/readyz")
async def readiness(request: Request):
    extractor = getattr(request.app.state, "extractor", None)
    repo = getattr(request.app.state, "gallery_repo", None)
    if extractor is None or repo is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": extractor is not None, "gallery_ready": repo is not None})
    return {"status": "ready", "model_loaded": True, "gallery_ready": True}


@router.post("/embedding", response_model=ApiResponse)
async def create_embedding(
    request: Request,
    file: UploadFile | None = File(None),
    body: EmbeddingRequest | None = None,
):
    extractor = _get_extractor(request)

    try:
        if file is not None:
            contents = await file.read()
            image = _validate_image_bytes(contents)
        elif body and body.image:
            image = base64_to_image(body.image)
        elif body and body.image_url:
            import requests as http_requests
            resp = http_requests.get(body.image_url, timeout=30)
            resp.raise_for_status()
            image = _validate_image_bytes(resp.content)
        else:
            return ApiResponse(code=400, message="请提供图片 (file, image, 或 image_url)", data=None)

        embedding = extractor.extract(image)
        return ApiResponse(
            code=0,
            message="success",
            data={"embedding": embedding.tolist(), "dimension": int(embedding.shape[0])},
        )

    except FaceNotFoundError as e:
        return ApiResponse(code=1001, message=str(e), data=None)
    except ValueError as e:
        return ApiResponse(code=1002, message=str(e), data=None)
    except Exception as e:
        logger.exception("处理请求时出错")
        return ApiResponse(code=-1, message=f"处理失败: {str(e)}", data=None)


@router.get("/stats", response_model=ApiResponse)
async def get_stats(request: Request):
    import time
    repo = _get_gallery_repo(request)
    gallery_stats = repo.get_stats()
    data = {
        "gallery": gallery_stats,
        "server": {
            "uptime_seconds": int(time.time() - request.app.state.start_time) if hasattr(request.app.state, "start_time") else 0,
            "device": request.app.state.config.device,
        },
    }
    return ApiResponse(code=0, message="success", data=data)
