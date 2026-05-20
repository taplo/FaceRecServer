import logging
from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
import numpy as np
from PIL import Image
import io

from facerecserver.api.schemas import EmbeddingRequest, ApiResponse
from facerecserver.face_recognition.embedding import FaceEmbeddingExtractor
from facerecserver.face_recognition.utils import base64_to_image
from facerecserver.face_detection.detector import FaceNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _get_extractor(request: Request) -> FaceEmbeddingExtractor:
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "模型未加载"})
    return extractor


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
            image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        elif body and body.image:
            image = base64_to_image(body.image)
        elif body and body.image_url:
            import requests as http_requests
            resp = http_requests.get(body.image_url, timeout=30)
            resp.raise_for_status()
            image = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
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
