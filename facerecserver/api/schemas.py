from pydantic import BaseModel
from typing import Optional


class EmbeddingRequest(BaseModel):
    image: Optional[str] = None
    image_url: Optional[str] = None


class ApiResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None


class ErrorResponse(BaseModel):
    code: int
    message: str
    data: None = None
