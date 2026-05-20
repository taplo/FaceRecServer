from pydantic import BaseModel
from typing import Optional


class GalleryAddRequest(BaseModel):
    image: Optional[str] = None
    image_url: Optional[str] = None
    name: Optional[str] = None


class GalleryAddResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None


class GalleryItem(BaseModel):
    face_id: str
    name: str
    created_at: str


class GalleryListResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None


class GalleryDeleteResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None


class RecognizeRequest(BaseModel):
    image: Optional[str] = None
    image_url: Optional[str] = None


class RecognizeItem(BaseModel):
    face_id: str
    name: str
    score: float
