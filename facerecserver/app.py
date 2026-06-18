import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from facerecserver.api.routes import router
from facerecserver.gallery.routes import router as gallery_router
from facerecserver.gallery.repository import GalleryRepository
from facerecserver.face_recognition.embedding import FaceEmbeddingExtractor
from facerecserver.config import AppConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    config: AppConfig = app.state.config
    app.state.extractor = None
    app.state.gallery_repo = None
    try:
        extractor = FaceEmbeddingExtractor(config)
        app.state.extractor = extractor
        logger.info("模型已加载: %s on %s", config.model.name, config.device)
    except Exception as e:
        logger.warning("模型加载失败: %s", e)
        logger.info("API 端点将返回模型未加载错误，请先下载模型")
    try:
        repo = GalleryRepository(config.gallery.db_dir, config.gallery.db_name)
        app.state.gallery_repo = repo
    except Exception as e:
        logger.warning("底库初始化失败: %s", e)
    yield
    repo = getattr(app.state, "gallery_repo", None)
    if repo:
        repo.close()
    logger.info("服务停止")


def create_app(config: AppConfig | None = None) -> FastAPI:
    if config is None:
        from facerecserver.config import load_config
        config = load_config()

    app = FastAPI(title="FaceRecServer", version="0.1.0", lifespan=lifespan)
    app.state.config = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    app.include_router(gallery_router)

    from facerecserver.web.routes import mount_frontend
    mount_frontend(app)
    return app
