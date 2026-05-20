from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from facerecserver.api.routes import router
from facerecserver.face_recognition.embedding import FaceEmbeddingExtractor
from facerecserver.config import AppConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: AppConfig = app.state.config
    try:
        extractor = FaceEmbeddingExtractor(config)
        app.state.extractor = extractor
        print(f"[启动] 模型已加载: {config.model.name} on {config.device}")
    except Exception as e:
        print(f"[警告] 模型加载失败: {e}")
        print("[提示] API 端点将返回模型未加载错误，请先下载模型")
        app.state.extractor = None
    yield
    print("[关闭] 服务停止")


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
    return app
