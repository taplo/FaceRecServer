# Architecture: FaceRecServer

## Project Type
基于 PETALface (WACV 2025) 算法的人脸识别 API 服务 + Vue 3 管理后台。

## File Map

| Path | Purpose |
|------|---------|
| `facerecserver/__main__.py` | CLI 入口: `python -m facerecserver` |
| `facerecserver/app.py` | FastAPI 应用工厂 + lifespan 生命周期管理 |
| `facerecserver/config.py` | YAML 配置加载器 (AppConfig dataclass) |
| `facerecserver/config.yaml` | 默认配置: 模型路径/检测参数/服务端口 |
| `facerecserver/api/routes.py` | 主路由: POST /embedding, GET /stats |
| `facerecserver/api/schemas.py` | Pydantic 模型: EmbeddingRequest, ApiResponse |
| `facerecserver/face_detection/detector.py` | MTCNN 人脸检测器 |
| `facerecserver/face_detection/aligner.py` | 5 关键点仿射变换对齐 |
| `facerecserver/face_recognition/model.py` | PETALface Swin-T + LoRALinear 定义 |
| `facerecserver/face_recognition/embedding.py` | FaceEmbeddingExtractor: 检测→对齐→质量→推理流水线 |
| `facerecserver/face_recognition/utils.py` | 图片加载/Base64/质量检查/Alpha 估计 |
| `facerecserver/gallery/repository.py` | SQLite + Faiss 存储层 |
| `facerecserver/gallery/routes.py` | Gallery CRUD + 识别路由 |
| `facerecserver/gallery/schemas.py` | Gallery Pydantic 模型 |
| `facerecserver/web/routes.py` | Vue 前端 SPA 静态文件挂载 |
| `scripts/import_gallery.py` | ZIP 批量导入脚本 |
| `scripts/download_model.py` | HuggingFace 模型下载 |
| `frontend/` | Vue 3 + TypeScript 前端 (4 页面) |

## Current State

已完成全部 4 个阶段:

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1: PETALface | ✅ 完成 | 模型加载/检测/对齐/特征提取/Embedding API |
| Phase 2: Gallery | ✅ 完成 | SQLite + Faiss 存储/CRUD/ZIP 批量导入 |
| Phase 3: Recognition | ✅ 完成 | 1:N 搜索/1:1 比对/IQA 质量评估/LoRA 双分支 |
| Phase 4: Web Admin | ✅ 完成 | Vue 3 前台/仪表盘/底库管理/识别/系统设置 |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Vue 3 SPA  │ ──→ │  FastAPI     │ ──→ │  MTCNN 检测    │
│  (Frontend) │     │  (Backend)   │     │  + 对齐        │
└─────────────┘     └──────┬───────┘     └────────────────┘
                           │                     │
                           │              ┌──────▼───────┐
                           │              │  PETALface    │
                           │              │  Swin-T + LoRA│
                           │              └──────┬───────┘
                           │                     │
                     ┌─────▼─────────────────────▼──────┐
                     │        SQLite + Faiss            │
                     │  (faces.db + faces.faiss)        │
                     └──────────────────────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 向量检索 | Faiss IndexFlatIP | 暴力搜索，底库<1万时性能足够，精度100% |
| 元数据 | SQLite | 零配置，单机部署简单 |
| 图像质量 | CNN-IQA | PETALface 原生支持的双分支架构 |
| 前端 | Vue 3 + Vite | 轻量、快速开发 SPA |
| 包管理 | uv | 现代 Python 包管理，速度快、依赖锁可靠 |
| 部署 | 单进程 uvicorn | 适合低并发场景，无需额外组件 |

## Key Metrics

| Metric | Value |
|--------|-------|
| 特征维度 | 512 |
| 检索方式 | 内积 (L2 归一化后 = 余弦相似度) |
| CPU 推理速度 | ~500ms-2s/张 |
| GPU 推理速度 | ~50ms/张 |
| 搜索时间 | 10μs + SQL 查询 (底库<1万) |
