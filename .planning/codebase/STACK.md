# Stack: FaceRecServer

## Current Stack

| Component | Choice | Version | Purpose |
|-----------|--------|---------|---------|
| Language | Python | >=3.12 | |
| Package Manager | uv | latest | |
| Entry Format | `python -m facerecserver` | via pyproject.toml | |
| Web Framework | FastAPI | >=0.104 | REST API |
| ASGI Server | uvicorn | >=0.24 | HTTP 服务 |
| Face Detection | MTCNN (facenet-pytorch) | >=2.6 | 人脸检测 + 关键点 |
| Face Recognition | PETALface (Swin-T + LoRA) | custom | 512-d 特征提取 |
| Image Quality | CNN-IQA (pyiqa) | >=0.1.15 | 质量评分 |
| Vector Search | Faiss (IndexFlatIP) | >=1.7 | 相似度搜索 |
| Metadata Store | SQLite | stdlib | 人脸元数据 |
| Image Processing | OpenCV + Pillow | >=4.9 + >=10.0 | 图像对齐/处理 |
| Config | PyYAML | >=6.0 | YAML 配置 |
| Model Hub | HuggingFace Hub | >=0.20 | 模型权重下载 |
| Frontend | Vue 3 + Vite + TS | 3.4 + 5.4 + 5.4 | 管理后台 |
| HTTP Client | requests | >=2.31 | 内置 image_url 获取 |
| Testing | pytest + httpx | >=9.0 + >=0.28 | 测试 |

## Dependencies

See `pyproject.toml` for full dependency list with version constraints.

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Faiss 而非 Milvus/Pinecone | 单机部署，无需额外服务，底库 < 1 万时 IndexFlatIP 足够 |
| SQLite 而非 PostgreSQL/MySQL | 零配置，元数据量小，单文件备份方便 |
| LoRA 双分支 | PETALface 原生设计，在低质量图片上更鲁棒 |
| 余弦相似度 (内积) | 标准人脸识别度量，与大部分学术 benchmark 一致 |
