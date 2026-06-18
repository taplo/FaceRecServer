# FaceRecServer

基于 PETALface (WACV 2025) 算法的人脸识别 API 服务，提供人脸底库管理、1:1 比对、1:N 搜索等功能，并附带基于 Web 的管理后台。

## 硬件要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| CPU | 支持 AVX2 指令集 | x86-64 4 核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 10 GB 可用空间 | 50 GB+（SSD 更佳） |
| GPU | 可选（CUDA） | NVIDIA 6 GB+ |

> **磁盘说明**: 仅模型文件约 1.6 GB（PETALface 默认模型 + CNN-IQA 质量评估模型）。人脸照片底库按每张 10-50 KB 估算，6949 人约 200 MB。如需测试多个模型，磁盘需求会进一步增加。

## 功能

- **人脸特征提取** — 从图片中提取 512 维人脸特征向量
- **底库管理** — 人脸注册、批量导入（ZIP）、列表/搜索、删除、清空
- **1:N 搜索** — 上传照片，在底库中搜索最相似的人脸
- **1:1 比对** — 上传两张照片，计算相似度
- **Web 管理后台** — Vue 3 前端，可视化操作
- **工号支持** — 自动解析 `姓名-工号.jpg` 文件名格式

## 快速开始

### Docker 部署（推荐）

```bash
# 构建镜像
docker compose build

# 启动服务（自动下载模型）
docker compose up -d

# 查看日志
docker compose logs -f
```

首次启动会自动下载模型（约 800 MB），稍后访问：
- **API 文档**: http://localhost:8000/docs
- **Web 后台**: http://localhost:8000
- **健康检查**: http://localhost:8000/api/v1/health

> GPU 加速：如果宿主机有 NVIDIA GPU，安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) 后自动启用。

### 本地开发

#### 前置要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器

#### 安装

```bash
git clone https://github.com/your-org/FaceRecServer.git
cd FaceRecServer
uv sync
```

#### 下载模型

```bash
uv run python scripts/download_model.py --model swin_arcface_webface4m_tinyface
```

模型文件保存到 `models/swin_arcface_webface4m_tinyface/model.pt`（约 800 MB）。首次启动时还会自动下载 CNN-IQA 质量评估模型（约 800 MB，`pyiqa` 依赖），总模型占用约 1.6 GB。

#### 启动服务

```bash
# 生产模式
uv run python -m facerecserver

# 开发模式（热加载）
uv run uvicorn facerecserver.app:create_app --factory --reload --port 8000
```

访问：
- **API 文档**: http://localhost:8000/docs
- **Web 后台**: http://localhost:8000（需先构建前端: `cd frontend && npm run build`）

## API 参考

所有 API 统一返回格式：

```json
{"code": 0, "message": "success", "data": { ... }}
```

错误码：

| code | 说明 |
|------|------|
| 0 | 成功 |
| 400 | 请求参数错误 |
| 1001 | 未检测到人脸 |
| 1002 | 图片处理失败（质量检查等） |
| 2002 | 人脸/图片不存在 |
| 5000 | 服务未初始化 |
| -1 | 服务器内部错误 |

### 提取特征

```
POST /api/v1/embedding
```

三种输入方式：

```bash
# 方式 1: 文件上传
curl -X POST http://localhost:8000/api/v1/embedding \
  -F "file=@photo.jpg"

# 方式 2: Base64
curl -X POST http://localhost:8000/api/v1/embedding \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_data>"}'

# 方式 3: 图片 URL
curl -X POST http://localhost:8000/api/v1/embedding \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg"}'
```

响应：

```json
{"code": 0, "data": {"embedding": [0.123, -0.456, ...], "dimension": 512}}
```

### 注册人脸

```
POST /api/v1/gallery
```

```bash
# 上传文件，文件名作为姓名
curl -X POST http://localhost:8000/api/v1/gallery \
  -F "file=@张三-ENG001.jpg"

# 上传文件，指定姓名和工号
curl -X POST http://localhost:8000/api/v1/gallery \
  -F "file=@photo.jpg"

# JSON 方式，显式指定姓名和工号
curl -X POST http://localhost:8000/api/v1/gallery \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "name": "张三", "employee_id": "ENG001"}'
```

如果文件名格式为 `姓名-工号.jpg`，系统自动解析工号。响应：

```json
{"code": 0, "data": {"face_id": "uuid", "name": "张三", "employee_id": "ENG001"}}
```

### 批量注册（ZIP）

```
POST /api/v1/gallery/batch
```

```bash
curl -X POST http://localhost:8000/api/v1/gallery/batch \
  -F "file=@faces.zip"
```

ZIP 文件名格式：`姓名-工号.jpg`。响应：

```json
{"code": 0, "data": {"total": 100, "succeeded": 98, "failed": 2, "failures": [...]}}
```

### 1:1 比对

```
POST /api/v1/compare
```

```bash
# 文件上传方式
curl -X POST http://localhost:8000/api/v1/compare \
  -F "file1=@face1.jpg" -F "file2=@face2.jpg"

# JSON 方式
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"image1": "<base64_1>", "image2": "<base64_2>"}'
```

响应：

```json
{"code": 0, "data": {"similarity": 0.8723}}
```

> similarity 为余弦相似度，范围 [-1, 1]。建议阈值 ≥0.6 判为同一人。

### 1:N 搜索

```
POST /api/v1/gallery/recognize?top_k=5
```

```bash
curl -X POST "http://localhost:8000/api/v1/gallery/recognize?top_k=5" \
  -F "file=@query.jpg"
```

响应：

```json
{
  "code": 0,
  "data": {
    "results": [
      {"face_id": "uuid1", "name": "张三", "employee_id": "ENG001", "score": 0.8723, "image_url": "/api/v1/gallery/uuid1/image"},
      {"face_id": "uuid2", "name": "李四", "employee_id": "ENG002", "score": 0.6541, "image_url": "/api/v1/gallery/uuid2/image"}
    ]
  }
}
```

> score 为余弦相似度（内积），范围 [-1, 1]。建议阈值 ≥0.6 判为同一人。

### 健康检查

```
GET /api/v1/health
```

```json
{"status": "ok", "model_loaded": true, "gallery_ready": true, "device": "cuda", "uptime_seconds": 3600}
```

### 列出底库（按人脸）

```
GET /api/v1/gallery?page=1&page_size=20&search=张三
```

```bash
curl "http://localhost:8000/api/v1/gallery?page=1&page_size=10&search=张三"
```

支持按姓名或工号搜索。响应：

```json
{
  "code": 0,
  "data": {
    "items": [
      {"face_id": "uuid", "name": "张三", "employee_id": "ENG001", "created_at": "2026-05-21T12:00:00+00:00", "image_url": "/api/v1/gallery/uuid/image"}
    ],
    "total": 1, "page": 1, "page_size": 20
  }
}
```

### 列出人员（按人聚合）

```
GET /api/v1/gallery/persons?page=1&page_size=20&search=张三
```

```json
{
  "code": 0,
  "data": {
    "items": [
      {"person_id": 1, "name": "张三", "employee_id": "ENG001", "created_at": "...", "face_count": 3}
    ],
    "total": 1, "page": 1, "page_size": 20
  }
}
```

> 同一人注册多张照片后，`face_count` 表示该人的底库照片数，1:N 搜索会自动按人聚合取最高分。

### 删除人脸

```
DELETE /api/v1/gallery/{face_id}
```

### 删除人员（级联删除其所有人脸）

```
DELETE /api/v1/gallery/persons/{person_id}
```

### 清空底库

```
DELETE /api/v1/gallery
```

### 获取人脸图片

```
GET /api/v1/gallery/{face_id}/image
```

### 系统统计

```
GET /api/v1/stats
```

```json
{
  "code": 0,
  "data": {
    "gallery": {"total_faces": 6949, "total_persons": 6949, "index_size": 6949, "dimension": 512},
    "server": {"uptime_seconds": 3600, "device": "cuda"}
  }
}
```

## 底库导入脚本

项目附带命令行导入工具，用于从 ZIP 文件批量导入人脸：

```bash
# 编辑 scripts/import_gallery.py 修改 ZIP_PATH 变量
uv run python scripts/import_gallery.py
```

或运行批处理文件：

```bash
run_import.bat
```

导入流程：
1. 清空现有底库
2. 遍历 ZIP 内所有图片
3. 对每张图片：MTCNN 检测 → 对齐 → PETALface 提取特征 → 注册到底库
4. 输出 CSV 报告到 `gallery/import_report.csv`

## 配置

默认配置 `facerecserver/config.yaml`：

```yaml
model:
  path: models/swin_arcface_webface4m_tinyface/model.pt
  name: swin_arcface_webface4m_tinyface
  lora_rank: 8
  lora_scale: 1.0
  use_lora: true

detection:
  confidence: 0.95
  min_face_size: 40

preprocess:
  image_size: 112
  do_alignment: true
  do_quality_check: true
  iqa:
    enabled: true
    threshold: 0.5

server:
  host: 0.0.0.0
  port: 8000

gallery:
  db_dir: gallery
  db_name: faces
  page_size_default: 20
  page_size_max: 100
```

环境变量 `FACEREC_CONFIG` 可指定自定义配置路径。

### 关键配置说明

| 配置项 | 说明 | 建议 |
|--------|------|------|
| `detection.confidence` | MTCNN 检测置信度阈值 | 0.9-0.98，越低召回越高 |
| `detection.min_face_size` | 最小人脸尺寸(px) | 30-60 |
| `preprocess.do_quality_check` | 是否检查模糊/过暗 | 批量导入建议关闭 |
| `preprocess.iqa.threshold` | 图像质量评分阈值 | 0.3-0.7 |
| `model.use_lora` | 是否启用 LoRA 双分支 | 建议开启 |

## 项目结构

```
facerecserver/
├── api/                 # 主路由: /embedding, /compare, /stats, /health
├── face_detection/      # MTCNN 检测 + 关键点对齐（自动 GPU）
├── face_recognition/    # PETALface 模型 + 特征提取（自动 GPU）
├── gallery/             # 底库: 人脸/人员 CRUD + Faiss/SQLite 存储
├── web/                 # 前端 SPA 静态文件挂载
├── app.py               # FastAPI 应用工厂
├── config.py            # 配置 dataclass + YAML 加载
└── config.yaml          # 默认配置

frontend/                # Vue 3 + TypeScript 前端
├── src/views/           # 4 个页面: 仪表盘/底库/识别/设置
├── src/api/client.ts    # API 客户端
└── dist/                # 构建产物

scripts/
├── download_model.py    # HuggingFace 模型下载
├── import_gallery.py    # ZIP 批量导入
└── entrypoint.sh        # Docker 入口脚本

Dockerfile               # 基于 CUDA 的多阶段构建
docker-compose.yml       # 容器编排（含 GPU 支持）
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 语言/运行时 | Python >=3.12, uv |
| Web 框架 | FastAPI + uvicorn |
| 人脸检测 | MTCNN (facenet-pytorch) |
| 特征提取 | PETALface (Swin-Tiny, 512-dim) |
| 质量评估 | CNN-IQA (pyiqa) |
| 向量检索 | Faiss (IndexFlatIP, 余弦相似度) |
| 元数据存储 | SQLite |
| 前端 | Vue 3 + TypeScript + Vite |

## License

MIT
