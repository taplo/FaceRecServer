# FaceRecServer

基于 PETALface (WACV 2025) 算法的人脸识别 API 服务，提供人脸底库管理、人脸比对 (1:1)、人脸识别 (1:N) 等功能，并附带基于 Web 的管理后台。

## 硬件要求

- **CPU**: 支持 AVX2 指令集（用于 Faiss 索引）
- **内存**: 最低 4GB，推荐 8GB+
- **GPU**: 可选（CUDA），通过 `torch.cuda.is_available()` 自动检测
- **磁盘**: 100MB（不含模型文件，模型约 200MB）

## 快速开始

### 1. 安装依赖

```bash
# 安装 Python 3.12+
# 安装 uv（包管理器）
pip install uv

# 安装项目依赖
uv sync
```

### 2. 下载模型

从 HuggingFace 下载模型权重到项目目录：

```bash
# 模型存放路径（默认）
models/swin_arcface_webface4m_tinyface/model.pt
```

> 网络受限时，可手动下载后将文件放入对应目录。

### 3. 启动服务

```bash
# 开发模式（热加载）
uv run uvicorn facerecserver.app:create_app --factory --reload --port 8000

# 生产模式
uv run python -m facerecserver

# 或直接使用 uv run
uv run python -m facerecserver
```

服务启动后访问：
- **API 文档**: http://localhost:8000/docs
- **管理后台**: http://localhost:8000（需先构建前端）

## Web 管理后台

前端位于 `frontend/` 目录，基于 Vue 3 + Vite + TypeScript。

```bash
# 安装前端依赖
cd frontend
npm install

# 开发模式（热更新，API 自动代理到 8000 端口）
npm run dev

# 生产构建（构建产物由 FastAPI 直接服务）
npm run build
```

管理后台包含 4 个页面：

| 页面 | 功能 |
|------|------|
| 仪表盘 | 底库统计、服务器运行时间、运行设备 |
| 人脸底库 | 列表/搜索/注册/删除，支持单张注册 |
| 人脸识别 | 1:1 比对 + 1:N 搜索，支持 Top-K 选择 |
| 系统设置 | 模型信息、底库维护 |

## API 文档

所有 API 统一返回格式：

```json
{
  "code": 0,
  "message": "success",
  "data": { ... }
}
```

错误码说明：

| code | 说明 |
|------|------|
| 0 | 成功 |
| 400 | 请求参数错误 |
| 1001 | 未检测到人脸 |
| 1002 | 图片处理失败 |
| 2002 | 人脸不存在 |
| 5000 | 服务未初始化（模型/底库未加载） |
| -1 | 服务器内部错误 |

### 人脸特征提取

```http
POST /api/v1/embedding
```

三种输入方式：

| 方式 | Content-Type | 参数 |
|------|-------------|------|
| 文件上传 | multipart/form-data | `file` |
| Base64 | application/json | `{"image": "<base64>"}` |
| URL | application/json | `{"image_url": "https://..."}` |

**响应示例：**

```json
{
  "code": 0,
  "data": {
    "embedding": [0.123, -0.456, ...],
    "dimension": 512
  }
}
```

### 底库管理

#### 注册人脸

```http
POST /api/v1/gallery
```

与 `/embedding` 相同的三种输入方式，额外支持 `name` 参数。

**响应：**

```json
{
  "code": 0,
  "data": {
    "face_id": "uuid-string",
    "name": "张三"
  }
}
```

#### 批量注册（ZIP 上传）

```http
POST /api/v1/gallery/batch
Content-Type: multipart/form-data

file: @faces.zip
```

- ZIP 内支持 JPG/PNG/BMP/TIFF/WebP 格式
- 文件名作为姓名（不含扩展名）
- 支持 GBK 编码文件名
- 返回成功/失败统计

#### 底库列表

```http
GET /api/v1/gallery?page=1&page_size=20&search=张三
```

#### 删除人脸

```http
DELETE /api/v1/gallery/{face_id}
```

#### 清空底库

```http
DELETE /api/v1/gallery
```

#### 获取人脸图片

```http
GET /api/v1/gallery/{face_id}/image
```

返回 JPEG 图片（注册时保存的人脸裁剪图）。

### 人脸识别

#### 1:N 搜索

```http
POST /api/v1/gallery/recognize?top_k=5
```

三种输入方式（同注册），在底库中搜索最相似的 Top-K 条结果。

**响应：**

```json
{
  "code": 0,
  "data": {
    "results": [
      {"face_id": "...", "name": "张三", "score": 0.8723, "image_url": "/api/v1/gallery/.../image"},
      {"face_id": "...", "name": "张伟", "score": 0.6541, "image_url": "/api/v1/gallery/.../image"}
    ]
  }
}
```

> 注意：结果不内置阈值过滤，相似度阈值由调用方自行判断。一般建议阈值 ≥0.6 为同一人。

### 系统统计

```http
GET /api/v1/stats
```

**响应：**

```json
{
  "code": 0,
  "data": {
    "gallery": {"total_faces": 6199, "index_size": 6199, "dimension": 512},
    "server": {"uptime_seconds": 7200, "device": "cpu"}
  }
}
```

## 配置

配置文件：`facerecserver/config.yaml`

```yaml
model:
  path: models/swin_arcface_webface4m_tinyface/model.pt
  name: swin_arcface_webface4m_tinyface
  lora_rank: 8
  lora_scale: 1.0
  use_lora: true

detection:
  confidence: 0.95          # MTCNN 人脸检测置信度阈值
  min_face_size: 40          # 最小人脸尺寸

preprocess:
  image_size: 112            # 模型输入尺寸
  do_alignment: true         # 人脸对齐
  do_quality_check: true     # 图像质量检查

server:
  host: 0.0.0.0
  port: 8000

gallery:
  db_dir: gallery            # 底库存储目录
  db_name: faces             # 数据库文件名前缀
  page_size_default: 20
  page_size_max: 100
```

配置可通过环境变量 `FACEREC_CONFIG` 指定路径覆盖。

## 底库导入

项目附带命令行导入脚本，用于从 ZIP 批量导入人脸照片：

```bash
# 编辑 scripts/import_gallery.py 修改 ZIP_PATH
uv run python scripts/import_gallery.py
```

或使用批处理文件：

```bash
run_import.bat
```

导入流程：
1. 清空现有底库
2. 从 ZIP 读取图片文件（支持 GBK 编码文件名）
3. MTCNN 检测 + PETALface 提取特征
4. 存入 Faiss 索引 + SQLite
5. 输出 CSV 格式的导入报表到 `gallery/import_report.csv`

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言 | Python >=3.12 |
| 包管理 | uv |
| Web 框架 | FastAPI |
| 人脸检测 | MTCNN (facenet-pytorch) |
| 特征提取 | PETALface (Swin-Tiny + ArcFace, 512-dim) |
| 向量检索 | Faiss (IndexFlatIP, 内积相似度) |
| 元数据存储 | SQLite |
| 前端 | Vue 3 + Vite + TypeScript + Vue Router 4 |

## 项目结构

```
facerecserver/
├── api/                # API 路由和请求模型
│   ├── routes.py       # /embedding, /stats 端点
│   └── schemas.py      # Pydantic 模型
├── face_detection/     # MTCNN 人脸检测
├── face_recognition/   # PETALface 特征提取
├── gallery/            # 底库管理
│   ├── repository.py   # SQLite + Faiss 存储层
│   ├── routes.py       # 底库 CRUD + 识别端点
│   └── schemas.py      # 请求/响应模型
├── web/                # 前端静态文件服务
│   └── routes.py       # SPA 挂载
├── app.py              # FastAPI 应用工厂 + 生命周期
├── config.py           # 配置加载
└── config.yaml         # 配置文件

frontend/               # Vue 3 前端项目
├── src/
│   ├── views/          # 4 个页面组件
│   ├── components/     # 通用组件
│   ├── api/client.ts   # API 客户端
│   └── router/         # 路由配置
└── vite.config.ts      # Vite 配置（API 代理）

scripts/
└── import_gallery.py   # 底库批量导入脚本

gallery/                # 运行时数据
├── faces.db            # SQLite 元数据
├── faces.faiss         # Faiss 向量索引
└── faces/              # 人脸裁剪图片
```

## 常见问题

### 启动时模型加载失败

确保模型权重已下载到配置中 `model.path` 指定的路径。

### 注册时返回 "未检测到人脸"

图片中人脸太小、角度过大、或质量过低。可尝试降低配置中的 `detection.confidence` 阈值。

### 识别准确率不理想

- 确保注册图片质量良好（正面、光线均匀）
- 增大 MTCNN 的 `min_face_size` 过滤过小的人脸
- 相似度阈值建议 ≥0.6

### 服务器启动慢

首次启动需要加载 PyTorch 模型（约 500ms-2s，取决于 CPU）和重建 Faiss 索引。启动后正常请求不受影响。

## License

MIT
