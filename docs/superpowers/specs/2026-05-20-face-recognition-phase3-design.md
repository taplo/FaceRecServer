# Phase 3: 人脸识别 1:N — 设计文档

**日期:** 2026-05-20
**状态:** 草案

## 1. 概述

在已有的人脸底库（Gallery）基础上，实现 1:N 人脸识别能力。用户上传一张人脸照片，系统自动检测、提取特征，并在底库中搜索最相似的 Top-K 条记录。

## 2. 架构

### 2.1 现有依赖

- `GalleryRepository` — Faiss `IndexIDMap(IndexFlatIP)` + SQLite 元数据存储
- `FaceEmbeddingExtractor` — 人脸检测 + 特征提取（512-dim）
- `ApiResponse` — 统一响应格式

### 2.2 新增组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `search()` 方法 | `gallery/repository.py` | Faiss 相似度搜索 |
| `POST /api/v1/recognize` | `gallery/routes.py` | 识别端点（追加到现有路由文件） |
| 路由注册 | `app.py` | 注册识别路由 |

## 3. Repository 层

在 `GalleryRepository` 中添加方法：

```python
def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """搜索最相似的 top_k 个人脸。

    Args:
        embedding: 512-dim 特征向量（未归一化，方法内部会归一化）
        top_k: 返回结果数

    Returns:
        [{"face_id": str, "name": str, "score": float}, ...]
    """
```

实现：
1. L2 归一化 query embedding
2. `self._index.search(query.reshape(1, -1).astype(np.float32), top_k)`
3. 返回 `(distances, indices)`，其中 `distances` 是 cosine similarity（IndexFlatIP 返回内积）
4. 根据 `indices` 从 SQLite 查 `face_id` 和 `name`
5. 组装结果列表，`score` 范围 [-1, 1]

**边界情况：**
- 空底库：返回空列表 `[]`
- `top_k > 底库数量`：返回所有结果（Faiss 默认行为）
- 非法值：`top_k < 1` 时默认为 1

## 4. API 层

### 4.1 端点

```
POST /api/v1/gallery/recognize
```

**请求（三种输入方式）：**

方式 1 — form-data 文件上传：
```
Content-Type: multipart/form-data
file: <image>
```

方式 2 — JSON base64：
```json
{
    "image": "/9j/4AAQ..."
}
```

方式 3 — JSON image_url：
```json
{
    "image_url": "http://example.com/photo.jpg"
}
```

Query 参数：`?top_k=5`

**成功响应：**
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "results": [
            {"face_id": "uuid", "name": "张三", "score": 0.9234},
            {"face_id": "uuid", "name": "李四", "score": 0.7211}
        ]
    }
}
```

**错误响应：**
| code | 含义 |
|------|------|
| 400 | 未提供图片 |
| 1001 | 未检测到人脸 |
| 1002 | 图片质检不合格 |
| 5000 | 底库/模型未初始化 |

### 4.2 路由注册

识别端点使用现有 `gallery_router` (prefix `/api/v1/gallery`)，路径为 `POST /api/v1/gallery/recognize`，无需新增路由注册。

## 5. 配置项

无需新增配置项。`top_k` 通过 Query 参数动态传入。

## 6. 实现计划

### Task 1: 添加 `GalleryRepository.search()` 方法

文件：`facerecserver/gallery/repository.py`

- 实现 `search(embedding, top_k=5)` 方法
- L2 归一化 + Faiss search + SQLite 查询
- 验证：单元测试

### Task 2: 创建识别请求/响应 Schemas

文件：`facerecserver/gallery/schemas.py`

- `RecognizeRequest(BaseModel)` — `image: Optional[str]`, `image_url: Optional[str]`
- `RecognizeItem(BaseModel)` — `face_id`, `name`, `score`
- `RecognizeResponse` — 复用 `ApiResponse`

### Task 3: 创建识别 API 路由

文件：`facerecserver/gallery/routes.py`（追加到现有文件）

- `POST /api/v1/recognize` 端点
- 三种输入方式支持
- 调用 extractor.extract() → repo.search()
- 错误处理

### Task 4: 确认路由注册

`recognize` 端点使用现有 `gallery_router`，无需修改 `app.py`。

### Task 5: 验证

- 启动服务
- 用 curl/requests 测试三种输入方式
- 验证 top_k 参数
- 验证空底库场景
