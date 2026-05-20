# Phase 2: 人脸底库管理 (Gallery) 设计规格

## 概述

基于 Phase 1 的 Embedding API，构建人脸底库管理系统，支持人脸注册、删除、列表、清空、ZIP 批量导入。

## 存储架构

### Faiss 向量索引
- **类型**: `faiss.IndexIDMap(faiss.IndexFlatIP(512))`
- 512 维 embedding 先 L2 归一化，IP 内积 = 余弦相似度
- `add_with_ids` / `remove_ids` 管理向量
- ID 为整数，映射到 SQLite 的 `rowid`

### SQLite 元数据
- **位置**: `gallery/{db_name}.db`（默认 `gallery/faces.db`）
- **表结构**:

```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT UNIQUE NOT NULL,       -- UUID
    name TEXT NOT NULL,                  -- 中文名称
    created_at TEXT NOT NULL,            -- ISO 8601
    image_path TEXT                      -- 原图路径（可选）
);
```

- `id` 与 Faiss 的 ID 一一对应，作为关联键

### 启动行为
- 服务启动时扫描 SQLite，将所有向量加载到 Faiss 内存索引
- 支持从 SQLite 重建 Faiss 索引

## API 设计

所有响应使用 Phase 1 的统一格式 `{code, message, data}`。

### 1. 单张上传

```
POST /api/v1/gallery
Content-Type: multipart/form-data

file=<image> 或 body: { image: "base64...", image_url: "https://...", name: "张三" }
```

```json
// 201
{"code": 0, "message": "success", "data": {"face_id": "uuid", "name": "张三"}}
```

### 2. ZIP 批量上传

```
POST /api/v1/gallery/batch
Content-Type: multipart/form-data

file=<gallery.zip>
```

ZIP 内图片自动解压、检测人脸、提取 embedding、注册到底库。
图片文件名作为人脸名称（去掉扩展名）。

```json
// 201
{"code": 0, "message": "success", "data": {"total": 10, "succeeded": 8, "failed": 2, "failures": [{"file": "blurred.jpg", "reason": "未检测到人脸"}]}}
```

### 3. 列表查询

```
GET /api/v1/gallery?page=1&page_size=20&search=张
```

```json
{"code": 0, "message": "success", "data": {
    "items": [{"face_id": "uuid", "name": "张三", "created_at": "2026-05-20T..."}],
    "total": 1, "page": 1, "page_size": 20
}}
```

### 4. 删除单个人脸

```
DELETE /api/v1/gallery/{face_id}
```

```json
{"code": 0, "message": "success", "data": null}
```

### 5. 清空底库

```
DELETE /api/v1/gallery
```

```json
{"code": 0, "message": "success", "data": null}
```

## 错误码扩展

| code | message | 说明 |
|------|---------|------|
| 1001 | 未检测到人脸 | MTCNN 无人脸 |
| 1002 | 图片质量不合格 | 模糊/过暗 |
| 2001 | 人脸已存在 | 重复注册（可选做） |
| 2002 | 人脸不存在 | face_id 未找到 |
| 2003 | 底库为空 | 空库操作 |

## 文件结构

新文件置于 `facerecserver/gallery/` 模块下：

```
facerecserver/gallery/
├── __init__.py
├── repository.py      # SQLite + Faiss 存储层
├── schemas.py         # Pydantic 请求/响应模型
└── routes.py          # FastAPI 路由
```

更新文件：
- `facerecserver/app.py` — 注册新路由
- `facerecserver/config.yaml` — 添加 gallery 配置
- `facerecserver/config.py` — 添加 gallery 配置项

## 性能考虑

- 小规模底库（<1万张），FlatIP 精确搜索绰绰有余
- Faiss 索引全量内存加载，启动时从 SQLite 重建
- ZIP 上传使用临时目录解压，处理后清理

## 配置项

```yaml
gallery:
  db_dir: gallery               # SQLite 和 Faiss 文件目录
  db_name: faces                # 数据库文件名（不含扩展名）
  page_size_default: 20         # 默认分页大小
  page_size_max: 100            # 最大分页大小
```
