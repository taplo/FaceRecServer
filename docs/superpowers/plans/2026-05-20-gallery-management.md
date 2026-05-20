# 人脸底库管理 (Gallery) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建人脸底库管理系统，支持单张/ZIP注册、删除、列表、清空

**Architecture:** `facerecserver/gallery/repository.py` 封装 SQLite（元数据）+ Faiss（向量），提供统一存储接口；`routes.py` 暴露 REST API

**Tech Stack:** faiss-cpu, sqlite3, zipfile, uuid

---

### Task 1: 添加 faiss-cpu 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 编辑 pyproject.toml**

在 `dependencies` 列表中插入 `"faiss-cpu>=1.7"`：

```
    "requests>=2.31",
    "faiss-cpu>=1.7",
```

- [ ] **Step 2: 安装依赖**

```
uv sync
```

- [ ] **Step 3: 验证安装**

```
python -c "import faiss; print(faiss.IndexFlatIP(512)); print('faiss ok')"
```

Expected: `<faiss.swigfaiss.IndexFlatIP object at 0x...>`

- [ ] **Step 4: Commit**

```
git add -A && git commit -m "chore: add faiss-cpu dependency"
```

---

### Task 2: Gallery 配置项

**Files:**
- Create: `facerecserver/gallery/__init__.py`
- Modify: `facerecserver/config.py`
- Modify: `facerecserver/config.yaml`

- [ ] **Step 1: 创建 `facerecserver/gallery/__init__.py`**

空文件。

- [ ] **Step 2: 编辑 `facerecserver/config.py` 添加 GalleryConfig**

在 `ServerConfig` 之后、`AppConfig` 之前插入：

```python
@dataclass
class GalleryConfig:
    db_dir: str = "gallery"
    db_name: str = "faces"
    page_size_default: int = 20
    page_size_max: int = 100
```

在 `AppConfig` 中添加字段并导入：

```python
@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    device: str = "cpu"
```

在 `load_config` 中添加加载逻辑（在 `s` 块之后）：

```python
    g = raw.get("gallery", {})
    cfg.gallery.db_dir = g.get("db_dir", cfg.gallery.db_dir)
    cfg.gallery.db_name = g.get("db_name", cfg.gallery.db_name)
    cfg.gallery.page_size_default = g.get("page_size_default", cfg.gallery.page_size_default)
    cfg.gallery.page_size_max = g.get("page_size_max", cfg.gallery.page_size_max)
```

- [ ] **Step 3: 编辑 `facerecserver/config.yaml`**

在文件末尾追加：

```yaml
gallery:
  db_dir: gallery
  db_name: faces
  page_size_default: 20
  page_size_max: 100
```

- [ ] **Step 4: 验证**

```
python -c "from facerecserver.config import load_config; c=load_config(); print(c.gallery)"
```

Expected: `GalleryConfig(db_dir='gallery', db_name='faces', page_size_default=20, page_size_max=100)`

- [ ] **Step 5: Commit**

```
git add -A && git commit -m "feat: add gallery config"
```

---

### Task 3: Gallery 存储层 (SQLite + Faiss)

**Files:**
- Create: `facerecserver/gallery/repository.py`

- [ ] **Step 1: 创建 `facerecserver/gallery/repository.py`**

```python
import os
import sqlite3
import uuid
import numpy as np
import faiss
from datetime import datetime, timezone
from dataclasses import dataclass, field


@dataclass
class FaceRecord:
    face_id: str
    name: str
    created_at: str
    image_path: str = ""


@dataclass
class GalleryStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    failures: list = field(default_factory=list)


class GalleryRepository:
    DIM = 512

    def __init__(self, db_dir: str, db_name: str):
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, f"{db_name}.db")
        self.index_path = os.path.join(db_dir, f"{db_name}.faiss")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
        self._load_index()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                image_path TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_id ON faces(face_id)
        """)
        self._conn.commit()

    def _load_index(self):
        rows = self._conn.execute("SELECT id, name, created_at, image_path, face_id FROM faces").fetchall()
        if not rows:
            return
        ids = []
        vectors = []
        for row in rows:
            ids.append(row[0])
        vec_path = self.index_path
        if os.path.exists(vec_path):
            self._index = faiss.read_index(vec_path)
        print(f"[Gallery] 加载底库: {len(ids)} 条记录")

    def _save_index(self):
        faiss.write_index(self._index, self.index_path)

    def add(self, embedding: np.ndarray, name: str, image_path: str = "") -> str:
        face_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        normalized = embedding / np.linalg.norm(embedding)
        cursor = self._conn.execute(
            "INSERT INTO faces (face_id, name, created_at, image_path) VALUES (?, ?, ?, ?)",
            (face_id, name, now, image_path),
        )
        faiss_id = cursor.lastrowid
        self._index.add_with_ids(normalized.reshape(1, -1).astype(np.float32), np.array([faiss_id]))
        self._save_index()
        return face_id

    def add_batch(self, embeddings: list, names: list, image_paths: list | None = None) -> GalleryStats:
        stats = GalleryStats(total=len(embeddings))
        for i, (emb, name) in enumerate(zip(embeddings, names)):
            try:
                path = image_paths[i] if image_paths else ""
                self.add(emb, name, path)
                stats.succeeded += 1
            except Exception as e:
                stats.failed += 1
                stats.failures.append({"file": name, "reason": str(e)})
        return stats

    def delete(self, face_id: str) -> bool:
        row = self._conn.execute("SELECT id FROM faces WHERE face_id = ?", (face_id,)).fetchone()
        if row is None:
            return False
        faiss_id = row[0]
        self._index.remove_ids(np.array([faiss_id]))
        self._conn.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
        self._conn.commit()
        self._save_index()
        return True

    def clear(self):
        self._conn.execute("DELETE FROM faces")
        self._conn.commit()
        dim = self.DIM
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._save_index()

    def list_faces(self, page: int = 1, page_size: int = 20, search: str = "") -> tuple:
        offset = (page - 1) * page_size
        if search:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM faces WHERE name LIKE ?", (f"%{search}%",)
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT face_id, name, created_at, image_path FROM faces WHERE name LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (f"%{search}%", page_size, offset),
            ).fetchall()
        else:
            count = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            rows = self._conn.execute(
                "SELECT face_id, name, created_at, image_path FROM faces ORDER BY id DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        items = [FaceRecord(face_id=r[0], name=r[1], created_at=r[2], image_path=r[3]) for r in rows]
        return items, count

    def get_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def close(self):
        self._conn.close()
```

- [ ] **Step 2: 验证导入**

```
python -c "from facerecserver.gallery.repository import GalleryRepository; print('ok')"
```

Expected: no error

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: gallery SQLite + Faiss storage layer"
```

---

### Task 4: Gallery Pydantic Schemas

**Files:**
- Create: `facerecserver/gallery/schemas.py`

- [ ] **Step 1: 创建 `facerecserver/gallery/schemas.py`**

```python
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
```

- [ ] **Step 2: 验证导入**

```
python -c "from facerecserver.gallery.schemas import GalleryAddRequest; print('ok')"
```

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: gallery pydantic schemas"
```

---

### Task 5: Gallery API Routes

**Files:**
- Create: `facerecserver/gallery/routes.py`

- [ ] **Step 1: 创建 `facerecserver/gallery/routes.py`**

```python
import io
import os
import zipfile
import tempfile
import logging
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Query

from facerecserver.api.schemas import ApiResponse
from facerecserver.gallery.repository import GalleryRepository
from facerecserver.gallery.schemas import GalleryAddRequest
from facerecserver.face_recognition.utils import base64_to_image, load_image
from facerecserver.face_detection.detector import FaceNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gallery")


def _get_repo(request: Request) -> GalleryRepository:
    repo = getattr(request.app.state, "gallery_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "底库未初始化"})
    return repo


def _get_extractor(request: Request):
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(status_code=503, detail={"code": 5000, "message": "模型未加载"})
    return extractor


@router.post("", response_model=ApiResponse)
async def add_face(
    request: Request,
    file: UploadFile | None = File(None),
    body: GalleryAddRequest | None = None,
):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    try:
        if file is not None:
            contents = await file.read()
            image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
            name = os.path.splitext(file.filename or "unknown")[0]
        elif body and body.image:
            image = base64_to_image(body.image)
            name = body.name or "unknown"
        elif body and body.image_url:
            import requests as http_requests
            resp = http_requests.get(body.image_url, timeout=30)
            resp.raise_for_status()
            image = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
            name = body.name or "unknown"
        else:
            return ApiResponse(code=400, message="请提供图片 (file, image, 或 image_url)", data=None)

        embedding = extractor.extract(image)
        face_id = repo.add(embedding, name)
        return ApiResponse(code=0, message="success", data={"face_id": face_id, "name": name})

    except FaceNotFoundError as e:
        return ApiResponse(code=1001, message=str(e), data=None)
    except ValueError as e:
        return ApiResponse(code=1002, message=str(e), data=None)
    except Exception as e:
        logger.exception("注册人脸失败")
        return ApiResponse(code=-1, message=f"处理失败: {str(e)}", data=None)


@router.post("/batch", response_model=ApiResponse)
async def add_faces_batch(request: Request, file: UploadFile = File(...)):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    contents = await file.read()
    if not contents:
        return ApiResponse(code=400, message="ZIP 文件为空", data=None)

    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            zf.extractall(temp_dir)

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []
        for root, _dirs, files in os.walk(temp_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in image_exts:
                    image_paths.append(os.path.join(root, fname))

        embeddings = []
        names = []
        errors = []
        for img_path in image_paths:
            try:
                image = load_image(img_path)
                emb = extractor.extract(image)
                embeddings.append(emb)
                names.append(os.path.splitext(os.path.basename(img_path))[0])
            except FaceNotFoundError:
                errors.append({"file": os.path.basename(img_path), "reason": "未检测到人脸"})
            except Exception as e:
                errors.append({"file": os.path.basename(img_path), "reason": str(e)})

        stats = repo.add_batch(embeddings, names)
        for err in errors:
            stats.failed += 1
            stats.failures.append(err)

        return ApiResponse(code=0, message="success", data={
            "total": stats.total + len(errors),
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "failures": stats.failures,
        })
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("", response_model=ApiResponse)
async def list_faces(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query("", max_length=50),
):
    repo = _get_repo(request)
    items, total = repo.list_faces(page, page_size, search)
    return ApiResponse(code=0, message="success", data={
        "items": [{"face_id": f.face_id, "name": f.name, "created_at": f.created_at} for f in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    })


@router.delete("/{face_id}", response_model=ApiResponse)
async def delete_face(request: Request, face_id: str):
    repo = _get_repo(request)
    if not repo.delete(face_id):
        return ApiResponse(code=2002, message="人脸不存在", data=None)
    return ApiResponse(code=0, message="success", data=None)


@router.delete("", response_model=ApiResponse)
async def clear_gallery(request: Request):
    repo = _get_repo(request)
    repo.clear()
    return ApiResponse(code=0, message="success", data=None)
```

- [ ] **Step 2: 验证导入**

```
python -c "from facerecserver.gallery.routes import router; print('ok')"
```

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: gallery API routes"
```

---

### Task 6: 注册 Gallery 路由到 App

**Files:**
- Modify: `facerecserver/app.py`

- [ ] **Step 1: 编辑 `facerecserver/app.py`**

在文件顶部导入 gallery router：

```python
from facerecserver.gallery.routes import router as gallery_router
from facerecserver.gallery.repository import GalleryRepository
```

在 `lifespan` 函数中，模型加载成功后初始化仓库：

```python
        extractor = FaceEmbeddingExtractor(config)
        app.state.extractor = extractor
        repo = GalleryRepository(config.gallery.db_dir, config.gallery.db_name)
        app.state.gallery_repo = repo
        print(f"[启动] 模型已加载: {config.model.name} on {config.device}")
```

并在 yield 前添加关闭 repository：

```python
    yield
    repo = getattr(app.state, "gallery_repo", None)
    if repo:
        repo.close()
    print("[关闭] 服务停止")
```

在 `create_app` 中注册 gallery 路由（在已有 router 之后）：

```python
    app.include_router(router)
    app.include_router(gallery_router)
```

- [ ] **Step 2: 验证应用启动**

```
python -c "from facerecserver.app import create_app; app=create_app(); print('routes:', [r.path for r in app.routes][:10])"
```

Expected: 包含 `/api/v1/gallery` 相关路由

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: register gallery routes in app"
```

---

### Task 7: 端到端验证

- [ ] **Step 1: 创建测试脚本**

创建 `test_gallery.py`：

```python
import requests, sys, time, os

SERVER = "http://127.0.0.1:8010"
URL = SERVER + "/api/v1/gallery"

# wait for server
for i in range(30):
    try:
        r = requests.get(SERVER + "/openapi.json", timeout=2)
        if r.status_code == 200:
            print("Server ready")
            break
    except:
        pass
    time.sleep(1)
else:
    print("Server not ready")
    sys.exit(1)

# test add face
with open("test/01/xxx.jpg", "rb") as f:
    r = requests.post(URL, files={"file": f})
    print("Add:", r.json())
    data = r.json()
    assert data["code"] == 0, f"Add failed: {data}"
    face_id = data["data"]["face_id"]

# test list
r = requests.get(URL)
print("List:", r.json())
assert r.json()["code"] == 0
assert r.json()["data"]["total"] >= 1

# test search
r = requests.get(URL, params={"search": "xxx", "page": 1, "page_size": 10})
print("Search:", r.json())

# test delete
r = requests.delete(f"{URL}/{face_id}")
print("Delete:", r.json())
assert r.json()["code"] == 0

# test delete not found
r = requests.delete(f"{URL}/nonexistent")
print("Delete not found:", r.json())
assert r.json()["code"] == 2002

print("All tests passed!")
```

- [ ] **Step 2: 启动服务并运行测试**

```
start uvicorn in background, then python test_gallery.py
```

- [ ] **Step 3: 完成后清理 test_gallery.py（可选）**

---

### Verification

1. `/api/v1/gallery` POST 返回 face_id
2. `/api/v1/gallery` GET 返回分页列表
3. `/api/v1/gallery/{face_id}` DELETE 删除成功
4. `/api/v1/gallery` DELETE 清空成功
5. `/api/v1/gallery/batch` POST 处理 ZIP
6. SQLite + Faiss 文件持久化在 `gallery/` 目录
7. 服务重启后数据可恢复
