# Phase 3: 人脸识别 1:N Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于已有底库实现 1:N 人脸识别 API

**Architecture:** GalleryRepository 新增 `search()` 方法（Faiss IndexIDMap.search），API 层 `POST /api/v1/gallery/recognize` 复用现有输入方式（file/base64/URL）和响应格式

**Tech Stack:** faiss, fastapi, pydantic

---

### Task 1: GalleryRepository 添加 search() 方法

**Files:**
- Modify: `facerecserver/gallery/repository.py`

- [ ] **Step 1: 在 `GalleryRepository` 中添加 `search()` 方法**

在 `get_count()` 之前插入：

```python
    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self._index.ntotal == 0:
            return []
        if top_k < 1:
            top_k = 1
        normalized = embedding / np.linalg.norm(embedding)
        query = normalized.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, top_k)
        results = []
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue
            row = self._conn.execute(
                "SELECT face_id, name FROM faces WHERE id = ?", (int(faiss_id),)
            ).fetchone()
            if row:
                results.append({
                    "face_id": row[0],
                    "name": row[1],
                    "score": float(score),
                })
        return results
```

- [ ] **Step 2: 验证导入**

```
uv run python -c "from facerecserver.gallery.repository import GalleryRepository; print('ok')"
```

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: add search() method to GalleryRepository"
```

---

### Task 2: 添加 Recognize Schemas

**Files:**
- Modify: `facerecserver/gallery/schemas.py`

- [ ] **Step 1: 在 `facerecserver/gallery/schemas.py` 末尾追加**

```python

class RecognizeRequest(BaseModel):
    image: Optional[str] = None
    image_url: Optional[str] = None


class RecognizeItem(BaseModel):
    face_id: str
    name: str
    score: float
```

- [ ] **Step 2: 验证导入**

```
uv run python -c "from facerecserver.gallery.schemas import RecognizeRequest, RecognizeItem; print('ok')"
```

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: add recognize pydantic schemas"
```

---

### Task 3: 添加识别 API 路由

**Files:**
- Modify: `facerecserver/gallery/routes.py`

- [ ] **Step 1: 在 `router` 定义之后追加识别端点**

```python
@router.post("/recognize", response_model=ApiResponse)
async def recognize_face(request: Request, top_k: int = Query(5, ge=1, le=50)):
    repo = _get_repo(request)
    extractor = _get_extractor(request)

    try:
        content_type = (request.headers.get("content-type") or "").lower()

        if "multipart/form-data" in content_type:
            form = await request.form()
            up_file = form.get("file")
            if up_file is None or not hasattr(up_file, "read"):
                return ApiResponse(code=400, message="请上传图片文件", data=None)
            contents = await up_file.read()
            image = _make_image_from_bytes(contents)
        else:
            body = await request.json()
            if body.get("image"):
                image = base64_to_image(body["image"])
            elif body.get("image_url"):
                resp = requests.get(body["image_url"], timeout=30)
                resp.raise_for_status()
                image = _make_image_from_bytes(resp.content)
            else:
                return ApiResponse(code=400, message="请提供图片 (file, image, 或 image_url)", data=None)

        embedding = extractor.extract(image)
        results = repo.search(embedding, top_k)
        return ApiResponse(code=0, message="success", data={"results": results})

    except FaceNotFoundError as e:
        return ApiResponse(code=1001, message=str(e), data=None)
    except ValueError as e:
        return ApiResponse(code=1002, message=str(e), data=None)
    except Exception as e:
        logger.exception("人脸识别失败")
        return ApiResponse(code=-1, message=f"处理失败: {str(e)}", data=None)
```

注意：`_make_image_from_bytes` 已在 routes.py 中存在，无需重复定义。

- [ ] **Step 2: 验证导入**

```
uv run python -c "from facerecserver.gallery.routes import router; print('ok')"
```

- [ ] **Step 3: Commit**

```
git add -A && git commit -m "feat: add /api/v1/gallery/recognize endpoint"
```

---

### Task 4: 确认路由注册

**Files:**
- （无需修改代码）

`recognize` 端点使用现有 `gallery_router` (prefix `/api/v1/gallery`)，路径为 `POST /api/v1/gallery/recognize`，无需修改 `app.py`。

- [ ] **Step 1: 验证路由已注册**

```
uv run python -c "from facerecserver.app import create_app; app=create_app(); paths=[r.path for r in app.routes]; print([p for p in paths if 'recognize' in p])"
```

预期输出包含 `['/api/v1/gallery/recognize']`

- [ ] **Step 2: （无代码修改，无需提交）**

---

### Task 5: 端到端验证

- [ ] **Step 1: 启动服务**

在一个终端中启动：
```
uv run python -m uvicorn facerecserver.app:create_app --factory --host 127.0.0.1 --port 8010
```

- [ ] **Step 2: 运行测试**

```python
# test_recognize.py
import requests, sys, time

SERVER = "http://127.0.0.1:8010"
URL = SERVER + "/api/v1/gallery/recognize"
TEST_IMG = "test/CYM-13-1_face_1768542709_74.62.jpg"

for i in range(60):
    try:
        r = requests.get(SERVER + "/openapi.json", timeout=2)
        if r.status_code == 200:
            print("Server ready")
            break
    except:
        pass
    time.sleep(2)
else:
    print("Server not ready")
    sys.exit(1)

# 1. 文件上传识别
with open(TEST_IMG, "rb") as f:
    r = requests.post(URL, params={"top_k": 3}, files={"file": f})
    print("Recognize (file):", r.json())
    data = r.json()
    assert data["code"] == 0, f"Recognize failed: {data}"
    assert len(data["data"]["results"]) > 0

# 2. 验证 top_k 有效
r = requests.get(URL, params={"top_k": 50})
print("top_k check:", r.json())

# 3. 空参数
r = requests.post(URL)
print("No image:", r.json())
assert r.json()["code"] == 400

print("All recognize tests passed!")
```

- [ ] **Step 3: 清理测试文件（可选）**

```
Remove-Item test_recognize.py -Force
```

---

### Verification

1. `POST /api/v1/gallery/recognize?top_k=5` 返回结果包含 `face_id`, `name`, `score`
2. 三种输入方式（file/base64/URL）均可正常工作
3. `top_k` 参数正确限制返回条数
4. 空底库返回 `{"results": []}`
5. `top_k` 校验（>=1, <=50）
