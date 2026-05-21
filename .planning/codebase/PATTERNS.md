# Patterns: FaceRecServer

## Established Patterns

### Package Structure
- **Namespace package**: `facerecserver.*` with `__init__.py` in each subpackage
- **CLI entry**: `facerecserver.__main__.py` via `python -m facerecserver`
- **App factory**: `create_app()` in `app.py` returning FastAPI instance

### Configuration
- **YAML-based**: `config.yaml` with Python dataclass mirror
- **Environment overrides**: `FACEREC_CONFIG` env variable for custom path
- **Lazy init**: Config loaded once at startup, stored in `app.state`

### API Design
- **Unified response**: All endpoints return `{"code": int, "message": str, "data": dict|null}`
- **Input flexibility**: 3 input modes per endpoint (file/base64/URL)
- **Error codes**: Named error codes (1001=face not found, 1002=quality fail, etc.)
- **Async handlers**: FastAPI async routes, sync detection/inference via `run_in_executor`

### Data Flow
- **Pipeline pattern**: `detect → align → quality_check → alpha_estimate → model → normalize`
- **Return both**: `extract()` can return `(embedding, face_crop)` for storage
- **L2 normalize**: Embedding always L2-normalized before storage and comparison

### Storage
- **Dual storage**: SQLite for metadata + Faiss for vectors
- **Index persistence**: Faiss index saved to disk after each `add`/`delete`
- **Auto migration**: SQL ALTER TABLE wrapped in try/except for schema evolution
- **Image thumbnails**: Crop images resized to 360px max dimension, JPEG q85

### Testing
- **pytest fixtures**: `conftest.py` provides shared test gallery, sample data
- **Separate test dirs**: `tests/*.py` mirroring source structure
- **Temp directory**: Tests use temporary gallery directories to avoid side effects

### Frontend
- **SPA pattern**: Vue 3 SPA with catch-all route served by FastAPI
- **API proxy**: Vite dev server proxies `/api` to backend
- **No state management**: Per-component reactive state (no Pinia/Vuex)
- **Fetch-based client**: Custom `api/client.ts` wrapping `fetch()`

### Code Conventions
- **Python**: PEP 8, no auto-formatter, f-strings preferred, type hints used
- **TS**: TypeScript strict mode, interfaces for all data types
- **Imports**: stdlib → third-party → local (3 groups with blank line)
- **Error handling**: Wrap route handlers in try/except, return ApiResponse with code

### Naming Conventions
- **Routes**: `/{resource}` for CRUD, `/{resource}/recognize` for search
- **Files**: `repository.py` for data access, `routes.py` for endpoints, `schemas.py` for models
- **Config keys**: snake_case, nested YAML matching dataclass hierarchy
