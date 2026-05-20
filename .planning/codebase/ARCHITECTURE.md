# Architecture: FaceRecServer

## Project Type
Python CLI/Server scaffold — currently a minimal entry point with no real logic.

## File Map

| Path | Purpose |
|------|---------|
| `main.py` | Entry point with `main()` stub (prints "Hello from facerecserver!") |
| `pyproject.toml` | Project metadata, Python 3.12+, no dependencies |
| `.python-version` | Python version pin (3.12) |
| `README.md` | Empty |

## Current State
Scaffold only — no server logic, no API, no face recognition functionality.

## Patterns
- Uses `uv` for project management (pyproject.toml, no requirements.txt)
- Python 3.12 minimum

## What's Missing (for a face recognition server)
- HTTP server/framework (FastAPI recommended)
- Face detection library (OpenCV, face_recognition, etc.)
- Image processing pipeline
- Storage/DB for face data
- API endpoints for registration, recognition, management
