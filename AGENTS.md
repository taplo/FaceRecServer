<!-- GSD:project-start source:PROJECT.md -->
## Project

**FaceRecServer**

基于 PETALface (WACV 2025) 算法的人脸识别 API 服务，提供人脸底库管理、人脸比对 (1:1)、人脸识别 (1:N) 等功能，并附带基于 Web 的管理后台。面向需要集成人脸识别能力的公共服务场景。

**Core Value:** 提供基于 PETALface 算法的、可直接通过 API 使用的人脸识别服务，让任何人都能方便地集成人脸识别能力。

### Constraints

- **Tech Stack**: Python >=3.12 + FastAPI + Vue 3 前后端分离
- **Hardware**: CPU only (通过 torch.cuda.is_available() 检测 GPU 支持)
- **Language**: 全中文支持（界面 + 文件路径/名称）
- **Algorithm**: 基于 PETALface 开源实现，需研究其模型结构并进行推理服务化
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Current Stack
| Component | Choice | Version |
|-----------|--------|---------|
| Language | Python | >=3.12 |
| Package Manager | uv | latest |
| Entry Format | `python -m facerecserver` | via pyproject.toml |
## No Existing Dependencies
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Project Type
## File Map
| Path | Purpose |
|------|---------|
| `main.py` | Entry point with `main()` stub (prints "Hello from facerecserver!") |
| `pyproject.toml` | Project metadata, Python 3.12+, no dependencies |
| `.python-version` | Python version pin (3.12) |
| `README.md` | Empty |
## Current State
## Patterns
- Uses `uv` for project management (pyproject.toml, no requirements.txt)
- Python 3.12 minimum
## What's Missing (for a face recognition server)
- HTTP server/framework (FastAPI recommended)
- Face detection library (OpenCV, face_recognition, etc.)
- Image processing pipeline
- Storage/DB for face data
- API endpoints for registration, recognition, management
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
