# FaceRecServer

## What This Is

基于 PETALface (WACV 2025) 算法的人脸识别 API 服务，提供人脸底库管理、人脸比对 (1:1)、人脸识别 (1:N) 等功能，并附带基于 Web 的管理后台。面向需要集成人脸识别能力的公共服务场景。

## Core Value

提供基于 PETALface 算法的、可直接通过 API 使用的人脸识别服务，让任何人都能方便地集成人脸识别能力。

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] **FACE-API-01**: 基于 PETALface 算法的 Face Embedding 提取服务
- [ ] **FACE-API-02**: 人脸比对 API (1:1)
- [ ] **FACE-API-03**: 人脸识别 API (1:N)
- [ ] **GALLERY-01**: 单张人脸图片上传到底库
- [ ] **GALLERY-02**: ZIP 批量上传人脸到底库
- [ ] **GALLERY-03**: 从底库删除人脸
- [ ] **GALLERY-04**: 查看底库列表/查询
- [ ] **GALLERY-05**: 清空底库
- [ ] **WEB-01**: 基于 Vue 3 的管理后台
- [ ] **WEB-02**: 后台支持中文界面
- [ ] **SYS-01**: 文件操作支持中文文件名
- [ ] **SYS-02**: 基于 CPU 运行，保留 GPU 支持能力

### Out of Scope

- **用户鉴权/权限管理** — 初期不做，后续可按需添加
- **GPU 加速** — 保留 torch.cuda 检测能力，但初期仅 CPU 运行

## Context

- 基于 PETALface 算法（WACV 2025 Oral），使用 LoRA + PEFT 的低分辨率人脸识别技术
- Python 3.12，使用 uv 包管理
- FastAPI 提供 API 服务
- Vue 3 独立前端（前后端分离）
- 预期底库规模 < 1 万张人脸
- 面向公共服务场景
- 项目当前为空白脚手架，尚无实际功能代码

## Constraints

- **Tech Stack**: Python >=3.12 + FastAPI + Vue 3 前后端分离
- **Hardware**: CPU only (通过 torch.cuda.is_available() 检测 GPU 支持)
- **Language**: 全中文支持（界面 + 文件路径/名称）
- **Algorithm**: 基于 PETALface 开源实现，需研究其模型结构并进行推理服务化

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 前后端分离 | 管理后台与 API 解耦，便于独立开发和扩展 | — Pending |
| Vue 3 | 生态成熟，社区活跃，用户明确选择 | — Pending |
| 无鉴权 | 开发阶段简化，后续按需添加 | — Pending |
| CPU 优先 | 当前硬件限制，保留 CUDA 检测逻辑 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-20 after initialization*
