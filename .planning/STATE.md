# Project State: FaceRecServer

## Current Phase
Phase 1 — Plan 01-01 executed (Walking Skeleton complete)

## Project Reference
See: .planning/PROJECT.md (updated 2026-05-20)

**Core value:** 提供基于 PETALface 算法的、可直接通过 API 使用的人脸识别服务
**Current focus:** Phase 1 - PETALface 模型集成

## Phase History

| Phase | Status | Date |
|-------|--------|------|
| 1 | Plan 01-01 Executed | 2026-05-20 |

## Milestones

| Milestone | Status | Date |
|-----------|--------|------|
| Walking Skeleton (Phase 1) | Complete | 2026-05-20 |

## Active Decisions

| ID | Decision | Status |
|----|----------|--------|
| D-01 | 使用 facenet-pytorch MTCNN 替代 mtcnn (tensorflow 依赖) | Active |
| D-02 | 模型权重通过 huggingface-hub 手动下载 | Active |
| D-03 | PETALface 模型定义从 GitHub 源码移植 | Active |
| D-04 | alpha 参数基于 Laplacian variance 自动估计 | Active |

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| HuggingFace 网络连接受限 | Medium | Open |
| timm deprecation warning | Low | Open |
