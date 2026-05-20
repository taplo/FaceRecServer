# Phase 1: PETALface 模型集成 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-20
**Phase:** 1-PETALface 模型集成
**Areas discussed:** 模型获取与格式, 人脸检测器选择, API 接口设计, 图像预处理方式, 项目模块结构

---

## 模型获取与格式

| Option | Description | Selected |
|--------|-------------|----------|
| HuggingFace 直接加载 | 直接从 HuggingFace hub 下载 PyTorch 权重 | |
| 本地下载后加载 | 先下载模型文件到本地，离线加载 | ✓ |
| 转 ONNX 格式 | 转换成 ONNX 再加载 | |

**User's choice:** 本地下载后加载
**Notes:** 手动下载，提供文档

| Option | Description | Selected |
|--------|-------------|----------|
| models/ 目录 | 项目目录内的 models/ 文件夹 | ✓ |
| 可配置路径 | 通过环境变量 MODEL_PATH 指定路径 | |
| 先 models/ 再改配置 | 先放 models/，后续再支持配置 | |

**User's choice:** models/ 目录

| Option | Description | Selected |
|--------|-------------|----------|
| 固定版本 | 固定一个模型版本，直接硬编码模型文件路径 | |
| 多版本可切换 | 支持配置文件指定不同模型权重，方便切换 | ✓ |

**User's choice:** 多版本可切换

| Option | Description | Selected |
|--------|-------------|----------|
| 启动时自动下载 | 启动时提供脚本/命令自动从 HuggingFace 下载 | |
| 手动下载 | 提供文档说明手动下载步骤 | ✓ |

**User's choice:** 手动下载

---

## 人脸检测器选择

| Option | Description | Selected |
|--------|-------------|----------|
| OpenCV Haar Cascade | 轻量级，CPU 友好 | |
| MTCNN | 精度较高，支持人脸关键点检测和对齐 | ✓ |
| RetinaFace | 精度最高，但较重 | |
| InsightFace | insightface 库，功能全面 | |

**User's choice:** MTCNN
**Notes:** 推荐选项

| Option | Description | Selected |
|--------|-------------|----------|
| 取最大人脸 | 只检测最大的人脸 | ✓ |
| 检测所有人脸 | 检测所有人脸 | |
| 可配置 | 由 API 参数控制行为 | |

**User's choice:** 取最大人脸

| Option | Description | Selected |
|--------|-------------|----------|
| facenet-pytorch | facenet-pytorch 中的 MTCNN 实现 | |
| mtcnn 包 | mtcnn 包（基于 TensorFlow/OpenCV） | ✓ |

**User's choice:** mtcnn 包

| Option | Description | Selected |
|--------|-------------|----------|
| 做对齐 | 检测后做仿射变换对齐人脸 | ✓ |
| 不对齐 | 仅检测裁切，不做对齐 | |

**User's choice:** 做对齐

---

## API 接口设计

| Option | Description | Selected |
|--------|-------------|----------|
| POST /api/v1/embedding | RESTful 风格 | ✓ |
| POST /api/v1/extract-embedding | RPC 风格 | |
| POST /api/v1/face/embedding | 直接根路径 | |

**User's choice:** POST /api/v1/embedding

| Option | Description | Selected |
|--------|-------------|----------|
| 全支持 | URL + Base64 + File 全支持 | ✓ |
| File + Base64 | 只支持 File 上传和 Base64 | |
| 仅 File | 只支持 File 上传 | |

**User's choice:** 全支持

| Option | Description | Selected |
|--------|-------------|----------|
| {code, message, data} | 标准 RESTful | ✓ |
| 直接返回 embedding | 直接返回结果数据 | |
| 让 FastAPI 自动处理 | FastAPI 默认自动文档 | |

**User's choice:** {code, message, data}

---

## 图像预处理方式

| Option | Description | Selected |
|--------|-------------|----------|
| 112x112 | 主流人脸识别尺寸 | ✓ |
| 160x160 | 保留更多细节 | |
| 按论文默认 | 跟 PETALface 原文保持一致 | |

**User's choice:** 112x112

| Option | Description | Selected |
|--------|-------------|----------|
| 返回错误 | 抛出明确错误 | ✓ |
| 返回空结果 | 返回空 embedding 或 null | |
| 可配置 | 可配置策略 | |

**User's choice:** 返回错误

| Option | Description | Selected |
|--------|-------------|----------|
| 不做质量检查 | 只靠检测器过滤 | |
| 简单质量检查 | 亮度、对比度过滤 | ✓ |

**User's choice:** 简单质量检查

---

## 项目模块结构

| Option | Description | Selected |
|--------|-------------|----------|
| 按功能分层 | api/, core/, models/, utils/ | |
| 按领域组织 | face_detection/, face_recognition/, gallery/ | ✓ |
| 扁平结构 | 所有模块在根目录 | |

**User's choice:** 按领域组织

| Option | Description | Selected |
|--------|-------------|----------|
| facerecserver | 与 pyproject.toml 一致 | ✓ |
| faceapi | 更简短 | |

**User's choice:** facerecserver

| Option | Description | Selected |
|--------|-------------|----------|
| YAML 配置文件 | 支持 config.yaml 文件 | ✓ |
| 环境变量 | 仅通过环境变量配置 | |
| Python 硬编码 | 简单 settings.py 硬编码 | |

**User's choice:** YAML 配置文件

---

## the agent's Discretion

- FastAPI 应用初始化方式（app factory vs 全局实例）
- 日志框架选择
- 依赖管理细节
- 错误处理具体实现

## Deferred Ideas

None
