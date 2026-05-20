# Phase 1: PETALface 模型集成 - Context

**Gathered:** 2026-05-20
**Status:** Ready for planning

<domain>
## Phase Boundary

加载 PETALface 预训练模型，实现人脸特征向量提取核心能力。这是所有后续阶段的基础。

**Requirements covered:**
- API-01: 基于 PETALface 模型提取人脸特征向量 (Embedding)
- SYS-02: 基于 CPU 运行，通过 `torch.cuda.is_available()` 保留 GPU 支持
- SYS-03: 模型加载和推理性能优化（小规模底库）

</domain>

<decisions>
## Implementation Decisions

### 模型获取与格式
- **D-01:** 模型从 HuggingFace (`kartiknarayan/PETALface`) 手动下载到本地 `models/` 目录
- **D-02:** 支持多版本模型切换，通过配置文件指定模型路径
- **D-03:** 提供文档说明手动下载步骤，启动时不做自动下载
- **D-04:** 模型格式为 PyTorch 权重，不做 ONNX 转换

### 人脸检测器选择
- **D-05:** 使用 `mtcnn` 包做人脸检测
- **D-06:** 多人脸场景取最大人脸
- **D-07:** MTCNN 检测到关键点后做仿射变换对齐

### API 接口设计
- **D-08:** 端点 `POST /api/v1/embedding`
- **D-09:** 支持图片 URL、Base64、File 三种输入
- **D-10:** 统一响应格式 `{code, message, data}`
- **D-11:** 检测不到人脸时返回错误

### 图像预处理方式
- **D-12:** 人脸图片缩放到 112x112
- **D-13:** 做简单质量检查（亮度、对比度过滤）

### 项目模块结构
- **D-14:** 按领域组织，包名 `facerecserver`
- **D-15:** 配置管理使用 YAML 配置文件

### the agent's Discretion
- FastAPI 应用初始化方式（app factory vs 全局实例）
- 日志框架选择
- 依赖管理细节
- 错误处理具体实现

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project docs
- `.planning/PROJECT.md` — 项目整体上下文和约束
- `.planning/REQUIREMENTS.md` — v1 需求定义（Phase 1 覆盖 API-01, SYS-02, SYS-03）
- `.planning/ROADMAP.md` — Phase 1 目标和成功标准
- `.planning/codebase/STACK.md` — 当前技术栈
- `.planning/codebase/ARCHITECTURE.md` — 当前架构

### External references
- `https://github.com/Kartik-3004/PETALface` — PETALface 算法源码
- `https://huggingface.co/kartiknarayan/PETALface` — 预训练模型权重
- PETALface 论文: https://arxiv.org/abs/2412.07771

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- 项目为空白脚手架，无可复用资产

### Established Patterns
- 使用 `uv` 包管理，入口为 `python -m facerecserver`

### Integration Points
- `main.py` 需要重构为 FastAPI 应用入口
- `pyproject.toml` 需要添加新依赖

</code_context>

<specifics>
## Specific Ideas

- MTCNN 使用 `mtcnn` 包（非 `facenet-pytorch`）
- 模型切换通过 YAML 配置中的 `model.path` 字段控制
- 图片质量检查用简单方法（亮度均值、拉普拉斯方差）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 1-PETALface 模型集成*
*Context gathered: 2026-05-20*
