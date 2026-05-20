# Requirements: FaceRecServer

**Defined:** 2026-05-20
**Core Value:** 提供基于 PETALface 算法的、可直接通过 API 使用的人脸识别服务

## v1 Requirements

### 人脸底库管理 (Gallery)

- [ ] **GALLERY-01**: 用户可以通过 API 单张上传人脸到底库
- [ ] **GALLERY-02**: 用户可以通过 ZIP 批量上传人脸到底库
- [ ] **GALLERY-03**: 用户可以从底库删除指定人脸
- [ ] **GALLERY-04**: 用户可以查看底库人脸列表及详情
- [ ] **GALLERY-05**: 用户可以清空整个底库
- [ ] **GALLERY-06**: 底库人脸信息支持中文名称

### API 服务 (API)

- [ ] **API-01**: 基于 PETALface 模型提取人脸特征向量 (Embedding)
- [ ] **API-02**: 提供人脸比对 API (1:1)，返回相似度分数
- [ ] **API-03**: 提供人脸识别 API (1:N)，返回最匹配的人脸
- [ ] **API-04**: 支持图片 URL 和 Base64 两种输入方式
- [ ] **API-05**: API 响应统一格式，包含状态码和消息

### Web 管理后台 (WEB)

- [ ] **WEB-01**: 基于 Vue 3 的管理后台界面
- [ ] **WEB-02**: 底库管理页面（上传、删除、查询、清空）
- [ ] **WEB-03**: 人脸比对测试页面（上传两张图对比）
- [ ] **WEB-04**: 人脸识别测试页面（上传待识别人脸）
- [ ] **WEB-05**: 全部界面使用中文

### 系统能力 (SYS)

- [ ] **SYS-01**: 系统支持中文文件名和解压路径
- [ ] **SYS-02**: 基于 CPU 运行，通过 `torch.cuda.is_available()` 保留 GPU 支持
- [ ] **SYS-03**: 模型加载和推理性能优化（小规模底库）

## v2 Requirements

Deferred to future release.

- **AUTH-01**: API 鉴权（JWT/API Key）
- **AUTH-02**: 管理后台登录
- **PERF-01**: GPU 加速推理
- **PERF-02**: 批量识别性能优化

## Out of Scope

| Feature | Reason |
|---------|--------|
| 用户鉴权/权限管理 | v1 聚焦核心功能，后续再添加 |
| GPU 加速推理 | 当前仅 CPU，保留 CUDA 检测能力 |
| 人脸检测/质量评估 | PETALface 仅做识别，检测需额外模型 |
| 模型训练/微调 | 仅做推理服务化，不涉及训练 |
| 移动端 APP | 仅提供 Web 管理后台和 API |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 | Phase 1 | Pending |
| SYS-02 | Phase 1 | Pending |
| SYS-03 | Phase 1 | Pending |
| GALLERY-01 | Phase 2 | Pending |
| GALLERY-02 | Phase 2 | Pending |
| GALLERY-03 | Phase 2 | Pending |
| GALLERY-04 | Phase 2 | Pending |
| GALLERY-05 | Phase 2 | Pending |
| GALLERY-06 | Phase 2 | Pending |
| SYS-01 | Phase 2 | Pending |
| API-02 | Phase 3 | Pending |
| API-03 | Phase 3 | Pending |
| API-04 | Phase 3 | Pending |
| API-05 | Phase 3 | Pending |
| WEB-01 | Phase 4 | Pending |
| WEB-02 | Phase 4 | Pending |
| WEB-03 | Phase 4 | Pending |
| WEB-04 | Phase 4 | Pending |
| WEB-05 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-20*
*Last updated: 2026-05-20 after initialization*
