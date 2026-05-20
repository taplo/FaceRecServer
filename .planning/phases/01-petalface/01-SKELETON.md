# Walking Skeleton: FaceRecServer Phase 1

## The Thinnest Working Slice

```
用户上传图片 → FastAPI → MTCNN检测 → PETALface推理 → 返回embedding
```

## What It Proves

1. **FastAPI 服务可启动** — `python -m facerecserver` 正常启动
2. **MTCNN 人脸检测可用** — 能检测并对齐人脸
3. **PETALface 模型可加载** — 在 CPU 上正常加载权重
4. **端到端推理可行** — 输入图片 → 输出 512-d embedding
5. **中文路径支持** — 中文文件名和解压路径正常
6. **GPU 检测** — `torch.cuda.is_available()` 正确检测

## Deliverable

一个可运行的 REST API: `POST /api/v1/embedding`

## Non-Goals (for this skeleton)

- 人脸底库管理（Phase 2）
- 人脸比对/识别（Phase 3）
- Web 管理后台（Phase 4）
- 高吞吐优化
- 完整测试覆盖
