# Phase 1 Research: PETALface 模型集成

## RESEARCH COMPLETE

## PETALface 模型概述

- **论文**: WACV 2025 Oral — Parameter Efficient Transfer Learning for Low-Resolution Face Recognition
- **架构**: Swin Transformer backbone + 双 LoRA 模块（高低分辨率自适应）
- **输入尺寸**: 120x120（训练），112x112（推理可接受）
- **Loss**: ArcFace / CosFace
- **LoRA rank**: 8 (TinyFace) / 32 (BRIAR)
- **特征维度**: 512-d embedding

## 模型权重下载

HuggingFace 仓库: `kartiknarayan/PETALface`

可用模型：
| 模型 | 预训练数据 | Loss | 说明 |
|------|-----------|------|------|
| swin_arcface_webface4m | WebFace4M | ArcFace | 基础预训练 |
| swin_cosface_webface4m | WebFace4M | CosFace | 基础预训练 |
| swin_arcface_webface4m_tinyface | WebFace4M + TinyFace | ArcFace | 微调低分辨率 |
| swin_cosface_webface4m_tinyface | WebFace4M + TinyFace | CosFace | 微调低分辨率 |
| swin_cosface_webface4m_briar | WebFace4M + BRIAR | CosFace | 微调 |
| swin_arcface_webface12m | WebFace12M | ArcFace | 大数据预训练 |
| swin_cosface_webface12m | WebFace12M | CosFace | 大数据预训练 |
| swin_cosface_webface12m_briar | WebFace12M + BRIAR | CosFace | 大数据微调 |

**建议**: 使用 `swin_arcface_webface4m_tinyface`（通用性较好）

## 依赖分析

| 包 | 用途 | 版本建议 |
|---|------|---------|
| torch | 深度学习框架 | >=2.0 |
| torchvision | 图像处理 | >=0.15 |
| mtcnn | 人脸检测 | >=0.1.1 |
| numpy | 数值计算 | >=1.24 |
| opencv-python | 图像IO | >=4.8 |
| pillow | 图像处理 | >=10.0 |
| fastapi | API框架 | >=0.104 |
| uvicorn | ASGI服务器 | >=0.24 |
| python-multipart | 文件上传 | >=0.0.6 |
| pyyaml | 配置文件读取 | >=6.0 |
| huggingface-hub | 模型下载工具 | >=0.20 |

## 人脸检测方案

MTCNN (`mtcnn` 包):
- 优点: CPU 友好，支持关键点检测，可用于对齐
- 速度: ~5 FPS on CPU (720p)
- 输出: 边界框 + 5个关键点（眼睛、鼻子、嘴角）

## 推理流程

1. 读取图片 → OpenCV/PIL
2. MTCNN 检测人脸 → 取最大人脸
3. 人脸对齐（基于关键点仿射变换）
4. 缩放至 112x112
5. 归一化（减均值除方差）
6. PETALface 模型前向 → 512-d embedding

## 项目结构参考

参考类似项目实践（InsightFace-REST、face-recognition-service）:

```
facerecserver/
├── __init__.py
├── __main__.py          # python -m facerecserver 入口
├── config/
│   └── config.yaml      # YAML 配置文件
├── api/
│   └── routes.py        # FastAPI 路由
├── face_detection/
│   ├── detector.py      # MTCNN 检测封装
│   └── aligner.py       # 人脸对齐
├── face_recognition/
│   ├── model.py         # PETALface 模型加载
│   ├── embedding.py     # 特征提取
│   └── utils.py         # 工具函数
├── models/              # 模型权重存放目录
└── app.py               # FastAPI 应用工厂
```

## 风险与注意事项

- **模型结构**: PETALface 使用自定义 Swin + LoRA 架构，需研究 GitHub 源码确定模型定义
- **LoRA 加载**: 推理时需要正确加载 LoRA 权重到 backbone
- **IQA 网络**: 论文使用 CNN-IQA 做质量评估，推理时可以用固定权重或简化处理
- **CPU 性能**: Swin Transformer 在 CPU 上较慢，单次推理预估 500ms-2s
- **中文路径**: OpenCV 不支持中文路径，需用 PIL/PIL 或 numpy 方式读取
