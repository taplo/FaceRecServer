# FaceRecServer 使用手册

## 目录

1. [概述](#1-概述)
2. [安装与部署](#2-安装与部署)
3. [配置详解](#3-配置详解)
4. [Web 管理后台](#4-web-管理后台)
5. [API 使用指南](#5-api-使用指南)
6. [底库导入与管理](#6-底库导入与管理)
7. [人脸识别最佳实践](#7-人脸识别最佳实践)
8. [常见问题排查](#8-常见问题排查)
9. [开发指南](#9-开发指南)

---

## 1. 概述

FaceRecServer 是基于 PETALface (WACV 2025) 算法的人脸识别服务。

### 核心能力

| 能力 | 说明 |
|------|------|
| 人脸检测 | MTCNN 算法，支持多角度、多尺度检测 |
| 特征提取 | PETALface Swin-Tiny，输出 512 维特征向量 |
| 图像质量评估 | CNN-IQA 双分支架构，低质量图片自动降权 |
| 1:1 比对 | 计算两张人脸的余弦相似度 |
| 1:N 搜索 | 在底库中搜索最相似的人脸，支持 Top-K |
| 底库管理 | 增删查改、ZIP 批量导入、CSV 报表导出 |

### 系统架构

```
用户请求 → FastAPI → MTCNN 检测 → 关键点对齐 → 质量评估 → PETALface 推理 → SQLite + Faiss
               ↓                                                        ↓
           Web 后台 ← 响应 ←—————————————————————————————————————————— 结果返回
```

- **检测层**: MTCNN 检测人脸边界框和 5 个关键点
- **对齐层**: 基于关键点的仿射变换，对齐到标准姿态
- **质量评估**: Laplacian 模糊检测 + CNN-IQA 深度学习评分
- **推理层**: PETALface Swin-Tiny 双分支架构，输出 L2 归一化特征
- **检索层**: Faiss IndexFlatIP 进行向量相似度搜索
- **存储层**: SQLite 存储元数据 + Faiss 文件存储索引

### 关于 PETALface 双分支架构

PETALface 使用 LoRA 双分支设计：

- **身份分支** (LoRALinear): 提取身份相关特征
- **质量分支** (LoRALinearTwo): 受图像质量影响更大
- **融合**: `output = alpha × quality_branch + (1 - alpha) × identity_branch`
- **alpha**: 由 CNN-IQA 模型估计，质量越好 alpha 越大

这种设计使得低质量人脸（模糊、暗光、大角度）更多地依赖身份特征，高质量人脸充分利用质量分支的辨别力。

---

## 2. 安装与部署

### 2.1 环境要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| CPU | 支持 AVX2 指令集 | x86-64 4 核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 5 GB 可用空间 | 20 GB+（SSD 更佳） |
| GPU | 可选（CUDA） | NVIDIA 6 GB+ |
| Python | 3.12 | 3.12+ |

> **磁盘说明**: 仅模型文件约 1.6 GB（含 PETALface 默认模型约 800 MB + CNN-IQA 质量评估模型约 800 MB）。人脸照片底库按每张 10-50 KB 估算，6949 人约 200 MB。如需测试多个模型或处理更多底库照片，磁盘需求会进一步增加。

### 2.2 安装步骤

```bash
# 1. 安装 Python 3.12+ 和 uv
pip install uv

# 2. 克隆或复制项目
# 3. 安装项目依赖
cd FaceRecServer
uv sync

# 4. 下载模型（默认模型约 200 MB）
uv run python scripts/download_model.py --model swin_arcface_webface4m_tinyface

# 5. 启动服务
uv run python -m facerecserver
```

### 2.3 启动方式

#### 生产模式

```bash
uv run python -m facerecserver
```

监听 `0.0.0.0:8000`。

#### 开发模式（热加载）

```bash
uv run uvicorn facerecserver.app:create_app --factory --reload --port 8000
```

#### 指定自定义配置

```bash
FACEREC_CONFIG=/path/to/my-config.yaml uv run python -m facerecserver
```

#### 进程管理（Windows）

```bash
# 后台启动（PowerShell）
Start-Process -FilePath "uv" -ArgumentList "run python -m facerecserver" -WorkingDirectory "D:\projects\FaceRecServer" -WindowStyle Hidden

# 查看进程
Get-Process -Name python

# 停止
Stop-Process -Name python -Force
```

### 2.4 首次启动验证

服务启动后，用浏览器或 curl 验证：

```bash
# API 文档
http://localhost:8000/docs

# 系统统计
curl http://localhost:8000/api/v1/stats

# 预期输出:
# {"code":0,"message":"success","data":{"gallery":{"total_faces":0,"index_size":0,"dimension":512},"server":{"uptime_seconds":5,"device":"cpu"}}}
```

### 2.5 前端构建

```bash
cd frontend
npm install
npm run build   # 生产构建
# npm run dev   # 开发模式（热更新，API 自动代理到 8000）
```

构建后访问 `http://localhost:8000` 即可打开管理后台。

---

## 3. 配置详解

### 3.1 配置文件

默认位置: `facerecserver/config.yaml`

```yaml
model:
  path: models/swin_arcface_webface4m_tinyface/model.pt
  name: swin_arcface_webface4m_tinyface
  lora_rank: 8
  lora_scale: 1.0
  use_lora: true

detection:
  confidence: 0.95
  min_face_size: 40

preprocess:
  image_size: 112
  do_alignment: true
  do_quality_check: true
  iqa:
    enabled: true
    threshold: 0.5

server:
  host: 0.0.0.0
  port: 8000

gallery:
  db_dir: gallery
  db_name: faces
  page_size_default: 20
  page_size_max: 100
```

### 3.2 配置项参考

#### Model（模型）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `path` | `models/.../model.pt` | 模型文件路径 |
| `name` | `swin_arcface_webface4m_tinyface` | 模型注册名称 |
| `lora_rank` | `8` | LoRA 低秩矩阵的秩 |
| `lora_scale` | `1.0` | LoRA 输出缩放因子 |
| `use_lora` | `true` | 是否启用 LoRA 双分支 |

可用模型列表（下载脚本）:

```bash
uv run python scripts/download_model.py --list
```

| 模型 | 训练数据 | 特征 | 适用场景 |
|------|----------|------|----------|
| `swin_arcface_webface4m_tinyface` | WebFace4M + TinyFace | 512-d | 通用（默认） |
| `swin_cosface_webface4m_tinyface` | WebFace4M + TinyFace | 512-d | 低质量/跨域 |
| `swin_arcface_webface4m` | WebFace4M | 512-d | 高质量受控场景 |
| `swin_cosface_webface12m` | WebFace12M | 512-d | 大规模 |

#### Detection（检测）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `confidence` | `0.95` | MTCNN 检测置信度阈值。越低召回越高，但可能误检 |
| `min_face_size` | `40` | 最小人脸边长(像素)。过小可能误检背景 |

调优建议：

- **高精度场景**（门禁、闸机）: `confidence: 0.98`, `min_face_size: 60`
- **高召回场景**（批量导入、监控）: `confidence: 0.90`, `min_face_size: 30`

#### Preprocess（预处理）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_size` | `112` | 模型输入尺寸。PETALface 固定为 112x112 |
| `do_alignment` | `true` | 基于 5 关键点的仿射对齐。建议开启 |
| `do_quality_check` | `true` | 前置质量过滤（Laplacian 模糊检测 + 亮度检查） |

质量检查规则：
- Laplacian 方差 < 100 → "图片模糊"
- 平均亮度 < 30 或 > 240 → "图片过暗/过亮"

#### IQA（图像质量评估）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用 CNN-IQA 评分 |
| `threshold` | `0.5` | alpha 基准值。`alpha = 0.5 + (score - threshold)`，clip 到 [0,1] |

影响：
- `threshold` 越低 → alpha 越大 → 更依赖质量分支
- `threshold` 越高 → alpha 越小 → 更依赖身份分支
- 批量导入低质量图片时建议 `threshold: 0.3`

#### Server（服务）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `0.0.0.0` | 监听地址 |
| `port` | `8000` | 监听端口 |

#### Gallery（底库）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `db_dir` | `gallery` | 数据库和索引文件存储目录 |
| `db_name` | `faces` | 数据库名称（生成 `faces.db`, `faces.faiss`） |
| `page_size_default` | `20` | 分页默认每页条数 |
| `page_size_max` | `100` | 分页最大每页条数 |

---

## 4. Web 管理后台

管理后台提供 4 个功能页面。

### 4.1 仪表盘 (/dashboard)

显示系统概览：
- **底库人脸总数** — SQLite 中人脸记录数
- **特征维度** — 固定 512 维
- **服务器运行时间** — 自启动以来的时长
- **运行设备** — CPU 或 CUDA

### 4.2 底库管理 (/gallery)

人脸底库的完整 CRUD 操作：

- **搜索** — 按姓名或工号搜索，支持模糊匹配
- **列表** — 分页展示，每页 20 条，选中查看详情
- **注册** — 点击"+ 注册人脸"，选择图片，确认注册
- **删除** — 列表行直接删除，或选中后从详情面板删除
- **清空** — 清空整个底库（不可恢复，需确认）

每条记录显示：头像、姓名、工号、ID、注册时间。

### 4.3 人脸识别 (/recognize)

#### 1:1 比对（左侧面板）

1. 分别上传两张图片（图片 A 和图片 B）
2. 点击"开始比对"
3. 系统显示相似度分数和颜色评级

注意：1:1 比对通过服务端"注册临时人脸 → 搜索 → 删除临时记录"的方式实现。

#### 1:N 搜索（右侧面板）

1. 上传查询图片
2. 选择 Top-K 值（5/10/20）
3. 点击"开始搜索"
4. 结果按相似度降序排列，显示名次、姓名、工号、分数、头像

### 4.4 系统设置 (/settings)

- **模型信息** — 当前使用的模型名称、路径、维度
- **底库统计** — 底库总数、索引大小、特征维度
- **重建索引** — 提示重启服务
- **清空底库** — 确认后清空
- **关于信息** — 版本号、运行设备、运行时间

---

## 5. API 使用指南

### 5.1 通用约定

**Base URL**: `http://localhost:8000/api/v1`

**响应格式**:
```json
{"code": 0, "message": "success", "data": { ... }}
```

**错误码**:

| code | message | 说明 |
|------|---------|------|
| 0 | success | 成功 |
| 400 | 请提供图片 | 请求未包含图片 |
| 1001 | 未检测到人脸 | 图片中无人脸或置信度过低 |
| 1002 | 图片模糊/过暗 | 质量检查未通过 |
| 2002 | 人脸不存在 | face_id 无效 |
| 5000 | 模型未加载/底库未初始化 | 服务未准备好 |
| -1 | 处理失败: ... | 服务器内部错误 |

### 5.2 图片输入方式

所有需要图片输入的端点（embedding、register、recognize）都支持三种方式：

**方式 1: 文件上传** (multipart/form-data)
```bash
curl -X POST http://localhost:8000/api/v1/gallery \
  -F "file=@photo.jpg"
```

**方式 2: Base64 编码** (application/json)
```bash
curl -X POST http://localhost:8000/api/v1/gallery \
  -H "Content-Type: application/json" \
  -d '{"image": "/9j/4AAQ...base64data...", "name": "张三", "employee_id": "001"}'
```

**方式 3: 图片 URL** (application/json)
```bash
curl -X POST http://localhost:8000/api/v1/gallery \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg"}'
```

### 5.3 端点速查

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/api/v1/embedding` | 提取人脸特征向量 |
| GET | `/api/v1/stats` | 系统统计信息 |
| POST | `/api/v1/gallery` | 注册单张人脸 |
| POST | `/api/v1/gallery/batch` | ZIP 批量注册 |
| GET | `/api/v1/gallery` | 列出底库（分页+搜索） |
| GET | `/api/v1/gallery/{face_id}/image` | 获取人脸图片 |
| DELETE | `/api/v1/gallery/{face_id}` | 删除单条记录 |
| DELETE | `/api/v1/gallery` | 清空整个底库 |
| POST | `/api/v1/gallery/recognize` | 1:N 人脸搜索 |

### 5.4 提取特征 → 1:1 比对

两步实现 1:1 比对：

```python
import requests
import numpy as np

SERVER = "http://localhost:8000"

def extract_embedding(image_path):
    with open(image_path, "rb") as f:
        resp = requests.post(f"{SERVER}/api/v1/embedding", files={"file": f})
    data = resp.json()
    if data["code"] != 0:
        raise Exception(data["message"])
    return np.array(data["data"]["embedding"])

emb_a = extract_embedding("person_a.jpg")
emb_b = extract_embedding("person_b.jpg")

# 余弦相似度
similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
print(f"相似度: {similarity:.4f}")
assert similarity >= -1.0 and similarity <= 1.0
```

### 5.5 注册 → 搜索（完整流程）

```python
import requests

SERVER = "http://localhost:8000"

# 第一步: 批量注册
with open("employees.zip", "rb") as f:
    resp = requests.post(f"{SERVER}/api/v1/gallery/batch", files={"file": f})
result = resp.json()
print(f"注册完成: {result['data']['succeeded']} 成功, {result['data']['failed']} 失败")

# 第二步: 搜索
with open("query.jpg", "rb") as f:
    resp = requests.post(f"{SERVER}/api/v1/gallery/recognize?top_k=3", files={"file": f})
result = resp.json()
for r in result["data"]["results"]:
    print(f"姓名: {r['name']}, 工号: {r['employee_id']}, 相似度: {r['score']:.4f}")

# 第三步: 获取匹配的人脸图片
import requests as req
face_id = result["data"]["results"][0]["face_id"]
img_resp = req.get(f"{SERVER}/api/v1/gallery/{face_id}/image")
with open("matched.jpg", "wb") as f:
    f.write(img_resp.content)
```

### 5.6 搜索阈值建议

| 相似度 | 判断 | 说明 |
|--------|------|------|
| ≥ 0.70 | 高度可能同一人 | 通常为同一人，低误识率 |
| 0.60 - 0.70 | 可能同一人 | 建议结合人工确认 |
| 0.45 - 0.60 | 不确定 | 需人工核实 |
| < 0.45 | 不同人 | 大概率不是同一人 |

> 阈值受以下因素影响：注册图片质量、查询图片质量、光照条件、面部角度、底库规模。建议根据实际场景数据校准。

### 5.7 错误处理示例

```python
import requests

def safe_register(url, image_path, name, employee_id=""):
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    result = resp.json()

    code = result["code"]
    if code == 0:
        return result["data"]
    elif code == 1001:
        print(f" [{name}] 未检测到人脸，跳过")
    elif code == 1002:
        print(f" [{name}] 图片质量不合格: {result['message']}")
    else:
        print(f" [{name}] 注册失败: [{code}] {result['message']}")
    return None
```

---

## 6. 底库导入与管理

### 6.1 文件名命名规范

建议使用以下格式组织照片文件：

```
姓名-工号.jpg          # 推荐：自动解析姓名和工号
张三-ENG001.jpg
李四-ADM002.jpg
王五-FIN003.jpg

姓名.jpg               # 只有姓名，工号留空
张三.jpg

英文名.jpg             # 也支持英文
John Doe.jpg
```

### 6.2 ZIP 导入

```bash
uv run python scripts/import_gallery.py
```

默认读取 `D:\faces.zip`。如需修改，编辑脚本中的 `ZIP_PATH` 变量。

导入过程中：
- 每 10 张图片显示一次进度：`[200/6949] 成功=200 失败=0 | 2.5 img/s | 已用 80s`
- 所有检测失败的图片会被记录下来
- 导入完成后生成 CSV 报表

导入报表 `gallery/import_report.csv` 包含：
| 列 | 说明 |
|----|------|
| filename | ZIP 中的文件路径 |
| status | success 或 failed |
| reason | 失败原因（如为空表示成功） |
| face_id | 成功时的 face_id |
| employee_id | 解析到的工号 |

### 6.3 通过 API 批量注册

```python
import requests
import zipfile
import io

# 方法: 通过 API 的 ZIP 批量端点
with open("faces.zip", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/v1/gallery/batch",
        files={"file": ("faces.zip", f, "application/zip")}
    )
print(resp.json())
```

### 6.4 底库维护

**备份:**
```bash
# 数据库
copy gallery\faces.db backup\
copy gallery\faces.faiss backup\

# 或关停服务后直接复制整个 gallery/ 目录
```

**迁移:**
```bash
# 1. 在新机器安装服务
# 2. 复制 gallery/ 目录到新服务器
# 3. 确保 config.yaml 中 gallery.db_dir 指向正确路径
# 4. 启动服务
```

**清空:**
- Web 后台 → 系统设置 → 清空底库
- 或 API: `DELETE /api/v1/gallery`

---

## 7. 人脸识别最佳实践

### 7.1 图片要求

| 条件 | 建议 | 备注 |
|------|------|------|
| 分辨率 | 不低于 80×80 像素 | 实际检测取决于人脸大小而非图片尺寸 |
| 人脸比例 | 占图片至少 10% | 太小的人脸特征不充分 |
| 光照 | 均匀、正面光 | 侧光/逆光/LoRA 质量分支自动补偿 |
| 角度 | 正面 ≤30°偏转 | MTCNN 支持一定角度，但极侧脸会漏检 |
| 模糊 | 清晰可见 | Laplacian 方差 ≥ 100 |
| 格式 | JPG/PNG/BMP/TIFF/WebP | 彩色图片自动转 RGB |

### 7.2 注册质量（决定搜索效果）

- 使用正面、光线均匀、无遮挡的高质量照片注册
- 不建议使用身份证照片等有损压缩图片
- 同一人不要重复注册（会导致搜索返回多条记录）

### 7.3 搜索优化

- 查询图片角度/光照与被搜索图片越接近，效果越好
- 批量导入时建议关闭质量检查 (`preprocess.do_quality_check: false`)
- 生产环境建议设置合理的阈值并做二次确认

### 7.4 性能调优

| 场景 | 建议 |
|------|------|
| 底库 < 1 万 | 当前 Faiss IndexFlatIP（暴力搜索），性能足够 |
| 底库 1 万 - 100 万 | 考虑切换为 Faiss IVF 或 HNSW |
| 底库 > 100 万 | 需要 GPU 加速或分布式部署 |
| 图片包含多张人脸 | MTCNN 只返回置信度最高的一张 |
| 高并发请求 | 使用 uvicorn 多 worker: `uvicorn ... --workers 4` |

### 7.5 LoRA 质量分支调优

| 场景 | iqa.threshold | 效果 |
|------|---------------|------|
| 所有图片高质量 | 0.7 | 更依赖质量分支，辨别力更强 |
| 混合质量 | 0.5（默认） | 平衡身份和质量分支 |
| 大量低质量图片 | 0.3 | 更依赖身份分支，对低质量更鲁棒 |
| 关闭 IQA | `iqa.enabled: false` | alpha 固定为 0，纯身份特征 |

---

## 8. 常见问题排查

### 8.1 启动问题

**Q: 模型加载失败**

确保模型文件存在且路径与 `config.yaml` 一致:
```bash
# 检查模型文件
ls -la models/swin_arcface_webface4m_tinyface/model.pt

# 重新下载
uv run python scripts/download_model.py --model swin_arcface_webface4m_tinyface
```

**Q: 端口被占用**

```bash
# 查看占用 8000 端口的进程
netstat -ano | findstr :8000

# 修改 config.yaml 中的 server.port
# 或杀进程: taskkill /PID <pid> /F
```

**Q: "No module named 'facerecserver'"**

确保在项目根目录运行:
```bash
cd D:\projects\FaceRecServer
uv run python -m facerecserver
```

### 8.2 识别问题

**Q: "未检测到人脸"**

- 检查图片中是否确实有人脸
- 尝试降低 `detection.confidence`（默认 0.95）
- 检查 `detection.min_face_size` 是否太大
- 检查图片是否过大（> 4000×4000），MTCNN 可能漏检

**Q: "图片模糊" / "图片过暗"**

- 上传更高清、更亮的图片
- 或在配置中关闭质量检查: `preprocess.do_quality_check: false`

**Q: 识别准确率低**

```yaml
# 尝试以下调整:
model:
  use_lora: true            # 确保 LoRA 启用
preprocess:
  do_alignment: true        # 确保对齐启用
  iqa:
    threshold: 0.5          # 可尝试 0.3 或 0.7
detection:
  confidence: 0.95          # 可降低到 0.9 提高召回
```

如果底库中的照片质量参差不齐，建议关闭 IQA:
```yaml
preprocess:
  iqa:
    enabled: false
```

**Q: 不同人相似度很高**

- 检查注册图片是否包含背景人脸
- 检查 MTCNN 是否检测到正确的人脸
- 尝试更换模型:
  ```bash
  uv run python scripts/download_model.py --model swin_cosface_webface4m_tinyface
  # 然后修改 config.yaml 中的 model.path 和 model.name
  ```

### 8.3 性能问题

**Q: 推理速度慢**

- PETALface 在 CPU 上每张约 500ms-2s，这是正常的
- 有 GPU 时自动使用 CUDA（~50ms/张）
- 批量导入速度约 2-3 img/s（CPU）

**Q: Faiss 索引重建慢**

首次启动时如果有大量数据，重建索引可能较慢。索引重建在 `_load_index()` 中完成：
- 从文件读取索引（秒级）
- 或从 SQLite 逐条重建（分钟级，仅在 faiss 文件丢失时）

---

## 9. 开发指南

### 9.1 项目设置

```bash
# 安装开发依赖
uv sync --group dev

# 运行测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_repository.py -v
```

### 9.2 测试架构

测试文件位于 `tests/` 目录:

| 文件 | 测试内容 |
|------|----------|
| `test_repository.py` | GalleryRepository 的 18 个测试用例 |
| `test_api_routes.py` | 主路由端点测试 |
| `test_config.py` | 配置加载与覆盖 |
| `test_schemas.py` | Pydantic 模型验证 |
| `test_utils.py` | 图片工具函数 |

使用 pytest fixtures（`conftest.py`）提供测试用图库目录、样本 embedding、样本图片等。

### 9.3 代码风格

- Python: 遵循 PEP 8，不使用额外格式化工具
- TypeScript: 使用 Vite 默认配置
- 导入顺序: 标准库 → 第三方库 → 本地模块
- 类型注解: Python 使用 typing 模块，前端使用 TypeScript

### 9.4 添加新端点

```python
# facerecserver/gallery/routes.py
@router.get("/search-by-employee", response_model=ApiResponse)
async def search_by_employee(request: Request, employee_id: str):
    repo = _get_repo(request)
    # 实现搜索逻辑
    return ApiResponse(code=0, message="success", data={...})
```

### 9.5 数据流说明

```
                     ┌──────────────┐
                     │  用户请求     │
                     └──────┬───────┘
                            ▼
                 ┌──────────────────┐
                 │  _parse_image_   │
                 │  _from_request   │ ← 支持文件/Base64/URL
                 └──────┬──────────┘
                        ▼
              ┌──────────────────┐
              │ FaceDetector     │
              │ .detect()        │ ← MTCNN
              └──────┬──────────┘
                     ▼
              ┌──────────────────┐
              │ align_face()     │ ← 关键点仿射变换
              └──────┬──────────┘
                     ▼
              ┌──────────────────┐
              │ check_image_     │
              │ quality()        │ ← Laplacian 模糊检测
              └──────┬──────────┘
                     ▼
              ┌──────────────────┐
              │ estimate_alpha() │ ← CNN-IQA
              └──────┬──────────┘
                     ▼
              ┌──────────────────┐
              │ PETALface 模型    │
              │ .forward()       │ ← Swin-T + LoRA
              └──────┬──────────┘
                     ▼
              ┌──────────────────┐
              │ L2 Normalize     │
              └──────┬──────────┘
                     ▼
        ┌────────────┴────────────┐
        ▼                         ▼
  ┌──────────┐            ┌──────────────┐
  │ Faiss    │            │ SQLite       │
  │ 搜索     │            │ 存储         │
  └──────────┘            └──────────────┘
        ▼
  ┌──────────┐
  │ 返回结果  │
  └──────────┘
```

### 9.6 模型下载方式

```bash
# 列出所有可用模型
uv run python scripts/download_model.py --list

# 下载指定模型
uv run python scripts/download_model.py --model swin_cosface_webface4m_tinyface

# 指定输出目录
uv run python scripts/download_model.py --model swin_arcface_webface4m --output-dir /data/models
```

### 9.7 底库数据结构

```sql
-- SQLite faces 表
CREATE TABLE faces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id     TEXT UNIQUE NOT NULL,         -- UUID v4
    name        TEXT NOT NULL,                -- 人名
    employee_id TEXT DEFAULT '',              -- 工号
    created_at  TEXT NOT NULL,                -- ISO 8601
    image_path  TEXT DEFAULT ''               -- 人脸裁剪图路径
);

-- Faiss 索引
IndexIDMap(IndexFlatIP(512))                  -- 512 维内积索引
-- faiss_id = SQLite faces.id
```

---

> 本文档对应 FaceRecServer v0.1.0。
> 基于 PETALface (WACV 2025) 算法构建。
