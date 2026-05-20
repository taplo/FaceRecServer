# Roadmap: FaceRecServer

**4 phases** | **19 v1 requirements** | All mapped ✓

---

### Phase 1: PETALface 模型集成
**Goal:** 加载 PETALface 预训练模型，实现人脸特征向量提取核心能力
**Mode:** mvp

**Requirements:** API-01, SYS-02, SYS-03

**Success Criteria:**
1. 成功加载 PETALface 预训练模型（CPU）
2. 输入人脸图片，能正确输出特征向量
3. 特征向量维度与预期一致
4. 同一个人的两张照片特征向量相似度 > 不同人
5. FastAPI 服务启动，Embedding 提取接口可调用
6. 支持中文图片路径

---

### Phase 2: 人脸底库管理 (Gallery)
**Goal:** 实现人脸底库的增删查清空管理功能，支持单张和 ZIP 批量上传
**Mode:** mvp

**Requirements:** GALLERY-01, GALLERY-02, GALLERY-03, GALLERY-04, GALLERY-05, GALLERY-06, SYS-01

**Success Criteria:**
1. 单张人脸图片上传到底库，返回唯一 ID
2. ZIP 文件批量上传，自动解压并注册所有人脸到底库
3. 根据 ID 从底库删除人脸
4. 查看底库人脸列表（分页、搜索）
5. 清空整个底库
6. 底库支持中文人脸名称和中文文件路径

---

### Phase 3: 人脸比对与识别 API
**Goal:** 提供 1:1 人脸比对和 1:N 人脸识别 API 服务
**Mode:** mvp

**Requirements:** API-02, API-03, API-04, API-05

**Success Criteria:**
1. 1:1 API：输入两张图片/特征向量，返回相似度分数
2. 1:N API：输入待识别人脸，返回底库中 Top-K 匹配结果
3. 支持图片 URL 和 Base64 两种输入
4. 统一响应格式（code, message, data）
5. API 文档可访问（FastAPI 自动生成）

---

### Phase 4: Vue 3 Web 管理后台
**Goal:** 提供可视化的人脸管理、比对、识别测试 Web 界面
**Mode:** mvp

**Requirements:** WEB-01, WEB-02, WEB-03, WEB-04, WEB-05

**Success Criteria:**
1. Vue 3 项目搭建，与 FastAPI 后端通信
2. 底库管理页面：上传、批量上传、删除、查询、清空
3. 人脸比对页面：上传两张图片，显示相似度
4. 人脸识别页面：上传待识别人脸，显示匹配结果
5. 全部界面使用中文

---

**Total:** 4 phases | 19 requirements | Coverage: 100% ✓
