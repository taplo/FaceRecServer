"""导入 d:\faces.zip 人脸底库照片到 Gallery。

文件名格式: 姓名-工号.jpg (GBK 编码)
用法: uv run python scripts/import_gallery.py
"""

import os
import sys
import zipfile
import time
import io
import csv
import logging

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

ZIP_PATH = r"D:\faces.zip"
GALLERY_DIR = "gallery"
DB_NAME = "faces"
REPORT_PATH = "gallery/import_report.csv"


def main():
    os.makedirs(GALLERY_DIR, exist_ok=True)

    logger.info("加载模型...")
    from facerecserver.face_recognition.embedding import FaceEmbeddingExtractor
    from facerecserver.config import load_config

    config = load_config()
    config.preprocess.do_quality_check = False
    config.preprocess.do_alignment = True
    extractor = FaceEmbeddingExtractor(config)
    logger.info("模型加载完成")

    from facerecserver.gallery.repository import GalleryRepository
    repo = GalleryRepository(GALLERY_DIR, DB_NAME)

    logger.info("清空现有底库...")
    repo.clear()
    logger.info("底库已清空")

    with zipfile.ZipFile(ZIP_PATH, "r", metadata_encoding="gbk") as zf:
        names = zf.namelist()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_entries = [
            n for n in names
            if os.path.splitext(n)[1].lower() in image_exts
        ]
        logger.info("ZIP 中共 %d 个图片文件", len(image_entries))

        records = []
        start_time = time.time()

        for idx, entry_name in enumerate(image_entries, 1):
            result = {"filename": entry_name, "status": "", "reason": "", "face_id": ""}

            try:
                basename = os.path.splitext(os.path.basename(entry_name))[0]
                if "-" in basename:
                    parts = basename.rsplit("-", 1)
                    person_name = parts[0]
                    employee_id = parts[1]
                else:
                    person_name = basename
                    employee_id = ""

                raw = zf.read(entry_name)
                image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

                embedding, face = extractor.extract(image, return_face=True)
                face_id = repo.add(embedding, person_name, employee_id=employee_id, image=face)
                result["status"] = "success"
                result["face_id"] = face_id
                result["employee_id"] = employee_id

            except Exception as e:
                result["status"] = "failed"
                result["reason"] = str(e)

            records.append(result)

            if idx % 10 == 0 or idx == len(image_entries):
                elapsed = time.time() - start_time
                ok = sum(1 for r in records if r["status"] == "success")
                fail = sum(1 for r in records if r["status"] == "failed")
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] 成功=%d 失败=%d | %.1f img/s | 已用 %.0fs",
                    idx, len(image_entries), ok, fail, rate, elapsed,
                )

        elapsed = time.time() - start_time
        ok = sum(1 for r in records if r["status"] == "success")
        fail = sum(1 for r in records if r["status"] == "failed")

        # 写入 CSV 报表
        with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "status", "reason", "face_id", "employee_id"])
            writer.writeheader()
            writer.writerows(records)

        logger.info("=" * 50)
        logger.info("导入完成!")
        logger.info("总计: %d | 成功: %d | 失败: %d | 成功率: %.1f%%",
                    len(image_entries), ok, fail, ok / len(image_entries) * 100 if image_entries else 0)
        logger.info("耗时: %.0fs (%.1f 分钟)", elapsed, elapsed / 60)
        logger.info("报表已保存到: %s", REPORT_PATH)

        # 打印统计摘要
        logger.info("=" * 50)
        logger.info("失败原因统计:")
        from collections import Counter
        reasons = Counter(r["reason"] for r in records if r["status"] == "failed")
        for reason, count in reasons.most_common(10):
            logger.info("  %s: %d", reason, count)

    repo.close()


if __name__ == "__main__":
    main()
