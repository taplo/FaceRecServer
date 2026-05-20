"""导入 d:\faces.zip 人脸底库照片到 Gallery。

文件名格式: 姓名-工号.jpg (GBK 编码)
用法: uv run python scripts/import_gallery.py
"""

import os
import sys
import zipfile
import time
import io
import logging

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

ZIP_PATH = r"D:\faces.zip"
GALLERY_DIR = "gallery"
DB_NAME = "faces"

BATCH_SIZE = 100


def main():
    os.makedirs(GALLERY_DIR, exist_ok=True)

    logger.info("加载模型...")
    from facerecserver.face_recognition.embedding import FaceEmbeddingExtractor
    from facerecserver.config import load_config

    config = load_config()
    extractor = FaceEmbeddingExtractor(config)
    logger.info("模型加载完成")

    from facerecserver.gallery.repository import GalleryRepository
    repo = GalleryRepository(GALLERY_DIR, DB_NAME)
    existing = repo.get_count()
    logger.info("当前底库已有 %d 条记录", existing)

    with zipfile.ZipFile(ZIP_PATH, "r", metadata_encoding="gbk") as zf:
        names = zf.namelist()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_entries = [
            n for n in names
            if os.path.splitext(n)[1].lower() in image_exts
        ]
        logger.info("ZIP 中共 %d 个图片文件", len(image_entries))

        succeeded = 0
        failed = 0
        skipped = 0
        start_time = time.time()

        for idx, entry_name in enumerate(image_entries, 1):
            try:
                basename = os.path.splitext(os.path.basename(entry_name))[0]
                if "-" in basename:
                    person_name = basename.rsplit("-", 1)[0]
                else:
                    person_name = basename

                raw = zf.read(entry_name)
                image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

                embedding = extractor.extract(image)
                repo.add(embedding, person_name, entry_name)
                succeeded += 1

            except Exception as e:
                failed += 1
                logger.error("失败 [%d/%d] %s: %s", idx, len(image_entries), entry_name, e)

            if idx % 10 == 0 or idx == len(image_entries):
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] 成功=%d 失败=%d 跳过=%d | %.1f img/s | 已用 %.0fs",
                    idx, len(image_entries), succeeded, failed, skipped, rate, elapsed,
                )

        elapsed = time.time() - start_time
        total_processed = succeeded + failed
        logger.info("=" * 50)
        logger.info("导入完成!")
        logger.info("总计: %d | 成功: %d | 失败: %d | 跳过: %d", total_processed, succeeded, failed, skipped)
        logger.info("耗时: %.0fs (%.1f 分钟)", elapsed, elapsed / 60)

    repo.close()


if __name__ == "__main__":
    main()
