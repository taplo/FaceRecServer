import base64
import io
import numpy as np
from PIL import Image
import warnings
from facerecserver.face_recognition.utils import (
    load_image,
    base64_to_image,
    preprocess_image,
    check_image_quality,
    estimate_alpha,
)


class TestImageUtils:

    def test_base64_to_image(self):
        img = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        result = base64_to_image(b64)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)

    def test_base64_to_image_with_data_uri(self):
        img = Image.new("RGB", (5, 5), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"
        result = base64_to_image(data_uri)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5, 3)

    def test_load_image(self, tmp_path):
        img = Image.new("RGB", (20, 20), color="green")
        path = tmp_path / "test.jpg"
        img.save(path)
        result = load_image(str(path))
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 20, 3)

    def test_preprocess_image_no_resize(self):
        img_arr = np.zeros((112, 112, 3), dtype=np.uint8)
        result = preprocess_image(img_arr, target_size=112)
        assert result.shape == (112, 112, 3)

    def test_preprocess_image_resize(self):
        img_arr = np.zeros((200, 100, 3), dtype=np.uint8)
        result = preprocess_image(img_arr, target_size=112)
        assert result.shape[0] == 112  # height resized to 112
        assert result.shape[1] <= 112

    def test_check_image_quality_sharp(self):
        # High contrast image = sharp
        img_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ok, msg = check_image_quality(img_arr, blur_threshold=1.0)
        assert ok is True

    def test_check_image_quality_blur(self):
        # Uniform image = blurry
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        ok, msg = check_image_quality(img_arr, blur_threshold=100.0)
        assert ok is False
        assert "模糊" in msg

    def test_estimate_alpha_sharp(self):
        img_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        alpha = estimate_alpha(img_arr)
        assert 0.0 <= alpha <= 1.0

    def test_estimate_alpha_uniform(self):
        img_arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        alpha = estimate_alpha(img_arr)
        assert 0.0 <= alpha <= 1.0

    def test_estimate_alpha_clamp(self):
        img_arr = np.ones((100, 100, 3), dtype=np.uint8) * 255
        alpha = estimate_alpha(img_arr, threshold=-2.0)
        assert alpha == 1.0
        alpha = estimate_alpha(img_arr, threshold=2.0)
        assert alpha == 0.0
