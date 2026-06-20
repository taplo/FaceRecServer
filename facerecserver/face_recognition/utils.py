import base64
import io
import numpy as np
import cv2
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def base64_to_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    buf = io.BytesIO(raw)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def preprocess_image(image: np.ndarray, target_size: int = 112) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


def check_image_quality(image: np.ndarray, blur_threshold: float = 100.0, min_brightness: float = 10.0) -> tuple[bool, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, f"图片模糊 (Laplacian variance={laplacian_var:.1f} < {blur_threshold})"

    mean_brightness = gray.mean()
    if mean_brightness < min_brightness:
        return False, f"图片过暗 (mean brightness={mean_brightness:.1f} < {min_brightness})"

    return True, "ok"


def estimate_alpha(image: np.ndarray, threshold: float = 0.5) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = 1.0 - np.exp(-laplacian_var / 400.0)
    alpha = 0.5 + (score - threshold)
    return max(0.0, min(1.0, alpha))
