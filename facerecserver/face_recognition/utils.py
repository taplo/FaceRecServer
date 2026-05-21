import base64
import io
import numpy as np
import cv2
import torch
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


_iqa_model = None


def _get_iqa_model():
    global _iqa_model
    if _iqa_model is None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pyiqa
            _iqa_model = pyiqa.create_metric("cnniqa", device="cpu")
    return _iqa_model


def estimate_alpha(image: np.ndarray, threshold: float = 0.5) -> float:
    model = _get_iqa_model()
    if len(image.shape) == 2:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        score = model(tensor).item()
    alpha = 0.5 + (score - threshold)
    return max(0.0, min(1.0, alpha))
