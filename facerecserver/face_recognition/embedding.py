import numpy as np
import torch

from facerecserver.face_detection.detector import FaceDetector, FaceNotFoundError
from facerecserver.face_detection.aligner import align_face
from facerecserver.face_recognition.model import load_model
from facerecserver.face_recognition.utils import (
    load_image, base64_to_image, check_image_quality, estimate_alpha, _set_iqa_device,
)
from facerecserver.config import AppConfig


class FaceEmbeddingExtractor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = FaceDetector(
            min_face_size=config.detection.min_face_size,
            confidence_threshold=config.detection.confidence,
            device=config.device,
        )
        _set_iqa_device(config.device)
        self.model = load_model(
            model_path=config.model.path,
            model_name=config.model.name,
            lora_rank=config.model.lora_rank,
            lora_scale=config.model.lora_scale,
            use_lora=config.model.use_lora,
            device=config.device,
        )
        self.device = config.device

    def extract(self, image: np.ndarray, return_face: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.config.preprocess.do_quality_check:
            ok, msg = check_image_quality(image)
            if not ok:
                raise ValueError(f"质量检查不合格: {msg}")

        bbox, landmarks, conf = self.detector.detect(image)

        if self.config.preprocess.do_alignment:
            face = align_face(image, landmarks, output_size=120)
        else:
            x1, y1, x2, y2 = bbox
            face = image[y1:y2, x1:x2]
            face = np.array(Image.fromarray(face).resize((120, 120)))

        face_crop = None
        if return_face:
            x1, y1, x2, y2 = bbox
            fw, fh = x2 - x1, y2 - y1
            margin_w, margin_h = int(fw * 0.3), int(fh * 0.3)
            h_img, w_img = image.shape[:2]
            cx1 = max(0, x1 - margin_w)
            cy1 = max(0, y1 - margin_h)
            cx2 = min(w_img, x2 + margin_w)
            cy2 = min(h_img, y2 + margin_h)
            face_crop = image[cy1:cy2, cx1:cx2]

        alpha = estimate_alpha(face, self.config.preprocess.iqa.threshold)

        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0)
        face_tensor = face_tensor / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.to(self.device)

        alpha_tensor = torch.tensor([alpha], device=self.device).float()

        with torch.no_grad():
            embedding = self.model(face_tensor, alpha_tensor)

        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        emb = embedding.cpu().numpy().flatten()
        if return_face:
            return emb, face_crop
        return emb

    def extract_from_file(self, path: str) -> np.ndarray:
        image = load_image(path)
        return self.extract(image)

    def extract_from_base64(self, base64_str: str) -> np.ndarray:
        image = base64_to_image(base64_str)
        return self.extract(image)
