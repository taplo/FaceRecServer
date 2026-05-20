import numpy as np
import torch

from facerecserver.face_detection.detector import FaceDetector, FaceNotFoundError
from facerecserver.face_detection.aligner import align_face
from facerecserver.face_recognition.model import load_model
from facerecserver.face_recognition.utils import (
    load_image, base64_to_image, check_image_quality, estimate_alpha,
)
from facerecserver.config import AppConfig


class FaceEmbeddingExtractor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = FaceDetector(
            min_face_size=config.detection.min_face_size,
            confidence_threshold=config.detection.confidence,
        )
        self.model = load_model(
            model_path=config.model.path,
            model_name=config.model.name,
            lora_rank=config.model.lora_rank,
            lora_scale=config.model.lora_scale,
            use_lora=config.model.use_lora,
            device=config.device,
        )
        self.device = config.device

    def extract(self, image: np.ndarray) -> np.ndarray:
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

        alpha = estimate_alpha(face)

        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0)
        face_tensor = face_tensor / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.to(self.device)

        alpha_tensor = torch.tensor([alpha], device=self.device).float()

        with torch.no_grad():
            embedding = self.model(face_tensor, alpha_tensor)

        return embedding.cpu().numpy().flatten()

    def extract_from_file(self, path: str) -> np.ndarray:
        image = load_image(path)
        return self.extract(image)

    def extract_from_base64(self, base64_str: str) -> np.ndarray:
        image = base64_to_image(base64_str)
        return self.extract(image)
