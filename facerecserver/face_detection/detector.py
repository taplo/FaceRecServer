import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN as _MTCNN


class FaceNotFoundError(Exception):
    pass


class FaceDetector:
    def __init__(self, min_face_size: int = 40, confidence_threshold: float = 0.95):
        self._detector = _MTCNN(min_face_size=min_face_size, thresholds=[0.3, 0.4, 0.5], device="cpu")
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray | Image.Image):
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 4:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(image)

        boxes, probs, landmarks = self._detector.detect(image, landmarks=True)

        if boxes is None or len(boxes) == 0:
            raise FaceNotFoundError("未检测到人脸")

        valid = [(b, p, l) for b, p, l in zip(boxes, probs, landmarks) if p >= self.confidence_threshold]
        if not valid:
            raise FaceNotFoundError("未检测到人脸")

        best_box, best_prob, best_landmarks = max(valid, key=lambda x: x[1])
        x1, y1, x2, y2 = [int(v) for v in best_box]
        x1, y1 = max(0, x1), max(0, y1)

        landmarks_dict = {
            "left_eye": (int(best_landmarks[0][0]), int(best_landmarks[0][1])),
            "right_eye": (int(best_landmarks[1][0]), int(best_landmarks[1][1])),
            "nose": (int(best_landmarks[2][0]), int(best_landmarks[2][1])),
            "mouth_left": (int(best_landmarks[3][0]), int(best_landmarks[3][1])),
            "mouth_right": (int(best_landmarks[4][0]), int(best_landmarks[4][1])),
        }

        return (x1, y1, x2, y2), landmarks_dict, float(best_prob)
