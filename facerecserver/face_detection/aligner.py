import numpy as np
import cv2


def align_face(image: np.ndarray, landmarks: dict, output_size: int = 112) -> np.ndarray:
    left_eye = np.array(landmarks["left_eye"], dtype=np.float32)
    right_eye = np.array(landmarks["right_eye"], dtype=np.float32)
    nose = np.array(landmarks["nose"], dtype=np.float32)
    left_mouth = np.array(landmarks["mouth_left"], dtype=np.float32)
    right_mouth = np.array(landmarks["mouth_right"], dtype=np.float32)

    src_pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

    scale = output_size / 112.0
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    dst_pts *= scale

    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if tform is None:
        tform, _ = cv2.estimateAffine2D(src_pts, dst_pts)

    aligned = cv2.warpAffine(image, tform, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
    return aligned
