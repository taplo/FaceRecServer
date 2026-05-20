import torch
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path

img_path = next(Path("test/01").glob("*.jpg"))
print(f"Image: {img_path}")
print(f"Exists: {img_path.exists()}")

img = Image.open(img_path).convert("RGB")
print(f"Size: {img.size}, Mode: {img.mode}")

# Convert to numpy and back to debug
import numpy as np
arr = np.array(img)
print(f"Numpy shape: {arr.shape}, dtype: {arr.dtype}, range: [{arr.min()}, {arr.max()}]")

# MTCNN with lower thresholds
detector = MTCNN(min_face_size=20, thresholds=[0.3, 0.4, 0.5], device="cpu")
boxes, probs, landmarks = detector.detect(img, landmarks=True)
print(f"boxes: {boxes}")
print(f"probs: {probs}")
if boxes is not None:
    print(f"Faces: {len(boxes)}")
else:
    print("No faces detected")

# Try with the tensor approach we use in our code
img_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
print(f"Tensor shape: {img_tensor.shape}")
boxes2, probs2, landmarks2 = detector.detect(img_tensor, landmarks=True)
print(f"boxes2: {boxes2}")
print(f"probs2: {probs2}")
