import tempfile
import os
import numpy as np
import pytest
from PIL import Image
import io


@pytest.fixture
def gallery_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(42)
    emb = rng.random(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def sample_image():
    img = Image.new("RGB", (112, 112), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))
