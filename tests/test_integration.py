import io
import os
import zipfile
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from PIL import Image
import numpy as np


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.device = "cpu"
    config.model.name = "swin_arcface_webface4m_tinyface"
    config.preprocess.do_quality_check = False
    config.preprocess.do_alignment = True
    config.gallery.db_dir = "/tmp/test_gallery"
    config.gallery.db_name = "test_faces"
    return config


@pytest.fixture
def app(mock_config):
    from facerecserver.app import create_app
    app = create_app(mock_config)
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _make_test_image(size=(112, 112), color="red"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _make_zip_with_images(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            zf.writestr(name, content)
    buf.seek(0)
    return buf.getvalue()


class TestHealth:
    def test_livez(self, client):
        resp = client.get("/api/v1/livez")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readyz_degraded(self, client):
        resp = client.get("/api/v1/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"

    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "device" in data


class TestCompare:
    def test_compare_no_images(self, client):
        resp = client.post("/api/v1/compare", json={})
        assert resp.status_code == 200
        assert resp.json()["code"] == 400

    def test_compare_missing_one_image(self, client):
        buf = _make_test_image()
        resp = client.post("/api/v1/compare", files={"file1": ("a.jpg", buf, "image/jpeg")})
        assert resp.status_code == 200
        assert resp.json()["code"] == 400

    def test_compare_same_image_bytes(self, client):
        buf = _make_test_image()
        data = buf.getvalue()
        resp = client.post(
            "/api/v1/compare",
            files={"file1": ("a.jpg", data, "image/jpeg"), "file2": ("b.jpg", data, "image/jpeg")},
        )
        assert resp.status_code in (200, 503)


class TestEmbedding:
    def test_embedding_no_image(self, client):
        resp = client.post("/api/v1/embedding", json={})
        assert resp.status_code == 200
        assert resp.json()["code"] == 400

    def test_embedding_with_file(self, client):
        buf = _make_test_image()
        resp = client.post("/api/v1/embedding", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert data["code"] == 0

    def test_embedding_with_base64(self, client):
        import base64
        buf = _make_test_image()
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = client.post("/api/v1/embedding", json={"image": b64})
        assert resp.status_code in (200, 503)

    def test_embedding_too_large(self, client):
        buf = _make_test_image((5000, 5000), "blue")
        resp = client.post("/api/v1/embedding", files={"file": ("big.jpg", buf, "image/jpeg")})
        assert resp.status_code == 200
        assert resp.json()["code"] == 1002


class TestStats:
    def test_stats_endpoint(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "gallery" in data["data"]


class TestGallery:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/gallery")
        assert resp.status_code in (200, 503)

    def test_list_with_search(self, client):
        resp = client.get("/api/v1/gallery?search=test&page=1&page_size=10")
        assert resp.status_code in (200, 503)

    def test_register_face_no_file(self, client):
        resp = client.post("/api/v1/gallery", json={})
        assert resp.status_code in (200, 503)

    def test_clear_gallery(self, client):
        resp = client.delete("/api/v1/gallery")
        assert resp.status_code in (200, 503)

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/v1/gallery/nonexistent-id")
        assert resp.status_code in (200, 503)

    def test_reindex(self, client):
        resp = client.post("/api/v1/gallery/reindex")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "total_faces" in data["data"]


class TestZipBatch:
    def _make_zip_bytes(self, filenames: list[str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in filenames:
                zf.writestr(name, _make_test_image().getvalue())
        buf.seek(0)
        return buf.getvalue()

    def test_batch_upload(self, client):
        zip_data = self._make_zip_bytes(["test-face.jpg"])
        resp = client.post("/api/v1/gallery/batch", files={"file": ("faces.zip", zip_data, "application/zip")})
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "data" in data

    def test_batch_empty_zip(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            pass
        buf.seek(0)
        resp = client.post("/api/v1/gallery/batch", files={"file": ("empty.zip", buf, "application/zip")})
        assert resp.status_code in (200, 503)

    def test_batch_invalid_file(self, client):
        resp = client.post("/api/v1/gallery/batch", files={"file": ("test.txt", b"not a zip", "text/plain")})
        assert resp.status_code in (200, 503)


class TestRecognize:
    def test_recognize_no_image(self, client):
        resp = client.post("/api/v1/gallery/recognize", json={})
        assert resp.status_code in (200, 503)

    def test_recognize_with_image(self, client):
        buf = _make_test_image()
        resp = client.post(
            "/api/v1/gallery/recognize?top_k=5",
            files={"file": ("query.jpg", buf, "image/jpeg")},
        )
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            if data["code"] == 0:
                assert "results" in data["data"]


class TestChineseFilenames:
    def test_register_with_chinese_name_json(self, client):
        resp = client.post("/api/v1/gallery", json={"name": "张三-001", "image": ""})
        assert resp.status_code in (200, 503)

    def test_gbk_zip_filenames(self, client):
        img_data = _make_test_image().getvalue()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("张三-001.jpg", img_data)
            zf.writestr("李四-002.jpg", img_data)
        buf.seek(0)
        resp = client.post("/api/v1/gallery/batch", files={"file": ("faces.zip", buf, "application/zip")})
        assert resp.status_code in (200, 503)


class TestPersons:
    def test_list_persons(self, client):
        resp = client.get("/api/v1/gallery/persons")
        assert resp.status_code in (200, 503)

    def test_list_persons_with_search(self, client):
        resp = client.get("/api/v1/gallery/persons?search=test")
        assert resp.status_code in (200, 503)

    def test_delete_person_nonexistent(self, client):
        resp = client.delete("/api/v1/gallery/persons/99999")
        assert resp.status_code in (200, 503)
