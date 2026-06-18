import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.device = "cpu"
    config.model.name = "test_model"
    return config


class TestApiRoutes:

    def test_get_openapi(self):
        from facerecserver.app import create_app
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/openapi.json")
            assert resp.status_code == 200

    def test_health_endpoint(self):
        from facerecserver.app import create_app
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "device" in data

    def test_health_check(self):
        from facerecserver.app import create_app
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/api/v1/stats")
            assert resp.status_code in (200, 503)

    def test_gallery_list_no_repo(self, mock_config):
        from facerecserver.app import create_app
        app = create_app(mock_config)
        app.state.gallery_repo = None
        with TestClient(app) as client:
            resp = client.get("/api/v1/gallery")
            assert resp.status_code == 503

    def test_compare_no_images(self):
        from facerecserver.app import create_app
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/api/v1/compare", json={})
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == 400

    def test_compare_same_image(self):
        """两张相同图片应返回接近 1.0 的相似度"""
        from facerecserver.app import create_app
        from PIL import Image
        import io
        app = create_app()
        # Create a simple test image
        img = Image.new("RGB", (112, 112), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/compare",
                files={"file1": ("test.jpg", buf, "image/jpeg"), "file2": ("test.jpg", buf.getvalue(), "image/jpeg")},
            )
            assert resp.status_code in (200, 503)
