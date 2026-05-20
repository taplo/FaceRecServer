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
