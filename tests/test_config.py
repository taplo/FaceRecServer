from facerecserver.config import load_config, AppConfig
import os


class TestConfig:
    def test_load_default(self):
        """Config should load with defaults when no file specified."""
        # Set empty config path to force defaults
        os.environ["FACEREC_CONFIG"] = ""
        cfg = load_config()
        assert isinstance(cfg, AppConfig)
        assert cfg.model.name == "swin_arcface_webface4m_tinyface"
        assert cfg.model.lora_rank == 8
        assert cfg.model.use_lora is True
        assert cfg.detection.confidence == 0.95
        assert cfg.detection.min_face_size == 40
        assert cfg.preprocess.image_size == 112
        assert cfg.preprocess.do_alignment is True
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.port == 8000
        assert cfg.gallery.db_dir == "gallery"
        assert cfg.gallery.db_name == "faces"
        assert cfg.device in ("cpu", "cuda")

    def test_device_detection(self):
        """Device should be cpu/cuda based on torch availability."""
        import torch
        cfg = AppConfig()
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            assert cfg.device == "cuda"
        else:
            assert cfg.device == "cpu"
