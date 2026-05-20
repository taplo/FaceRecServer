import os
import yaml
from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    path: str = "models/swin_arcface_webface4m_tinyface/model.pt"
    name: str = "swin_arcface_webface4m_tinyface"
    lora_rank: int = 8
    lora_scale: float = 1.0
    use_lora: bool = True


@dataclass
class DetectionConfig:
    confidence: float = 0.95
    min_face_size: int = 40


@dataclass
class PreprocessConfig:
    image_size: int = 112
    do_alignment: bool = True
    do_quality_check: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    device: str = "cpu"


def load_config(path: str | None = None) -> AppConfig:
    if path is None:
        path = os.environ.get("FACEREC_CONFIG", "")
    if not path:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "config.yaml")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig()

    m = raw.get("model", {})
    cfg.model.path = m.get("path", cfg.model.path)
    cfg.model.name = m.get("name", cfg.model.name)
    cfg.model.lora_rank = m.get("lora_rank", cfg.model.lora_rank)
    cfg.model.lora_scale = m.get("lora_scale", cfg.model.lora_scale)
    cfg.model.use_lora = m.get("use_lora", cfg.model.use_lora)

    d = raw.get("detection", {})
    cfg.detection.confidence = d.get("confidence", cfg.detection.confidence)
    cfg.detection.min_face_size = d.get("min_face_size", cfg.detection.min_face_size)

    p = raw.get("preprocess", {})
    cfg.preprocess.image_size = p.get("image_size", cfg.preprocess.image_size)
    cfg.preprocess.do_alignment = p.get("do_alignment", cfg.preprocess.do_alignment)
    cfg.preprocess.do_quality_check = p.get("do_quality_check", cfg.preprocess.do_quality_check)

    s = raw.get("server", {})
    cfg.server.host = s.get("host", cfg.server.host)
    cfg.server.port = s.get("port", cfg.server.port)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg
