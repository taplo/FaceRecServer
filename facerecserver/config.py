import os
import platform
import logging
import yaml
from dataclasses import dataclass, field
import torch

logger = logging.getLogger(__name__)


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
class IQAConfig:
    enabled: bool = True
    threshold: float = 0.5


@dataclass
class PreprocessConfig:
    image_size: int = 112
    do_alignment: bool = True
    do_quality_check: bool = True
    iqa: IQAConfig = field(default_factory=IQAConfig)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class GalleryConfig:
    db_dir: str = "gallery"
    db_name: str = "faces"
    page_size_default: int = 20
    page_size_max: int = 100


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    device: str = "cpu"


def _detect_cpu_capabilities() -> dict:
    caps = {"avx2": False, "avx": False, "sse4_2": False}
    if platform.system() == "Linux" and os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.lower()
                    caps["avx2"] = "avx2" in flags
                    caps["avx"] = "avx" in flags
                    caps["sse4_2"] = "sse4_2" in flags
                    break
    if not caps["avx2"] and caps["avx"]:
        os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX")
        logger.info("CPU 不支持 AVX2（仅支持 AVX），已设置 MKL 兼容模式")
    elif caps["avx2"]:
        logger.info("CPU 支持 AVX2，已启用加速")
    else:
        logger.info("CPU 不支持 AVX 指令集，性能可能受限")
    return caps


def load_config(path: str | None = None) -> AppConfig:
    cpu_caps = _detect_cpu_capabilities()
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
    iq = p.get("iqa", {})
    cfg.preprocess.iqa.enabled = iq.get("enabled", cfg.preprocess.iqa.enabled)
    cfg.preprocess.iqa.threshold = iq.get("threshold", cfg.preprocess.iqa.threshold)

    s = raw.get("server", {})
    cfg.server.host = s.get("host", cfg.server.host)
    cfg.server.port = s.get("port", cfg.server.port)

    g = raw.get("gallery", {})
    cfg.gallery.db_dir = g.get("db_dir", cfg.gallery.db_dir)
    cfg.gallery.db_name = g.get("db_name", cfg.gallery.db_name)
    cfg.gallery.page_size_default = g.get("page_size_default", cfg.gallery.page_size_default)
    cfg.gallery.page_size_max = g.get("page_size_max", cfg.gallery.page_size_max)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        logger.info("CUDA 可用，已启用 GPU 加速 (device=%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA 不可用，使用 CPU")

    cfg._cpu_caps = cpu_caps
    return cfg
