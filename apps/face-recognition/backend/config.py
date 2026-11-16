from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODELS_DIR = _REPO_ROOT / "benchmarks" / "corpus" / "face" / "models"


@dataclass(frozen=True)
class FaceAppSettings:
    """Configuración de la aplicación independiente de reconocimiento facial."""

    dbms_api_url: str = os.getenv("DBMS_API_URL", "http://127.0.0.1:8000")
    extractor_backend: str = os.getenv("FACE_BACKEND", "sface")
    yunet_model_path: Path = Path(
        os.getenv(
            "FACE_YUNET_MODEL",
            str(_DEFAULT_MODELS_DIR / "face_detection_yunet_2026may.onnx"),
        )
    )
    sface_model_path: Path = Path(
        os.getenv(
            "FACE_SFACE_MODEL",
            str(_DEFAULT_MODELS_DIR / "face_recognition_sface_2021dec.onnx"),
        )
    )
    facenet_pretrained: str = os.getenv("FACENET_PRETRAINED", "vggface2")
    embedding_dim: int = int(os.getenv("FACE_EMBEDDING_DIM", "0"))  # 0 = nativa del backend
    default_index_type: str = os.getenv("FACE_INDEX_TYPE", "hnsw")
    default_metric: str = os.getenv("FACE_METRIC", "cosine")
    default_k: int = int(os.getenv("FACE_DEFAULT_K", "5"))
    default_threshold: float = float(os.getenv("FACE_THRESHOLD", "0.45"))
    hnsw_ef_search: int = int(os.getenv("FACE_HNSW_EF_SEARCH", "100"))
    hnsw_ef_construction: int = int(os.getenv("FACE_HNSW_EF_CONSTRUCTION", "200"))
    hnsw_m: int = int(os.getenv("FACE_HNSW_M", "16"))
    ivf_n_lists: int = int(os.getenv("FACE_IVF_N_LISTS", "16"))
    ivf_n_probe: int = int(os.getenv("FACE_IVF_N_PROBE", "4"))
    cache_ttl_seconds: int = int(os.getenv("FACE_CACHE_TTL", "300"))
    cache_maxsize: int = int(os.getenv("FACE_CACHE_MAXSIZE", "1024"))
    redis_url: str | None = os.getenv("REDIS_URL") or None

    def resolved_dim(self) -> int:
        if self.embedding_dim is not None and self.embedding_dim > 0:
            return int(self.embedding_dim)
        if self.extractor_backend.lower() == "facenet":
            return 512
        return 128

    @classmethod
    def from_env(cls, **overrides) -> FaceAppSettings:
        clean = {k: v for k, v in overrides.items() if v is not None}
        return cls(**clean)
