from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
APP_ROOT = REPO_ROOT / "apps" / "face-recognition"
for p in (
    str(THIS_DIR),
    str(APP_ROOT),
    str(REPO_ROOT / "api"),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "benchmarks"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.config import FaceAppSettings
from backend.dbms_client import DBMSClient
from backend.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from backend.extractor import ExtractedFace
from backend.main import create_app
from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from face_helpers import ASGITestClient
from transaction.transaction_manager import TransactionManager


def _load_src_main():
    spec = importlib.util.spec_from_file_location("dbms_src_main_for_face", REPO_ROOT / "src" / "main.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dbms_api_main():
    spec = importlib.util.spec_from_file_location("dbms_api_main_for_face_tests", REPO_ROOT / "api" / "main.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeFaceExtractor:
    """Extractor determinista sin pesos externos para pruebas E2E de la app y el DBMS."""

    def __init__(self, dim: int = 128, model_name: str = "fake-sface") -> None:
        self.embedding_dim = dim
        self.model_name = model_name

    def _vec_for_subject(self, subject: str, noise: float = 0.0, seed_offset: int = 0) -> np.ndarray:
        seed = abs(hash(subject)) % (2**31)
        base_rng = np.random.default_rng(seed)
        vec = base_rng.standard_normal(self.embedding_dim).astype(np.float32)
        if noise > 0.0:
            n_rng = np.random.default_rng(seed + 17 + seed_offset)
            vec = vec + noise * n_rng.standard_normal(self.embedding_dim).astype(np.float32)
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def extract_primary_face(
        self,
        source: str | Path | bytes | bytearray | np.ndarray,
        policy: str = "largest",
    ) -> tuple[ExtractedFace, int]:
        raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)

        if raw.startswith(b"NO_FACE"):
            raise NoFaceDetectedError("No se detectó ningún rostro en la imagen")

        if raw.startswith(b"MULTI_FACE"):
            if policy == "strict_single":
                raise MultipleFacesDetectedError("Se detectaron 2 rostros; el registro exige exactamente 1 rostro")
            vec = self._vec_for_subject(raw.decode("utf-8", errors="ignore"))
            crop = np.zeros((112, 112, 3), dtype=np.uint8)
            return ExtractedFace(bbox=(10.0, 10.0, 80.0, 80.0), confidence=0.98, aligned_crop=crop, embedding=vec), 2

        try:
            meta = json.loads(raw.decode("utf-8"))
            subject = str(meta["subject"])
            noise = float(meta.get("noise", 0.0))
            seed_offset = int(meta.get("seed_offset", 0))
        except Exception:
            subject = raw.decode("utf-8", errors="replace")
            noise = 0.0
            seed_offset = 0

        vec = self._vec_for_subject(subject, noise=noise, seed_offset=seed_offset)
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        return ExtractedFace(bbox=(5.0, 5.0, 64.0, 64.0), confidence=0.99, aligned_crop=crop, embedding=vec), 1


@pytest.fixture
def dbms_and_face_env(tmp_path, monkeypatch):
    """Levanta el DBMS real en tmp_path conectado vía ASGI a la app facial."""
    import engine as api_engine
    from routers import query as query_router
    from routers import tables as tables_router
    from routers import vectors as vectors_router

    src_main = _load_src_main()
    dbms_api_main = _load_dbms_api_main()
    data_dir = tmp_path / "dbms_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    heap_factory = src_main.make_heap_factory(str(data_dir))
    cat = Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: src_main.make_sequential_factory(str(data_dir)),
        },
        index_buffer_factory=src_main.make_index_buffer_factory(str(data_dir)),
    )
    cat._transaction_manager = TransactionManager(str(data_dir / "transactions.wal"))

    monkeypatch.setattr(api_engine, "catalog", cat)
    monkeypatch.setattr(query_router, "catalog", cat)
    monkeypatch.setattr(tables_router, "catalog", cat)
    monkeypatch.setattr(vectors_router, "catalog", cat)

    def build_app(dim: int = 128, **setting_overrides):
        settings = FaceAppSettings(
            embedding_dim=dim,
            ivf_n_lists=2,
            ivf_n_probe=2,
            hnsw_ef_search=32,
            **setting_overrides,
        )
        dbms_client = DBMSClient(base_url="http://dbms.test", asgi_app=dbms_api_main.app)
        extractor = FakeFaceExtractor(dim=dim)
        face_app = create_app(settings=settings, client=dbms_client, extractor=extractor)
        return face_app, face_app.state.service, dbms_client

    face_app, service, dbms_client = build_app()
    yield {
        "client": ASGITestClient(face_app),
        "service": service,
        "catalog": cat,
        "build_app": build_app,
    }
    dbms_client.close()
