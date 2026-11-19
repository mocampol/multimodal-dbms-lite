from __future__ import annotations

import numpy as np
import pytest

from backend.config import FaceAppSettings
from backend.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from backend.extractor import (
    ExtractedFace,
    _adapt_embedding_dim,
    _build_projection_matrix,
    select_face,
)


def _dummy_face(area_side: float, dim: int = 128, seed: int = 1) -> ExtractedFace:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return ExtractedFace(
        bbox=(0.0, 0.0, area_side, area_side),
        confidence=0.95,
        aligned_crop=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=vec,
    )


def test_select_face_policies():
    with pytest.raises(NoFaceDetectedError):
        select_face([], "strict_single")

    f_small = _dummy_face(40.0, seed=1)
    f_big = _dummy_face(90.0, seed=2)

    single, total = select_face([f_small], "strict_single")
    assert single is f_small
    assert total == 1

    with pytest.raises(MultipleFacesDetectedError):
        select_face([f_small, f_big], "strict_single")

    largest, total_multi = select_face([f_small, f_big], "largest")
    assert largest is f_big
    assert total_multi == 2


@pytest.mark.parametrize("in_dim,out_dim", [(128, 64), (128, 128), (128, 256), (512, 128), (512, 512)])
def test_dynamic_embedding_projection_is_unit_norm_and_deterministic(in_dim: int, out_dim: int):
    rng = np.random.default_rng(123)
    raw = rng.standard_normal(in_dim).astype(np.float32)
    raw /= np.linalg.norm(raw)

    proj = None if in_dim == out_dim else _build_projection_matrix(in_dim, out_dim)
    out1 = _adapt_embedding_dim(raw, proj, out_dim)
    out2 = _adapt_embedding_dim(raw, proj, out_dim)

    assert out1.shape == (out_dim,)
    assert out1.dtype == np.float32
    assert float(np.linalg.norm(out1)) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_settings_resolved_dim_defaults_and_override():
    sface_default = FaceAppSettings(extractor_backend="sface", embedding_dim=None)
    assert sface_default.resolved_dim() == 128

    facenet_default = FaceAppSettings(extractor_backend="facenet", embedding_dim=None)
    assert facenet_default.resolved_dim() == 512

    custom = FaceAppSettings(extractor_backend="facenet", embedding_dim=256)
    assert custom.resolved_dim() == 256


def test_real_facenet_extractor_produces_normalized_512d_and_dynamic_dim(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("facenet_pytorch")
    from backend.extractor import FaceNetExtractorAdapter, build_extractor

    adapter = build_extractor(FaceAppSettings(extractor_backend="facenet", embedding_dim=0))
    assert isinstance(adapter, FaceNetExtractorAdapter)
    assert adapter.embedding_dim == 512
    assert adapter.model_name == "mtcnn_facenet_vggface2"

    # Imagen en blanco -> 0 rostros con MTCNN real
    with pytest.raises(NoFaceDetectedError):
        adapter.extract_primary_face(np.zeros((200, 200, 3), dtype=np.uint8))

    # Caja detectada -> InceptionResnetV1 real produce vector 512-d L2 normalizado
    noise_img = np.random.default_rng(77).integers(0, 256, size=(220, 220, 3), dtype=np.uint8)
    monkeypatch.setattr(
        adapter._mtcnn,
        "detect",
        lambda _img: (np.array([[20.0, 20.0, 140.0, 140.0]]), np.array([0.99])),
    )
    face, total = adapter.extract_primary_face(noise_img, policy="strict_single")
    assert total == 1
    assert face.embedding.shape == (512,)
    assert face.embedding.dtype == np.float32
    assert float(np.linalg.norm(face.embedding)) == pytest.approx(1.0, abs=1e-4)
    assert face.aligned_crop.shape == (160, 160, 3)

    # Proyección dinámica a 256-d y manejo de múltiples rostros
    adapter_256 = build_extractor(FaceAppSettings(extractor_backend="facenet", embedding_dim=256))
    monkeypatch.setattr(
        adapter_256._mtcnn,
        "detect",
        lambda _img: (
            np.array([[10.0, 10.0, 60.0, 60.0], [80.0, 80.0, 190.0, 190.0]]),
            np.array([0.91, 0.98]),
        ),
    )
    with pytest.raises(MultipleFacesDetectedError):
        adapter_256.extract_primary_face(noise_img, policy="strict_single")

    largest_face, total_multi = adapter_256.extract_primary_face(noise_img, policy="largest")
    assert total_multi == 2
    assert largest_face.embedding.shape == (256,)
    assert float(np.linalg.norm(largest_face.embedding)) == pytest.approx(1.0, abs=1e-4)
    assert largest_face.bbox == (80.0, 80.0, 110.0, 110.0)


def test_real_sface_adapter_if_models_present():
    from backend.extractor import SFaceExtractorAdapter, build_extractor

    settings = FaceAppSettings(extractor_backend="sface", embedding_dim=0)
    if not settings.yunet_model_path.is_file() or not settings.sface_model_path.is_file():
        pytest.skip("Modelos YuNet/SFace locales no descargados")

    adapter = build_extractor(settings)
    assert isinstance(adapter, SFaceExtractorAdapter)
    assert adapter.embedding_dim == 128
    with pytest.raises(NoFaceDetectedError):
        adapter.extract_primary_face(np.zeros((240, 240, 3), dtype=np.uint8))


def test_sface_adapter_serializes_concurrent_inference():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from types import SimpleNamespace

    from backend.extractor import SFaceExtractorAdapter

    settings = FaceAppSettings(extractor_backend="sface", embedding_dim=0)
    if not settings.yunet_model_path.is_file() or not settings.sface_model_path.is_file():
        pytest.skip("Modelos YuNet/SFace locales no descargados")

    adapter = SFaceExtractorAdapter(
        detector_model=settings.yunet_model_path,
        recognizer_model=settings.sface_model_path,
    )

    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class _FakeInner:
        def detect_and_extract(self, _source):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1
            vec = np.ones(128, dtype=np.float32)
            return [
                SimpleNamespace(
                    bbox=(0.0, 0.0, 50.0, 50.0),
                    confidence=0.99,
                    aligned_crop=np.zeros((112, 112, 3), dtype=np.uint8),
                    embedding=vec / np.linalg.norm(vec),
                )
            ]

    adapter._inner = _FakeInner()  # type: ignore[assignment]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: adapter.extract_primary_face(b"fake"), range(8)))

    assert len(results) == 8
    assert max_active == 1
