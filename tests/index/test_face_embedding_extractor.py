from pathlib import Path

import cv2
import numpy as np
import pytest

from index.multimedia import FaceEmbeddingExtractor


_DETECTION = np.array(
    [[20, 20, 120, 120, 55, 70, 105, 70, 80, 95, 60, 120, 100, 120, 0.99]],
    dtype=np.float32,
)


class _FakeDetector:
    def __init__(self, detections: np.ndarray | None = _DETECTION) -> None:
        self.detections = detections
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        return 1, self.detections


class _FakeRecognizer:
    def __init__(self, embedding_size: int = 128) -> None:
        self.embedding_size = embedding_size
        self.aligned_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        self.received_face: np.ndarray | None = None

    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        self.received_face = face
        return self.aligned_crop

    def feature(self, aligned_crop: np.ndarray) -> np.ndarray:
        return np.arange(1, self.embedding_size + 1, dtype=np.float32).reshape(1, -1)


def _create_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detections: np.ndarray | None = _DETECTION,
    embedding_size: int = 128,
) -> tuple[FaceEmbeddingExtractor, _FakeDetector, _FakeRecognizer]:
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    detector_model.write_bytes(b"test model")
    recognizer_model.write_bytes(b"test model")
    detector = _FakeDetector(detections)
    recognizer = _FakeRecognizer(embedding_size)
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *_args: detector)
    monkeypatch.setattr(cv2, "FaceRecognizerSF_create", lambda *_args: recognizer)
    extractor = FaceEmbeddingExtractor(detector_model, recognizer_model)
    return extractor, detector, recognizer


def test_detects_aligns_and_extracts_a_normalized_128d_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, detector, recognizer = _create_extractor(tmp_path, monkeypatch)
    image = np.zeros((200, 300, 3), dtype=np.uint8)

    faces = extractor.detect_and_extract(image)

    assert len(faces) == 1
    face = faces[0]
    assert face.bbox == (20.0, 20.0, 120.0, 120.0)
    assert face.confidence == pytest.approx(0.99)
    assert face.aligned_crop.shape == (112, 112, 3)
    assert face.embedding.shape == (128,)
    assert face.embedding.dtype == np.float32
    assert np.linalg.norm(face.embedding) == pytest.approx(1.0)
    assert detector.input_size == (300, 200)
    assert recognizer.received_face is not None
    assert recognizer.received_face.shape == (14,)


@pytest.mark.parametrize("detections", [None, np.empty((0, 15), dtype=np.float32)])
def test_returns_no_faces_when_detector_finds_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detections: np.ndarray | None,
) -> None:
    extractor, _, _ = _create_extractor(tmp_path, monkeypatch, detections)

    assert extractor.detect_and_extract(np.zeros((40, 50, 3), dtype=np.uint8)) == []


def test_converts_grayscale_images_to_bgr_before_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, _, recognizer = _create_extractor(tmp_path, monkeypatch)
    image = np.zeros((200, 300), dtype=np.uint8)

    faces = extractor.detect_and_extract(image)

    assert len(faces) == 1
    assert recognizer.received_face is not None
    assert faces[0].aligned_crop.shape == (112, 112, 3)


def test_accepts_image_path_and_encoded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, _, _ = _create_extractor(tmp_path, monkeypatch)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    image_path = tmp_path / "image.png"
    image_path.write_bytes(buffer.tobytes())

    assert len(extractor.detect_and_extract(image_path)) == 1
    assert len(extractor.detect_and_extract(buffer.tobytes())) == 1


def test_rejects_unexpected_embedding_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, _, _ = _create_extractor(tmp_path, monkeypatch, embedding_size=127)

    with pytest.raises(RuntimeError, match="127 dimensiones"):
        extractor.detect_and_extract(np.zeros((200, 300, 3), dtype=np.uint8))


def test_rejects_missing_model_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="modelo YuNet"):
        FaceEmbeddingExtractor(tmp_path / "missing.onnx", tmp_path / "sface.onnx")


def test_rejects_invalid_image() -> None:
    with pytest.raises(ValueError, match="uint8"):
        FaceEmbeddingExtractor._load_image(np.zeros((10, 10, 3), dtype=np.float32))


_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "corpus"
    / "face"
    / "models"
)
_YUNET_MODEL = _MODEL_DIR / "face_detection_yunet_2026may.onnx"
_SFACE_MODEL = _MODEL_DIR / "face_recognition_sface_2021dec.onnx"


@pytest.mark.skipif(
    not _YUNET_MODEL.is_file() or not _SFACE_MODEL.is_file(),
    reason="Descarga los modelos locales descritos en docs/face_embeddings.md",
)
def test_pretrained_sface_model_produces_128d_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _FakeDetector()
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *_args: detector)
    extractor = FaceEmbeddingExtractor(_YUNET_MODEL, _SFACE_MODEL)
    image = np.random.default_rng(42).integers(
        0, 256, size=(160, 160, 3), dtype=np.uint8
    )

    faces = extractor.detect_and_extract(image)

    assert len(faces) == 1
    assert faces[0].embedding.shape == (128,)
    assert np.linalg.norm(faces[0].embedding) == pytest.approx(1.0)
    assert faces[0].aligned_crop.shape == (112, 112, 3)


@pytest.mark.skipif(
    not _YUNET_MODEL.is_file() or not _SFACE_MODEL.is_file(),
    reason="Descarga los modelos locales descritos en docs/face_embeddings.md",
)
def test_pretrained_yunet_returns_no_faces_for_blank_image() -> None:
    extractor = FaceEmbeddingExtractor(_YUNET_MODEL, _SFACE_MODEL)

    assert extractor.detect_and_extract(np.zeros((320, 320, 3), dtype=np.uint8)) == []
