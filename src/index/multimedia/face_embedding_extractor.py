from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

import cv2
import numpy as np


FACE_EMBEDDING_DIM = 128
ALIGNED_FACE_SIZE = (112, 112)

ImageSource = str | os.PathLike[str] | bytes | bytearray | np.ndarray


@dataclass(frozen=True)
class FaceEmbedding:
    """A detected face, its aligned crop, and its normalized SFace embedding."""

    bbox: tuple[float, float, float, float]
    confidence: float
    aligned_crop: np.ndarray
    embedding: np.ndarray

    def __post_init__(self) -> None:
        crop = np.asarray(self.aligned_crop)
        if crop.ndim != 3 or crop.shape[:2] != ALIGNED_FACE_SIZE:
            raise ValueError(
                f"El recorte alineado debe medir {ALIGNED_FACE_SIZE[1]}x"
                f"{ALIGNED_FACE_SIZE[0]} píxeles"
            )

        embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        if embedding.shape != (FACE_EMBEDDING_DIM,):
            raise ValueError(
                f"El embedding SFace debe tener {FACE_EMBEDDING_DIM} dimensiones"
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError("El embedding contiene valores no finitos")

        object.__setattr__(self, "aligned_crop", crop)
        object.__setattr__(self, "embedding", embedding)


class FaceEmbeddingExtractor:
    """Detect faces with YuNet and produce normalized 128-D SFace embeddings."""

    def __init__(
        self,
        detector_model: str | os.PathLike[str],
        recognizer_model: str | os.PathLike[str],
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        detector_path = Path(detector_model)
        recognizer_path = Path(recognizer_model)
        if not detector_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo YuNet: {detector_path}")
        if not recognizer_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo SFace: {recognizer_path}")
        if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold debe estar entre 0 y 1")
        if not math.isfinite(nms_threshold) or not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold debe estar entre 0 y 1")
        if top_k <= 0:
            raise ValueError("top_k debe ser un entero positivo")

        self._detector = cv2.FaceDetectorYN_create(
            str(detector_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF_create(str(recognizer_path), "")

    def detect_and_extract(self, source: ImageSource) -> list[FaceEmbedding]:
        """Return one aligned crop and embedding per detected face."""
        image = self._load_image(source)
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, detections = self._detector.detect(image)
        if detections is None:
            return []

        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            return []
        if detections.ndim != 2 or detections.shape[1] != 15:
            raise RuntimeError("YuNet devolvió detecciones con un formato inesperado")

        results: list[FaceEmbedding] = []
        for detection in detections:
            if not np.all(np.isfinite(detection)):
                raise RuntimeError("YuNet devolvió una detección con valores no finitos")

            aligned_crop = self._recognizer.alignCrop(image, detection[:14])
            embedding = np.asarray(
                self._recognizer.feature(aligned_crop), dtype=np.float32
            ).reshape(-1)
            if embedding.shape != (FACE_EMBEDDING_DIM,):
                raise RuntimeError(
                    f"SFace devolvió {embedding.size} dimensiones; se esperaban "
                    f"{FACE_EMBEDDING_DIM}"
                )
            if not np.all(np.isfinite(embedding)):
                raise RuntimeError("SFace devolvió un embedding con valores no finitos")

            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                raise RuntimeError("SFace devolvió un embedding de norma cero")
            embedding /= norm

            results.append(
                FaceEmbedding(
                    bbox=tuple(float(value) for value in detection[:4]),
                    confidence=float(detection[14]),
                    aligned_crop=aligned_crop,
                    embedding=embedding,
                )
            )

        return results

    @staticmethod
    def _load_image(source: ImageSource) -> np.ndarray:
        if isinstance(source, np.ndarray):
            image = source
        elif isinstance(source, (str, os.PathLike)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"No se encontró la imagen: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"No se pudo decodificar la imagen: {path}")
        elif isinstance(source, (bytes, bytearray)):
            if not source:
                raise ValueError("El buffer de imagen está vacío")
            image = cv2.imdecode(
                np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            if image is None:
                raise ValueError("No se pudo decodificar el buffer de imagen")
        else:
            raise TypeError(f"Tipo de entrada de imagen no soportado: {type(source)}")

        if image.size == 0:
            raise ValueError("La imagen está vacía")
        if image.dtype != np.uint8:
            raise ValueError("La imagen debe usar valores uint8")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Formato de imagen no soportado: {image.shape}")

        return np.ascontiguousarray(image)
