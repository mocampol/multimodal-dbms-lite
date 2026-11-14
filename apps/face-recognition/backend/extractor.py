from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Protocol, Sequence

import cv2
import numpy as np

from index.multimedia import FACE_EMBEDDING_DIM, FaceEmbeddingExtractor

from .config import FaceAppSettings
from .exceptions import MultipleFacesDetectedError, NoFaceDetectedError


@dataclass(frozen=True)
class ExtractedFace:
    """Rostro detectado y su embedding L2-normalizado de dimensión `embedding.size`."""

    bbox: tuple[float, float, float, float]
    confidence: float
    aligned_crop: np.ndarray
    embedding: np.ndarray


class FaceVectorExtractor(Protocol):
    """Contrato común para extractores faciales intercambiables."""

    model_name: str
    embedding_dim: int

    def extract_primary_face(
        self,
        source: str | Path | bytes | bytearray | np.ndarray,
        policy: str = "largest",
    ) -> tuple[ExtractedFace, int]:
        """Extrae el rostro principal según `policy` ('largest' o 'strict_single').
        Retorna (rostro_seleccionado, total_rostros_detectados)."""
        ...


def _build_projection_matrix(in_dim: int, out_dim: int) -> np.ndarray:
    """Proyección ortogonal determinista (QR) para experimentar con distintas dimensiones."""
    rng = np.random.default_rng(seed=in_dim * 1009 + out_dim)
    if out_dim <= in_dim:
        q, _ = np.linalg.qr(rng.standard_normal((in_dim, out_dim)))
        return q.astype(np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((out_dim, in_dim)))
    return q.T.astype(np.float32)


def _adapt_embedding_dim(
    embedding: np.ndarray,
    projection: np.ndarray | None,
    target_dim: int,
) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if projection is not None:
        vec = (vec @ projection).astype(np.float32, copy=False)
    if vec.size != target_dim:
        raise ValueError(
            f"Dimensión del embedding ({vec.size}) no coincide con target_dim={target_dim}"
        )
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("El embedding proyectado tiene norma cero")
    return (vec / norm).astype(np.float32, copy=False)


def select_face(faces: Sequence[ExtractedFace], policy: str) -> tuple[ExtractedFace, int]:
    """Aplica la política de selección de rostros ('largest' o 'strict_single')."""
    if not faces:
        raise NoFaceDetectedError("No se detectó ningún rostro en la imagen")
    count = len(faces)
    if policy == "strict_single":
        if count != 1:
            raise MultipleFacesDetectedError(
                f"Se detectaron {count} rostros; el registro exige exactamente 1 rostro"
            )
        return faces[0], count
    if policy == "largest":
        largest = max(faces, key=lambda f: max(0.0, f.bbox[2]) * max(0.0, f.bbox[3]))
        return largest, count
    raise ValueError(f"Política de detección desconocida: {policy!r}")


class SFaceExtractorAdapter:
    """Adaptador sobre FaceEmbeddingExtractor (YuNet + SFace) con soporte de dim ajustable."""

    def __init__(
        self,
        detector_model: str | Path,
        recognizer_model: str | Path,
        target_dim: int = FACE_EMBEDDING_DIM,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
    ) -> None:
        if target_dim <= 0:
            raise ValueError("target_dim debe ser positivo")
        self._inner = FaceEmbeddingExtractor(
            detector_model=detector_model,
            recognizer_model=recognizer_model,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        self.embedding_dim = int(target_dim)
        self._projection = (
            _build_projection_matrix(FACE_EMBEDDING_DIM, self.embedding_dim)
            if self.embedding_dim != FACE_EMBEDDING_DIM
            else None
        )
        self.model_name = (
            "yunet_sface"
            if self.embedding_dim == FACE_EMBEDDING_DIM
            else f"yunet_sface_proj{self.embedding_dim}"
        )
        self._infer_lock = threading.Lock()

    def extract_primary_face(
        self,
        source: str | Path | bytes | bytearray | np.ndarray,
        policy: str = "largest",
    ) -> tuple[ExtractedFace, int]:
        # YuNet/SFace (cv2 DNN) muta buffers internos por llamada; serializar evita race conditions.
        with self._infer_lock:
            raw_faces = self._inner.detect_and_extract(source)
        adapted = [
            ExtractedFace(
                bbox=item.bbox,
                confidence=item.confidence,
                aligned_crop=item.aligned_crop,
                embedding=_adapt_embedding_dim(item.embedding, self._projection, self.embedding_dim),
            )
            for item in raw_faces
        ]
        return select_face(adapted, policy)


class FaceNetExtractorAdapter:
    """Adaptador opcional para facenet-pytorch (MTCNN + InceptionResnetV1, 512-d nativa)."""

    NATIVE_DIM = 512

    def __init__(
        self,
        pretrained: str = "vggface2",
        target_dim: int = NATIVE_DIM,
        device: str | None = None,
    ) -> None:
        if target_dim <= 0:
            raise ValueError("target_dim debe ser positivo")
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError as exc:
            raise ImportError(
                "Para usar backend='facenet' instala torch, torchvision y facenet-pytorch"
            ) from exc

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._device = torch.device(resolved_device)
        self._mtcnn = MTCNN(image_size=160, margin=14, keep_all=True, device=self._device)
        self._resnet = InceptionResnetV1(pretrained=pretrained).eval().to(self._device)
        self.embedding_dim = int(target_dim)
        self._projection = (
            _build_projection_matrix(self.NATIVE_DIM, self.embedding_dim)
            if self.embedding_dim != self.NATIVE_DIM
            else None
        )
        self.model_name = (
            f"mtcnn_facenet_{pretrained}"
            if self.embedding_dim == self.NATIVE_DIM
            else f"mtcnn_facenet_{pretrained}_proj{self.embedding_dim}"
        )
        self._infer_lock = threading.Lock()

    def extract_primary_face(
        self,
        source: str | Path | bytes | bytearray | np.ndarray,
        policy: str = "largest",
    ) -> tuple[ExtractedFace, int]:
        from PIL import Image

        bgr = FaceEmbeddingExtractor._load_image(source)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with self._infer_lock:
            boxes, probs = self._mtcnn.detect(pil_img)
            if boxes is None or len(boxes) == 0:
                raise NoFaceDetectedError("No se detectó ningún rostro en la imagen")

            aligned_batch = self._mtcnn.extract(pil_img, boxes, save_path=None)
            if aligned_batch is None:
                raise NoFaceDetectedError("No se pudo alinear ningún rostro detectado")
            if aligned_batch.ndim == 3:
                aligned_batch = aligned_batch.unsqueeze(0)

            with self._torch.no_grad():
                embeddings = self._resnet(aligned_batch.to(self._device)).cpu().numpy()

        faces: list[ExtractedFace] = []
        for idx, box in enumerate(boxes):
            if idx >= len(embeddings):
                break
            x1, y1, x2, y2 = [float(v) for v in box]
            crop_tensor = aligned_batch[idx].permute(1, 2, 0).cpu().numpy()
            crop_uint8 = np.clip((crop_tensor * 128.0) + 127.5, 0, 255).astype(np.uint8)
            faces.append(
                ExtractedFace(
                    bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                    confidence=float(probs[idx]) if probs is not None and probs[idx] is not None else 1.0,
                    aligned_crop=cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2BGR),
                    embedding=_adapt_embedding_dim(embeddings[idx], self._projection, self.embedding_dim),
                )
            )
        return select_face(faces, policy)


def build_extractor(settings: FaceAppSettings) -> FaceVectorExtractor:
    """Construye el extractor facial configurado (sface por defecto o facenet opcional)."""
    backend = settings.extractor_backend.strip().lower()
    target_dim = settings.resolved_dim()
    if backend == "sface":
        return SFaceExtractorAdapter(
            detector_model=settings.yunet_model_path,
            recognizer_model=settings.sface_model_path,
            target_dim=target_dim,
        )
    if backend == "facenet":
        return FaceNetExtractorAdapter(
            pretrained=settings.facenet_pretrained,
            target_dim=target_dim,
        )
    raise ValueError(f"FACE_BACKEND desconocido: {settings.extractor_backend!r} (usa 'sface' o 'facenet')")
