from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np


class VectorMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    COSINE = "cosine"

    @classmethod
    def from_str(cls, name: str | VectorMetric) -> VectorMetric:
        if isinstance(name, VectorMetric):
            return name
        clean = name.strip().lower()
        if clean in ("euclidean", "l2"):
            return cls.EUCLIDEAN
        if clean in ("dot_product", "dot", "ip", "inner_product"):
            return cls.DOT_PRODUCT
        if clean in ("cosine", "cos"):
            return cls.COSINE
        raise ValueError(f"Métrica desconocida: '{name}'. Opciones válidas: euclidean, dot_product, cosine")


@dataclass
class SearchResult:
    """Resultado individual de una búsqueda de similitud vectorial."""
    rid: Any
    distance: float
    score: float

    def __iter__(self):
        # Permite desempaquetar como (rid, score)
        return iter((self.rid, self.score))

    def __getitem__(self, idx: int) -> Any:
        if idx == 0:
            return self.rid
        if idx == 1:
            return self.score
        if idx == 2:
            return self.distance
        raise IndexError("SearchResult index fuera de rango [0..2]")


def compute_distance(v1: np.ndarray, v2: np.ndarray, metric: VectorMetric) -> float:
    """
    Calcula la distancia entre dos vectores según la métrica indicada.
    Menor distancia siempre significa mayor cercanía/similitud.
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    if metric == VectorMetric.EUCLIDEAN:
        return float(np.linalg.norm(v1 - v2))

    elif metric == VectorMetric.COSINE:
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 1.0
        similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        similarity = max(-1.0, min(1.0, similarity))
        # Distancia coseno en [0, 2]
        return 1.0 - similarity

    elif metric == VectorMetric.DOT_PRODUCT:
        # Mayor producto punto -> menor distancia negativa
        return -float(np.dot(v1, v2))

    raise ValueError(f"Métrica no soportada: {metric}")


def distance_to_score(distance: float, metric: VectorMetric) -> float:
    """
    Convierte una distancia en un score de similitud (mayor score = más similar).
    """
    if metric == VectorMetric.EUCLIDEAN:
        return 1.0 / (1.0 + max(0.0, distance))
    elif metric == VectorMetric.COSINE:
        # Distancia = 1 - sim => sim = 1 - dist
        return 1.0 - distance
    elif metric == VectorMetric.DOT_PRODUCT:
        # Distancia = -dot => score = -dist = dot
        return -distance
    raise ValueError(f"Métrica no soportada: {metric}")
