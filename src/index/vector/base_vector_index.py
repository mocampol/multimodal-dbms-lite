from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np

from .metric import VectorMetric, SearchResult


class BaseVectorIndex(ABC):
    """Interfaz abstracta base para índices vectoriales (IVF, HNSW)."""

    def __init__(self, dim: int, metric: VectorMetric | str = VectorMetric.COSINE) -> None:
        if dim <= 0:
            raise ValueError(f"dim debe ser un entero positivo, recibido {dim}")
        self.dim = dim
        self.metric = VectorMetric.from_str(metric)

    @abstractmethod
    def build(self, vectors: Sequence[np.ndarray], rids: Sequence[Any]) -> None:
        """Construye el índice a partir de una colección inicial de vectores y RIDs."""
        pass

    @abstractmethod
    def add(self, vector: np.ndarray, rid: Any) -> None:
        """Inserta un nuevo vector con su correspondiente RID en el índice."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        metric: VectorMetric | str | None = None,
    ) -> list[SearchResult]:
        """
        Busca los k vecinos más cercanos ordenados según la métrica indicada.

        Returns:
            list[SearchResult]: lista ordenada de los k elementos más cercanos.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persiste el índice a disco en la ruta especificada."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> BaseVectorIndex:
        """Carga el índice desde disco."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Número de elementos indexados."""
        pass
