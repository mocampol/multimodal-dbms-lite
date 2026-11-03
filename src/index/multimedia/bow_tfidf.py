from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class BoWTFIDF:
    """
    Construye histogramas de frecuencias (Bag of Words) y aplica ponderación TF-IDF con normalización L2.
    """

    def __init__(self, k: int = 64) -> None:
        if k <= 0:
            raise ValueError(f"k debe ser un entero positivo, recibido {k}")
        self.k = k
        self.idf_: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, assignments_list: Sequence[np.ndarray]) -> BoWTFIDF:
        """
        Calcula las frecuencias de documentos (DF) y valores IDF para las K palabras.

        Args:
            assignments_list: lista donde cada elemento es un array 1D con los índices de cluster de un objeto.

        Returns:
            self
        """
        n_docs = len(assignments_list)
        if n_docs == 0:
            self.idf_ = np.ones(self.k, dtype=np.float32)
            self._is_fitted = True
            return self

        df = np.zeros(self.k, dtype=np.float32)
        for assignments in assignments_list:
            if assignments is not None and len(assignments) > 0:
                unique_words = np.unique(assignments)
                for word in unique_words:
                    if 0 <= word < self.k:
                        df[word] += 1.0

        # Fórmula estándar de smooth IDF: log((N + 1) / (DF + 1)) + 1
        self.idf_ = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        self.idf_ = self.idf_.astype(np.float32)
        self._is_fitted = True
        return self

    def compute(self, assignments: np.ndarray) -> np.ndarray:
        """
        Calcula el vector de características de dimensión K ponderado por TF-IDF y normalizado L2.

        Args:
            assignments: array 1D con las asignaciones de palabras/clusters para el objeto.

        Returns:
            np.ndarray: vector 1D de dimensión K, normalizado L2 (norma euclidiana = 1).
        """
        if assignments is None or len(assignments) == 0:
            raise ValueError("Las asignaciones de palabras no pueden estar vacías")

        counts = np.bincount(assignments, minlength=self.k)[: self.k].astype(np.float32)
        total = float(len(assignments))
        tf = counts / total

        if self.idf_ is not None and len(self.idf_) == self.k:
            tfidf = tf * self.idf_
        else:
            tfidf = tf

        norm = np.linalg.norm(tfidf)
        if norm > 1e-12:
            tfidf = tfidf / norm
        else:
            tfidf = np.zeros(self.k, dtype=np.float32)

        return tfidf.astype(np.float32)
