from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans


class Quantizer:
    """
    Cuantizador basado en K-Means para generar palabras visuales o auditivas (codebook).
    """

    def __init__(self, k: int = 64, random_state: int = 42) -> None:
        if k <= 0:
            raise ValueError(f"k debe ser un entero positivo, recibido {k}")
        self.k = k
        self.random_state = random_state
        self._is_fitted = False
        self._model: MiniBatchKMeans | KMeans | None = None
        self._effective_k = k

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def centroids(self) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise ValueError("El cuantizador aún no ha sido entrenado")
        return self._model.cluster_centers_

    def codebook_size(self) -> int:
        return self.k

    def fit(self, descriptors: np.ndarray) -> Quantizer:
        """
        Entrena el diccionario de centroides K con los descriptores proporcionados.

        Args:
            descriptors: matriz de forma (N, D) con descriptores.

        Returns:
            self
        """
        if descriptors is None or len(descriptors) == 0:
            raise ValueError("No se proporcionaron descriptores para entrenar el cuantizador")

        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim != 2:
            raise ValueError(f"Los descriptores deben tener 2 dimensiones (N, D), forma recibida: {descriptors.shape}")

        n_samples = descriptors.shape[0]
        # Si hay menos muestras que clusters, ajustamos el número de clusters al número de muestras
        actual_k = min(self.k, n_samples)
        self._effective_k = actual_k

        if n_samples < 500:
            self._model = KMeans(n_clusters=actual_k, random_state=self.random_state, n_init="auto")
        else:
            self._model = MiniBatchKMeans(
                n_clusters=actual_k,
                random_state=self.random_state,
                batch_size=min(1024, n_samples),
                n_init="auto",
            )

        self._model.fit(descriptors)
        self._is_fitted = True
        return self

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Asigna cada descriptor al centroide más cercano.

        Args:
            descriptors: matriz de forma (N, D).

        Returns:
            np.ndarray: array 1D de enteros con índices de clusters [0, k-1].
        """
        if not self._is_fitted or self._model is None:
            raise ValueError("El cuantizador no está entrenado. Llame a fit() primero.")

        if descriptors is None or len(descriptors) == 0:
            raise ValueError("No se proporcionaron descriptores para cuantizar")

        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim != 2:
            raise ValueError(f"Los descriptores deben tener 2 dimensiones (N, D), forma recibida: {descriptors.shape}")

        return self._model.predict(descriptors).astype(np.int64)
