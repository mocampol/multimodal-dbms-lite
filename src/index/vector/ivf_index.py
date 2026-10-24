from __future__ import annotations

import os
import pickle
from typing import Any, Sequence
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from .base_vector_index import BaseVectorIndex
from .metric import VectorMetric, SearchResult, compute_distance, distance_to_score


class IVFIndex(BaseVectorIndex):
    """
    Índice de Archivo Invertido (Inverted File Index - IVF).
    Agrupa vectores en celdas Voronoi mediante centroides K-Means
    y examina las listas invertidas de las n_probe celdas más cercanas.
    """

    def __init__(
        self,
        dim: int,
        metric: VectorMetric | str = VectorMetric.COSINE,
        n_lists: int = 16,
        n_probe: int = 4,
    ) -> None:
        super().__init__(dim=dim, metric=metric)
        if n_lists <= 0:
            raise ValueError(f"n_lists debe ser mayor a 0, recibido {n_lists}")
        if n_probe <= 0:
            raise ValueError(f"n_probe debe ser mayor a 0, recibido {n_probe}")

        self.n_lists = n_lists
        self.n_probe = n_probe
        self.centroids: np.ndarray | None = None
        # inverted_lists: centroid_id -> list of (rid, vector)
        self.inverted_lists: dict[int, list[tuple[Any, np.ndarray]]] = {}
        self._count: int = 0

    def __len__(self) -> int:
        return self._count

    def build(self, vectors: Sequence[np.ndarray], rids: Sequence[Any]) -> None:
        if len(vectors) != len(rids):
            raise ValueError("La cantidad de vectores debe ser igual a la cantidad de RIDs")

        self.inverted_lists = {}
        self._count = 0

        if len(vectors) == 0:
            self.centroids = None
            return

        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Los vectores deben tener forma (N, {self.dim}), forma recibida: {X.shape}")

        n_samples = len(X)
        actual_clusters = min(self.n_lists, n_samples)

        if n_samples < 200:
            clusterer = KMeans(n_clusters=actual_clusters, random_state=42, n_init="auto")
        else:
            clusterer = MiniBatchKMeans(
                n_clusters=actual_clusters,
                random_state=42,
                batch_size=min(1024, n_samples),
                n_init="auto",
            )

        clusterer.fit(X)
        self.centroids = clusterer.cluster_centers_.astype(np.float32)

        for i in range(len(self.centroids)):
            self.inverted_lists[i] = []

        labels = clusterer.predict(X)
        for i, (rid, vec, label) in enumerate(zip(rids, X, labels)):
            self.inverted_lists[int(label)].append((rid, vec))
            self._count += 1

    def add(self, vector: np.ndarray, rid: Any) -> None:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if len(vec) != self.dim:
            raise ValueError(f"Dimensión del vector ({len(vec)}) no coincide con la dimensión del índice ({self.dim})")

        if self.centroids is None or len(self.centroids) == 0:
            self.centroids = vec.reshape(1, self.dim)
            self.inverted_lists = {0: [(rid, vec)]}
            self._count = 1
            return

        # Encontrar centroide más cercano (por distancia Euclidiana a los centroides)
        dists = np.linalg.norm(self.centroids - vec, axis=1)
        best_cluster = int(np.argmin(dists))

        if best_cluster not in self.inverted_lists:
            self.inverted_lists[best_cluster] = []

        self.inverted_lists[best_cluster].append((rid, vec))
        self._count += 1

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        metric: VectorMetric | str | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        if self._count == 0 or self.centroids is None or len(self.centroids) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if len(q) != self.dim:
            raise ValueError(f"Dimensión del query ({len(q)}) no coincide con dim={self.dim}")

        target_metric = VectorMetric.from_str(metric) if metric is not None else self.metric

        # Identificar los n_probe centroides más cercanos a la consulta
        centroid_dists = np.linalg.norm(self.centroids - q, axis=1)
        probe_count = min(self.n_probe, len(self.centroids))
        probed_clusters = np.argsort(centroid_dists)[:probe_count]

        candidates: list[SearchResult] = []
        for cluster_id in probed_clusters:
            items = self.inverted_lists.get(int(cluster_id), [])
            for rid, item_vec in items:
                dist = compute_distance(q, item_vec, target_metric)
                score = distance_to_score(dist, target_metric)
                candidates.append(SearchResult(rid=rid, distance=dist, score=score))

        # Ordenar por distancia ascendente (menor distancia = más similar)
        candidates.sort(key=lambda r: r.distance)
        return candidates[:k]

    def save(self, path: str | None = None) -> None:
        target_path = path or getattr(self, "_storage_path", None)
        if target_path is None:
            raise ValueError("Ruta de archivo requerida para guardar IVFIndex")
        self._storage_path = target_path

        data = {
            "dim": self.dim,
            "metric": self.metric.value,
            "n_lists": self.n_lists,
            "n_probe": self.n_probe,
            "centroids": self.centroids,
            "inverted_lists": self.inverted_lists,
            "count": self._count,
        }
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> IVFIndex:
        with open(path, "rb") as f:
            data = pickle.load(f)

        idx = cls(
            dim=data["dim"],
            metric=data["metric"],
            n_lists=data["n_lists"],
            n_probe=data["n_probe"],
        )
        idx.centroids = data["centroids"]
        idx.inverted_lists = data["inverted_lists"]
        idx._count = data["count"]
        return idx
