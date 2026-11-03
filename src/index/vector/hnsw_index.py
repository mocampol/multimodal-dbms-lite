from __future__ import annotations

import math
import os
import pickle
from typing import Any, Sequence
import hnswlib
import numpy as np

from .base_vector_index import BaseVectorIndex
from .metric import VectorMetric, SearchResult, distance_to_score, compute_distance


class HNSWIndex(BaseVectorIndex):
    """
    Índice basado en Grafos de Mundo Pequeño Navegable Jerárquico (HNSW).
    Implementado sobre hnswlib con soporte para métricas Euclidiana, Coseno y Producto Punto.
    """

    SPACE_MAP = {
        VectorMetric.EUCLIDEAN: "l2",
        VectorMetric.COSINE: "cosine",
        VectorMetric.DOT_PRODUCT: "ip",
    }

    def __init__(
        self,
        dim: int,
        metric: VectorMetric | str = VectorMetric.COSINE,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        super().__init__(dim=dim, metric=metric)
        self.max_elements = max(100, max_elements)
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = 50

        self.space = self.SPACE_MAP[self.metric]
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.set_ef(self.ef_search)

        self.label_to_rid: dict[int, Any] = {}
        self.rid_to_label: dict[Any, int] = {}
        self._next_label: int = 0
        self._raw_vectors: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.label_to_rid)

    def build(self, vectors: Sequence[np.ndarray], rids: Sequence[Any]) -> None:
        if len(vectors) != len(rids):
            raise ValueError("La cantidad de vectores debe ser igual a la cantidad de RIDs")

        self.label_to_rid.clear()
        self.rid_to_label.clear()
        self._raw_vectors.clear()
        self._next_label = 0

        n_samples = len(vectors)
        if n_samples == 0:
            return

        target_max = max(self.max_elements, n_samples * 2)
        self.max_elements = target_max
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.set_ef(self.ef_search)

        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Los vectores deben tener forma (N, {self.dim}), recibido: {X.shape}")

        labels = np.arange(n_samples, dtype=np.int64)
        for label, rid, vec in zip(labels, rids, X):
            lbl = int(label)
            self.label_to_rid[lbl] = rid
            self.rid_to_label[rid] = lbl
            self._raw_vectors[lbl] = vec

        self._index.add_items(X, labels)
        self._next_label = n_samples

    def add(self, vector: np.ndarray, rid: Any) -> None:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if len(vec) != self.dim:
            raise ValueError(f"Dimensión del vector ({len(vec)}) no coincide con la dimensión del índice ({self.dim})")

        if len(self.label_to_rid) >= self.max_elements:
            new_max = self.max_elements * 2
            self._index.resize_index(new_max)
            self.max_elements = new_max

        label = self._next_label
        self._next_label += 1

        self.label_to_rid[label] = rid
        self.rid_to_label[rid] = label
        self._raw_vectors[label] = vec

        self._index.add_items(vec.reshape(1, self.dim), np.array([label], dtype=np.int64))

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        metric: VectorMetric | str | None = None,
    ) -> list[SearchResult]:
        if k <= 0 or len(self.label_to_rid) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if len(q) != self.dim:
            raise ValueError(f"Dimensión del query ({len(q)}) no coincide con dim={self.dim}")

        target_metric = VectorMetric.from_str(metric) if metric is not None else self.metric
        actual_k = min(k, len(self.label_to_rid))

        # Si la métrica coincide con la construcción de HNSW, usamos directamente el grafo HNSW
        if target_metric == self.metric:
            labels, distances = self._index.knn_query(q.reshape(1, self.dim), k=actual_k)
            labels = labels[0]
            distances = distances[0]

            results: list[SearchResult] = []
            for lbl, raw_dist in zip(labels, distances):
                lbl = int(lbl)
                rid = self.label_to_rid[lbl]
                if self.metric == VectorMetric.EUCLIDEAN:
                    # hnswlib devuelve L2 al cuadrado
                    dist = math.sqrt(max(0.0, float(raw_dist)))
                elif self.metric == VectorMetric.COSINE:
                    dist = float(raw_dist)
                elif self.metric == VectorMetric.DOT_PRODUCT:
                    # hnswlib IP devuelve 1.0 - dot_product
                    # Nuestra distancia de dot_product es -dot_product
                    dot_prod = 1.0 - float(raw_dist)
                    dist = -dot_prod
                else:
                    dist = float(raw_dist)

                score = distance_to_score(dist, self.metric)
                results.append(SearchResult(rid=rid, distance=dist, score=score))

            results.sort(key=lambda r: r.distance)
            return results

        # Si se consulta con una métrica distinta a la nativa del grafo HNSW, recalculamos
        # de forma transparente preservando compatibilidad de API independiente de métrica
        fetch_k = min(max(actual_k * 3, 20), len(self.label_to_rid))
        labels, _ = self._index.knn_query(q.reshape(1, self.dim), k=fetch_k)
        candidates: list[SearchResult] = []
        for lbl in labels[0]:
            lbl = int(lbl)
            rid = self.label_to_rid[lbl]
            vec = self._raw_vectors[lbl]
            dist = compute_distance(q, vec, target_metric)
            score = distance_to_score(dist, target_metric)
            candidates.append(SearchResult(rid=rid, distance=dist, score=score))

        candidates.sort(key=lambda r: r.distance)
        return candidates[:actual_k]

    def save(self, path: str | None = None) -> None:
        target_path = path or getattr(self, "_storage_path", None)
        if target_path is None:
            raise ValueError("Ruta de archivo requerida para guardar HNSWIndex")
        self._storage_path = target_path

        bin_path = target_path + ".bin"
        self._index.save_index(bin_path)

        meta = {
            "dim": self.dim,
            "metric": self.metric.value,
            "max_elements": self.max_elements,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "ef_search": self.ef_search,
            "label_to_rid": self.label_to_rid,
            "rid_to_label": self.rid_to_label,
            "next_label": self._next_label,
            "raw_vectors": self._raw_vectors,
            "bin_file": os.path.basename(bin_path),
        }
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> HNSWIndex:
        with open(path, "rb") as f:
            meta = pickle.load(f)

        idx = cls(
            dim=meta["dim"],
            metric=meta["metric"],
            max_elements=meta["max_elements"],
            ef_construction=meta["ef_construction"],
            M=meta["M"],
        )
        idx.ef_search = meta["ef_search"]
        idx.label_to_rid = meta["label_to_rid"]
        idx.rid_to_label = meta["rid_to_label"]
        idx._next_label = meta["next_label"]
        idx._raw_vectors = meta["raw_vectors"]

        bin_path = os.path.join(os.path.dirname(os.path.abspath(path)), meta["bin_file"])
        idx._index.load_index(bin_path, max_elements=meta["max_elements"])
        idx._index.set_ef(idx.ef_search)
        return idx
