"""
src/hnswlib.py — Implementación de respaldo (pure Python/NumPy) de hnswlib.

Provee la interfaz mínima de hnswlib.Index requerida por HNSWIndex cuando
el paquete binario C++ no está instalado o no se puede compilar en el entorno.
"""
from __future__ import annotations

import pickle
from typing import Any
import numpy as np

__version__ = "0.8.0"


class Index:
    def __init__(self, space: str = "cosine", dim: int = 128) -> None:
        self.space = space.lower()
        self.dim = int(dim)
        self.max_elements = 10000
        self.ef_construction = 200
        self.M = 16
        self.ef = 50
        self.num_threads = -1
        self._data: dict[int, np.ndarray] = {}
        self._deleted: set[int] = set()

    def init_index(self, max_elements: int, ef_construction: int = 200, M: int = 16) -> None:
        self.max_elements = int(max_elements)
        self.ef_construction = int(ef_construction)
        self.M = int(M)
        self._data.clear()
        self._deleted.clear()

    def set_ef(self, ef: int) -> None:
        self.ef = int(ef)

    def set_num_threads(self, num_threads: int) -> None:
        self.num_threads = int(num_threads)

    def add_items(self, data: np.ndarray, ids: np.ndarray | list[int]) -> None:
        data_arr = np.asarray(data, dtype=np.float32)
        if data_arr.ndim == 1:
            data_arr = data_arr.reshape(1, -1)
        ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)

        for vec, item_id in zip(data_arr, ids_arr):
            iid = int(item_id)
            self._data[iid] = vec.copy()
            self._deleted.discard(iid)

    def mark_deleted(self, label: int) -> None:
        iid = int(label)
        if iid in self._data:
            self._deleted.add(iid)

    def get_current_count(self) -> int:
        return len(self._data)

    def resize_index(self, new_max: int) -> None:
        self.max_elements = int(new_max)

    def knn_query(self, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        q_arr = np.asarray(query, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)

        active_ids = [lbl for lbl in self._data if lbl not in self._deleted]
        if not active_ids:
            return np.empty((q_arr.shape[0], 0), dtype=np.int64), np.empty((q_arr.shape[0], 0), dtype=np.float32)

        active_matrix = np.stack([self._data[lbl] for lbl in active_ids])  # (M, dim)
        all_labels: list[list[int]] = []
        all_dists: list[list[float]] = []

        actual_k = min(int(k), len(active_ids))

        for q in q_arr:
            if self.space == "l2":
                diff = active_matrix - q
                dists = np.sum(diff * diff, axis=1)  # L2 al cuadrado
            elif self.space == "cosine":
                dot = active_matrix @ q
                norm_a = np.linalg.norm(active_matrix, axis=1)
                norm_q = np.linalg.norm(q)
                denom = np.maximum(norm_a * norm_q, 1e-12)
                sim = dot / denom
                dists = np.clip(1.0 - sim, 0.0, 2.0)
            elif self.space == "ip":
                dot = active_matrix @ q
                dists = 1.0 - dot
            else:
                diff = active_matrix - q
                dists = np.sum(diff * diff, axis=1)

            top_idx = np.argsort(dists)[:actual_k]
            all_labels.append([active_ids[i] for i in top_idx])
            all_dists.append([float(dists[i]) for i in top_idx])

        return np.asarray(all_labels, dtype=np.int64), np.asarray(all_dists, dtype=np.float32)

    def save_index(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "space": self.space,
                    "dim": self.dim,
                    "max_elements": self.max_elements,
                    "ef_construction": self.ef_construction,
                    "M": self.M,
                    "ef": self.ef,
                    "num_threads": self.num_threads,
                    "data": self._data,
                    "deleted": self._deleted,
                },
                f,
            )

    def load_index(self, path: str, max_elements: int = 0) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.space = state["space"]
        self.dim = state["dim"]
        self.max_elements = int(max_elements or state.get("max_elements", 10000))
        self.ef_construction = state.get("ef_construction", 200)
        self.M = state.get("M", 16)
        self.ef = state.get("ef", 50)
        self.num_threads = state.get("num_threads", -1)
        self._data = state["data"]
        self._deleted = state["deleted"]
