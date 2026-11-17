"""
apps/audio-copyright/features/segment_index.py — Índices vectoriales de audio_segments.

Crea, si faltan, un índice HNSW y uno IVF sobre audio_segments.embedding, con la API
del DBMS (POST /tables/{t}/indexes). Los dos usan coseno, porque los vectores tienen
norma 1. Tener ambos permite elegir en cada búsqueda entre la velocidad de HNSW y el
menor costo de IVF.

Se llama al final de la carga, no antes: si el índice existe mientras se insertan miles
de filas, el motor lo actualiza y lo guarda en disco con cada una. Después de crearlo,
el motor lo mantiene solo en cada INSERT y DELETE.
"""

from __future__ import annotations

import math
from typing import Protocol

SEGMENTS_TABLE = "audio_segments"
EMBEDDING_COLUMN = "embedding"
METRIC = "cosine"

HNSW_INDEX = "idx_audio_segments_hnsw"
IVF_INDEX = "idx_audio_segments_ivf"

# Parámetros elegidos con el benchmark del #85 (benchmarks/results/vector_summary.md).
HNSW_PARAMS = {"M": 16, "ef_construction": 200, "ef_search": 100}  # en audio, ef=20 ya daba recall 0.97
IVF_N_PROBE = 8


class IndexClient(Protocol):
    def get_table(self, table: str) -> dict: ...
    def create_index(self, table: str, column: str, index_type: str, index_name: str | None = None,
                     dim: int | None = None, index_params: dict | None = None) -> dict: ...


def ivf_lists(n_segments: int) -> int:
    """√N listas: con las 16 fijas del motor, IVF era más lento que la fuerza bruta (#85)."""
    return max(1, round(math.sqrt(n_segments)))


def existing_indexes(client: IndexClient) -> set[str]:
    """Tipos de índice ('hnsw', 'ivf') que ya tiene la columna de vectores."""
    return {
        index["index_type"]
        for index in client.get_table(SEGMENTS_TABLE).get("indexes", [])
        if index["column_name"] == EMBEDDING_COLUMN
    }


def ensure_segment_indexes(client: IndexClient, dim: int, n_segments: int) -> list[str]:
    """Crea los índices que falten y devuelve cuáles creó. Con la tabla vacía no crea
    ninguno, porque IVF necesita datos para calcular sus listas."""
    if n_segments <= 0:
        return []

    present = existing_indexes(client)
    created = []
    if "hnsw" not in present:
        client.create_index(SEGMENTS_TABLE, EMBEDDING_COLUMN, "hnsw", index_name=HNSW_INDEX, dim=dim,
                            index_params={"metric": METRIC, **HNSW_PARAMS})
        created.append("hnsw")
    if "ivf" not in present:
        client.create_index(SEGMENTS_TABLE, EMBEDDING_COLUMN, "ivf", index_name=IVF_INDEX, dim=dim,
                            index_params={"metric": METRIC, "n_lists": ivf_lists(n_segments),
                                          "n_probe": IVF_N_PROBE})
        created.append("ivf")
    return created
