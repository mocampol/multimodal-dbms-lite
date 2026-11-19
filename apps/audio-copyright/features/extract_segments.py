"""
apps/audio-copyright/features/extract_segments.py — Extrae un vector MFCC por segmento
de cada canción del corpus y lo guarda en el DBMS, junto con la canción y el intervalo
de tiempo que cubre.

Pasa por la API, como load_songs.py. Es idempotente:
- Entrena el codebook solo si no existe o si cambió su configuración (k, ventana o
  salto). En ese caso los vectores guardados dejan de ser comparables y se recalculan
  todos.
- Extrae solo las canciones que todavía no tienen segmentos.
- Borra los segmentos de las canciones que ya no están en el manifiesto.
- Al final crea los índices HNSW e IVF si faltan (segment_index.py).

Cada petición de inserción lleva canciones completas, y la API la hace en una sola
transacción: si algo falla, ninguna canción queda con segmentos a medias.

Uso:
    python apps/audio-copyright/features/extract_segments.py --dbms-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps" / "audio-copyright" / "dataset"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fma_corpus import MANIFEST_NAME, read_manifest  # noqa: E402
from load_songs import HttpDBMSClient, create_table_sql  # noqa: E402
from segment_extractor import (  # noqa: E402
    CODEBOOK_SIZE,
    HOP_S,
    WINDOW_S,
    SegmentMFCCExtractor,
    SegmentVector,
    SongFrames,
)
from segment_index import SEGMENTS_TABLE, ensure_segment_indexes  # noqa: E402

CODEBOOK_NAME = "segment_codebook.pkl"
INSERT_BATCH = 200          # filas por petición, redondeado a canciones completas
SEGMENT_ID_STRIDE = 1000    # segment_id = song_id * 1000 + segment_index

SEGMENT_COLUMNS = (
    ("segment_id", "INTEGER PRIMARY KEY"),
    ("song_id", "INTEGER NOT NULL"),
    ("segment_index", "INTEGER NOT NULL"),
    ("start_s", "DOUBLE PRECISION NOT NULL"),
    ("end_s", "DOUBLE PRECISION NOT NULL"),
    ("embedding", "BYTEA NOT NULL"),          # float32 little-endian, como face_samples
    ("embedding_dim", "INTEGER NOT NULL"),
    ("model_name", "VARCHAR(64) NOT NULL"),   # con qué codebook se calculó el vector
)


class SegmentsClient(Protocol):
    def table_names(self) -> set[str]: ...
    def query(self, sql: str) -> dict: ...
    def insert_rows(self, table: str, rows: list[dict]) -> dict: ...
    def get_table(self, table: str) -> dict: ...
    def create_index(self, table: str, column: str, index_type: str, index_name: str | None = None,
                     dim: int | None = None, index_params: dict | None = None) -> dict: ...


class HttpSegmentsClient(HttpDBMSClient):
    """El cliente de load_songs.py más las interfaces vectoriales de la API: inserción con
    el vector en base64 para la columna BYTEA y creación de índices con dim y parámetros."""

    def insert_rows(self, table: str, rows: list[dict]) -> dict:
        return self._request("POST", f"/tables/{table}/rows", {"rows": rows})

    def get_table(self, table: str) -> dict:
        return self._request("GET", f"/tables/{table}")

    def create_index(self, table: str, column: str, index_type: str, index_name: str | None = None,
                     dim: int | None = None, index_params: dict | None = None) -> dict:
        payload = {"column_name": column, "index_type": index_type}
        if index_name is not None:
            payload["index_name"] = index_name
        if dim is not None:
            payload["dim"] = dim
        if index_params is not None:
            payload["index_params"] = index_params
        return self._request("POST", f"/tables/{table}/indexes", payload)


# =============================================================================
# Extracción
# =============================================================================

def _song_frames(path: str) -> SongFrames | str:
    """Tramas MFCC de una canción, o el error como texto para reportarlo sin cortar la corrida.
    Va a nivel de módulo para que ProcessPoolExecutor pueda enviarla a otros procesos."""
    try:
        return SegmentMFCCExtractor().song_frames(path)
    except ValueError as error:
        return str(error)


def compute_frames(paths: dict[int, Path], workers: int) -> tuple[dict[int, SongFrames], dict[int, str]]:
    """Decodifica y calcula el MFCC de cada canción, en paralelo si workers > 1."""
    song_ids = sorted(paths)
    sources = [str(paths[song_id]) for song_id in song_ids]
    if workers <= 1 or len(song_ids) <= 1:
        results = [_song_frames(source) for source in sources]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_song_frames, sources, chunksize=8))

    frames, failed = {}, {}
    for song_id, result in zip(song_ids, results):
        if isinstance(result, str):
            failed[song_id] = result
        else:
            frames[song_id] = result
    return frames, failed


def segment_row(song_id: int, segment: SegmentVector, model_name: str) -> dict:
    if segment.segment_index >= SEGMENT_ID_STRIDE:
        raise ValueError(f"La canción {song_id} tiene más de {SEGMENT_ID_STRIDE} segmentos")
    vector = np.asarray(segment.vector, dtype="<f4")
    return {
        "segment_id": song_id * SEGMENT_ID_STRIDE + segment.segment_index,
        "song_id": song_id,
        "segment_index": segment.segment_index,
        "start_s": segment.start_s,
        "end_s": segment.end_s,
        "embedding": base64.b64encode(vector.tobytes()).decode("ascii"),
        "embedding_dim": int(vector.size),
        "model_name": model_name,
    }


# =============================================================================
# Carga
# =============================================================================

def _load_codebook(path: Path, model_name: str) -> SegmentMFCCExtractor | None:
    """El codebook guardado, si existe y se calculó con la misma configuración."""
    if not path.exists():
        return None
    extractor = SegmentMFCCExtractor.load(path)
    return extractor if extractor.model_name == model_name and extractor.is_fitted else None


def _stored_models(client: SegmentsClient) -> dict[int, set[str]]:
    """song_id -> modelos con que están calculados sus segmentos en la tabla."""
    stored: dict[int, set[str]] = {}
    for song_id, model_name in client.query(f"SELECT song_id, model_name FROM {SEGMENTS_TABLE};")["rows"]:
        stored.setdefault(int(song_id), set()).add(model_name)
    return stored


def _insert(client: SegmentsClient, extractor: SegmentMFCCExtractor, frames: dict[int, SongFrames]) -> int:
    """Inserta los segmentos por lotes que solo contienen canciones completas."""
    batch: list[dict] = []
    inserted = 0
    for song_id in sorted(frames):
        batch.extend(segment_row(song_id, segment, extractor.model_name)
                     for segment in extractor.transform(frames[song_id]))
        if len(batch) >= INSERT_BATCH:
            client.insert_rows(SEGMENTS_TABLE, batch)
            inserted += len(batch)
            batch = []
    if batch:
        client.insert_rows(SEGMENTS_TABLE, batch)
        inserted += len(batch)
    return inserted


def extract(
    client: SegmentsClient,
    corpus_dir: Path,
    k: int = CODEBOOK_SIZE,
    window_s: float = WINDOW_S,
    hop_s: float = HOP_S,
    workers: int = 1,
) -> dict:
    songs = read_manifest(corpus_dir / MANIFEST_NAME)
    if not songs:
        raise RuntimeError(f"No hay canciones en {corpus_dir / MANIFEST_NAME}; corre antes prepare_corpus.py")
    corpus_dir = corpus_dir.resolve()
    paths = {song_id: corpus_dir / song.relative_path for song_id, song in songs.items()}
    codebook_path = corpus_dir / CODEBOOK_NAME

    frames: dict[int, SongFrames] = {}
    failed: dict[int, str] = {}
    extractor = _load_codebook(codebook_path, SegmentMFCCExtractor(k, window_s, hop_s).model_name)
    codebook = "loaded"
    if extractor is None:
        frames, failed = compute_frames(paths, workers)
        if not frames:
            raise RuntimeError(f"No se pudo extraer ninguna canción: {failed}")
        extractor = SegmentMFCCExtractor(k, window_s, hop_s).fit(list(frames.values()))
        extractor.save(codebook_path)
        codebook = "fitted"

    tables = client.table_names()
    if codebook == "fitted" and SEGMENTS_TABLE in tables:
        # con un codebook nuevo, ningún vector guardado es comparable: se empieza de cero
        client.query(f"DROP TABLE {SEGMENTS_TABLE};")
        tables.discard(SEGMENTS_TABLE)
    if SEGMENTS_TABLE not in tables:
        client.query(create_table_sql(SEGMENTS_TABLE, SEGMENT_COLUMNS))

    stored = _stored_models(client)
    stale = sorted(song_id for song_id, models in stored.items()
                   if song_id not in songs or models != {extractor.model_name})
    for song_id in stale:
        client.query(f"DELETE FROM {SEGMENTS_TABLE} WHERE song_id = {song_id};")

    missing = sorted(set(songs) - (set(stored) - set(stale)))
    pending = {song_id: paths[song_id] for song_id in missing if song_id not in frames and song_id not in failed}
    new_frames, new_failed = compute_frames(pending, workers) if pending else ({}, {})
    frames.update(new_frames)
    failed.update(new_failed)
    inserted = _insert(client, extractor, {song_id: frames[song_id] for song_id in missing if song_id in frames})

    n_segments = len(client.query(f"SELECT segment_id FROM {SEGMENTS_TABLE};")["rows"])
    created_indexes = ensure_segment_indexes(client, extractor.dim, n_segments)

    result = {
        "codebook": codebook,
        "model_name": extractor.model_name,
        "songs": len(songs),
        "inserted_songs": sum(1 for song_id in missing if song_id in frames),
        "inserted_segments": inserted,
        "deleted_songs": len(stale),
        "created_indexes": created_indexes,
        "failed": failed,
    }
    print(f"DBMS: {SEGMENTS_TABLE} {result}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dbms-url", default=os.getenv("DBMS_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--corpus-dir", type=Path, default=Path(os.getenv("AUDIO_CORPUS_DIR", "data/audio/fma")))
    parser.add_argument("--k", type=int, default=CODEBOOK_SIZE, help="tamaño del codebook (dimensión del vector)")
    parser.add_argument("--window-s", type=float, default=WINDOW_S, help="duración de cada segmento")
    parser.add_argument("--hop-s", type=float, default=HOP_S, help="salto entre segmentos")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    extract(HttpSegmentsClient(args.dbms_url), args.corpus_dir, args.k, args.window_s, args.hop_s, args.workers)


if __name__ == "__main__":
    main()
