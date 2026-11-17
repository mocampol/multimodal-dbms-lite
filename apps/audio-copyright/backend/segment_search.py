"""
apps/audio-copyright/backend/segment_search.py — Busca, para cada segmento de un audio,
los segmentos más parecidos de la base de referencia (audio_segments).

El audio se divide y se vectoriza con el mismo codebook que la base (#111), y cada
segmento se busca con POST /tables/audio_segments/vector-search, por HNSW o IVF. Cada
candidato vuelve con su score, su canción y dos referencias temporales: el intervalo
del audio buscado y el de la canción original.

Los metadatos de canciones y artistas se leen una sola vez y se unen en memoria: el
motor no admite búsqueda por similitud junto con JOIN.

Uso (con la API corriendo):
    python apps/audio-copyright/backend/segment_search.py remix.mp3 --k 5 --index hnsw
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parents[1]
for path in (ROOT / "src", APP_DIR / "dataset", APP_DIR / "features"):
    sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from extract_segments import CODEBOOK_NAME, HttpSegmentsClient  # noqa: E402
from load_songs import ARTISTS_TABLE, SONGS_TABLE  # noqa: E402
from segment_extractor import SegmentMFCCExtractor  # noqa: E402
from segment_index import EMBEDDING_COLUMN, HNSW_PARAMS, IVF_N_PROBE, METRIC, SEGMENTS_TABLE  # noqa: E402

INDEX_TYPES = ("hnsw", "ivf")
METRICS = ("cosine", "euclidean", "dot_product")


@dataclass(frozen=True)
class SearchSettings:
    """Valores por defecto de la búsqueda; cada uno se puede cambiar por variable de entorno."""

    dbms_api_url: str = os.getenv("DBMS_API_URL", "http://127.0.0.1:8000")
    codebook_path: Path = Path(os.getenv(
        "AUDIO_CODEBOOK", str(Path(os.getenv("AUDIO_CORPUS_DIR", "data/audio/fma")) / CODEBOOK_NAME)))
    index_type: str = os.getenv("AUDIO_INDEX_TYPE", "hnsw")
    metric: str = os.getenv("AUDIO_METRIC", METRIC)
    k: int = int(os.getenv("AUDIO_DEFAULT_K", "5"))
    ef_search: int = int(os.getenv("AUDIO_HNSW_EF_SEARCH", str(HNSW_PARAMS["ef_search"])))
    n_probe: int = int(os.getenv("AUDIO_IVF_N_PROBE", str(IVF_N_PROBE)))


@dataclass(frozen=True)
class Candidate:
    """Un segmento de la base parecido al buscado."""

    song_id: int
    title: str
    artist: str
    license_family: str
    track_url: str | None
    segment_id: int
    start_s: float      # intervalo dentro de la canción original
    end_s: float
    score: float        # con coseno, la similitud: 1.0 es idéntico
    distance: float


@dataclass(frozen=True)
class SegmentMatches:
    """Un segmento del audio buscado y sus candidatos, del más al menos parecido."""

    segment_index: int
    start_s: float      # intervalo dentro del audio buscado
    end_s: float
    candidates: list[Candidate]


class SearchClient(Protocol):
    def query(self, sql: str) -> dict: ...
    def vector_search(self, table: str, vector, column: str, k: int, index_type: str, metric: str,
                      ef_search: int | None = None, n_probe: int | None = None) -> dict: ...


class HttpSearchClient(HttpSegmentsClient):
    """El cliente de extract_segments.py más POST /tables/{t}/vector-search."""

    def vector_search(self, table: str, vector, column: str, k: int, index_type: str, metric: str,
                      ef_search: int | None = None, n_probe: int | None = None) -> dict:
        packed = np.asarray(vector, dtype="<f4").reshape(-1)
        payload = {
            "column": column,
            "vector_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
            "k": int(k),
            "index_type": index_type,
            "metric": metric,
        }
        if ef_search is not None:
            payload["ef_search"] = int(ef_search)
        if n_probe is not None:
            payload["n_probe"] = int(n_probe)
        return self._request("POST", f"/tables/{table}/vector-search", payload)


class SegmentSearcher:
    def __init__(self, client: SearchClient, extractor: SegmentMFCCExtractor,
                 settings: SearchSettings | None = None) -> None:
        if not extractor.is_fitted:
            raise ValueError("El codebook no está entrenado; corre antes extract_segments.py")
        self.client = client
        self.extractor = extractor
        self.settings = settings or SearchSettings()
        self._songs: dict[int, dict] | None = None

    @classmethod
    def from_settings(cls, settings: SearchSettings | None = None) -> SegmentSearcher:
        settings = settings or SearchSettings()
        if not settings.codebook_path.exists():
            raise FileNotFoundError(f"No existe el codebook {settings.codebook_path}; corre antes extract_segments.py")
        return cls(HttpSearchClient(settings.dbms_api_url), SegmentMFCCExtractor.load(settings.codebook_path), settings)

    # ---------------------------------------------------------------- metadatos

    def songs(self) -> dict[int, dict]:
        """song_id -> título, artista, licencia y URL. Se lee una vez y queda en memoria."""
        if self._songs is None:
            artists = {
                int(artist_id): name
                for artist_id, name in self.client.query(f"SELECT artist_id, name FROM {ARTISTS_TABLE};")["rows"]
            }
            rows = self.client.query(
                f"SELECT song_id, title, artist_id, license_family, track_url FROM {SONGS_TABLE};")["rows"]
            self._songs = {
                int(song_id): {"title": title, "artist": artists.get(int(artist_id), ""),
                               "license_family": license_family, "track_url": track_url}
                for song_id, title, artist_id, license_family, track_url in rows
            }
        return self._songs

    def refresh(self) -> None:
        """Vuelve a leer los metadatos, por ejemplo después de cargar canciones nuevas."""
        self._songs = None

    # ---------------------------------------------------------------- búsqueda

    def _resolve(self, k: int | None, index_type: str | None, metric: str | None) -> tuple[int, str, str]:
        k = self.settings.k if k is None else k
        index_type = (index_type or self.settings.index_type).lower()
        metric = (metric or self.settings.metric).lower()
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")
        if index_type not in INDEX_TYPES:
            raise ValueError(f"Índice desconocido: {index_type!r} (usa {' o '.join(INDEX_TYPES)})")
        if metric not in METRICS:
            raise ValueError(f"Métrica desconocida: {metric!r} (usa {', '.join(METRICS)})")
        return k, index_type, metric

    def _candidate(self, result: dict) -> Candidate:
        row = result["row"]
        song = self.songs().get(int(row["song_id"]), {})
        return Candidate(
            song_id=int(row["song_id"]),
            title=song.get("title", ""),
            artist=song.get("artist", ""),
            license_family=song.get("license_family", ""),
            track_url=song.get("track_url"),
            segment_id=int(row["segment_id"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            score=float(result["score"]),
            distance=float(result["distance"]),
        )

    def search_vector(self, vector, k: int | None = None, index_type: str | None = None,
                      metric: str | None = None) -> tuple[list[Candidate], bool, float]:
        """Los k segmentos más parecidos a un vector, si la API usó el índice y su tiempo en ms."""
        k, index_type, metric = self._resolve(k, index_type, metric)
        response = self.client.vector_search(
            SEGMENTS_TABLE, vector, column=EMBEDDING_COLUMN, k=k, index_type=index_type, metric=metric,
            ef_search=self.settings.ef_search if index_type == "hnsw" else None,
            n_probe=self.settings.n_probe if index_type == "ivf" else None,
        )
        candidates = [self._candidate(result) for result in response.get("results", [])]
        return candidates, bool(response.get("used_index", False)), float(response.get("search_ms", 0.0))

    def search_audio(self, source: str | Path | np.ndarray, k: int | None = None,
                     index_type: str | None = None, metric: str | None = None) -> dict:
        """Divide el audio en segmentos y busca los k candidatos de cada uno."""
        k, index_type, metric = self._resolve(k, index_type, metric)

        started = time.perf_counter()
        song = self.extractor.song_frames(source)
        segments = self.extractor.transform(song)
        extraction_ms = (time.perf_counter() - started) * 1000.0

        matches, used_index, dbms_ms = [], True, 0.0
        for segment in segments:
            candidates, used, search_ms = self.search_vector(segment.vector, k, index_type, metric)
            used_index = used_index and used
            dbms_ms += search_ms
            matches.append(SegmentMatches(segment.segment_index, segment.start_s, segment.end_s, candidates))

        return {
            "duration_s": round(song.duration_s, 3),
            "model_name": self.extractor.model_name,
            "index_type": index_type,
            "metric": metric,
            "k": k,
            "used_index": used_index,
            "segments": [asdict(match) for match in matches],
            "timings_ms": {
                "extraction_ms": round(extraction_ms, 3),
                "dbms_search_ms": round(dbms_ms, 3),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("audio", type=Path, help="audio a buscar (MP3, WAV, ...)")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--index", choices=INDEX_TYPES, default=None)
    parser.add_argument("--metric", choices=METRICS, default=None)
    args = parser.parse_args()
    result = SegmentSearcher.from_settings().search_audio(args.audio, args.k, args.index, args.metric)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
