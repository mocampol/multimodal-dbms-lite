"""
apps/audio-copyright/backend/song_ingest.py — Agrega una canción a la base de referencia
desde la aplicación, procesada igual que las de FMA.

    1. El audio se divide en segmentos y se vectoriza con el codebook y el IDF ya
       entrenados (transform, nunca fit): la canción nueva usa las palabras ya
       encontradas y el BoVW no cambia.
    2. Se guarda el archivo en data/audio/uploads/ y se insertan el artista (si es nuevo),
       la canción y sus segmentos por la API del DBMS. Al recibir los segmentos, el motor
       los agrega a los índices HNSW e IVF de audio_segments sin reconstruirlos.
    3. Recién entonces se anota en uploads.csv, para que audio-data la conserve.

Si algo falla a mitad, se deshace lo que alcanzó a hacerse: no quedan archivos, filas ni
vectores sueltos.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for path in (APP_DIR.parents[1] / "src", APP_DIR / "dataset", APP_DIR / "features"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from extract_segments import segment_row  # noqa: E402
from load_songs import (  # noqa: E402
    ARTIST_COLUMNS, ARTISTS_TABLE, SONG_COLUMNS, SONGS_TABLE, _check_lengths, insert_statements, song_row,
)
from reference_songs import UploadStore  # noqa: E402
from segment_extractor import SAMPLE_RATE  # noqa: E402
from segment_index import SEGMENTS_TABLE  # noqa: E402


def add_song(searcher, store: UploadStore, signal: np.ndarray, data: bytes, extension: str,
             title: str, artist: str, license_family: str, album: str = "", genre: str = "",
             source_url: str = "") -> dict:
    """Agrega la canción y devuelve sus datos. searcher aporta el codebook cargado y el
    cliente del DBMS (query e insert_rows)."""
    if not title.strip() or not artist.strip():
        raise ValueError("El título y el artista son obligatorios")
    started = time.perf_counter()
    extractor = searcher.extractor
    segments = extractor.transform(extractor.song_frames(signal))   # rechaza audio en silencio
    extraction_ms = (time.perf_counter() - started) * 1000.0
    client = searcher.client

    with store.lock:
        song = store.new_song(data, extension, title, artist, len(signal) / SAMPLE_RATE, license_family,
                              album=album, genre=genre, source_url=source_url)
        artist_row = (song.artist_id, song.artist_name, song.artist_url)
        row = song_row(song, store.corpus_dir)
        segment_rows = [segment_row(song.song_id, segment, extractor.model_name) for segment in segments]
        done = []   # lo que ya se escribió en el DBMS, para deshacerlo si algo falla
        mark = time.perf_counter()
        try:
            _check_lengths(ARTIST_COLUMNS, artist_row, "artista")
            _check_lengths(SONG_COLUMNS, row, "canción")
            known = client.query(f"SELECT artist_id FROM {ARTISTS_TABLE} WHERE artist_id = {song.artist_id};")["rows"]
            if not known:
                client.query(insert_statements(ARTISTS_TABLE, [artist_row])[0])
                done.append(f"DELETE FROM {ARTISTS_TABLE} WHERE artist_id = {song.artist_id};")
            client.query(insert_statements(SONGS_TABLE, [row])[0])
            done.append(f"DELETE FROM {SONGS_TABLE} WHERE song_id = {song.song_id};")
            done.append(f"DELETE FROM {SEGMENTS_TABLE} WHERE song_id = {song.song_id};")
            client.insert_rows(SEGMENTS_TABLE, segment_rows)   # el motor actualiza HNSW e IVF
            store.commit(song)
        except Exception:
            for statement in reversed(done):
                try:
                    client.query(statement)
                except Exception:
                    pass   # se intenta deshacer todo; el error que importa es el original
            store.discard(song)
            raise
        dbms_ms = (time.perf_counter() - mark) * 1000.0

    searcher.refresh()   # los resultados ya pueden mostrar el título y el artista nuevos
    return {
        "song_id": song.song_id,
        "title": song.title,
        "artist": song.artist_name,
        "artist_id": song.artist_id,
        "album_title": song.album_title,
        "genre": song.genre,
        "license_family": song.license_family,
        "track_url": song.track_url,
        "duration_s": song.duration_s,
        "segments": len(segment_rows),
        "model_name": extractor.model_name,
        "timings_ms": {"extraction_ms": round(extraction_ms, 1), "dbms_ms": round(dbms_ms, 1),
                       "total_ms": round((time.perf_counter() - started) * 1000.0, 1)},
    }


def upload_summary(song) -> dict:
    """Lo que GET /songs muestra de una canción agregada."""
    return {
        "song_id": song.song_id, "title": song.title, "artist": song.artist_name,
        "album_title": song.album_title, "genre": song.genre, "license_family": song.license_family,
        "track_url": song.track_url, "duration_s": song.duration_s,
    }
