"""
apps/audio-copyright/dataset/prepare_corpus.py — Descarga los clips de referencia de
fma_small en el corpus y escribe su manifiesto (songs.csv) y su resumen.

Es idempotente: si el manifiesto ya tiene la cuota de cada género y todos los
archivos coinciden con su SHA-256, termina sin descargar nada. Si falta algo, completa
solo lo que falta, y borra del corpus las pistas que ya no corresponden (por ejemplo,
las que se agregaron a excluded_tracks.txt).

Uso:
    python apps/audio-copyright/dataset/prepare_corpus.py                      # 1,000 clips
    python apps/audio-copyright/dataset/prepare_corpus.py --per-genre 2        # prueba: 16 clips
    python apps/audio-copyright/dataset/prepare_corpus.py --corpus-dir /app/data/audio/fma
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fma_corpus import (  # noqa: E402
    EXCLUDED_FILE,
    MANIFEST_NAME,
    METADATA_BUFFER,
    METADATA_URL,
    PER_GENRE,
    SMALL_URL,
    SUMMARY_NAME,
    Song,
    Track,
    audio_duration,
    candidates_by_genre,
    fetch_song,
    file_name,
    is_intact,
    open_zip,
    read_excluded,
    read_manifest,
    read_small_tracks,
    summarize,
    write_manifest,
)

FMA_SMALL_GENRES = ("Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock")
DEFAULT_CORPUS_DIR = Path(os.getenv("AUDIO_CORPUS_DIR", "data/audio/fma"))
DEFAULT_WORKERS = 16


def is_complete(songs: dict[int, Song], corpus_dir: Path, per_genre: int, excluded: set[int]) -> bool:
    """La cuota exacta de cada género, nada excluido y cada archivo intacto."""
    counts = {genre: 0 for genre in FMA_SMALL_GENRES}
    for song in songs.values():
        if song.song_id in excluded or song.genre not in counts:
            return False
        counts[song.genre] += 1
    return (all(count == per_genre for count in counts.values())
            and all(is_intact(song, corpus_dir) for song in songs.values()))


class Downloader:
    """Descarga clips en paralelo. Cada hilo abre su propio zip: zipfile serializa las
    lecturas de un mismo archivo, así que compartirlo no permitiría paralelizar."""

    def __init__(self, audio_source: str, corpus_dir: Path, workers: int,
                 duration_of: Callable[[Path], float]) -> None:
        self.audio_source = audio_source
        self.corpus_dir = corpus_dir
        self.workers = workers
        self.duration_of = duration_of
        self.local = threading.local()
        self.readers = []
        self.lock = threading.Lock()

    def _archive(self):
        if not hasattr(self.local, "archive"):
            archive, reader = open_zip(self.audio_source)
            self.local.archive = archive
            with self.lock:
                self.readers.append(reader)
        return self.local.archive

    def fetch(self, tracks: list[Track]) -> list[tuple[Track, Song | None, str | None]]:
        def one(track):
            song, failure = fetch_song(self._archive(), track, self.corpus_dir, self.duration_of)
            return track, song, failure

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, tracks))

    @property
    def downloaded_bytes(self) -> int:
        return sum(reader.downloaded for reader in self.readers if reader is not None)


def prepare(corpus_dir: Path = DEFAULT_CORPUS_DIR, per_genre: int = PER_GENRE, workers: int = DEFAULT_WORKERS,
            metadata_source: str = METADATA_URL, audio_source: str = SMALL_URL,
            excluded_path: Path = EXCLUDED_FILE,
            duration_of: Callable[[Path], float] = audio_duration) -> dict:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / MANIFEST_NAME
    existing = read_manifest(manifest_path)
    excluded = read_excluded(excluded_path)

    if existing and is_complete(existing, corpus_dir, per_genre, excluded):
        print(f"El corpus ya está completo: {len(existing)} canciones en {corpus_dir}", flush=True)
        return _finish(existing.values(), corpus_dir, downloaded=0, rejected={})

    print("Leyendo los metadatos de FMA...", flush=True)
    metadata, _ = open_zip(metadata_source, METADATA_BUFFER)
    candidates = candidates_by_genre(read_small_tracks(metadata), excluded)
    missing = [genre for genre in FMA_SMALL_GENRES if len(candidates.get(genre, [])) < per_genre]
    if missing:
        raise RuntimeError(f"No hay {per_genre} pistas con licencia aceptada en: {', '.join(missing)}")

    chosen: dict[str, list[Song]] = {genre: [] for genre in FMA_SMALL_GENRES}
    next_candidate = {genre: 0 for genre in FMA_SMALL_GENRES}
    rejected: dict[int, str] = {}
    downloader = Downloader(audio_source, corpus_dir, workers, duration_of)
    started = time.perf_counter()

    while True:
        # Por género, los siguientes candidatos hasta completar la cuota.
        batch = []
        for genre in FMA_SMALL_GENRES:
            needed = per_genre - len(chosen[genre])
            while needed > 0 and next_candidate[genre] < len(candidates[genre]):
                track = candidates[genre][next_candidate[genre]]
                next_candidate[genre] += 1
                song = existing.get(track.song_id)
                if song and is_intact(song, corpus_dir):
                    chosen[genre].append(song)
                else:
                    batch.append(track)
                needed -= 1
        if not batch:
            break
        for track, song, failure in downloader.fetch(batch):
            if song:
                chosen[track.genre].append(song)
            else:
                rejected[track.song_id] = failure
        done = sum(len(songs) for songs in chosen.values())
        print(f"  {done} de {per_genre * len(FMA_SMALL_GENRES)} canciones, "
              f"{downloader.downloaded_bytes / 2**20:.0f} MB descargados, "
              f"{time.perf_counter() - started:.0f} s", flush=True)

    short = [genre for genre, songs in chosen.items() if len(songs) < per_genre]
    if short:
        raise RuntimeError(f"Se acabaron los candidatos válidos en: {', '.join(short)}")

    songs = [song for genre_songs in chosen.values() for song in genre_songs]
    write_manifest(songs, manifest_path)
    _remove_leftovers(corpus_dir, {song.relative_path for song in songs})
    return _finish(songs, corpus_dir, downloader.downloaded_bytes, rejected)


def _remove_leftovers(corpus_dir: Path, keep: set[str]) -> None:
    """Borra los clips que ya no están en el manifiesto y las descargas a medias."""
    for path in corpus_dir.iterdir():
        if path.suffix in (".mp3", ".part") and path.name not in keep:
            path.unlink()


def _finish(songs, corpus_dir: Path, downloaded: int, rejected: dict[int, str]) -> dict:
    summary = summarize(songs)
    (corpus_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if rejected:
        print(f"Rechazadas {len(rejected)}: " + ", ".join(f"{file_name(i)} ({why})" for i, why in sorted(rejected.items())))
    print(f"Corpus: {summary['songs']} canciones de {summary['artists']} artistas, "
          f"{summary['total_bytes'] / 2**20:.0f} MB en {corpus_dir} "
          f"({downloaded / 2**20:.0f} MB descargados en esta corrida)", flush=True)
    return {**summary, "downloaded_bytes": downloaded, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                        help="carpeta del corpus (por defecto $AUDIO_CORPUS_DIR o data/audio/fma)")
    parser.add_argument("--per-genre", type=int, default=PER_GENRE, help="clips por cada uno de los 8 géneros")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="descargas en paralelo")
    parser.add_argument("--summary-out", type=Path, default=None, help="copia el resumen aquí (p. ej., para versionarlo)")
    args = parser.parse_args()

    result = prepare(args.corpus_dir, args.per_genre, args.workers)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text((args.corpus_dir / SUMMARY_NAME).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Resumen: {args.summary_out}")
    return result


if __name__ == "__main__":
    main()
