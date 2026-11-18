"""
Corpus de canciones de referencia para la detección de copyright (issue #110).

Elige clips de fma_small (Free Music Archive) con licencias que permiten obras
derivadas, los descarga sin bajar el zip entero y escribe sus metadatos. La política
que justifica cada regla está en docs/audio_data_policy.md.

Los zips de FMA se leen por rangos HTTP: el índice del zip ocupa unos pocos KB y
cada clip unos 900 KB, así que 1,000 clips son ~1 GB en lugar de los 7.2 GiB del zip.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Iterable

METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
# SHA1 publicados en el README de FMA, para quien descargue los zips completos.
METADATA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
SMALL_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"

SOURCE = "fma_small"
SEED = 42
PER_GENRE = 125                 # 8 géneros balanceados: 1,000 clips
MIN_DURATION_S = 25.0           # los clips son de 30 s; algunos de FMA vienen dañados o cortos
MANIFEST_NAME = "songs.csv"
SUMMARY_NAME = "corpus_summary.json"
EXCLUDED_FILE = Path(__file__).with_name("excluded_tracks.txt")

# Familias aceptadas (docs/audio_data_policy.md). Ninguna variante ND ni Sampling Plus.
ACCEPTED_LICENSES = ("CC BY", "CC BY-SA", "CC BY-NC", "CC BY-NC-SA", "CC0", "Public Domain")


# =============================================================================
# Licencias
# =============================================================================

_CC_LICENSE = re.compile(r"creativecommons\.org/licenses/([a-z+\-]+)/", re.IGNORECASE)


def license_family(license_url: str) -> str | None:
    """Familia de la licencia según su URL, o None si no es una que se acepte.

    Se usa la URL y no el nombre: en FMA el mismo tipo de licencia aparece escrito
    de muchas formas ("Attribution-Noncommercial-Share Alike 3.0 United States",
    "Attribution-NonCommercial-ShareAlike", ...).
    """
    url = (license_url or "").strip().lower()
    if "creativecommons.org/publicdomain/zero" in url:
        return "CC0"
    if "creativecommons.org/publicdomain/mark" in url or "creativecommons.org/licenses/publicdomain" in url:
        return "Public Domain"
    match = _CC_LICENSE.search(url)
    if not match:
        return None
    terms = match.group(1).split("-")
    if terms[0] != "by" or "nd" in terms or any("sampling" in term for term in terms):
        return None
    family = "CC " + "-".join(term.upper() for term in terms)
    return family if family in ACCEPTED_LICENSES else None


# =============================================================================
# Lectura de zips remotos por rangos HTTP
# =============================================================================

class HttpRangeFile(io.RawIOBase):
    """Archivo de solo lectura sobre una URL que acepta Range. zipfile solo necesita
    seek y read, así que puede leer el índice y algunos miembros sin bajar el resto."""

    def __init__(self, url: str, retries: int = 4, timeout: float = 60.0) -> None:
        self.url = url
        self.retries = retries
        self.timeout = timeout
        self.position = 0
        self.downloaded = 0
        request = urllib.request.Request(url, method="HEAD")
        with self._open(request) as response:
            if response.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError(f"{url} no acepta descargas por rangos")
            self.size = int(response.headers["Content-Length"])

    def _open(self, request):
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size or len(buffer) == 0:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with self._open(request) as response:
            if response.status != 206:
                raise RuntimeError(f"Se esperaba 206 Partial Content y llegó {response.status}")
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        self.downloaded += len(data)
        return len(data)


AUDIO_BUFFER = 256 * 1024       # por rango; lo que sobra al final de cada clip se descarta
METADATA_BUFFER = 8 * 1024 * 1024  # los CSV se leen enteros: conviene pedir tramos grandes


def open_zip(source: str | Path, buffer_size: int = AUDIO_BUFFER) -> tuple[zipfile.ZipFile, HttpRangeFile | None]:
    """Un zip local o remoto. Devuelve también el lector HTTP, para contar lo descargado."""
    if str(source).startswith(("http://", "https://")):
        remote = HttpRangeFile(str(source))
        return zipfile.ZipFile(io.BufferedReader(remote, buffer_size=buffer_size)), remote
    return zipfile.ZipFile(source), None


# =============================================================================
# Metadatos de FMA
# =============================================================================

@dataclass(frozen=True)
class Track:
    song_id: int
    title: str
    artist_id: int
    artist_name: str
    album_title: str
    genre: str
    license_title: str
    license_url: str
    license_family: str | None
    track_url: str
    artist_url: str


def _multiheader_rows(handle) -> Iterable[dict]:
    """tracks.csv tiene dos filas de encabezado (grupo y columna) y una con track_id."""
    reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", newline=""))
    groups, columns = next(reader), next(reader)
    next(reader)
    names = ["track_id"] + [f"{group}.{column}" for group, column in zip(groups[1:], columns[1:])]
    for row in reader:
        yield dict(zip(names, row))


def read_small_tracks(metadata: zipfile.ZipFile) -> list[Track]:
    """Las pistas de fma_small, con título y artista de tracks.csv y la licencia y
    las URLs de raw_tracks.csv."""
    small = {}
    with metadata.open("fma_metadata/tracks.csv") as handle:
        for row in _multiheader_rows(handle):
            if row["set.subset"] == "small":
                small[int(row["track_id"])] = row
    tracks = []
    with metadata.open("fma_metadata/raw_tracks.csv") as handle:
        for raw in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", newline="")):
            row = small.get(int(raw["track_id"]))
            if row is None:
                continue
            tracks.append(Track(
                song_id=int(raw["track_id"]),
                title=row["track.title"],
                artist_id=int(row["artist.id"]),
                artist_name=row["artist.name"],
                album_title=row["album.title"],
                genre=row["track.genre_top"],
                license_title=raw["license_title"].strip(),
                license_url=raw["license_url"].strip(),
                license_family=license_family(raw["license_url"]),
                track_url=raw["track_url"].strip(),
                artist_url=raw["artist_url"].strip(),
            ))
    return sorted(tracks, key=lambda track: track.song_id)


def read_excluded(path: Path = EXCLUDED_FILE) -> set[int]:
    """song_id retirados del corpus, uno por línea; '#' empieza un comentario."""
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            ids.add(int(value))
    return ids


def candidates_by_genre(tracks: list[Track], excluded: set[int] = frozenset(),
                        seed: int = SEED) -> dict[str, list[Track]]:
    """Por género, las pistas aceptables en un orden aleatorio fijo. Se toman en ese
    orden hasta completar la cuota, saltando las que no se puedan usar.

    Las excluidas se quitan después de barajar: si se quitaran antes, retirar una sola
    pista cambiaría el orden de todo su género y, con él, la mitad de sus canciones."""
    by_genre = defaultdict(list)
    for track in tracks:
        if track.license_family and track.genre:
            by_genre[track.genre].append(track)
    shuffled = {}
    for genre in sorted(by_genre):
        candidates = sorted(by_genre[genre], key=lambda track: track.song_id)
        random.Random(f"{seed}-{genre}").shuffle(candidates)
        shuffled[genre] = [track for track in candidates if track.song_id not in excluded]
    return shuffled


def member_name(song_id: int) -> str:
    """Ruta del clip dentro de fma_small.zip: fma_small/000/000002.mp3."""
    padded = f"{song_id:06d}"
    return f"fma_small/{padded[:3]}/{padded}.mp3"


def file_name(song_id: int) -> str:
    """Nombre en el corpus. Todos en una carpeta: GET /files solo sirve archivos que
    estén directamente dentro de una carpeta importada."""
    return f"{song_id:06d}.mp3"


# =============================================================================
# Manifiesto
# =============================================================================

@dataclass(frozen=True)
class Song:
    song_id: int
    title: str
    artist_id: int
    artist_name: str
    album_title: str
    genre: str
    duration_s: float
    license_title: str
    license_url: str
    license_family: str
    track_url: str
    artist_url: str
    source: str
    relative_path: str
    sha256: str
    size_bytes: int


MANIFEST_FIELDS = [field.name for field in fields(Song)]


def read_manifest(path: Path) -> dict[int, Song]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        songs = {}
        for row in csv.DictReader(handle):
            songs[int(row["song_id"])] = Song(**{
                **row,
                "song_id": int(row["song_id"]),
                "artist_id": int(row["artist_id"]),
                "duration_s": float(row["duration_s"]),
                "size_bytes": int(row["size_bytes"]),
            })
        return songs


def write_manifest(songs: Iterable[Song], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for song in sorted(songs, key=lambda song: (song.genre, song.song_id)):
            writer.writerow(asdict(song))
    os.replace(temporary, path)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_intact(song: Song, corpus_dir: Path) -> bool:
    path = corpus_dir / song.relative_path
    return path.is_file() and path.stat().st_size == song.size_bytes and sha256_of(path) == song.sha256


def summarize(songs: Iterable[Song]) -> dict:
    songs = list(songs)
    return {
        "source": SOURCE,
        "songs": len(songs),
        "artists": len({song.artist_id for song in songs}),
        "by_genre": dict(sorted(Counter(song.genre for song in songs).items())),
        "by_license": dict(sorted(Counter(song.license_family for song in songs).items())),
        "total_duration_s": round(sum(song.duration_s for song in songs), 1),
        "total_bytes": sum(song.size_bytes for song in songs),
        "metadata_url": METADATA_URL,
        "audio_url": SMALL_URL,
        "policy": "docs/audio_data_policy.md",
    }


# =============================================================================
# Descarga
# =============================================================================

SILENCE_PEAK = 1e-6  # el mismo umbral con que MFCCExtractor rechaza una señal en silencio


class SilentAudioError(ValueError):
    """El clip se decodifica, pero no tiene sonido: no se le puede extraer MFCC."""


def audio_duration(path: Path) -> float:
    """Duración del clip. Lo decodifica entero, no solo el encabezado, para rechazar los
    que están en silencio: FMA tiene alguno con duración normal y sin sonido."""
    import numpy as np
    import soundfile

    samples, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if samples.size == 0 or float(np.max(np.abs(samples))) < SILENCE_PEAK:
        raise SilentAudioError(path.name)
    return len(samples) / sample_rate


def fetch_song(archive: zipfile.ZipFile, track: Track, corpus_dir: Path,
               duration_of: Callable[[Path], float] = audio_duration) -> tuple[Song | None, str | None]:
    """Extrae un clip al corpus. Devuelve la canción, o None y el motivo del rechazo.
    zipfile comprueba el CRC al leer, así que un miembro dañado lanza BadZipFile."""
    target = corpus_dir / file_name(track.song_id)
    temporary = target.with_suffix(".part")
    try:
        with archive.open(member_name(track.song_id)) as source, open(temporary, "wb") as destination:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                destination.write(chunk)
        duration = duration_of(temporary)
    except KeyError:
        return None, "no está en el zip"
    except zipfile.BadZipFile:
        temporary.unlink(missing_ok=True)
        return None, "CRC inválido"
    except SilentAudioError:
        temporary.unlink(missing_ok=True)
        return None, "en silencio"
    except Exception as error:  # audio que soundfile no puede leer
        temporary.unlink(missing_ok=True)
        return None, f"audio ilegible ({type(error).__name__})"
    if duration < MIN_DURATION_S:
        temporary.unlink(missing_ok=True)
        return None, f"dura {duration:.1f} s"
    os.replace(temporary, target)
    return Song(
        **{name: value for name, value in asdict(track).items() if name in MANIFEST_FIELDS},
        duration_s=round(duration, 3),
        source=SOURCE,
        relative_path=target.name,
        sha256=sha256_of(target),
        size_bytes=target.stat().st_size,
    ), None
