"""
apps/audio-copyright/dataset/reference_songs.py — Las canciones de referencia: las de
fma_small (prepare_corpus.py) y las que se agregan desde la aplicación.

Las agregadas viven en su propia carpeta (data/audio/uploads/, junto a data/audio/fma/) y
en su propio manifiesto (uploads.csv), con el mismo formato que songs.csv. Así
prepare_corpus.py, que borra de la carpeta de FMA lo que no está en su manifiesto, no
las toca; y load_songs.py y extract_segments.py, que leen las dos listas, no las borran.

Sus song_id empiezan en UPLOAD_ID_START, lejos de los track_id de FMA (hasta 155066), y
caben en el segment_id (song_id * 1000 + segmento) de un INTEGER de 32 bits.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

from fma_corpus import ACCEPTED_LICENSES, MANIFEST_NAME, Song, read_manifest, write_manifest

UPLOAD_SOURCE = "upload"
UPLOADS_DIRNAME = "uploads"
UPLOADS_MANIFEST = "uploads.csv"
UPLOAD_ID_START = 1_000_001
UPLOAD_LICENSES = ("Propia", *ACCEPTED_LICENSES)   # "Propia": quien la sube es el autor
# audio_songs.license_url es obligatoria: la URL oficial de cada licencia, y un texto si no tiene.
LICENSE_URLS = {
    "Propia": "sin URL: la subió su autor",
    "CC BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-SA": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "Public Domain": "https://creativecommons.org/publicdomain/mark/1.0/",
}
DEFAULT_GENRE = "Sin género"


def uploads_dir(corpus_dir: Path) -> Path:
    """data/audio/uploads, al lado de data/audio/fma."""
    return Path(corpus_dir).parent / UPLOADS_DIRNAME


def read_uploads(corpus_dir: Path) -> dict[int, Song]:
    return read_manifest(uploads_dir(corpus_dir) / UPLOADS_MANIFEST)


def reference_songs(corpus_dir: Path) -> dict[int, Song]:
    """Las canciones de FMA y las agregadas, por song_id."""
    songs = read_manifest(Path(corpus_dir) / MANIFEST_NAME)
    uploads = read_uploads(corpus_dir)
    repeated = set(songs) & set(uploads)
    if repeated:
        raise RuntimeError(f"song_id repetidos entre el corpus y las canciones agregadas: {sorted(repeated)}")
    return {**songs, **uploads}


def song_path(corpus_dir: Path, song: Song) -> Path:
    """La ruta absoluta del archivo de una canción, según de dónde viene."""
    base = uploads_dir(corpus_dir) if song.source == UPLOAD_SOURCE else Path(corpus_dir)
    return (base / song.relative_path).resolve()


def artist_id_for(name: str, songs: dict[int, Song]) -> int:
    """El artist_id de un artista que ya está en la base (mismo nombre, sin distinguir
    mayúsculas), o uno nuevo a partir de UPLOAD_ID_START."""
    wanted = name.strip().casefold()
    for song in songs.values():
        if song.artist_name.strip().casefold() == wanted:
            return song.artist_id
    taken = [song.artist_id for song in songs.values() if song.artist_id >= UPLOAD_ID_START]
    return max(taken, default=UPLOAD_ID_START - 1) + 1


class UploadStore:
    """Guarda los audios agregados y su manifiesto. Un lock evita que dos subidas a la vez
    tomen el mismo song_id o se pisen el manifiesto."""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.directory = uploads_dir(corpus_dir)
        self.manifest = self.directory / UPLOADS_MANIFEST
        self.lock = threading.Lock()

    def list(self) -> list[Song]:
        return sorted(read_uploads(self.corpus_dir).values(), key=lambda song: song.song_id)

    def next_song_id(self) -> int:
        return max(read_uploads(self.corpus_dir), default=UPLOAD_ID_START - 1) + 1

    def new_song(self, data: bytes, extension: str, title: str, artist: str, duration_s: float,
                 license_family: str, album: str = "", genre: str = "", source_url: str = "") -> Song:
        """Escribe el archivo y devuelve su Song, sin agregarlo todavía al manifiesto:
        eso se hace con commit() cuando ya está en el DBMS. Hay que llamarlo con el lock."""
        if license_family not in UPLOAD_LICENSES:
            raise ValueError(f"Licencia no admitida: {license_family!r} (usa {', '.join(UPLOAD_LICENSES)})")
        song_id = self.next_song_id()
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{song_id}{extension.lower()}"
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        return Song(
            song_id=song_id,
            title=title.strip(),
            artist_id=artist_id_for(artist, reference_songs(self.corpus_dir)),
            artist_name=artist.strip(),
            album_title=album.strip(),
            genre=genre.strip() or DEFAULT_GENRE,
            duration_s=round(duration_s, 3),
            license_title=license_family,
            license_url=LICENSE_URLS[license_family],
            license_family=license_family,
            track_url=source_url.strip(),
            artist_url="",
            source=UPLOAD_SOURCE,
            relative_path=target.name,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def commit(self, song: Song) -> None:
        """Agrega la canción al manifiesto. Hay que llamarlo con el lock."""
        write_manifest([*read_uploads(self.corpus_dir).values(), song], self.manifest)

    def discard(self, song: Song) -> None:
        """Borra el archivo de una canción que no llegó a guardarse."""
        (self.directory / song.relative_path).unlink(missing_ok=True)

