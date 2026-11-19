"""
apps/audio-copyright/dataset/load_songs.py — Carga las canciones y artistas del
manifiesto (songs.csv) en el DBMS, a través de su API REST.

Pasa por la API y no abre el catálogo directamente porque la API puede estar
corriendo y escribiendo en los mismos archivos de data/. Es idempotente: crea las
tablas si faltan, inserta solo las filas nuevas y borra las que ya no están en el
manifiesto (por ejemplo, canciones excluidas).

Uso:
    python apps/audio-copyright/dataset/load_songs.py --dbms-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fma_corpus import MANIFEST_NAME, Song, read_manifest  # noqa: E402
from reference_songs import reference_songs, song_path  # noqa: E402

ARTISTS_TABLE = "audio_artists"
SONGS_TABLE = "audio_songs"
INSERT_BATCH = 100

# Columnas en el orden de la tabla. Los VARCHAR tienen margen sobre lo más largo de
# fma_small con licencia aceptada (título: 152; track_url: 224; artista: 54).
ARTIST_COLUMNS = (
    ("artist_id", "INTEGER PRIMARY KEY"),
    ("name", "VARCHAR(128) NOT NULL"),
    ("artist_url", "VARCHAR(256)"),
)
SONG_COLUMNS = (
    ("song_id", "INTEGER PRIMARY KEY"),
    ("title", "VARCHAR(256) NOT NULL"),
    ("artist_id", "INTEGER NOT NULL"),
    ("album_title", "VARCHAR(256)"),
    ("genre", "VARCHAR(32) NOT NULL"),
    ("duration_s", "DOUBLE PRECISION NOT NULL"),
    ("license_family", "VARCHAR(16) NOT NULL"),
    ("license_title", "VARCHAR(128) NOT NULL"),
    ("license_url", "VARCHAR(128) NOT NULL"),
    ("track_url", "VARCHAR(512)"),
    ("source", "VARCHAR(16) NOT NULL"),
    ("path", "VARCHAR(256) NOT NULL"),       # ruta absoluta donde la API abre el archivo
    ("sha256", "VARCHAR(64) NOT NULL"),
)


class DBMSClient(Protocol):
    def table_names(self) -> set[str]: ...
    def query(self, sql: str) -> dict: ...


class HttpDBMSClient:
    """Lo mínimo de la API REST del DBMS, con la biblioteca estándar."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} respondió {error.code}: {detail[:500]}") from error

    def table_names(self) -> set[str]:
        return {table["name"] for table in self._request("GET", "/tables")}

    def query(self, sql: str) -> dict:
        return self._request("POST", "/query", {"sql": sql})


# =============================================================================
# SQL
# =============================================================================

def sql_literal(value) -> str:
    if value is None or value == "":
        return "NULL"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _check_lengths(columns, values: tuple, label: str) -> None:
    """Un VARCHAR demasiado largo haría fallar todo el lote; mejor decir cuál es."""
    for (name, definition), value in zip(columns, values):
        if definition.startswith("VARCHAR(") and isinstance(value, str):
            limit = int(definition[len("VARCHAR("):definition.index(")")])
            if len(value) > limit:
                raise ValueError(f"{label}: {name} tiene {len(value)} caracteres y la columna admite {limit}")


def create_table_sql(table: str, columns) -> str:
    return f"CREATE TABLE {table} ({', '.join(f'{name} {definition}' for name, definition in columns)});"


def insert_statements(table: str, rows: list[tuple], batch: int = INSERT_BATCH) -> list[str]:
    return [
        f"INSERT INTO {table} VALUES "
        + ", ".join("(" + ", ".join(sql_literal(value) for value in row) + ")" for row in rows[start:start + batch])
        + ";"
        for start in range(0, len(rows), batch)
    ]


def artist_rows(songs: list[Song]) -> dict[int, tuple]:
    rows = {}
    for song in songs:
        rows.setdefault(song.artist_id, (song.artist_id, song.artist_name, song.artist_url))
    return rows


def song_row(song: Song, corpus_dir: Path) -> tuple:
    """La fila de audio_songs. La ruta depende de si es de FMA o una canción agregada."""
    return (
        song.song_id, song.title, song.artist_id, song.album_title, song.genre, song.duration_s,
        song.license_family, song.license_title, song.license_url, song.track_url, song.source,
        song_path(corpus_dir, song).as_posix(), song.sha256,
    )


# =============================================================================
# Carga
# =============================================================================

def _ids(client: DBMSClient, table: str, column: str) -> set[int]:
    return {int(row[0]) for row in client.query(f"SELECT {column} FROM {table};")["rows"]}


def _sync(client: DBMSClient, table: str, key: str, columns, rows: dict[int, tuple]) -> dict:
    """Inserta las filas que faltan y borra las que sobran. No actualiza las que ya están:
    una canción no cambia mientras siga en el manifiesto con el mismo song_id."""
    present = _ids(client, table, key)
    new = [rows[row_id] for row_id in sorted(set(rows) - present)]
    for row in new:
        _check_lengths(columns, row, f"{table} {row[0]}")
    for statement in insert_statements(table, new):
        client.query(statement)
    stale = sorted(present - set(rows))
    for row_id in stale:
        client.query(f"DELETE FROM {table} WHERE {key} = {row_id};")
    return {"inserted": len(new), "deleted": len(stale), "total": len(rows)}


def load(client: DBMSClient, corpus_dir: Path) -> dict:
    """Sincroniza las tablas con las canciones de FMA y las agregadas desde la aplicación:
    si se leyera solo songs.csv, esta carga borraría las agregadas."""
    if not read_manifest(corpus_dir / MANIFEST_NAME):
        raise RuntimeError(f"No hay canciones en {corpus_dir / MANIFEST_NAME}; corre antes prepare_corpus.py")
    corpus_dir = corpus_dir.resolve()
    songs = list(reference_songs(corpus_dir).values())

    existing = client.table_names()
    for table, columns in ((ARTISTS_TABLE, ARTIST_COLUMNS), (SONGS_TABLE, SONG_COLUMNS)):
        if table not in existing:
            client.query(create_table_sql(table, columns))

    # El motor no tiene claves foráneas: el orden solo importa para que, si algo falla
    # a mitad, cada canción insertada ya tenga su artista.
    artists = _sync(client, ARTISTS_TABLE, "artist_id", ARTIST_COLUMNS, artist_rows(songs))
    result = {
        "artists": artists,
        "songs": _sync(client, SONGS_TABLE, "song_id", SONG_COLUMNS,
                       {song.song_id: song_row(song, corpus_dir) for song in songs}),
    }
    print(f"DBMS: {SONGS_TABLE} {result['songs']}, {ARTISTS_TABLE} {result['artists']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dbms-url", default=os.getenv("DBMS_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--corpus-dir", type=Path, default=Path(os.getenv("AUDIO_CORPUS_DIR", "data/audio/fma")))
    args = parser.parse_args()
    load(HttpDBMSClient(args.dbms_url), args.corpus_dir)


if __name__ == "__main__":
    main()
