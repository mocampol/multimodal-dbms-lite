"""
Carga de canciones y artistas en el DBMS (apps/audio-copyright/dataset/load_songs.py).
El cliente ejecuta el SQL en un catálogo temporal, como lo haría POST /query.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright" / "dataset"
sys.path.insert(0, str(DATA_DIR))

from catalog.catalog import Catalog  # noqa: E402
from catalog.table_metadata import StorageType  # noqa: E402
from fma_corpus import Song, write_manifest  # noqa: E402
from load_songs import ARTISTS_TABLE, SONGS_TABLE, load, sql_literal  # noqa: E402
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory  # noqa: E402
from query.parser.ast_nodes import SelectStm  # noqa: E402
from query.query_engine import execute, execute_many  # noqa: E402


class LocalClient:
    def __init__(self, catalog):
        self.catalog = catalog
        self.statements = []

    def table_names(self):
        return set(self.catalog.tables)

    def query(self, sql):
        """Como POST /query: solo un SELECT devuelve filas."""
        self.statements.append(sql)
        statement, result = execute_many(sql, self.catalog)[-1]
        if isinstance(statement, SelectStm):
            return {"rows": [[value.data for value in record.values] for record in result]}
        return {}


@pytest.fixture
def client(tmp_path):
    directory = str(tmp_path / "db")
    heap = make_heap_factory(directory)
    return LocalClient(Catalog(
        heap_factory=heap,
        storage_factories={StorageType.HEAP: heap, StorageType.SEQUENTIAL: make_sequential_factory(directory)},
        index_buffer_factory=make_index_buffer_factory(directory),
    ))


def song(song_id, artist_id=1, **changes):
    values = dict(
        song_id=song_id, title=f"Canción {song_id}", artist_id=artist_id, artist_name=f"Artista {artist_id}",
        album_title="Álbum", genre="Rock", duration_s=29.98, license_title="Attribution-ShareAlike",
        license_url="http://creativecommons.org/licenses/by-sa/3.0/", license_family="CC BY-SA",
        track_url=f"http://freemusicarchive.org/music/{song_id}", artist_url=f"http://freemusicarchive.org/a/{artist_id}",
        source="fma_small", relative_path=f"{song_id:06d}.mp3", sha256="0" * 64, size_bytes=1000,
    )
    values.update(changes)
    return Song(**values)


@pytest.fixture
def corpus(tmp_path):
    directory = tmp_path / "audio" / "fma"
    write_manifest([song(2, title="Don't Stop"), song(3), song(5, artist_id=7, album_title="")], directory / "songs.csv")
    return directory


def rows(client, sql):
    return [[value.data for value in record.values] for record in execute(sql, client.catalog)]


def test_sql_literals_escape_quotes_and_use_null_for_missing_values():
    assert sql_literal("Don't") == "'Don''t'"
    assert sql_literal("") == "NULL" and sql_literal(None) == "NULL"
    assert sql_literal(29.98) == "29.98"


def test_load_creates_the_tables_with_songs_and_their_artists(client, corpus):
    result = load(client, corpus)

    assert result["songs"] == {"inserted": 3, "deleted": 0, "total": 3}
    assert result["artists"] == {"inserted": 2, "deleted": 0, "total": 2}
    songs = rows(client, f"SELECT song_id, title, artist_id, album_title, duration_s, path FROM {SONGS_TABLE};")
    by_id = {row[0]: row for row in songs}
    assert by_id[2][1] == "Don't Stop"
    assert by_id[2][4] == 29.98
    assert by_id[5][3] is None
    assert by_id[3][5] == (corpus.resolve() / "000003.mp3").as_posix()
    assert sorted(rows(client, f"SELECT artist_id, name FROM {ARTISTS_TABLE};")) == [[1, "Artista 1"], [7, "Artista 7"]]


def test_a_second_load_inserts_and_deletes_nothing(client, corpus):
    load(client, corpus)
    client.statements.clear()

    result = load(client, corpus)

    assert result["songs"]["inserted"] == result["songs"]["deleted"] == 0
    assert not any(statement.startswith(("CREATE", "INSERT", "DELETE")) for statement in client.statements)


def test_songs_removed_from_the_manifest_are_deleted_with_their_lonely_artist(client, corpus):
    load(client, corpus)
    write_manifest([song(2), song(3)], corpus / "songs.csv")

    result = load(client, corpus)

    assert result["songs"]["deleted"] == 1 and result["artists"]["deleted"] == 1
    assert sorted(row[0] for row in rows(client, f"SELECT song_id FROM {SONGS_TABLE};")) == [2, 3]
    assert rows(client, f"SELECT artist_id FROM {ARTISTS_TABLE};") == [[1]]


def test_a_value_too_long_is_reported_before_inserting(client, corpus):
    write_manifest([song(2), replace(song(3), title="x" * 300)], corpus / "songs.csv")

    with pytest.raises(ValueError, match="title tiene 300 caracteres"):
        load(client, corpus)
    assert rows(client, f"SELECT song_id FROM {SONGS_TABLE};") == []


def test_an_empty_corpus_is_an_error(client, tmp_path):
    with pytest.raises(RuntimeError, match="corre antes prepare_corpus.py"):
        load(client, tmp_path / "vacio")
