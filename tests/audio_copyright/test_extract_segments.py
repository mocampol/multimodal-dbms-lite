"""
Extracción y carga de segmentos MFCC (apps/audio-copyright/features/extract_segments.py).
El cliente ejecuta el SQL y las inserciones en un catálogo temporal, como la API.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FEATURES_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright" / "features"
DATA_DIR = FEATURES_DIR.parent / "dataset"
sys.path.insert(0, str(FEATURES_DIR))
sys.path.insert(0, str(DATA_DIR))

import extract_segments  # noqa: E402
from catalog.catalog import Catalog  # noqa: E402
from catalog.table_metadata import StorageType  # noqa: E402
from common.record import Record  # noqa: E402
from common.value import DataType, Value  # noqa: E402
from extract_segments import CODEBOOK_NAME, SEGMENTS_TABLE, extract, segment_row  # noqa: E402
from fma_corpus import Song, write_manifest  # noqa: E402
from loader.csv_loader import _insert_one_record, _transaction_manager  # noqa: E402
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory  # noqa: E402
from query.parser.ast_nodes import SelectStm  # noqa: E402
from query.query_engine import execute, execute_many  # noqa: E402
from segment_extractor import SAMPLE_RATE, SegmentMFCCExtractor, SegmentVector, segment_intervals  # noqa: E402


class LocalClient:
    """Como la API: POST /query para el SQL y POST /tables/{t}/rows en una transacción."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.statements = []
        self.batches = []

    def table_names(self):
        return set(self.catalog.tables)

    def query(self, sql):
        self.statements.append(sql)
        statement, result = execute_many(sql, self.catalog)[-1]
        if isinstance(statement, SelectStm):
            return {"rows": [[value.data for value in record.values] for record in result]}
        return {}

    def insert_rows(self, table, rows):
        schema = self.catalog.get_schema(table)
        manager = _transaction_manager(self.catalog)
        manager.begin()
        try:
            for row in rows:
                values = [
                    Value(column.data_type,
                          base64.b64decode(row[column.name]) if column.data_type == DataType.BYTEA else row[column.name])
                    for column in schema.columns
                ]
                _insert_one_record(self.catalog, manager, table, Record(values))
            manager.commit()
        except BaseException:
            manager.abort()
            raise
        self.batches.append(len(rows))
        return {"inserted": len(rows)}

    def get_table(self, table):
        return {"indexes": self.catalog.get_indexes(table)}

    def create_index(self, table, column, index_type, index_name=None, dim=None, index_params=None):
        return self.catalog.create_index(table, column, index_type, index_name=index_name,
                                         dim=dim, index_params=index_params)


@pytest.fixture
def client(tmp_path):
    directory = str(tmp_path / "db")
    heap = make_heap_factory(directory)
    return LocalClient(Catalog(
        heap_factory=heap,
        storage_factories={StorageType.HEAP: heap, StorageType.SEQUENTIAL: make_sequential_factory(directory)},
        index_buffer_factory=make_index_buffer_factory(directory),
    ))


def melody(frequencies, seconds_each=4.0):
    t = np.arange(int(seconds_each * SAMPLE_RATE)) / SAMPLE_RATE
    return np.concatenate([0.5 * np.sin(2 * np.pi * f * t) for f in frequencies]).astype(np.float32)


def song(song_id):
    return Song(
        song_id=song_id, title=f"Canción {song_id}", artist_id=1, artist_name="Artista", album_title="Álbum",
        genre="Rock", duration_s=12.0, license_title="Attribution", license_family="CC BY",
        license_url="http://creativecommons.org/licenses/by/4.0/", track_url="", artist_url="",
        source="fma_small", relative_path=f"{song_id:06d}.wav", sha256="0" * 64, size_bytes=1,
    )


FREQUENCIES = {2: [220, 440, 880], 3: [330, 660, 990], 5: [150, 300, 600], 7: [250, 500, 750]}


def write_corpus(directory, song_ids):
    directory.mkdir(parents=True, exist_ok=True)
    for song_id in song_ids:
        sf.write(str(directory / f"{song_id:06d}.wav"), melody(FREQUENCIES[song_id]), SAMPLE_RATE)
    write_manifest([song(song_id) for song_id in song_ids], directory / "songs.csv")


@pytest.fixture
def corpus(tmp_path):
    directory = tmp_path / "audio" / "fma"
    write_corpus(directory, [2, 3, 5])
    return directory


def rows(client, sql):
    return [[value.data for value in record.values] for record in execute(sql, client.catalog)]


def test_the_first_run_fits_the_codebook_and_stores_every_segment(client, corpus):
    result = extract(client, corpus, k=16)

    assert result["codebook"] == "fitted" and result["failed"] == {}
    assert result["inserted_songs"] == 3 and result["inserted_segments"] == 12
    assert (corpus / CODEBOOK_NAME).exists()
    stored = rows(client, f"SELECT segment_id, segment_index, start_s, end_s, embedding_dim, model_name "
                          f"FROM {SEGMENTS_TABLE} WHERE song_id = 2;")
    assert sorted(row[0] for row in stored) == [2000, 2001, 2002, 2003]
    assert sorted((row[2], row[3]) for row in stored) == segment_intervals(12.0)
    assert {(row[4], row[5]) for row in stored} == {(16, "mfcc20_boaw16_w5_h2.5")}


def test_each_embedding_is_the_vector_of_its_interval(client, corpus):
    extract(client, corpus, k=16)
    extractor = SegmentMFCCExtractor.load(corpus / CODEBOOK_NAME)
    expected = extractor.transform(extractor.song_frames(corpus / "000003.wav"))[1]

    [[raw, start, end]] = rows(client, f"SELECT embedding, start_s, end_s FROM {SEGMENTS_TABLE} WHERE segment_id = 3001;")

    np.testing.assert_allclose(np.frombuffer(raw, dtype="<f4"), expected.vector, rtol=1e-6)
    assert (start, end) == (expected.start_s, expected.end_s) == (2.5, 7.5)


def test_a_second_run_reuses_the_codebook_and_changes_nothing(client, corpus):
    extract(client, corpus, k=16)
    client.statements.clear()

    result = extract(client, corpus, k=16)

    assert result["codebook"] == "loaded"
    assert result["inserted_segments"] == result["deleted_songs"] == 0
    assert client.batches == [12]
    assert not any(statement.startswith(("CREATE", "DROP", "DELETE")) for statement in client.statements)


def test_removed_songs_are_deleted_and_new_ones_extracted_with_the_same_codebook(client, corpus):
    extract(client, corpus, k=16)
    write_corpus(corpus, [2, 3, 7])

    result = extract(client, corpus, k=16)

    assert result["codebook"] == "loaded"
    assert result["deleted_songs"] == 1 and result["inserted_songs"] == 1
    assert {row[0] for row in rows(client, f"SELECT song_id FROM {SEGMENTS_TABLE};")} == {2, 3, 7}


def test_a_new_configuration_refits_and_recomputes_every_vector(client, corpus):
    extract(client, corpus, k=16)

    result = extract(client, corpus, k=16, window_s=4.0, hop_s=2.0)

    assert result["codebook"] == "fitted"
    stored = rows(client, f"SELECT model_name FROM {SEGMENTS_TABLE};")
    assert len(stored) == 3 * len(segment_intervals(12.0, 4.0, 2.0))
    assert {row[0] for row in stored} == {"mfcc20_boaw16_w4_h2"}


def test_each_insert_carries_only_complete_songs(client, corpus, monkeypatch):
    monkeypatch.setattr(extract_segments, "INSERT_BATCH", 5)

    extract(client, corpus, k=16)

    assert client.batches == [8, 4]  # 4 segmentos por canción: nunca una canción partida


def test_an_unreadable_song_is_reported_and_the_rest_are_stored(client, corpus):
    (corpus / "000005.wav").write_bytes(b"esto no es audio")

    result = extract(client, corpus, k=16)

    assert set(result["failed"]) == {5} and "No se pudo cargar" in result["failed"][5]
    assert {row[0] for row in rows(client, f"SELECT song_id FROM {SEGMENTS_TABLE};")} == {2, 3}


def test_an_empty_corpus_is_an_error(client, tmp_path):
    with pytest.raises(RuntimeError, match="corre antes prepare_corpus.py"):
        extract(client, tmp_path / "vacio")


def test_segment_ids_are_reserved_for_up_to_1000_segments_per_song():
    segment = SegmentVector(1000, 0.0, 5.0, np.ones(4, dtype=np.float32) / 2)

    with pytest.raises(ValueError, match="más de 1000 segmentos"):
        segment_row(7, segment, "modelo")
