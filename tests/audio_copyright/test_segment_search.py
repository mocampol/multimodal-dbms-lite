"""
Búsqueda de candidatos por segmento (apps/audio-copyright/backend/segment_search.py).

Arma una base real pequeña con load_songs.load y extract_segments.extract, y el cliente
de prueba responde vector-search con los índices del catálogo, como la API.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

APP_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright"
for folder in ("dataset", "features", "backend"):
    sys.path.insert(0, str(APP_DIR / folder))

from catalog.catalog import Catalog  # noqa: E402
from catalog.table_metadata import StorageType  # noqa: E402
from common.record import Record  # noqa: E402
from common.value import DataType, Value  # noqa: E402
from extract_segments import CODEBOOK_NAME, extract  # noqa: E402
from fma_corpus import Song, write_manifest  # noqa: E402
from load_songs import load  # noqa: E402
from loader.csv_loader import _insert_one_record, _transaction_manager  # noqa: E402
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory  # noqa: E402
from query.parser.ast_nodes import SelectStm  # noqa: E402
from query.query_engine import execute_many  # noqa: E402
from segment_extractor import SAMPLE_RATE, SegmentMFCCExtractor  # noqa: E402
from segment_search import Candidate, SearchSettings, SegmentSearcher, merge_candidates  # noqa: E402


class LocalClient:
    """Las interfaces de la API que usan la carga, la indexación y la búsqueda."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.statements = []
        self.searches = []

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
        for row in rows:
            values = [Value(c.data_type, base64.b64decode(row[c.name]) if c.data_type == DataType.BYTEA else row[c.name])
                      for c in schema.columns]
            _insert_one_record(self.catalog, manager, table, Record(values))
        manager.commit()
        return {"inserted": len(rows)}

    def get_table(self, table):
        return {"indexes": self.catalog.get_indexes(table)}

    def create_index(self, table, column, index_type, index_name=None, dim=None, index_params=None):
        return self.catalog.create_index(table, column, index_type, index_name=index_name,
                                         dim=dim, index_params=index_params)

    def vector_search(self, table, vector, column, k, index_type, metric, ef_search=None, n_probe=None):
        """Como POST /vector-search: busca en el índice y devuelve cada fila con su score."""
        self.searches.append({"index_type": index_type, "ef_search": ef_search, "n_probe": n_probe})
        index = self.catalog.get_physical_index(table, column, index_type=index_type)
        schema = self.catalog.get_schema(table)
        storage = self.catalog.get_storage(table)
        results = []
        for hit in index.search(np.asarray(vector, dtype=np.float32), k=k, metric=metric):
            record = storage.get(hit.rid)
            results.append({"score": hit.score, "distance": hit.distance,
                            "row": {c.name: v.data for c, v in zip(schema.columns, record.values)}})
        return {"results": results, "used_index": True, "search_ms": 0.0}


FREQUENCIES = {2: [220, 440, 880], 3: [330, 660, 990], 5: [150, 300, 600]}


def melody(frequencies, seed=0, noise=0.0, seconds_each=4.0):
    t = np.arange(int(seconds_each * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.concatenate([0.5 * np.sin(2 * np.pi * f * t) for f in frequencies])
    return (signal + np.random.default_rng(seed).normal(0.0, noise, signal.size)).astype(np.float32)


def reference(song_id):
    """El audio de una canción de la base. Lleva un poco de ruido, como el audio real: un
    seno puro deja bandas Mel casi vacías, y el logaritmo del MFCC las vuelve muy sensibles
    a cualquier ruido de la consulta."""
    return melody(FREQUENCIES[song_id], seed=song_id, noise=0.01)


def song(song_id):
    return Song(
        song_id=song_id, title=f"Canción {song_id}", artist_id=song_id * 10, artist_name=f"Artista {song_id}",
        album_title="Álbum", genre="Rock", duration_s=12.0, license_title="Attribution", license_family="CC BY",
        license_url="http://creativecommons.org/licenses/by/4.0/", track_url=f"http://fma.org/{song_id}",
        artist_url="", source="fma_small", relative_path=f"{song_id:06d}.wav", sha256="0" * 64, size_bytes=1,
    )


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """Base de referencia con 3 canciones: tablas, segmentos, codebook e índices."""
    root = tmp_path_factory.mktemp("audio-search")
    corpus = root / "audio" / "fma"
    corpus.mkdir(parents=True)
    for song_id in FREQUENCIES:
        sf.write(str(corpus / f"{song_id:06d}.wav"), reference(song_id), SAMPLE_RATE)
    write_manifest([song(song_id) for song_id in FREQUENCIES], corpus / "songs.csv")

    heap = make_heap_factory(str(root / "db"))
    client = LocalClient(Catalog(
        heap_factory=heap,
        storage_factories={StorageType.HEAP: heap, StorageType.SEQUENTIAL: make_sequential_factory(str(root / "db"))},
        index_buffer_factory=make_index_buffer_factory(str(root / "db")),
    ))
    load(client, corpus)
    assert extract(client, corpus, k=16)["created_indexes"] == ["hnsw", "ivf"]
    return client, corpus


def searcher(base, **settings):
    client, corpus = base
    client.statements.clear()
    client.searches.clear()
    config = SearchSettings(codebook_path=corpus / CODEBOOK_NAME, k=3, **settings)
    return SegmentSearcher(client, SegmentMFCCExtractor.load(config.codebook_path), config)


@pytest.mark.parametrize("index_type", ["hnsw", "ivf"])
def test_a_noisy_copy_finds_its_original_segment_by_segment(base, index_type):
    result = searcher(base, index_type=index_type).search_audio(melody(FREQUENCIES[3], seed=7, noise=0.02))

    assert result["index_type"] == index_type and result["used_index"] is True
    assert result["k"] == 3 and len(result["segments"]) == 4
    for segment in result["segments"]:
        best = segment["candidates"][0]
        assert (best["song_id"], best["title"], best["artist"]) == (3, "Canción 3", "Artista 3")
        # Con 50 % de solape, el segmento vecino comparte la mitad del audio: el mejor
        # candidato está a lo sumo a un salto (2.5 s), y el tramo equivalente está en el top k.
        # Si lo encontró una consulta corrida, su inicio se corrige en ese corrimiento, así que
        # cae a lo sumo a un corrimiento (0.625 s) del inicio exacto.
        assert abs(best["start_s"] - segment["start_s"]) <= 2.5
        assert any(c["song_id"] == 3 and abs(c["start_s"] - segment["start_s"]) <= 0.625 + 1e-6
                   for c in segment["candidates"])
        scores = [candidate["score"] for candidate in segment["candidates"]]
        assert len(scores) == 3 and scores == sorted(scores, reverse=True) and scores[0] <= 1.0 + 1e-6


def test_a_fragment_shifted_in_time_reports_both_intervals(base):
    shifted = reference(2)[int(2.5 * SAMPLE_RATE):]  # empieza en 0:02.5 de la canción 2

    result = searcher(base).search_audio(shifted)

    first = result["segments"][0]
    assert (first["start_s"], first["end_s"]) == (0.0, 5.0)
    assert first["candidates"][0]["song_id"] == 2
    assert (first["candidates"][0]["start_s"], first["candidates"][0]["end_s"]) == (2.5, 7.5)


def test_a_fragment_off_the_segment_grid_finds_its_interval_with_the_shifts(base):
    # Empieza en 0:01.25: sin corrimientos, cada segmento cae entre dos de la base.
    shifted = reference(2)[int(1.25 * SAMPLE_RATE):]

    result = searcher(base).search_audio(shifted)

    assert result["query_shifts_s"] == [0.0, 0.625, 1.25, 1.875]
    for segment in result["segments"][:3]:
        best = segment["candidates"][0]
        assert best["song_id"] == 2
        # el desfase entre ambos audios es 1.25 s, salvo el redondeo de la grilla
        assert abs((best["start_s"] - segment["start_s"]) - 1.25) <= 0.625 + 1e-6


def test_without_shifts_only_one_search_is_made_per_segment(base):
    single = searcher(base, query_shifts_s=(0.0,))
    single_result = single.search_audio(reference(5))
    assert len(single.client.searches) == len(single_result["segments"])

    several = searcher(base)   # el mismo cliente: searcher() vacía sus búsquedas
    several_result = several.search_audio(reference(5))
    assert len(several.client.searches) > len(several_result["segments"])


def test_merging_shifted_candidates_keeps_the_best_score_and_corrects_the_interval():
    def candidate(segment_id, start, score):
        return Candidate(2, "Canción 2", "Artista 2", "CC BY", None, segment_id, start, start + 5.0, score, 1 - score)

    merged = merge_candidates([(0.0, [candidate(1, 2.5, 0.90), candidate(9, 0.0, 0.50)]),
                               (1.25, [candidate(1, 2.5, 0.97), candidate(2, 5.0, 0.95)])], k=2)

    assert [(c.segment_id, c.start_s, c.end_s, c.score) for c in merged] == [(1, 1.25, 6.25, 0.97), (2, 3.75, 8.75, 0.95)]


@pytest.mark.parametrize("shifts", [(), (0.5, 1.0), (0.0, -1.0)])
def test_query_shifts_must_start_at_zero(shifts):
    with pytest.raises(ValueError, match="query_shifts_s"):
        SearchSettings(query_shifts_s=shifts)


def test_each_candidate_carries_its_song_metadata(base):
    result = searcher(base).search_audio(reference(5))

    best = result["segments"][0]["candidates"][0]
    assert best["license_family"] == "CC BY" and best["track_url"] == "http://fma.org/5"
    assert best["segment_id"] == 5000
    assert set(result["timings_ms"]) == {"extraction_ms", "dbms_search_ms", "total_ms"}


def test_only_the_parameter_of_the_chosen_index_is_sent(base):
    hnsw = searcher(base, index_type="hnsw", ef_search=64)
    hnsw.search_audio(melody(FREQUENCIES[2]))
    assert {(s["ef_search"], s["n_probe"]) for s in hnsw.client.searches} == {(64, None)}

    ivf = searcher(base, index_type="ivf", n_probe=2)
    ivf.search_audio(melody(FREQUENCIES[2]))
    assert {(s["ef_search"], s["n_probe"]) for s in ivf.client.searches} == {(None, 2)}


def test_song_metadata_is_read_once_until_refresh(base):
    search = searcher(base)
    search.search_audio(melody(FREQUENCIES[2]))
    search.search_audio(melody(FREQUENCIES[3]))
    assert sum("FROM audio_songs" in sql for sql in search.client.statements) == 1

    search.refresh()
    search.search_audio(melody(FREQUENCIES[2]))
    assert sum("FROM audio_songs" in sql for sql in search.client.statements) == 2


@pytest.mark.parametrize("arguments, message", [
    ({"k": 0}, "k debe ser"),
    ({"index_type": "btree"}, "Índice desconocido"),
    ({"metric": "manhattan"}, "Métrica desconocida"),
])
def test_invalid_search_parameters(base, arguments, message):
    with pytest.raises(ValueError, match=message):
        searcher(base).search_vector(np.ones(16, dtype=np.float32) / 4, **arguments)


def test_an_untrained_codebook_is_rejected(base):
    client, _ = base

    with pytest.raises(ValueError, match="no está entrenado"):
        SegmentSearcher(client, SegmentMFCCExtractor(k=16))


def test_a_missing_codebook_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="corre antes extract_segments.py"):
        SegmentSearcher.from_settings(SearchSettings(codebook_path=tmp_path / "no_existe.pkl"))
