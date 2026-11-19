"""
Agregar canciones a la base de referencia desde la aplicación
(apps/audio-copyright/backend/song_ingest.py, dataset/reference_songs.py y POST /songs).

Arma una base real pequeña (tablas, segmentos, codebook e índices) con el cliente de
test_segment_search, que ejecuta el SQL, las inserciones y las búsquedas en un catálogo.
"""

import io
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
from detection import DetectionSettings, detect  # noqa: E402
from detection_api import create_app  # noqa: E402
from extract_segments import CODEBOOK_NAME, extract  # noqa: E402
from fma_corpus import write_manifest  # noqa: E402
from load_songs import load  # noqa: E402
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory  # noqa: E402
from query.query_engine import execute  # noqa: E402
from reference_songs import (  # noqa: E402
    UPLOAD_ID_START, UploadStore, artist_id_for, read_uploads, reference_songs, song_path, uploads_dir,
)
from segment_extractor import SAMPLE_RATE, SegmentMFCCExtractor  # noqa: E402
from segment_search import SearchSettings, SegmentSearcher  # noqa: E402
from song_ingest import add_song  # noqa: E402
from test_detection_api import call  # noqa: E402
from test_segment_search import FREQUENCIES, LocalClient, melody, reference, song  # noqa: E402

NEW_SONG = [500, 1000, 1500]


@pytest.fixture
def base(tmp_path):
    """Base de referencia con 3 canciones de FMA: tablas, segmentos, codebook e índices."""
    corpus = tmp_path / "audio" / "fma"
    corpus.mkdir(parents=True)
    for song_id in FREQUENCIES:
        sf.write(str(corpus / f"{song_id:06d}.wav"), reference(song_id), SAMPLE_RATE)
    write_manifest([song(song_id) for song_id in FREQUENCIES], corpus / "songs.csv")
    heap = make_heap_factory(str(tmp_path / "db"))
    client = LocalClient(Catalog(
        heap_factory=heap,
        storage_factories={StorageType.HEAP: heap, StorageType.SEQUENTIAL: make_sequential_factory(str(tmp_path / "db"))},
        index_buffer_factory=make_index_buffer_factory(str(tmp_path / "db")),
    ))
    load(client, corpus)
    extract(client, corpus, k=16)
    settings = SearchSettings(codebook_path=corpus / CODEBOOK_NAME, k=3)
    return SegmentSearcher(client, SegmentMFCCExtractor.load(settings.codebook_path), settings), UploadStore(corpus)


def new_audio(frequencies=NEW_SONG, seed=11):
    signal = melody(frequencies, seed=seed, noise=0.01)
    buffer = io.BytesIO()
    sf.write(buffer, signal, SAMPLE_RATE, format="WAV")
    return signal, buffer.getvalue()


def rows(searcher, sql):
    return [[v.data for v in record.values] for record in execute(sql, searcher.client.catalog)]


def index_sizes(searcher):
    catalog = searcher.client.catalog
    return {kind: len(catalog.get_physical_index("audio_segments", "embedding", index_type=kind))
            for kind in ("hnsw", "ivf")}


# Agregar una canción

def test_an_added_song_enters_the_indexes_and_is_detected(base):
    searcher, store = base
    before = index_sizes(searcher)
    signal, data = new_audio()

    added = add_song(searcher, store, signal, data, ".wav", "Nueva", "Yo", "Propia")

    assert added["song_id"] == UPLOAD_ID_START and added["segments"] == 4
    assert index_sizes(searcher) == {kind: size + 4 for kind, size in before.items()}
    result = detect(searcher.search_audio(signal), DetectionSettings(threshold=0.7, min_segments=2, margin=0.0))
    assert [(m["song_id"], m["title"], m["artist"]) for m in result["matches"]] == [(UPLOAD_ID_START, "Nueva", "Yo")]


def test_the_file_the_manifest_and_the_tables_are_written(base):
    searcher, store = base
    signal, data = new_audio()

    added = add_song(searcher, store, signal, data, ".WAV", "Nueva", "Yo", "CC BY", album="Demo",
                     source_url="http://ejemplo.org/nueva")

    path = uploads_dir(store.corpus_dir) / f"{added['song_id']}.wav"
    assert path.read_bytes() == data
    [uploaded] = read_uploads(store.corpus_dir).values()
    assert (uploaded.title, uploaded.source, uploaded.license_family, uploaded.album_title) == ("Nueva", "upload", "CC BY", "Demo")
    [song_row] = rows(searcher, f"SELECT title, path, source FROM audio_songs WHERE song_id = {added['song_id']};")
    assert song_row == ["Nueva", path.resolve().as_posix(), "upload"]
    segments = rows(searcher, f"SELECT model_name FROM audio_segments WHERE song_id = {added['song_id']};")
    assert {row[0] for row in segments} == {searcher.extractor.model_name} and len(segments) == 4


def test_the_codebook_is_not_retrained(base):
    searcher, store = base
    codebook = store.corpus_dir / CODEBOOK_NAME
    saved = codebook.read_bytes()
    centroids = np.array(searcher.extractor.quantizer.centroids, copy=True)
    signal, data = new_audio()

    add_song(searcher, store, signal, data, ".wav", "Nueva", "Yo", "Propia")

    assert codebook.read_bytes() == saved
    np.testing.assert_array_equal(searcher.extractor.quantizer.centroids, centroids)


def test_an_existing_artist_is_reused_and_a_new_one_gets_its_own_id(base):
    searcher, store = base
    first_signal, first = new_audio()
    second_signal, second = new_audio([700, 1400, 2100], seed=12)

    known = add_song(searcher, store, first_signal, first, ".wav", "Otra", "  artista 2 ", "Propia")
    new = add_song(searcher, store, second_signal, second, ".wav", "Más", "Nadie", "Propia")

    assert known["artist_id"] == 20 and new["artist_id"] == UPLOAD_ID_START
    assert len(rows(searcher, "SELECT artist_id FROM audio_artists;")) == len(FREQUENCIES) + 1
    assert (known["song_id"], new["song_id"]) == (UPLOAD_ID_START, UPLOAD_ID_START + 1)


def test_a_failure_in_the_dbms_undoes_everything(base, monkeypatch):
    searcher, store = base
    before = index_sizes(searcher)

    def fail(table, rows):
        raise RuntimeError("POST /tables/audio_segments/rows respondió 500")

    monkeypatch.setattr(searcher.client, "insert_rows", fail)
    signal, data = new_audio()

    with pytest.raises(RuntimeError, match="respondió 500"):
        add_song(searcher, store, signal, data, ".wav", "Nueva", "Nadie", "Propia")

    assert read_uploads(store.corpus_dir) == {}
    assert not any(uploads_dir(store.corpus_dir).glob("*.wav"))
    assert rows(searcher, f"SELECT song_id FROM audio_songs WHERE song_id = {UPLOAD_ID_START};") == []
    assert rows(searcher, f"SELECT artist_id FROM audio_artists WHERE artist_id = {UPLOAD_ID_START};") == []
    assert index_sizes(searcher) == before


@pytest.mark.parametrize("title, artist, license, signal, message", [
    ("", "Yo", "Propia", None, "obligatorios"),
    ("Nueva", "Yo", "Robada", None, "Licencia no admitida"),
    ("Nueva", "Yo", "Propia", np.zeros(8 * SAMPLE_RATE, dtype=np.float32), "silencio"),
])
def test_invalid_songs_are_rejected_without_writing_anything(base, title, artist, license, signal, message):
    searcher, store = base
    audio, data = new_audio()

    with pytest.raises(ValueError, match=message):
        add_song(searcher, store, audio if signal is None else signal, data, ".wav", title, artist, license)

    assert read_uploads(store.corpus_dir) == {}
    assert not any(uploads_dir(store.corpus_dir).glob("*.wav"))


# Que audio-data no las borre

def test_reloading_the_corpus_keeps_the_added_songs(base):
    searcher, store = base
    signal, data = new_audio()
    added = add_song(searcher, store, signal, data, ".wav", "Nueva", "Yo", "Propia")

    loaded = load(searcher.client, store.corpus_dir)
    extracted = extract(searcher.client, store.corpus_dir, k=16)

    assert loaded["songs"] == {"inserted": 0, "deleted": 0, "total": len(FREQUENCIES) + 1}
    assert extracted["codebook"] == "loaded" and extracted["inserted_segments"] == 0 and extracted["deleted_songs"] == 0
    assert len(rows(searcher, f"SELECT segment_id FROM audio_segments WHERE song_id = {added['song_id']};")) == 4


def test_a_new_codebook_is_trained_only_with_the_fma_corpus(base, monkeypatch):
    searcher, store = base
    signal, data = new_audio()
    add_song(searcher, store, signal, data, ".wav", "Nueva", "Yo", "Propia")
    trained_with = []
    original_fit = SegmentMFCCExtractor.fit
    monkeypatch.setattr(SegmentMFCCExtractor, "fit",
                        lambda self, songs, *args, **kwargs: trained_with.append(len(songs)) or original_fit(self, songs))

    result = extract(searcher.client, store.corpus_dir, k=8)   # otra configuración: se reentrena

    assert trained_with == [len(FREQUENCIES)]
    assert result["inserted_songs"] == len(FREQUENCIES) + 1    # pero la agregada también se vectoriza


# Manifiestos

def test_reference_songs_join_both_manifests_and_resolve_each_path(base):
    searcher, store = base
    signal, data = new_audio()
    added = add_song(searcher, store, signal, data, ".wav", "Nueva", "Yo", "Propia")

    songs = reference_songs(store.corpus_dir)

    assert set(songs) == set(FREQUENCIES) | {added["song_id"]}
    assert song_path(store.corpus_dir, songs[2]) == (store.corpus_dir / "000002.wav").resolve()
    assert song_path(store.corpus_dir, songs[added["song_id"]]).parent == uploads_dir(store.corpus_dir).resolve()


def test_a_repeated_song_id_between_manifests_is_an_error(base):
    _, store = base
    write_manifest([song(2)], uploads_dir(store.corpus_dir) / "uploads.csv")

    with pytest.raises(RuntimeError, match="repetidos"):
        reference_songs(store.corpus_dir)


def test_artist_ids_for_uploads_start_after_fma_ones():
    assert artist_id_for("Nuevo", {}) == UPLOAD_ID_START


# API

def test_post_songs_adds_a_song_and_get_songs_lists_it(base):
    searcher, store = base
    app = create_app(searcher=searcher, settings=DetectionSettings(threshold=0.7, min_segments=2, margin=0.0), store=store)
    _, data = new_audio()

    status, added = call(app, "POST", "/songs?filename=nueva.wav&title=Nueva&artist=Yo&license=CC%20BY", data)
    _, listed = call(app, "GET", "/songs")

    assert status == 201 and added["segments"] == 4 and added["license_family"] == "CC BY"
    assert listed["count"] == 1 and listed["songs"][0]["title"] == "Nueva"
    assert "Propia" in listed["licenses"]


@pytest.mark.parametrize("path, status, message", [
    ("/songs?filename=nueva.wav&title=Nueva&artist=Yo&license=Robada", 422, "Licencia no admitida"),
    ("/songs?filename=nueva.txt&title=Nueva&artist=Yo", 415, "Formato no admitido"),
    ("/songs?filename=nueva.wav&artist=Yo", 422, None),
])
def test_post_songs_rejects_invalid_requests(base, path, status, message):
    searcher, store = base
    app = create_app(searcher=searcher, store=store)
    _, data = new_audio()

    got_status, body = call(app, "POST", path, data)

    assert got_status == status
    if message:
        assert message in body["detail"]
    assert read_uploads(store.corpus_dir) == {}
