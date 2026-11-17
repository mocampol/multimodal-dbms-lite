"""
Índices vectoriales de audio_segments (apps/audio-copyright/features/segment_index.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

FEATURES_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from catalog.catalog import Catalog  # noqa: E402
from catalog.table_metadata import StorageType  # noqa: E402
from common.record import Record  # noqa: E402
from common.value import DataType, Value  # noqa: E402
from loader.csv_loader import _insert_one_record, _transaction_manager  # noqa: E402
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory  # noqa: E402
from query.query_engine import execute  # noqa: E402
from segment_index import (  # noqa: E402
    HNSW_INDEX,
    IVF_INDEX,
    IVF_N_PROBE,
    SEGMENTS_TABLE,
    ensure_segment_indexes,
    ivf_lists,
)


class FakeClient:
    """Registra qué índices se piden; get_table refleja los ya creados."""

    def __init__(self, indexes=()):
        self.indexes = [dict(index) for index in indexes]
        self.created = []

    def get_table(self, table):
        return {"indexes": list(self.indexes)}

    def create_index(self, table, column, index_type, index_name=None, dim=None, index_params=None):
        self.created.append((table, column, index_type, index_name, dim, index_params))
        self.indexes.append({"column_name": column, "index_type": index_type})
        return {}


class TestEnsureSegmentIndexes:
    def test_creates_hnsw_and_ivf_with_their_parameters(self):
        client = FakeClient()

        assert ensure_segment_indexes(client, dim=64, n_segments=11_000) == ["hnsw", "ivf"]

        assert client.created == [
            (SEGMENTS_TABLE, "embedding", "hnsw", HNSW_INDEX, 64,
             {"metric": "cosine", "M": 16, "ef_construction": 200, "ef_search": 100}),
            (SEGMENTS_TABLE, "embedding", "ivf", IVF_INDEX, 64,
             {"metric": "cosine", "n_lists": 105, "n_probe": IVF_N_PROBE}),
        ]

    def test_a_second_call_creates_nothing(self):
        client = FakeClient()
        ensure_segment_indexes(client, dim=64, n_segments=100)

        assert ensure_segment_indexes(client, dim=64, n_segments=100) == []
        assert len(client.created) == 2

    def test_only_the_missing_index_is_created(self):
        client = FakeClient(indexes=[{"column_name": "embedding", "index_type": "hnsw"}])

        assert ensure_segment_indexes(client, dim=64, n_segments=100) == ["ivf"]

    def test_an_index_on_another_column_does_not_count(self):
        client = FakeClient(indexes=[{"column_name": "otra", "index_type": "hnsw"}])

        assert ensure_segment_indexes(client, dim=64, n_segments=100) == ["hnsw", "ivf"]

    def test_an_empty_table_gets_no_index(self):
        client = FakeClient()

        assert ensure_segment_indexes(client, dim=64, n_segments=0) == []
        assert client.created == []

    @pytest.mark.parametrize("n_segments, lists", [(0, 1), (1, 1), (100, 10), (11_000, 105)])
    def test_ivf_uses_the_square_root_of_the_segments(self, n_segments, lists):
        assert ivf_lists(n_segments) == lists


# =============================================================================
# Con el catálogo real
# =============================================================================

class CatalogClient:
    def __init__(self, catalog):
        self.catalog = catalog

    def get_table(self, table):
        return {"indexes": self.catalog.get_indexes(table)}

    def create_index(self, table, column, index_type, index_name=None, dim=None, index_params=None):
        return self.catalog.create_index(table, column, index_type, index_name=index_name,
                                         dim=dim, index_params=index_params)


@pytest.fixture
def catalog(tmp_path):
    directory = str(tmp_path / "db")
    heap = make_heap_factory(directory)
    catalog = Catalog(
        heap_factory=heap,
        storage_factories={StorageType.HEAP: heap, StorageType.SEQUENTIAL: make_sequential_factory(directory)},
        index_buffer_factory=make_index_buffer_factory(directory),
    )
    execute(f"CREATE TABLE {SEGMENTS_TABLE} (segment_id INTEGER PRIMARY KEY, embedding BYTEA NOT NULL);", catalog)
    return catalog


def insert_vectors(catalog, first_id, count, dim=8, seed=0):
    vectors = np.random.default_rng(seed).normal(size=(count, dim)).astype("<f4")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    manager = _transaction_manager(catalog)
    manager.begin()
    for offset, vector in enumerate(vectors):
        record = Record([Value(DataType.INTEGER, first_id + offset), Value(DataType.BYTEA, vector.tobytes())])
        _insert_one_record(catalog, manager, SEGMENTS_TABLE, record)
    manager.commit()


def test_the_engine_builds_both_indexes_with_the_requested_parameters(catalog):
    insert_vectors(catalog, first_id=0, count=100)

    assert ensure_segment_indexes(CatalogClient(catalog), dim=8, n_segments=100) == ["hnsw", "ivf"]

    hnsw = catalog.get_physical_index(SEGMENTS_TABLE, "embedding", index_type="hnsw")
    ivf = catalog.get_physical_index(SEGMENTS_TABLE, "embedding", index_type="ivf")
    assert (hnsw.dim, hnsw.M, hnsw.ef_construction, hnsw.ef_search, len(hnsw)) == (8, 16, 200, 100, 100)
    assert (ivf.dim, ivf.n_lists, ivf.n_probe, len(ivf)) == (8, 10, IVF_N_PROBE, 100)
    assert hnsw.metric.value == ivf.metric.value == "cosine"


def test_rows_inserted_later_enter_both_indexes(catalog):
    insert_vectors(catalog, first_id=0, count=100)
    ensure_segment_indexes(CatalogClient(catalog), dim=8, n_segments=100)

    insert_vectors(catalog, first_id=100, count=5, seed=1)

    assert len(catalog.get_physical_index(SEGMENTS_TABLE, "embedding", index_type="hnsw")) == 105
    assert len(catalog.get_physical_index(SEGMENTS_TABLE, "embedding", index_type="ivf")) == 105
