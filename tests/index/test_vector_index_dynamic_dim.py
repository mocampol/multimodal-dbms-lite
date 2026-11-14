import time

import numpy as np
import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from common.record import Record
from common.value import DataType, Value
from index.multimedia import RawVectorExtractor, decode_raw_vector, encode_raw_vector
from index.vector import HNSWIndex, IVFIndex, VectorMetric
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute
from storage.heap.rid import RID


def _make_catalog(base_dir):
    heap_factory = make_heap_factory(str(base_dir))
    return Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )


def _flush_all(catalog: Catalog) -> None:
    for storage in catalog._table_storage.values():
        storage.bm.flush_all()
        if hasattr(storage, "overflow"):
            storage.overflow.bm.flush_all()
    catalog._sys_tables.bm.flush_all()
    catalog._sys_columns.bm.flush_all()
    catalog._sys_indexes.bm.flush_all()


def _normalized_random_vectors(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    mat = rng.standard_normal((count, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("metric", [VectorMetric.COSINE, VectorMetric.EUCLIDEAN])
def test_vectorized_ivf_search_matches_exact_ranking_and_meets_latency(dim: int, metric: VectorMetric):
    rng = np.random.default_rng(42 + dim)
    vectors = _normalized_random_vectors(rng, 250, dim)
    rids = [RID(i // 10, i % 10) for i in range(len(vectors))]

    ivf = IVFIndex(dim=dim, metric=metric, n_lists=10, n_probe=10)
    ivf.build(vectors, rids)

    query = vectors[13].copy()
    t0 = time.perf_counter()
    results = ivf.search(query, k=5, metric=metric)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(results) == 5
    assert results[0].rid == rids[13]
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert elapsed_ms < 100.0


def test_raw_vector_extractor_roundtrip_arbitrary_dim():
    vec = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    raw = encode_raw_vector(vec)
    assert isinstance(raw, bytes)
    assert len(raw) == 256 * 4

    decoded = decode_raw_vector(raw, expected_dim=256)
    np.testing.assert_allclose(decoded, vec, atol=1e-6)

    extractor = RawVectorExtractor(dim=256)
    fv = extractor.extract(list(vec))
    assert fv.k == 256
    np.testing.assert_allclose(fv.data, vec, atol=1e-6)


@pytest.mark.parametrize("dim,index_type", [(128, "hnsw"), (512, "ivf")])
def test_catalog_bytea_vector_index_infers_dim_and_survives_reboot(
    tmp_path, dim: int, index_type: str
):
    rng = np.random.default_rng(100 + dim)
    vecs = _normalized_random_vectors(rng, 20, dim)

    cat = _make_catalog(tmp_path)
    execute(
        "CREATE TABLE faces (id INTEGER PRIMARY KEY, label VARCHAR(64), emb BYTEA NOT NULL);",
        cat,
    )

    storage = cat.get_storage("faces")
    schema = cat.get_schema("faces")
    for idx, vec in enumerate(vecs, start=1):
        rec = Record([
            Value(DataType.INTEGER, idx),
            Value(DataType.VARCHAR, f"person-{idx}"),
            Value(DataType.BYTEA, encode_raw_vector(vec)),
        ])
        assert rec.validate(schema)
        rid = storage.insert(rec)
        cat.register_insert("faces", rec, rid)

    entry = cat.create_index(
        "faces",
        "emb",
        index_type,
        index_name=f"idx_faces_{index_type}",
        index_params={"n_lists": 4, "n_probe": 4} if index_type == "ivf" else {"ef_search": 64},
    )
    assert entry["index_type"] == index_type

    phys = cat.get_physical_index("faces", "emb", index_type=index_type)
    assert isinstance(phys, (HNSWIndex, IVFIndex))
    assert phys.dim == dim
    assert len(phys) == 20

    hits = phys.search(vecs[5], k=3, metric=VectorMetric.COSINE)
    rec = storage.get(hits[0].rid)
    assert rec[0].data == 6

    _flush_all(cat)

    # Reboot: nuevo Catalog sobre el mismo directorio debe reabrir el índice con la misma dim
    reopened = _make_catalog(tmp_path)
    phys_reopened = reopened.get_physical_index("faces", "emb", index_type=index_type)
    assert phys_reopened is not None
    assert phys_reopened.dim == dim
    assert len(phys_reopened) == 20

    hits_after = phys_reopened.search(vecs[5], k=1, metric=VectorMetric.COSINE)
    rec_after = reopened.get_storage("faces").get(hits_after[0].rid)
    assert rec_after[0].data == 6

    # Verificar que DELETE mantiene sincronizado el índice
    execute("DELETE FROM faces WHERE id = 6;", reopened)
    hits_after_del = reopened.get_physical_index("faces", "emb", index_type=index_type).search(
        vecs[5], k=3, metric=VectorMetric.COSINE
    )
    returned_ids = {reopened.get_storage("faces").get(h.rid)[0].data for h in hits_after_del}
    assert 6 not in returned_ids
