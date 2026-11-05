"""
Métricas y control de la comparación de similitud (benchmarks/benchmark_vector.py).
No necesita los datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from benchmark_vector import (  # noqa: E402
    FIELDNAMES,
    VectorizedIVF,
    distances_to,
    index_configs,
    ivf_lists_for,
    precision_at_k,
    read_rows,
    recall_at_k,
    time_summary,
    top_k_ids,
    write_rows,
)
from index.vector import IVFIndex, VectorMetric, compute_distance  # noqa: E402


def random_vectors(n, dim=8, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


# Distancias

@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_distances_match_the_engine(metric):
    base, query = random_vectors(30), random_vectors(1, seed=1)[0]
    expected = [compute_distance(query, row, VectorMetric.from_str(metric)) for row in base]
    assert distances_to(base, query, metric) == pytest.approx(expected, abs=1e-5)


def test_cosine_distance_of_a_zero_vector_is_one_like_the_engine():
    base = np.array([[0, 0], [1, 0]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    assert distances_to(base, query, "cosine") == pytest.approx([1.0, 0.0])


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="no soportada"):
        distances_to(random_vectors(2), random_vectors(1)[0], "manhattan")


def test_top_k_breaks_ties_by_id():
    distances = np.array([0.5, 0.1, 0.5, 0.5, 0.9])
    assert top_k_ids(distances, 3).tolist() == [1, 0, 2]


def test_top_k_with_fewer_rows_than_k():
    assert top_k_ids(np.array([0.2, 0.1]), 10).tolist() == [1, 0]


# Recall y precisión

def test_recall_counts_results_within_the_kth_distance():
    distances = np.array([0.1, 0.2, 0.3, 0.9])
    assert recall_at_k([0, 1], distances, kth_distance=0.2, k=2) == 1.0
    assert recall_at_k([0, 3], distances, kth_distance=0.2, k=2) == 0.5


def test_recall_accepts_a_tied_neighbor_that_is_not_in_the_exact_list():
    """Con histogramas idénticos, cualquiera de los empatados es correcto."""
    distances = np.array([0.1, 0.2, 0.2])
    assert recall_at_k([0, 2], distances, kth_distance=0.2, k=2) == 1.0


def test_recall_penalizes_returning_too_few_results():
    assert recall_at_k([0], np.array([0.1, 0.2]), kth_distance=0.2, k=2) == 0.5
    assert recall_at_k([], np.array([0.1, 0.2]), kth_distance=0.2, k=2) == 0.0


def test_precision_counts_neighbors_of_the_same_class():
    labels = np.array(["a", "b", "a", "c"])
    assert precision_at_k([0, 1, 2, 3], labels, "a", k=4) == 0.5
    assert precision_at_k([], labels, "a", k=4) == 0.0


def test_time_summary_averages_repetitions_per_query():
    times = np.array([[0.001, 0.003], [0.003, 0.005]])
    summary = time_summary(times)
    assert summary["avg_time_ms"] == pytest.approx(3.0)
    assert summary["min_time_ms"] == pytest.approx(2.0)
    assert summary["max_time_ms"] == pytest.approx(4.0)


# Control vectorizado

@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
@pytest.mark.parametrize("n_probe", [1, 4, 16])
def test_vectorized_control_returns_the_same_as_ivf_index(metric, n_probe):
    vectors, queries = random_vectors(500), random_vectors(20, seed=1)
    index = IVFIndex(dim=8, metric=metric, n_lists=16, n_probe=n_probe)
    index.build(vectors, list(range(500)))
    control = VectorizedIVF(index)
    for query in queries:
        expected = [result.rid for result in index.search(query, k=10)]
        ids, candidates = control.search(query, n_probe)
        assert ids.tolist() == expected
        assert candidates >= 10


def test_vectorized_control_counts_every_vector_when_probing_all_lists():
    vectors = random_vectors(100)
    index = IVFIndex(dim=8, metric="euclidean", n_lists=4)
    index.build(vectors, list(range(100)))
    _, candidates = VectorizedIVF(index).search(vectors[0], 4)
    assert candidates == 100


# Configuraciones y CSV

def test_ivf_lists_follow_the_square_root_of_n():
    assert [ivf_lists_for(n) for n in (1_000, 10_000, 100_000)] == [32, 100, 316]


def test_configs_do_not_probe_more_lists_than_exist():
    configs = {(c.kind, c.config): c for c in index_configs(1_000)}
    assert max(configs["ivf", "sqrt"].values) == 32
    assert configs["ivf", "engine"].n_lists == 16 and configs["ivf", "engine"].values == (4,)
    assert 50 in configs["hnsw", "sweep"].values
    assert max({(c.kind, c.config): c for c in index_configs(100_000)}["ivf", "sqrt"].values) == 64


def row(**values):
    base = {field: "" for field in FIELDNAMES}
    base.update(modality="image", n=1000, variant="raw", metric="cosine", index_kind="hnsw", config="sweep")
    base.update(values)
    return base


def test_write_rows_replaces_the_same_key_and_keeps_the_others(tmp_path):
    path = tmp_path / "results.csv"
    write_rows([row(ef_search=10, avg_time_ms=1), row(n=10000, avg_time_ms=2)], path)
    write_rows([row(ef_search=10, avg_time_ms=3), row(ef_search=50, avg_time_ms=4)], path)

    rows = read_rows(path)
    assert sorted((r["n"], r["avg_time_ms"]) for r in rows) == [("1000", "3"), ("1000", "4"), ("10000", "2")]


def test_read_rows_of_a_missing_file_is_empty(tmp_path):
    assert read_rows(tmp_path / "nada.csv") == []
