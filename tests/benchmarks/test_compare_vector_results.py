"""
Verificaciones de la comparación de similitud (benchmarks/compare_vector_results.py)
sobre filas sintéticas. No necesita los datasets.
"""

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from benchmark_vector import FIELDNAMES  # noqa: E402
from compare_vector_results import (  # noqa: E402
    check,
    expected_rows_per_combination,
    fastest_at_recall,
    missing_combinations,
)


def row(**values):
    base = {field: "" for field in FIELDNAMES}
    base.update(modality="image", n="1000", variant="normalized", metric="cosine", k="10", num_queries="200",
                recall_at_k="0.9", precision_at_k="0.1", avg_time_ms="1.0")
    base.update({key: str(value) for key, value in values.items()})
    return base


def ivf_pair(n_probe=1, **values):
    common = dict(config="sqrt", n_probe=n_probe, avg_candidates=10, **values)
    return [row(index_kind="ivf", **common), row(index_kind="ivf_numpy", **common)]


def test_valid_rows_pass():
    assert check(ivf_pair(1, recall_at_k=0.5) + ivf_pair(2, recall_at_k=0.9) + [row(index_kind="exact", recall_at_k=1)]) == []


def test_a_different_k_or_query_count_fails():
    assert "k=5" in check([row(index_kind="hnsw", k=5)])[0]
    assert "100 consultas" in check([row(index_kind="hnsw", num_queries=100)])[0]


def test_recall_out_of_range_fails():
    assert "fuera de [0, 1]" in check([row(index_kind="hnsw", recall_at_k=1.5)])[0]


def test_exact_search_must_have_full_recall():
    assert "fuerza bruta" in check([row(index_kind="exact", recall_at_k=0.99)])[0]


def test_ivf_and_its_control_must_agree():
    ivf, control = ivf_pair()
    control["recall_at_k"] = "0.8"
    assert "recall_at_k distinto" in check([ivf, control])[0]


def test_ivf_without_its_control_fails():
    assert "falta ivf_numpy" in check(ivf_pair()[:1])[0]


def test_ivf_recall_cannot_drop_when_probing_more_lists():
    errors = check(ivf_pair(1, recall_at_k=0.9) + ivf_pair(2, recall_at_k=0.8))
    assert any("el recall baja" in error for error in errors)


def test_equivalent_metrics_must_find_the_same_neighbors():
    rows = [row(index_kind="exact", recall_at_k=1, precision_at_k=0.1),
            row(index_kind="exact", recall_at_k=1, precision_at_k=0.2, metric="euclidean")]
    assert "normalized euclidean" in check(rows)[0]


def test_raw_euclidean_may_find_other_neighbors():
    rows = [row(index_kind="exact", recall_at_k=1, precision_at_k=0.1),
            row(index_kind="exact", recall_at_k=1, precision_at_k=0.2, metric="euclidean", variant="raw")]
    assert check(rows) == []


def test_hnsw_that_does_not_use_every_core_fails():
    slow = row(index_kind="hnsw", n=10000, index_build_time_s=2.5, build_time_all_cores_s=2.5)
    assert "todos los núcleos" in check([slow])[0]
    fast = row(index_kind="hnsw", n=10000, index_build_time_s=2.5, build_time_all_cores_s=0.5)
    assert check([fast]) == []


def test_small_hnsw_builds_are_not_compared_by_threads():
    assert check([row(index_kind="hnsw", index_build_time_s=0.1, build_time_all_cores_s=0.2)]) == []


def test_fastest_at_recall_ignores_configs_below_the_target_and_outside_the_sweep():
    rows = [row(index_kind="hnsw", config="sweep", ef_search=10, recall_at_k=0.5, avg_time_ms=0.1),
            row(index_kind="hnsw", config="sweep", ef_search=50, recall_at_k=0.96, avg_time_ms=0.3),
            row(index_kind="hnsw", config="sweep", ef_search=100, recall_at_k=0.99, avg_time_ms=0.6),
            row(index_kind="ivf", config="engine", n_probe=4, recall_at_k=0.99, avg_time_ms=0.1)]
    assert fastest_at_recall(rows, "hnsw")["ef_search"] == "50"
    assert fastest_at_recall(rows, "ivf") is None


def test_expected_rows_match_the_published_grid():
    assert [expected_rows_per_combination(n) for n in (1_000, 10_000, 100_000)] == [21, 23, 23]


def test_missing_combinations_are_reported():
    warnings = missing_combinations([row(index_kind="exact")])
    assert "image N=1000 normalized cosine: 1 filas de 21" in warnings
    assert len(warnings) == 4
