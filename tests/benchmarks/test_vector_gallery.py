"""
Selección de casos de la galería (benchmarks/plot_vector_gallery.py).
No necesita los datasets.
"""

import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from plot_vector_gallery import count_hits, exact_neighbors, select_cases  # noqa: E402


def test_exact_neighbors_are_sorted_by_similarity():
    base = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    query = np.array([[1, 0.1]], dtype=np.float32)
    assert exact_neighbors(base, query, k=2).tolist() == [[0, 2]]


def test_count_hits_compares_with_the_class_of_each_query():
    neighbors = np.array([[0, 1], [1, 2]])
    base_labels = np.array(["a", "b", "b"])
    query_labels = np.array(["a", "b"])
    assert count_hits(neighbors, base_labels, query_labels).tolist() == [1, 2]


def test_cases_are_the_best_then_the_worst():
    hits = np.array([1, 5, 0, 3, 2, 0])
    neighbors = np.arange(6).reshape(6, 1)
    cases = select_cases(neighbors, hits, per_group=2)
    assert [case.query for case in cases] == [1, 3, 2, 5]
    assert [case.hits for case in cases] == [5, 3, 0, 0]
    assert cases[0].neighbors == (1,)


def test_a_query_is_not_repeated_when_there_are_few():
    hits = np.array([2, 1, 0])
    cases = select_cases(np.zeros((3, 1), dtype=int), hits, per_group=2)
    assert [case.query for case in cases] == [0, 1, 2]
