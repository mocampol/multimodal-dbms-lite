"""
Lectura de la carga congelada (benchmarks/text_frozen.py), con el corpus sintético.
"""

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import text_workload as workload  # noqa: E402
from text_frozen import load_frozen_workload  # noqa: E402
from text_manifest import qrels_path  # noqa: E402

CORPUS = "synthetic"
SIZES = [1_000, 3_000]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("frozen")
    workload.build_workload(CORPUS, out_dir, SIZES)
    return out_dir


@pytest.mark.parametrize("n", SIZES)
def test_the_relevant_documents_match_the_manifest_counts_at_each_n(data_dir, n):
    queries, relevant = load_frozen_workload(CORPUS, n, data_dir)

    assert len(queries) == len(relevant) == 30
    for query in queries:
        ids = relevant[query["query_id"]]
        assert len(ids) == query["relevant_docs"][str(n)] > 0
        assert max(ids) <= n
        assert query["seed_doc_id"] in ids


def test_the_smaller_n_is_the_larger_n_filtered_by_id(data_dir):
    _, small = load_frozen_workload(CORPUS, SIZES[0], data_dir)
    _, large = load_frozen_workload(CORPUS, SIZES[-1], data_dir)

    assert all(small[query_id] == {d for d in ids if d <= SIZES[0]} for query_id, ids in large.items())


def test_an_n_that_was_not_frozen_is_rejected(data_dir):
    with pytest.raises(ValueError, match="no está congelado"):
        load_frozen_workload(CORPUS, 5_000, data_dir)


def test_missing_files_tell_how_to_generate_them(tmp_path):
    with pytest.raises(ValueError, match="text_workload.py --corpus synthetic"):
        load_frozen_workload(CORPUS, 1_000, tmp_path)


def test_tampered_labels_are_rejected_unless_validation_is_off(data_dir, tmp_path):
    import shutil
    copy = tmp_path / "data"
    shutil.copytree(data_dir, copy)
    path = qrels_path(CORPUS, copy)
    path.write_bytes(path.read_bytes() + b"G1-01 0 999999 1\n")

    with pytest.raises(ValueError, match="no es válida"):
        load_frozen_workload(CORPUS, 1_000, copy)
    load_frozen_workload(CORPUS, 1_000, copy, validate=False)  # no lanza