"""
Comparación por dimensión (benchmarks/benchmark_vector_dims.py) y los codebooks de
varias dimensiones de prepare_vector_data.py. No necesita los datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import prepare_vector_data  # noqa: E402
import vector_common  # noqa: E402
from benchmark_vector import ENGINE_HNSW_EF, HNSW_EFS, exact_baseline, read_rows  # noqa: E402
from benchmark_vector_dims import (  # noqa: E402
    FIELDNAMES,
    METRIC,
    PYTHON_IVF_PROBES,
    build_phase,
    index_rows,
    ivf_probes_for,
    measured_groups,
    query_phase,
    same_neighbors,
    write_rows,
)
from compare_vector_dims import check  # noqa: E402
from index.multimedia.quantizer import Quantizer  # noqa: E402


def normalized_vectors(n, dim, seed=0):
    vectors = np.random.default_rng(seed).random((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def saved_vectors(tmp_path):
    base, queries = normalized_vectors(300, 128), normalized_vectors(12, 128, seed=1)
    np.save(tmp_path / "base.npy", base)
    np.save(tmp_path / "queries.npy", queries)
    return base, queries, str(tmp_path / "base.npy"), str(tmp_path / "queries.npy")


# Rutas por dimensión

def test_dimension_64_keeps_the_file_names_of_the_previous_benchmark():
    assert vector_common.features_path("image").name == "features_k64.npz"
    assert vector_common.manifest_path("audio").name == "vector_dataset_audio.json"


def test_other_dimensions_have_their_own_files():
    assert vector_common.features_path("image", dim=256).name == "features_k256.npz"
    assert vector_common.features_path("image", quick=True, dim=512).name == "features_k512_quick.npz"
    assert vector_common.manifest_path("audio", quick=True, dim=128).name == "vector_dataset_audio_k128_quick.json"


# Preparación

def test_prepare_skips_dimensions_that_already_have_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_vector_data, "features_path",
                        lambda modality, quick, dim: tmp_path / f"features_k{dim}.npz")
    (tmp_path / "features_k64.npz").touch()

    assert prepare_vector_data.pending_dims("image", False, [64, 128], force=False) == [128]
    assert prepare_vector_data.pending_dims("image", False, [64, 128], force=True) == [64, 128]


def test_one_histogram_per_codebook_from_the_same_descriptors(monkeypatch):
    rng = np.random.default_rng(0)
    descriptors = rng.random((40, 8)).astype(np.float32)
    quantizers = [Quantizer(k=k).fit(rng.random((600, 8)).astype(np.float32)) for k in (4, 16)]
    monkeypatch.setattr(prepare_vector_data, "_descriptors_of", lambda path: (descriptors, None))
    monkeypatch.setattr(prepare_vector_data, "_quantizers", quantizers)

    histograms, failure = prepare_vector_data._histogram_of("objeto.png")

    assert failure is None
    assert [len(histogram) for histogram in histograms] == [4, 16]
    assert all(histogram.sum() == len(descriptors) for histogram in histograms)


def test_a_rejected_object_has_no_histograms(monkeypatch):
    monkeypatch.setattr(prepare_vector_data, "_descriptors_of", lambda path: (None, "sin puntos"))

    assert prepare_vector_data._histogram_of("vacia.png") == (None, "sin puntos")


# Guardado y carga

def test_same_neighbors_tolerates_float32_rounding_but_not_other_results():
    expected = [[(1, 0.1), (2, 0.2)]]

    assert same_neighbors(expected, [[(1, 0.1 + 1e-8), (2, 0.2)]])
    assert not same_neighbors(expected, [[(2, 0.2), (1, 0.1)]])
    assert not same_neighbors(expected, [[(1, 0.1), (2, 0.25)]])
    assert not same_neighbors(expected, [])


@pytest.mark.parametrize("kind, n_lists", [("ivf", 17), ("hnsw", 0)])
def test_build_phase_reloads_an_identical_index(saved_vectors, tmp_path, kind, n_lists):
    _, _, base_path, queries_path = saved_vectors

    built = build_phase(kind, n_lists, base_path, queries_path, str(tmp_path / f"{kind}.idx"))

    assert built["save_load_identical"] is True
    assert built["index_space_bytes"] > 0
    assert built["save_time_s"] >= 0 and built["load_time_s"] >= 0


# Consultas y filas

def test_ivf_probes_never_exceed_the_number_of_lists():
    assert ivf_probes_for(1_000) == (1, 2, 4, 8, 16, 32)
    assert max(ivf_probes_for(10_000)) <= 100


def test_query_phase_measures_python_ivf_only_with_the_engine_probe(saved_vectors, tmp_path):
    _, _, base_path, queries_path = saved_vectors
    index_path = str(tmp_path / "ivf.idx")
    build_phase("ivf", 17, base_path, queries_path, index_path)

    measured = query_phase("ivf", index_path, queries_path, [1, 4, 17], list(PYTHON_IVF_PROBES), 1)

    assert sorted(measured["runs"]) == list(PYTHON_IVF_PROBES)
    assert sorted(measured["control"]) == [1, 4, 17]
    assert measured["control"][17]["avg_candidates"] == 300


def test_rows_report_recall_and_the_engine_defaults(saved_vectors, tmp_path):
    base, queries, base_path, queries_path = saved_vectors
    labels, query_labels = np.zeros(len(base), dtype=int), np.zeros(len(queries), dtype=int)
    truth, _, _ = exact_baseline(base, queries, METRIC, 1)
    common = {"modality": "image", "dim": 128, "n": len(base)}

    rows = []
    for kind, n_lists, values in [("hnsw", "", list(HNSW_EFS)), ("ivf", 17, [1, 4, 17])]:
        index_path = str(tmp_path / f"{kind}.idx")
        built = build_phase(kind, n_lists or 0, base_path, queries_path, index_path)
        measured = query_phase(kind, index_path, queries_path, values, list(PYTHON_IVF_PROBES), 1)
        rows += index_rows(kind, n_lists, built, measured, common, truth, labels, query_labels)

    assert all(set(row) == set(FIELDNAMES) for row in rows)
    kinds = [row["index_kind"] for row in rows]
    assert kinds.count("hnsw") == len(HNSW_EFS)
    assert kinds.count("ivf") == len(PYTHON_IVF_PROBES)
    assert kinds.count("ivf_numpy") == 3
    assert [row["ef_search"] for row in rows if row["is_engine_default"]] == [ENGINE_HNSW_EF]
    probing_everything = next(r for r in rows if r["index_kind"] == "ivf_numpy" and r["n_probe"] == 17)
    assert probing_everything["recall_at_k"] == 1.0


def test_write_rows_replaces_the_same_key_and_keeps_the_others(tmp_path):
    path = tmp_path / "dims.csv"
    empty = {field: "" for field in FIELDNAMES}
    first = {**empty, "modality": "image", "dim": 64, "n": 1000, "index_kind": "hnsw", "ef_search": 50, "recall_at_k": 0.9}
    other = {**first, "dim": 128}
    write_rows([first, other], path)

    write_rows([{**first, "recall_at_k": 0.95}], path)

    rows = read_rows(path)
    assert sorted((row["dim"], row["recall_at_k"]) for row in rows) == [("128", "0.9"), ("64", "0.95")]
    assert measured_groups(path) == {("image", "64", "1000", "hnsw"), ("image", "128", "1000", "hnsw")}


# Verificación del CSV

def csv_row(index_kind, dim=64, **values):
    """Una fila como las lee read_rows: todo texto."""
    row = {field: "" for field in FIELDNAMES}
    row.update(modality="image", dim=dim, n=1000, k=10, num_queries=200, index_kind=index_kind,
               recall_at_k=1.0, precision_at_k=0.1, save_load_identical=index_kind != "exact",
               index_space_bytes=dim * 1000)
    row.update(values)
    return {key: str(value) for key, value in row.items()}


def valid_rows():
    rows = []
    for dim in (64, 128):
        rows += [csv_row("exact", dim), csv_row("hnsw", dim, ef_search=50, recall_at_k=0.9)]
        for probe, recall in ((1, 0.5), (4, 0.8)):
            rows.append(csv_row("ivf_numpy", dim, n_probe=probe, recall_at_k=recall, avg_candidates=probe * 10))
        rows.append(csv_row("ivf", dim, n_probe=4, recall_at_k=0.8, avg_candidates=40))
    return rows


def test_valid_results_pass_every_check():
    assert check(valid_rows()) == []


@pytest.mark.parametrize("change, expected", [
    ({"index_kind": "hnsw", "save_load_identical": "False"}, "no devolvió lo mismo"),
    ({"index_kind": "ivf", "recall_at_k": "0.7"}, "distinto entre ivf e ivf_numpy"),
    ({"index_kind": "exact", "recall_at_k": "0.9"}, "la fuerza bruta tiene recall"),
    ({"index_kind": "hnsw", "num_queries": "100"}, "consultas"),
])
def test_each_check_reports_its_problem(change, expected):
    rows = valid_rows()
    target = next(row for row in rows if row["index_kind"] == change["index_kind"])
    target.update(change)

    assert any(expected in error for error in check(rows))


def test_recall_that_drops_with_more_probes_is_reported():
    rows = valid_rows()
    next(row for row in rows if row["index_kind"] == "ivf_numpy" and row["n_probe"] == "4")["recall_at_k"] = "0.4"

    assert any("el recall baja" in error for error in check(rows))


def test_an_index_that_does_not_grow_with_the_dimension_is_reported():
    rows = valid_rows()
    for row in rows:
        if row["dim"] == "128" and row["index_kind"] == "hnsw":
            row["index_space_bytes"] = "64000"

    assert any("no crece con la dimensión" in error for error in check(rows))
