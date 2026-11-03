"""
Formato de los qrels del benchmark de texto (benchmarks/text_qrels.py):
TREC "query_id 0 doc_id relevancia", escritura determinista, filtrado por N y
errores explícitos. No necesita el motor ni el corpus.
"""

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from text_qrels import read_qrels, relevant_ids, write_qrels  # noqa: E402


def write(tmp_path, qrels, name="qrels.tsv"):
    return write_qrels(tmp_path / name, qrels)


def lines(path):
    return path.read_bytes().decode("utf-8").splitlines()


# --------------------------------------------------------------------------
# Escritura
# --------------------------------------------------------------------------

def test_writes_trec_lines(tmp_path):
    path = write(tmp_path, {"G1-01": {7, 3}, "G3-01": {5}})
    assert path.read_bytes() == (
        b"G1-01 0 3 1\n"
        b"G1-01 0 7 1\n"
        b"G3-01 0 5 1\n"
    )


def test_a_set_of_ids_means_relevance_one_and_a_mapping_keeps_its_grades(tmp_path):
    path = write(tmp_path, {"A": {1, 2}, "B": {1: 2, 2: 0, 3: 1}})
    assert lines(path) == ["A 0 1 1", "A 0 2 1", "B 0 1 2", "B 0 2 0", "B 0 3 1"]


def test_output_is_ordered_and_documents_sort_numerically(tmp_path):
    shuffled = {"G5-02": [100, 2, 10], "G1-01": [30, 4]}
    path = write(tmp_path, shuffled)
    assert lines(path) == [
        "G1-01 0 4 1", "G1-01 0 30 1",
        "G5-02 0 2 1", "G5-02 0 10 1", "G5-02 0 100 1",  # 2 < 10 < 100, not "10" < "100" < "2"
    ]


def test_same_qrels_produce_identical_bytes_in_any_insertion_order(tmp_path):
    first = write(tmp_path, {"B": {2, 1}, "A": {9, 3}}, "first.tsv")
    second = write(tmp_path, {"A": {3, 9}, "B": {1, 2}}, "second.tsv")
    assert first.read_bytes() == second.read_bytes()


def test_file_uses_lf_line_endings_and_utf8_on_every_platform(tmp_path):
    data = write(tmp_path, {"A": {1, 2}}).read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")


def test_a_query_without_documents_leaves_no_lines_and_empty_qrels_write_an_empty_file(tmp_path):
    assert lines(write(tmp_path, {"A": set(), "B": {1}})) == ["B 0 1 1"]
    assert write(tmp_path, {}, "empty.tsv").read_bytes() == b""


def test_parent_directories_are_created(tmp_path):
    path = write_qrels(tmp_path / "data" / "nested" / "qrels.tsv", {"A": {1}})
    assert path.exists()


@pytest.mark.parametrize("query_id", ["", "G1 01", "G1\t01", " G1"])
def test_write_rejects_query_ids_that_would_break_the_format(tmp_path, query_id):
    with pytest.raises(ValueError, match="query_id inválido"):
        write(tmp_path, {query_id: {1}})


@pytest.mark.parametrize("doc_id", [0, -4, 1.5, "7", True])
def test_write_rejects_invalid_document_ids(tmp_path, doc_id):
    with pytest.raises(ValueError, match="doc_id inválido"):
        write(tmp_path, {"A": {doc_id: 1}})


@pytest.mark.parametrize("grade", [-1, 0.5, "1", True])
def test_write_rejects_invalid_relevance_grades(tmp_path, grade):
    with pytest.raises(ValueError, match="relevancia inválida"):
        write(tmp_path, {"A": {1: grade}})


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------

def test_round_trip(tmp_path):
    qrels = {"G1-01": {3: 1, 7: 1}, "G3-02": {5: 2, 6: 0}, "G5-01": {1: 1}}
    assert read_qrels(write(tmp_path, qrels)) == qrels


def test_read_accepts_crlf_blank_lines_and_extra_whitespace(tmp_path):
    path = tmp_path / "windows.tsv"
    path.write_bytes(b"A 0 1 1\r\n\r\nA   0\t2 1\r\n   \r\nB 0 5 1\r\n")
    assert read_qrels(path) == {"A": {1: 1, 2: 1}, "B": {5: 1}}


def test_max_doc_id_keeps_only_the_prefix_of_the_corpus(tmp_path):
    path = write(tmp_path, {"A": {5, 900, 5000}, "B": {2, 80}})

    assert read_qrels(path, max_doc_id=1000) == {"A": {5: 1, 900: 1}, "B": {2: 1, 80: 1}}
    assert read_qrels(path, max_doc_id=100) == {"A": {5: 1}, "B": {2: 1, 80: 1}}


def test_a_query_left_without_documents_by_the_filter_is_still_present(tmp_path):
    path = write(tmp_path, {"A": {5000}, "B": {1}})
    qrels = read_qrels(path, max_doc_id=1000)

    assert qrels == {"A": {}, "B": {1: 1}}
    assert relevant_ids(qrels, "A") == set()  # labelled, with no relevant document in this prefix


@pytest.mark.parametrize("content, message", [
    ("A 0 1\n", "se esperaban 4 campos"),
    ("A 0 1 1 extra\n", "se esperaban 4 campos"),
    ("A 0 uno 1\n", "deben ser enteros"),
    ("A 0 1 alto\n", "deben ser enteros"),
    ("A 0 0 1\n", "doc_id debe ser >= 1"),
    ("A 0 1 -1\n", "relevancia >= 0"),
    ("A 0 1 1\nA 0 1 1\n", "etiqueta dos veces el documento 1"),
    ("A 0 1 1\nA 0 1 0\n", "etiqueta dos veces el documento 1"),
])
def test_read_reports_malformed_lines_with_their_position(tmp_path, content, message):
    path = tmp_path / "bad.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        read_qrels(path)
    assert "bad.tsv:" in str(error.value)  # file and line number


def test_the_error_names_the_offending_line(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("A 0 1 1\nA 0 2 1\nA 0 x 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.tsv:3:"):
        read_qrels(path)


# --------------------------------------------------------------------------
# relevant_ids
# --------------------------------------------------------------------------

def test_relevant_ids_applies_the_minimum_grade():
    qrels = {"A": {1: 0, 2: 1, 3: 2}}
    assert relevant_ids(qrels, "A") == {2, 3}
    assert relevant_ids(qrels, "A", min_grade=2) == {3}
    assert relevant_ids(qrels, "A", min_grade=0) == {1, 2, 3}


def test_a_query_missing_from_the_qrels_is_an_error_not_an_empty_set():
    with pytest.raises(KeyError, match="G9-99"):
        relevant_ids({"A": {1: 1}}, "G9-99")