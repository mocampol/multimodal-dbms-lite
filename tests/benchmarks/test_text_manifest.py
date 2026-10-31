"""
Huellas y manifiesto de la carga de trabajo de texto (benchmarks/text_manifest.py):
huellas de documentos y tokenizador, manifiesto determinista y validación de que
las consultas y los qrels guardados sean coherentes y permitan calcular
Precision@k / Recall@k. No necesita el corpus.
"""

import hashlib
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from index.text.text_processor import TextProcessor  # noqa: E402
from text_manifest import (  # noqa: E402
    FORMAT_VERSION,
    dataset_digest,
    file_sha256,
    read_manifest,
    tokenizer_fingerprint,
    validate_workload_files,
    write_manifest,
)
from text_qrels import write_qrels  # noqa: E402


# --------------------------------------------------------------------------
# Huellas
# --------------------------------------------------------------------------

DOCS = [(1, "el gato come"), (2, "un perro ladra"), (3, "")]


def test_dataset_digest_is_deterministic_and_order_sensitive():
    assert dataset_digest(DOCS) == dataset_digest(list(DOCS))
    assert dataset_digest(DOCS) != dataset_digest(list(reversed(DOCS)))


@pytest.mark.parametrize("other", [
    [(1, "el gato come"), (2, "un perro ladra"), (3, " ")],          # a text changed
    [(1, "el gato come"), (2, "un perro ladra"), (4, "")],           # an id changed
    [(1, "el gato come"), (2, "un perro ladra")],                    # a document is missing
    [(1, "el gato come"), (2, "un perro ladra"), (3, ""), (4, "x")],  # a document was added
])
def test_dataset_digest_detects_any_change(other):
    assert dataset_digest(DOCS) != dataset_digest(other)


def test_dataset_digest_framing_is_unambiguous():
    # without a separator these would both serialize as "12" + "3"
    assert dataset_digest([(1, "23")]) != dataset_digest([(12, "3")])
    assert dataset_digest([(1, "a"), (2, "b")]) != dataset_digest([(1, "a2"), (0, "b")])


def test_dataset_digest_handles_non_ascii_text_and_streams():
    assert dataset_digest([(1, "canción año")]) == dataset_digest(iter([(1, "canción año")]))
    assert len(dataset_digest([])) == 64  # SHA-256 hex, also for no documents


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    payload = bytes(range(256)) * 9000  # more than one read block
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_tokenizer_fingerprint_changes_with_any_rule():
    base = tokenizer_fingerprint(TextProcessor(stopwords=["de", "la"]))

    assert base == tokenizer_fingerprint(TextProcessor(stopwords=["la", "de"]))  # order is irrelevant
    assert base["stopwords"] == 2
    assert base != tokenizer_fingerprint(TextProcessor(stopwords=["de", "la", "el"]))
    assert base != tokenizer_fingerprint(TextProcessor(stopwords=["de", "la"], lowercase=False))
    assert base != tokenizer_fingerprint(TextProcessor(stopwords=["de", "la"], strip_accents=False))
    assert base != tokenizer_fingerprint(TextProcessor(stopwords=["de", "la"], stemmer=str.upper))
    assert tokenizer_fingerprint(TextProcessor())["stopwords"] == 0


# --------------------------------------------------------------------------
# Manifiesto
# --------------------------------------------------------------------------

def test_manifest_round_trip_with_non_ascii_text_and_key_order(tmp_path):
    manifest = {"format_version": 1, "corpus": "spanish", "z": "canción", "a": [1, 2, {"b": None}]}
    path = write_manifest(tmp_path / "nested" / "queries.json", manifest)

    assert read_manifest(path) == manifest
    assert list(read_manifest(path)) == list(manifest)  # insertion order is kept
    assert "canción" in path.read_text(encoding="utf-8")  # readable, not \u escaped


def test_manifest_bytes_are_deterministic_lf_utf8_with_a_final_newline(tmp_path):
    manifest = {"corpus": "spanish", "queries": [{"text": "año"}]}
    first = write_manifest(tmp_path / "a.json", manifest).read_bytes()
    second = write_manifest(tmp_path / "b.json", dict(manifest)).read_bytes()

    assert first == second
    assert b"\r" not in first and first.endswith(b"}\n")


# --------------------------------------------------------------------------
# validate_workload_files
# --------------------------------------------------------------------------

N_VALUES = [10, 20]
QUERIES = [  # (query_id, group, terms)
    ("G1-01", "1", ["a"]),
    ("G1-02", "1", ["b"]),
    ("G3-01", "3", ["a", "b", "c"]),
    ("G3-02", "3", ["d", "e", "f"]),
]
QRELS = {"G1-01": {1, 5, 15}, "G1-02": {2, 12}, "G3-01": {3}, "G3-02": {4, 14, 18}}


def count_up_to(relevant, n):
    return sum(1 for doc_id in relevant if doc_id <= n)


def write_pair(tmp_path, qrels=None, edit=None):
    """Writes a consistent manifest + qrels pair (optionally altered) and returns both paths."""
    qrels = {key: set(value) for key, value in (QRELS if qrels is None else qrels).items()}
    qrels_file = write_qrels(tmp_path / "qrels.tsv", qrels)
    manifest = {
        "format_version": FORMAT_VERSION,
        "parameters": {
            "n_values": N_VALUES,
            "queries_per_group": 2,
            "query_groups": {"1": [1, 1], "3": [3, 3]},
        },
        "qrels": {"sha256": file_sha256(qrels_file)},
        "queries": [
            {
                "query_id": query_id, "query_group": group, "query_terms": len(terms), "terms": terms,
                "text": " ".join(terms),
                "relevant_docs": {str(n): count_up_to(QRELS.get(query_id, ()), n) for n in N_VALUES},
            }
            for query_id, group, terms in QUERIES
        ],
    }
    if edit:
        edit(manifest)
    return write_manifest(tmp_path / "queries.json", manifest), qrels_file


def assert_problem(problems, fragment):
    assert any(fragment in problem for problem in problems), problems


def test_consistent_files_have_no_problems(tmp_path):
    assert validate_workload_files(*write_pair(tmp_path)) == []


def test_missing_files_are_reported(tmp_path):
    problems = validate_workload_files(tmp_path / "queries.json", tmp_path / "qrels.tsv")
    assert len(problems) == 2 and all("falta el archivo" in problem for problem in problems)


def test_qrels_edited_after_the_manifest_was_written_are_detected(tmp_path):
    queries_file, qrels_file = write_pair(tmp_path)
    with open(qrels_file, "a", encoding="utf-8", newline="\n") as f:
        f.write("G3-01 0 19 1\n")
    assert_problem(validate_workload_files(queries_file, qrels_file), "SHA-256 del archivo es distinto")


def test_a_query_without_labels_cannot_be_evaluated(tmp_path):
    qrels = {key: value for key, value in QRELS.items() if key != "G3-02"}
    problems = validate_workload_files(*write_pair(tmp_path, qrels))
    assert_problem(problems, "G3-02: sin etiquetas")
    assert_problem(problems, "no se pueden calcular Precision@k ni Recall@k")


def test_a_query_with_no_relevant_document_at_some_n_is_reported(tmp_path):
    qrels = {**QRELS, "G1-02": {12, 15}}  # nothing within the first 10 documents
    problems = validate_workload_files(*write_pair(tmp_path, qrels))
    assert_problem(problems, "G1-02: ningún documento relevante con N=10")


def test_declared_relevant_counts_must_match_the_qrels(tmp_path):
    def edit(manifest):
        manifest["queries"][0]["relevant_docs"]["20"] = 99
    assert_problem(validate_workload_files(*write_pair(tmp_path, edit=edit)),
                   "declara 99 relevantes con N=20 y los qrels tienen 3")


def test_every_group_must_have_the_declared_number_of_queries(tmp_path):
    def edit(manifest):
        manifest["queries"].pop()
    assert_problem(validate_workload_files(*write_pair(tmp_path, edit=edit)),
                   "el grupo '3' tiene 1 consultas, se esperaban 2")


def test_term_counts_must_fit_their_group(tmp_path):
    def edit(manifest):
        manifest["queries"][2].update(terms=["a", "b"], query_terms=2, text="a b")
    assert_problem(validate_workload_files(*write_pair(tmp_path, edit=edit)),
                   "G3-01: 2 términos fuera del grupo '3' (3-3)")


def test_query_terms_must_equal_the_number_of_terms(tmp_path):
    def edit(manifest):
        manifest["queries"][0]["query_terms"] = 4
    assert_problem(validate_workload_files(*write_pair(tmp_path, edit=edit)),
                   "G1-01: query_terms=4 pero tiene 1 términos")


def test_duplicate_query_ids_are_reported(tmp_path):
    def edit(manifest):
        manifest["queries"][1]["query_id"] = "G1-01"
    assert_problem(validate_workload_files(*write_pair(tmp_path, edit=edit)), "query_id repetidos")


def test_qrels_for_an_unknown_query_are_reported(tmp_path):
    problems = validate_workload_files(*write_pair(tmp_path, {**QRELS, "G9-99": {1}}))
    assert_problem(problems, "G9-99, que no está en las consultas")


def test_an_unknown_format_version_is_reported(tmp_path):
    def edit(manifest):
        manifest["format_version"] = 99
    assert_problem(validate_workload_files(*write_pair(tmp_path, edit=edit)), "format_version 99")