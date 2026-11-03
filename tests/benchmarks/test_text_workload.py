"""
Generación reproducible de las consultas y los qrels de la comparación de texto
(benchmarks/text_workload.py), con el corpus sintético: determinista y sin
descargas. Cubre los criterios del benchmark:

  * la generación usa semillas fijas y es reproducible (mismos bytes),
  * hay consultas de 1, 3 y 5 o más términos,
  * cada consulta tiene documentos relevantes etiquetados, en todos los N,
  * un cambio en el corpus, el tokenizador, las semillas o las etiquetas se detecta.

Los relevantes esperados se recalculan aquí por fuerza bruta sobre los documentos
(sin postings), independientes del generador.
"""

import shutil
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import benchmark_text  # noqa: E402
import text_workload as workload  # noqa: E402
from text_common import (  # noqa: E402
    MAX_DOC_BYTES, QUERIES_PER_GROUP, QUERY_GROUPS, QUERY_POOL_DOCS, SEED, generate_queries,
    load_documents, tokenize,
)
from text_manifest import (  # noqa: E402
    file_sha256, qrels_path, queries_path, read_manifest, validate_workload_files, write_manifest,
)
from text_qrels import read_qrels, relevant_ids  # noqa: E402

CORPUS = "synthetic"
SIZES = [1_000, 3_000]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("workload")
    manifest = workload.build_workload(CORPUS, out_dir, SIZES)
    return out_dir, manifest


@pytest.fixture
def copy_of_built(built, tmp_path):
    """A private copy of the generated files, safe to tamper with."""
    shutil.copytree(built[0], tmp_path / "data")
    return tmp_path / "data"


def brute_force_relevant(terms, documents):
    """Documents that contain every term, found by scanning (no postings)."""
    return {doc_id for doc_id, text in documents if set(terms) <= set(tokenize(text))}


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_generation_is_reproducible_byte_for_byte(built, tmp_path):
    workload.build_workload(CORPUS, tmp_path, SIZES)

    for path_of in (queries_path, qrels_path):
        assert path_of(CORPUS, tmp_path).read_bytes() == path_of(CORPUS, built[0]).read_bytes()


def test_the_manifest_records_the_fixed_seeds_and_the_parameters(built):
    manifest = built[1]

    assert manifest["seeds"] == {"documents": SEED, "queries": SEED + 1}
    assert manifest["parameters"]["n_values"] == SIZES
    assert manifest["parameters"]["max_doc_bytes"] == MAX_DOC_BYTES
    assert manifest["parameters"]["query_pool_docs"] == QUERY_POOL_DOCS
    assert manifest["corpus_source"] == {"generator": CORPUS, "seed": SEED}


def test_the_frozen_queries_are_the_ones_generate_queries_produces(built):
    expected = generate_queries(CORPUS)
    frozen = built[1]["queries"]

    assert [{key: query[key] for key in expected[0]} for query in frozen] == expected


def test_the_manifest_has_no_machine_specific_values(built):
    text = queries_path(CORPUS, built[0]).read_text(encoding="utf-8")
    assert str(built[0]) not in text and "\\" not in text.replace("\\n", "")  # no absolute or Windows paths


def test_each_dataset_is_fingerprinted_with_its_own_digest(built):
    datasets = built[1]["datasets"]

    assert set(datasets) == {str(n) for n in SIZES}
    assert datasets["1000"]["documents"] == 1_000 and datasets["3000"]["documents"] == 3_000
    assert len({info["sha256"] for info in datasets.values()}) == len(SIZES)
    assert all(1 <= info["distinct_documents"] <= info["documents"] for info in datasets.values())


def test_the_corpus_file_is_fingerprinted_by_name_size_and_sha256(tmp_path, monkeypatch):
    csv_file = tmp_path / "data_larazon_publico_v2.csv"
    csv_file.write_bytes(b"id,texto\n1,hola\n")
    monkeypatch.setattr(benchmark_text, "CORPUS_DIR", tmp_path)

    source = workload.corpus_source("spanish")

    assert source == {"file": "data_larazon_publico_v2.csv", "size_bytes": 16, "sha256": file_sha256(csv_file)}
    assert str(tmp_path) not in str(source)  # the name only, never the machine's path


def test_a_missing_real_corpus_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_text, "CORPUS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="spanish"):
        workload.corpus_source("spanish")


# --------------------------------------------------------------------------
# Queries of 1, 3 and 5 or more terms
# --------------------------------------------------------------------------

def test_there_are_queries_of_one_three_and_five_or_more_terms(built):
    queries = built[1]["queries"]

    assert len(queries) == QUERIES_PER_GROUP * len(QUERY_GROUPS)
    for group, (low, high) in QUERY_GROUPS.items():
        members = [query for query in queries if query["query_group"] == group]
        assert len(members) == QUERIES_PER_GROUP
        assert all(low <= query["query_terms"] <= high == high for query in members)
    assert {query["query_terms"] for query in queries if query["query_group"] == "1"} == {1}
    assert {query["query_terms"] for query in queries if query["query_group"] == "3"} == {3}
    assert min(query["query_terms"] for query in queries if query["query_group"] == "5+") >= 5


def test_query_terms_are_already_normalized(built):
    for query in built[1]["queries"]:
        assert tokenize(query["text"]) == query["terms"]  # re-tokenizing the text gives the same terms


# --------------------------------------------------------------------------
# Labelled relevant documents
# --------------------------------------------------------------------------

def test_the_qrels_are_exactly_the_documents_containing_all_the_terms(built):
    out_dir, manifest = built
    qrels = read_qrels(qrels_path(CORPUS, out_dir))
    documents = load_documents(CORPUS, SIZES[-1])

    for query in manifest["queries"]:
        expected = brute_force_relevant(query["terms"], documents)
        assert relevant_ids(qrels, query["query_id"]) == expected, query["query_id"]


def test_qrels_of_the_largest_n_filtered_by_id_are_the_qrels_of_the_smaller_n(built):
    out_dir, manifest = built
    qrels_small = read_qrels(qrels_path(CORPUS, out_dir), max_doc_id=SIZES[0])
    documents = load_documents(CORPUS, SIZES[0])

    for query in manifest["queries"]:
        expected = brute_force_relevant(query["terms"], documents)
        assert relevant_ids(qrels_small, query["query_id"]) == expected, query["query_id"]


def test_every_query_has_relevant_documents_at_every_n(built):
    out_dir, manifest = built
    for n in SIZES:
        qrels = read_qrels(qrels_path(CORPUS, out_dir), max_doc_id=n)
        for query in manifest["queries"]:
            relevant = relevant_ids(qrels, query["query_id"])
            assert relevant, f"{query['query_id']} sin relevantes con N={n}"  # Recall@k needs |R| > 0
            assert query["seed_doc_id"] in relevant
            assert len(relevant) == query["relevant_docs"][str(n)]


def test_the_generated_files_pass_their_own_validation(built):
    out_dir, manifest = built
    assert validate_workload_files(queries_path(CORPUS, out_dir), qrels_path(CORPUS, out_dir)) == []
    assert manifest["qrels"]["lines"] == len(qrels_path(CORPUS, out_dir).read_text().splitlines())
    assert manifest["qrels"]["sha256"] == file_sha256(qrels_path(CORPUS, out_dir))


def test_a_smallest_n_below_the_query_pool_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"debe ser >= 1,000"):
        workload.build_workload(CORPUS, tmp_path, [500, 1_000])


def test_documents_that_are_not_a_prefix_of_the_largest_n_are_rejected(tmp_path, monkeypatch):
    original = workload.load_documents

    def not_a_prefix(corpus, n):
        documents = original(corpus, n)
        if n == SIZES[-1]:
            documents[10] = (documents[10][0], "texto distinto")
        return documents

    monkeypatch.setattr(workload, "load_documents", not_a_prefix)
    with pytest.raises(ValueError, match="no son un prefijo"):
        workload.build_workload(CORPUS, tmp_path, SIZES)


# --------------------------------------------------------------------------
# --check detects any drift
# --------------------------------------------------------------------------

def check(data_dir):
    return workload.check_workload(CORPUS, data_dir, SIZES)


def test_check_passes_on_freshly_generated_files(built):
    assert check(built[0]) == []


def test_check_reports_missing_files(tmp_path):
    assert any("falta el archivo" in problem for problem in check(tmp_path))


def test_check_detects_a_changed_corpus(copy_of_built, monkeypatch):
    original = workload.load_documents

    def altered(corpus, n):
        documents = original(corpus, n)
        documents[4] = (documents[4][0], documents[4][1] + " palabranueva")
        return documents

    monkeypatch.setattr(workload, "load_documents", altered)
    problems = check(copy_of_built)

    assert any("los documentos de N=1,000 son distintos" in problem for problem in problems), problems


def test_check_detects_a_changed_tokenizer(copy_of_built, monkeypatch):
    real = workload.tokenizer_fingerprint
    monkeypatch.setattr(workload, "tokenizer_fingerprint", lambda p: {**real(p), "stopwords_sha256": "otra-lista"})

    assert any("el tokenizador cambió" in problem for problem in check(copy_of_built))


def test_check_detects_changed_seeds(copy_of_built, monkeypatch):
    monkeypatch.setattr(workload, "SEED", SEED + 1)
    assert any("semillas distintas" in problem for problem in check(copy_of_built))


def test_check_detects_a_tampered_label_file(copy_of_built):
    path = qrels_path(CORPUS, copy_of_built)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_bytes("".join(lines[1:]).encode("utf-8"))  # one label removed

    problems = check(copy_of_built)
    assert any("SHA-256 del archivo es distinto" in problem for problem in problems), problems
    assert any("el manifiesto declara" in problem for problem in problems), problems


def test_check_detects_labels_forged_consistently_with_the_manifest(copy_of_built):
    """Dropping a label AND fixing the manifest's hash and counts passes the
    internal validation; only regenerating from the corpus exposes it."""
    qrels_file, queries_file = qrels_path(CORPUS, copy_of_built), queries_path(CORPUS, copy_of_built)
    qrels = read_qrels(qrels_file)
    first = read_manifest(queries_file)["queries"][0]["query_id"]
    dropped = max(qrels[first])
    del qrels[first][dropped]
    from text_qrels import write_qrels
    write_qrels(qrels_file, qrels)

    manifest = read_manifest(queries_file)
    manifest["qrels"].update(sha256=file_sha256(qrels_file), lines=manifest["qrels"]["lines"] - 1)
    for n in SIZES:
        if dropped <= n:
            manifest["queries"][0]["relevant_docs"][str(n)] -= 1
    write_manifest(queries_file, manifest)

    assert validate_workload_files(queries_file, qrels_file) == []  # internally consistent...
    problems = check(copy_of_built)                                  # ...but not what the corpus yields
    assert any("los qrels regenerados son distintos" in problem for problem in problems), problems
    assert any(first in problem and "cambió" in problem for problem in problems), problems


def test_check_detects_an_edited_query(copy_of_built):
    path = queries_path(CORPUS, copy_of_built)
    manifest = read_manifest(path)
    manifest["queries"][0]["text"] = "otra consulta"
    write_manifest(path, manifest)

    assert any(manifest["queries"][0]["query_id"] in problem and "cambió" in problem for problem in check(copy_of_built))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def test_cli_generates_then_checks_then_fails_after_tampering(tmp_path, capsys):
    args = ["--corpus", CORPUS, "--sizes", "1000,3000", "--out-dir", str(tmp_path)]

    assert workload.main(args) == 0
    assert "30 consultas" in capsys.readouterr().out
    assert workload.main(args + ["--check"]) == 0
    assert "✓" in capsys.readouterr().out

    path = qrels_path(CORPUS, tmp_path)
    path.write_bytes(path.read_bytes() + b"G1-01 0 999999 1\n")
    assert workload.main(args + ["--check"]) == 1
    assert "✗" in capsys.readouterr().out