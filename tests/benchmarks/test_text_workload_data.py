"""
Guarda de los archivos congelados que se versionan en benchmarks/workload/:
las consultas y las etiquetas de relevancia (qrels) de la comparación de texto.

Sin ellas no se pueden calcular Precision@10 ni Recall@10, y sin que sean las
mismas para todos los motores la comparación no vale. Estas pruebas fallan si:

  * los archivos no están versionados (o git los alteró: CRLF),
  * las consultas y los qrels no son coherentes entre sí,
  * alguna consulta se queda sin documentos relevantes con algún N,
  * faltan consultas de 1, 3 o 5 o más términos.

Se generan con:  python benchmarks/text_workload.py --corpus spanish
La regeneración completa contra el corpus (descarga 261 MB y tarda ~1 minuto)
es opcional: TEXT_WORKLOAD_FULL_CHECK=1 pytest tests/benchmarks/test_text_workload_data.py
"""

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from text_common import N_VALUES, QUERIES_PER_GROUP, QUERY_GROUPS, SEED  # noqa: E402
from text_manifest import (  # noqa: E402
    FORMAT_VERSION, qrels_path, queries_path, read_manifest, validate_workload_files,
)
from text_qrels import read_qrels, relevant_ids  # noqa: E402

# Corpora whose frozen workload is versioned in the repository.
WORKLOAD_CORPORA = ["spanish"]
GENERATE = "python benchmarks/text_workload.py --corpus {corpus}"


@pytest.fixture(params=WORKLOAD_CORPORA)
def corpus(request):
    return request.param


@pytest.fixture
def manifest(corpus):
    return read_manifest(queries_path(corpus))


def test_the_frozen_files_are_versioned(corpus):
    missing = [path for path in (queries_path(corpus), qrels_path(corpus)) if not path.exists()]
    assert not missing, (
        f"faltan {[p.name for p in missing]} en benchmarks/workload/. "
        f"Genera con: {GENERATE.format(corpus=corpus)} y súbelos al repositorio"
    )


def test_the_files_are_consistent_with_each_other(corpus):
    assert validate_workload_files(queries_path(corpus), qrels_path(corpus)) == []


def test_every_query_has_relevant_documents_labelled_at_every_n(corpus, manifest):
    """Precision@10 and Recall@10 need |R(q)| > 0 for every query and every N."""
    for n in N_VALUES:
        qrels = read_qrels(qrels_path(corpus), max_doc_id=n)
        for query in manifest["queries"]:
            relevant = relevant_ids(qrels, query["query_id"])  # KeyError if the query has no labels
            assert relevant, f"{query['query_id']} no tiene relevantes con N={n:,}"
            assert query["seed_doc_id"] in relevant


def test_there_are_queries_of_one_three_and_five_or_more_terms(manifest):
    by_group = {group: [q for q in manifest["queries"] if q["query_group"] == group] for group in QUERY_GROUPS}

    assert {group: len(queries) for group, queries in by_group.items()} == {g: QUERIES_PER_GROUP for g in QUERY_GROUPS}
    assert {q["query_terms"] for q in by_group["1"]} == {1}
    assert {q["query_terms"] for q in by_group["3"]} == {3}
    assert all(5 <= q["query_terms"] <= 8 for q in by_group["5+"])


def test_the_workload_covers_the_three_corpus_sizes_with_the_fixed_seeds(manifest):
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["parameters"]["n_values"] == N_VALUES == [1_000, 10_000, 100_000]
    assert manifest["seeds"] == {"documents": SEED, "queries": SEED + 1}
    assert manifest["qrels"]["max_doc_id"] == max(N_VALUES)


def test_the_labels_are_binary(corpus):
    qrels = read_qrels(qrels_path(corpus))
    assert {grade for labels in qrels.values() for grade in labels.values()} == {1}


def test_the_manifest_pins_the_corpus_and_every_dataset(corpus, manifest):
    sha256 = re.compile(r"^[0-9a-f]{64}$")

    assert sha256.match(manifest["corpus_source"]["sha256"])  # the corpus file is pinned by hash
    assert sha256.match(manifest["tokenizer"]["stopwords_sha256"])
    assert set(manifest["datasets"]) == {str(n) for n in N_VALUES}
    for n, info in manifest["datasets"].items():
        assert sha256.match(info["sha256"]) and info["documents"] == int(n)
        assert 1 <= info["distinct_documents"] <= info["documents"]


def test_the_files_have_lf_line_endings(corpus):
    for path in (queries_path(corpus), qrels_path(corpus)):
        assert b"\r" not in path.read_bytes(), f"{path.name} tiene CRLF: revisa .gitattributes"


def test_gitattributes_keeps_git_from_rewriting_the_frozen_files():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "benchmarks/workload/*.tsv" in attributes and "benchmarks/workload/*.json" in attributes
    assert attributes.count("eol=lf") >= 2


@pytest.mark.skipif(os.environ.get("TEXT_WORKLOAD_FULL_CHECK") != "1",
                    reason="regeneración completa contra el corpus: TEXT_WORKLOAD_FULL_CHECK=1")
def test_regenerating_from_the_corpus_gives_the_versioned_files(corpus):
    import text_workload
    assert text_workload.check_workload(corpus) == []