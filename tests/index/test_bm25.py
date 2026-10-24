"""
BM25 over the inverted index, checked against the contract in
docs/text_search_policy.md §5-7 (k1=1.2, b=0.75, non-negative IDF,
OR semantics, unknown terms add 0, ties broken by doc_id ascending).
"""

import math

import pytest

from index.text import InvertedIndex, SPIMIIndexer
from index.text.inverted_index import BM25_B, BM25_K1


# N = 4, avgdl = (3 + 2 + 1 + 0) / 4 = 1.5
CORPUS = {
    1: ["bases", "datos", "bases"],
    2: ["sistemas", "operativos"],
    3: ["datos"],
    4: [],  # document with no indexable terms
}


def build_index(corpus=CORPUS):
    index = InvertedIndex()
    for doc_id, tokens in corpus.items():
        index.add_document(doc_id, tokens)
    return index


def contract_bm25(tf, doc_len, df, n, avgdl, k1=1.2, b=0.75):
    """Okapi BM25 term contribution written straight from the contract."""
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))


# Puntajes y parámetros

def test_parameters_match_the_contract():
    assert (BM25_K1, BM25_B) == (1.2, 0.75)
    index = build_index()
    query = ["bases", "datos"]
    expected = index.score_bm25(query, k1=1.2, b=0.75)
    assert index.score_bm25(query) == expected
    assert index.bm25_search(query) == expected


def test_single_term_score_computed_by_hand():
    # idf(bases) = ln(1 + 3.5/1.5); tf=2, |D|=3
    assert build_index().score_bm25(["bases"]) == [(1, pytest.approx(1.2920683753741757))]


def test_multi_term_score_is_the_sum_over_query_terms():
    results = dict(build_index().score_bm25(["bases", "datos"]))
    assert results[1] == pytest.approx(1.7839792777070402)
    assert results[3] == pytest.approx(0.8025914722273051)
    assert results[1] == pytest.approx(
        contract_bm25(tf=2, doc_len=3, df=1, n=4, avgdl=1.5)
        + contract_bm25(tf=1, doc_len=3, df=2, n=4, avgdl=1.5)
    )


def test_shorter_document_scores_higher_for_same_tf():
    results = dict(build_index().score_bm25(["datos"]))
    assert results[3] > results[1] > 0  # same tf, doc 3 is shorter


def test_b_zero_disables_length_normalization():
    results = dict(build_index().score_bm25(["datos"], b=0.0))
    assert results[1] == pytest.approx(results[3])


def test_scores_are_non_negative_even_for_very_common_terms():
    index = build_index({1: ["comun"], 2: ["comun"], 3: ["comun", "raro"]})
    assert all(score > 0 for _, score in index.score_bm25(["comun"]))


def test_repeated_query_terms_count_once():
    index = build_index()
    assert index.score_bm25(["datos", "datos"]) == index.score_bm25(["datos"])
    assert index.score_bm25(["bases", "datos", "bases"]) == index.score_bm25(["bases", "datos"])


# Documentos vacíos

def test_empty_documents_count_in_corpus_statistics():
    index = build_index()
    assert index.total_docs == 4
    assert index.avg_doc_length == pytest.approx(1.5)


def test_empty_documents_are_never_candidates():
    doc_ids = [doc_id for doc_id, _ in build_index().score_bm25(["bases", "datos", "sistemas"], top_k=None)]
    assert 4 not in doc_ids


def test_corpus_of_only_empty_documents_returns_nothing():
    assert build_index({1: [], 2: []}).score_bm25(["bases"]) == []


def test_empty_index_and_empty_query_return_nothing():
    assert InvertedIndex().score_bm25(["bases"]) == []
    assert build_index().score_bm25([]) == []


# Términos desconocidos

def test_unknown_term_has_zero_idf():
    assert build_index().idf("desconocido") == 0.0


def test_unknown_terms_neither_add_nor_penalize():
    index = build_index()
    assert index.score_bm25(["bases", "desconocido"]) == index.score_bm25(["bases"])


def test_query_with_only_unknown_terms_returns_nothing():
    assert build_index().score_bm25(["desconocido", "inexistente"]) == []


# Desempates y top_k

def test_ties_are_broken_by_doc_id_ascending():
    # inserted in reverse order on purpose
    index = build_index({9: ["empate"], 5: ["empate"], 7: ["empate"], 1: ["otro"]})
    results = index.score_bm25(["empate"])
    assert [doc_id for doc_id, _ in results] == [5, 7, 9]
    assert len({score for _, score in results}) == 1


def test_results_are_sorted_by_score_descending():
    scores = [score for _, score in build_index().score_bm25(["bases", "datos"])]
    assert scores == sorted(scores, reverse=True)


def test_top_k_cuts_after_ranking_and_none_returns_every_candidate():
    index = build_index()
    assert index.score_bm25(["bases", "datos"], top_k=1) == index.score_bm25(["bases", "datos"])[:1]
    # OR semantics: every document with at least one term, and only those
    wide = build_index({doc_id: ["t"] for doc_id in range(1, 16)})
    assert len(wide.score_bm25(["t"])) == 10          # default top_k
    assert len(wide.score_bm25(["t"], top_k=None)) == 15
    assert {doc_id for doc_id, _ in index.score_bm25(["datos", "sistemas"], top_k=None)} == {1, 2, 3}


# Reutiliza el índice construido por SPIMI

def test_spimi_built_index_gives_the_same_scores():
    corpus = {doc_id: " ".join(tokens) for doc_id, tokens in CORPUS.items()}
    spimi_index = SPIMIIndexer(max_records_per_block=2).index_corpus(corpus.items())
    query = ["bases", "datos", "sistemas"]
    assert spimi_index.score_bm25(query, top_k=None) == build_index().score_bm25(query, top_k=None)
