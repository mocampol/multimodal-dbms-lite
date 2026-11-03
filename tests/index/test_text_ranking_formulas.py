"""
Ranking algorithms of the inverted index: Okapi BM25 and TF-IDF with cosine
similarity (docs/text_search_policy.md sections 7 and 8).

The corpus is indexed from explicit token lists, so these tests exercise the
scoring math in isolation (no TextProcessor, no catalog, no SQL). Expected
values come from two independent sources: arithmetic written out by hand
from the policy formulas, and small reference implementations (oracles)
that work on the raw documents without touching the index structures.
"""

import math
from collections import Counter

import pytest

from index.text.inverted_index import BM25_B, BM25_K1, InvertedIndex

CORPUS = {
    1: ["bases", "datos", "bases"],
    2: ["datos", "redes"],
    3: ["redes", "sistemas", "operativos", "redes"],
    4: ["cocina"],
}


def build(corpus=None) -> InvertedIndex:
    index = InvertedIndex()
    for doc_id, tokens in (CORPUS if corpus is None else corpus).items():
        index.add_document(doc_id, tokens)
    return index


# --------------------------------------------------------------------------
# Oracles: policy formulas applied directly to the raw documents
# --------------------------------------------------------------------------

def oracle_bm25(corpus, query, k1=BM25_K1, b=BM25_B):
    n = len(corpus)
    avgdl = sum(len(tokens) for tokens in corpus.values()) / n
    scores = {}
    for term in dict.fromkeys(query):  # distinct query terms (policy section 7)
        df = sum(1 for tokens in corpus.values() if term in tokens)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for doc_id, tokens in corpus.items():
            tf = tokens.count(term)
            if tf:
                norm = k1 * (1 - b + b * len(tokens) / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + norm)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def oracle_tfidf(corpus, query):
    n = len(corpus)

    def idf(term):
        df = sum(1 for tokens in corpus.values() if term in tokens)
        return math.log(1 + n / df) if df else 0.0

    query_weights = {
        term: count * idf(term)
        for term, count in Counter(query).items()
        if idf(term) > 0
    }
    if not query_weights:
        return []
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

    scores = {}
    for doc_id, tokens in corpus.items():
        counts = Counter(tokens)
        doc_norm = math.sqrt(sum((c * idf(t)) ** 2 for t, c in counts.items()))
        if doc_norm == 0:
            continue
        dot = sum(w * counts[t] * idf(t) for t, w in query_weights.items() if t in counts)
        if dot > 0:
            scores[doc_id] = min(1.0, dot / (query_norm * doc_norm))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def assert_same_ranking(actual, expected):
    assert [doc_id for doc_id, _ in actual] == [doc_id for doc_id, _ in expected]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected])


QUERIES = [
    ["bases"],
    ["datos"],
    ["redes", "datos"],
    ["bases", "datos", "redes"],
    ["cocina", "operativos", "sistemas", "bases", "redes"],
]


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def test_bm25_idf_is_the_non_negative_lucene_variant():
    index = build()
    # df(datos) = 2, N = 4  ->  ln(1 + (4 - 2 + 0.5) / (2 + 0.5)) = ln 2
    assert index.idf("datos") == pytest.approx(math.log(2))
    assert index.idf("inexistente") == 0.0

    everywhere = build({1: ["x", "a"], 2: ["x", "b"], 3: ["x", "c"]})
    assert everywhere.idf("x") > 0.0  # a term in every document never goes negative


def test_bm25_single_term_matches_hand_computed_value():
    index = build()
    # 'bases' only in doc 1: tf = 2, |D| = 3, N = 4, df = 1, avgdl = 10 / 4 = 2.5
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    tf_part = (2 * (1.2 + 1)) / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2.5))

    assert index.score_bm25(["bases"]) == [(1, pytest.approx(idf * tf_part))]


@pytest.mark.parametrize("query", QUERIES)
def test_bm25_matches_reference_implementation(query):
    index = build()
    assert_same_ranking(index.score_bm25(query, top_k=None), oracle_bm25(CORPUS, query))


def test_bm25_repeated_query_term_counts_once():
    index = build()
    assert index.score_bm25(["datos", "datos", "datos"]) == index.score_bm25(["datos"])


def test_bm25_unknown_terms_are_ignored_without_error():
    index = build()
    # policy section 6: df = 0 -> IDF 0, no penalty, no exception
    assert index.score_bm25(["bases", "zzz"]) == index.score_bm25(["bases"])
    assert index.score_bm25(["zzz", "yyy"]) == []


def test_bm25_empty_query_or_empty_index_returns_nothing():
    assert build().score_bm25([]) == []
    assert InvertedIndex().score_bm25(["bases"]) == []


def test_bm25_penalizes_longer_documents_for_the_same_term_frequency():
    index = build({1: ["gato", "x"], 2: ["gato", "x", "y", "z", "w", "v"], 3: ["otro"]})
    scores = dict(index.score_bm25(["gato"], top_k=None))
    assert scores[1] > scores[2]


def test_bm25_without_length_normalization_when_b_is_zero():
    index = build({1: ["gato", "x"], 2: ["gato", "x", "y", "z", "w", "v"], 3: ["otro"]})
    scores = dict(index.score_bm25(["gato"], b=0.0, top_k=None))
    assert scores[1] == pytest.approx(scores[2])


def test_bm25_term_frequency_saturates():
    # same length, increasing tf: score grows, but with diminishing returns
    docs = {i: ["gato"] * i + ["x"] * (6 - i) for i in (1, 2, 3)}
    docs[4] = ["otro"] * 6
    scores = dict(build(docs).score_bm25(["gato"], top_k=None))
    assert scores[1] < scores[2] < scores[3]
    assert scores[2] - scores[1] > scores[3] - scores[2]


def test_bm25_ties_break_by_ascending_doc_id():
    index = build({7: ["gato", "x"], 3: ["gato", "x"], 5: ["gato", "x"], 9: ["otro", "y"]})
    assert [doc_id for doc_id, _ in index.score_bm25(["gato"])] == [3, 5, 7]


def test_bm25_top_k_truncates_after_ranking():
    index = build()
    full = index.score_bm25(["redes", "datos", "bases"], top_k=None)
    assert len(full) == 3
    assert index.score_bm25(["redes", "datos", "bases"], top_k=2) == full[:2]
    assert index.score_bm25(["redes", "datos", "bases"], top_k=0) == []


# --------------------------------------------------------------------------
# TF-IDF + cosine similarity
# --------------------------------------------------------------------------

def test_tfidf_idf_is_the_smooth_variant():
    index = build()
    # df(datos) = 2, N = 4  ->  ln(1 + 4 / 2)
    assert index.idf_smooth("datos") == pytest.approx(math.log(3))
    assert index.idf_smooth("bases") == pytest.approx(math.log(1 + 4 / 1))
    assert index.idf_smooth("inexistente") == 0.0


def test_tfidf_single_term_matches_hand_computed_cosine():
    index = build()
    # query ['bases']: V(Q) = {bases: ln 5}; doc 1 = {bases: 2 ln 5, datos: ln 3}
    idf_bases, idf_datos = math.log(5), math.log(3)
    expected = (idf_bases * 2 * idf_bases) / (
        idf_bases * math.sqrt((2 * idf_bases) ** 2 + idf_datos ** 2)
    )

    assert index.score_tfidf_cosine(["bases"]) == [(1, pytest.approx(expected))]


@pytest.mark.parametrize("query", QUERIES)
def test_tfidf_matches_reference_implementation(query):
    index = build()
    assert_same_ranking(index.score_tfidf_cosine(query), oracle_tfidf(CORPUS, query))


def test_tfidf_scores_are_bounded_and_an_aligned_document_scores_one():
    index = build({1: ["bases", "datos"], 2: ["bases", "datos", "redes"], 3: ["cocina"]})
    ranked = index.score_tfidf_cosine(["bases", "datos"])

    assert all(0.0 <= score <= 1.0 for _, score in ranked)
    assert ranked[0] == (1, pytest.approx(1.0))  # same vector direction as the query
    assert ranked[1][0] == 2 and ranked[1][1] < 1.0


def test_tfidf_repeated_query_terms_weight_the_query_vector():
    index = build()
    # unlike BM25, the query vector uses tf(t, Q), so repeating a term changes the ranking scores
    once = dict(index.score_tfidf_cosine(["bases", "datos"]))
    twice = dict(index.score_tfidf_cosine(["bases", "bases", "datos"]))
    assert once[1] != pytest.approx(twice[1])


def test_tfidf_empty_or_unknown_queries_return_nothing_without_error():
    index = build()
    assert index.score_tfidf_cosine([]) == []
    assert index.score_tfidf_cosine(["zzz", "yyy"]) == []
    assert InvertedIndex().score_tfidf_cosine(["bases"]) == []


def test_tfidf_ignores_unknown_terms_in_a_mixed_query():
    index = build()
    assert index.score_tfidf_cosine(["bases", "zzz"]) == index.score_tfidf_cosine(["bases"])


def test_tfidf_skips_documents_without_indexable_terms():
    index = build()
    index.add_document(99, [])  # e.g. a document made only of stopwords
    ranked = index.score_tfidf_cosine(["bases", "datos"])  # must not divide by zero
    assert 99 not in [doc_id for doc_id, _ in ranked]


def test_tfidf_ties_break_by_ascending_doc_id():
    index = build({7: ["gato", "x"], 3: ["gato", "x"], 5: ["gato", "x"], 9: ["otro", "y"]})
    assert [doc_id for doc_id, _ in index.score_tfidf_cosine(["gato"])] == [3, 5, 7]


def test_tfidf_top_k_truncates_after_ranking():
    index = build()
    full = index.score_tfidf_cosine(["redes", "datos", "bases"])
    assert index.score_tfidf_cosine(["redes", "datos", "bases"], top_k=2) == full[:2]


def test_cached_document_norms_are_invalidated_by_add_and_remove():
    index = build()
    index.score_tfidf_cosine(["bases", "datos"])  # fills the norm cache

    index.add_document(5, ["bases", "datos", "bases", "datos"])
    extended = dict(CORPUS) | {5: ["bases", "datos", "bases", "datos"]}
    assert_same_ranking(
        index.score_tfidf_cosine(["bases", "datos"]),
        oracle_tfidf(extended, ["bases", "datos"]),
    )

    index.remove_document(5)
    assert_same_ranking(
        index.score_tfidf_cosine(["bases", "datos"]),
        oracle_tfidf(CORPUS, ["bases", "datos"]),
    )


# --------------------------------------------------------------------------
# rank(): algorithm selection
# --------------------------------------------------------------------------

def test_rank_dispatches_to_the_named_algorithm():
    index = build()
    query = ["redes", "datos"]
    assert index.rank(query, "bm25") == index.score_bm25(query, top_k=None)
    assert index.rank(query, "tf_idf") == index.score_tfidf_cosine(query)
    assert index.rank(query) == index.rank(query, "bm25")  # BM25 is the default


@pytest.mark.parametrize("name", ["coseno", "BM25", "tfidf", "", "bm 25"])
def test_rank_rejects_unsupported_algorithms(name):
    with pytest.raises(ValueError, match="no soportado"):
        build().rank(["redes"], name)


def test_algorithms_agree_on_candidates_but_not_on_scores():
    index = build()
    query = ["redes", "datos", "bases"]
    bm25 = dict(index.rank(query, "bm25"))
    tfidf = dict(index.rank(query, "tf_idf"))

    assert set(bm25) == set(tfidf) == {1, 2, 3}  # same candidate documents
    assert all(bm25[d] != pytest.approx(tfidf[d]) for d in bm25)  # different scales