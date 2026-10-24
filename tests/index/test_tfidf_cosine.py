"""
Unit tests for TF-IDF Vector Space Model and Cosine Similarity in InvertedIndex.

Acceptance Criteria:
1. Exact numerical validation against hand-calculated micro-corpus (N=3).
2. Well-defined, error-free behavior for empty queries, non-indexable terms,
   stopwords-only queries, out-of-vocabulary terms, and empty/zero-norm documents.
3. Deterministic tie-breaking: scores descending, then doc_id ascending (-score, doc_id).
4. Full integration with SPIMI indexer and TextProcessor.
"""

from __future__ import annotations

import math
import pytest
from index.text.inverted_index import InvertedIndex
from index.text.spimi import SPIMIIndexer
from index.text.text_processor import SPANISH_STOPWORDS, TextProcessor


# ---------------------------------------------------------------------------
# Hand-Calculated Micro-Corpus Test Fixture (N = 3)
# ---------------------------------------------------------------------------
# Doc 1: "gato perro"
# Doc 2: "gato gato raton"
# Doc 3: "perro leon"
#
# Hand-calculated values:
# idf("gato") = ln(1 + 3/2) = ln(2.5) ≈ 0.91629073
# idf("perro") = ln(1 + 3/2) = ln(2.5) ≈ 0.91629073
# idf("raton") = ln(1 + 3/1) = ln(4.0) ≈ 1.38629436
# idf("leon") = ln(1 + 3/1) = ln(4.0) ≈ 1.38629436
#
# ||V(D1)|| = sqrt(ln(2.5)^2 + ln(2.5)^2) ≈ 1.29582998
# ||V(D2)|| = sqrt((2*ln(2.5))^2 + ln(4.0)^2) ≈ 2.29786137
# ||V(D3)|| = sqrt(ln(2.5)^2 + ln(4.0)^2) ≈ 1.66174630
#
# Query Q: "gato perro"
# ||V(Q)|| = sqrt(ln(2.5)^2 + ln(2.5)^2) ≈ 1.29582998
#
# sim(Q, D1) = 2*ln(2.5)^2 / (1.29583 * 1.29583) = 1.00000000
# sim(Q, D2) = 2*ln(2.5)^2 / (1.29583 * 2.29786) ≈ 0.56392943
# sim(Q, D3) = ln(2.5)^2 / (1.29583 * 1.66175) ≈ 0.38989938
# ---------------------------------------------------------------------------

@pytest.fixture
def micro_corpus_index() -> InvertedIndex:
    """Constructs the canonical 3-document test index."""
    indexer = SPIMIIndexer(processor=TextProcessor(stopwords=set()))
    corpus = [
        (1, "gato perro"),
        (2, "gato gato raton"),
        (3, "perro leon"),
    ]
    return indexer.index_corpus(corpus)


def test_hand_calculated_idf_and_doc_norms(micro_corpus_index: InvertedIndex):
    """Validates smooth IDF and document Euclidean norms against analytical calculation."""
    idx = micro_corpus_index
    assert idx.total_docs == 3

    # Check IDF: idf(t) = ln(1 + N / df)
    expected_idf_gato = math.log(1.0 + 3.0 / 2.0)
    expected_idf_perro = math.log(1.0 + 3.0 / 2.0)
    expected_idf_raton = math.log(1.0 + 3.0 / 1.0)
    expected_idf_leon = math.log(1.0 + 3.0 / 1.0)

    assert math.isclose(idx.idf_smooth("gato"), expected_idf_gato, rel_tol=1e-7)
    assert math.isclose(idx.idf_smooth("perro"), expected_idf_perro, rel_tol=1e-7)
    assert math.isclose(idx.idf_smooth("raton"), expected_idf_raton, rel_tol=1e-7)
    assert math.isclose(idx.idf_smooth("leon"), expected_idf_leon, rel_tol=1e-7)

    # Check Document Euclidean Norms: ||V(D)|| = sqrt(sum (tf * idf)^2)
    norm_1 = math.sqrt(expected_idf_gato**2 + expected_idf_perro**2)
    norm_2 = math.sqrt((2.0 * expected_idf_gato)**2 + expected_idf_raton**2)
    norm_3 = math.sqrt(expected_idf_perro**2 + expected_idf_leon**2)

    assert math.isclose(idx.get_doc_norm(1), norm_1, rel_tol=1e-6)
    assert math.isclose(idx.get_doc_norm(2), norm_2, rel_tol=1e-6)
    assert math.isclose(idx.get_doc_norm(3), norm_3, rel_tol=1e-6)

    # Analytical values check
    assert math.isclose(idx.get_doc_norm(1), 1.29583, abs_tol=1e-4)
    assert math.isclose(idx.get_doc_norm(2), 2.29786, abs_tol=1e-4)
    assert math.isclose(idx.get_doc_norm(3), 1.66175, abs_tol=1e-4)


def test_hand_calculated_cosine_scores(micro_corpus_index: InvertedIndex):
    """
    CRITICAL ACCEPTANCE CRITERION 1:
    Verifies that cosine similarity scores exactly match hand-calculated values:
    Doc 1 -> 1.00000
    Doc 2 -> 0.56393
    Doc 3 -> 0.38990
    """
    idx = micro_corpus_index
    query = ["gato", "perro"]

    results = idx.score_tfidf_cosine(query)

    assert len(results) == 3

    doc_ids = [doc_id for doc_id, _ in results]
    scores = [score for _, score in results]

    # Deterministic ranking order by score descending
    assert doc_ids == [1, 2, 3]

    # Compare against analytical values
    assert math.isclose(scores[0], 1.00000, abs_tol=1e-4)
    assert math.isclose(scores[1], 0.56393, abs_tol=1e-4)
    assert math.isclose(scores[2], 0.38990, abs_tol=1e-4)

    # Test top_k slicing
    top_2 = idx.score_tfidf_cosine(query, top_k=2)
    assert len(top_2) == 2
    assert top_2[0][0] == 1
    assert top_2[1][0] == 2

    top_1 = idx.score_tfidf_cosine(query, top_k=1)
    assert len(top_1) == 1
    assert top_1[0][0] == 1


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: Edge Cases & Non-Indexable Terms
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty_list(micro_corpus_index: InvertedIndex):
    """An empty sequence of query terms returns [] safely."""
    assert micro_corpus_index.score_tfidf_cosine([]) == []


def test_stopwords_only_query_behavior():
    """
    When a query contains only stopwords, the TextProcessor filters them to [],
    and score_tfidf_cosine returns [] without error.
    """
    processor = TextProcessor(stopwords=SPANISH_STOPWORDS)
    indexer = SPIMIIndexer(processor=processor)
    corpus = [
        (1, "el gato corre"),
        (2, "la casa grande"),
    ]
    idx = indexer.index_corpus(corpus)

    # Query composed purely of stopwords
    stopwords_query = "de la el y en"
    tokens = processor.process(stopwords_query)
    assert tokens == []

    results = idx.score_tfidf_cosine(tokens)
    assert results == []


def test_symbols_and_punctuation_only_query(micro_corpus_index: InvertedIndex):
    """Query with only punctuation or whitespace produces no tokens and returns []."""
    processor = TextProcessor()
    tokens = processor.process("¡!¿? --- ... ***")
    assert tokens == []
    assert micro_corpus_index.score_tfidf_cosine(tokens) == []


def test_out_of_vocabulary_query(micro_corpus_index: InvertedIndex):
    """
    Terms absent from the index have df=0 and idf=0, contributing 0 to query norm.
    If all query terms are OOV, query norm is 0 and returns [].
    """
    assert micro_corpus_index.score_tfidf_cosine(["palabrainventada123"]) == []
    assert micro_corpus_index.score_tfidf_cosine(["desconocido1", "desconocido2"]) == []


def test_mixed_query_with_oov_term(micro_corpus_index: InvertedIndex):
    """
    If query contains both known and unknown terms, unknown terms are safely ignored
    and scoring proceeds with known terms.
    """
    results_pure = micro_corpus_index.score_tfidf_cosine(["gato"])
    results_mixed = micro_corpus_index.score_tfidf_cosine(["gato", "palabrainexistente"])

    # Results should be identical because palabrainexistente has idf=0
    assert results_pure == results_mixed
    assert len(results_pure) == 2  # Doc 1 and Doc 2 contain "gato"


def test_document_without_indexable_terms_zero_norm():
    """
    Corpus containing documents that reduce to empty text (e.g. only stopwords or symbols)
    should receive norm=0.0 and NEVER trigger ZeroDivisionError.
    """
    processor = TextProcessor(stopwords=SPANISH_STOPWORDS)
    indexer = SPIMIIndexer(processor=processor)
    corpus = [
        (1, "perro policia"),
        (2, "de la el y"),       # Empty after stopword removal
        (3, "--- ... ¡!"),        # Empty after punctuation filtering
        (4, "gato bombero"),
    ]
    idx = indexer.index_corpus(corpus)

    assert idx.total_docs == 4
    # Docs 2 and 3 have 0 tokens and norm 0.0
    assert idx.get_doc_norm(2) == 0.0
    assert idx.get_doc_norm(3) == 0.0

    # Query matching doc 1 and doc 4
    query_tokens = processor.process("perro bombero")
    results = idx.score_tfidf_cosine(query_tokens)

    # Candidate docs are only 1 and 4. Zero-norm docs 2 and 3 must not fail or appear
    result_doc_ids = [d for d, _ in results]
    assert 2 not in result_doc_ids
    assert 3 not in result_doc_ids
    assert set(result_doc_ids) == {1, 4}


def test_empty_inverted_index_behavior():
    """Querying an empty InvertedIndex returns [] gracefully."""
    idx = InvertedIndex()
    assert idx.total_docs == 0
    assert idx.score_tfidf_cosine(["gato"]) == []
    assert idx.get_doc_norm(1) == 0.0
    assert idx.idf_smooth("gato") == 0.0


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: Deterministic Tie-Breaking
# ---------------------------------------------------------------------------

def test_deterministic_tie_breaking_identical_documents():
    """
    CRITICAL ACCEPTANCE CRITERION 3:
    When multiple documents have identical cosine similarity scores,
    they must be deterministically sorted by doc_id in ASCENDING order.
    Key: (-score, doc_id)
    """
    indexer = SPIMIIndexer(processor=TextProcessor(stopwords=set()))
    # 4 documents with identical content, inserted in arbitrary ID order
    corpus = [
        (50, "algoritmo busqueda"),
        (10, "algoritmo busqueda"),
        (30, "algoritmo busqueda"),
        (20, "algoritmo busqueda"),
    ]
    idx = indexer.index_corpus(corpus)

    query = ["algoritmo", "busqueda"]
    results = idx.score_tfidf_cosine(query)

    assert len(results) == 4

    # All scores must be 1.0
    for doc_id, score in results:
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    # Ordering MUST strictly be ascending by doc_id: [10, 20, 30, 50]
    doc_ids = [doc_id for doc_id, _ in results]
    assert doc_ids == [10, 20, 30, 50]


def test_deterministic_tie_breaking_partial_overlap():
    """
    Validates tie-breaking when two distinct documents achieve the exact same partial score.
    """
    idx = InvertedIndex(
        index={
            "python": [(100, 1), (200, 1)],
            "java": [(100, 1), (200, 1)],
        },
        doc_lengths={100: 2, 200: 2},
        total_docs=2,
    )

    results = idx.score_tfidf_cosine(["python"])
    assert len(results) == 2
    assert results[0][0] == 100
    assert results[1][0] == 200
    assert math.isclose(results[0][1], results[1][1], rel_tol=1e-7)
