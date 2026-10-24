"""
Unit tests for the SPIMI indexer and InvertedIndex (Issue 2).

Validates:
- Construction without requiring the whole corpus in RAM (streaming/generator).
- Postings structure storing (doc_id, tf) correctly.
- Strict determinism for identical corpus across multiple runs.
- Edge cases: multiple blocks, repeated terms, empty corpus, BM25 scoring.
"""

import os
import pytest
from index.text.spimi import SPIMIIndexer, SPIMIStats
from index.text.text_processor import TextProcessor


def test_empty_corpus():
    """Validates indexing an empty corpus produces an empty index without error."""
    indexer = SPIMIIndexer()
    inverted_index = indexer.index_corpus([])

    assert inverted_index.total_docs == 0
    assert inverted_index.get_terms() == []
    assert inverted_index.avg_doc_length == 0.0
    assert inverted_index.get_postings("cualquiera") == []
    assert inverted_index.score_bm25(["test"]) == []


def test_single_block_indexing():
    """Indexes a small corpus fitting into a single memory block."""
    indexer = SPIMIIndexer(max_records_per_block=100)
    corpus = [
        (1, "base de datos relacional"),
        (2, "sistema de base de datos"),
    ]

    index = indexer.index_corpus(corpus)

    assert index.total_docs == 2
    # 'base' appears in doc 1 (tf=1) and doc 2 (tf=1)
    assert index.get_postings("base") == [(1, 1), (2, 1)]
    # 'de' appears in doc 1 (tf=1) and doc 2 (tf=2)
    assert index.get_postings("de") == [(1, 1), (2, 2)]
    # 'relacional' only in doc 1
    assert index.get_postings("relacional") == [(1, 1)]
    assert index.get_document_frequency("base") == 2
    assert index.get_document_frequency("relacional") == 1
    assert index.get_term_frequency("de", 2) == 2


def test_multiple_blocks_spill_and_merge():
    """
    CRITICAL ACCEPTANCE CRITERION:
    Forces multiple blocks by setting max_records_per_block=2 with 6 documents.
    Verifies that all blocks spill to disk and K-way merge correctly consolidates them.
    """
    indexer = SPIMIIndexer(max_records_per_block=2)
    corpus = [
        (10, "gato perro"),
        (20, "gato raton"),
        (30, "perro leon"),
        (40, "leon tigre"),
        (50, "gato tigre"),
        (60, "perro perro perro"),  # Repeated term in doc 60
    ]

    index = indexer.index_corpus(corpus)

    assert index.total_docs == 6
    # 'gato' appears in docs 10, 20, 50
    assert index.get_postings("gato") == [(10, 1), (20, 1), (50, 1)]
    # 'perro' appears in doc 10 (tf=1), 30 (tf=1), 60 (tf=3)
    assert index.get_postings("perro") == [(10, 1), (30, 1), (60, 3)]
    # 'leon' appears in doc 30 (tf=1), 40 (tf=1)
    assert index.get_postings("leon") == [(30, 1), (40, 1)]
    # 'tigre' appears in doc 40 (tf=1), 50 (tf=1)
    assert index.get_postings("tigre") == [(40, 1), (50, 1)]
    # 'raton' appears in doc 20 (tf=1)
    assert index.get_postings("raton") == [(20, 1)]


def test_repeated_terms_within_document():
    """Term frequency (tf) accumulates accurately for duplicate words in the same document."""
    indexer = SPIMIIndexer(max_records_per_block=5)
    corpus = [
        (1, "algoritmo algoritmo algoritmo estructura"),
        (2, "algoritmo estructura estructura"),
    ]

    index = indexer.index_corpus(corpus)

    assert index.get_term_frequency("algoritmo", 1) == 3
    assert index.get_term_frequency("estructura", 1) == 1
    assert index.get_term_frequency("algoritmo", 2) == 1
    assert index.get_term_frequency("estructura", 2) == 2

    assert index.get_postings("algoritmo") == [(1, 3), (2, 1)]
    assert index.get_postings("estructura") == [(1, 1), (2, 2)]


def test_streaming_generator_input():
    """
    CRITICAL ACCEPTANCE CRITERION:
    Documents are passed as a generator stream, proving the whole corpus
    is not preloaded in memory.
    """
    def doc_stream():
        for i in range(1, 11):
            yield (i, f"termino_{i % 3} comun")

    indexer = SPIMIIndexer(max_records_per_block=3)
    index = indexer.index_corpus(doc_stream())

    assert index.total_docs == 10
    # 'comun' appears in all 10 documents
    assert index.get_document_frequency("comun") == 10
    assert len(index.get_postings("comun")) == 10


def test_deterministic_output():
    """
    CRITICAL ACCEPTANCE CRITERION:
    Indexing the same corpus twice generates completely deterministic,
    identical vocabulary and postings lists.
    """
    corpus = [
        (1, "zarigüeya zorro manzana pera"),
        (2, "pera platano manzana"),
        (3, "zorro zarigüeya pera"),
    ]

    indexer1 = SPIMIIndexer(max_records_per_block=2)
    index1 = indexer1.index_corpus(corpus)

    indexer2 = SPIMIIndexer(max_records_per_block=2)
    index2 = indexer2.index_corpus(corpus)

    assert index1.get_terms() == index2.get_terms()
    for term in index1.get_terms():
        assert index1.get_postings(term) == index2.get_postings(term)
    assert index1.doc_lengths == index2.doc_lengths


def test_bm25_ranking_evaluation():
    """Validates Okapi BM25 ranking: top matches score highest."""
    indexer = SPIMIIndexer(max_records_per_block=2)
    corpus = [
        (1, "computacion y algoritmos grafos y algoritmos"),  # 2x algoritmos, 1x grafos
        (2, "receta de cocina y comida"),                     # No match
        (3, "introduccion a grafos y redes"),                 # 1x grafos
        (4, "algoritmos de optimizacion matematica"),         # 1x algoritmos
    ]
    index = indexer.index_corpus(corpus)

    # Search for "algoritmos grafos"
    results = index.score_bm25(["algoritmos", "grafos"], k1=1.2, b=0.75, top_k=5)

    assert len(results) == 3  # Docs 1, 3, 4 match; Doc 2 has score 0
    top_doc_id, top_score = results[0]
    # Doc 1 has both terms and 'algoritmos' twice, so it should rank first
    assert top_doc_id == 1
    assert top_score > results[1][1]


def test_spill_block_isolated(tmp_path):
    """
    White-box test: verifies _spill_block directly.
    Checks that the binary block file contains sorted terms and sorted doc_ids.
    """
    import pickle
    indexer = SPIMIIndexer()
    block_data = {
        "perro": {20: 1, 10: 2},
        "arbol": {5: 1},
        "gato": {15: 3, 5: 1},
    }

    block_file = indexer._spill_block(block_data, str(tmp_path), block_index=0)
    assert os.path.exists(block_file)

    # Read binary stream and verify sequential sorted terms
    items = []
    with open(block_file, "rb") as f:
        while True:
            try:
                items.append(pickle.load(f))
            except EOFError:
                break

    # Terms must be lexicographically sorted: "arbol", "gato", "perro"
    terms = [term for term, _ in items]
    assert terms == ["arbol", "gato", "perro"]

    # Postings inside each term must be sorted by doc_id ascending
    arbol_postings = dict(items)["arbol"]
    assert arbol_postings == [(5, 1)]

    gato_postings = dict(items)["gato"]
    assert gato_postings == [(5, 1), (15, 3)]

    perro_postings = dict(items)["perro"]
    assert perro_postings == [(10, 2), (20, 1)]


def test_merge_blocks_isolated(tmp_path):
    """
    White-box test: verifies _merge_blocks directly.
    Simulates K independent block files and confirms the Min-Heap merges them accurately.
    """
    indexer = SPIMIIndexer()

    # Block 0: 'algoritmo' in doc 1, 'estructura' in doc 1
    b0 = indexer._spill_block(
        {"algoritmo": {1: 2}, "estructura": {1: 1}},
        str(tmp_path),
        block_index=0,
    )

    # Block 1: 'algoritmo' in doc 2, 'grafo' in doc 2
    b1 = indexer._spill_block(
        {"algoritmo": {2: 1}, "grafo": {2: 3}},
        str(tmp_path),
        block_index=1,
    )

    # Block 2: 'estructura' in doc 3, 'grafo' in doc 3
    b2 = indexer._spill_block(
        {"estructura": {3: 2}, "grafo": {3: 1}},
        str(tmp_path),
        block_index=2,
    )

    merged = indexer._merge_blocks([b0, b1, b2])

    assert sorted(merged.keys()) == ["algoritmo", "estructura", "grafo"]
    assert merged["algoritmo"] == [(1, 2), (2, 1)]
    assert merged["estructura"] == [(1, 1), (3, 2)]
    assert merged["grafo"] == [(2, 3), (3, 1)]


def test_spimi_telemetry_metrics():
    """
    White-box test: verifies SPIMIStats telemetry records blocks, sizes, and execution times.
    """
    indexer = SPIMIIndexer(max_records_per_block=2)
    corpus = [
        (1, "primero dos tres"),
        (2, "cuatro cinco seis"),
        (3, "siete ocho nueve"),
        (4, "diez once doce"),
        (5, "trece catorce quince"),
    ]

    index = indexer.index_corpus(corpus, keep_blocks=True)

    stats = indexer.stats
    assert stats.blocks_created == 3
    assert stats.block_sizes == [2, 2, 1]
    assert stats.total_tokens_processed == 15
    assert stats.spill_time_sec >= 0.0
    assert stats.merge_time_sec >= 0.0
    assert stats.total_time_sec >= stats.spill_time_sec + stats.merge_time_sec

    # With keep_blocks=True, intermediate files exist until manually cleaned
    assert len(indexer.last_block_paths) == 3
    for p in indexer.last_block_paths:
        assert os.path.exists(p)
        os.remove(p)  # Clean up test files

