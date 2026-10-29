"""
SQL-to-results coverage of full-text queries by number of terms (1, 3 and 5 or
more), over the three ways the engine can answer them, with both ranking
algorithms (docs/text_search_policy.md sections 5, 6, 7, 8 and 11.3):

  docs_scan   no index        -> SeqScan + TextSearch (transient InvertedIndex)
  docs_spimi  USING SPIMI     -> TextIndexScan
  docs_gin    USING GIN       -> TextIndexScan

All three tables hold the same documents inserted in the same order, so the
policy requires them to return the same documents, scores and order. Expected
values come from a reference implementation of the policy formulas applied to
the raw documents (it shares nothing with InvertedIndex except the
TextProcessor that tokenizes documents and queries).
"""

import math
from collections import Counter

import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from index.text import default_text_processor
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute

# Fixed literals (not generated at import time) so the corpus never changes
# with the Python version: 40 documents, 5-13 indexable terms each.
DOCS = [
    (1, "Catálogo esquema en pagina de bloqueo y buffer tabla latencia la caché escritura la lectura."),
    (2, "Tabla plan clave cache lectura lectura página el columna escritura de caché latencia ranking buffer los."),
    (3, "Registro el latencia lectura de relevancia transacción los tabla."),
    (4, "Disco el ranking esquema transaccion en plan escritura arbol frecuencia escritura clave la memoria caché."),
    (5, "Costo disco arbol catalogo los fila de latencia escritura de."),
    (6, "Árbol la lectura el catalogo latencia en cache escritura de."),
    (7, "Recuperación y costo hash tabla ranking clave índice arbol columna indice."),
    (8, "Columna índice hash escritura recuperación de árbol arbol y relevancia."),
    (9, "Relevancia página buffer los tabla plan caché vocabulario en esquema relevancia ranking y."),
    (10, "Árbol árbol relevancia la recuperación pagina en caché los indice."),
    (11, "Fila tabla en vocabulario transaccion indice catálogo latencia el catálogo indice índice árbol termino frecuencia."),
    (12, "Ranking índice transaccion transacción vocabulario los costo vocabulario latencia la."),
    (13, "Consulta recuperación la cache los esquema y transaccion hash clave plan la tabla el consulta clave la ranking."),
    (14, "Esquema los catálogo latencia de vocabulario índice lectura de hash y ranking documento."),
    (15, "Indice la ranking consulta y caché de clave de."),
    (16, "Latencia columna indice hash los consulta vocabulario frecuencia los cache plan en tabla pagina la fila de plan."),
    (17, "Latencia documento los consulta de bloqueo de transacción disco columna la escritura clave arbol."),
    (18, "Catalogo disco recuperación memoria de página catálogo bloqueo tabla tabla ranking transacción índice."),
    (19, "Recuperación la recuperación hash costo los relevancia buffer la."),
    (20, "Costo página recuperación y ranking disco memoria de lectura en catálogo arbol hash la."),
    (21, "Lectura los buffer catalogo en recuperación de indice costo la."),
    (22, "Escritura la esquema documento de cache tabla documento árbol de documento registro el documento tabla."),
    (23, "Transacción los arbol relevancia registro página en esquema el vocabulario transacción y latencia los latencia y columna."),
    (24, "Vocabulario el documento registro frecuencia fila catalogo latencia el columna consulta la buffer clave documento."),
    (25, "Catalogo bloqueo en fila ranking registro y."),
    (26, "Tabla de recuperación ranking página termino registro."),
    (27, "Ranking registro el cache el caché buffer disco indice la lectura columna la esquema costo."),
    (28, "Ranking la fila escritura la plan esquema y columna árbol relevancia disco recuperación y."),
    (29, "Fila bloqueo ranking los bloqueo los clave esquema frecuencia el."),
    (30, "Escritura fila transaccion los cache y latencia el registro en."),
    (31, "Relevancia página los registro esquema bloqueo esquema los bloqueo frecuencia y."),
    (32, "Arbol tabla consulta caché hash los ranking."),
    (33, "Memoria consulta documento término documento en frecuencia escritura."),
    (34, "Cache fila hash ranking y frecuencia clave la consulta la catalogo."),
    (35, "Documento buffer en esquema costo el consulta la índice indice consulta."),
    (36, "Registro el registro la plan los árbol cache pagina el memoria en ranking de memoria."),
    (37, "Pagina latencia buffer pagina clave esquema ranking."),
    (38, "Recuperación esquema columna de árbol fila y transaccion clave el fila caché en tabla de buffer y."),
    (39, "Buffer transacción ranking memoria ranking tabla los transacción y clave pagina y relevancia termino."),
    (40, "Lectura y consulta fila esquema plan escritura en frecuencia en índice la frecuencia."),
]

TABLES = {"docs_scan": None, "docs_spimi": "SPIMI", "docs_gin": "GIN"}
ALGORITHMS = ["BM25", "TF_IDF"]

QUERIES = [
    pytest.param("indice", id="1-termino"),
    pytest.param("indice arbol pagina", id="3-terminos"),
    pytest.param("indice arbol pagina buffer bloqueo", id="5-terminos"),
    pytest.param("indice arbol pagina buffer bloqueo transaccion consulta plan", id="8-terminos"),
    pytest.param("indice arbol pagina buffer zzzdesconocido", id="5-terminos-uno-desconocido"),
    pytest.param("indice indice arbol", id="3-terminos-uno-repetido"),  # BM25 counts it once; TF-IDF weights it twice
]

PROCESSOR = default_text_processor()


# --------------------------------------------------------------------------
# Reference implementation of the policy (sections 7 and 8)
# --------------------------------------------------------------------------

def _corpus():
    return {doc_id: PROCESSOR.process(text) for doc_id, text in DOCS}


def reference_bm25(corpus, query, k1=1.2, b=0.75):
    n = len(corpus)
    avgdl = sum(len(tokens) for tokens in corpus.values()) / n
    scores = {}
    for term in dict.fromkeys(query):  # distinct query terms
        df = sum(1 for tokens in corpus.values() if term in tokens)
        if df == 0:
            continue  # section 6: unknown terms contribute nothing
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for doc_id, tokens in corpus.items():
            tf = tokens.count(term)
            if tf:
                norm = k1 * (1 - b + b * len(tokens) / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + norm)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def reference_tfidf(corpus, query):
    n = len(corpus)

    def idf(term):
        df = sum(1 for tokens in corpus.values() if term in tokens)
        return math.log(1 + n / df) if df else 0.0

    weights = {t: c * idf(t) for t, c in Counter(query).items() if idf(t) > 0}
    if not weights:
        return []
    query_norm = math.sqrt(sum(w * w for w in weights.values()))
    scores = {}
    for doc_id, tokens in corpus.items():
        counts = Counter(tokens)
        doc_norm = math.sqrt(sum((c * idf(t)) ** 2 for t, c in counts.items()))
        dot = sum(w * counts[t] * idf(t) for t, w in weights.items() if t in counts)
        if doc_norm and dot > 0:
            scores[doc_id] = min(1.0, dot / (query_norm * doc_norm))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def reference(algorithm, query):
    terms = PROCESSOR.process(query)
    corpus = _corpus()
    return reference_bm25(corpus, terms) if algorithm == "BM25" else reference_tfidf(corpus, terms)


# --------------------------------------------------------------------------
# Engine fixtures and SQL helpers
# --------------------------------------------------------------------------

def make_catalog(base_dir):
    heap_factory = make_heap_factory(str(base_dir))
    return Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    catalog = make_catalog(tmp_path_factory.mktemp("text_terms"))
    inserts = ", ".join(f"({doc_id}, '{text}')" for doc_id, text in DOCS)
    for table, index_type in TABLES.items():
        execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, contenido TEXT);", catalog)
        execute(f"INSERT INTO {table} VALUES {inserts};", catalog)
        if index_type is not None:
            execute(
                f"CREATE INDEX idx_{table} ON {table} USING {index_type} (contenido);",
                catalog,
            )
    return catalog


def ranked(engine, table, algorithm, query):
    """[(id, score)] in the order the SQL query returns them."""
    rows = execute(
        f"SELECT id, SCORE() AS relevancia FROM {table} "
        f"WHERE MATCH(contenido, '{query}') USING {algorithm} "
        "ORDER BY relevancia DESC;",
        engine,
    )
    return [(row.values[0].data, row.values[1].data) for row in rows]


def matched_ids(engine, table, query):
    rows = execute(
        f"SELECT id FROM {table} WHERE MATCH(contenido, '{query}') ORDER BY id;", engine
    )
    return [row.values[0].data for row in rows]


def assert_same_ranking(actual, expected):
    assert [doc_id for doc_id, _ in actual] == [doc_id for doc_id, _ in expected]
    assert [score for _, score in actual] == pytest.approx(
        [score for _, score in expected], rel=1e-9
    )


def plan_nodes(description):
    names = [description["node"]]
    for child in description.get("children", []):
        names.extend(plan_nodes(child))
    return names


# --------------------------------------------------------------------------
# The three access paths really are the ones under test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_each_table_is_answered_by_the_expected_access_path(engine, table):
    result = execute(
        f"EXPLAIN SELECT id, SCORE() AS relevancia FROM {table} "
        "WHERE MATCH(contenido, 'indice arbol') ORDER BY relevancia DESC;",
        engine,
    )
    nodes = plan_nodes(result.description)

    if TABLES[table] is None:
        assert "TextSearch" in nodes and "SeqScan" in nodes
        assert "TextIndexScan" not in nodes
    else:
        assert "TextIndexScan" in nodes
        assert "TextSearch" not in nodes and "SeqScan" not in nodes


# --------------------------------------------------------------------------
# Ranking by number of terms, per algorithm and access path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("query", QUERIES)
def test_ranking_matches_the_reference_implementation(engine, table, algorithm, query):
    rows = ranked(engine, table, algorithm, query)
    expected = reference(algorithm, query)

    assert expected, "the fixture corpus must produce candidates for every query"
    assert_same_ranking(rows, expected)
    assert all(isinstance(score, float) and score > 0 for _, score in rows)
    assert [score for _, score in rows] == sorted((s for _, s in rows), reverse=True)
    if algorithm == "TF_IDF":
        assert all(score <= 1.0 for _, score in rows)  # cosine similarity is bounded


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("query", QUERIES)
def test_index_and_full_scan_return_identical_results(engine, algorithm, query):
    scan = ranked(engine, "docs_scan", algorithm, query)
    for table in ("docs_spimi", "docs_gin"):
        assert_same_ranking(ranked(engine, table, algorithm, query), scan)


# --------------------------------------------------------------------------
# Multi-term semantics: OR candidates (section 5) and unknown terms (section 6)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_match_without_score_returns_every_document_sharing_a_term(engine, table):
    previous = set()
    for query in ("indice", "indice arbol pagina", "indice arbol pagina buffer bloqueo",
                  "indice arbol pagina buffer bloqueo transaccion consulta plan"):
        ids = matched_ids(engine, table, query)
        expected = sorted(doc_id for doc_id, _ in reference("BM25", query))

        assert ids == expected
        assert previous <= set(ids)  # adding terms can only add candidates (OR)
        previous = set(ids)
    assert len(previous) > len(matched_ids(engine, table, "indice"))


@pytest.mark.parametrize("table", ["docs_scan", "docs_spimi"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_unknown_terms_do_not_change_the_ranking(engine, table, algorithm):
    with_unknown = ranked(engine, table, algorithm, "indice arbol pagina buffer zzzdesconocido")
    without = ranked(engine, table, algorithm, "indice arbol pagina buffer")
    assert_same_ranking(with_unknown, without)


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_query_made_only_of_unknown_terms_returns_no_rows(engine, table, algorithm):
    assert ranked(engine, table, algorithm, "zzzdesconocido yyyausente") == []


@pytest.mark.parametrize("table", ["docs_scan", "docs_spimi"])
def test_queries_are_normalized_like_documents(engine, table):
    # case and accents do not matter (sections 1.2 and 1.3)
    assert_same_ranking(
        ranked(engine, table, "BM25", "ÍNDICE Árbol PÁGINA"),
        ranked(engine, table, "BM25", "indice arbol pagina"),
    )


@pytest.mark.parametrize("table", ["docs_scan", "docs_spimi"])
def test_default_and_legacy_forms_rank_with_bm25(engine, table):
    expected = ranked(engine, table, "BM25", "indice arbol pagina")
    for condition in ("MATCH(contenido, 'indice arbol pagina')",
                      "contenido MATCH 'indice arbol pagina'"):
        rows = execute(
            f"SELECT id, SCORE() AS relevancia FROM {table} WHERE {condition} "
            "ORDER BY relevancia DESC;",
            engine,
        )
        assert_same_ranking([(r.values[0].data, r.values[1].data) for r in rows], expected)