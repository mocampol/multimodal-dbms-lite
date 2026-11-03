"""
SQL-to-results coverage of SCORE(), its alias, descending order and LIMIT in
full-text queries (docs/text_search_policy.md section 11):

  * SCORE() is a numeric column (DOUBLE_PRECISION) exposed under an alias
  * SELECT * keeps every document column and appends the alias
  * ORDER BY <alias> DESC orders by the real BM25 / TF-IDF score
  * the ranking is applied before LIMIT (Projection -> Limit -> Sort -> access)
  * both access paths answer identically: no index (SeqScan + TextSearch) and
    a SPIMI index (TextIndexScan)

Expected rankings come from a reference implementation of the policy formulas
applied to the raw documents; it shares nothing with InvertedIndex except the
TextProcessor that tokenizes documents and queries.
"""

import math
from collections import Counter

import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from common.value import DataType
from index.text import default_text_processor
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute

# (id, titulo, contenido, anio)
DOCS = [
    (1, "Índices B+", "Los índices B+ aceleran la búsqueda de registros. Un índice B+ mantiene las claves ordenadas y el árbol balanceado.", 2001),
    (2, "Índices hash", "Un índice hash ofrece búsqueda por igualdad en tiempo casi constante, pero no soporta rangos.", 2003),
    (3, "Optimizador", "El optimizador elige el plan de ejecución con menor costo estimado usando estadísticas del catálogo.", 2005),
    (4, "Transacciones", "Las transacciones garantizan atomicidad, consistencia, aislamiento y durabilidad.", 2007),
    (5, "Buffer pool", "El buffer pool mantiene páginas del disco en memoria para evitar lecturas repetidas.", 2009),
    (6, "Recuperación", "La recuperación usa el registro de transacciones para deshacer y rehacer cambios tras un fallo.", 2011),
    (7, "Índice invertido", "Un índice invertido asocia cada término con la lista de documentos que lo contienen, el término y su frecuencia.", 2013),
    (8, "Ranking BM25", "BM25 ordena documentos por relevancia combinando la frecuencia del término y la longitud del documento.", 2015),
    (9, "TF-IDF", "TF-IDF pondera cada término por su frecuencia en el documento y su rareza en la colección de documentos.", 2017),
    (10, "Receta", "Receta de pasta con salsa de tomate y albahaca fresca.", 2019),
    (11, "Bloqueos", "El gestor de bloqueos evita conflictos entre transacciones que escriben el mismo registro.", 2021),
    (12, "Archivos heap", "Un archivo heap guarda registros sin orden y se recorre completo en cada búsqueda.", 2023),
]
DOCS_BY_ID = {doc[0]: doc for doc in DOCS}

# 6 candidate documents, no tied scores. The best one (id 2) is not the first
# in insertion order, so a LIMIT applied before ranking would return other rows.
QUERY = "indice registro busqueda"

TABLES = {"docs_scan": None, "docs_spimi": "SPIMI"}
PROCESSOR = default_text_processor()


# --------------------------------------------------------------------------
# Reference implementation of the policy (sections 7 and 8)
# --------------------------------------------------------------------------

def _corpus():
    return {doc_id: PROCESSOR.process(content) for doc_id, _, content, _ in DOCS}


def reference_bm25(query, k1=1.2, b=0.75):
    corpus, terms = _corpus(), PROCESSOR.process(query)
    n = len(corpus)
    avgdl = sum(len(tokens) for tokens in corpus.values()) / n
    scores = {}
    for term in dict.fromkeys(terms):
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


def reference_tfidf(query):
    corpus, terms = _corpus(), PROCESSOR.process(query)
    n = len(corpus)

    def idf(term):
        df = sum(1 for tokens in corpus.values() if term in tokens)
        return math.log(1 + n / df) if df else 0.0

    weights = {t: c * idf(t) for t, c in Counter(terms).items() if idf(t) > 0}
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


def with_zero_scores(ranking):
    """Ranking of a SCORE() without MATCH: every document, the non-candidates
    last with 0.0 (ties keep ascending id)."""
    seen = {doc_id for doc_id, _ in ranking}
    return list(ranking) + [(doc[0], 0.0) for doc in DOCS if doc[0] not in seen]


# --------------------------------------------------------------------------
# Engine fixtures and helpers
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
    catalog = make_catalog(tmp_path_factory.mktemp("text_score"))
    inserts = ", ".join(
        f"({doc_id}, '{titulo}', '{contenido}', {anio})" for doc_id, titulo, contenido, anio in DOCS
    )
    for table, index_type in TABLES.items():
        execute(
            f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, titulo VARCHAR(40), "
            "contenido TEXT, anio INTEGER);",
            catalog,
        )
        execute(f"INSERT INTO {table} VALUES {inserts};", catalog)
        if index_type is not None:
            execute(
                f"CREATE INDEX idx_{table} ON {table} USING {index_type} (contenido);",
                catalog,
            )
    return catalog


def pairs(rows):
    """[(id, score)] from rows shaped (id, ..., score)."""
    return [(row.values[0].data, row.values[-1].data) for row in rows]


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


def explain(engine, sql):
    return execute("EXPLAIN " + sql, engine)


def search(table, items="id, SCORE() AS relevancia", query=QUERY, using="",
           order="ORDER BY relevancia DESC", limit=""):
    return (
        f"SELECT {items} FROM {table} WHERE MATCH(contenido, '{query}'){using} {order}{limit};"
    ).replace(" ;", ";")


# --------------------------------------------------------------------------
# SCORE() as a numeric column with an alias
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_score_is_a_numeric_column_equal_to_the_reference_score(engine, table):
    rows = execute(search(table), engine)
    expected = reference_bm25(QUERY)

    assert len(expected) == 6
    assert_same_ranking(pairs(rows), expected)
    for row in rows:
        score = row.values[1]
        assert score.data_type == DataType.DOUBLE_PRECISION
        assert isinstance(score.data, float) and score.data > 0


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("item, alias", [
    ("SCORE() AS relevancia", "relevancia"),
    ("SCORE() AS puntaje", "puntaje"),
    ("SCORE() puntaje", "puntaje"),   # AS is optional
    ("SCORE()", "score"),             # default alias
])
def test_alias_names_the_column_and_can_be_ordered_by(engine, table, item, alias):
    sql = search(table, items=f"id, {item}", order=f"ORDER BY {alias} DESC")

    assert explain(engine, sql).plan_root.columns == ["id", alias]
    assert_same_ranking(pairs(execute(sql, engine)), reference_bm25(QUERY))


@pytest.mark.parametrize("table", TABLES)
def test_score_can_be_combined_with_other_selected_columns(engine, table):
    rows = execute(search(table, items="titulo, anio, SCORE() AS relevancia"), engine)
    expected = reference_bm25(QUERY)

    assert [(r.values[0].data, r.values[1].data) for r in rows] == [
        (DOCS_BY_ID[doc_id][1], DOCS_BY_ID[doc_id][3]) for doc_id, _ in expected
    ]
    assert [r.values[2].data for r in rows] == pytest.approx([s for _, s in expected], rel=1e-9)


# --------------------------------------------------------------------------
# SELECT *: document columns preserved, alias appended
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_select_star_keeps_the_document_columns_and_appends_the_alias(engine, table):
    rows = execute(search(table, items="*, SCORE() AS relevancia"), engine)
    expected = reference_bm25(QUERY)

    assert [r.values[0].data for r in rows] == [doc_id for doc_id, _ in expected]
    for row, (doc_id, score) in zip(rows, expected):
        assert len(row.values) == 5
        assert [v.data for v in row.values[:4]] == list(DOCS_BY_ID[doc_id])  # the stored row
        assert row.values[4].data == pytest.approx(score, rel=1e-9)
        assert row.values[4].data_type == DataType.DOUBLE_PRECISION


@pytest.mark.parametrize("table", TABLES)
def test_select_star_output_schema_lists_document_columns_then_alias(engine, table):
    result = explain(engine, search(table, items="*, SCORE() AS relevancia"))
    assert [c.name for c in result.plan_root.schema.columns] == [
        "id", "titulo", "contenido", "anio", "relevancia"
    ]


# --------------------------------------------------------------------------
# ORDER BY <alias> DESC orders by the real score
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_order_by_alias_desc_orders_by_score_not_by_insertion_order(engine, table):
    ranking = pairs(execute(search(table), engine))
    ids = [doc_id for doc_id, _ in ranking]

    assert [s for _, s in ranking] == sorted((s for _, s in ranking), reverse=True)
    assert ids == [doc_id for doc_id, _ in reference_bm25(QUERY)]
    assert ids != sorted(ids)  # guard: the fixture would be vacuous if both orders agreed


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("order", ["ORDER BY relevancia", "ORDER BY relevancia ASC"])
def test_ascending_order_is_the_exact_reverse_of_descending(engine, table, order):
    descending = pairs(execute(search(table), engine))
    ascending = pairs(execute(search(table, order=order), engine))
    assert [d for d, _ in ascending] == [d for d, _ in reversed(descending)]  # no tied scores


@pytest.mark.parametrize("table", TABLES)
def test_selecting_the_score_does_not_force_its_ordering(engine, table):
    rows = execute(search(table, order="ORDER BY anio DESC"), engine)
    candidates = {doc_id for doc_id, _ in reference_bm25(QUERY)}

    assert [r.values[0].data for r in rows] == sorted(candidates, reverse=True)  # anio grows with id
    assert all(r.values[1].data > 0 for r in rows)


# --------------------------------------------------------------------------
# LIMIT is applied after the ranking
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("n", [1, 2, 3, 6, 50])
def test_limit_returns_the_top_n_of_the_ranking(engine, table, n):
    ranking = reference_bm25(QUERY)
    first_in_insertion_order = sorted(doc_id for doc_id, _ in ranking)[:n]
    if n < len(ranking):
        # guard: limiting before ranking would return these other documents
        assert [doc_id for doc_id, _ in ranking[:n]] != first_in_insertion_order

    rows = execute(search(table, limit=f" LIMIT {n}"), engine)
    assert_same_ranking(pairs(rows), ranking[:n])


@pytest.mark.parametrize("table", TABLES)
def test_limit_with_select_star_and_tf_idf(engine, table):
    rows = execute(
        search(table, items="*, SCORE() AS relevancia", using=" USING TF_IDF", limit=" LIMIT 3"),
        engine,
    )
    expected = reference_tfidf(QUERY)

    assert_same_ranking(pairs(rows), expected[:3])
    assert all(0.0 < row.values[4].data <= 1.0 for row in rows)  # cosine similarity
    assert all(len(row.values) == 5 for row in rows)


def test_plan_ranks_before_it_limits(engine):
    for table, index_type in TABLES.items():
        nodes = plan_nodes(explain(engine, search(table, limit=" LIMIT 2")).description)

        assert nodes[:3] == ["Projection", "Limit", "Sort"]  # Sort feeds Limit
        if index_type is None:
            assert nodes[3:] == ["TextSearch", "SeqScan"]
        else:
            assert nodes[3:] == ["TextIndexScan"]


# --------------------------------------------------------------------------
# SCORE(column, 'query') in its explicit forms
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_explicit_score_without_match_scores_every_document(engine, table):
    sql = (
        f"SELECT id, SCORE(contenido, '{QUERY}') AS relevancia FROM {table} "
        "ORDER BY relevancia DESC;"
    )
    expected = with_zero_scores(reference_bm25(QUERY))

    assert len(expected) == len(DOCS)  # no MATCH filter: all documents come back
    assert_same_ranking(pairs(execute(sql, engine)), expected)
    assert_same_ranking(pairs(execute(sql.replace(";", " LIMIT 3;"), engine)), expected[:3])


@pytest.mark.parametrize("table", TABLES)
def test_explicit_score_with_the_same_query_as_match_equals_bare_score(engine, table):
    explicit = search(table, items=f"id, SCORE(contenido, '{QUERY}') AS relevancia")
    assert_same_ranking(pairs(execute(explicit, engine)), pairs(execute(search(table), engine)))

    nodes = plan_nodes(explain(engine, explicit).description)
    assert ("TextIndexScan" in nodes) == (TABLES[table] is not None)  # still served by the index


def test_score_over_a_different_query_than_match_scores_independently(engine):
    match_query, score_query = "indice busqueda", "registro busqueda"
    sql = (
        f"SELECT id, SCORE(contenido, '{score_query}') AS relevancia FROM docs_spimi "
        f"WHERE MATCH(contenido, '{match_query}') ORDER BY relevancia DESC;"
    )
    candidates = {doc_id for doc_id, _ in reference_bm25(match_query)}
    own_scores = dict(reference_bm25(score_query))
    expected = sorted(
        ((doc_id, own_scores.get(doc_id, 0.0)) for doc_id in candidates),
        key=lambda item: (-item[1], item[0]),
    )

    assert len(candidates) == 4 and 0.0 in [s for _, s in expected]
    assert_same_ranking(pairs(execute(sql, engine)), expected)
    # the index cannot answer two different queries: the planner falls back to a scan
    nodes = plan_nodes(explain(engine, sql).description)
    assert "TextSearch" in nodes and "SeqScan" in nodes and "TextIndexScan" not in nodes