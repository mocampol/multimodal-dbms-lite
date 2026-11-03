"""
SQL-level error handling of full-text search (docs/text_search_policy.md
section 11.2): wrong column, wrong type, missing index, unsupported ranking
algorithm, and the other invalid forms of MATCH / SCORE() / LIMIT.

Every failure must surface as a QueryError with a message that says what is
wrong, and must not leave the engine in a bad state. Statements are rejected
by the parser or the semantic check before anything is executed, so these
tests never depend on stored data.
"""

import re

import pytest

from catalog.catalog import Catalog
from catalog.exceptions import TableNotFoundError
from catalog.table_metadata import StorageType
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import QueryError, execute

DOCS = [
    (1, "Indices", "los indices aceleran la busqueda de registros por clave"),
    (2, "Optimizador", "el optimizador elige el plan de ejecucion con menor costo"),
    (3, "Ranking", "el ranking ordena documentos por relevancia"),
]

SEARCH = "SELECT id FROM docs WHERE MATCH(contenido, 'indice')"
RANKED = "SELECT id, SCORE() AS relevancia FROM docs WHERE MATCH(contenido, 'indice')"


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


def build_catalog(base_dir):
    catalog = make_catalog(base_dir)
    execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, titulo VARCHAR(40), contenido TEXT);",
        catalog,
    )
    execute("CREATE TABLE otros (doc_id INTEGER, nota INTEGER);", catalog)
    execute("CREATE TABLE lugares (id INTEGER PRIMARY KEY, nombre TEXT, ubicacion POINT);", catalog)
    inserts = ", ".join(f"({i}, '{titulo}', '{contenido}')" for i, titulo, contenido in DOCS)
    execute(f"INSERT INTO docs VALUES {inserts};", catalog)
    return catalog


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Shared by the tests whose statements are rejected before they run."""
    return build_catalog(tmp_path_factory.mktemp("text_errors"))


@pytest.fixture
def fresh(tmp_path):
    """Own catalog per test, for the ones that create and drop indexes."""
    return build_catalog(tmp_path)


def plan_nodes(description):
    names = [description["node"]]
    for child in description.get("children", []):
        names.extend(plan_nodes(child))
    return names


def assert_rejected(engine, sql, fragment):
    with pytest.raises(QueryError, match=re.escape(fragment)):
        execute(sql, engine)


# --------------------------------------------------------------------------
# Unsupported ranking algorithm
# --------------------------------------------------------------------------

ALGORITHM = "se esperaba un algoritmo de ranking ('BM25' o 'TF_IDF')"
INDEX_TYPE = "se esperaba 'BTREE', 'HASH', 'RTREE', 'GIN' o 'SPIMI'"


@pytest.mark.parametrize("sql, fragment", [
    pytest.param(f"{SEARCH} USING COSENO;", ALGORITHM, id="match-using-unknown-name"),
    pytest.param(f"{RANKED} USING TFIDF;", ALGORITHM, id="match-using-misspelled-tf-idf"),
    pytest.param(f"{SEARCH} USING;", ALGORITHM, id="match-using-without-name"),
    pytest.param("CREATE INDEX i ON docs USING FOO (contenido);", INDEX_TYPE, id="index-using-unknown-type"),
    pytest.param("CREATE INDEX i ON docs USING BM25 (contenido);", INDEX_TYPE, id="ranking-name-is-not-an-index-type"),
])
def test_unsupported_algorithm_is_rejected(engine, sql, fragment):
    assert_rejected(engine, sql, fragment)


@pytest.mark.parametrize("name, same_as", [("bm25", "BM25"), ("Tf_Idf", "TF_IDF")])
def test_algorithm_names_are_case_insensitive(engine, name, same_as):
    def ids(algorithm):
        rows = execute(
            f"SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'indices busqueda') "
            f"USING {algorithm} ORDER BY r DESC;",
            engine,
        )
        return [(row.values[0].data, row.values[1].data) for row in rows]

    assert ids(name) == ids(same_as) != []


# --------------------------------------------------------------------------
# Wrong column
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    pytest.param("SELECT id FROM docs WHERE MATCH(nope, 'indice');", id="match"),
    pytest.param("SELECT id, SCORE(nope, 'indice') AS r FROM docs;", id="score"),
    pytest.param(f"{SEARCH} ORDER BY nope;", id="order-by"),
    pytest.param(f"{SEARCH} ORDER BY relevancia DESC;", id="order-by-alias-that-was-not-selected"),
    pytest.param("CREATE INDEX i ON docs USING GIN (nope);", id="create-index"),
])
def test_unknown_column_is_rejected(engine, sql):
    assert_rejected(engine, sql, "La columna '" + re.search(r"nope|relevancia", sql).group() + "' no existe en la tabla 'docs'")


@pytest.mark.parametrize("sql", [
    pytest.param("SELECT id FROM nope WHERE MATCH(contenido, 'indice');", id="select"),
    pytest.param("CREATE INDEX i ON nope USING SPIMI (contenido);", id="create-index"),
])
def test_unknown_table_is_rejected(engine, sql):
    # raised by the semantic check or, for CREATE INDEX, by the catalog itself
    with pytest.raises((QueryError, TableNotFoundError), match="no existe"):
        execute(sql, engine)


# --------------------------------------------------------------------------
# Wrong type
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql, function, column, type_name", [
    ("SELECT id FROM docs WHERE MATCH(id, 'indice');", "MATCH", "id", "integer"),
    ("SELECT id, SCORE(id, 'indice') AS r FROM docs;", "SCORE", "id", "integer"),
    ("SELECT id FROM lugares WHERE MATCH(ubicacion, 'indice');", "MATCH", "ubicacion", "point"),
])
def test_match_and_score_require_a_text_column(engine, sql, function, column, type_name):
    assert_rejected(
        engine, sql,
        f"{function} requiere una columna de texto (CHAR, VARCHAR o TEXT); '{column}' es {type_name}",
    )


@pytest.mark.parametrize("index_type", ["GIN", "SPIMI"])
def test_text_indexes_require_a_text_column(engine, index_type):
    assert_rejected(
        engine, f"CREATE INDEX i ON docs USING {index_type} (id);",
        "GIN y SPIMI requieren una columna de tipo texto (TEXT, VARCHAR o CHAR), no integer",
    )


@pytest.mark.parametrize("sql", [
    "SELECT id FROM docs WHERE MATCH(contenido, 5);",
    "SELECT id, SCORE(contenido, 5) AS r FROM docs;",
])
def test_the_query_must_be_a_quoted_string(engine, sql):
    with pytest.raises(QueryError, match="Error sintáctico"):
        execute(sql, engine)


# --------------------------------------------------------------------------
# Queries without indexable terms, and malformed MATCH / SCORE
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["", "...", "de la", "   "])
def test_match_needs_at_least_one_indexable_term(engine, query):
    assert_rejected(
        engine, f"SELECT id FROM docs WHERE MATCH(contenido, '{query}');",
        "MATCH requiere al menos un término de búsqueda",
    )


@pytest.mark.parametrize("query", ["", "...", "de la"])  # empty, punctuation, only stopwords
def test_score_needs_at_least_one_indexable_term(engine, query):
    assert_rejected(
        engine, f"SELECT id, SCORE(contenido, '{query}') AS r FROM docs;",
        "SCORE requiere al menos un término de búsqueda",
    )


@pytest.mark.parametrize("sql, fragment", [
    pytest.param("SELECT id FROM docs WHERE MATCH(contenido);",
                 "se esperaba ',' y la consulta: MATCH(columna, 'consulta')", id="match-without-query"),
    pytest.param("SELECT id, SCORE(contenido) AS r FROM docs;",
                 "se esperaba ',' y la consulta (o SCORE() sin argumentos", id="score-without-query"),
])
def test_malformed_match_and_score_are_syntax_errors(engine, sql, fragment):
    assert_rejected(engine, sql, "Error sintáctico: " + fragment)


# --------------------------------------------------------------------------
# SCORE() usage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql, fragment", [
    pytest.param("SELECT SCORE() AS r FROM docs;",
                 "SCORE() sin argumentos requiere un MATCH(columna, 'consulta') en el WHERE",
                 id="bare-score-without-match"),
    pytest.param(f"{RANKED} USING TF_IDF;".replace("SCORE()", "SCORE(contenido, 'indice')"),
                 "SCORE(columna, 'consulta') siempre usa BM25 y el MATCH pide USING TF_IDF",
                 id="explicit-score-with-tf-idf-match"),
    pytest.param("SELECT SCORE() AS a, SCORE() AS b FROM docs WHERE MATCH(contenido, 'indice');",
                 "Solo se admite un SCORE(...) por consulta", id="two-scores"),
    pytest.param("SELECT SCORE() AS id FROM docs WHERE MATCH(contenido, 'indice');",
                 "El alias 'id' de SCORE choca con una columna de 'docs'", id="alias-clashes-with-a-column"),
    pytest.param("SELECT *, id FROM docs WHERE MATCH(contenido, 'indice');",
                 "Después de '*' solo se admite SCORE(...)", id="extra-column-after-star"),
])
def test_invalid_score_usage_is_rejected(engine, sql, fragment):
    assert_rejected(engine, sql, fragment)


# --------------------------------------------------------------------------
# Combinations the search does not support
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql, fragment", [
    pytest.param(
        "SELECT docs.id, SCORE() AS r FROM docs JOIN otros ON docs.id = otros.doc_id "
        "WHERE MATCH(docs.contenido, 'indice');",
        "SCORE/MATCH no se admiten junto con JOIN", id="join"),
    pytest.param(
        "SELECT titulo, COUNT(*), SCORE() AS r FROM docs "
        "WHERE MATCH(contenido, 'indice') GROUP BY titulo;",
        "SCORE/MATCH no se admiten junto con GROUP BY ni agregados", id="group-by"),
    pytest.param("SELECT COUNT(*) FROM docs WHERE MATCH(contenido, 'indice');",
                 "SCORE/MATCH no se admiten junto con GROUP BY ni agregados", id="aggregate"),
    pytest.param(
        "SELECT nombre, SCORE() AS r FROM lugares WHERE MATCH(nombre, 'indice') "
        "ORDER BY DISTANCIA(ubicacion, POINT(0, 0));",
        "SCORE/MATCH no se admiten junto con ORDER BY DISTANCIA", id="order-by-distance"),
    pytest.param("DELETE FROM docs WHERE MATCH(contenido, 'indice');",
                 "MATCH solo se admite en el WHERE de un SELECT", id="delete"),
    pytest.param("UPDATE docs SET titulo = 'x' WHERE MATCH(contenido, 'indice');",
                 "MATCH solo se admite en el WHERE de un SELECT", id="update"),
])
def test_unsupported_combinations_are_rejected(engine, sql, fragment):
    assert_rejected(engine, sql, fragment)


@pytest.mark.parametrize("limit, fragment", [
    ("0", "LIMIT debe ser un entero positivo"),
    ("-3", "LIMIT debe ser un entero positivo"),
    ("1.5", "LIMIT requiere un entero"),
    ("abc", "se esperaba 'NUM'"),
])
def test_limit_must_be_a_positive_integer(engine, limit, fragment):
    with pytest.raises(QueryError, match=re.escape(fragment)):
        execute(f"{RANKED} ORDER BY relevancia DESC LIMIT {limit};", engine)


# --------------------------------------------------------------------------
# Missing index
# --------------------------------------------------------------------------

def test_dropping_an_index_that_does_not_exist_is_an_error(fresh):
    assert_rejected(fresh, "DROP INDEX no_existe;", "El índice 'no_existe' no existe")


def test_drop_index_if_exists_tolerates_a_missing_index(fresh):
    assert execute("DROP INDEX IF EXISTS no_existe;", fresh) is None
    assert execute("DROP INDEX IF EXISTS no_existe;", fresh) is None


def test_dropping_an_index_twice_fails_the_second_time(fresh):
    execute("CREATE INDEX idx_docs ON docs USING SPIMI (contenido);", fresh)
    execute("DROP INDEX idx_docs;", fresh)

    assert_rejected(fresh, "DROP INDEX idx_docs;", "El índice 'idx_docs' no existe")
    assert execute("DROP INDEX IF EXISTS idx_docs;", fresh) is None


def test_search_does_not_need_an_index(fresh):
    # section 11.3: MATCH without an index is allowed and uses a full scan
    sql = "SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'indices busqueda') ORDER BY r DESC;"
    without_index = [(r.values[0].data, r.values[1].data) for r in execute(sql, fresh)]
    assert without_index and without_index[0][0] == 1

    execute("CREATE INDEX idx_docs ON docs USING SPIMI (contenido);", fresh)
    assert "TextIndexScan" in plan_nodes(execute("EXPLAIN " + sql, fresh).description)
    with_index = [(r.values[0].data, r.values[1].data) for r in execute(sql, fresh)]

    execute("DROP INDEX idx_docs;", fresh)  # the index is gone: same answer, now by scan
    nodes = plan_nodes(execute("EXPLAIN " + sql, fresh).description)
    assert "TextSearch" in nodes and "TextIndexScan" not in nodes
    after_drop = [(r.values[0].data, r.values[1].data) for r in execute(sql, fresh)]

    assert [i for i, _ in with_index] == [i for i, _ in without_index] == [i for i, _ in after_drop]
    assert [s for _, s in with_index] == pytest.approx([s for _, s in without_index], rel=1e-9)


def test_duplicate_index_name_is_rejected(fresh):
    execute("CREATE INDEX idx_docs ON docs USING SPIMI (contenido);", fresh)
    # the catalog rejects it; depending on the layer it arrives wrapped or as a ValueError
    with pytest.raises((QueryError, ValueError), match="Ya existe un índice con nombre 'idx_docs'"):
        execute("CREATE INDEX idx_docs ON docs USING GIN (titulo);", fresh)


@pytest.mark.parametrize("sql", [
    "CREATE INDEX idx_bad ON docs USING SPIMI (id);",
    "CREATE INDEX idx_bad ON docs USING GIN (nope);",
])
def test_a_rejected_create_index_leaves_nothing_behind(fresh, sql):
    with pytest.raises(QueryError):
        execute(sql, fresh)

    assert not fresh.index_exists("idx_bad")
    # the name is still free
    execute("CREATE INDEX idx_bad ON docs USING SPIMI (contenido);", fresh)
    assert fresh.index_exists("idx_bad")


def test_the_engine_keeps_working_after_errors(fresh):
    for sql in ("SELECT id FROM docs WHERE MATCH(nope, 'indice');",
                f"{SEARCH} USING COSENO;",
                "SELECT SCORE() AS r FROM docs;"):
        with pytest.raises(QueryError):
            execute(sql, fresh)

    rows = execute("SELECT id FROM docs WHERE MATCH(contenido, 'ranking');", fresh)
    assert [row.values[0].data for row in rows] == [3]