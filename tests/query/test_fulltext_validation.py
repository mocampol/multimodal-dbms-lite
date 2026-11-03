"""
Semantic validation of full-text queries, end to end through execute():
every invalid query must surface as a QueryError with an understandable
message (docs/text_search_policy.md §11.2), and MATCH over a column
without an index is allowed via a full scan (§11.3).
"""

import pytest

from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from query.query_engine import QueryError, execute


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


@pytest.fixture
def catalog(tmp_path):
    catalog = make_catalog(tmp_path)
    execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, titulo VARCHAR(40), contenido TEXT, "
        "ubicacion POINT);",
        catalog,
    )
    execute(
        "INSERT INTO docs (id, titulo, contenido) VALUES "
        "(1, 'Intro', 'introduccion a las bases de datos'), "
        "(2, 'SO', 'sistemas operativos'), "
        "(3, 'BD', 'bases de datos relacionales bases de datos bases');",
        catalog,
    )
    return catalog


def rejects(catalog, sql, fragment):
    with pytest.raises(QueryError) as error:
        execute(sql, catalog)
    assert fragment in str(error.value)


W = "SELECT id FROM docs WHERE "


# Columnas: deben existir y ser textuales

@pytest.mark.parametrize("sql, fragment", [
    (W + "MATCH(nope, 'x');", "La columna 'nope' no existe"),
    (W + "MATCH(otra.contenido, 'x');", "La tabla 'otra' no participa en la consulta"),
    ("SELECT SCORE(nope, 'x') AS r FROM docs;", "La columna 'nope' no existe"),
])
def test_rejects_unknown_columns(catalog, sql, fragment):
    rejects(catalog, sql, fragment)


@pytest.mark.parametrize("sql", [
    W + "MATCH(id, 'x');",
    W + "MATCH(ubicacion, 'x');",
    W + "id MATCH 'x';",
    "SELECT SCORE(id, 'x') AS r FROM docs;",
])
def test_rejects_non_text_columns(catalog, sql):
    rejects(catalog, sql, "requiere una columna de texto (CHAR, VARCHAR o TEXT)")


def test_accepts_every_text_type(catalog):
    assert execute(W + "MATCH(titulo, 'intro');", catalog)[0][0].data == 1   # VARCHAR
    assert execute(W + "MATCH(contenido, 'sistemas');", catalog)[0][0].data == 2  # TEXT


# Algoritmos de ranking

@pytest.mark.parametrize("sql", [
    W + "MATCH(contenido, 'x') USING COSENO;",
    W + "MATCH(contenido, 'x') USING BTREE;",
    W + "MATCH(contenido, 'x') USING 'BM25';",
    W + "MATCH(contenido, 'x') USING;",
])
def test_rejects_unsupported_algorithms(catalog, sql):
    rejects(catalog, sql, "algoritmo de ranking ('BM25' o 'TF_IDF')")


@pytest.mark.parametrize("using", ["", "USING BM25", "USING TF_IDF", "using tf_idf"])
def test_accepts_supported_algorithms(catalog, using):
    rows = execute(
        f"SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'bases') {using} ORDER BY r DESC;",
        catalog,
    )
    assert [row[0].data for row in rows] == [3, 1]


# Formato de la consulta y cláusulas incompletas

@pytest.mark.parametrize("sql, fragment", [
    (W + "MATCH(contenido);", "MATCH(columna, 'consulta')"),
    (W + "MATCH(contenido, 5);", "consulta de MATCH entre comillas simples"),
    (W + "MATCH(contenido, bases);", "consulta de MATCH entre comillas simples"),
    (W + "MATCH(contenido, 'x';", "')'"),
    (W + "MATCH('x');", "'ID'"),
    (W + "contenido MATCH;", "consulta de MATCH"),
    ("SELECT SCORE(contenido) AS r FROM docs;", "SCORE() sin argumentos"),
    ("SELECT SCORE(contenido, 5) AS r FROM docs;", "consulta de SCORE"),
])
def test_rejects_malformed_clauses_as_syntax_errors(catalog, sql, fragment):
    rejects(catalog, sql, "Error sintáctico")
    rejects(catalog, sql, fragment)


@pytest.mark.parametrize("query", ["", "   ", "...", "¡¿?!", "de la", "el y de"])
def test_rejects_queries_without_indexable_terms(catalog, query):
    rejects(catalog, W + f"MATCH(contenido, '{query}');", "MATCH requiere al menos un término")
    rejects(catalog, f"SELECT SCORE(contenido, '{query}') AS r FROM docs;", "SCORE requiere al menos un término")


@pytest.mark.parametrize("sql", [
    "SELECT id, SCORE() AS r FROM docs;",
    "SELECT id, SCORE() AS r FROM docs WHERE id = 1;",
    "SELECT *, SCORE() FROM docs ORDER BY score DESC;",
])
def test_rejects_score_without_match(catalog, sql):
    rejects(catalog, sql, "SCORE() sin argumentos requiere un MATCH(columna, 'consulta')")


# Uso de SCORE() y USING

def test_rejects_explicit_score_with_a_different_using(catalog):
    rejects(
        catalog,
        "SELECT SCORE(contenido, 'bases') AS r FROM docs WHERE MATCH(contenido, 'bases') USING TF_IDF;",
        "siempre usa BM25 y el MATCH pide USING TF_IDF",
    )


@pytest.mark.parametrize("sql, fragment", [
    ("SELECT SCORE() AS a, SCORE() AS b FROM docs WHERE MATCH(contenido, 'x');",
     "Solo se admite un SCORE"),
    ("SELECT SCORE() AS titulo FROM docs WHERE MATCH(contenido, 'x');",
     "El alias 'titulo' de SCORE choca"),
    ("SELECT *, id FROM docs WHERE MATCH(contenido, 'x');",
     "Después de '*' solo se admite SCORE"),
    ("SELECT id FROM docs WHERE MATCH(contenido, 'x') ORDER BY relevancia;",
     "La columna 'relevancia' no existe"),
])
def test_rejects_invalid_score_usage(catalog, sql, fragment):
    rejects(catalog, sql, fragment)


@pytest.mark.parametrize("sql, fragment", [
    ("DELETE FROM docs WHERE MATCH(contenido, 'x');", "MATCH solo se admite en el WHERE de un SELECT"),
    ("UPDATE docs SET titulo = 'z' WHERE MATCH(contenido, 'x');", "MATCH solo se admite en el WHERE de un SELECT"),
    ("SELECT docs.id FROM docs JOIN docs ON docs.id = docs.id WHERE MATCH(contenido, 'x');",
     "no se admiten junto con JOIN"),
    ("SELECT COUNT(*) FROM docs WHERE MATCH(contenido, 'x');", "no se admiten junto con GROUP BY ni agregados"),
    ("SELECT titulo FROM docs WHERE MATCH(contenido, 'x') GROUP BY titulo;",
     "no se admiten junto con GROUP BY ni agregados"),
    ("SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'x') "
     "ORDER BY DISTANCIA(ubicacion, POINT(0, 0));", "no se admiten junto con ORDER BY DISTANCIA"),
])
def test_rejects_unsupported_combinations(catalog, sql, fragment):
    rejects(catalog, sql, fragment)


def test_rejected_queries_do_not_modify_data(catalog):
    with pytest.raises(QueryError):
        execute("DELETE FROM docs WHERE MATCH(contenido, 'bases');", catalog)
    assert len(execute("SELECT * FROM docs;", catalog)) == 3


# MATCH sin índice creado: se permite con escaneo completo

def test_match_without_index_runs_with_a_full_scan(catalog):
    sql = (
        "SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'bases datos') USING BM25 "
        "ORDER BY r DESC LIMIT 5;"
    )
    plan = str(execute("EXPLAIN " + sql, catalog).description)
    assert "TextSearch" in plan and "SeqScan" in plan and "TextIndexScan" not in plan
    assert [row[0].data for row in execute(sql, catalog)] == [3, 1]


def test_same_results_before_and_after_creating_the_index(catalog):
    sql = (
        "SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'bases datos') "
        "ORDER BY r DESC;"
    )
    without_index = execute(sql, catalog)
    execute("CREATE INDEX idx_docs_contenido ON docs USING SPIMI (contenido);", catalog)
    assert "TextIndexScan" in str(execute("EXPLAIN " + sql, catalog).description)
    with_index = execute(sql, catalog)
    assert [row[0].data for row in with_index] == [row[0].data for row in without_index]
    assert [row[1].data for row in with_index] == pytest.approx([row[1].data for row in without_index])


def test_match_on_a_column_without_index_when_another_has_one(catalog):
    execute("CREATE INDEX idx_docs_contenido ON docs USING SPIMI (contenido);", catalog)
    plan = str(execute("EXPLAIN " + W + "MATCH(titulo, 'intro');", catalog).description)
    assert "TextSearch" in plan and "TextIndexScan" not in plan
    assert [row[0].data for row in execute(W + "MATCH(titulo, 'intro');", catalog)] == [1]
