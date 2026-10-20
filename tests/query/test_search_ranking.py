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


def seed_catalog(tmp_path):
    catalog = make_catalog(tmp_path)
    execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, titulo VARCHAR(40), contenido TEXT);",
        catalog,
    )
    # the best document for 'bases datos' is inserted LAST, so a LIMIT applied
    # before ranking would return the wrong row
    execute(
        "INSERT INTO docs VALUES "
        "(1, 'Intro', 'introduccion a las bases de datos'), "
        "(2, 'SO', 'sistemas operativos'), "
        "(3, 'BD', 'bases de datos relacionales bases de datos bases');",
        catalog,
    )
    return catalog


def values(rows):
    return [[value.data for value in row.values] for row in rows]


def test_select_star_keeps_document_columns_and_adds_alias(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT *, SCORE(contenido, 'bases datos') AS relevancia FROM docs "
        "WHERE contenido MATCH 'bases datos' ORDER BY relevancia DESC LIMIT 5;",
        catalog,
    )

    assert [row[0].data for row in rows] == [3, 1]          # filtered + ranked
    assert all(len(row.values) == 4 for row in rows)         # id, titulo, contenido, alias
    scores = [row[3].data for row in rows]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[1] > 0


def test_ranking_applies_before_limit(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT id, SCORE(contenido, 'bases datos') AS relevancia FROM docs "
        "WHERE contenido MATCH 'bases datos' ORDER BY relevancia DESC LIMIT 1;",
        catalog,
    )

    assert [row[0].data for row in rows] == [3]


def test_projected_columns_expose_score_alias(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT titulo, SCORE(contenido, 'bases') AS puntaje FROM docs "
        "WHERE contenido MATCH 'bases' ORDER BY puntaje DESC;",
        catalog,
    )

    assert [row[0].data for row in rows] == ["BD", "Intro"]
    assert all(len(row.values) == 2 for row in rows)


def test_score_without_match_scores_every_row(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT id, SCORE(contenido, 'operativos') AS r FROM docs ORDER BY r DESC;",
        catalog,
    )

    assert values(rows)[0][0] == 2
    assert len(rows) == 3


def test_explain_shows_text_search_node(tmp_path):
    catalog = seed_catalog(tmp_path)

    result = execute(
        "EXPLAIN SELECT *, SCORE(contenido, 'bases') AS r FROM docs "
        "WHERE contenido MATCH 'bases' ORDER BY r DESC LIMIT 2;",
        catalog,
    )

    text = str(result.description)
    assert "TextSearch" in text and "Sort" in text and "Limit" in text


@pytest.mark.parametrize("sql", [
    "SELECT *, SCORE(id, 'x') AS r FROM docs;",
    "SELECT *, SCORE(contenido, 'x') AS titulo FROM docs;",
    "DELETE FROM docs WHERE contenido MATCH 'x';",
])
def test_invalid_search_queries_raise_query_error(tmp_path, sql):
    with pytest.raises(QueryError):
        execute(sql, seed_catalog(tmp_path))