import pytest

from index.text import rid_to_doc_id
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from query.query_engine import execute


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


DOCS = [
    (1, "Intro", "introduccion a las bases de datos"),
    (2, "SO", "sistemas operativos"),
    (3, "BD", "bases de datos relacionales bases de datos bases"),
    (4, "Nada", None),
]


@pytest.fixture
def catalog(tmp_path):
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, titulo VARCHAR(40), contenido TEXT);", catalog)
    for doc_id, titulo, contenido in DOCS:
        text = "NULL" if contenido is None else f"'{contenido}'"
        execute(f"INSERT INTO docs VALUES ({doc_id}, '{titulo}', {text});", catalog)
    execute("CREATE INDEX idx_docs_contenido ON docs USING SPIMI (contenido);", catalog)
    return catalog


def index_scores(catalog, query):
    """{id: score} straight from the catalog's inverted index."""
    index = catalog.get_text_index("docs", "contenido")
    ranked = dict(index.score_bm25(catalog.text_processor.process(query), top_k=None))
    storage = catalog.get_storage("docs")
    return {
        record[0].data: ranked[rid_to_doc_id(rid.page_id, rid.slot)]
        for rid, record in storage.scan_with_rid()
        if rid_to_doc_id(rid.page_id, rid.slot) in ranked
    }


def explain(catalog, sql):
    return str(execute("EXPLAIN " + sql, catalog).description)


MATCH_AND_SCORE = (
    "SELECT id, SCORE() AS relevancia FROM docs "
    "WHERE MATCH(contenido, 'bases datos') USING BM25 ORDER BY relevancia DESC"
)


def test_match_with_index_uses_text_index_scan(catalog):
    plan = explain(catalog, MATCH_AND_SCORE + ";")
    assert "TextIndexScan" in plan
    assert "TextSearch" not in plan and "SeqScan" not in plan


def test_sql_scores_are_the_index_scores(catalog):
    rows = execute(MATCH_AND_SCORE + ";", catalog)
    expected = index_scores(catalog, "bases datos")
    assert {row[0].data: row[1].data for row in rows} == pytest.approx(expected)
    assert [row[0].data for row in rows] == [3, 1]


def test_ranking_applies_before_limit(catalog):
    rows = execute(MATCH_AND_SCORE + " LIMIT 1;", catalog)
    assert [row[0].data for row in rows] == [3]


def test_match_without_score_filters_by_index(catalog):
    rows = execute("SELECT id FROM docs WHERE MATCH(contenido, 'operativos');", catalog)
    assert [row[0].data for row in rows] == [2]
    assert "TextIndexScan" in explain(catalog, "SELECT id FROM docs WHERE contenido MATCH 'operativos';")


def test_select_star_keeps_columns_and_appends_score(catalog):
    rows = execute(
        "SELECT *, SCORE() AS r FROM docs WHERE MATCH(contenido, 'bases') ORDER BY r DESC;",
        catalog,
    )
    assert all(len(row.values) == 4 for row in rows)
    assert [row[1].data for row in rows] == ["BD", "Intro"]


def test_score_without_match_scores_every_row_from_the_index(catalog):
    rows = execute(
        "SELECT id, SCORE(contenido, 'operativos') AS r FROM docs ORDER BY id;",
        catalog,
    )
    scores = {row[0].data: row[1].data for row in rows}
    assert set(scores) == {1, 2, 3, 4}                 # NULL row included, scored 0
    assert scores[2] == pytest.approx(index_scores(catalog, "operativos")[2])
    assert scores[1] == scores[3] == scores[4] == 0.0
    assert "TextIndexScan" in explain(
        catalog, "SELECT id, SCORE(contenido, 'operativos') AS r FROM docs;",
    )


def test_unknown_terms_return_no_rows(catalog):
    assert execute("SELECT id FROM docs WHERE MATCH(contenido, 'inexistente');", catalog) == []


def test_index_is_used_after_inserts_and_deletes(catalog):
    execute("INSERT INTO docs VALUES (5, 'Nuevo', 'bases bases bases');", catalog)
    execute("DELETE FROM docs WHERE id = 3;", catalog)
    rows = execute(MATCH_AND_SCORE + ";", catalog)
    scores = {row[0].data: row[1].data for row in rows}
    assert set(scores) == {1, 5}                       # deleted doc gone, new doc found
    assert scores == pytest.approx(index_scores(catalog, "bases datos"))
    assert [row[1].data for row in rows] == sorted(scores.values(), reverse=True)


def test_score_over_another_query_falls_back_to_text_search(catalog):
    plan = explain(
        catalog,
        "SELECT id, SCORE(contenido, 'sistemas') AS r FROM docs WHERE MATCH(contenido, 'bases');",
    )
    assert "TextSearch" in plan and "TextIndexScan" not in plan


def test_column_without_text_index_falls_back_to_text_search(tmp_path):
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE notas (id INTEGER PRIMARY KEY, texto TEXT);", catalog)
    execute("INSERT INTO notas VALUES (1, 'bases de datos');", catalog)
    plan = explain(catalog, "SELECT id FROM notas WHERE MATCH(texto, 'bases');")
    assert "TextSearch" in plan and "TextIndexScan" not in plan
    assert [row[0].data for row in execute("SELECT id FROM notas WHERE MATCH(texto, 'bases');", catalog)] == [1]
