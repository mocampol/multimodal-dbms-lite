import pytest

from common.record import Record
from common.schema import Column, Schema
from common.value import DataType, Value
from catalog.table_metadata import StorageType  # noqa: F401  (visitor import path)
from query.executor.executor import run_plan
from query.executor.plan_node import PlanNode
from query.executor.processing.sort import Sort
from query.executor.processing.text_search import TextSearch
from query.parser.ast_nodes import MatchExp, ScoreSpec, SelectStm
from query.parser.exceptions import SemanticError
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.parser.visitor import InMemoryCatalog, SemanticVisitor
from index.text import InvertedIndex, default_text_processor


class RecordsNode(PlanNode):
    def __init__(self, records):
        self.records = records
        self.index = 0

    def open(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.records):
            return None
        self.index += 1
        return self.records[self.index - 1]

    def close(self):
        pass


DOCS = [
    (1, "introduccion a las bases de datos"),
    (2, "sistemas operativos"),
    (3, "bases de datos relacionales bases de datos bases"),
    (4, None),
]


def make_schema():
    return Schema("docs", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("contenido", DataType.TEXT),
    ])


def make_records():
    return [
        Record([Value(DataType.INTEGER, i), Value(DataType.TEXT, t)])
        for i, t in DOCS
    ]


def out_schema(schema):
    return Schema("docs", list(schema.columns) + [
        Column("relevancia", DataType.DOUBLE_PRECISION, nullable=False)
    ])


def test_processor_normalizes_case_accents_and_drops_stopwords():
    process = default_text_processor().process
    assert process("Introducción, a las BASES_de-datos!") == ["introduccion", "bases", "datos"]
    assert process(None) == []


def test_scores_are_those_of_an_inverted_index_over_the_same_rows():
    schema = make_schema()
    node = TextSearch(
        RecordsNode(make_records()), schema, out_schema(schema),
        score=("contenido", "bases datos", "relevancia"),
    )
    scores = {r[0].data: r[2].data for r in run_plan(node)}

    process = default_text_processor().process
    index = InvertedIndex()
    for doc_id, text in DOCS:
        if text is not None:  # NULL is not a document, as in the catalog
            index.add_document(doc_id, process(text))
    expected = dict(index.score_bm25(process("bases datos"), top_k=None))

    assert scores == {doc_id: expected.get(doc_id, 0.0) for doc_id, _ in DOCS}
    assert scores[3] > scores[1] > 0
    assert scores[2] == 0.0 and scores[4] == 0.0


def test_text_search_filters_and_appends_score():
    schema = make_schema()
    node = TextSearch(
        RecordsNode(make_records()), schema, out_schema(schema),
        match=("contenido", "bases datos"),
        score=("contenido", "bases datos", "relevancia"),
    )
    rows = list(run_plan(node))
    assert [r[0].data for r in rows] == [1, 3]
    assert all(len(r) == 3 and r[2].data_type == DataType.DOUBLE_PRECISION for r in rows)


def test_score_without_match_keeps_every_document():
    schema = make_schema()
    node = TextSearch(
        RecordsNode(make_records()), schema, out_schema(schema),
        score=("contenido", "bases", "relevancia"),
    )
    rows = list(run_plan(node))
    assert [r[0].data for r in rows] == [1, 2, 3, 4]
    assert rows[1][2].data == 0.0


def test_rank_then_limit_uses_real_score():
    from query.executor.processing.limit import Limit  # may be the real module in the repo
    schema = make_schema()
    full = out_schema(schema)
    node = TextSearch(
        RecordsNode(make_records()), schema, full,
        match=("contenido", "bases datos"),
        score=("contenido", "bases datos", "relevancia"),
    )
    node = Sort(node, ["relevancia"], full, descending=[True])
    node = Limit(node, 1)
    rows = list(run_plan(node))
    assert [r[0].data for r in rows] == [3]  # best doc, not first-inserted


def test_sort_descending_with_nulls_and_runs():
    schema = Schema("t", [Column("v", DataType.INTEGER, nullable=True)])
    records = [Record([Value(DataType.INTEGER, v)]) for v in [3, None, 1, 2]]
    node = Sort(RecordsNode(records), ["v"], schema, memory_records=2, descending=[True])
    assert [r[0].data for r in run_plan(node)] == [3, 2, 1, None]


def test_sort_descending_strings():
    schema = Schema("t", [Column("s", DataType.TEXT)])
    records = [Record([Value(DataType.TEXT, v)]) for v in ["b", "c", "a"]]
    node = Sort(RecordsNode(records), ["s"], schema, memory_records=2, descending=[True])
    assert [r[0].data for r in run_plan(node)] == ["c", "b", "a"]


def parse(sql):
    return Parser(Scanner(sql)).parse_sql_statements()[0]


QUERY_STAR = (
    "SELECT *, SCORE(contenido, 'bases datos') AS relevancia FROM docs "
    "WHERE contenido MATCH 'bases datos' ORDER BY relevancia DESC LIMIT 5;"
)


def test_parser_builds_score_match_and_desc():
    stm = parse(QUERY_STAR)
    assert isinstance(stm, SelectStm)
    assert stm.columns[0] == "*" and isinstance(stm.columns[1], ScoreSpec)
    assert stm.columns[1].alias == "relevancia"
    assert isinstance(stm.where_cond, MatchExp)
    assert stm.order_by.columns == ["relevancia"] and stm.order_by.descending == [True]
    assert stm.limit.value == 5


def test_parser_alias_without_as_and_default_alias():
    assert parse("SELECT id, SCORE(contenido, 'x') rel FROM docs;").columns[1].alias == "rel"
    assert parse("SELECT id, SCORE(contenido, 'x') FROM docs;").columns[1].alias == "score"


def make_catalog():
    catalog = InMemoryCatalog()
    catalog.define_table(make_schema())
    return catalog


def test_semantic_accepts_both_query_shapes():
    for sql in (
        QUERY_STAR,
        "SELECT id, SCORE() AS r FROM docs WHERE MATCH(contenido, 'x') USING TF_IDF;",
        "SELECT id, SCORE(contenido, 'x') AS r FROM docs WHERE MATCH(contenido, 'x') USING BM25;",
        "SELECT id FROM docs WHERE MATCH(contenido, 'x') USING TF_IDF;",
        "SELECT id, SCORE(contenido, 'bases') AS relevancia FROM docs "
        "WHERE contenido MATCH 'bases' ORDER BY relevancia DESC LIMIT 3;",
    ):
        SemanticVisitor(make_catalog()).check(parse(sql))


@pytest.mark.parametrize("sql", [
    "SELECT *, SCORE(id, 'x') AS r FROM docs;",                       # no es texto
    "SELECT *, SCORE(contenido, 'x') AS id FROM docs;",               # alias choca
    "SELECT *, SCORE(contenido, '...') AS r FROM docs;",              # sin términos
    "SELECT id FROM docs WHERE MATCH(contenido, 'de la');",           # solo stopwords
    "SELECT id, SCORE() AS r FROM docs;",                             # SCORE() sin MATCH
    "SELECT id, SCORE() AS r FROM docs WHERE id = 1;",                # WHERE sin MATCH
    "SELECT id, SCORE(contenido, 'x') AS r FROM docs "
    "WHERE MATCH(contenido, 'x') USING TF_IDF;",                      # SCORE explícito + TF_IDF
    "SELECT *, id FROM docs;",                                        # extra tras *
    "DELETE FROM docs WHERE contenido MATCH 'x';",                    # MATCH en DELETE
    "SELECT id FROM docs WHERE contenido MATCH 'x' ORDER BY nope;",   # columna inexistente
])
def test_semantic_rejects_invalid(sql):
    with pytest.raises(SemanticError):
        SemanticVisitor(make_catalog()).check(parse(sql))


def test_plain_queries_still_parse_without_direction():
    stm = parse("SELECT * FROM docs ORDER BY id, contenido;")
    assert stm.order_by.descending == [False, False]