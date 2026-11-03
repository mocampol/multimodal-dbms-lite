import pytest

from query.parser.ast_nodes import (
    BinaryExp,
    DeleteStm,
    MatchExp,
    RankingAlgorithm,
    ScoreSpec,
    SelectStm,
)
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.parser.token_ import TokenType


def parse(sql):
    return Parser(Scanner(sql)).parse_sql_statements()[0]


def token_types(source):
    scanner = Scanner(source)
    types = []
    while (token := scanner.next_token()).type != TokenType.END:
        types.append(token.type)
    return types


FULL_QUERY = (
    "SELECT id, SCORE() AS relevancia FROM docs "
    "WHERE MATCH(contenido, 'bases de datos') USING BM25 "
    "ORDER BY relevancia DESC LIMIT 10;"
)


# Lexer

def test_scanner_recognizes_ranking_keywords_case_insensitive():
    assert token_types("BM25 bm25 TF_IDF tf_idf") == [
        TokenType.BM25, TokenType.BM25, TokenType.TF_IDF, TokenType.TF_IDF,
    ]


def test_match_and_score_stay_contextual_identifiers():
    assert token_types("MATCH SCORE") == [TokenType.ID, TokenType.ID]


# MATCH(columna, 'consulta') [USING ...]

def test_full_query_from_the_issue():
    stm = parse(FULL_QUERY)
    assert isinstance(stm, SelectStm)
    assert isinstance(stm.where_cond, MatchExp)
    assert stm.where_cond.left.value == "contenido"
    assert stm.where_cond.query == "bases de datos"
    assert stm.where_cond.ranking == RankingAlgorithm.BM25
    assert stm.order_by.columns == ["relevancia"] and stm.order_by.descending == [True]
    assert stm.limit.value == 10


@pytest.mark.parametrize("clause, ranking", [
    ("USING BM25", RankingAlgorithm.BM25),
    ("USING TF_IDF", RankingAlgorithm.TF_IDF),
    ("using tf_idf", RankingAlgorithm.TF_IDF),
    ("", RankingAlgorithm.BM25),
])
def test_match_ranking_algorithm(clause, ranking):
    stm = parse(f"SELECT * FROM docs WHERE MATCH(contenido, 'x') {clause};")
    assert stm.where_cond.ranking == ranking


def test_match_accepts_qualified_column():
    stm = parse("SELECT * FROM docs WHERE MATCH(docs.contenido, 'x');")
    assert stm.where_cond.left.value == "docs.contenido"


def test_legacy_match_form_still_parses_as_bm25():
    stm = parse("SELECT * FROM docs WHERE contenido MATCH 'x';")
    assert isinstance(stm.where_cond, MatchExp)
    assert stm.where_cond.left.value == "contenido"
    assert stm.where_cond.ranking == RankingAlgorithm.BM25


def test_column_named_match_is_not_the_function():
    stm = parse("SELECT * FROM t WHERE match = 3;")
    assert isinstance(stm.where_cond, BinaryExp)
    assert stm.where_cond.left.value == "match"


def test_match_in_delete_parses_and_is_left_to_the_visitor():
    stm = parse("DELETE FROM docs WHERE MATCH(contenido, 'x');")
    assert isinstance(stm, DeleteStm) and isinstance(stm.where_cond, MatchExp)


# SCORE() AS alias

def test_empty_score_binds_to_match_column_query_and_ranking():
    stm = parse(
        "SELECT *, SCORE() AS r FROM docs WHERE MATCH(contenido, 'bases') USING TF_IDF;"
    )
    score = stm.columns[1]
    assert isinstance(score, ScoreSpec) and score.is_bound
    assert (score.column, score.query, score.alias) == ("contenido", "bases", "r")
    assert score.ranking == RankingAlgorithm.TF_IDF


def test_empty_score_without_match_stays_unbound():
    score = parse("SELECT SCORE() AS r FROM docs;").columns[0]
    assert not score.is_bound and score.column is None and score.query is None


def test_empty_score_alias_without_as_and_default_alias():
    sql = "SELECT id, SCORE() {} FROM docs WHERE MATCH(contenido, 'x');"
    assert parse(sql.format("rel")).columns[1].alias == "rel"
    assert parse(sql.format("")).columns[1].alias == "score"


def test_explicit_score_keeps_its_own_arguments():
    stm = parse(
        "SELECT SCORE(titulo, 'otra') AS r FROM docs WHERE MATCH(contenido, 'x') USING TF_IDF;"
    )
    score = stm.columns[0]
    assert (score.column, score.query) == ("titulo", "otra")
    assert score.ranking == RankingAlgorithm.BM25


# ORDER BY relevancia [ASC|DESC] y LIMIT

@pytest.mark.parametrize("direction, descending", [
    ("DESC", True), ("desc", True), ("ASC", False), ("", False),
])
def test_order_by_score_alias_direction(direction, descending):
    stm = parse(
        "SELECT id, SCORE() AS relevancia FROM docs WHERE MATCH(contenido, 'x') "
        f"ORDER BY relevancia {direction} LIMIT 3;"
    )
    assert stm.order_by.columns == ["relevancia"]
    assert stm.order_by.descending == [descending]
    assert stm.limit.value == 3


def test_order_by_score_alias_and_tiebreak_column():
    stm = parse(
        "SELECT id, SCORE() AS relevancia FROM docs WHERE MATCH(contenido, 'x') USING TF_IDF "
        "ORDER BY relevancia DESC, id ASC LIMIT 5;"
    )
    assert stm.order_by.columns == ["relevancia", "id"]
    assert stm.order_by.descending == [True, False]
    assert stm.limit.value == 5


def test_limit_without_order_by():
    stm = parse("SELECT * FROM docs WHERE MATCH(contenido, 'x') USING BM25 LIMIT 2;")
    assert stm.order_by is None and stm.limit.value == 2


def test_reprs_show_the_new_syntax():
    stm = parse(FULL_QUERY)
    assert repr(stm.where_cond) == "MATCH(contenido, 'bases de datos') USING BM25"
    assert repr(parse("SELECT SCORE() AS r FROM docs;").columns[0]) == "SCORE() AS r"


# Errores sintácticos

@pytest.mark.parametrize("sql, fragment", [
    ("SELECT * FROM docs WHERE MATCH(contenido) USING BM25;", "MATCH(columna, 'consulta')"),
    ("SELECT * FROM docs WHERE MATCH(contenido, 5);", "consulta de MATCH entre comillas"),
    ("SELECT * FROM docs WHERE MATCH(contenido, 'x';", "')'"),
    ("SELECT * FROM docs WHERE MATCH(contenido, 'x') USING;", "algoritmo de ranking"),
    ("SELECT * FROM docs WHERE MATCH(contenido, 'x') USING COSENO;", "'BM25' o 'TF_IDF'"),
    ("SELECT * FROM docs WHERE MATCH(contenido, 'x') USING BTREE;", "algoritmo de ranking"),
    ("SELECT * FROM docs WHERE contenido MATCH 5;", "consulta de MATCH"),
    ("SELECT SCORE(contenido) AS r FROM docs;", "SCORE() sin argumentos"),
    ("SELECT SCORE(contenido, x) AS r FROM docs;", "consulta de SCORE"),
    ("SELECT SCORE( AS r FROM docs;", "'ID'"),
    ("SELECT SCORE() AS FROM docs;", "'ID'"),
    ("SELECT * FROM docs WHERE MATCH(contenido, 'x') ORDER BY r DESC LIMIT;", "'NUM'"),
])
def test_syntax_errors_are_descriptive(sql, fragment):
    with pytest.raises(RuntimeError, match="Error sintáctico") as error:
        parse(sql)
    assert fragment in str(error.value)
