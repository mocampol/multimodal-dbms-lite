import pytest

from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.parser.ast_nodes import (
    DistanceExp,
    LimitClause,
    PointLiteral,
    PolygonLiteral,
    SpatialPredicate,
    WithinExp,
)


def parse(sql):
    return Parser(Scanner(sql)).parse_sql_statements()[0]


def test_parse_distance_comparison_with_point_literal():
    statement = parse(
        "SELECT * FROM tiendas "
        "WHERE distancia(ubicacion, POINT(-12.0464, -77.0428)) < 5000;"
    )

    condition = statement.where_cond
    assert isinstance(condition, SpatialPredicate)
    assert isinstance(condition.left, DistanceExp)
    assert condition.left.geometry.value == "ubicacion"
    assert isinstance(condition.left.point, PointLiteral)
    assert (condition.left.point.longitude, condition.left.point.latitude) == (
        -12.0464,
        -77.0428,
    )
    assert condition.right.value == 5000


def test_parse_order_by_distance_and_limit():
    statement = parse(
        "SELECT * FROM restaurantes "
        "ORDER BY distancia(ubicacion, POINT(-12.05, -77.04)) LIMIT 10;"
    )

    order_expression = statement.order_by.columns[0]
    assert isinstance(order_expression, DistanceExp)
    assert isinstance(order_expression.point, PointLiteral)
    assert isinstance(statement.limit, LimitClause)
    assert statement.limit.value == 10


def test_parse_within_polygon_predicate():
    statement = parse(
        "SELECT * FROM restaurantes WHERE dentro_de(ubicacion, POLYGON("
        "POINT(-78, -13), POINT(-77, -13), POINT(-77, -12), POINT(-78, -13)"
        "));"
    )

    predicate = statement.where_cond
    assert isinstance(predicate, WithinExp)
    assert predicate.geometry.value == "ubicacion"
    assert isinstance(predicate.polygon, PolygonLiteral)
    assert len(predicate.polygon.points) == 4
    assert (predicate.polygon.points[0].longitude, predicate.polygon.points[0].latitude) == (
        -78,
        -13,
    )


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        (
            "SELECT * FROM tiendas WHERE distancia(ubicacion, POINT(-12.0, )) < 5000;",
            "se esperaba",
        ),
        (
            "SELECT * FROM restaurantes ORDER BY ubicacion LIMIT 1.5;",
            "LIMIT requiere un entero",
        ),
    ],
)
def test_spatial_parser_reports_clear_syntax_errors(sql, message):
    with pytest.raises(RuntimeError, match=message):
        parse(sql)