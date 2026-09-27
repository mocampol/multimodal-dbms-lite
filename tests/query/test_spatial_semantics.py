import pytest

from common import Column, DataType, Schema
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.parser.visitor import InMemoryCatalog, SemanticVisitor
from query.query_engine import QueryError, execute


def make_catalog(point_type=DataType.POINT, with_rtree=True):
    catalog = InMemoryCatalog()
    catalog.define_table(
        Schema(
            "tiendas",
            [
                Column("id", DataType.INTEGER),
                Column("ubicacion", point_type),
                Column("punto_referencia", DataType.POINT),
                Column("nombre", DataType.TEXT),
            ],
        )
    )
    if with_rtree:
        catalog.create_index("tiendas", "ubicacion", "rtree")
    return catalog


def parse(sql):
    return Parser(Scanner(sql)).parse_sql_statements()[0]


def assert_semantically_valid(sql, catalog):
    SemanticVisitor(catalog).check(parse(sql))


def assert_query_error(sql, catalog, message):
    with pytest.raises(QueryError) as error:
        execute(sql, catalog)
    assert message in str(error.value)


def test_semantics_accept_distance_with_supported_metric():
    catalog = make_catalog()

    assert_semantically_valid(
        "SELECT * FROM tiendas WHERE distancia(ubicacion, "
        "POINT(-12.0464, -77.0428), HAVERSINE) < 5000;",
        catalog,
    )


def test_semantics_accept_point_column_as_distance_argument():
    assert_semantically_valid(
        "SELECT * FROM tiendas WHERE distancia(ubicacion, punto_referencia) < 5000;",
        make_catalog(),
    )


def test_semantics_accept_distance_order_without_spatial_filter_index():
    catalog = make_catalog(with_rtree=False)

    assert_semantically_valid(
        "SELECT * FROM tiendas ORDER BY distancia(ubicacion, "
        "POINT(-12.05, -77.04), EUCLIDEAN) LIMIT 10;",
        catalog,
    )


def test_semantics_accept_within_polygon_with_rtree():
    catalog = make_catalog()

    assert_semantically_valid(
        "SELECT * FROM tiendas WHERE dentro_de(ubicacion, POLYGON("
        "POINT(-78, -13), POINT(-77, -13), POINT(-77, -12)));",
        catalog,
    )


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        (
            "SELECT * FROM tiendas WHERE distancia(nombre, POINT(0, 0)) < 10;",
            "de tipo POINT",
        ),
        (
            "SELECT * FROM tiendas WHERE distancia(ubicacion, 5) < 10;",
            "segundo argumento de distancia debe ser POINT",
        ),
        (
            "SELECT * FROM tiendas WHERE distancia(ubicacion, POINT(0, 0)) < 'lejos';",
            "distancia debe compararse con un valor numérico",
        ),
        (
            "SELECT * FROM tiendas WHERE dentro_de(ubicacion, POLYGON("
            "POINT(0, 0), POINT(1, 1)));",
            "al menos tres puntos",
        ),
        (
            "SELECT * FROM tiendas WHERE dentro_de(nombre, POLYGON("
            "POINT(0, 0), POINT(1, 0), POINT(0, 1)));",
            "usada en dentro_de debe ser de tipo POINT",
        ),
        (
            "SELECT * FROM tiendas WHERE distancia(ubicacion, POINT(-181, 0)) < 10;",
            "longitude",
        ),
        (
            "SELECT * FROM tiendas WHERE dentro_de(ubicacion, POLYGON("
            "POINT(181, 0), POINT(1, 0), POINT(0, 1)));",
            "longitude",
        ),
        (
            "SELECT * FROM tiendas WHERE distancia(ubicacion, "
            "POINT(0, 0), MANHATTAN) < 10;",
            "métrica debe ser EUCLIDEAN o HAVERSINE",
        ),
    ],
)
def test_spatial_semantic_errors_become_query_error(sql, message):
    assert_query_error(sql, make_catalog(), message)


def test_spatial_filter_requires_rtree_but_ordering_does_not():
    sql = "SELECT * FROM tiendas WHERE distancia(ubicacion, POINT(0, 0)) < 10;"

    assert_query_error(sql, make_catalog(with_rtree=False), "índice RTREE")


def test_limit_must_be_positive():
    assert_query_error(
        "SELECT * FROM tiendas LIMIT 0;",
        make_catalog(),
        "LIMIT debe ser un entero positivo",
    )


def test_join_columns_must_be_compatible_for_spatial_lookup():
    catalog = make_catalog()
    catalog.define_table(
        Schema(
            "restaurantes",
            [Column("id", DataType.INTEGER), Column("ubicacion", DataType.POINT)],
        )
    )
    catalog.create_index("restaurantes", "ubicacion", "rtree")

    assert_query_error(
        "SELECT * FROM tiendas JOIN restaurantes "
        "ON tiendas.id = restaurantes.ubicacion "
        "WHERE distancia(restaurantes.ubicacion, POINT(0, 0)) < 10;",
        catalog,
        "mismo tipo",
    )

    assert_semantically_valid(
        "SELECT * FROM tiendas JOIN restaurantes "
        "ON tiendas.id = restaurantes.id "
        "WHERE distancia(restaurantes.ubicacion, POINT(0, 0)) < 10;",
        catalog,
    )