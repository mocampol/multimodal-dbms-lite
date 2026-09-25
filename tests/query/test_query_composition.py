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


def seed_catalog(tmp_path):
    catalog = make_catalog(tmp_path)
    execute(
        "CREATE TABLE alumnos (id INTEGER PRIMARY KEY, nombre VARCHAR(20));",
        catalog,
    )
    execute(
        "CREATE TABLE matriculas (alumno_id INTEGER, curso VARCHAR(20), nota INTEGER);",
        catalog,
    )
    execute(
        "INSERT INTO alumnos VALUES "
        "(1, 'Ana'), (2, 'Luis'), (3, 'Marta');",
        catalog,
    )
    execute(
        "INSERT INTO matriculas VALUES "
        "(1, 'BD', 95), (1, 'SO', 70), (2, 'BD', 80), (3, 'BD', 60);",
        catalog,
    )
    return catalog


def values(rows):
    return [[value.data for value in row.values] for row in rows]


def test_join_where_and_order_by_compose(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT alumnos.nombre, matriculas.nota "
        "FROM alumnos JOIN matriculas ON alumnos.id = matriculas.alumno_id "
        "WHERE matriculas.nota >= 70 ORDER BY matriculas.nota;",
        catalog,
    )

    assert values(rows) == [["Ana", 70], ["Luis", 80], ["Ana", 95]]


def test_join_group_by_aggregate_where_and_order_by_compose(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT alumnos.id, COUNT(*), AVG(matriculas.nota) "
        "FROM alumnos JOIN matriculas ON alumnos.id = matriculas.alumno_id "
        "WHERE matriculas.nota >= 70 GROUP BY alumnos.id ORDER BY alumnos.id;",
        catalog,
    )

    assert values(rows) == [[1, 2, 82.5], [2, 1, 80.0]]


def test_group_by_and_order_by_compose_without_join(tmp_path):
    catalog = seed_catalog(tmp_path)

    rows = execute(
        "SELECT curso, COUNT(*) FROM matriculas "
        "GROUP BY curso ORDER BY curso;",
        catalog,
    )

    assert values(rows) == [["BD", 3], ["SO", 1]]