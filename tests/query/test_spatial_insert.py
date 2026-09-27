import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from common.value import Point
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import QueryError, execute
from serialize import serialize_record


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


def flush_catalog(catalog):
    for storage in catalog._table_storage.values():
        storage.bm.flush_all()
        if hasattr(storage, "overflow"):
            storage.overflow.bm.flush_all()
    catalog._sys_tables.bm.flush_all()
    catalog._sys_columns.bm.flush_all()
    catalog._sys_indexes.bm.flush_all()


def test_insert_point_round_trips_and_serializes_after_reboot(tmp_path):
    catalog = make_catalog(tmp_path)
    execute(
        "CREATE TABLE tiendas (id INTEGER PRIMARY KEY, nombre VARCHAR(40), ubicacion POINT);",
        catalog,
    )
    execute(
        "INSERT INTO tiendas VALUES (1, 'Tienda Centro', POINT(-12.0464, -77.0428));",
        catalog,
    )
    flush_catalog(catalog)

    rows = execute("SELECT * FROM tiendas;", make_catalog(tmp_path))

    assert rows[0].values[2].data == Point(-12.0464, -77.0428)
    assert serialize_record(rows[0]) == [
        1,
        "Tienda Centro",
        {"longitude": -12.0464, "latitude": -77.0428},
    ]


def test_insert_null_into_nullable_point_column(tmp_path):
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE ubicaciones (punto POINT);", catalog)

    execute("INSERT INTO ubicaciones VALUES (NULL);", catalog)

    rows = execute("SELECT * FROM ubicaciones;", catalog)
    assert rows[0].values[0].data is None


@pytest.mark.parametrize(
    "point",
    ["POINT(-181, 0)", "POINT(0, 91)"],
)
def test_insert_rejects_out_of_range_point_coordinates(tmp_path, point):
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE ubicaciones (punto POINT);", catalog)

    with pytest.raises(QueryError):
        execute(f"INSERT INTO ubicaciones VALUES ({point});", catalog)