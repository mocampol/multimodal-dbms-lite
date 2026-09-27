import os

from catalog.catalog import Catalog
from common import Column, DataType, Point, Record, Schema, Value
from spatial.geometry import Point2D
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.heap_file import HeapFile
from query.parser.ast_nodes import IndexType as AstIndexType
from query.parser.parser import Parser
from query.parser.scanner import Scanner


def _boot_engine(base_dir):
    buffer_managers = {}

    def heap_factory(schema):
        manager = BufferManager(
            FileManager(os.path.join(base_dir, f"{schema.table_name}.tbl")),
            pool_size=16,
        )
        buffer_managers[f"heap:{schema.table_name}"] = manager
        return HeapFile(schema, manager)

    def index_buffer_factory(table_name, column_name, index_id):
        manager = BufferManager(
            FileManager(
                os.path.join(base_dir, f"{table_name}.{column_name}.{index_id}.idx")
            ),
            pool_size=16,
        )
        buffer_managers[f"index:{index_id}"] = manager
        return manager

    return Catalog(heap_factory, index_buffer_factory=index_buffer_factory), buffer_managers


def _flush(buffer_managers):
    for manager in buffer_managers.values():
        manager.flush_all()


def _insert_place(catalog, identifier, point):
    record = Record([
        Value(DataType.INTEGER, identifier),
        Value(DataType.POINT, point),
    ])
    rid = catalog.get_storage("places").insert(record)
    catalog.register_insert("places", record, rid)
    return rid


def test_parser_accepts_point_columns_and_rtree_indexes():
    create_table, create_index = Parser(Scanner(
        "CREATE TABLE places (location POINT); "
        "CREATE INDEX location_idx ON places (location) USING RTREE;"
    )).parse_sql_statements()

    assert create_table.columns[0].data_type == DataType.POINT
    assert create_index.index_type == AstIndexType.RTREE


def test_rtree_index_survives_catalog_reboot_and_drop(tmp_path):
    base_dir = str(tmp_path)
    catalog, managers = _boot_engine(base_dir)
    catalog.create_table(Schema("places", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("location", DataType.POINT),
    ]))

    for identifier in range(200):
        _insert_place(
            catalog,
            identifier,
            Point(identifier * 0.5 - 50, identifier * 0.25 - 25),
        )

    index_entry = catalog.create_index("places", "location", "rtree")
    index_file = os.path.join(base_dir, "places.location.1.idx")
    assert index_entry["index_type"] == "rtree"
    assert index_entry["root_page_id"] != 0
    assert catalog.get_indexes("places") == [index_entry]
    _flush(managers)
    del catalog, managers

    catalog, managers = _boot_engine(base_dir)
    assert catalog.get_indexes("places")[0]["root_page_id"] == index_entry["root_page_id"]
    rtree = catalog.get_physical_index("places", "location")
    assert rtree.search_point(Point2D(25.0, 12.5))

    new_point = Point(12.25, -3.5)
    new_rid = _insert_place(catalog, 201, new_point)
    assert new_rid in rtree.search_point(Point2D(12.25, -3.5))
    _flush(managers)

    catalog.drop_table("places")
    assert not os.path.exists(index_file)
    assert not catalog.table_exists("places")
    assert catalog.get_indexes("places") == []