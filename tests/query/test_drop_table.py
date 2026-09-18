from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.parser.parser import Parser
from query.parser.scanner import Scanner
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


def flush_catalog(catalog):
    for storage in catalog._table_storage.values():
        storage.bm.flush_all()
        if hasattr(storage, "overflow"):
            storage.overflow.bm.flush_all()
    catalog._sys_tables.bm.flush_all()
    catalog._sys_columns.bm.flush_all()
    catalog._sys_indexes.bm.flush_all()


def test_parser_accepts_drop_table():
    statements = Parser(Scanner("DROP TABLE alumnos;")).parse_sql_statements()

    assert len(statements) == 1
    assert statements[0].table == "alumnos"


def test_drop_table_can_recreate_after_reboot(tmp_path):
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE alumnos (id INTEGER PRIMARY KEY, nombre VARCHAR(20));", catalog)
    execute("INSERT INTO alumnos VALUES (1, 'Daniela');", catalog)
    execute("DROP TABLE alumnos;", catalog)

    assert "alumnos" not in catalog.tables
    execute("CREATE TABLE alumnos (id INTEGER PRIMARY KEY, nombre VARCHAR(20));", catalog)
    execute("INSERT INTO alumnos VALUES (2, 'Valentin');", catalog)
    flush_catalog(catalog)

    rebooted = make_catalog(tmp_path)
    rows = execute("SELECT * FROM alumnos;", rebooted)

    assert [[value.data for value in row.values] for row in rows] == [[2, "Valentin"]]


def test_sequential_clustered_index_rebuilds_after_reboot(tmp_path):
    catalog = make_catalog(tmp_path)
    execute(
        "CREATE TABLE matriculas (id INTEGER PRIMARY KEY, curso VARCHAR(20)) "
        "USING SEQUENTIAL;",
        catalog,
    )
    execute("INSERT INTO matriculas VALUES (1, 'BD2');", catalog)
    flush_catalog(catalog)

    rebooted = make_catalog(tmp_path)
    execute("INSERT INTO matriculas VALUES (2, 'Cloud');", rebooted)
    flush_catalog(rebooted)

    rows = execute("SELECT * FROM matriculas WHERE id = 2;", rebooted)

    assert [[value.data for value in row.values] for row in rows] == [[2, "Cloud"]]


def test_create_index_replaces_orphaned_index_file(tmp_path):
    catalog = make_catalog(tmp_path)
    execute(
        "CREATE TABLE alumnos (id INTEGER PRIMARY KEY, nombre VARCHAR(20));",
        catalog,
    )

    orphan_manager = make_index_buffer_factory(str(tmp_path))(
        "alumnos", "nombre", 1
    )
    orphan_manager.allocate_page()

    execute("CREATE INDEX idx_nombre ON alumnos (nombre) USING BTREE;", catalog)

    assert len(catalog.get_indexes("alumnos")) == 1
