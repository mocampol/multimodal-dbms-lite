"""
System catalog: metadata about tables, columns, and indexes.

Persisted using the engine's own Heap File, as sys_tables / sys_columns /
sys_indexes. The same approach Postgres uses with pg_catalog. All three
system tables are loaded fully into memory at engine startup, since they
are small and are consulted on every query.
"""

from common.value import DataType, Value
from common.schema import Schema, Column
from common.record import Record

from .table_metadata import TableMetadata, StorageType
from .column import ColumnMetadata
from .exceptions import (
    TableAlreadyExistsError,
    TableNotFoundError,
    ColumnNotFoundError,
    UniqueConstraintError,
)


# System table schemas

SYS_TABLES_SCHEMA = Schema("sys_tables", [
    Column("table_id", DataType.INTEGER, is_primary_key=True),
    Column("table_name", DataType.VARCHAR, size=64, is_unique=True),
    Column("storage_type", DataType.VARCHAR, size=16),
    Column("root_page_id", DataType.INTEGER),
])

SYS_COLUMNS_SCHEMA = Schema("sys_columns", [
    Column("table_id", DataType.INTEGER),
    Column("column_name", DataType.VARCHAR, size=64),
    Column("col_type", DataType.VARCHAR, size=32),
    Column("col_size", DataType.INTEGER, nullable=True),
    Column("position", DataType.INTEGER),
    Column("is_primary_key", DataType.BOOLEAN),
])

SYS_INDEXES_SCHEMA = Schema("sys_indexes", [
    Column("index_id", DataType.INTEGER, is_primary_key=True),
    Column("table_id", DataType.INTEGER),
    Column("column_name", DataType.VARCHAR, size=64),
    Column("index_type", DataType.VARCHAR, size=16),
    Column("root_page_id", DataType.INTEGER),
])


class Catalog:
    """
    Central metadata registry.

    Attributes:
        tables (dict[str, TableMetadata]): table_name -> metadata (schema included).
        indexes (dict[str, list[dict]]): table_name -> list of index metadata.
    """

    def __init__(self, heap_factory):
        """
        Args:
            heap_factory: callable(schema: Schema) -> heap-like object exposing:
                - insert(record: Record) -> RID
                - scan() -> Iterator[Record]
              This is how the Catalog stays decoupled from the concrete
              storage.heap.HeapFile implementation. Once heap.py is ready,
              pass e.g. `lambda schema: HeapFile(schema)` from main.py.
        """
        self._sys_tables = heap_factory(SYS_TABLES_SCHEMA)
        self._sys_columns = heap_factory(SYS_COLUMNS_SCHEMA)
        self._sys_indexes = heap_factory(SYS_INDEXES_SCHEMA)

        self.tables: dict[str, TableMetadata] = {}
        self.indexes: dict[str, list[dict]] = {}

        self._unique_values: dict[tuple, set] = {}

        self._next_table_id = 1
        self._next_index_id = 1

        self._load()


    def _load(self):
        """
        Loads sys_tables, sys_columns and sys_indexes fully into memory.
        Called once at engine startup.
        """
        columns_by_table_id: dict[int, list[ColumnMetadata]] = {}
        for record in self._sys_columns.scan():
            cm = ColumnMetadata.from_values(tuple(v.data for v in record))
            columns_by_table_id.setdefault(cm.table_id, []).append(cm)

        for record in self._sys_tables.scan():
            tm = TableMetadata.from_values(tuple(v.data for v in record))
            col_metas = sorted(
                columns_by_table_id.get(tm.table_id, []),
                key=lambda c: c.position,
            )
            tm.schema = Schema(tm.table_name, [cm.to_column() for cm in col_metas])
            self.tables[tm.table_name] = tm
            self._next_table_id = max(self._next_table_id, tm.table_id + 1)

            for col_name in tm.schema.unique_columns():
                self._unique_values[(tm.table_name, col_name)] = set()

        for record in self._sys_indexes.scan():
            values = tuple(v.data for v in record)
            index_id, table_id, column_name, index_type, root_page_id = values
            table_name = self._table_name_by_id(table_id)
            self.indexes.setdefault(table_name, []).append({
                "index_id": index_id,
                "column_name": column_name,
                "index_type": index_type,
                "root_page_id": root_page_id,
            })
            self._next_index_id = max(self._next_index_id, index_id + 1)

    def _table_name_by_id(self, table_id: int) -> str:
        for name, tm in self.tables.items():
            if tm.table_id == table_id:
                return name
        raise TableNotFoundError(f"No existe una tabla con table_id={table_id}")


    def create_table(self, schema: Schema, storage_type: StorageType = StorageType.HEAP) -> TableMetadata:
        """
        Registers a new table: persists it into sys_tables/sys_columns
        and makes it available in memory immediately.
        """
        if schema.table_name in self.tables:
            raise TableAlreadyExistsError(f"La tabla '{schema.table_name}' ya existe")

        table_id = self._next_table_id
        self._next_table_id += 1

        tm = TableMetadata(
            table_id=table_id,
            table_name=schema.table_name,
            storage_type=storage_type,
            root_page_id=-1,  # storage assigns this once the physical file is created
            schema=schema,
        )

        self._sys_tables.insert(Record([
            Value(DataType.INTEGER, tm.table_id),
            Value(DataType.VARCHAR, tm.table_name),
            Value(DataType.VARCHAR, tm.storage_type.value),
            Value(DataType.INTEGER, tm.root_page_id),
        ]))

        for position, column in enumerate(schema.columns):
            cm = ColumnMetadata.from_column(table_id, position, column)
            self._sys_columns.insert(Record([
                Value(DataType.INTEGER, cm.table_id),
                Value(DataType.VARCHAR, cm.column_name),
                Value(DataType.VARCHAR, cm.col_type),
                Value(DataType.INTEGER, cm.col_size) if cm.col_size is not None else Value(DataType.INTEGER, None),
                Value(DataType.INTEGER, cm.position),
                Value(DataType.BOOLEAN, cm.is_primary_key),
            ]))

        self.tables[schema.table_name] = tm
        for col_name in schema.unique_columns():
            self._unique_values[(schema.table_name, col_name)] = set()

        return tm

    def drop_table(self, table_name: str):
        """
        Removes a table from the catalog's in-memory view.
        NOTE: does not delete sys_tables/sys_columns rows physically yet.
        Heap File delete is lazy (per spec), so this should call the
        equivalent delete on the sys_* heaps once heap.delete(rid) exists.
        """
        if table_name not in self.tables:
            raise TableNotFoundError(f"La tabla '{table_name}' no existe")

        del self.tables[table_name]
        self.indexes.pop(table_name, None)
        for key in [k for k in self._unique_values if k[0] == table_name]:
            del self._unique_values[key]

    def create_index(self, table_name: str, column_name: str, index_type: str) -> dict:
        """
        Registers an index over a column. Does not build the physical
        index structure, that's the index subsystem's job. This only
        records the metadata so the planner knows the index exists.
        """
        tm = self.get_table(table_name)
        if not tm.schema.get_column(column_name):
            raise ColumnNotFoundError(f"'{column_name}' no existe en '{table_name}'")

        index_id = self._next_index_id
        self._next_index_id += 1

        self._sys_indexes.insert(Record([
            Value(DataType.INTEGER, index_id),
            Value(DataType.INTEGER, tm.table_id),
            Value(DataType.VARCHAR, column_name),
            Value(DataType.VARCHAR, index_type),
            Value(DataType.INTEGER, -1),  # root_page_id, set once the index is built
        ]))

        entry = {
            "index_id": index_id,
            "column_name": column_name,
            "index_type": index_type,
            "root_page_id": -1,
        }
        self.indexes.setdefault(table_name, []).append(entry)
        return entry

    # queries used by the semantic analyzer

    def table_exists(self, table_name: str) -> bool:
        return table_name in self.tables

    def get_table(self, table_name: str) -> TableMetadata:
        if table_name not in self.tables:
            raise TableNotFoundError(f"La tabla '{table_name}' no existe")
        return self.tables[table_name]

    def get_schema(self, table_name: str) -> Schema:
        return self.get_table(table_name).schema

    def column_exists(self, table_name: str, column_name: str) -> bool:
        return self.get_schema(table_name).get_column(column_name) is not None

    def get_column_type(self, table_name: str, column_name: str) -> DataType:
        column = self.get_schema(table_name).get_column(column_name)
        if column is None:
            raise ColumnNotFoundError(f"'{column_name}' no existe en '{table_name}'")
        return column.data_type

    def get_indexes(self, table_name: str) -> list[dict]:
        return self.indexes.get(table_name, [])

    # UNIQUE / PK enforcement

    def check_unique(self, table_name: str, column_name: str, value) -> bool:
        """
        Returns True if `value` does not violate a UNIQUE/PK constraint
        on this column. Called before inserting into storage.
        """
        key = (table_name, column_name)
        if key not in self._unique_values:
            return True
        return value not in self._unique_values[key]

    def register_unique(self, table_name: str, column_name: str, value):
        """
        Marks `value` as used for a UNIQUE/PK column. Call this AFTER
        a successful insert into storage, so the in-memory set stays
        in sync with what's actually on disk.
        """
        key = (table_name, column_name)
        if key in self._unique_values:
            self._unique_values[key].add(value)

    def check_insert_uniques(self, table_name: str, record: Record) -> str | None:
        """
        Convenience helper: checks ALL unique columns of a record at once.
        Returns the name of the first violated column, or None if OK.
        """
        schema = self.get_schema(table_name)
        for col_name in schema.unique_columns():
            idx = schema.column_index(col_name)
            if not self.check_unique(table_name, col_name, record[idx].data):
                return col_name
        return None

    def register_insert_uniques(self, table_name: str, record: Record):
        """
        Registers all unique-column values of a record after a successful insert.
        """
        schema = self.get_schema(table_name)
        for col_name in schema.unique_columns():
            idx = schema.column_index(col_name)
            self.register_unique(table_name, col_name, record[idx].data)
