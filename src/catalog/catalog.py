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
from index.btree.btree import BTree
from index.extendible_hash.extendible_hash_index import ExtendibleHashIndex
from index.btree.clustered_index import ClusteredIndex

from .table_metadata import TableMetadata, StorageType
from .column import ColumnMetadata
from .exceptions import (
    TableAlreadyExistsError,
    TableNotFoundError,
    ColumnNotFoundError,
    UniqueConstraintError,
)


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

    def __init__(self, heap_factory, storage_factories=None, index_buffer_factory=None):
        """
        Args:
            heap_factory: callable(schema: Schema) -> heap-like object exposing:
                - insert(record: Record) -> RID
                - scan() -> Iterator[Record]
            This is how the Catalog stays decoupled from the concrete
            storage.heap.HeapFile implementation. Once heap.py is ready,
            pass from main.py.
        """
        self._sys_tables = heap_factory(SYS_TABLES_SCHEMA)
        self._sys_columns = heap_factory(SYS_COLUMNS_SCHEMA)
        self._sys_indexes = heap_factory(SYS_INDEXES_SCHEMA)

        self._storage_factories = storage_factories or {StorageType.HEAP: heap_factory}
        self._index_buffer_factory = index_buffer_factory

        self.tables: dict[str, TableMetadata] = {}
        self.indexes: dict[str, list[dict]] = {}
        self._table_storage: dict[str, object] = {}
        self._physical_indexes: dict[int, object] = {}
        self._index_catalog_rids: dict[int, object] = {}
        self._clustered_indexes: dict[str, ClusteredIndex] = {}

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

            factory = self._storage_factories[tm.storage_type]
            self._table_storage[tm.table_name] = factory(tm.schema)
            if tm.storage_type == StorageType.SEQUENTIAL and self._index_buffer_factory is not None:
                primary_key = tm.schema.primary_key()
                if primary_key is not None:
                    manager = self._index_buffer_factory(tm.table_name, "clustered", tm.table_id)
                    self._clustered_indexes[tm.table_name] = ClusteredIndex(
                        primary_key.data_type, manager, self._table_storage[tm.table_name]
                    )

            for col_name in tm.schema.unique_columns():
                values = {
                    record[tm.schema.column_index(col_name)].data
                    for record in self._table_storage[tm.table_name].scan()
                }
                self._unique_values[(tm.table_name, col_name)] = values

        for sys_rid, record in self._sys_indexes.scan_with_rid():
            values = tuple(v.data for v in record)
            index_id, table_id, column_name, index_type, root_page_id = values
            table_name = self._table_name_by_id(table_id)
            self.indexes.setdefault(table_name, []).append({
                "index_id": index_id,
                "column_name": column_name,
                "index_type": index_type,
                "root_page_id": root_page_id,
            })
            self._index_catalog_rids[index_id] = sys_rid
            if self._index_buffer_factory is not None and root_page_id >= 0:
                self._physical_indexes[index_id] = self._open_index(
                    table_name, column_name, index_type, root_page_id, index_id
                )
            self._next_index_id = max(self._next_index_id, index_id + 1)

    def _table_name_by_id(self, table_id: int) -> str:
        for name, tm in self.tables.items():
            if tm.table_id == table_id:
                return name
        raise TableNotFoundError(f"No existe una tabla con table_id={table_id}")


    def create_table(self, schema: Schema, storage_type: StorageType = StorageType.HEAP) -> TableMetadata:
        """
        Registers a new table: persists its metadata into sys_tables/
        sys_columns AND creates its physical data storage, then corrects
        the persisted root_page_id once that storage exists.
        """
        if schema.table_name in self.tables:
            raise TableAlreadyExistsError(f"La tabla '{schema.table_name}' ya existe")
        if storage_type not in self._storage_factories:
            raise ValueError(f"No hay una fábrica de storage registrada para {storage_type}")

        table_id = self._next_table_id
        self._next_table_id += 1

        tm = TableMetadata(
            table_id=table_id,
            table_name=schema.table_name,
            storage_type=storage_type,
            root_page_id=-1,
            schema=schema,
        )

        table_rid = self._sys_tables.insert(Record([
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

        storage = self._storage_factories[storage_type](schema)
        self._table_storage[schema.table_name] = storage

        if storage_type == StorageType.SEQUENTIAL and self._index_buffer_factory is not None:
            primary_key = schema.primary_key()
            if primary_key is None:
                raise ValueError("SequentialFile requiere una clave primaria para el índice agrupado")
            manager = self._index_buffer_factory(schema.table_name, "clustered", table_id)
            self._clustered_indexes[schema.table_name] = ClusteredIndex(
                primary_key.data_type, manager, storage
            )

        if hasattr(storage, "root_page_id"):
            tm.root_page_id = storage.root_page_id
            self._sys_tables.update(table_rid, Record([
                Value(DataType.INTEGER, tm.table_id),
                Value(DataType.VARCHAR, tm.table_name),
                Value(DataType.VARCHAR, tm.storage_type.value),
                Value(DataType.INTEGER, tm.root_page_id),
            ]))

        self.tables[schema.table_name] = tm
        for col_name in schema.unique_columns():
            self._unique_values[(schema.table_name, col_name)] = set()

        return tm

    def drop_table(self, table_name: str):
        if table_name not in self.tables:
            raise TableNotFoundError(f"La tabla '{table_name}' no existe")

        del self.tables[table_name]
        self._table_storage.pop(table_name, None)
        self.indexes.pop(table_name, None)
        for key in [k for k in self._unique_values if k[0] == table_name]:
            del self._unique_values[key]

    def get_storage(self, table_name: str):
        """
        Returns the physical storage object (HeapFile, SequentialFile, ...)
        backing table_name's data. This is what the Executor's access
        nodes (Sequential Scan, Index Scan) call insert()/scan()/get() on.
        They never construct or manage storage objects themselves.
        """
        if table_name not in self._table_storage:
            raise TableNotFoundError(f"La tabla '{table_name}' no existe")
        return self._table_storage[table_name]

    def get_clustered_index(self, table_name: str):
        return self._clustered_indexes.get(table_name)

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

        column = tm.schema.get_column(column_name)
        if index_type not in {"btree", "hash"}:
            raise ValueError(f"Tipo de índice no soportado: {index_type}")
        if self._index_buffer_factory is None:
            raise ValueError("No hay fábrica de BufferManager para índices")

        index = self._create_index(index_type, table_name, column_name, index_id, column.data_type)
        for rid, record in self._scan_with_rids(self._table_storage[table_name]):
            index.insert(record[tm.schema.column_index(column_name)], rid)

        root_page_id = (
            index.root_page_id if index_type == "btree" else index.directory_page_id
        )
        index_rid = self._sys_indexes.insert(Record([
            Value(DataType.INTEGER, index_id),
            Value(DataType.INTEGER, tm.table_id),
            Value(DataType.VARCHAR, column_name),
            Value(DataType.VARCHAR, index_type),
            Value(DataType.INTEGER, root_page_id),
        ]))

        entry = {
            "index_id": index_id,
            "column_name": column_name,
            "index_type": index_type,
            "root_page_id": root_page_id,
        }
        self.indexes.setdefault(table_name, []).append(entry)
        self._physical_indexes[index_id] = index
        self._index_catalog_rids[index_id] = index_rid
        return entry

    def _create_index(self, index_type, table_name, column_name, index_id, key_type):
        manager = self._index_buffer_factory(table_name, column_name, index_id)
        if index_type == "btree":
            return BTree(key_type, manager)
        return ExtendibleHashIndex.create(manager, key_type)

    def _open_index(self, table_name, column_name, index_type, root_page_id, index_id):
        manager = self._index_buffer_factory(table_name, column_name, index_id)
        key_type = self.get_schema(table_name).get_column(column_name).data_type
        if index_type == "btree":
            return BTree(key_type, manager, root_page_id=root_page_id)
        return ExtendibleHashIndex(manager, root_page_id, key_type)

    @staticmethod
    def _scan_with_rids(storage):
        if not hasattr(storage, "scan_with_rid"):
            raise ValueError("Los índices secundarios requieren HeapFile con RIDs")
        return storage.scan_with_rid()

    def get_physical_index(self, table_name: str, column_name: str):
        for entry in self.indexes.get(table_name, []):
            if entry["column_name"] == column_name:
                return self._physical_indexes.get(entry["index_id"])
        return None

    def register_insert(self, table_name: str, record: Record, rid):
        clustered = self._clustered_indexes.get(table_name)
        if clustered is not None:
            clustered.sync()
        for entry in self.indexes.get(table_name, []):
            index = self._physical_indexes.get(entry["index_id"])
            if index is not None:
                column_index = self.get_schema(table_name).column_index(entry["column_name"])
                index.insert(record[column_index], rid)
                self._persist_index_root(entry, index)

    def unregister_delete(self, table_name: str, record: Record, rid):
        clustered = self._clustered_indexes.get(table_name)
        if clustered is not None:
            clustered.sync()
        for entry in self.indexes.get(table_name, []):
            index = self._physical_indexes.get(entry["index_id"])
            if index is not None:
                column_index = self.get_schema(table_name).column_index(entry["column_name"])
                key = record[column_index]
                if hasattr(index, "remove"):
                    index.remove(key, rid)
                else:
                    index.delete(key, rid)
                self._persist_index_root(entry, index)
        schema = self.get_schema(table_name)
        for column_name in schema.unique_columns():
            index = schema.column_index(column_name)
            self._unique_values[(table_name, column_name)].discard(record[index].data)

    def register_update(self, table_name: str, rid, old_record: Record, new_record: Record):
        schema = self.get_schema(table_name)
        for entry in self.indexes.get(table_name, []):
            index = self._physical_indexes.get(entry["index_id"])
            if index is None:
                continue
            column_index = schema.column_index(entry["column_name"])
            old_key = old_record[column_index]
            new_key = new_record[column_index]
            if old_key.data != new_key.data:
                if hasattr(index, "remove"):
                    index.remove(old_key, rid)
                else:
                    index.delete(old_key, rid)
                index.insert(new_key, rid)
                self._persist_index_root(entry, index)
        for column_name in schema.unique_columns():
            column_index = schema.column_index(column_name)
            old_value = old_record[column_index].data
            new_value = new_record[column_index].data
            if old_value != new_value:
                self._unique_values[(table_name, column_name)].discard(old_value)
                self._unique_values[(table_name, column_name)].add(new_value)

    def _persist_index_root(self, entry, index):
        if entry["index_type"] != "btree":
            return
        root_page_id = index.root_page_id
        if root_page_id == entry["root_page_id"]:
            return
        entry["root_page_id"] = root_page_id
        catalog_rid = self._index_catalog_rids.get(entry["index_id"])
        if catalog_rid is None:
            return
        self._sys_indexes.update(catalog_rid, Record([
            Value(DataType.INTEGER, entry["index_id"]),
            Value(DataType.INTEGER, self.get_table_by_index_entry(entry)[0]),
            Value(DataType.VARCHAR, entry["column_name"]),
            Value(DataType.VARCHAR, entry["index_type"]),
            Value(DataType.INTEGER, root_page_id),
        ]))

    def get_table_by_index_entry(self, entry):
        for table_name, entries in self.indexes.items():
            if entry in entries:
                return self.tables[table_name].table_id, table_name
        raise TableNotFoundError(f"No se encontró el índice {entry['index_id']}")

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
        Returns True if value does not violate a UNIQUE/PK constraint
        on this column. Called before inserting into storage.
        """
        key = (table_name, column_name)
        if key not in self._unique_values:
            return True
        return value not in self._unique_values[key]

    def register_unique(self, table_name: str, column_name: str, value):
        """
        Marks value as used for a UNIQUE/PK column. Call this AFTER
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
