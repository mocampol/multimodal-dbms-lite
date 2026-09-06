"""
Table metadata as stored in the system catalog (sys_tables).
"""

from enum import Enum
from common.schema import Schema


class StorageType(Enum):
    HEAP = "heap"
    SEQUENTIAL = "sequential"


class TableMetadata:
    """
    In-memory representation of one row of sys_tables, enriched with
    the full Schema (built by joining with sys_columns at load time).

    Attributes:
        table_id (int): Unique id, primary key of sys_tables.
        table_name (str): Name of the table.
        storage_type (StorageType): HEAP or SEQUENTIAL.
        root_page_id (int): First page of the table's physical file.
        schema (Schema | None): Set by Catalog after loading columns.
    """

    def __init__(
        self,
        table_id: int,
        table_name: str,
        storage_type: StorageType,
        root_page_id: int,
        schema: Schema = None,
    ):
        self.table_id = table_id
        self.table_name = table_name
        self.storage_type = storage_type
        self.root_page_id = root_page_id
        self.schema = schema

    def to_values(self) -> tuple:
        """
        Returns the raw field values in sys_tables column order,
        ready to be wrapped into a Record by the Catalog.
        """
        return (self.table_id, self.table_name, self.storage_type.value, self.root_page_id)

    @classmethod
    def from_values(cls, values: tuple) -> "TableMetadata":
        """
        Rebuilds a TableMetadata from a raw sys_tables row (schema not set yet;
        Catalog fills it in after reading sys_columns).
        """
        table_id, table_name, storage_type, root_page_id = values
        return cls(table_id, table_name, StorageType(storage_type), root_page_id)

    def __repr__(self):
        return f"TableMetadata(id={self.table_id}, name={self.table_name}, type={self.storage_type.value})"
