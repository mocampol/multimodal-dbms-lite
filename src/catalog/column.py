"""
Column metadata as stored in the system catalog (sys_columns).

This is intentionally separate from common.schema.Column:
    - common.schema.Column is the RUNTIME definition used to validate Values.
    - ColumnMetadata is the PERSISTENCE-shaped row that maps directly
      to sys_columns(table_id, column_name, col_type, col_size, position, is_primary_key).
"""

from common.value import DataType
from common.schema import Column


class ColumnMetadata:
    def __init__(
        self,
        table_id: int,
        column_name: str,
        col_type: str,
        col_size: int,
        position: int,
        is_primary_key: bool,
    ):
        self.table_id = table_id
        self.column_name = column_name
        self.col_type = col_type
        self.col_size = col_size
        self.position = position
        self.is_primary_key = is_primary_key

  
    def to_values(self) -> tuple:
        """
        Returns raw field values in sys_columns column order.
        """
        return (
            self.table_id,
            self.column_name,
            self.col_type,
            self.col_size,
            self.position,
            self.is_primary_key,
        )

  
    @classmethod
    def from_values(cls, values: tuple) -> "ColumnMetadata":
        table_id, column_name, col_type, col_size, position, is_primary_key = values
        return cls(table_id, column_name, col_type, col_size, position, is_primary_key)

    def to_column(self) -> Column:
        """
        Builds the runtime common.schema.Column used for validation.
        NOTE: is_unique is not part of sys_columns per the spec — only
        is_primary_key is tracked there. A PK is unique by definition
        (enforced inside Column itself).
        """
        return Column(
            name=self.column_name,
            data_type=DataType(self.col_type),
            size=self.col_size,
            is_primary_key=self.is_primary_key,
        )

    @classmethod
    def from_column(cls, table_id: int, position: int, column: Column) -> "ColumnMetadata":
        """
        Builds a persistence-shaped ColumnMetadata from a runtime Column,
        to be written into sys_columns when a table is created.
        """
        return cls(
            table_id=table_id,
            column_name=column.name,
            col_type=column.data_type.value,
            col_size=column.size,
            position=position,
            is_primary_key=column.is_primary_key,
        )

    def __repr__(self):
        return f"ColumnMetadata({self.column_name} {self.col_type}, pos={self.position})"
