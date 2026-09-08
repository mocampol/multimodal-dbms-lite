"""
Type system.

Classes:
    DataType: Enumeration of supported SQL data types.
    Value: A SQL value paired with its corresponding DataType.
"""

from enum import Enum
from decimal import Decimal
from datetime import date, datetime, time


class DataType(Enum):
    """
    SQL data types supported by the database system, modeled after
    PostgreSQL's type system.

    Data types are grouped into six categories:
        1) Integer types
        2) Decimal types
        3) String types
        4) Boolean type
        5) Date and Time types
        6) Binary type
    """

    # Integer types
    SMALLINT = "smallint"
    INTEGER = "integer"
    BIGINT = "bigint"

    # Decimal types
    NUMERIC = "numeric"
    REAL = "real"
    DOUBLE_PRECISION = "double_precision"

    # String types
    CHAR = "char"
    VARCHAR = "varchar"
    TEXT = "text"

    # Boolean type
    BOOLEAN = "boolean"

    # Date and Time types
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"

    # Binary type
    BYTEA = "bytea"


FIXED_SIZE = {
    DataType.SMALLINT: 2,
    DataType.INTEGER: 4,
    DataType.BIGINT: 8,
    DataType.REAL: 4,
    DataType.DOUBLE_PRECISION: 8,
    DataType.BOOLEAN: 1,
    DataType.DATE: 4,
    DataType.TIME: 8,
    DataType.TIMESTAMP: 8,
}

VARIABLE_SIZE_TYPES = {
    DataType.CHAR,
    DataType.VARCHAR,
}

UNBOUNDED_TYPES = {
    DataType.TEXT,
    DataType.BYTEA,
    DataType.NUMERIC,
}


class Value:
    """
    Value represents a typed value within the database engine.

    A Value contains:
        data_type: the SQL type
        data: the actual Python value
    """

    def __init__(self, data_type: DataType, data):
        self.data_type = data_type
        self.data = data

    def validate(self) -> bool:
        """
        Checks that self.data is a Python type consistent with self.data_type.
        NULL (self.data is None) is always structurally valid here.
        Whether NULL is actually allowed depends on the Column (nullable).
        """
        if self.data is None:
            return True

        if self.data_type in {DataType.CHAR, DataType.VARCHAR, DataType.TEXT}:
            return isinstance(self.data, str)

        if self.data_type == DataType.BYTEA:
            return isinstance(self.data, (bytes, bytearray))

        if self.data_type in {DataType.SMALLINT, DataType.INTEGER, DataType.BIGINT}:
            return isinstance(self.data, int) and not isinstance(self.data, bool)

        if self.data_type in {DataType.REAL, DataType.DOUBLE_PRECISION}:
            return isinstance(self.data, (int, float)) and not isinstance(self.data, bool)

        if self.data_type == DataType.NUMERIC:
            return isinstance(self.data, (Decimal, int, float)) and not isinstance(self.data, bool)

        if self.data_type == DataType.BOOLEAN:
            return isinstance(self.data, bool)

        if self.data_type == DataType.DATE:
            return isinstance(self.data, date) and not isinstance(self.data, datetime)

        if self.data_type == DataType.TIMESTAMP:
            return isinstance(self.data, datetime)

        if self.data_type == DataType.TIME:
            return isinstance(self.data, time)

        return False

    def byte_size(self) -> int:
        """
        Returns the number of bytes this value occupies in a serialized record.
        Used by storage when writing/reading pages.
        """
        if self.data is None:
            return 0

        if not self.validate():
            raise ValueError(f"Valor inválido para el tipo {self.data_type}: {self.data!r}")

        if self.data_type in FIXED_SIZE:
            return FIXED_SIZE[self.data_type]

        if self.data_type in {DataType.CHAR, DataType.VARCHAR, DataType.TEXT}:
            return len(self.data.encode("utf-8"))

        if self.data_type == DataType.BYTEA:
            return len(self.data)

        if self.data_type == DataType.NUMERIC:
            return len(str(self.data).encode("utf-8"))

        raise ValueError(f"No se conoce el tamaño físico de {self.data_type}")

    def __repr__(self):
        return f"Value({self.data_type.value}, {self.data!r})"
