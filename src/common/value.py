"""
SQL type system.

Classes:
    DataType: Enumeration of supported SQL data types.
    Value: A SQL value paired with its corresponding DataType.
"""


from enum import Enum
from decimal import Decimal
from datetime import date, datetime, time


class DataType(Enum):
    """
    SQL data types supported by the database system.

    Data types are grouped into three categories:
        1) String types
        2) Numeric types
        3) Date and time types
    """

    # String types
    CHAR = "char"
    VARCHAR = "varchar"
    BINARY = "binary"
    VARBINARY = "varbinary"
    TEXT = "text"
    TINYTEXT = "tinytext"
    MEDIUMTEXT = "mediumtext"
    LONGTEXT = "longtext"
    SET = "set"

    # Numeric types
    BIT = "bit"
    TINYINT = "tinyint"
    BOOL = "bool"
    BOOLEAN = "boolean"
    SMALLINT = "smallint" 
    MEDIUMINT = "mediumint" 
    INT = "int" 
    INTEGER = "integer" 
    BIGINT = "bigint" 
    DECIMAL = "decimal" 
    NUMERIC = "numeric" 
    FLOAT = "float" 
    DOUBLE = "double" 
    DOUBLE_PRECISION = "double precision" 

    # Date and time types
    DATE = "date" 
    DATETIME = "datetime" 
    TIMESTAMP = "timestamp" 
    TIME = "time" 
    YEAR = "year" 


# TPhysical storage sizes for fixed-size types
FIXED_SIZE = {
    DataType.TINYINT: 1,
    DataType.SMALLINT: 2,
    DataType.MEDIUMINT: 3,
    DataType.INT: 4,
    DataType.INTEGER: 4,
    DataType.BIGINT: 8,
    DataType.FLOAT: 4,
    DataType.DOUBLE: 8,
    DataType.DOUBLE_PRECISION: 8,
    DataType.YEAR: 1,
}


# Types whose declared size (`Column.size`) is required to know their max footprint.
VARIABLE_SIZE_TYPES = {
    DataType.CHAR,
    DataType.VARCHAR,
    DataType.BINARY,
    DataType.VARBINARY,
    DataType.TEXT,
    DataType.TINYTEXT,
    DataType.MEDIUMTEXT,
    DataType.LONGTEXT,
    DataType.SET,
}


class Value:
    """
    Value represents a typed value within the database engine.

    A Value contains:
        data_type: the SQL type
        data: the actual Python value
    """
    def __init__(self,  data_type: DataType,  data):
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
        
        if self.data_type in {
            DataType.VARCHAR,
            DataType.CHAR,
            DataType.TEXT,
            DataType.TINYTEXT,
            DataType.MEDIUMTEXT,
            DataType.LONGTEXT,
            DataType.SET
        }:
            return isinstance(self.data, str)

        if self.data_type in {
            DataType.BINARY,
            DataType.VARBINARY
        }:
            return isinstance(self.data, (bytes, bytearray))

        if self.data_type in {
            DataType.TINYINT,
            DataType.SMALLINT,
            DataType.MEDIUMINT,
            DataType.INT,
            DataType.INTEGER,
            DataType.BIGINT,
            DataType.BIT
        }:
            return isinstance(self.data, int) and not isinstance(self.data, bool)

        if self.data_type in {
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.DOUBLE_PRECISION
        }:
            return isinstance(self.data, (int, float)) and not isinstance(self.data, bool)

        if self.data_type in {
            DataType.DECIMAL,
            DataType.NUMERIC
        }:
            return isinstance(self.data, (Decimal, int, float)) and not isinstance(self.data, bool)

        if self.data_type in {
            DataType.BOOL,
            DataType.BOOLEAN
        }:
            return isinstance(self.data, bool)

        if self.data_type == DataType.DATE:
            return isinstance(self.data, date) and not isinstance(self.data, datetime)

        if self.data_type in {
            DataType.DATETIME,
            DataType.TIMESTAMP
        }:
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
            raise ValueError(
                f"Valor inválido para el tipo {self.data_type}: {self.data!r}"
            )

        if self.data_type in FIXED_SIZE:
            return FIXED_SIZE[self.data_type]

        if self.data_type in {
            DataType.CHAR,
            DataType.VARCHAR,
            DataType.TEXT,
            DataType.TINYTEXT,
            DataType.MEDIUMTEXT,
            DataType.LONGTEXT,
            DataType.SET
        }:
            return len(self.data.encode("utf-8"))

        if self.data_type in {
            DataType.BINARY,
            DataType.VARBINARY
        }:
            return len(self.data)

        if self.data_type in {
            DataType.BOOL,
            DataType.BOOLEAN
        }:
            return 1

        if self.data_type == DataType.BIT:
            return 1

        if self.data_type in {
            DataType.DECIMAL,
            DataType.NUMERIC
        }:
            # Temporary physical representation.
            # # The exact decimal encoding can be defined later.
            return len(str(self.data).encode("utf-8"))

        if self.data_type == DataType.DATE:
            return 4

        if self.data_type in {
            DataType.DATETIME,
            DataType.TIMESTAMP
        }:
            return 8

        if self.data_type == DataType.TIME:
            return 8

        raise ValueError(
            f"No se conoce el tamaño físico de {self.data_type}"
        )