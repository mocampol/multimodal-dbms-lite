"""
SQL type system.

Classes:
    DataType: Enumeration of supported SQL data types.
    Value: A SQL value paired with its corresponding DataType.
"""

from enum import Enum

class DataType(Enum):
    """
    SQL data types supported by the database system.

    Data types are grouped into three categories:
        - String types
        - Numeric types
        - Date and time types
    """

    # String types
    CHAR = "char",
    VARCHAR = "varchar",
    BINARY = "binary",
    VARBINARY = "varbinary",
    TEXT = "text",
    TINYTEXT = "tinytext",
    MEDIUMTEXT = "mediumtext",
    LONGTEXT = "longtext",
    SET = "set",

    # Numeric types
    BIT = "bit",
    TINYINT = "tinyint",
    BOOL = "bool",
    BOOLEAN = "boolean",
    SMALLINT = "smallint",
    MEDIUMINT = "mediumint",
    INT = "int",
    INTEGER = "integer",
    BIGINT = "bigint",
    DECIMAL = "decimal",
    NUMERIC = "numeric",
    FLOAT = "float",
    DOUBLE = "double",
    DOUBLE_PRECISION = "double precision",

    # Date and time types
    DATE = "date",
    DATETIME = "datetime",
    TIMESTAMP = "timestamp",
    TIME = "time",
    YEAR = "year",


class Value:
    """
    Value represents a typed value within the database engine.
    """
    
    def __init__(self, data_type: DataType, data):
        self.data_type = data_type
        self.data = data