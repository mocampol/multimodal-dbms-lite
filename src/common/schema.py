"""
Database schema definitions.

Classes:
    Column: Describes the definition of a single table column.
    Schema: Describes the structure of a database table.
"""

from .value import DataType

class Column:
    """
    Describes a column in a database table.

    Attributes:
        name (str): Name of the column.
        data_type: Data type assigned to the column.
        size (int | None): Optional size constraint for the column.
        is_primary_key (bool): Whether the column is part of the table's primary key.
    """

    def __init__(
        self,
        name: str,
        data_type: DataType,
        size: int = None,
        is_primary_key: bool =False,
    ):
        self.name = name
        self.data_type = data_type
        self.size = size
        self.is_primary_key = is_primary_key

class Schema:
    """
    Describes the structure of a database table.
    
    A Schema associates a table name with an ordered collection of Column definitions.
    
    Attributes:
        table_name (str): Name of the table.
        columns (list[Column]): Ordered list of column definitions.
    """

    def __init__(self, table_name: str, columns: list[Column]):
        self.table_name = table_name
        self.columns = columns

    def get_column(self, name: str) -> Column | None:
        """
        Returns the column with the given name.
        If no such column exists, returns None.
        """

        for c in self.columns:
            if c.name == name:
                return c
        return None

    def column_index(self, name: str) -> int:
        """
        Returns the index of the column with the given name.
        Raises ValueError if no such column exists.
        """

        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        raise ValueError(f"La columna {name} no existe")