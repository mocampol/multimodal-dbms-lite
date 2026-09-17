"""
Database schema definitions.

Classes:
    Column: Describes the definition of a single table column.
    Schema: Describes the structure of a database table.
"""


from .value import DataType, VARIABLE_SIZE_TYPES, Value


class Column:
    """
    Describes a column in a database table.

    Attributes:
        name (str): Name of the column.
        data_type (DataType): Data type assigned to the column.
        size (int | None): Required for variable-size types (CHAR, VARCHAR, BINARY, VARBINARY, ...).
        is_primary_key (bool): Whether the column is the table's primary key.
        nullable (bool): Whether NULL is allowed. A primary key is always NOT NULL.
        is_unique (bool): Whether values must be unique. A primary key is always UNIQUE.
            NOTE: this class only stores the flag — actually enforcing uniqueness
            requires checking existing rows, which is catalog/storage's job, not this class's.
    """
    def __init__(
        self,
        name: str,
        data_type: DataType,
        size: int = None,
        is_primary_key: bool = False,
        nullable: bool = True,
        is_unique: bool = False,
    ):
        self.name = name
        self.data_type = data_type
        self.size = size
        self.is_primary_key = is_primary_key
        self.nullable = False if is_primary_key else nullable
        self.is_unique = True if is_primary_key else is_unique

        self._validate_definition()


    def _validate_definition(self):
        """
        Validates the definition of the column (structural checks only,
        not the data that will later be stored in it).
        """
        if self.size is not None and self.size <= 0:
            raise ValueError(f"El size de '{self.name}' debe ser mayor que 0")

        if self.data_type in VARIABLE_SIZE_TYPES and self.size is None:
            raise ValueError(f"{self.data_type.value.upper()} '{self.name}' requiere un size")


    def validate(self, value: Value) -> bool:
        """
        Checks whether a Value is valid for this column: type, NULL, and size.
        Does NOT check UNIQUE, that requires looking at other rows (catalog/storage).
        """
        if not isinstance(value, Value):
            return False

        if value.data is None:
            return self.nullable

        if value.data_type != self.data_type:
            return False

        if not value.validate():
            return False

        if self.size is not None and self.data_type in VARIABLE_SIZE_TYPES:
            if len(value.data) > self.size:
                return False

        return True


    def __repr__(self):
        flags = []
        if self.is_primary_key:
            flags.append("PK")
        if self.is_unique and not self.is_primary_key:
            flags.append("UNIQUE")
        if not self.nullable:
            flags.append("NOT NULL")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        size_str = f"({self.size})" if self.size is not None else ""
        return f"{self.name} {self.data_type.value}{size_str}{flag_str}"


class Schema:
    """
    Describes the structure of a database table.

    Attributes:
        table_name (str): Name of the table.
        columns (list[Column]): Ordered list of column definitions.
    """
    def __init__(self, table_name: str, columns: list[Column]):
        self.table_name = table_name
        self.columns = columns
        self._validate_definition()


    def _validate_definition(self):
        """
        Validates the schema definition: non-empty name, at least one column,
        and no duplicate column names.
        """
        if not self.table_name:
            raise ValueError("El nombre de la tabla no puede estar vacío")
        if not self.columns:
            raise ValueError("Una tabla debe tener al menos una columna")

        names = set()
        pk_count = 0
        for column in self.columns:
            if column.name in names:
                raise ValueError(f"Columna duplicada: {column.name}")
            names.add(column.name)
            if column.is_primary_key:
                pk_count += 1

        if pk_count > 1:
            raise ValueError(
                f"'{self.table_name}' tiene más de una columna marcada como primary key "
                "(primary key compuesta no soportada por ahora)"
            )


    def get_column(self, name: str) -> Column | None:
        """
        Returns the column with the given name, or None if it doesn't exist.
        """
        for column in self.columns:
            if column.name == name:
                return column
        return None


    def column_index(self, name: str) -> int:
        """
        Returns the index of the column with the given name.
        Raises ValueError if no such column exists.
        """
        for i, column in enumerate(self.columns):
            if column.name == name:
                return i
        raise ValueError(f"La columna '{name}' no existe")


    def unique_columns(self) -> list[str]:
        """
        Returns names of columns with a UNIQUE constraint (primary key included).
        Useful for catalog/heap to know which columns need a uniqueness check
        before inserting a new record.
        """
        return [c.name for c in self.columns if c.is_unique]


    def primary_key(self) -> Column | None:
        """
        Returns the primary key column, or None if the table has no PK.
        """
        for c in self.columns:
            if c.is_primary_key:
                return c
        return None


    def __len__(self) -> int:
        return len(self.columns)


    def __repr__(self):
        cols = ", ".join(repr(c) for c in self.columns)
        return f"Schema({self.table_name}: {cols})"
