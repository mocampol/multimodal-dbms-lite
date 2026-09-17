from typing import Optional, Protocol, Tuple, Union

from common import DataType, Value, Column, Schema

from ast_nodes import (
    Visitor,
    Exp,
    NumExp,
    IdExp,
    StringExp,
    BinaryExp,
    BinaryOp,
    Stm,
    SelectStm,
    InsertStm,
    DeleteStm,
    OrderByClause,
    GroupByClause,
    CreateTableStm,
    CreateIndexStm,
    IndexType,
)


class SemanticError(Exception):
    """Error de análisis semántico: tabla/columna inexistente, tipos
    incompatibles, aridad incorrecta en INSERT, valor que excede el
    tamaño de una columna, etc. Análogo a los std::runtime_error que
    lanza el TypeCheckerVisitor en C++."""


_NUMERIC_TYPES = {
    DataType.SMALLINT, DataType.INTEGER, DataType.BIGINT,
    DataType.NUMERIC, DataType.REAL, DataType.DOUBLE_PRECISION,
}

_DATE_TYPES = {
    DataType.DATE, DataType.TIME, DataType.TIMESTAMP,
}

_STRING_TYPES = {
    DataType.CHAR, DataType.VARCHAR, DataType.TEXT,
}

ORDERABLE_TYPES = _NUMERIC_TYPES | _DATE_TYPES | _STRING_TYPES


class CatalogProtocol(Protocol):
    """Minimum contract that SemanticVisitor requires from the catalog. It is a
    Protocol (duck typing): the actual Storage Manager catalog does NOT
    need to inherit from this class; it only needs to implement these
    four methods.

    table_exists / get_schema: used to validate SELECT/INSERT/DELETE
    against a table that must already exist.

    create_table / create_index: used to validate AND EXECUTE
    CREATE TABLE / CREATE INDEX. Unlike the read-only methods above,
    these are mutating calls — visit_create_table_stm/visit_create_index_stm
    invoke them directly as part of semantic checking, since for these
    two statements "is it valid?" and "make it so" are the same step
    (there's no separate physical plan to build afterwards, unlike
    SELECT/INSERT/DELETE, which the executor runs later).
    """

    def table_exists(self, table_name: str) -> bool: ...

    def get_schema(self, table_name: str) -> Schema: ...

    def create_table(self, schema: Schema) -> None: ...

    def create_index(self, table_name: str, column_name: str, index_type: str) -> None: ...


class InMemoryCatalog:
    """An in-memory reference catalog based on actual Schema/Columns from
    common. Useful for testing the parser and visitor in isolation
    while the actual catalog (backed by a heap file, sys_tables/
    sys_columns) is not yet ready."""

    def __init__(self):
        self._schemas: dict[str, Schema] = {}
        self._indexes: dict[str, list[dict]] = {}

    def define_table(self, schema: Schema) -> None:
        self._schemas[schema.table_name] = schema

    def table_exists(self, table_name: str) -> bool:
        return table_name in self._schemas

    def get_schema(self, table_name: str) -> Schema:
        if table_name not in self._schemas:
            raise SemanticError(f"La tabla '{table_name}' no existe en el catálogo")
        return self._schemas[table_name]

    def create_table(self, schema: Schema) -> None:
        if schema.table_name in self._schemas:
            raise SemanticError(f"La tabla '{schema.table_name}' ya existe")
        self._schemas[schema.table_name] = schema

    def create_index(self, table_name: str, column_name: str, index_type: str) -> None:
        self.get_schema(table_name)  # raises if the table doesn't exist
        self._indexes.setdefault(table_name, []).append({
            "column_name": column_name,
            "index_type": index_type,
        })


_ExpResult = Tuple[str, Union[Column, int, str]]


class SemanticVisitor(Visitor):
    """Iterates through the AST of a parsed SQL statement and validates that it is
    semantically correct against the catalog, before the Query
    Executor executes it. Analogous to TypeCheckerVisitor in the general
    compiler.

    Usage:
        from common import DataType, Column, Schema

        catalog = InMemoryCatalog()
        catalog.define_table(Schema("students", [
            Column("id", DataType.INTEGER, is_primary_key=True),
            Column("name", DataType.VARCHAR, size=50),
            Column("age", DataType.INTEGER),
        ]))

        stm = parser.parse_sql_statement()
        SemanticVisitor(catalog).check(stm)
    """

    def __init__(self, catalog: CatalogProtocol):
        self.catalog = catalog
        self._current_schema: Optional[Schema] = None

    def check(self, stm: Stm) -> None:
        stm.accept(self)

    def visit_select_stm(self, stm: SelectStm):
        schema = self.catalog.get_schema(stm.table)
        self._current_schema = schema
        try:
            if stm.columns != ["*"]:
                for col in stm.columns:
                    self._require_column(schema, col)

            if stm.where_cond is not None:
                stm.where_cond.accept(self)

            if stm.order_by is not None:
                stm.order_by.accept(self)

            if stm.group_by is not None:
                stm.group_by.accept(self)
        finally:
            self._current_schema = None

        return None

    def visit_insert_stm(self, stm: InsertStm):
        schema = self.catalog.get_schema(stm.table)
        self._current_schema = schema
        try:
            if len(stm.values) != len(schema.columns):
                raise SemanticError(
                    f"INSERT INTO {stm.table}: se esperaban {len(schema.columns)} "
                    f"valores (uno por columna), pero se recibieron {len(stm.values)}"
                )

            for column, value_exp in zip(schema.columns, stm.values):
                kind, payload = value_exp.accept(self)

                if kind == "column":
                    raise SemanticError(
                        f"INSERT INTO {stm.table}: '{payload.name}' es una referencia a "
                        f"columna, no un literal. INSERT VALUES solo acepta literales "
                        f"(NUM o STRING), no nombres de columna."
                    )

                value = Value(column.data_type, payload)
                if not column.validate(value):
                    raise SemanticError(
                        f"INSERT INTO {stm.table}: el valor {payload!r} no es válido para "
                        f"la columna '{column.name}' ({column!r})"
                    )
        finally:
            self._current_schema = None

        return None

    def visit_delete_stm(self, stm: DeleteStm):
        schema = self.catalog.get_schema(stm.table)
        self._current_schema = schema
        try:
            if stm.where_cond is not None:
                stm.where_cond.accept(self)
        finally:
            self._current_schema = None

        return None

    def visit_create_table_stm(self, stm: CreateTableStm):
        """
        Validates and EXECUTES a CREATE TABLE statement. Unlike SELECT/
        INSERT/DELETE, there's no separate plan for the executor to run
        afterwards — the table is created right here, as part of semantic
        checking, by calling catalog.create_table().
        """
        if self.catalog.table_exists(stm.table):
            raise SemanticError(f"La tabla '{stm.table}' ya existe")

        if not stm.columns:
            raise SemanticError(f"CREATE TABLE {stm.table}: se requiere al menos una columna")

        seen_names = set()
        pk_count = 0
        for col_def in stm.columns:
            if col_def.name in seen_names:
                raise SemanticError(
                    f"CREATE TABLE {stm.table}: columna duplicada '{col_def.name}'"
                )
            seen_names.add(col_def.name)
            if col_def.is_primary_key:
                pk_count += 1

        if pk_count > 1:
            raise SemanticError(
                f"CREATE TABLE {stm.table}: más de una columna marcada como PRIMARY KEY "
                "(llave primaria compuesta no soportada)"
            )

        columns = [
            Column(
                name=col_def.name,
                data_type=col_def.data_type,
                size=col_def.size,
                is_primary_key=col_def.is_primary_key,
                nullable=col_def.nullable,
                is_unique=col_def.is_unique,
            )
            for col_def in stm.columns
        ]
        schema = Schema(stm.table, columns)

        self.catalog.create_table(schema)
        return None

    def visit_create_index_stm(self, stm: CreateIndexStm):
        """
        Validates and EXECUTES a CREATE INDEX statement, same rationale
        as visit_create_table_stm: no separate plan needed afterwards.
        """
        schema = self.catalog.get_schema(stm.table)  # raises SemanticError if missing
        self._require_column(schema, stm.column)

        self.catalog.create_index(stm.table, stm.column, stm.index_type.name.lower())
        return None

    def visit_order_by_clause(self, clause: OrderByClause):
        for col in clause.columns:
            self._require_column(self._current_schema, col)
        return None

    def visit_group_by_clause(self, clause: GroupByClause):
        for col in clause.columns:
            self._require_column(self._current_schema, col)
        return None

    def visit_num_exp(self, exp: NumExp) -> _ExpResult:
        return ("literal", exp.value)

    def visit_string_exp(self, exp: StringExp) -> _ExpResult:
        return ("literal", exp.value)

    def visit_id_exp(self, exp: IdExp) -> _ExpResult:
        """An IdExp always represents a column reference from the
        current table: the left-hand side of <Condition>, a column in
        ORDER BY/GROUP BY, or (in <Value>) a column-to-column comparison."""
        column = self._require_column(self._current_schema, exp.value)
        return ("column", column)

    def visit_binary_exp(self, exp: BinaryExp):
        left_kind, left_column = exp.left.accept(self)
        assert left_kind == "column"

        right_kind, right_payload = exp.right.accept(self)

        if right_kind == "column":
            right_column: Column = right_payload
            if left_column.data_type != right_column.data_type:
                raise SemanticError(
                    f"Tipos incompatibles en condición '{exp!r}': "
                    f"'{left_column.name}' es {left_column.data_type.value} pero "
                    f"'{right_column.name}' es {right_column.data_type.value}"
                )
            compare_type = left_column.data_type
        else:
            literal = right_payload
            value = Value(left_column.data_type, literal)
            if not left_column.validate(value):
                raise SemanticError(
                    f"Tipos incompatibles en condición '{exp!r}': el valor {literal!r} "
                    f"no es válido para la columna '{left_column.name}' ({left_column!r})"
                )
            compare_type = left_column.data_type

        if exp.op in (BinaryOp.LE_OP, BinaryOp.LEQ_OP, BinaryOp.GT_OP, BinaryOp.GEQ_OP):
            if compare_type not in ORDERABLE_TYPES:
                raise SemanticError(
                    f"El operador '{exp!r}' no aplica sobre columnas de tipo "
                    f"{compare_type.value} (no son ordenables)"
                )

        return None

    def _require_column(self, schema: Optional[Schema], col_name: str) -> Column:
        if schema is None:
            raise SemanticError(f"No hay tabla activa para resolver la columna '{col_name}'")
        column = schema.get_column(col_name)
        if column is None:
            raise SemanticError(f"La columna '{col_name}' no existe en la tabla '{schema.table_name}'")
        return column