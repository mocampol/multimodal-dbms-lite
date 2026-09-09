# =============================================================================
# visitor.py — Visitantes del AST del compilador SQL
# =============================================================================
# Análogo a visitor.h / visitor.cpp del compilador general de tu profesor,
# adaptado al dominio de SQL (tablas/columnas en vez de variables/funciones).
#
# Se implementa un visitante:
#
#   SemanticVisitor — Análisis semántico (Fase 3 del Query Parser):
#     · Verifica que la tabla referenciada exista en el catálogo.
#     · Verifica que las columnas referenciadas existan en esa tabla.
#     · Verifica compatibilidad de tipos en condiciones del WHERE.
#     · Verifica aridad (cantidad de valores) en INSERT.
#     · Verifica compatibilidad de tipos en INSERT.
#
# Diferencias de diseño frente al visitor.h/.cpp del compilador general:
#   - Ese AST tenía variables con scope (funciones, bloques anidados), por
#     lo que el TypeCheckerVisitor necesitaba un Environment<T> para
#     manejar scopes. SQL no tiene scopes anidados de variables: el
#     "entorno" es simplemente el catálogo (tablas/columnas), así que no
#     se necesita Environment aquí.
#   - No existe un GenCodeVisitor que emita ensamblador: quien "ejecuta"
#     una sentencia SQL ya validada es el Query Executor de tu equipo,
#     que recorre el AST (posiblemente con su propio Visitor) contra el
#     Storage Manager. Este archivo se limita a la fase de validación
#     semántica, que es lo que le corresponde al Query Parser.
#   - No hay equivalentes a ConstantTaggingVisitor, SethiUllmanLabelVisitor
#     ni CascadeFunctionEvalVisitor: son optimizaciones de generación de
#     código de bajo nivel (plegado de constantes, asignación de
#     registros, evaluación en cascada de funciones) que no aplican a un
#     árbol de una sola sentencia SQL sin funciones ni bucles.
#
# Este archivo NO depende del Storage Manager real: define un
# CatalogProtocol mínimo (duck typing) más una implementación en memoria
# (InMemoryCatalog) para poder probar el visitor de forma aislada. Cuando
# el catálogo real (sys_tables/sys_columns sobre Heap File) esté listo,
# solo necesita cumplir table_exists()/get_table() para conectarse aquí
# sin cambiar una sola línea de SemanticVisitor.
# =============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from ast_nodes import (
    Visitor,
    Exp,
    NumExp,
    IdExp,
    BinaryExp,
    BinaryOp,
    Stm,
    SelectStm,
    InsertStm,
    DeleteStm,
    OrderByClause,
    GroupByClause,
)


class SemanticError(Exception):
    """Error de análisis semántico: tabla/columna inexistente, tipos
    incompatibles, aridad incorrecta en INSERT, etc. Análogo a los
    std::runtime_error que lanza el TypeCheckerVisitor en C++."""


# =============================================================================
# Catálogo — contrato mínimo que necesita el análisis semántico
# =============================================================================

class ColumnType:
    """Tipos de columna soportados. Ajusta/extiende esto según lo que tu
    equipo termine definiendo en sys_columns.col_type."""
    INT = "INT"
    TEXT = "TEXT"


@dataclass
class ColumnInfo:
    name: str
    type: str  # uno de ColumnType


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]

    def has_column(self, name: str) -> bool:
        return any(c.name == name for c in self.columns)

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


class CatalogProtocol(Protocol):
    """Contrato mínimo que SemanticVisitor necesita del catálogo. Es un
    Protocol (duck typing): el catálogo real del Storage Manager NO
    necesita heredar de esta clase, solo implementar estos dos métodos."""

    def table_exists(self, table_name: str) -> bool: ...

    def get_table(self, table_name: str) -> TableInfo: ...


class InMemoryCatalog:
    """Catálogo de referencia en memoria. Útil para probar el parser y el
    visitor de forma aislada mientras el catálogo real (respaldado por
    Heap File, sys_tables/sys_columns) no está listo."""

    def __init__(self):
        self._tables: Dict[str, TableInfo] = {}

    def define_table(self, name: str, columns: List[ColumnInfo]) -> None:
        self._tables[name] = TableInfo(name, columns)

    def table_exists(self, table_name: str) -> bool:
        return table_name in self._tables

    def get_table(self, table_name: str) -> TableInfo:
        if table_name not in self._tables:
            raise SemanticError(f"La tabla '{table_name}' no existe en el catálogo")
        return self._tables[table_name]


# =============================================================================
# SemanticVisitor — Análisis semántico
# =============================================================================

class SemanticVisitor(Visitor):
    """Recorre el AST de una sentencia SQL ya parseada y valida que sea
    semánticamente correcta contra el catálogo, antes de que el Query
    Executor la ejecute. Análogo a TypeCheckerVisitor en el compilador
    general.

    Uso:
        catalog = InMemoryCatalog()
        catalog.define_table("alumnos", [
            ColumnInfo("id", ColumnType.INT),
            ColumnInfo("nombre", ColumnType.TEXT),
            ColumnInfo("edad", ColumnType.INT),
        ])

        stm = parser.parse_sql_statement()
        SemanticVisitor(catalog).check(stm)   # lanza SemanticError si algo está mal
    """

    def __init__(self, catalog: CatalogProtocol):
        self.catalog = catalog
        # Tabla contra la que se resuelven las columnas "sueltas"
        # mencionadas en la sentencia que se está visitando actualmente
        # (columnas del SELECT, lado izquierdo de <Condition>, columnas
        # de ORDER BY / GROUP BY). Análogo al concepto de "scope actual"
        # del Environment<T> en el compilador general, pero mucho más
        # simple: aquí solo hay un nivel.
        self._current_table: Optional[TableInfo] = None

    # -------------------------------------------------------------------
    # Punto de entrada — análogo a TypeChecker(Program*) en C++
    # -------------------------------------------------------------------
    def check(self, stm: Stm) -> None:
        stm.accept(self)

    # -------------------------------------------------------------------
    # Sentencias
    # -------------------------------------------------------------------
    def visit_select_stm(self, stm: SelectStm):
        table = self.catalog.get_table(stm.table)
        self._current_table = table
        try:
            if stm.columns != ["*"]:
                for col in stm.columns:
                    self._require_column(table, col)

            if stm.where_cond is not None:
                stm.where_cond.accept(self)

            if stm.order_by is not None:
                stm.order_by.accept(self)

            if stm.group_by is not None:
                stm.group_by.accept(self)
        finally:
            self._current_table = None

        return None

    def visit_insert_stm(self, stm: InsertStm):
        table = self.catalog.get_table(stm.table)
        self._current_table = table
        try:
            if len(stm.values) != len(table.columns):
                raise SemanticError(
                    f"INSERT INTO {stm.table}: se esperaban {len(table.columns)} "
                    f"valores (uno por columna), pero se recibieron {len(stm.values)}"
                )

            for column, value_exp in zip(table.columns, stm.values):
                value_type = value_exp.accept(self)
                self._check_type_compatible(column, value_type, context=f"INSERT INTO {stm.table}")
        finally:
            self._current_table = None

        return None

    def visit_delete_stm(self, stm: DeleteStm):
        table = self.catalog.get_table(stm.table)
        self._current_table = table
        try:
            if stm.where_cond is not None:
                stm.where_cond.accept(self)
        finally:
            self._current_table = None

        return None

    # -------------------------------------------------------------------
    # Cláusulas auxiliares
    # -------------------------------------------------------------------
    def visit_order_by_clause(self, clause: OrderByClause):
        for col in clause.columns:
            self._require_column(self._current_table, col)
        return None

    def visit_group_by_clause(self, clause: GroupByClause):
        for col in clause.columns:
            self._require_column(self._current_table, col)
        return None

    # -------------------------------------------------------------------
    # Expresiones — cada visit_* devuelve el ColumnType resuelto, así
    # visit_binary_exp puede comparar tipos de sus dos operandos.
    # -------------------------------------------------------------------
    def visit_num_exp(self, exp: NumExp):
        return ColumnType.INT

    def visit_id_exp(self, exp: IdExp):
        """Todo IdExp dentro de una sentencia se trata como referencia a
        una columna de la tabla activa (ya sea la columna evaluada en
        <Condition>, o un <Value> que compara columna contra columna).
        Se devuelve el tipo declarado de esa columna en el catálogo."""
        column = self._require_column(self._current_table, exp.value)
        return column.type

    def visit_binary_exp(self, exp: BinaryExp):
        left_type = exp.left.accept(self)
        right_type = exp.right.accept(self)

        if left_type != right_type:
            raise SemanticError(
                f"Tipos incompatibles en condición '{exp!r}': "
                f"'{exp.left!r}' es {left_type} pero '{exp.right!r}' es {right_type}"
            )

        # <, <=, >, >= solo tienen sentido sobre columnas numéricas.
        # (= sí aplica a TEXT, por eso EQ_OP queda fuera de esta validación.)
        if exp.op in (BinaryOp.LE_OP, BinaryOp.LEQ_OP, BinaryOp.GT_OP, BinaryOp.GEQ_OP):
            if left_type != ColumnType.INT:
                raise SemanticError(
                    f"El operador '{exp!r}' solo aplica a columnas numéricas, "
                    f"pero '{exp.left!r}' es de tipo {left_type}"
                )

        return ColumnType.INT  # una condición es booleana; no hay BOOL en ColumnType todavía

    # -------------------------------------------------------------------
    # Helpers internos
    # -------------------------------------------------------------------
    def _require_column(self, table: Optional[TableInfo], col_name: str) -> ColumnInfo:
        if table is None:
            raise SemanticError(f"No hay tabla activa para resolver la columna '{col_name}'")
        column = table.get_column(col_name)
        if column is None:
            raise SemanticError(f"La columna '{col_name}' no existe en la tabla '{table.name}'")
        return column

    def _check_type_compatible(self, column: ColumnInfo, value_type: str, context: str) -> None:
        if column.type != value_type:
            raise SemanticError(
                f"{context}: la columna '{column.name}' es de tipo {column.type}, "
                f"pero se recibió un valor de tipo {value_type}"
            )