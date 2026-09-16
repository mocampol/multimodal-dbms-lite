from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional


# =============================================================================
# Operadores de comparación (<Operator> ::= EQ | LE | LEQ | GT | GEQ)
# =============================================================================

class BinaryOp(Enum):
    EQ_OP = auto()   # =
    LE_OP = auto()   # <
    LEQ_OP = auto()  # <=
    GT_OP = auto()   # >
    GEQ_OP = auto()  # >=


_BINOP_CHARS = {
    BinaryOp.EQ_OP: "=",
    BinaryOp.LE_OP: "<",
    BinaryOp.LEQ_OP: "<=",
    BinaryOp.GT_OP: ">",
    BinaryOp.GEQ_OP: ">=",
}


def binop_to_char(op: BinaryOp) -> str:
    return _BINOP_CHARS.get(op, "?")


# =============================================================================
# Visitor — interfaz para recorrer el AST (patrón Visitor)
# =============================================================================

class Visitor(ABC):

    # ---- Expresiones ----
    @abstractmethod
    def visit_num_exp(self, exp: "NumExp"):
        ...

    @abstractmethod
    def visit_id_exp(self, exp: "IdExp"):
        ...

    @abstractmethod
    def visit_string_exp(self, exp: "StringExp"):
        ...

    @abstractmethod
    def visit_binary_exp(self, exp: "BinaryExp"):
        ...

    # ---- Sentencias ----
    @abstractmethod
    def visit_select_stm(self, stm: "SelectStm"):
        ...

    @abstractmethod
    def visit_insert_stm(self, stm: "InsertStm"):
        ...

    @abstractmethod
    def visit_delete_stm(self, stm: "DeleteStm"):
        ...

    # ---- Cláusulas auxiliares ----
    @abstractmethod
    def visit_order_by_clause(self, clause: "OrderByClause"):
        ...

    @abstractmethod
    def visit_group_by_clause(self, clause: "GroupByClause"):
        ...


# =============================================================================
# Expresiones (<Condition>, <Value>)
# =============================================================================

class Exp(ABC):
    """Nodo base abstracto para toda expresión."""

    @abstractmethod
    def accept(self, visitor: Visitor):
        ...


class NumExp(Exp):
    """<Value> ::= NUM"""

    def __init__(self, value: int):
        self.value = value

    def accept(self, visitor: Visitor):
        return visitor.visit_num_exp(self)

    def __repr__(self):
        return str(self.value)


class IdExp(Exp):
    """<Value> ::= ID  (también se usa como lado izquierdo de <Condition>)"""

    def __init__(self, value: str):
        self.value = value

    def accept(self, visitor: Visitor):
        return visitor.visit_id_exp(self)

    def __repr__(self):
        return self.value


class StringExp(Exp):
    """<Value> ::= STRING — literal de texto entre comillas simples: 'texto'"""

    def __init__(self, value: str):
        self.value = value

    def accept(self, visitor: Visitor):
        return visitor.visit_string_exp(self)

    def __repr__(self):
        return f"'{self.value}'"


class BinaryExp(Exp):
    """<Condition> ::= ID <Operator> <Value>"""

    def __init__(self, left: Exp, right: Exp, op: BinaryOp):
        self.left = left
        self.right = right
        self.op = op

    def accept(self, visitor: Visitor):
        return visitor.visit_binary_exp(self)

    def __repr__(self):
        return f"({self.left!r} {binop_to_char(self.op)} {self.right!r})"


# =============================================================================
# Cláusulas auxiliares (<GroupOrOrder>)
# =============================================================================

class OrderByClause:
    """ORDER_BY ID { COMA ID }"""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns: List[str] = columns if columns is not None else []

    def accept(self, visitor: Visitor):
        return visitor.visit_order_by_clause(self)

    def __repr__(self):
        return f"ORDER BY {', '.join(self.columns)}"


class GroupByClause:
    """GROUP_BY ID { COMA ID }"""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns: List[str] = columns if columns is not None else []

    def accept(self, visitor: Visitor):
        return visitor.visit_group_by_clause(self)

    def __repr__(self):
        return f"GROUP BY {', '.join(self.columns)}"


# =============================================================================
# Sentencias (<SelectStmt> | <InsertStmt> | <DeleteStmt>)
# =============================================================================

class Stm(ABC):
    """Nodo base abstracto para toda sentencia SQL."""

    @abstractmethod
    def accept(self, visitor: Visitor):
        ...


class SelectStm(Stm):
    """<SelectStmt> ::= SELECT <SelectList> FROM ID [ <WhereClause> ] [ <GroupOrOrder> ]"""

    def __init__(
        self,
        columns: List[str],
        table: str,
        where_cond: Optional[Exp] = None,
        order_by: Optional[OrderByClause] = None,
        group_by: Optional[GroupByClause] = None,
    ):
        self.columns = columns
        self.table = table
        self.where_cond = where_cond
        self.order_by = order_by
        self.group_by = group_by

    def accept(self, visitor: Visitor):
        return visitor.visit_select_stm(self)

    def __repr__(self):
        parts = [f"SELECT {', '.join(self.columns)} FROM {self.table}"]
        if self.where_cond is not None:
            parts.append(f"WHERE {self.where_cond!r}")
        if self.order_by is not None:
            parts.append(repr(self.order_by))
        if self.group_by is not None:
            parts.append(repr(self.group_by))
        return " ".join(parts)


class InsertStm(Stm):
    """<InsertStmt> ::= INSERT_INTO ID VALUES LPAREN <ValueList> RPAREN"""

    def __init__(self, table: str, values: List[Exp]):
        self.table = table
        self.values = values

    def accept(self, visitor: Visitor):
        return visitor.visit_insert_stm(self)

    def __repr__(self):
        vals = ", ".join(repr(v) for v in self.values)
        return f"INSERT INTO {self.table} VALUES ({vals})"


class DeleteStm(Stm):
    """<DeleteStmt> ::= DELETE FROM ID [ <WhereClause> ]"""

    def __init__(self, table: str, where_cond: Optional[Exp] = None):
        self.table = table
        self.where_cond = where_cond

    def accept(self, visitor: Visitor):
        return visitor.visit_delete_stm(self)

    def __repr__(self):
        base = f"DELETE FROM {self.table}"
        if self.where_cond is not None:
            base += f" WHERE {self.where_cond!r}"
        return base