from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional

from common import DataType


class BinaryOp(Enum):
    EQ_OP = auto()   # =
    LE_OP = auto()   # 
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


# Index type (<IndexType> ::= BTREE | HASH)
class IndexType(Enum):
    BTREE = auto()
    HASH = auto()


# Visitor — interface for traversing the AST (Visitor pattern)
class Visitor(ABC):

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

    @abstractmethod
    def visit_select_stm(self, stm: "SelectStm"):
        ...

    @abstractmethod
    def visit_insert_stm(self, stm: "InsertStm"):
        ...

    @abstractmethod
    def visit_delete_stm(self, stm: "DeleteStm"):
        ...

    @abstractmethod
    def visit_create_table_stm(self, stm: "CreateTableStm"):
        ...

    @abstractmethod
    def visit_create_index_stm(self, stm: "CreateIndexStm"):
        ...

    @abstractmethod
    def visit_order_by_clause(self, clause: "OrderByClause"):
        ...

    @abstractmethod
    def visit_group_by_clause(self, clause: "GroupByClause"):
        ...


# Expressions (<Condition>, <Value>)
class Exp(ABC):
    """Abstract base node for every expression."""

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
    """<Value> ::= ID  (also used as the left-hand side of <Condition>)"""

    def __init__(self, value: str):
        self.value = value

    def accept(self, visitor: Visitor):
        return visitor.visit_id_exp(self)

    def __repr__(self):
        return self.value


class StringExp(Exp):
    """<Value> ::= STRING — text literal enclosed in single quotes: 'text'"""

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


# Auxiliary clauses (<GroupOrOrder>)
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


# CREATE TABLE support (<ColumnDef>, <ColumnConstraint>)
class ColumnDef:
    """
    <ColumnDef> ::= ID <TypeName> [ LPAREN NUM RPAREN ] { <ColumnConstraint> }
    """

    def __init__(
        self,
        name: str,
        data_type: DataType,
        size: Optional[int] = None,
        is_primary_key: bool = False,
        nullable: bool = True,
        is_unique: bool = False,
    ):
        self.name = name
        self.data_type = data_type
        self.size = size
        self.is_primary_key = is_primary_key
        self.nullable = nullable
        self.is_unique = is_unique

    def __repr__(self):
        flags = []
        if self.is_primary_key:
            flags.append("PRIMARY KEY")
        if self.is_unique and not self.is_primary_key:
            flags.append("UNIQUE")
        if not self.nullable and not self.is_primary_key:
            flags.append("NOT NULL")
        flag_str = f" {' '.join(flags)}" if flags else ""
        size_str = f"({self.size})" if self.size is not None else ""
        return f"{self.name} {self.data_type.value}{size_str}{flag_str}"


# Statements (<SelectStmt> | <InsertStmt> | <DeleteStmt> | <CreateTableStmt> | <CreateIndexStmt>)
class Stm(ABC):
    """Abstract base node for every SQL statement."""

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


class CreateTableStm(Stm):
    """<CreateTableStmt> ::= CREATE_TABLE ID LPAREN <ColumnDefList> RPAREN"""

    def __init__(self, table: str, columns: List[ColumnDef]):
        self.table = table
        self.columns = columns

    def accept(self, visitor: Visitor):
        return visitor.visit_create_table_stm(self)

    def __repr__(self):
        cols = ", ".join(repr(c) for c in self.columns)
        return f"CREATE TABLE {self.table} ({cols})"


class CreateIndexStm(Stm):
    """<CreateIndexStmt> ::= CREATE_INDEX ID ON ID LPAREN ID RPAREN USING <IndexType>"""

    def __init__(self, index_name: str, table: str, column: str, index_type: IndexType):
        self.index_name = index_name
        self.table = table
        self.column = column
        self.index_type = index_type

    def accept(self, visitor: Visitor):
        return visitor.visit_create_index_stm(self)

    def __repr__(self):
        return (
            f"CREATE INDEX {self.index_name} ON {self.table} "
            f"({self.column}) USING {self.index_type.name}"
        )