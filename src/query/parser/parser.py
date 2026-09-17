from typing import List, Optional

from query.parser.token_ import Token, TokenType
from query.parser.scanner import Scanner
from common import DataType
from query.parser.ast_nodes import (
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
    ColumnDef,
    CreateTableStm,
    CreateIndexStm,
    IndexType,
    StorageKind,
)


# Maps an operator token type to its corresponding BinaryOp
_OP_MAP = {
    TokenType.EQ: BinaryOp.EQ_OP,
    TokenType.NEQ: BinaryOp.NEQ_OP,
    TokenType.LE: BinaryOp.LE_OP,
    TokenType.LEQ: BinaryOp.LEQ_OP,
    TokenType.GT: BinaryOp.GT_OP,
    TokenType.GEQ: BinaryOp.GEQ_OP,
}

# Maps a type-name token to its DataType (common), and whether it
# REQUIRES a declared size, per common.value.VARIABLE_SIZE_TYPES.
_TYPE_MAP = {
    TokenType.T_SMALLINT: DataType.SMALLINT,
    TokenType.T_INTEGER: DataType.INTEGER,
    TokenType.T_BIGINT: DataType.BIGINT,
    TokenType.T_NUMERIC: DataType.NUMERIC,
    TokenType.T_REAL: DataType.REAL,
    TokenType.T_DOUBLE_PRECISION: DataType.DOUBLE_PRECISION,
    TokenType.T_CHAR: DataType.CHAR,
    TokenType.T_VARCHAR: DataType.VARCHAR,
    TokenType.T_TEXT: DataType.TEXT,
    TokenType.T_BOOLEAN: DataType.BOOLEAN,
    TokenType.T_DATE: DataType.DATE,
    TokenType.T_TIME: DataType.TIME,
    TokenType.T_TIMESTAMP: DataType.TIMESTAMP,
    TokenType.T_BYTEA: DataType.BYTEA,
}

_INDEX_TYPE_MAP = {
    TokenType.BTREE: IndexType.BTREE,
    TokenType.HASH: IndexType.HASH,
}

_STORAGE_TYPE_MAP = {
    TokenType.HEAP: StorageKind.HEAP,
    TokenType.SEQUENTIAL: StorageKind.SEQUENTIAL,
}


class Parser:
    def __init__(self, scanner: Scanner):
        self.scanner = scanner
        self.previous: Optional[Token] = None
        self.current: Token = self.scanner.next_token()
        if self.current.type == TokenType.ERR:
            raise RuntimeError(f"Error léxico: carácter no reconocido '{self.current.text}'")

    # =========================================================================
    # Token consumption primitives
    # =========================================================================

    def is_at_end(self) -> bool:
        return self.current.type == TokenType.END

    def check(self, ttype: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.current.type == ttype

    def advance(self) -> bool:
        if not self.is_at_end():
            self.previous = self.current
            self.current = self.scanner.next_token()
            if self.current.type == TokenType.ERR:
                raise RuntimeError(f"Error léxico: carácter no reconocido '{self.current.text}'")
            return True
        return False

    def match(self, ttype: TokenType) -> bool:
        if self.check(ttype):
            self.advance()
            return True
        return False

    # =========================================================================
    # Error reporting
    # =========================================================================

    def error(self, expected: str):
        if self.is_at_end():
            found = "fin de entrada"
        else:
            found = Token.type_name(self.current.type)
            if self.current.text:
                found += f" '{self.current.text}'"
        raise RuntimeError(f"Error sintáctico: se esperaba {expected}, pero se encontró {found}")

    def expect(self, ttype: TokenType) -> Token:
        """Consume and return the token if it matches; otherwise raise a descriptive error."""
        if self.check(ttype):
            tok = self.current
            self.advance()
            return tok
        self.error(Token.type_name(ttype))

    # =========================================================================
    # Grammar rules
    # =========================================================================

    def parse_sql_statement(self) -> Stm:
        """<Statement> ::= ( <SelectStmt> | <InsertStmt> | <DeleteStmt>
                            | <CreateTableStmt> | <CreateIndexStmt> ) SEMICOL"""
        if self.check(TokenType.SELECT):
            stm: Stm = self.parse_select()
        elif self.check(TokenType.INSERT_INTO):
            stm = self.parse_insert()
        elif self.check(TokenType.DELETE):
            stm = self.parse_delete()
        elif self.check(TokenType.CREATE_TABLE):
            stm = self.parse_create_table()
        elif self.check(TokenType.CREATE_INDEX):
            stm = self.parse_create_index()
        else:
            self.error(
                "'SELECT', 'INSERT INTO', 'DELETE', 'CREATE TABLE' o 'CREATE INDEX'"
            )

        self.expect(TokenType.SEMICOL)
        return stm

    def parse_select(self) -> SelectStm:
        """<SelectStmt> ::= SELECT <SelectList> FROM ID [ <WhereClause> ] [ <GroupOrOrder> ]"""
        self.expect(TokenType.SELECT)
        columns = self.parse_select_list()
        self.expect(TokenType.FROM)
        table_tok = self.expect(TokenType.ID)
        table = table_tok.text

        where_cond = None
        if self.check(TokenType.WHERE):
            where_cond = self.parse_where_clause()

        order_by: Optional[OrderByClause] = None
        group_by: Optional[GroupByClause] = None
        if self.check(TokenType.ORDER_BY) or self.check(TokenType.GROUP_BY):
            order_by, group_by = self.parse_group_or_order()

        return SelectStm(columns, table, where_cond, order_by, group_by)

    def parse_select_list(self) -> List[str]:
        """<SelectList> ::= MUL | ID { COMA ID }"""
        if self.match(TokenType.MUL):
            return ["*"]

        columns = [self.expect(TokenType.ID).text]
        while self.match(TokenType.COMA):
            columns.append(self.expect(TokenType.ID).text)
        return columns

    def parse_where_clause(self) -> Exp:
        """<WhereClause> ::= WHERE <Condition>"""
        self.expect(TokenType.WHERE)
        return self.parse_condition()

    def parse_condition(self) -> Exp:
        """<Condition> ::= ID <Operator> <Value>"""
        id_tok = self.expect(TokenType.ID)
        left = IdExp(id_tok.text)

        op = self.parse_operator()
        right = self.parse_value()

        return BinaryExp(left, right, op)

    def parse_operator(self) -> BinaryOp:
        """<Operator> ::= EQ | NEQ | LE | LEQ | GT | GEQ"""
        for ttype, op in _OP_MAP.items():
            if self.match(ttype):
                return op
        self.error("un operador ('=', '!=', '<>', '<', '<=', '>' o '>=')")

    def parse_value(self) -> Exp:
        """<Value> ::= NUM | STRING | ID"""
        if self.check(TokenType.NUM):
            tok = self.expect(TokenType.NUM)
            return NumExp(int(tok.text))
        if self.check(TokenType.STRING):
            tok = self.expect(TokenType.STRING)
            return StringExp(tok.text)
        if self.check(TokenType.ID):
            tok = self.expect(TokenType.ID)
            return IdExp(tok.text)
        self.error("un valor (NUM, STRING o ID)")

    def parse_group_or_order(self):
        """<GroupOrOrder> ::= ORDER_BY ID { COMA ID } | GROUP_BY ID { COMA ID }"""
        order_by: Optional[OrderByClause] = None
        group_by: Optional[GroupByClause] = None

        if self.match(TokenType.ORDER_BY):
            columns = [self.expect(TokenType.ID).text]
            while self.match(TokenType.COMA):
                columns.append(self.expect(TokenType.ID).text)
            order_by = OrderByClause(columns)
        elif self.match(TokenType.GROUP_BY):
            columns = [self.expect(TokenType.ID).text]
            while self.match(TokenType.COMA):
                columns.append(self.expect(TokenType.ID).text)
            group_by = GroupByClause(columns)
        else:
            self.error("'ORDER BY' o 'GROUP BY'")

        return order_by, group_by

    def parse_insert(self) -> InsertStm:
        """<InsertStmt> ::= INSERT_INTO ID VALUES LPAREN <ValueList> RPAREN"""
        self.expect(TokenType.INSERT_INTO)
        table_tok = self.expect(TokenType.ID)
        self.expect(TokenType.VALUES)
        self.expect(TokenType.LPAREN)
        values = self.parse_value_list()
        self.expect(TokenType.RPAREN)
        return InsertStm(table_tok.text, values)

    def parse_value_list(self) -> List[Exp]:
        """<ValueList> ::= <Value> { COMA <Value> }"""
        values = [self.parse_value()]
        while self.match(TokenType.COMA):
            values.append(self.parse_value())
        return values

    def parse_delete(self) -> DeleteStm:
        """<DeleteStmt> ::= DELETE FROM ID [ <WhereClause> ]"""
        self.expect(TokenType.DELETE)
        self.expect(TokenType.FROM)
        table_tok = self.expect(TokenType.ID)

        where_cond = None
        if self.check(TokenType.WHERE):
            where_cond = self.parse_where_clause()

        return DeleteStm(table_tok.text, where_cond)

    # =========================================================================
    # CREATE TABLE
    # =========================================================================

    def parse_create_table(self) -> CreateTableStm:
        """<CreateTableStmt> ::= CREATE_TABLE ID LPAREN <ColumnDefList> RPAREN [ USING <StorageType> ]"""
        self.expect(TokenType.CREATE_TABLE)
        table_tok = self.expect(TokenType.ID)
        self.expect(TokenType.LPAREN)
        columns = self.parse_column_def_list()
        self.expect(TokenType.RPAREN)

        storage_kind = StorageKind.HEAP  # default, per spec
        if self.match(TokenType.USING):
            storage_kind = self.parse_storage_type()

        return CreateTableStm(table_tok.text, columns, storage_kind)

    def parse_storage_type(self) -> StorageKind:
        """<StorageType> ::= HEAP | SEQUENTIAL"""
        for ttype, storage_kind in _STORAGE_TYPE_MAP.items():
            if self.match(ttype):
                return storage_kind
        self.error("'HEAP' o 'SEQUENTIAL'")

    def parse_column_def_list(self) -> List[ColumnDef]:
        """<ColumnDefList> ::= <ColumnDef> { COMA <ColumnDef> }"""
        columns = [self.parse_column_def()]
        while self.match(TokenType.COMA):
            columns.append(self.parse_column_def())
        return columns

    def parse_column_def(self) -> ColumnDef:
        """<ColumnDef> ::= ID <TypeName> [ LPAREN NUM RPAREN ] { <ColumnConstraint> }"""
        name_tok = self.expect(TokenType.ID)
        data_type = self.parse_type_name()

        size = None
        if self.match(TokenType.LPAREN):
            size_tok = self.expect(TokenType.NUM)
            size = int(size_tok.text)
            self.expect(TokenType.RPAREN)

        is_primary_key = False
        nullable = True
        is_unique = False

        while self.check(TokenType.PRIMARY_KEY) or self.check(TokenType.NOT_NULL) or self.check(TokenType.UNIQUE):
            if self.match(TokenType.PRIMARY_KEY):
                is_primary_key = True
            elif self.match(TokenType.NOT_NULL):
                nullable = False
            elif self.match(TokenType.UNIQUE):
                is_unique = True

        return ColumnDef(
            name=name_tok.text,
            data_type=data_type,
            size=size,
            is_primary_key=is_primary_key,
            nullable=nullable,
            is_unique=is_unique,
        )

    def parse_type_name(self) -> DataType:
        """<TypeName> ::= SMALLINT | INTEGER | BIGINT | NUMERIC | REAL | DOUBLE_PRECISION
                         | CHAR | VARCHAR | TEXT | BOOLEAN | DATE | TIME | TIMESTAMP | BYTEA"""
        for ttype, data_type in _TYPE_MAP.items():
            if self.match(ttype):
                return data_type
        self.error("un nombre de tipo (SMALLINT, INTEGER, BIGINT, NUMERIC, REAL, "
                    "DOUBLE PRECISION, CHAR, VARCHAR, TEXT, BOOLEAN, DATE, TIME, "
                    "TIMESTAMP o BYTEA)")

    # =========================================================================
    # CREATE INDEX
    # =========================================================================

    def parse_create_index(self) -> CreateIndexStm:
        """<CreateIndexStmt> ::= CREATE_INDEX ID ON ID LPAREN ID RPAREN USING <IndexType>"""
        self.expect(TokenType.CREATE_INDEX)
        index_name_tok = self.expect(TokenType.ID)
        self.expect(TokenType.ON)
        table_tok = self.expect(TokenType.ID)
        self.expect(TokenType.LPAREN)
        column_tok = self.expect(TokenType.ID)
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.USING)
        index_type = self.parse_index_type()

        return CreateIndexStm(index_name_tok.text, table_tok.text, column_tok.text, index_type)

    def parse_index_type(self) -> IndexType:
        """<IndexType> ::= BTREE | HASH"""
        for ttype, index_type in _INDEX_TYPE_MAP.items():
            if self.match(ttype):
                return index_type
        self.error("'BTREE' o 'HASH'")