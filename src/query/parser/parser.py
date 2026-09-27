from typing import List, Optional

from query.parser.token_ import Token, TokenType
from query.parser.scanner import Scanner
from common import DataType
from query.parser.ast_nodes import (
    Exp,
    NumExp,
    IdExp,
    StringExp,
    PointExp,
    PointLiteral,
    PolygonLiteral,
    DistanceExp,
    WithinExp,
    SpatialPredicate,
    LimitClause,
    NullExp,
    BinaryExp,
    BinaryOp,
    Stm,
    SelectStm,
    ExplainStm,
    InsertStm,
    DeleteStm,
    OrderByClause,
    GroupByClause,
    JoinClause,
    BeginTransactionStm,
    EndTransactionStm,
    UpdateStm,
    AggregateSpec,
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
    TokenType.T_POINT: DataType.POINT,
}

_INDEX_TYPE_MAP = {
    TokenType.BTREE: IndexType.BTREE,
    TokenType.HASH: IndexType.HASH,
    TokenType.RTREE: IndexType.RTREE,
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

    def parse_sql_statements(self) -> list[Stm]:
        """Parse all semicolon-terminated statements until end of input."""
        statements = []
        while not self.is_at_end():
            statements.append(self.parse_sql_statement())
        return statements

    def parse_sql_statement(self) -> Stm:
        """<Statement> ::= [ EXPLAIN [ ANALYZE ] ] ( <SelectStmt> | <InsertStmt>
                            | <DeleteStmt> | <CreateTableStmt> | <CreateIndexStmt> ) SEMICOL

        EXPLAIN / EXPLAIN ANALYZE solo envuelven un <SelectStmt> por ahora
        (ver ExplainStm en ast_nodes.py)."""
        if self.check(TokenType.EXPLAIN):
            stm: Stm = self.parse_explain()
        elif self.check(TokenType.SELECT):
            stm = self.parse_select()
        elif self.check(TokenType.INSERT_INTO):
            stm = self.parse_insert()
        elif self.check(TokenType.DELETE):
            stm = self.parse_delete()
        elif self.check(TokenType.UPDATE):
            stm = self.parse_update()
        elif self.check(TokenType.BEGIN_TRANSACTION):
            self.advance()
            stm = BeginTransactionStm()
        elif self.check(TokenType.END_TRANSACTION):
            self.advance()
            stm = EndTransactionStm()
        elif self.check(TokenType.CREATE_TABLE):
            stm = self.parse_create_table()
        elif self.check(TokenType.CREATE_INDEX):
            stm = self.parse_create_index()
        else:
            self.error(
                "'EXPLAIN', 'SELECT', 'INSERT INTO', 'DELETE', 'CREATE TABLE' o 'CREATE INDEX'"
            )

        self.expect(TokenType.SEMICOL)
        return stm

    def parse_explain(self) -> ExplainStm:
        """<Statement> ::= EXPLAIN [ ANALYZE ] <SelectStmt>"""
        self.expect(TokenType.EXPLAIN)
        analyze = self.match(TokenType.ANALYZE)
        if not self.check(TokenType.SELECT):
            self.error("'SELECT' (EXPLAIN solo soporta SELECT por ahora)")
        inner = self.parse_select()
        return ExplainStm(inner, analyze)

    def parse_select(self) -> SelectStm:
        """<SelectStmt> ::= SELECT <SelectList> FROM ID [ <WhereClause> ] [ <GroupOrOrder> ]"""
        self.expect(TokenType.SELECT)
        columns = self.parse_select_list()
        self.expect(TokenType.FROM)
        table_tok = self.expect(TokenType.ID)
        table = table_tok.text

        join = None
        if self.match(TokenType.JOIN):
            join_table = self.expect(TokenType.ID).text
            self.expect(TokenType.ON)
            join_left = self.parse_qualified_name()
            self.expect(TokenType.EQ)
            join_right = self.parse_qualified_name()
            join = JoinClause(join_table, join_left, join_right)

        where_cond = None
        if self.check(TokenType.WHERE):
            where_cond = self.parse_where_clause()

        order_by, group_by = self.parse_group_or_order()
        limit = self.parse_limit_clause() if self.check(TokenType.LIMIT) else None

        return SelectStm(columns, table, where_cond, order_by, group_by, join, limit)

    def parse_select_list(self) -> List[str]:
        """<SelectList> ::= MUL | ID { COMA ID }"""
        if self.match(TokenType.MUL):
            return ["*"]

        columns = [self.parse_select_item()]
        while self.match(TokenType.COMA):
            columns.append(self.parse_select_item())
        return columns

    def parse_select_item(self):
        name = self.expect(TokenType.ID).text
        if not self.match(TokenType.LPAREN):
            if self.match(TokenType.DOT):
                name += "." + self.expect(TokenType.ID).text
            return name
        function = name.upper()
        if function not in {"COUNT", "SUM", "AVG", "MIN", "MAX"}:
            self.error("una función agregada válida")
        if self.match(TokenType.MUL):
            column = "*"
        else:
            column = self.parse_qualified_name()
        self.expect(TokenType.RPAREN)
        return AggregateSpec(function, column)

    def parse_qualified_name(self) -> str:
        name = self.expect(TokenType.ID).text
        if self.match(TokenType.DOT):
            name += "." + self.expect(TokenType.ID).text
        return name

    def parse_where_clause(self) -> Exp:
        """<WhereClause> ::= WHERE <Condition>"""
        self.expect(TokenType.WHERE)
        return self.parse_condition()

    def parse_condition(self) -> Exp:
        """<Condition> ::= distancia(...) <Operator> <Value> | dentro_de(...) | QualifiedName <Operator> <Value>"""
        if self.check(TokenType.WITHIN):
            return self.parse_within_exp()

        spatial_comparison = self.check(TokenType.DISTANCE)
        if spatial_comparison:
            left = self.parse_distance_exp()
        else:
            left = IdExp(self.parse_qualified_name())

        op = self.parse_operator()
        right = self.parse_value()

        if spatial_comparison:
            return SpatialPredicate(left, op, right)
        return BinaryExp(left, right, op)

    def parse_operator(self) -> BinaryOp:
        """<Operator> ::= EQ | NEQ | LE | LEQ | GT | GEQ"""
        for ttype, op in _OP_MAP.items():
            if self.match(ttype):
                return op
        self.error("un operador ('=', '!=', '<>', '<', '<=', '>' o '>=')")

    def parse_value(self) -> Exp:
        """<Value> ::= NUM | STRING | ID | NULL | <Point> | <Polygon>"""
        if self.check(TokenType.NUM):
            return NumExp(self.parse_number())
        if self.check(TokenType.STRING):
            tok = self.expect(TokenType.STRING)
            return StringExp(tok.text)
        if self.check(TokenType.T_POINT):
            return self.parse_point()
        if self.check(TokenType.POLYGON):
            return self.parse_polygon()
        if self.match(TokenType.NULL):
            return NullExp()
        if self.check(TokenType.ID):
            return IdExp(self.parse_qualified_name())
        self.error("un valor (NUM, STRING, ID, NULL, POINT o POLYGON)")

    def parse_number(self) -> int | float:
        token = self.expect(TokenType.NUM)
        return float(token.text) if "." in token.text else int(token.text)

    def parse_point(self) -> PointExp:
        self.expect(TokenType.T_POINT)
        self.expect(TokenType.LPAREN)
        longitude = self.parse_number()
        self.expect(TokenType.COMA)
        latitude = self.parse_number()
        self.expect(TokenType.RPAREN)
        return PointLiteral(longitude, latitude)

    def parse_distance_exp(self) -> DistanceExp:
        self.expect(TokenType.DISTANCE)
        self.expect(TokenType.LPAREN)
        geometry = IdExp(self.parse_qualified_name())
        self.expect(TokenType.COMA)
        point = self.parse_value()
        metric = "HAVERSINE"
        if self.match(TokenType.COMA):
            metric = self.expect(TokenType.ID).text.upper()
        self.expect(TokenType.RPAREN)
        return DistanceExp(geometry, point, metric)

    def parse_polygon(self) -> PolygonLiteral:
        self.expect(TokenType.POLYGON)
        self.expect(TokenType.LPAREN)
        points = [self.parse_point()]
        while self.match(TokenType.COMA):
            points.append(self.parse_point())
        self.expect(TokenType.RPAREN)
        return PolygonLiteral(points)

    def parse_within_exp(self) -> WithinExp:
        self.expect(TokenType.WITHIN)
        self.expect(TokenType.LPAREN)
        geometry = IdExp(self.parse_qualified_name())
        self.expect(TokenType.COMA)
        polygon = self.parse_polygon()
        self.expect(TokenType.RPAREN)
        return WithinExp(geometry, polygon)

    def parse_limit_clause(self) -> LimitClause:
        self.expect(TokenType.LIMIT)
        token = self.expect(TokenType.NUM)
        try:
            value = int(token.text)
        except ValueError:
            raise RuntimeError("Error sintáctico: LIMIT requiere un entero") from None
        return LimitClause(value)

    def parse_group_or_order(self):
        """<GroupOrOrder> ::= [ GROUP_BY QualifiedNameList ] [ ORDER_BY QualifiedNameList ]"""
        order_by: Optional[OrderByClause] = None
        group_by: Optional[GroupByClause] = None

        while self.check(TokenType.ORDER_BY) or self.check(TokenType.GROUP_BY):
            if self.match(TokenType.ORDER_BY):
                columns = [self.parse_order_by_item()]
                while self.match(TokenType.COMA):
                    columns.append(self.parse_order_by_item())
                order_by = OrderByClause(columns)
            else:
                self.expect(TokenType.GROUP_BY)
                columns = [self.parse_qualified_name()]
                while self.match(TokenType.COMA):
                    columns.append(self.parse_qualified_name())
                group_by = GroupByClause(columns)

        return order_by, group_by

    def parse_order_by_item(self):
        if self.check(TokenType.DISTANCE):
            return self.parse_distance_exp()
        return self.parse_qualified_name()

    def parse_insert(self) -> InsertStm:
        """<InsertStmt> ::= INSERT_INTO ID [ LPAREN <ColumnNameList> RPAREN ] VALUES <ValueRow> { COMA <ValueRow> }"""
        self.expect(TokenType.INSERT_INTO)
        table_tok = self.expect(TokenType.ID)

        columns = None
        if self.match(TokenType.LPAREN):
            columns = [self.expect(TokenType.ID).text]
            while self.match(TokenType.COMA):
                columns.append(self.expect(TokenType.ID).text)
            self.expect(TokenType.RPAREN)

        self.expect(TokenType.VALUES)
        rows = [self.parse_value_row()]
        while self.match(TokenType.COMA):
            rows.append(self.parse_value_row())
        return InsertStm(table_tok.text, rows, columns=columns)

    def parse_value_row(self) -> List[Exp]:
        """<ValueRow> ::= LPAREN <ValueList> RPAREN"""
        self.expect(TokenType.LPAREN)
        values = self.parse_value_list()
        self.expect(TokenType.RPAREN)
        return values

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

    def parse_update(self) -> UpdateStm:
        self.expect(TokenType.UPDATE)
        table = self.expect(TokenType.ID).text
        self.expect(TokenType.SET)
        column = self.expect(TokenType.ID).text
        self.expect(TokenType.EQ)
        value = self.parse_value()
        self.expect(TokenType.WHERE)
        return UpdateStm(table, column, value, self.parse_condition())

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
                         | CHAR | VARCHAR | TEXT | BOOLEAN | DATE | TIME | TIMESTAMP | BYTEA | POINT"""
        for ttype, data_type in _TYPE_MAP.items():
            if self.match(ttype):
                return data_type
        self.error("un nombre de tipo (SMALLINT, INTEGER, BIGINT, NUMERIC, REAL, "
                    "DOUBLE PRECISION, CHAR, VARCHAR, TEXT, BOOLEAN, DATE, TIME, "
                    "TIMESTAMP, BYTEA o POINT)")

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
        """<IndexType> ::= BTREE | HASH | RTREE"""
        for ttype, index_type in _INDEX_TYPE_MAP.items():
            if self.match(ttype):
                return index_type
        self.error("'BTREE', 'HASH' o 'RTREE'")