from typing import List, Optional

from token_ import Token, TokenType
from scanner import Scanner
from ast_nodes import (
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
)


# Mapea el tipo de token de operador a su BinaryOp correspondiente
_OP_MAP = {
    TokenType.EQ: BinaryOp.EQ_OP,
    TokenType.LE: BinaryOp.LE_OP,
    TokenType.LEQ: BinaryOp.LEQ_OP,
    TokenType.GT: BinaryOp.GT_OP,
    TokenType.GEQ: BinaryOp.GEQ_OP,
}


class Parser:
    def __init__(self, scanner: Scanner):
        self.scanner = scanner
        self.previous: Optional[Token] = None
        self.current: Token = self.scanner.next_token()
        if self.current.type == TokenType.ERR:
            raise RuntimeError(f"Error léxico: carácter no reconocido '{self.current.text}'")

    # =========================================================================
    # Primitivas de consumo de tokens
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
    # Reporte de errores
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
        """Consume el token si coincide y lo retorna; si no, lanza error descriptivo."""
        if self.check(ttype):
            tok = self.current
            self.advance()
            return tok
        self.error(Token.type_name(ttype))

    # =========================================================================
    # Reglas gramaticales
    # =========================================================================

    def parse_sql_statement(self) -> Stm:
        """<Statement> ::= ( <SelectStmt> | <InsertStmt> | <DeleteStmt> ) SEMICOL"""
        if self.check(TokenType.SELECT):
            stm: Stm = self.parse_select()
        elif self.check(TokenType.INSERT_INTO):
            stm = self.parse_insert()
        elif self.check(TokenType.DELETE):
            stm = self.parse_delete()
        else:
            self.error("'SELECT', 'INSERT INTO' o 'DELETE'")

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
        """<Operator> ::= EQ | LE | LEQ | GT | GEQ"""
        for ttype, op in _OP_MAP.items():
            if self.match(ttype):
                return op
        self.error("un operador ('=', '<', '<=', '>' o '>=')")

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