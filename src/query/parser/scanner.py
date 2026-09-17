import os
from query.parser.token_ import Token, TokenType


def _is_white_space(c: str) -> bool:
    return c in (" ", "\n", "\r", "\t")


class Scanner:
    def __init__(self, source: str):
        self.input = source
        self.first = 0
        self.current = 0

    def next_token(self) -> Token:
        while self.current < len(self.input) and _is_white_space(self.input[self.current]):
            self.current += 1
        if self.current >= len(self.input):
            return Token(TokenType.END)

        c = self.input[self.current]
        self.first = self.current
        if c.isdigit():
            self.current += 1
            while self.current < len(self.input) and self.input[self.current].isdigit():
                self.current += 1
            return Token(TokenType.NUM, self.input[self.first:self.current])

        if c.isalpha() or c == "_":
            self.current += 1
            while self.current < len(self.input) and (self.input[self.current].isalnum() or self.input[self.current] == "_"):
                self.current += 1
            lexeme = self.input[self.first:self.current]
            upper = lexeme.upper()
            words = {
                "SELECT": TokenType.SELECT, "FROM": TokenType.FROM, "JOIN": TokenType.JOIN,
                "WHERE": TokenType.WHERE, "DELETE": TokenType.DELETE, "VALUES": TokenType.VALUES,
                "ON": TokenType.ON, "USING": TokenType.USING, "UNIQUE": TokenType.UNIQUE,
                "SMALLINT": TokenType.T_SMALLINT, "INTEGER": TokenType.T_INTEGER,
                "BIGINT": TokenType.T_BIGINT, "NUMERIC": TokenType.T_NUMERIC,
                "REAL": TokenType.T_REAL, "CHAR": TokenType.T_CHAR, "VARCHAR": TokenType.T_VARCHAR,
                "TEXT": TokenType.T_TEXT, "BOOLEAN": TokenType.T_BOOLEAN, "DATE": TokenType.T_DATE,
                "TIME": TokenType.T_TIME, "TIMESTAMP": TokenType.T_TIMESTAMP, "BYTEA": TokenType.T_BYTEA,
                "BTREE": TokenType.BTREE, "HASH": TokenType.HASH, "HEAP": TokenType.HEAP,
                "SEQUENTIAL": TokenType.SEQUENTIAL,
            }
            if upper in words:
                return Token(words[upper], lexeme)
            compounds = {
                "ORDER": ("BY", TokenType.ORDER_BY), "GROUP": ("BY", TokenType.GROUP_BY),
                "INSERT": ("INTO", TokenType.INSERT_INTO), "CREATE": ("TABLE", TokenType.CREATE_TABLE),
                "PRIMARY": ("KEY", TokenType.PRIMARY_KEY), "NOT": ("NULL", TokenType.NOT_NULL),
                "DOUBLE": ("PRECISION", TokenType.T_DOUBLE_PRECISION),
            }
            if upper in compounds:
                expected, token_type = compounds[upper]
                probe = self.current
                while probe < len(self.input) and _is_white_space(self.input[probe]):
                    probe += 1
                end = probe
                while end < len(self.input) and self.input[end].isalpha():
                    end += 1
                if self.input[probe:end].upper() == expected:
                    self.current = end
                    return Token(token_type, self.input[self.first:self.current])
            if upper == "CREATE":
                probe = self.current
                while probe < len(self.input) and _is_white_space(self.input[probe]):
                    probe += 1
                end = probe
                while end < len(self.input) and self.input[end].isalpha():
                    end += 1
                if self.input[probe:end].upper() == "INDEX":
                    self.current = end
                    return Token(TokenType.CREATE_INDEX, self.input[self.first:self.current])
            return Token(TokenType.ID, lexeme)

        if c == "'":
            self.current += 1
            chars = []
            while self.current < len(self.input):
                char = self.input[self.current]
                if char == "'":
                    if self.current + 1 < len(self.input) and self.input[self.current + 1] == "'":
                        chars.append("'")
                        self.current += 2
                        continue
                    self.current += 1
                    return Token(TokenType.STRING, "".join(chars))
                chars.append(char)
                self.current += 1
            return Token(TokenType.ERR, self.input[self.first:self.current])

        self.current += 1
        simple = {"*": TokenType.MUL, "(": TokenType.LPAREN, ")": TokenType.RPAREN, ",": TokenType.COMA, ".": TokenType.DOT, ";": TokenType.SEMICOL, "=": TokenType.EQ}
        if c in simple:
            return Token(simple[c], c)
        if c == "!" and self.current < len(self.input) and self.input[self.current] == "=":
            self.current += 1
            return Token(TokenType.NEQ, "!=")
        if c == "<":
            if self.current < len(self.input) and self.input[self.current] in "=>":
                suffix = self.input[self.current]
                self.current += 1
                return Token(TokenType.LEQ if suffix == "=" else TokenType.NEQ, "<" + suffix)
            return Token(TokenType.LE, c)
        if c == ">":
            if self.current < len(self.input) and self.input[self.current] == "=":
                self.current += 1
                return Token(TokenType.GEQ, ">=")
            return Token(TokenType.GT, c)
        return Token(TokenType.ERR, c)


def ejecutar_scanner(scanner: Scanner, input_file: str) -> int:
    output_name, _ext = os.path.splitext(input_file)
    output_name += "_tokens.txt"
    with open(output_name, "w", encoding="utf-8") as out_file:
        out_file.write("Scanner\n\n")
        while True:
            token = scanner.next_token()
            out_file.write(f"{token}\n")
            if token.type == TokenType.ERR:
                out_file.write(f"Error léxico: carácter inválido '{token.text}'\n")
                return 1
            if token.type == TokenType.END:
                out_file.write("\nScanner exitoso\n")
                return 0
