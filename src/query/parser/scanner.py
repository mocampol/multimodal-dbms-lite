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
        # Skip whitespace
        while self.current < len(self.input) and _is_white_space(self.input[self.current]):
            self.current += 1

        # End of input
        if self.current >= len(self.input):
            return Token(TokenType.END)

        c = self.input[self.current]
        self.first = self.current

        # Numbers
        if c.isdigit():
            self.current += 1
            while self.current < len(self.input) and self.input[self.current].isdigit():
                self.current += 1
            lexema = self.input[self.first:self.current]
            return Token(TokenType.NUM, lexema)

        # Identifiers and SQL reserved words
        if c.isalpha() or c == "_":
            self.current += 1
            while self.current < len(self.input) and (
                self.input[self.current].isalnum() or self.input[self.current] == "_"
            ):
                self.current += 1

            lexema = self.input[self.first:self.current]
            upper_lexema = lexema.upper()

            single_word = {
                "SELECT": TokenType.SELECT,
                "FROM": TokenType.FROM,
                "WHERE": TokenType.WHERE,
                "DELETE": TokenType.DELETE,
                "VALUES": TokenType.VALUES,

                "ON": TokenType.ON,
                "USING": TokenType.USING,

                "UNIQUE": TokenType.UNIQUE,

                "SMALLINT": TokenType.T_SMALLINT,
                "INTEGER": TokenType.T_INTEGER,
                "BIGINT": TokenType.T_BIGINT,
                "NUMERIC": TokenType.T_NUMERIC,
                "REAL": TokenType.T_REAL,
                "CHAR": TokenType.T_CHAR,
                "VARCHAR": TokenType.T_VARCHAR,
                "TEXT": TokenType.T_TEXT,
                "BOOLEAN": TokenType.T_BOOLEAN,
                "DATE": TokenType.T_DATE,
                "TIME": TokenType.T_TIME,
                "TIMESTAMP": TokenType.T_TIMESTAMP,
                "BYTEA": TokenType.T_BYTEA,

                "BTREE": TokenType.BTREE,
                "HASH": TokenType.HASH,
                "HEAP": TokenType.HEAP,
                "SEQUENTIAL": TokenType.SEQUENTIAL,
            }
            if upper_lexema in single_word:
                return Token(single_word[upper_lexema], lexema)

            if upper_lexema in ("ORDER", "GROUP", "INSERT", "CREATE", "PRIMARY", "NOT", "DOUBLE"):
                temp_current = self.current

                while temp_current < len(self.input) and _is_white_space(self.input[temp_current]):
                    temp_current += 1

                if temp_current < len(self.input) and self.input[temp_current].isalpha():
                    second_first = temp_current
                    while temp_current < len(self.input) and self.input[temp_current].isalpha():
                        temp_current += 1

                    second_lexema = self.input[second_first:temp_current]
                    upper_second = second_lexema.upper()

                    if upper_lexema == "ORDER" and upper_second == "BY":
                        self.current = temp_current
                        return Token(TokenType.ORDER_BY, self.input[self.first:self.current])
                    if upper_lexema == "GROUP" and upper_second == "BY":
                        self.current = temp_current
                        return Token(TokenType.GROUP_BY, self.input[self.first:self.current])
                    if upper_lexema == "INSERT" and upper_second == "INTO":
                        self.current = temp_current
                        return Token(TokenType.INSERT_INTO, self.input[self.first:self.current])
                    if upper_lexema == "CREATE" and upper_second == "TABLE":
                        self.current = temp_current
                        return Token(TokenType.CREATE_TABLE, self.input[self.first:self.current])
                    if upper_lexema == "CREATE" and upper_second == "INDEX":
                        self.current = temp_current
                        return Token(TokenType.CREATE_INDEX, self.input[self.first:self.current])
                    if upper_lexema == "PRIMARY" and upper_second == "KEY":
                        self.current = temp_current
                        return Token(TokenType.PRIMARY_KEY, self.input[self.first:self.current])
                    if upper_lexema == "NOT" and upper_second == "NULL":
                        self.current = temp_current
                        return Token(TokenType.NOT_NULL, self.input[self.first:self.current])
                    if upper_lexema == "DOUBLE" and upper_second == "PRECISION":
                        self.current = temp_current
                        return Token(TokenType.T_DOUBLE_PRECISION, self.input[self.first:self.current])

            return Token(TokenType.ID, lexema)

        if c == "'":
            self.current += 1
            chars = []
            while True:
                if self.current >= len(self.input):
                    err = Token(TokenType.ERR, self.input[self.first:self.current])
                    return err

                ch = self.input[self.current]

                if ch == "'":
                    if self.current + 1 < len(self.input) and self.input[self.current + 1] == "'":
                        chars.append("'")
                        self.current += 2
                        continue
                    self.current += 1
                    break

                chars.append(ch)
                self.current += 1

            return Token(TokenType.STRING, "".join(chars))

        # SQL operators and delimiters
        if c in "*()=<>!;,":
            if c == "*":
                self.current += 1
                return Token(TokenType.MUL, c)
            if c == "(":
                self.current += 1
                return Token(TokenType.LPAREN, c)
            if c == ")":
                self.current += 1
                return Token(TokenType.RPAREN, c)
            if c == ",":
                self.current += 1
                return Token(TokenType.COMA, c)
            if c == ";":
                self.current += 1
                return Token(TokenType.SEMICOL, c)
            if c == "=":
                self.current += 1
                return Token(TokenType.EQ, c)
            if c == "!":
                if self.current + 1 < len(self.input) and self.input[self.current + 1] == "=":
                    lexema = self.input[self.current:self.current + 2]
                    self.current += 2
                    return Token(TokenType.NEQ, lexema)
                err = Token(TokenType.ERR, c)
                self.current += 1
                return err
            if c == "<":
                if self.current + 1 < len(self.input) and self.input[self.current + 1] == ">":
                    lexema = self.input[self.current:self.current + 2]
                    self.current += 2
                    return Token(TokenType.NEQ, lexema)
                if self.current + 1 < len(self.input) and self.input[self.current + 1] == "=":
                    lexema = self.input[self.current:self.current + 2]
                    self.current += 2
                    return Token(TokenType.LEQ, lexema)
                self.current += 1
                return Token(TokenType.LE, c)
            if c == ">":
                if self.current + 1 < len(self.input) and self.input[self.current + 1] == "=":
                    lexema = self.input[self.current:self.current + 2]
                    self.current += 2
                    return Token(TokenType.GEQ, lexema)
                self.current += 1
                return Token(TokenType.GT, c)

        # Lexical error
        err = Token(TokenType.ERR, c)
        self.current += 1
        return err


def ejecutar_scanner(scanner: Scanner, input_file: str) -> int:
    output_name, _ext = os.path.splitext(input_file)
    output_name += "_tokens.txt"

    with open(output_name, "w", encoding="utf-8") as out_file:
        out_file.write("Scanner\n\n")

        while True:
            tok = scanner.next_token()

            if tok.type == TokenType.ERR:
                out_file.write(f"{tok}\n")
                out_file.write(f"Error léxico: carácter inválido '{tok.text}'\n")
                return 1

            out_file.write(f"{tok}\n")

            if tok.type == TokenType.END:
                out_file.write("\nScanner exitoso\n")
                return 0