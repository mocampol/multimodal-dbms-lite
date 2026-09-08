import os
from token import Token, TokenType


def _is_white_space(c: str) -> bool:
    return c in (" ", "\n", "\r", "\t")


class Scanner:

    def __init__(self, source: str):
        self.input = source
        self.first = 0
        self.current = 0

    def next_token(self) -> Token:
        # Saltar espacios en blanco
        while self.current < len(self.input) and _is_white_space(self.input[self.current]):
            self.current += 1

        # Fin de la entrada
        if self.current >= len(self.input):
            return Token(TokenType.END)

        c = self.input[self.current]
        self.first = self.current

        # ---- Números ----
        if c.isdigit():
            self.current += 1
            while self.current < len(self.input) and self.input[self.current].isdigit():
                self.current += 1
            lexema = self.input[self.first:self.current]
            return Token(TokenType.NUM, lexema)

        # ---- Identificadores y palabras reservadas SQL ----
        if c.isalpha() or c == "_":
            self.current += 1
            while self.current < len(self.input) and (
                self.input[self.current].isalnum() or self.input[self.current] == "_"
            ):
                self.current += 1

            lexema = self.input[self.first:self.current]
            upper_lexema = lexema.upper()

            # Tokens de una sola palabra
            single_word = {
                "SELECT": TokenType.SELECT,
                "FROM": TokenType.FROM,
                "WHERE": TokenType.WHERE,
                "DELETE": TokenType.DELETE,
                "VALUES": TokenType.VALUES,
            }
            if upper_lexema in single_word:
                return Token(single_word[upper_lexema], lexema)

            # Tokens compuestos por dos palabras
            if upper_lexema in ("ORDER", "GROUP", "INSERT"):
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

            return Token(TokenType.ID, lexema)

        # ---- Operadores y delimitadores SQL ----
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
            if c == "<":
                if self.current + 1 < len(self.input) and self.input[self.current + 1] == "=":
                    lexema = self.input[self.current:self.current + 2]
                    self.current += 2
                    return Token(TokenType.LEQ, lexema)
                else:
                    self.current += 1
                    return Token(TokenType.LE, c)
            if c == ">":
                if self.current + 1 < len(self.input) and self.input[self.current + 1] == "=":
                    lexema = self.input[self.current:self.current + 2]
                    self.current += 2
                    return Token(TokenType.GEQ, lexema)
                else:
                    self.current += 1
                    return Token(TokenType.GT, c)

        # ---- Carácter no reconocido (Error léxico) ----
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