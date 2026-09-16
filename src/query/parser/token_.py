from enum import Enum, auto


class TokenType(Enum):
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    ORDER_BY = auto()
    GROUP_BY = auto()
    DELETE = auto()
    INSERT_INTO = auto()
    VALUES = auto()
    END = auto()
    ERR = auto()
    ID = auto()
    NUM = auto()
    STRING = auto()
    MUL = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMA = auto()
    SEMICOL = auto()
    EQ = auto()
    LE = auto()
    LEQ = auto()
    GT = auto()
    GEQ = auto()


# Nombres legibles para cada tipo de token (usados en mensajes de error del parser)
_TYPE_NAMES = {
    TokenType.SELECT: "'SELECT'",
    TokenType.FROM: "'FROM'",
    TokenType.WHERE: "'WHERE'",
    TokenType.ORDER_BY: "'ORDER BY'",
    TokenType.GROUP_BY: "'GROUP BY'",
    TokenType.DELETE: "'DELETE'",
    TokenType.INSERT_INTO: "'INSERT INTO'",
    TokenType.VALUES: "'VALUES'",
    TokenType.END: "'END'",
    TokenType.ERR: "'ERR'",
    TokenType.ID: "'ID'",
    TokenType.NUM: "'NUM'",
    TokenType.STRING: "'STRING'",
    TokenType.MUL: "'*'",
    TokenType.LPAREN: "'('",
    TokenType.RPAREN: "')'",
    TokenType.COMA: "','",
    TokenType.SEMICOL: "';'",
    TokenType.EQ: "'='",
    TokenType.LE: "'<'",
    TokenType.LEQ: "'<='",
    TokenType.GT: "'>'",
    TokenType.GEQ: "'>='",
}


class Token:

    def __init__(self, type_: TokenType, text: str = ""):
        self.type = type_
        self.text = text

    @staticmethod
    def type_name(t: TokenType) -> str:
        return _TYPE_NAMES.get(t, str(t))

    def __repr__(self) -> str:
        return f'TOKEN({self.type.name}, "{self.text}")'

    def __str__(self) -> str:
        return self.__repr__()