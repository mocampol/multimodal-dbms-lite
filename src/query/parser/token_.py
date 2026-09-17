from enum import Enum, auto


class TokenType(Enum):
    SELECT = auto()
    FROM = auto()
    JOIN = auto()
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
    DOT = auto()
    SEMICOL = auto()
    EQ = auto()
    NEQ = auto()
    LE = auto()
    LEQ = auto()
    GT = auto()
    GEQ = auto()

    CREATE_TABLE = auto()
    CREATE_INDEX = auto()
    ON = auto()
    USING = auto()

    PRIMARY_KEY = auto()
    NOT_NULL = auto()
    UNIQUE = auto()

    T_SMALLINT = auto()
    T_INTEGER = auto()
    T_BIGINT = auto()
    T_NUMERIC = auto()
    T_REAL = auto()
    T_DOUBLE_PRECISION = auto()
    T_CHAR = auto()
    T_VARCHAR = auto()
    T_TEXT = auto()
    T_BOOLEAN = auto()
    T_DATE = auto()
    T_TIME = auto()
    T_TIMESTAMP = auto()
    T_BYTEA = auto()

    BTREE = auto()
    HASH = auto()
    HEAP = auto()
    SEQUENTIAL = auto()


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
    TokenType.DOT: "'.'",
    TokenType.SEMICOL: "';'",
    TokenType.EQ: "'='",
    TokenType.NEQ: "'!=' o '<>'",
    TokenType.LE: "'<'",
    TokenType.LEQ: "'<='",
    TokenType.GT: "'>'",
    TokenType.GEQ: "'>='",

    TokenType.CREATE_TABLE: "'CREATE TABLE'",
    TokenType.CREATE_INDEX: "'CREATE INDEX'",
    TokenType.ON: "'ON'",
    TokenType.USING: "'USING'",

    TokenType.PRIMARY_KEY: "'PRIMARY KEY'",
    TokenType.NOT_NULL: "'NOT NULL'",
    TokenType.UNIQUE: "'UNIQUE'",

    TokenType.T_SMALLINT: "'SMALLINT'",
    TokenType.T_INTEGER: "'INTEGER'",
    TokenType.T_BIGINT: "'BIGINT'",
    TokenType.T_NUMERIC: "'NUMERIC'",
    TokenType.T_REAL: "'REAL'",
    TokenType.T_DOUBLE_PRECISION: "'DOUBLE PRECISION'",
    TokenType.T_CHAR: "'CHAR'",
    TokenType.T_VARCHAR: "'VARCHAR'",
    TokenType.T_TEXT: "'TEXT'",
    TokenType.T_BOOLEAN: "'BOOLEAN'",
    TokenType.T_DATE: "'DATE'",
    TokenType.T_TIME: "'TIME'",
    TokenType.T_TIMESTAMP: "'TIMESTAMP'",
    TokenType.T_BYTEA: "'BYTEA'",

    TokenType.BTREE: "'BTREE'",
    TokenType.HASH: "'HASH'",
    TokenType.HEAP: "'HEAP'",
    TokenType.SEQUENTIAL: "'SEQUENTIAL'",
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