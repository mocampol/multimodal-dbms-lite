from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation

from common.value import DataType, Value, VARIABLE_SIZE_TYPES, UNBOUNDED_TYPES

from .exceptions import UnsupportedTypeError


_BOOLEAN_TRUE_WORDS = {"true", "t", "yes", "y"}
_BOOLEAN_FALSE_WORDS = {"false", "f", "no", "n"}

_BOOLEAN_TRUE_LITERALS = _BOOLEAN_TRUE_WORDS | {"1"}
_BOOLEAN_FALSE_LITERALS = _BOOLEAN_FALSE_WORDS | {"0"}

_SIZE_STEPS = (16, 32, 64, 128, 256, 512)



def infer_column_type(raw_values: list[str]) -> tuple[DataType, int | None]:
    """
    Given every raw string seen for one CSV column (empty strings
    representing NULL cells already filtered out by the caller, or
    included and ignored here), returns the narrowest of
    {INTEGER, DOUBLE_PRECISION, BOOLEAN, VARCHAR} that fits every
    non-empty value, plus a `size` (only meaningful for VARCHAR).

    An all-empty column (every cell blank) defaults to VARCHAR(32),
    since there is no evidence to infer a narrower type from.
    """
    non_empty = [v for v in raw_values if v != ""]
    if not non_empty:
        return DataType.VARCHAR, 32

    if all(_looks_like_int(v) for v in non_empty):
        return DataType.INTEGER, None

    if all(_looks_like_float(v) for v in non_empty):
        return DataType.DOUBLE_PRECISION, None

    if all(v.strip().lower() in _BOOLEAN_TRUE_WORDS | _BOOLEAN_FALSE_WORDS for v in non_empty):
        return DataType.BOOLEAN, None

    max_len = max(len(v) for v in non_empty)
    return DataType.VARCHAR, _round_up_size(max_len)


def _looks_like_int(v: str) -> bool:
    try:
        int(v)
        return True
    except ValueError:
        return False


def _looks_like_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _round_up_size(n: int) -> int:
    for step in _SIZE_STEPS:
        if n <= step:
            return step
    return n + 32


# ---------- conversion for ANY supported DataType (inferred or user-overridden) ----------

def convert_value(raw: str, data_type: DataType) -> Value:
    """
    Converts one raw CSV cell into a Value of data_type. Used both for
    columns whose type came from infer_column_type() and for columns
    the user explicitly overrode to a different DataType in the
    frontend — every member of the engine's DataType enum is handled
    here, not just the four automatically inferrable ones.

    An empty string always means NULL, regardless of data_type.
    Raises UnsupportedTypeError (wrapping the original exception) if
    raw cannot be parsed as data_type, so the caller can report which
    row/column/value failed instead of a bare ValueError.
    """
    if raw == "":
        return Value(data_type, None)

    try:
        return Value(data_type, _parse_raw(raw, data_type))
    except UnsupportedTypeError:
        raise
    except (ValueError, InvalidOperation) as exc:
        raise UnsupportedTypeError(
            f"El valor {raw!r} no es válido para el tipo {data_type.value}"
        ) from exc


def _parse_raw(raw: str, data_type: DataType):
    if data_type in (DataType.SMALLINT, DataType.INTEGER, DataType.BIGINT):
        return int(raw)

    if data_type in (DataType.REAL, DataType.DOUBLE_PRECISION):
        return float(raw)

    if data_type == DataType.NUMERIC:
        return Decimal(raw)

    if data_type in (DataType.CHAR, DataType.VARCHAR, DataType.TEXT):
        return raw

    if data_type == DataType.BOOLEAN:
        normalized = raw.strip().lower()
        if normalized in _BOOLEAN_TRUE_LITERALS:
            return True
        if normalized in _BOOLEAN_FALSE_LITERALS:
            return False
        raise ValueError(f"'{raw}' no es un booleano reconocido")

    if data_type == DataType.DATE:
        return date.fromisoformat(raw.strip())

    if data_type == DataType.TIME:
        return time.fromisoformat(raw.strip())

    if data_type == DataType.TIMESTAMP:
        return datetime.fromisoformat(raw.strip())

    if data_type == DataType.BYTEA:
        return bytes.fromhex(raw.strip())

    raise UnsupportedTypeError(f"Tipo de dato no soportado para conversión: {data_type}")