from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation

from common.value import DataType, Value

from .exceptions import UnsupportedTypeError


_BOOLEAN_TRUE_WORDS = {"true", "t", "yes", "y"}
_BOOLEAN_FALSE_WORDS = {"false", "f", "no", "n"}

_BOOLEAN_TRUE_LITERALS = _BOOLEAN_TRUE_WORDS | {"1"}
_BOOLEAN_FALSE_LITERALS = _BOOLEAN_FALSE_WORDS | {"0"}

_SIZE_STEPS = (16, 32, 64, 128, 256, 512)


# ---------- streaming inference ----------

def narrow_type(current: DataType | None, raw: str) -> DataType:
    """
    Folds one new raw value into the running type candidate for a
    column. `current` is None before the first non-empty value seen;
    empty strings (NULL cells) never change the candidate.
    """
    if raw == "":
        return current if current is not None else DataType.VARCHAR

    if current is None:
        return _narrowest_type_for(raw)

    if current == DataType.INTEGER:
        if _looks_like_int(raw):
            return DataType.INTEGER
        if _looks_like_float(raw):
            return DataType.DOUBLE_PRECISION
        return DataType.VARCHAR

    if current == DataType.DOUBLE_PRECISION:
        if _looks_like_float(raw):
            return DataType.DOUBLE_PRECISION
        return DataType.VARCHAR

    if current == DataType.BOOLEAN:
        if _looks_like_bool_word(raw):
            return DataType.BOOLEAN
        return DataType.VARCHAR

    return DataType.VARCHAR


def narrow_varchar_size(current_max_len: int, raw: str) -> int:
    return max(current_max_len, len(raw))


def resolve_varchar_size(max_len: int) -> int:
    if max_len == 0:
        return 32
    for step in _SIZE_STEPS:
        if max_len <= step:
            return step
    return max_len + 32


def _narrowest_type_for(raw: str) -> DataType:
    if _looks_like_int(raw):
        return DataType.INTEGER
    if _looks_like_float(raw):
        return DataType.DOUBLE_PRECISION
    if _looks_like_bool_word(raw):
        return DataType.BOOLEAN
    return DataType.VARCHAR


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


def _looks_like_bool_word(v: str) -> bool:
    return v.strip().lower() in _BOOLEAN_TRUE_WORDS | _BOOLEAN_FALSE_WORDS


# ---------- conversion for ANY supported DataType (inferred or user-overridden) ----------

def convert_value(raw: str, data_type: DataType) -> Value:
    """
    Converts one raw CSV cell into a Value of data_type. Used both for
    columns whose type came from narrow_type() and for columns the
    user explicitly overrode to a different DataType — every member of
    the engine's DataType enum is handled here.

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