import math
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation

from common.value import DataType, Value

from .exceptions import UnsupportedTypeError


# Matched during AUTOMATIC inference only — alphabetic forms alone.
# "1"/"0" deliberately absent here: they already satisfy
# _looks_like_int() first, so a column of "1"/"0" converges on
# INTEGER and never reaches this check.
_BOOLEAN_TRUE_WORDS = {"true", "t", "yes", "y"}
_BOOLEAN_FALSE_WORDS = {"false", "f", "no", "n"}

# Matched when the user EXPLICITLY overrides a column to BOOLEAN.
_BOOLEAN_TRUE_LITERALS = _BOOLEAN_TRUE_WORDS | {"1"}
_BOOLEAN_FALSE_LITERALS = _BOOLEAN_FALSE_WORDS | {"0"}

_SIZE_STEPS = (16, 32, 64, 128, 256, 512)

_INT_RANGES = {
    DataType.SMALLINT: (-32_768, 32_767),
    DataType.INTEGER: (-2_147_483_648, 2_147_483_647),
    DataType.BIGINT: (-9_223_372_036_854_775_808, 9_223_372_036_854_775_807),
}


# ---------- streaming inference ----------

def narrow_type(current: DataType | None, raw: str) -> DataType | None:
    """
    Folds one new raw value into the running type candidate for a
    column. `current` is None before any non-empty value has been
    seen yet. An empty cell never changes the candidate.
    """
    if raw == "":
        return current

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

    return DataType.VARCHAR  # ya en VARCHAR: estado absorbente


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


# ---------- conversion for ANY supported DataType ----------

def convert_value(raw: str, data_type: DataType) -> Value:
    if raw == "":
        return Value(data_type, None)

    try:
        return Value(data_type, _parse_raw(raw, data_type))
    except UnsupportedTypeError:
        raise
    except (ValueError, InvalidOperation, OverflowError) as exc:
        raise UnsupportedTypeError(
            f"El valor {raw!r} no es válido para el tipo {data_type.value}"
        ) from exc


def _parse_raw(raw: str, data_type: DataType):
    if data_type in (DataType.SMALLINT, DataType.INTEGER, DataType.BIGINT):
        value = int(raw)
        low, high = _INT_RANGES[data_type]
        if not (low <= value <= high):
            raise UnsupportedTypeError(
                f"{value} está fuera de rango para {data_type.value} ({low} a {high})"
            )
        return value

    if data_type == DataType.REAL:
        import struct
        value = float(raw)
        truncated = struct.unpack("f", struct.pack("f", value))[0]
        if math.isinf(truncated) and not math.isinf(value):
            raise UnsupportedTypeError(
                f"{raw!r} está fuera de rango para REAL (32 bits, máx. ~3.4e38); "
                "usa DOUBLE_PRECISION"
            )
        if truncated != value:
            raise UnsupportedTypeError(
                f"{raw!r} pierde precisión al convertirse a REAL (32 bits); "
                "usa DOUBLE_PRECISION si necesitas conservarlo exacto"
            )
        return truncated

    if data_type == DataType.DOUBLE_PRECISION:
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
        raise UnsupportedTypeError(f"{raw!r} no es un booleano reconocido")

    if data_type == DataType.DATE:
        return date.fromisoformat(raw.strip())

    if data_type == DataType.TIME:
        return time.fromisoformat(raw.strip())

    if data_type == DataType.TIMESTAMP:
        return datetime.fromisoformat(raw.strip())

    if data_type == DataType.BYTEA:
        return bytes.fromhex(raw.strip())

    raise UnsupportedTypeError(f"Tipo de dato no soportado para conversión: {data_type}")