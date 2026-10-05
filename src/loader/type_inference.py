import math
import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation

from common.value import DataType, Point, Polygon, Rectangle, Value

from .exceptions import UnsupportedTypeError

_SPATIAL_TYPES = {DataType.POINT, DataType.POLYGON, DataType.RECTANGLE, DataType.GEOMETRY}
_SPATIAL_PREFIX = re.compile(r"^\s*(POINT|POLYGON|RECTANGLE)\s*\(", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


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

    if current in _SPATIAL_TYPES:
        spatial = _spatial_type_for(raw)
        if spatial is None:
            return DataType.VARCHAR
        # una columna que mezcla POINT y POLYGON, por ejemplo, es GEOMETRY
        return current if spatial == current else DataType.GEOMETRY

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
    return _spatial_type_for(raw) or DataType.VARCHAR


def _spatial_type_for(raw: str) -> DataType | None:
    """POINT/POLYGON/RECTANGLE if raw is a valid literal with that prefix."""
    match = _SPATIAL_PREFIX.match(raw)
    if match is None:
        return None
    data_type = DataType(match.group(1).lower())
    try:
        _parse_spatial(raw, data_type)
    except (ValueError, UnsupportedTypeError):
        return None
    return data_type


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

    if data_type in _SPATIAL_TYPES:
        return _parse_spatial(raw, data_type)

    raise UnsupportedTypeError(f"Tipo de dato no soportado para conversión: {data_type}")

# ---------- spatial literals ----------

def _parse_spatial(raw: str, data_type: DataType):
    """
    Parses a spatial CSV cell. Coordinates go as (longitude latitude),
    separated by spaces or commas; both WKT and the engine's SQL syntax work:

        POINT(-77.03 -12.12)          POINT(-77.03, -12.12)      -77.03 -12.12
        POLYGON((x y, x y, x y))      POLYGON(POINT(x, y), POINT(x, y), POINT(x, y))
        RECTANGLE(west south east north)

    GEOMETRY takes whichever kind the prefix names. A polygon that does not
    repeat its first vertex at the end is closed automatically, as in SQL.
    """
    text = raw.strip()
    match = _SPATIAL_PREFIX.match(text)
    kind = DataType(match.group(1).lower()) if match else None

    if data_type == DataType.GEOMETRY:
        data_type = kind or DataType.POINT
    elif kind is not None and kind != data_type:
        raise UnsupportedTypeError(f"{raw!r} es un {kind.value.upper()}, no un {data_type.value.upper()}")

    if match is not None:
        body = text[match.end():]
        if not body.endswith(")"):
            raise ValueError(f"{raw!r}: falta el paréntesis de cierre")
        body = body[:-1]
    else:
        body = text
    # dentro del cuerpo solo pueden quedar números, separadores y POINT(...)
    if re.sub(r"POINT", "", _NUMBER.sub("", body), flags=re.IGNORECASE).strip(" ,()\t"):
        raise ValueError(f"{raw!r} no es un literal {data_type.value.upper()} válido")
    numbers = [float(n) for n in _NUMBER.findall(body)]

    if data_type == DataType.POINT:
        if len(numbers) != 2:
            raise ValueError(f"POINT requiere 2 coordenadas (longitud latitud), se encontraron {len(numbers)}")
        return Point(*numbers)

    if data_type == DataType.RECTANGLE:
        if len(numbers) != 4:
            raise ValueError(f"RECTANGLE requiere 4 valores (oeste sur este norte), se encontraron {len(numbers)}")
        return Rectangle(*numbers)

    if len(numbers) % 2 != 0:
        raise ValueError("POLYGON requiere pares de coordenadas (longitud latitud)")
    points = [Point(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return Polygon(points)
