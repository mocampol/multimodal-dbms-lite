"""
Binary encoding/decoding of Record <-> bytes, using a Schema to know
column order and types. Used by heap_file to serialize records before
writing them into a page's data area, and to deserialize bytes read
back from a page.

Encoding layout for one record:
    [null_bitmap: ceil(n_columns / 8) bytes]
    for each column, in schema order, if NOT null:
        - FIXED types (SMALLINT, INTEGER, BIGINT, REAL, DOUBLE_PRECISION,
          BOOLEAN, DATE, TIME, TIMESTAMP): a fixed number of bytes,
          matching common.value.FIXED_SIZE exactly (see _FIXED_CODECS).
        - Variable-length types (CHAR, VARCHAR, TEXT, BYTEA, NUMERIC):
          a 4-byte big-endian length prefix, followed by that many bytes.

A field whose null_bitmap bit is 1 contributes NO bytes beyond the
bitmap itself — its value is reconstructed as Value(data_type, None).
"""

import struct
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from common.value import DataType, Point, Value, Polygon, Rectangle, FIXED_SIZE
from common.schema import Schema
from common.record import Record


_EPOCH_DATE = date(1970, 1, 1)
_EPOCH_DATETIME = datetime(1970, 1, 1)

_LENGTH_PREFIX_FORMAT = ">I"
_LENGTH_PREFIX_SIZE = struct.calcsize(_LENGTH_PREFIX_FORMAT)


def _encode_date(value: date) -> bytes:
    return struct.pack(">i", (value - _EPOCH_DATE).days)


def _decode_date(buf: bytes) -> date:
    days = struct.unpack(">i", buf)[0]
    return _EPOCH_DATE + timedelta(days=days)


def _encode_timestamp(value: datetime) -> bytes:
    micros = int((value - _EPOCH_DATETIME).total_seconds() * 1_000_000)
    return struct.pack(">q", micros)


def _decode_timestamp(buf: bytes) -> datetime:
    micros = struct.unpack(">q", buf)[0]
    return _EPOCH_DATETIME + timedelta(microseconds=micros)


def _encode_time(value: time) -> bytes:
    micros = (
        (value.hour * 3600 + value.minute * 60 + value.second) * 1_000_000
        + value.microsecond
    )
    return struct.pack(">q", micros)


def _decode_time(buf: bytes) -> time:
    micros = struct.unpack(">q", buf)[0]
    total_seconds, micro = divmod(micros, 1_000_000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return time(hour=hours, minute=minutes, second=seconds, microsecond=micro)


# Fixed-size types: (struct_format, encode_fn, decode_fn).
# When struct_format is set, encoding/decoding go straight through struct.
# DATE/TIME/TIMESTAMP use dedicated helpers instead (they wrap a Python
# date/time/datetime, not a plain number), but their resulting byte
# length still matches common.value.FIXED_SIZE exactly, by design.
_FIXED_CODECS = {
    DataType.SMALLINT: (">h", None, None),
    DataType.INTEGER: (">i", None, None),
    DataType.BIGINT: (">q", None, None),
    DataType.REAL: (">f", None, None),
    DataType.DOUBLE_PRECISION: (">d", None, None),
    DataType.BOOLEAN: (">?", None, None),
    DataType.DATE: (None, _encode_date, _decode_date),
    DataType.TIME: (None, _encode_time, _decode_time),
    DataType.TIMESTAMP: (None, _encode_timestamp, _decode_timestamp),
}

_VARIABLE_LENGTH_TYPES = {
    DataType.CHAR, DataType.VARCHAR, DataType.TEXT,
    DataType.BYTEA, DataType.NUMERIC, DataType.POLYGON,
    DataType.GEOMETRY,
}


def encode_scalar(value: Value) -> bytes:
    """
    Encodes a single non-null Value into its raw payload bytes,
    without any length prefix (the caller adds that for variable-length types).
    """
    dt = value.data_type

    if dt == DataType.POINT:
        return struct.pack(">dd", value.data.longitude, value.data.latitude)

    if dt == DataType.RECTANGLE:
        return struct.pack(">dddd", value.data.west, value.data.south, value.data.east, value.data.north)

    if dt == DataType.POLYGON:
        count = len(value.data.points)
        fmt = f">I{count * 2}d"
        coords = []
        for p in value.data.points:
            coords.extend([p.longitude, p.latitude])
        return struct.pack(fmt, count, *coords)

    if dt == DataType.GEOMETRY:
        from common.value import Point, Rectangle, Polygon
        if isinstance(value.data, Point):
            return b"\x01" + struct.pack(">dd", value.data.longitude, value.data.latitude)
        if isinstance(value.data, Rectangle):
            return b"\x02" + struct.pack(">dddd", value.data.west, value.data.south, value.data.east, value.data.north)
        if isinstance(value.data, Polygon):
            count = len(value.data.points)
            fmt = f">I{count * 2}d"
            coords = []
            for p in value.data.points:
                coords.extend([p.longitude, p.latitude])
            return b"\x03" + struct.pack(fmt, count, *coords)
        raise ValueError(f"Geometría no soportada para codificación: {type(value.data)}")

    if dt in _FIXED_CODECS:
        fmt, encode_fn, _ = _FIXED_CODECS[dt]
        return struct.pack(fmt, value.data) if fmt else encode_fn(value.data)

    if dt in {DataType.CHAR, DataType.VARCHAR, DataType.TEXT}:
        return value.data.encode("utf-8")

    if dt == DataType.BYTEA:
        return bytes(value.data)

    if dt == DataType.NUMERIC:
        return str(value.data).encode("utf-8")

    raise ValueError(f"No se sabe codificar el tipo {dt}")


def decode_scalar(data_type: DataType, buf: bytes):
    """
    Decodes raw payload bytes (already stripped of any length prefix)
    back into the Python value for data_type.
    """
    if data_type == DataType.POINT:
        return Point(*struct.unpack(">dd", buf))

    if data_type == DataType.RECTANGLE:
        return Rectangle(*struct.unpack(">dddd", buf))

    if data_type == DataType.POLYGON:
        count = struct.unpack_from(">I", buf, 0)[0]
        fmt = f">{count * 2}d"
        coords = struct.unpack_from(fmt, buf, 4)
        from common.value import Polygon
        points = []
        for i in range(count):
            points.append(Point(coords[i*2], coords[i*2+1]))
        return Polygon(points)

    if data_type == DataType.GEOMETRY:
        tag = buf[0]
        payload = buf[1:]
        if tag == 1:
            return Point(*struct.unpack(">dd", payload))
        if tag == 2:
            return Rectangle(*struct.unpack(">dddd", payload))
        if tag == 3:
            count = struct.unpack_from(">I", payload, 0)[0]
            fmt = f">{count * 2}d"
            coords = struct.unpack_from(fmt, payload, 4)
            from common.value import Polygon
            points = [Point(coords[i*2], coords[i*2+1]) for i in range(count)]
            return Polygon(points)
        raise ValueError(f"Tag de GEOMETRY desconocido: {tag}")

    if data_type in _FIXED_CODECS:
        fmt, _, decode_fn = _FIXED_CODECS[data_type]
        return struct.unpack(fmt, buf)[0] if fmt else decode_fn(buf)

    if data_type in {DataType.CHAR, DataType.VARCHAR, DataType.TEXT}:
        return buf.decode("utf-8")

    if data_type == DataType.BYTEA:
        return bytes(buf)

    if data_type == DataType.NUMERIC:
        return Decimal(buf.decode("utf-8"))

    raise ValueError(f"No se sabe decodificar el tipo {data_type}")


def encode_record(record: Record, schema: Schema) -> bytes:
    """
    Serializes record into bytes, using schema to know each column's
    type and nullability. Assumes record.validate(schema) already passed.
    """
    n = len(schema.columns)
    null_bitmap = bytearray((n + 7) // 8)
    payload = bytearray()

    for i, (value, column) in enumerate(zip(record.values, schema.columns)):
        if value.data is None:
            null_bitmap[i // 8] |= (1 << (i % 8))
            continue

        raw = encode_scalar(value)
        if column.data_type in _VARIABLE_LENGTH_TYPES:
            payload += struct.pack(_LENGTH_PREFIX_FORMAT, len(raw))
        payload += raw

    return bytes(null_bitmap) + bytes(payload)


def decode_record(buf: bytes, schema: Schema) -> Record:
    """
    Deserializes bytes produced by encode_record() back into a Record,
    using schema to know each column's type, order and nullability.
    """
    n = len(schema.columns)
    bitmap_size = (n + 7) // 8
    null_bitmap = buf[:bitmap_size]
    offset = bitmap_size

    values = []
    for i, column in enumerate(schema.columns):
        is_null = (null_bitmap[i // 8] >> (i % 8)) & 1

        if is_null:
            values.append(Value(column.data_type, None))
            continue

        if column.data_type in _VARIABLE_LENGTH_TYPES:
            length = struct.unpack(
                _LENGTH_PREFIX_FORMAT, buf[offset:offset + _LENGTH_PREFIX_SIZE]
            )[0]
            offset += _LENGTH_PREFIX_SIZE
            raw = buf[offset:offset + length]
            offset += length
        else:
            length = FIXED_SIZE[column.data_type]
            raw = buf[offset:offset + length]
            offset += length

        values.append(Value(column.data_type, decode_scalar(column.data_type, raw)))

    return Record(values)
