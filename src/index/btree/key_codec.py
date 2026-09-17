import struct

from common.value import DataType, Value, FIXED_SIZE
from storage.heap.record_codec import encode_scalar, decode_scalar


_LENGTH_PREFIX_FORMAT = ">I"
_LENGTH_PREFIX_SIZE = struct.calcsize(_LENGTH_PREFIX_FORMAT)

_VARIABLE_LENGTH_TYPES = {
    DataType.CHAR,
    DataType.VARCHAR,
    DataType.TEXT,
    DataType.BYTEA,
    DataType.NUMERIC,
}


def encode_key(value: Value) -> bytes:
    if value.data is None:
        raise ValueError(
            "No se pueden indexar valores NULL como clave de un B+ Tree"
        )

    raw = encode_scalar(value)

    if value.data_type in _VARIABLE_LENGTH_TYPES:
        return struct.pack(_LENGTH_PREFIX_FORMAT, len(raw)) + raw

    return raw


def decode_key(data_type: DataType, buf: bytes) -> Value:
    if data_type in _VARIABLE_LENGTH_TYPES:
        length = struct.unpack(
            _LENGTH_PREFIX_FORMAT, buf[:_LENGTH_PREFIX_SIZE]
        )[0]
        raw = buf[_LENGTH_PREFIX_SIZE:_LENGTH_PREFIX_SIZE + length]
        return Value(data_type, decode_scalar(data_type, raw))

    length = FIXED_SIZE[data_type]
    raw = buf[:length]
    return Value(data_type, decode_scalar(data_type, raw))


def key_size(value: Value) -> int:
    if value.data_type in _VARIABLE_LENGTH_TYPES:
        return _LENGTH_PREFIX_SIZE + len(encode_scalar(value))
    return FIXED_SIZE[value.data_type]
