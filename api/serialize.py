import base64
from datetime import date, datetime, time
from decimal import Decimal


def serialize_value(value) -> object:
    data = value.data
    if data is None:
        return None
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, (datetime, date, time)):
        return data.isoformat()
    if isinstance(data, (bytes, bytearray)):
        return base64.b64encode(bytes(data)).decode("ascii")
    return data


def serialize_record(record) -> list:
    return [serialize_value(v) for v in record.values]
