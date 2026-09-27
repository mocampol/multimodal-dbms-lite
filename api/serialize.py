import base64
from datetime import date, datetime, time
from decimal import Decimal

from common.value import Distance, Point
from spatial.geometry import Point2D


def serialize_value(value) -> object:
    data = value.data
    if data is None:
        return None
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, Point):
        return {"longitude": data.longitude, "latitude": data.latitude}
    if isinstance(data, Point2D):
        return {"longitude": data.x, "latitude": data.y}
    if isinstance(data, Distance):
        return {"value": data.value, "unit": data.unit.value}
    if isinstance(data, (datetime, date, time)):
        return data.isoformat()
    if isinstance(data, (bytes, bytearray)):
        return base64.b64encode(bytes(data)).decode("ascii")
    return data


def serialize_record(record) -> list:
    return [serialize_value(v) for v in record.values]
