"""Append-only write-ahead log."""

from datetime import date, datetime, time, timezone
from decimal import Decimal
import json
import os
import threading

from common.value import DataType, Point, Value, Polygon, Rectangle
from common.record import Record
from storage.heap.rid import RID


class LogManager:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        self._last_lsn = self._read_last_lsn()

    def append(self, txn_id: int, kind: str, **payload) -> dict:
        with self._lock:
            self._last_lsn += 1
            record = {
                "lsn": self._last_lsn,
                "txn_id": txn_id,
                "kind": kind,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            stream = open(self.path, "a", encoding="utf-8")
            stream.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        return record

    def flush(self):
        """Force all WAL bytes to stable storage."""
        with self._lock:
            stream = open(self.path, "a", encoding="utf-8")
            try:
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                stream.close()

    def begin(self, txn_id: int):
        return self.append(txn_id, "BEGIN")

    def update(self, txn_id: int, resource: str, old_value, new_value):
        return self.append(txn_id, "UPDATE", resource=resource, old=old_value, new=new_value)

    def data_change(self, txn_id, operation, table, rid, before, after, new_rid=None):
        return self.append(
            txn_id, "DATA", operation=operation, table=table,
            rid=rid, new_rid=new_rid, before=before, after=after,
        )

    def commit(self, txn_id: int):
        return self.append(txn_id, "COMMIT")

    def abort(self, txn_id: int):
        return self.append(txn_id, "ABORT")

    def checkpoint(self, active_transactions):
        return self.append(0, "CHECKPOINT", active=list(active_transactions))

    def records(self):
        with self._lock, open(self.path, encoding="utf-8") as stream:
            return [
                json.loads(line, object_hook=_json_object_hook)
                for line in stream if line.strip()
            ]

    def _read_last_lsn(self):
        last = 0
        with open(self.path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    last = json.loads(line)["lsn"]
        return last


def _json_default(value):
    if isinstance(value, RID):
        return {"__rid__": [value.page_id, value.slot]}

    if isinstance(value, Record):
        return {
            "__record__": [
                {"type": v.data_type.value, "data": _encode_value_data(v)}
                for v in value.values
            ]
        }

    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}

    if isinstance(value, Point):
        return {"__point__": [value.longitude, value.latitude]}

    if isinstance(value, Polygon):
        return {"__polygon__": [[p.longitude, p.latitude] for p in value.points]}

    if isinstance(value, Rectangle):
        return {"__rectangle__": [value.west, value.south, value.east, value.north]}

    # datetime before date: datetime is a subclass of date
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}

    if isinstance(value, date):
        return {"__date__": value.isoformat()}

    if isinstance(value, time):
        return {"__time__": value.isoformat()}

    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}

    raise TypeError(f"El WAL no sabe serializar {type(value).__name__}")


def _encode_value_data(value: Value):
    return value.data


def _json_object_hook(obj: dict):
    if "__rid__" in obj:
        page_id, slot = obj["__rid__"]
        return RID(page_id, slot)

    if "__record__" in obj:
        values = [
            Value(DataType(entry["type"]), entry["data"])
            for entry in obj["__record__"]
        ]
        return Record(values)

    if "__bytes__" in obj:
        return bytes.fromhex(obj["__bytes__"])

    if "__point__" in obj:
        return Point(*obj["__point__"])

    if "__polygon__" in obj:
        return Polygon([Point(*p) for p in obj["__polygon__"]])

    if "__rectangle__" in obj:
        return Rectangle(*obj["__rectangle__"])

    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])

    if "__date__" in obj:
        return date.fromisoformat(obj["__date__"])

    if "__time__" in obj:
        return time.fromisoformat(obj["__time__"])

    if "__decimal__" in obj:
        return Decimal(obj["__decimal__"])

    return obj