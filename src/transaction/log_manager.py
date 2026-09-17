"""Append-only write-ahead log."""

from datetime import datetime, timezone
import json
import os
import threading


class LogManager:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "a", encoding="utf-8").close()

    def append(self, txn_id: int, kind: str, **payload) -> dict:
        with self._lock:
            record = {
                "lsn": self._next_lsn_unlocked(),
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

    def begin(self, txn_id: int):
        return self.append(txn_id, "BEGIN")

    def update(self, txn_id: int, resource: str, old_value, new_value):
        return self.append(txn_id, "UPDATE", resource=resource, old=old_value, new=new_value)

    def data_change(self, txn_id, operation, table, rid, before, after):
        return self.append(
            txn_id, "DATA", operation=operation, table=table,
            rid=rid, before=before, after=after,
        )

    def commit(self, txn_id: int):
        return self.append(txn_id, "COMMIT")

    def abort(self, txn_id: int):
        return self.append(txn_id, "ABORT")

    def checkpoint(self, active_transactions):
        return self.append(0, "CHECKPOINT", active=list(active_transactions))

    def records(self):
        with self._lock, open(self.path, encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def _next_lsn_unlocked(self):
        with open(self.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return records[-1]["lsn"] + 1 if records else 1


def _json_default(value):
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if hasattr(value, "isoformat"):
        return {"__datetime__": value.isoformat()}
    if hasattr(value, "as_tuple"):
        return {"__decimal__": str(value)}
    return repr(value)
