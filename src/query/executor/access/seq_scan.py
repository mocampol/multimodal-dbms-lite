"""
Sequential Scan: the simplest access node, reading every record of a
table via its storage engine's scan().
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode
from storage.latch import latched_scan, table_latch


class SeqScan(PlanNode):
    """
    Reads every record of table_name, in whatever order the
    underlying storage engine's scan() yields them.
    """

    def __init__(self, table_name: str, catalog, lock_rid=None):
        self.table_name = table_name
        self.catalog = catalog
        self.lock_rid = lock_rid
        self._generator = None
        self._storage = None
        self._latch = None
        self._has_rid = False

    def open(self) -> None:
        self._storage = self.catalog.get_storage(self.table_name)
        self._latch = table_latch(self.catalog, self.table_name)
        if self.lock_rid is not None and hasattr(self._storage, "scan_with_rid"):
            rows = self._storage.scan_with_rid()
            self._has_rid = True
        else:
            rows = self._storage.scan()
            self._has_rid = False
        self._generator = latched_scan(rows, self._latch)

    def next(self) -> Optional[Record]:
        while True:
            item = next(self._generator, None)
            if item is None or not self._has_rid:
                return item
            rid, _ = item
            from transaction.lock_manager import LockMode
            self.lock_rid(rid, LockMode.SHARED)
            # re-read: a writer may have changed or deleted the row while we waited
            with self._latch:
                record = self._storage.get(rid)
            if record is not None:
                return record

    def close(self) -> None:
        # closing the generator unpins the page the storage scan was on
        if self._generator is not None:
            self._generator.close()
        self._generator = None
        self._storage = None
        self._latch = None

    def describe_self(self) -> dict:
        return {"node": "SeqScan", "access": "sequential_scan", "table": self.table_name}
