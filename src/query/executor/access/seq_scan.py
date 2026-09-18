"""
Sequential Scan: the simplest access node, reading every record of a
table via its storage engine's scan().
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


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
        self._has_rid = False

    def open(self) -> None:
        storage = self.catalog.get_storage(self.table_name)
        if self.lock_rid is not None and hasattr(storage, "scan_with_rid"):
            self._generator = storage.scan_with_rid()
            self._has_rid = True
        else:
            self._generator = storage.scan()
            self._has_rid = False

    def next(self) -> Optional[Record]:
        item = next(self._generator, None)
        if item is None:
            return None
        if self._has_rid:
            rid, record = item
            from transaction.lock_manager import LockMode
            self.lock_rid(rid, LockMode.SHARED)
            return record
        return item

    def close(self) -> None:
        # HeapFile.scan() already unpins each page as it iterates
        self._generator = None
