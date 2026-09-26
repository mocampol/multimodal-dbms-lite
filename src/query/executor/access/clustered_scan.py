from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class ClusteredScan(PlanNode):
    """Accesses records through a clustered index over SequentialFile."""

    def __init__(self, index, key):
        self.index = index
        self.key = key
        self._records = []
        self._cursor = 0

    def open(self) -> None:
        self._records = self.index.search(self.key)
        self._cursor = 0

    def next(self) -> Optional[Record]:
        if self._cursor >= len(self._records):
            return None
        record = self._records[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self._records = []
        self._cursor = 0

    def describe_self(self) -> dict:
        return {
            "node": "ClusteredScan",
            "access": "clustered_index_scan",
            "key": self.key.data,
        }