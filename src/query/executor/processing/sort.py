"""
ORDER BY: materializes its child's entire output in memory and sorts it.

LIMITATION: this is a plain in-memory sort, NOT the External Sorting
(k-way merge) algorithm required by Part II of the project for datasets
that don't fit in RAM. That algorithm is a separate implementation
(expected to live alongside the index subsystem's external algorithms)
and should replace this node's internals once ready.
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class Sort(PlanNode):
    """
    Sorts child's output by columns, ascending, all in memory.
    """

    def __init__(self, child: PlanNode, columns: list, schema):
        self.child = child
        self.columns = columns
        self.schema = schema
        self._indices = [schema.column_index(name) for name in columns]
        self._sorted_records = None
        self._cursor = 0

    def open(self) -> None:
        self.child.open()
        records = []
        while True:
            record = self.child.next()
            if record is None:
                break
            records.append(record)

        records.sort(key=lambda r: tuple(r[i].data for i in self._indices))
        self._sorted_records = records
        self._cursor = 0

    def next(self) -> Optional[Record]:
        if self._cursor >= len(self._sorted_records):
            return None
        record = self._sorted_records[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self.child.close()
        self._sorted_records = None
