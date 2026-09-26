"""
SELECT column list: reshapes each record from its child into one
containing only the requested columns (or passes it through unchanged
for SELECT *).
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class Projection(PlanNode):
    """
    Projects child's records onto columns (a list of column names,
    or ["*"] to keep every column as-is).
    """

    def __init__(self, child: PlanNode, columns: list, schema):
        self.child = child
        self.columns = columns
        self.schema = schema
        self._indices = None if columns == ["*"] else [
            schema.column_index(name) for name in columns
        ]

    def open(self) -> None:
        self.child.open()

    def next(self) -> Optional[Record]:
        record = self.child.next()
        if record is None:
            return None
        if self._indices is None:
            return record
        return Record([record[i] for i in self._indices])

    def close(self) -> None:
        self.child.close()

    def children(self) -> list:
        return [self.child]

    def replace_children(self, new_children: list) -> None:
        (self.child,) = new_children

    def describe_self(self) -> dict:
        return {"node": "Projection", "columns": self.columns}