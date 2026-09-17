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

    def __init__(self, table_name: str, catalog):
        self.table_name = table_name
        self.catalog = catalog
        self._generator = None

    def open(self) -> None:
        storage = self.catalog.get_storage(self.table_name)
        self._generator = storage.scan()

    def next(self) -> Optional[Record]:
        return next(self._generator, None)

    def close(self) -> None:
        # HeapFile.scan() already unpins each page as it iterates
        self._generator = None
