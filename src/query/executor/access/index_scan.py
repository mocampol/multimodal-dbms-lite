"""Index-backed access node.

The planner can provide any index object exposing ``search(key)`` and a
storage object exposing ``get(rid)``. Keeping those dependencies explicit
lets Catalog remain the owner of physical index construction.
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class IndexScan(PlanNode):
	def __init__(self, index, storage, key):
		self.index = index
		self.storage = storage
		self.key = key
		self._records = []
		self._cursor = 0

	def open(self) -> None:
		rids = self.index.search(self.key)
		if rids is None:
			rids = []
		if not isinstance(rids, list):
			rids = [rids]
		self._records = [
			record for rid in rids
			if (record := self.storage.get(rid)) is not None
		]
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

