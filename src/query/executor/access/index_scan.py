"""Index-backed access node.

The planner can provide any index object exposing ``search(key)`` and a
storage object exposing ``get(rid)``. Keeping those dependencies explicit
lets Catalog remain the owner of physical index construction.
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class IndexScan(PlanNode):
	def __init__(self, index, storage, key, lock_rid=None):
		self.index = index
		self.storage = storage
		self.key = key
		self.lock_rid = lock_rid
		self._records = []
		self._cursor = 0

	def open(self) -> None:
		search_all = getattr(self.index, "search_all", None)
		rids = search_all(self.key) if search_all is not None else self.index.search(self.key)
		if rids is None:
			rids = []
		if not isinstance(rids, list):
			rids = [rids]
		
		self._records = []
		for rid in rids:
			if self.lock_rid is not None:
				from transaction.lock_manager import LockMode
				self.lock_rid(rid, LockMode.SHARED)
			record = self.storage.get(rid)
			if record is not None:
				self._records.append(record)
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

