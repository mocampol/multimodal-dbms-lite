"""Bitmap-index access node."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class BitmapScan(PlanNode):
	"""Reads RIDs from a bitmap provider and fetches their records."""

	def __init__(self, bitmap_index, storage, key=None):
		self.bitmap_index = bitmap_index
		self.storage = storage
		self.key = key
		self._records = []
		self._cursor = 0

	def open(self) -> None:
		rids = self.bitmap_index.scan() if self.key is None else self.bitmap_index.search(self.key)
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

	def describe_self(self) -> dict:
		return {
			"node": "BitmapScan",
			"index": type(self.bitmap_index).__name__,
			"key": repr(self.key) if self.key is not None else None,
		}