"""In-memory hash join Volcano node for equality predicates."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class HashJoin(PlanNode):
	def __init__(self, left: PlanNode, right: PlanNode, left_key: int, right_key: int):
		self.left = left
		self.right = right
		self.left_key = left_key
		self.right_key = right_key
		self._rows = []
		self._cursor = 0

	def open(self) -> None:
		self.left.open()
		self.right.open()
		buckets = {}
		while True:
			record = self.right.next()
			if record is None:
				break
			buckets.setdefault(record[self.right_key].data, []).append(record)

		self._rows = []
		while True:
			left_record = self.left.next()
			if left_record is None:
				break
			for right_record in buckets.get(left_record[self.left_key].data, []):
				self._rows.append(Record(left_record.values + right_record.values))
		self._cursor = 0

	def next(self) -> Optional[Record]:
		if self._cursor >= len(self._rows):
			return None
		record = self._rows[self._cursor]
		self._cursor += 1
		return record

	def close(self) -> None:
		self.left.close()
		self.right.close()
		self._rows = []
		self._cursor = 0

