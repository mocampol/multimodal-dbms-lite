"""Hash-based GROUP BY node for the current aggregate-free grammar."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class HashAggregate(PlanNode):
	"""Emits one representative record for each distinct group key."""

	def __init__(self, child: PlanNode, group_columns: list[str], schema):
		self.child = child
		self._indices = [schema.column_index(name) for name in group_columns]
		self._rows = []
		self._cursor = 0

	def open(self) -> None:
		self.child.open()
		groups = {}
		while True:
			record = self.child.next()
			if record is None:
				break
			key = tuple(record[index].data for index in self._indices)
			groups.setdefault(key, record)
		self._rows = list(groups.values())
		self._cursor = 0

	def next(self) -> Optional[Record]:
		if self._cursor >= len(self._rows):
			return None
		record = self._rows[self._cursor]
		self._cursor += 1
		return record

	def close(self) -> None:
		self.child.close()
		self._rows = []
		self._cursor = 0

