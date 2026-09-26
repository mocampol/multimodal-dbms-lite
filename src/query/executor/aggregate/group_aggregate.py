"""Sort-based GROUP BY for the aggregate-free SQL grammar."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class GroupAggregate(PlanNode):
	"""Emits the first record of each distinct group.

	Aggregate functions are not part of the current parser grammar, so
	this node implements the useful supported form: grouping removes
	duplicate group keys while preserving one representative record.
	The child is expected to be sorted by ``group_columns``.
	"""

	def __init__(self, child: PlanNode, group_columns: list[str], schema):
		self.child = child
		self.group_columns = group_columns
		self._indices = [schema.column_index(name) for name in group_columns]
		self._pending = None
		self._last_key = None

	def open(self) -> None:
		self.child.open()
		self._pending = None
		self._last_key = None

	def next(self) -> Optional[Record]:
		while True:
			record = self.child.next()
			if record is None:
				return None
			key = tuple(record[index].data for index in self._indices)
			if key != self._last_key:
				self._last_key = key
				return record

	def close(self) -> None:
		self.child.close()
		self._pending = None

	def children(self) -> list:
		return [self.child]

	def replace_children(self, new_children: list) -> None:
		(self.child,) = new_children

	def describe_self(self) -> dict:
		return {"node": "GroupAggregate", "group_by": self.group_columns, "strategy": "external_sort_then_stream"}