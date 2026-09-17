"""Sort-based aggregation node for already ordered input."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode
from query.executor.aggregate.hash_aggregate import HashAggregate


class SortAggregate(PlanNode):
	"""Aggregates consecutive equal groups from a sorted child.

	The external ``Sort`` node is normally placed before this node by the
	planner, so the aggregation itself only keeps one group at a time.
	"""

	def __init__(self, child: PlanNode, group_columns: list[str], schema, output_items=None):
		self.child = child
		self.group_columns = group_columns
		self.schema = schema
		self.output_items = output_items or group_columns
		self._indices = [schema.column_index(name) for name in group_columns]
		self._pending = None
		self._finished = False

	def open(self) -> None:
		self.child.open()
		self._pending = self.child.next()
		self._finished = False

	def next(self) -> Optional[Record]:
		if self._finished or self._pending is None:
			return None
		group = [self._pending]
		group_key = tuple(self._pending[index].data for index in self._indices)
		while True:
			record = self.child.next()
			if record is None:
				self._pending = None
				self._finished = True
				break
			key = tuple(record[index].data for index in self._indices)
			if key == group_key:
				group.append(record)
			else:
				self._pending = record
				break
		helper = HashAggregate(_RowsNode(group), self.group_columns, self.schema, self.output_items)
		helper.open()
		result = helper.next()
		helper.close()
		return result

	def close(self) -> None:
		self.child.close()
		self._pending = None
		self._finished = True


class _RowsNode(PlanNode):
	def __init__(self, rows):
		self.rows = rows
		self.cursor = 0

	def open(self) -> None:
		self.cursor = 0

	def next(self):
		if self.cursor >= len(self.rows):
			return None
		row = self.rows[self.cursor]
		self.cursor += 1
		return row

	def close(self) -> None:
		self.cursor = 0

