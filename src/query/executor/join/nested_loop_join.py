"""Nested-loop join Volcano node."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class NestedLoopJoin(PlanNode):
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
		right_rows = []
		while True:
			record = self.right.next()
			if record is None:
				break
			right_rows.append(record)
		self._rows = []
		while True:
			left_record = self.left.next()
			if left_record is None:
				break
			for right_record in right_rows:
				if left_record[self.left_key].data == right_record[self.right_key].data:
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

	def children(self) -> list:
		return [self.left, self.right]

	def replace_children(self, new_children: list) -> None:
		self.left, self.right = new_children

	def describe_self(self) -> dict:
		return {
			"node": "NestedLoopJoin",
			"left_key": self.left_key,
			"right_key": self.right_key,
		}