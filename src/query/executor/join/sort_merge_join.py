"""Sort-merge join Volcano node for equality predicates."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class SortMergeJoin(PlanNode):
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
		left_rows = self._read_all(self.left)
		right_rows = self._read_all(self.right)
		left_rows.sort(key=lambda row: row[self.left_key].data)
		right_rows.sort(key=lambda row: row[self.right_key].data)

		self._rows = []
		left_index = right_index = 0
		while left_index < len(left_rows) and right_index < len(right_rows):
			left_key = left_rows[left_index][self.left_key].data
			right_key = right_rows[right_index][self.right_key].data
			if left_key < right_key:
				left_index += 1
				continue
			if left_key > right_key:
				right_index += 1
				continue

			left_end = left_index
			while left_end < len(left_rows) and left_rows[left_end][self.left_key].data == left_key:
				left_end += 1
			right_end = right_index
			while right_end < len(right_rows) and right_rows[right_end][self.right_key].data == right_key:
				right_end += 1
			for left_record in left_rows[left_index:left_end]:
				for right_record in right_rows[right_index:right_end]:
					self._rows.append(Record(left_record.values + right_record.values))
			left_index = left_end
			right_index = right_end
		self._cursor = 0

	@staticmethod
	def _read_all(node):
		rows = []
		while True:
			record = node.next()
			if record is None:
				return rows
			rows.append(record)

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
			"node": "SortMergeJoin",
			"join_type": "sort_merge",
			"strategy": "sort_merge_external",
			"left_key": self.left_key,
			"right_key": self.right_key,
		}