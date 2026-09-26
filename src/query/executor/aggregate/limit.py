"""LIMIT Volcano node."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class Limit(PlanNode):
	def __init__(self, child: PlanNode, limit: int):
		if limit < 0:
			raise ValueError("limit debe ser mayor o igual que 0")
		self.child = child
		self.limit = limit
		self._seen = 0

	def open(self) -> None:
		self.child.open()
		self._seen = 0

	def next(self) -> Optional[Record]:
		if self._seen >= self.limit:
			return None
		record = self.child.next()
		if record is not None:
			self._seen += 1
		return record

	def close(self) -> None:
		self.child.close()
		self._seen = 0

	def children(self) -> list:
		return [self.child]

	def replace_children(self, new_children: list) -> None:
		(self.child,) = new_children

	def describe_self(self) -> dict:
		return {"node": "Limit", "limit": self.limit}