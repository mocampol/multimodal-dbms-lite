"""Hash-based GROUP BY node for the current aggregate-free grammar."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode
from common.value import DataType, Value
from query.parser.ast_nodes import AggregateSpec

class HashAggregate(PlanNode):
	"""Emits one representative record for each distinct group key."""

	def __init__(self, child: PlanNode, group_columns: list[str], schema, output_items=None):
		self.child = child
		self.schema = schema
		self.group_columns = group_columns
		self._indices = [schema.column_index(name) for name in group_columns]
		self.output_items = output_items or group_columns
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
			groups.setdefault(key, []).append(record)
		if not groups and not self._indices:
			groups[()] = []
		self._rows = [self._build_row(key, records) for key, records in groups.items()]
		self._cursor = 0

	def _build_row(self, key, records):
		values = []
		for item in self.output_items:
			if isinstance(item, AggregateSpec):
				values.append(self._aggregate(item, records))
			else:
				index = self.schema.column_index(item)
				values.append(records[0][index])
		return Record(values)

	def _aggregate(self, item, records):
		if item.function == "COUNT":
			return Value(DataType.INTEGER, len(records))
		index = self.schema.column_index(item.column)
		data = [record[index].data for record in records if record[index].data is not None]
		if item.function == "SUM":
			return Value(self.schema.columns[index].data_type, sum(data))
		if item.function == "AVG":
			return Value(DataType.DOUBLE_PRECISION, sum(data) / len(data) if data else None)
		if item.function == "MIN":
			return Value(self.schema.columns[index].data_type, min(data) if data else None)
		if item.function == "MAX":
			return Value(self.schema.columns[index].data_type, max(data) if data else None)
		raise ValueError(f"Función agregada no soportada: {item.function}")

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

