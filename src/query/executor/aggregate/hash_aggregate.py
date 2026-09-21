"""Hash-based GROUP BY with external partitioning when memory is exceeded."""

import hashlib
import os
import pickle
import tempfile
from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode
from common.value import DataType, Value
from query.parser.ast_nodes import AggregateSpec

class HashAggregate(PlanNode):
	"""Emits one representative record for each distinct group key."""

	def __init__(self, child: PlanNode, group_columns: list[str], schema, output_items=None,
	             memory_records: int = 256, partition_count: int = 16):
		if memory_records <= 0 or partition_count <= 1:
			raise ValueError("memory_records debe ser > 0 y partition_count debe ser > 1")
		self.child = child
		self.schema = schema
		self.group_columns = group_columns
		self._indices = [schema.column_index(name) for name in group_columns]
		self.output_items = output_items or group_columns
		self.memory_records = memory_records
		self.partition_count = partition_count
		self._rows = []
		self._cursor = 0
		self._temp_dir = None
		self._partition_paths = []
		self._partition_index = 0

	def open(self) -> None:
		self.child.open()
		groups = {}
		record_count = 0
		spilled = False
		while True:
			record = self.child.next()
			if record is None:
				break
			if record_count >= self.memory_records:
				spilled = True
				break
			record_count += 1
			key = tuple(record[index].data for index in self._indices)
			groups.setdefault(key, []).append(record)
		if spilled:
			self._spill(groups, record)
			while True:
				record = self.child.next()
				if record is None:
					break
				self._spill_record(record)
			for handle in self._partition_handles:
				handle.close()
			self._partition_handles = []
			self.child.close()
			self._load_partition(0)
		else:
			if not groups and not self._indices:
				groups[()] = []
			self._rows = [self._build_row(key, records) for key, records in groups.items()]
			self._cursor = 0
			return

	def _spill(self, groups, first_record):
		self._temp_dir = tempfile.TemporaryDirectory(prefix="query-group-")
		self._partition_paths = [
			os.path.join(self._temp_dir.name, f"partition-{index}.bin")
			for index in range(self.partition_count)
		]
		self._partition_handles = [open(path, "wb") for path in self._partition_paths]
		for records in groups.values():
			for record in records:
				self._spill_record(record)
		self._spill_record(first_record)

	def _spill_record(self, record):
		key = tuple(record[index].data for index in self._indices)
		bucket = self._bucket(key)
		pickle.dump(record, self._partition_handles[bucket], protocol=pickle.HIGHEST_PROTOCOL)

	def _load_partition(self, partition_index):
		if partition_index >= len(self._partition_paths):
			self._rows = []
			self._cursor = 0
			return
		self._partition_index = partition_index
		groups = {}
		with open(self._partition_paths[partition_index], "rb") as handle:
			while True:
				try:
					record = pickle.load(handle)
				except EOFError:
					break
				key = tuple(record[index].data for index in self._indices)
				state = groups.setdefault(key, self._new_state(record))
				self._accumulate(state, record)
		if not groups and not self._indices:
			groups[()] = self._new_state(None)
		self._rows = [self._build_state_row(key, state) for key, state in groups.items()]
		self._cursor = 0

	def _new_state(self, representative):
		return {
			"representative": representative,
			"count": 0,
			"aggregates": {},
		}

	def _accumulate(self, state, record):
		state["count"] += 1
		for item in self.output_items:
			if not isinstance(item, AggregateSpec) or item.function == "COUNT":
				continue
			index = self.schema.column_index(item.column)
			value = record[index].data
			if value is None:
				continue
			aggregate = state["aggregates"].setdefault(item.name, {
				"count": 0, "sum": 0, "min": value, "max": value,
			})
			aggregate["count"] += 1
			aggregate["sum"] += value
			aggregate["min"] = min(aggregate["min"], value)
			aggregate["max"] = max(aggregate["max"], value)

	def _build_state_row(self, key, state):
		values = []
		for item in self.output_items:
			if not isinstance(item, AggregateSpec):
				index = self.schema.column_index(item)
				values.append(state["representative"][index])
				continue
			if item.function == "COUNT":
				values.append(Value(DataType.INTEGER, state["count"]))
				continue
			aggregate = state["aggregates"].get(item.name)
			index = self.schema.column_index(item.column)
			if aggregate is None:
				value = None
			elif item.function == "SUM":
				value = aggregate["sum"]
			elif item.function == "AVG":
				value = aggregate["sum"] / aggregate["count"]
			elif item.function == "MIN":
				value = aggregate["min"]
			elif item.function == "MAX":
				value = aggregate["max"]
			else:
				raise ValueError(f"Función agregada no soportada: {item.function}")
			data_type = (
				DataType.DOUBLE_PRECISION
				if item.function == "AVG"
				else self.schema.columns[index].data_type
			)
			values.append(Value(data_type, value))
		return Record(values)

	def _bucket(self, key):
		digest = hashlib.blake2b(
			pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL), digest_size=8
		).digest()
		return int.from_bytes(digest, "big") % self.partition_count

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
		while self._cursor >= len(self._rows):
			if not self._partition_paths or self._partition_index >= self.partition_count - 1:
				return None
			self._load_partition(self._partition_index + 1)
		record = self._rows[self._cursor]
		self._cursor += 1
		return record

	def close(self) -> None:
		self.child.close()
		for handle in getattr(self, "_partition_handles", []):
			handle.close()
		self._partition_handles = []
		self._rows = []
		self._cursor = 0
		self._partition_paths = []
		if self._temp_dir is not None:
			self._temp_dir.cleanup()
			self._temp_dir = None

