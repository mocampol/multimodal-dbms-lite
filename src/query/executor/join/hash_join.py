"""In-memory hash join Volcano node for equality predicates."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


"""Grace Hash Join Volcano node for equality predicates."""

import hashlib
import os
import pickle
import tempfile
from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class HashJoin(PlanNode):
	"""External hash join using disk-backed Grace hash partitions.

	``memory_records`` bounds the build-side hash table. Oversized
	partitions are repartitioned with a new hash seed; a pathological
	all-equal partition falls back to chunked nested-loop probing, still
	bounded in memory.
	"""

	def __init__(
		self,
		left: PlanNode,
		right: PlanNode,
		left_key: int,
		right_key: int,
		memory_records: int = 256,
		partition_count: int = 16,
	):
		if memory_records <= 0 or partition_count <= 1:
			raise ValueError("memory_records debe ser > 0 y partition_count debe ser > 1")
		self.left = left
		self.right = right
		self.left_key = left_key
		self.right_key = right_key
		self.memory_records = memory_records
		self.partition_count = partition_count
		self._temp_dir = None
		self._output = None

	def open(self) -> None:
		self.left.open()
		self.right.open()
		self._temp_dir = tempfile.TemporaryDirectory(prefix="grace-hash-join-")
		left_paths = self._partition(self.left, self.left_key, "left", 0)
		right_paths = self._partition(self.right, self.right_key, "right", 0)
		self.left.close()
		self.right.close()

		output_path = os.path.join(self._temp_dir.name, "join-output.bin")
		with open(output_path, "wb") as output:
			for partition in range(self.partition_count):
				self._join_partition(
					left_paths[partition], right_paths[partition], output, 0
				)
		self._output = open(output_path, "rb")

	def next(self) -> Optional[Record]:
		if self._output is None:
			return None
		try:
			return pickle.load(self._output)
		except EOFError:
			return None

	def close(self) -> None:
		if self._output is not None:
			self._output.close()
			self._output = None
		if self._temp_dir is not None:
			self._temp_dir.cleanup()
			self._temp_dir = None

	def _partition(self, node: PlanNode, key_index: int, prefix: str, depth: int):
		paths = [
			os.path.join(self._temp_dir.name, f"{prefix}-{depth}-{index}.bin")
			for index in range(self.partition_count)
		]
		handles = [open(path, "wb") for path in paths]
		try:
			while True:
				record = node.next()
				if record is None:
					break
				bucket = self._bucket(record[key_index].data, depth)
				pickle.dump(record, handles[bucket], protocol=pickle.HIGHEST_PROTOCOL)
		finally:
			for handle in handles:
				handle.close()
		return paths

	def _join_partition(self, left_path, right_path, output, depth):
		right_size = self._record_count(right_path)
		if right_size <= self.memory_records:
			self._build_and_probe(left_path, right_path, output)
			return

		if depth < 8:
			left_paths = self._repartition(left_path, self.left_key, "left", depth + 1)
			right_paths = self._repartition(right_path, self.right_key, "right", depth + 1)
			for partition in range(self.partition_count):
				self._join_partition(
					left_paths[partition], right_paths[partition], output, depth + 1
				)
			return

		self._chunked_probe(left_path, right_path, output)

	def _build_and_probe(self, left_path, right_path, output):
		build = {}
		with open(right_path, "rb") as right_file:
			while True:
				try:
					record = pickle.load(right_file)
				except EOFError:
					break
				build.setdefault(record[self.right_key].data, []).append(record)

		with open(left_path, "rb") as left_file:
			while True:
				try:
					left_record = pickle.load(left_file)
				except EOFError:
					break
				for right_record in build.get(left_record[self.left_key].data, []):
					pickle.dump(
						Record(left_record.values + right_record.values),
						output,
						protocol=pickle.HIGHEST_PROTOCOL,
					)

	def _chunked_probe(self, left_path, right_path, output):
		with open(right_path, "rb") as right_file:
			while True:
				chunk = []
				for _ in range(self.memory_records):
					try:
						chunk.append(pickle.load(right_file))
					except EOFError:
						break
				if not chunk:
					return
				with open(left_path, "rb") as left_file:
					while True:
						try:
							left_record = pickle.load(left_file)
						except EOFError:
							break
						for right_record in chunk:
							if left_record[self.left_key].data == right_record[self.right_key].data:
								pickle.dump(
									Record(left_record.values + right_record.values),
									output,
									protocol=pickle.HIGHEST_PROTOCOL,
								)

	def _repartition(self, path, key_index, prefix, depth):
		node = _PartitionReader(path)
		try:
			node.open()
			return self._partition(node, key_index, prefix, depth)
		finally:
			node.close()

	def _bucket(self, value, depth):
		digest = hashlib.blake2b(
			pickle.dumps((depth, value), protocol=pickle.HIGHEST_PROTOCOL),
			digest_size=8,
		).digest()
		return int.from_bytes(digest, "big") % self.partition_count

	@staticmethod
	def _record_count(path):
		count = 0
		with open(path, "rb") as handle:
			while True:
				try:
					pickle.load(handle)
				except EOFError:
					return count
				count += 1


class _PartitionReader(PlanNode):
	def __init__(self, path):
		self.path = path
		self.handle = None

	def open(self):
		self.handle = open(self.path, "rb")

	def next(self):
		try:
			return pickle.load(self.handle)
		except EOFError:
			return None

	def close(self):
		if self.handle is not None:
			self.handle.close()
			self.handle = None

