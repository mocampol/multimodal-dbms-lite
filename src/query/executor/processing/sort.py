"""External ORDER BY using sorted runs and a k-way merge."""

import heapq
import pickle
import tempfile
from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode


class Sort(PlanNode):
    """
    Sorts child's output externally. ``memory_records`` is the maximum
    number of records held while generating one initial run.
    """

    def __init__(self, child: PlanNode, columns: list, schema, memory_records: int = 256):
        if memory_records <= 0:
            raise ValueError("memory_records debe ser mayor que 0")
        self.child = child
        self.columns = columns
        self.schema = schema
        self._indices = [schema.column_index(name) for name in columns]
        self.memory_records = memory_records
        self._temp_dir = None
        self._runs = []
        self._handles = []
        self._heap = []

    def open(self) -> None:
        self.child.open()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="query-sort-")
        run = []
        while True:
            record = self.child.next()
            if record is None:
                break
            run.append(record)
            if len(run) >= self.memory_records:
                self._write_run(run)
                run = []
        if run:
            self._write_run(run)

        for run_index, path in enumerate(self._runs):
            handle = open(path, "rb")
            self._handles.append(handle)
            record = pickle.load(handle)
            heapq.heappush(self._heap, (self._sort_key(record), run_index, record))

    def next(self) -> Optional[Record]:
        if not self._heap:
            return None
        _, run_index, record = heapq.heappop(self._heap)
        handle = self._handles[run_index]
        try:
            next_record = pickle.load(handle)
        except EOFError:
            next_record = None
        if next_record is not None:
            heapq.heappush(
                self._heap,
                (self._sort_key(next_record), run_index, next_record),
            )
        return record

    def close(self) -> None:
        self.child.close()
        for handle in self._handles:
            handle.close()
        self._handles = []
        self._heap = []
        self._runs = []
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _sort_key(self, record: Record):
        return tuple(record[index].data for index in self._indices)

    def _write_run(self, records: list[Record]):
        records.sort(key=self._sort_key)
        path = tempfile.NamedTemporaryFile(
            dir=self._temp_dir.name, prefix="run-", suffix=".bin", delete=False
        ).name
        with open(path, "wb") as handle:
            for record in records:
                pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._runs.append(path)
