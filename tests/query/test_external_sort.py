from common.record import Record
from common.schema import Column, Schema
from common.value import DataType, Value
from query.executor.executor import run_plan
from query.executor.plan_node import PlanNode
from query.executor.processing.sort import Sort


class RecordsNode(PlanNode):
    def __init__(self, records):
        self.records = records
        self.index = 0

    def open(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.records):
            return None
        record = self.records[self.index]
        self.index += 1
        return record

    def close(self):
        pass


def test_sort_merges_multiple_external_runs():
    schema = Schema("items", [Column("value", DataType.INTEGER)])
    records = [Record([Value(DataType.INTEGER, value)]) for value in [4, 1, 3, 2]]
    node = Sort(RecordsNode(records), ["value"], schema, memory_records=2)

    result = list(run_plan(node))

    assert [row[0].data for row in result] == [1, 2, 3, 4]


def test_sort_handles_null_values_across_runs():
    schema = Schema("items", [Column("value", DataType.INTEGER, nullable=True)])
    records = [
        Record([Value(DataType.INTEGER, value)])
        for value in [3, None, 1, 2]
    ]
    node = Sort(RecordsNode(records), ["value"], schema, memory_records=2)

    result = list(run_plan(node))

    assert [row[0].data for row in result] == [None, 1, 2, 3]