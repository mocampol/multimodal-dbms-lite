from common.record import Record
from common.value import DataType, Value
from query.executor.executor import run_plan
from query.executor.join.hash_join import HashJoin
from query.executor.plan_node import PlanNode


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


def record(key, value):
    return Record([
        Value(DataType.INTEGER, key),
        Value(DataType.VARCHAR, value),
    ])


def test_hash_join_spills_and_preserves_duplicate_matches():
    left = RecordsNode([record(1, "l1"), record(1, "l2"), record(2, "l3")])
    right = RecordsNode([record(1, "r1"), record(1, "r2"), record(3, "r3")])
    node = HashJoin(left, right, 0, 0, memory_records=1, partition_count=2)

    result = list(run_plan(node))

    assert len(result) == 4
    assert {(row[1].data, row[3].data) for row in result} == {
        ("l1", "r1"), ("l1", "r2"), ("l2", "r1"), ("l2", "r2"),
    }