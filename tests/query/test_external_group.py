from common.record import Record
from common.schema import Column, Schema
from common.value import DataType, Value
from query.executor.aggregate.hash_aggregate import HashAggregate
from query.executor.executor import run_plan
from query.executor.plan_node import PlanNode
from query.parser.ast_nodes import AggregateSpec


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


def test_hash_group_by_spills_to_external_partitions():
    schema = Schema("items", [
        Column("category", DataType.VARCHAR, size=16),
        Column("amount", DataType.INTEGER),
    ])
    records = [
        Record([Value(DataType.VARCHAR, category), Value(DataType.INTEGER, amount)])
        for category, amount in [("a", 1), ("b", 2), ("a", 3), ("c", 4)]
    ]
    node = HashAggregate(
        RecordsNode(records), ["category"], schema,
        memory_records=1, partition_count=4,
    )

    result = list(run_plan(node))

    assert {row[0].data for row in result} == {"a", "b", "c"}


def test_external_group_aggregates_skewed_partition_without_row_lists():
    schema = Schema("items", [
        Column("category", DataType.VARCHAR, size=16),
        Column("amount", DataType.INTEGER),
    ])
    records = [
        Record([Value(DataType.VARCHAR, "same"), Value(DataType.INTEGER, amount)])
        for amount in range(1, 6)
    ]
    node = HashAggregate(
        RecordsNode(records), ["category"], schema,
        output_items=["category", AggregateSpec("COUNT", "*")],
        memory_records=1, partition_count=2,
    )

    result = list(run_plan(node))

    assert [(row[0].data, row[1].data) for row in result] == [("same", 5)]