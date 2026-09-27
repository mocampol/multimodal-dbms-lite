"""Exact predicates applied after spatial candidate access."""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode
from query.executor.spatial_utils import (
    column_index,
    point2d_from_expression,
    point2d_from_value,
    polygon_from_literal,
)
from query.parser.ast_nodes import (
    BinaryOp,
    DistanceExp,
    IdExp,
    NumExp,
    SpatialPredicate,
    WithinExp,
)
from spatial.distance import euclidean_distance, haversine_distance
from spatial.geometry import point_in_polygon


class SpatialFilter(PlanNode):
    """Evaluate spatial conditions exactly, including after an R-Tree scan."""

    def __init__(self, child: PlanNode, predicate: SpatialPredicate, schema):
        self.child = child
        self.predicate = predicate
        self.schema = schema
        if isinstance(predicate, WithinExp):
            self.geometry_index = column_index(schema, predicate.geometry.value)
            self.polygon = polygon_from_literal(predicate.polygon)
            self.distance = None
        else:
            self.distance = predicate.left
            self.geometry_index = column_index(schema, self.distance.geometry.value)
            self.polygon = None

    def open(self) -> None:
        self.child.open()

    def next(self) -> Optional[Record]:
        while True:
            record = self.child.next()
            if record is None:
                return None
            if self._matches(record):
                return record

    def close(self) -> None:
        self.child.close()

    def children(self) -> list[PlanNode]:
        return [self.child]

    def replace_children(self, new_children: list[PlanNode]) -> None:
        (self.child,) = new_children

    def describe_self(self) -> dict:
        return {"node": "SpatialFilter", "predicate": repr(self.predicate)}

    def _matches(self, record: Record) -> bool:
        geometry = point2d_from_value(record[self.geometry_index].data)
        if geometry is None:
            return False

        if isinstance(self.predicate, WithinExp):
            return point_in_polygon(geometry, self.polygon)

        target = point2d_from_expression(self.distance.point, record, self.schema)
        if target is None:
            return False
        actual_distance = (
            euclidean_distance(geometry, target)
            if self.distance.metric.upper() == "EUCLIDEAN"
            else haversine_distance(geometry, target)
        )
        threshold = self._value(self.predicate.right, record)
        if threshold is None:
            return False

        operations = {
            BinaryOp.EQ_OP: lambda: actual_distance == threshold,
            BinaryOp.NEQ_OP: lambda: actual_distance != threshold,
            BinaryOp.LE_OP: lambda: actual_distance < threshold,
            BinaryOp.LEQ_OP: lambda: actual_distance <= threshold,
            BinaryOp.GT_OP: lambda: actual_distance > threshold,
            BinaryOp.GEQ_OP: lambda: actual_distance >= threshold,
        }
        return operations[self.predicate.operator]()

    def _value(self, expression, record):
        if isinstance(expression, NumExp):
            return expression.value
        if isinstance(expression, IdExp):
            return record[column_index(self.schema, expression.value)].data
        return None