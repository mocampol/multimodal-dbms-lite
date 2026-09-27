"""R-Tree candidate scans and exact distance ordering."""

import math
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
    NumExp,
    PointExp,
    SpatialPredicate,
    WithinExp,
)
from spatial.distance import EARTH_MEAN_RADIUS_METERS, euclidean_distance, haversine_distance
from spatial.geometry import BoundingBox, Point2D
from storage.latch import NO_LATCH


WORLD_BOUNDS = BoundingBox(-180, -90, 180, 90)


def distance_bounding_boxes(center: Point2D, radius: float, metric: str) -> list[BoundingBox]:
    """Return conservative geographic MBRs for a distance-radius search."""
    if not math.isfinite(radius) or radius < 0:
        return [WORLD_BOUNDS]

    if metric.upper() == "EUCLIDEAN":
        west = max(-180.0, center.x - radius)
        east = min(180.0, center.x + radius)
        south = max(-90.0, center.y - radius)
        north = min(90.0, center.y + radius)
        return [BoundingBox(west, south, east, north)]

    angular_radius = radius / EARTH_MEAN_RADIUS_METERS
    latitude = math.radians(center.y)
    south = max(-90.0, center.y - math.degrees(angular_radius))
    north = min(90.0, center.y + math.degrees(angular_radius))
    if angular_radius >= math.pi or angular_radius >= math.pi / 2 or (
        abs(latitude) + angular_radius >= math.pi / 2
    ):
        return [BoundingBox(-180, south, 180, north)]

    longitude_radius = math.degrees(
        math.asin(min(1.0, math.sin(angular_radius) / math.cos(latitude)))
    )
    west = center.x - longitude_radius
    east = center.x + longitude_radius
    if west < -180:
        return [
            BoundingBox(-180, south, east, north),
            BoundingBox(west + 360, south, 180, north),
        ]
    if east > 180:
        return [
            BoundingBox(west, south, 180, north),
            BoundingBox(-180, south, east - 360, north),
        ]
    return [BoundingBox(west, south, east, north)]


def candidate_boxes_for_predicate(predicate, schema) -> list[BoundingBox]:
    if isinstance(predicate, WithinExp):
        return [polygon_from_literal(predicate.polygon).bounding_box]

    if not isinstance(predicate, SpatialPredicate) or not isinstance(
        predicate.left, DistanceExp
    ):
        return [WORLD_BOUNDS]

    distance = predicate.left
    if (
        predicate.operator not in (BinaryOp.LE_OP, BinaryOp.LEQ_OP, BinaryOp.EQ_OP)
        or not isinstance(distance.point, PointExp)
        or not isinstance(predicate.right, NumExp)
    ):
        return [WORLD_BOUNDS]

    center = point2d_from_expression(distance.point, None, schema)
    if center is None:
        return [WORLD_BOUNDS]
    return distance_bounding_boxes(
        center,
        float(predicate.right.value),
        distance.metric,
    )


class SpatialIndexScan(PlanNode):
    """Read R-Tree candidate RIDs for one or more bounding boxes.

    This node deliberately does not make final spatial decisions. Its child
    must be wrapped in SpatialFilter to remove MBR false positives.
    """

    def __init__(
        self,
        index,
        storage,
        table_name: str,
        boxes: list[BoundingBox],
        latch=NO_LATCH,
        lock_rid=None,
        include_nulls_column: int | None = None,
        index_id: int | None = None,
        column_name: str | None = None,
    ):
        self.index = index
        self.storage = storage
        self.table_name = table_name
        self.boxes = boxes
        self.latch = latch
        self.lock_rid = lock_rid
        self.include_nulls_column = include_nulls_column
        self.index_id = index_id
        self.column_name = column_name
        self._records = []
        self._cursor = 0

    def open(self) -> None:
        from transaction.lock_manager import LockMode

        with self.latch:
            rids = [
                rid
                for bbox in self.boxes
                for rid in self.index.range_query(bbox)
            ]

        if self.include_nulls_column is not None and hasattr(self.storage, "scan_with_rid"):
            rows = self.storage.scan_with_rid()
            try:
                while True:
                    with self.latch:
                        item = next(rows, None)
                    if item is None:
                        break
                    rid, record = item
                    if record[self.include_nulls_column].data is None:
                        rids.append(rid)
            finally:
                close = getattr(rows, "close", None)
                if close is not None:
                    with self.latch:
                        close()

        self._records = []
        for rid in dict.fromkeys(rids):
            if self.lock_rid is not None:
                self.lock_rid(rid, LockMode.SHARED)
            with self.latch:
                record = self.storage.get(rid)
            if record is not None:
                self._records.append(record)
        self._cursor = 0

    def next(self) -> Optional[Record]:
        if self._cursor >= len(self._records):
            return None
        record = self._records[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self._records = []
        self._cursor = 0

    def describe_self(self) -> dict:
        return {
            "node": "SpatialIndexScan",
            "access": "rtree_candidates",
            "index_type": "rtree",
            "index_id": self.index_id,
            "column_name": self.column_name,
            "table": self.table_name,
            "candidate_boxes": [
                [box.min_x, box.min_y, box.max_x, box.max_y]
                for box in self.boxes
            ],
        }


class KNNScan(PlanNode):
    """Sort source records by exact spatial distance and optional tie keys."""

    def __init__(self, child: PlanNode, order_items: list, schema):
        self.child = child
        self.order_items = order_items
        self.schema = schema
        self._records = []
        self._cursor = 0

    def open(self) -> None:
        self.child.open()
        self._records = []
        while True:
            record = self.child.next()
            if record is None:
                break
            self._records.append(record)
        self._records.sort(key=self._sort_key)
        self._cursor = 0

    def next(self) -> Optional[Record]:
        if self._cursor >= len(self._records):
            return None
        record = self._records[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self.child.close()
        self._records = []
        self._cursor = 0

    def children(self) -> list[PlanNode]:
        return [self.child]

    def replace_children(self, new_children: list[PlanNode]) -> None:
        (self.child,) = new_children

    def describe_self(self) -> dict:
        return {
            "node": "KNNScan",
            "order_by": [repr(item) for item in self.order_items],
            "metric": next(
                (item.metric.upper() for item in self.order_items if isinstance(item, DistanceExp)),
                None,
            ),
            "strategy": "exact_distance_sort",
        }

    def _sort_key(self, record: Record):
        keys = []
        for item in self.order_items:
            if isinstance(item, DistanceExp):
                geometry = point2d_from_value(
                    record[column_index(self.schema, item.geometry.value)].data
                )
                target = point2d_from_expression(item.point, record, self.schema)
                if geometry is None or target is None:
                    keys.append((True, 0.0))
                    continue
                distance = (
                    euclidean_distance(geometry, target)
                    if item.metric.upper() == "EUCLIDEAN"
                    else haversine_distance(geometry, target)
                )
                keys.append((False, distance))
                continue

            value = record[column_index(self.schema, item)].data
            keys.append((value is not None, value if value is not None else 0))
        tie_breaker = tuple(repr(value.data) for value in record.values)
        return tuple(keys) + (tie_breaker,)