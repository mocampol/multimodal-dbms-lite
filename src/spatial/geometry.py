"""Planar geometry primitives for geographic coordinates in degrees.

Point2D.x is longitude and Point2D.y is latitude. Bounding boxes are ordinary
ordered planar MBRs and do not wrap across the antimeridian; represent a
dateline-crossing region with two boxes. Polygon edges use direct segments in
the longitude/latitude plane.
"""

from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Real
from typing import Iterable


def _coordinate(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} debe ser un número finito")
    try:
        coordinate = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} debe ser un número finito") from None
    if not math.isfinite(coordinate):
        raise ValueError(f"{name} debe ser un número finito")
    return coordinate


@dataclass(frozen=True)
class Point2D:
    """A finite geographic point: x is longitude, y is latitude, in degrees."""

    x: float
    y: float

    def __post_init__(self):
        x_coordinate = _coordinate(self.x, "longitude")
        y_coordinate = _coordinate(self.y, "latitude")
        if not -180 <= x_coordinate <= 180:
            raise ValueError("longitude debe estar entre -180 y 180 grados")
        if not -90 <= y_coordinate <= 90:
            raise ValueError("latitude debe estar entre -90 y 90 grados")
        object.__setattr__(self, "x", x_coordinate)
        object.__setattr__(self, "y", y_coordinate)


def points_equal(first: Point2D, second: Point2D) -> bool:
    """Return whether two points have exactly equal coordinates."""
    if not isinstance(first, Point2D) or not isinstance(second, Point2D):
        raise TypeError("points_equal requiere dos Point2D")
    return first == second


@dataclass(frozen=True)
class BoundingBox:
    """An inclusive, non-wrapping minimum bounding rectangle (MBR)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self):
        min_x = _coordinate(self.min_x, "min_x")
        min_y = _coordinate(self.min_y, "min_y")
        max_x = _coordinate(self.max_x, "max_x")
        max_y = _coordinate(self.max_y, "max_y")
        Point2D(min_x, min_y)
        Point2D(max_x, max_y)
        if min_x > max_x or min_y > max_y:
            raise ValueError("los límites mínimos no pueden superar los máximos")
        object.__setattr__(self, "min_x", min_x)
        object.__setattr__(self, "min_y", min_y)
        object.__setattr__(self, "max_x", max_x)
        object.__setattr__(self, "max_y", max_y)

    @classmethod
    def from_points(cls, points: Iterable[Point2D]) -> "BoundingBox":
        points = tuple(points)
        if not points:
            raise ValueError("se requiere al menos un punto para crear el MBR")
        if any(not isinstance(point, Point2D) for point in points):
            raise TypeError("from_points requiere Point2D")
        return cls(
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    @property
    def area(self) -> float:
        """Return the planar area in square degrees."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def contains(self, point: Point2D) -> bool:
        if not isinstance(point, Point2D):
            raise TypeError("contains requiere un Point2D")
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
        )

    def expand(self, point: Point2D) -> "BoundingBox":
        """Return the smallest MBR containing this box and the point."""
        if not isinstance(point, Point2D):
            raise TypeError("expand requiere un Point2D")
        return BoundingBox(
            min(self.min_x, point.x),
            min(self.min_y, point.y),
            max(self.max_x, point.x),
            max(self.max_y, point.y),
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Return the smallest MBR containing both boxes."""
        if not isinstance(other, BoundingBox):
            raise TypeError("union requiere un BoundingBox")
        return BoundingBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Return whether the boxes overlap or touch at their boundaries."""
        if not isinstance(other, BoundingBox):
            raise TypeError("intersects requiere un BoundingBox")
        return not (
            self.max_x < other.min_x
            or other.max_x < self.min_x
            or self.max_y < other.min_y
            or other.max_y < self.min_y
        )

    def intersects_polygon(self, polygon: "Polygon") -> bool:
        return intersects_mbr_polygon(self, polygon)


MBR = BoundingBox


@dataclass(frozen=True)
class Polygon:
    """A closed ring of geographic points, represented by its vertices."""

    points: tuple[Point2D, ...]

    def __post_init__(self):
        try:
            points = tuple(self.points)
        except TypeError:
            raise ValueError("polygon debe ser una secuencia de puntos") from None
        if len(points) < 4 or any(not isinstance(point, Point2D) for point in points):
            raise ValueError("polygon requiere tres vértices y el punto de cierre")
        if not points_equal(points[0], points[-1]):
            raise ValueError("polygon debe repetir el primer punto al final")
        if len(set(points[:-1])) < 3:
            raise ValueError("polygon requiere al menos tres vértices distintos")
        if any(
            abs(first.x - second.x) > 180
            for first, second in zip(points, points[1:])
        ):
            raise ValueError("polygon no puede cruzar el antimeridiano")
        object.__setattr__(self, "points", points)

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox.from_points(self.points)

    def contains(self, point: Point2D) -> bool:
        return point_in_polygon(point, self)

    def intersects(self, bounding_box: BoundingBox) -> bool:
        return intersects_mbr_polygon(bounding_box, self)


def _cross(first: Point2D, second: Point2D, third: Point2D) -> float:
    return (second.x - first.x) * (third.y - first.y) - (
        second.y - first.y
    ) * (third.x - first.x)


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D) -> bool:
    return (
        _cross(start, end, point) == 0
        and min(start.x, end.x) <= point.x <= max(start.x, end.x)
        and min(start.y, end.y) <= point.y <= max(start.y, end.y)
    )


def point_in_polygon(point: Point2D, polygon: Polygon) -> bool:
    """Return whether the point is inside or on the polygon boundary."""
    if not isinstance(point, Point2D) or not isinstance(polygon, Polygon):
        raise TypeError("point_in_polygon requiere Point2D y Polygon")
    if not polygon.bounding_box.contains(point):
        return False

    inside = False
    for start, end in zip(polygon.points, polygon.points[1:]):
        if _point_on_segment(point, start, end):
            return True
        if (start.y > point.y) != (end.y > point.y):
            crossing_x = start.x + (point.y - start.y) * (end.x - start.x) / (
                end.y - start.y
            )
            if point.x < crossing_x:
                inside = not inside
    return inside


def _orientation(first: Point2D, second: Point2D, third: Point2D) -> int:
    cross_product = _cross(first, second, third)
    return (cross_product > 0) - (cross_product < 0)


def _segments_intersect(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> bool:
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)

    if (
        first_orientation * second_orientation < 0
        and third_orientation * fourth_orientation < 0
    ):
        return True
    return (
        (first_orientation == 0 and _point_on_segment(second_start, first_start, first_end))
        or (second_orientation == 0 and _point_on_segment(second_end, first_start, first_end))
        or (third_orientation == 0 and _point_on_segment(first_start, second_start, second_end))
        or (fourth_orientation == 0 and _point_on_segment(first_end, second_start, second_end))
    )


def intersects_mbr_polygon(bounding_box: BoundingBox, polygon: Polygon) -> bool:
    """Return whether an MBR and polygon overlap or touch at any boundary."""
    if not isinstance(bounding_box, BoundingBox) or not isinstance(polygon, Polygon):
        raise TypeError("intersects_mbr_polygon requiere BoundingBox y Polygon")
    if not bounding_box.intersects(polygon.bounding_box):
        return False

    if any(bounding_box.contains(point) for point in polygon.points):
        return True

    box_corners = (
        Point2D(bounding_box.min_x, bounding_box.min_y),
        Point2D(bounding_box.max_x, bounding_box.min_y),
        Point2D(bounding_box.max_x, bounding_box.max_y),
        Point2D(bounding_box.min_x, bounding_box.max_y),
    )
    if any(point_in_polygon(corner, polygon) for corner in box_corners):
        return True

    box_edges = tuple(zip(box_corners, box_corners[1:] + box_corners[:1]))
    polygon_edges = zip(polygon.points, polygon.points[1:])
    return any(
        _segments_intersect(box_start, box_end, polygon_start, polygon_end)
        for box_start, box_end in box_edges
        for polygon_start, polygon_end in polygon_edges
    )