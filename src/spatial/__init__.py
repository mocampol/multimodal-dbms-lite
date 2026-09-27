"""2D spatial primitives and distance metrics, independent of indexes."""

from .distance import EARTH_MEAN_RADIUS_METERS, euclidean_distance, haversine_distance
from .geometry import (
    MBR,
    BoundingBox,
    Point2D,
    Polygon,
    intersects_mbr_polygon,
    point_in_polygon,
    points_equal,
)

__all__ = [
    "EARTH_MEAN_RADIUS_METERS",
    "MBR",
    "BoundingBox",
    "Point2D",
    "Polygon",
    "euclidean_distance",
    "haversine_distance",
    "intersects_mbr_polygon",
    "point_in_polygon",
    "points_equal",
]