"""Euclidean and great-circle distance metrics for spatial points."""

import math

from .geometry import Point2D


EARTH_MEAN_RADIUS_METERS = 6_371_008.8


def _require_point(point: Point2D) -> None:
    if not isinstance(point, Point2D):
        raise TypeError("la distancia requiere valores Point2D")


def euclidean_distance(first: Point2D, second: Point2D) -> float:
    """Return planar Euclidean distance in coordinate units (degrees)."""
    _require_point(first)
    _require_point(second)
    return math.hypot(second.x - first.x, second.y - first.y)


def haversine_distance(
    first: Point2D,
    second: Point2D,
    earth_radius_meters: float = EARTH_MEAN_RADIUS_METERS,
) -> float:
    """Return great-circle distance in meters using the shortest longitude arc."""
    _require_point(first)
    _require_point(second)
    if isinstance(earth_radius_meters, bool) or not isinstance(
        earth_radius_meters, (int, float)
    ):
        raise ValueError("earth_radius_meters debe ser positivo y finito")
    earth_radius_meters = float(earth_radius_meters)
    if not math.isfinite(earth_radius_meters) or earth_radius_meters <= 0:
        raise ValueError("earth_radius_meters debe ser positivo y finito")

    first_latitude = math.radians(first.y)
    second_latitude = math.radians(second.y)
    latitude_delta = second_latitude - first_latitude
    longitude_delta = math.radians(second.x - first.x)
    longitude_delta = (longitude_delta + math.pi) % (2 * math.pi) - math.pi

    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude)
        * math.cos(second_latitude)
        * math.sin(longitude_delta / 2) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    central_angle = 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))
    return earth_radius_meters * central_angle