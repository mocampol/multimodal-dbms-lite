import pytest

from spatial.distance import euclidean_distance, haversine_distance
from spatial.geometry import Point2D


def test_euclidean_distance_uses_coordinate_units():
    assert euclidean_distance(Point2D(0, 0), Point2D(3, 4)) == 5
    assert euclidean_distance(Point2D(3, 4), Point2D(0, 0)) == 5


def test_haversine_identical_point_is_zero():
    point = Point2D(-122.3, 47.6)
    assert haversine_distance(point, point) == pytest.approx(0, abs=1e-9)


def test_haversine_matches_known_city_distance():
    london = Point2D(-0.1278, 51.5074)
    paris = Point2D(2.3522, 48.8566)
    assert haversine_distance(london, paris) == pytest.approx(343_556, abs=150)


def test_haversine_uses_short_arc_across_antimeridian():
    west = Point2D(179.9, 0)
    east = Point2D(-179.9, 0)
    assert haversine_distance(west, east) == pytest.approx(22_239, abs=2)


def test_haversine_validates_earth_radius():
    point = Point2D(0, 0)
    with pytest.raises(ValueError):
        haversine_distance(point, point, earth_radius_meters=0)