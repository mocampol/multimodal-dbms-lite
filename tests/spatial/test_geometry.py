import pytest

from spatial.geometry import (
    BoundingBox,
    Point2D,
    Polygon,
    intersects_mbr_polygon,
    point_in_polygon,
    points_equal,
)


def square():
    return Polygon(
        (
            Point2D(-1, -1),
            Point2D(1, -1),
            Point2D(1, 1),
            Point2D(-1, 1),
            Point2D(-1, -1),
        )
    )


def test_point_comparison_is_exact():
    assert points_equal(Point2D(10, 20), Point2D(10, 20))
    assert not points_equal(Point2D(10, 20), Point2D(10, 20.0001))


def test_mbr_area_expansion_and_union():
    original = BoundingBox(-1, -2, 2, 3)
    assert original.area == 15
    assert original.expand(Point2D(-2, 4)) == BoundingBox(-2, -2, 2, 4)
    assert original.union(BoundingBox(1, -4, 3, 1)) == BoundingBox(-1, -4, 3, 3)


def test_mbr_intersection_includes_touching_edges():
    first = BoundingBox(0, 0, 1, 1)
    assert first.intersects(BoundingBox(1, 0.25, 2, 0.75))
    assert first.contains(Point2D(1, 1))
    assert not first.intersects(BoundingBox(1.0001, 0, 2, 1))


def test_polygon_contains_inside_outside_and_boundary_points():
    polygon = square()
    assert point_in_polygon(Point2D(0, 0), polygon)
    assert not point_in_polygon(Point2D(2, 0), polygon)
    assert point_in_polygon(Point2D(1, 0), polygon)
    assert polygon.bounding_box == BoundingBox(-1, -1, 1, 1)


def test_polygon_intersects_mbr_on_edge_and_when_contained():
    polygon = square()
    assert intersects_mbr_polygon(BoundingBox(1, -0.5, 2, 0.5), polygon)
    assert intersects_mbr_polygon(BoundingBox(-0.25, -0.25, 0.25, 0.25), polygon)
    assert not intersects_mbr_polygon(BoundingBox(2, 2, 3, 3), polygon)


def test_polygon_rejects_open_or_antimeridian_crossing_ring():
    with pytest.raises(ValueError):
        Polygon((Point2D(0, 0), Point2D(1, 0), Point2D(0, 1)))
    with pytest.raises(ValueError):
        Polygon(
            (
                Point2D(170, 0),
                Point2D(-170, 0),
                Point2D(-170, 1),
                Point2D(170, 0),
            )
        )


def test_point_and_mbr_reject_invalid_geographic_coordinates():
    with pytest.raises(ValueError):
        Point2D(181, 0)
    with pytest.raises(ValueError):
        BoundingBox(0, 1, 1, 0)