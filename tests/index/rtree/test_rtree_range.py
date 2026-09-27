from spatial.geometry import BoundingBox, Point2D
from storage.heap.rid import RID


def test_range_query_returns_all_points_and_includes_rectangle_edges(rtree):
    points = [
        (Point2D(-2, 0), RID(1, 0)),
        (Point2D(0, 0), RID(2, 0)),
        (Point2D(2, 0), RID(3, 0)),
        (Point2D(0, 2), RID(4, 0)),
        (Point2D(0, 2.01), RID(5, 0)),
    ]
    for point, rid in points:
        rtree.insert(point, rid)

    assert set(rtree.search_bbox(Point2D(-2, 0), Point2D(2, 2))) == {
        RID(1, 0), RID(2, 0), RID(3, 0), RID(4, 0)
    }
    assert set(rtree.range_query(BoundingBox(-2, 0, 2, 2))) == {
        RID(1, 0), RID(2, 0), RID(3, 0), RID(4, 0)
    }


def test_range_query_filters_subtree_mbr_false_positives(rtree):
    points = [
        (Point2D(-10, -10), RID(1, 0)),
        (Point2D(-10, 10), RID(2, 0)),
        (Point2D(10, -10), RID(3, 0)),
        (Point2D(10, 10), RID(4, 0)),
        (Point2D(0, 0), RID(5, 0)),
    ]
    for point, rid in points:
        rtree.insert(point, rid)

    assert rtree.range_query(BoundingBox(-1, -1, 1, 1)) == [RID(5, 0)]


def test_range_query_on_empty_tree_returns_no_rids(rtree):
    assert rtree.range_query(BoundingBox(-1, -1, 1, 1)) == []