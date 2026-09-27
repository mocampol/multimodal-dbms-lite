from spatial.geometry import BoundingBox, Point2D
from storage.heap.rid import RID


def test_delete_removes_only_the_requested_duplicate_rid(rtree):
    point = Point2D(12.5, -8.25)
    first_rid = RID(5, 1)
    second_rid = RID(5, 2)
    rtree.insert(point, first_rid)
    rtree.insert(point, second_rid)

    assert rtree.delete(point, first_rid)
    assert rtree.search_point(point) == [second_rid]
    assert not rtree.delete(point, first_rid)


def test_delete_updates_mbrs_and_prunes_empty_nodes(rtree):
    entries = [(Point2D(index - 30, index % 13 - 6), RID(index, 0)) for index in range(60)]
    for point, rid in entries:
        rtree.insert(point, rid)

    for point, rid in entries:
        assert rtree.delete(point, rid)

    assert rtree.range_query(BoundingBox(-180, -90, 180, 90)) == []
    for point, rid in entries[:10]:
        rtree.insert(point, rid)
    assert set(rtree.range_query(BoundingBox(-180, -90, 180, 90))) == {
        rid for _, rid in entries[:10]
    }


def test_delete_requires_both_point_and_rid(rtree):
    point = Point2D(1, 2)
    rid = RID(7, 3)
    rtree.insert(point, rid)

    assert not rtree.delete(Point2D(1, 2.0001), rid)
    assert not rtree.delete(point, RID(7, 4))
    assert rtree.search_point(point) == [rid]