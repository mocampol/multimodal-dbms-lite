from index.rtree import NodeType, RTree
from spatial.geometry import BoundingBox, Point2D
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.rid import RID


def test_insert_points_through_multiple_leaf_and_internal_splits(rtree):
    entries = [(Point2D(index - 60, index % 17 - 8), RID(index, 0)) for index in range(120)]
    for point, rid in entries:
        rtree.insert(point, rid)

    assert set(rtree.range_query(BoundingBox(-180, -90, 180, 90))) == {
        rid for _, rid in entries
    }
    root = rtree._fetch_node(rtree.root_page_id)
    assert root.node_type == NodeType.INTERNAL
    rtree.bm.unpin_page(rtree.root_page_id, is_dirty=False)


def test_insert_allows_repeated_points_and_preserves_each_rid(rtree):
    point = Point2D(-73.98, 40.75)
    expected_rids = [RID(10, 0), RID(11, 0), RID(10, 1)]
    for rid in expected_rids:
        rtree.insert(point, rid)

    assert set(rtree.search_point(point)) == set(expected_rids)


def test_inserted_points_survive_flush_and_reopen_with_catalog_root(tmp_path):
    file_path = tmp_path / "reopened-rtree.idx"
    first_manager = BufferManager(
        FileManager(str(file_path), page_size=128),
        pool_size=8,
    )
    tree = RTree(first_manager)
    inserted = [(Point2D(index - 25, index % 11 - 5), RID(index, 2)) for index in range(50)]
    for point, rid in inserted:
        tree.insert(point, rid)
    catalog_root_page_id = tree.root_page_id
    first_manager.flush_all()

    second_manager = BufferManager(
        FileManager(str(file_path), page_size=128),
        pool_size=8,
    )
    reopened = RTree(second_manager, root_page_id=catalog_root_page_id)
    assert set(reopened.range_query(BoundingBox(-180, -90, 180, 90))) == {
        rid for _, rid in inserted
    }