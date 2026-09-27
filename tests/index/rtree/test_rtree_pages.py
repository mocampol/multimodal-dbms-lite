import pytest

from index.rtree import (
    InternalEntry,
    LeafEntry,
    NodeType,
    RTree,
    RTreeNode,
    RTreeNodeCapacityError,
    RTreePageCorruptionError,
    RTreePage,
)
from spatial.geometry import BoundingBox
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.rid import RID
from storage.page import Page


def _mbr(index: int) -> BoundingBox:
    return BoundingBox(index, index, index + 0.5, index + 0.5)


def _buffer_manager(path, page_size: int = 128) -> BufferManager:
    return BufferManager(FileManager(str(path), page_size=page_size), pool_size=8)


def test_leaf_and_internal_pages_round_trip():
    leaf_page = Page(page_id=0, size=128)
    leaf = RTreeNode.init_leaf(leaf_page)
    expected_leaf_entries = [LeafEntry(_mbr(1), RID(20, 3)), LeafEntry(_mbr(2), RID(21, 4))]
    for entry in expected_leaf_entries:
        assert leaf.insert_leaf_entry(entry.mbr, entry.rid)

    reloaded_leaf = RTreeNode(Page(page_id=0, size=128, data=bytes(leaf_page.data)))
    assert reloaded_leaf.is_leaf
    assert reloaded_leaf.entries == expected_leaf_entries

    internal_page = Page(page_id=1, size=128)
    internal = RTreeNode.init_internal(internal_page)
    expected_internal_entries = [InternalEntry(_mbr(3), 8), InternalEntry(_mbr(4), 9)]
    for entry in expected_internal_entries:
        assert internal.insert_internal_entry(entry.mbr, entry.child_page_id)

    reloaded_internal = RTreeNode(
        Page(page_id=1, size=128, data=bytes(internal_page.data))
    )
    assert not reloaded_internal.is_leaf
    assert reloaded_internal.entries == expected_internal_entries


def test_capacity_is_derived_from_page_size_and_never_overflows():
    page = Page(page_id=0, size=128)
    leaf = RTreeNode.init_leaf(page)
    assert leaf.capacity == 2
    assert leaf.insert_leaf_entry(_mbr(0), RID(0, 0))
    assert leaf.insert_leaf_entry(_mbr(1), RID(1, 0))
    assert not leaf.insert_leaf_entry(_mbr(2), RID(2, 0))
    assert len(page.data) == page.size == 128
    with pytest.raises(RTreeNodeCapacityError):
        RTreePage.serialize(
            page,
            NodeType.LEAF,
            leaf.entries + [LeafEntry(_mbr(3), RID(3, 0))],
        )


def test_split_conserves_existing_and_pending_leaf_entries():
    left_page = Page(page_id=0, size=128)
    node = RTreeNode.init_leaf(left_page)
    existing = [LeafEntry(_mbr(index), RID(index, 0)) for index in range(2)]
    pending = LeafEntry(_mbr(2), RID(2, 0))
    for entry in existing:
        assert node.insert_leaf_entry(entry.mbr, entry.rid)

    right_node = node.split(Page(page_id=1, size=128), pending)
    combined_entries = node.entries + right_node.entries
    assert {entry.rid for entry in combined_entries} == {RID(0, 0), RID(1, 0), RID(2, 0)}
    assert len(node.entries) <= node.capacity
    assert len(right_node.entries) <= right_node.capacity


def test_internal_split_conserves_existing_and_pending_children():
    node = RTreeNode.init_internal(Page(page_id=0, size=128))
    existing = [InternalEntry(_mbr(index), 10 + index) for index in range(2)]
    pending = InternalEntry(_mbr(2), 12)
    for entry in existing:
        assert node.insert_internal_entry(entry.mbr, entry.child_page_id)

    right_node = node.split(Page(page_id=1, size=128), pending)
    combined_children = {
        entry.child_page_id for entry in node.entries + right_node.entries
    }
    assert combined_children == {10, 11, 12}
    assert all(isinstance(entry, InternalEntry) for entry in node.entries + right_node.entries)


def test_corrupt_page_checksum_is_rejected():
    page = Page(page_id=4, size=128)
    RTreeNode.init_leaf(page)
    page.data[0] ^= 0x01
    with pytest.raises(RTreePageCorruptionError, match="Checksum"):
        RTreeNode(page)


def test_nodes_and_promoted_root_survive_flush_and_reboot(tmp_path):
    path = tmp_path / "rtree.idx"
    first_manager = _buffer_manager(path)
    tree = RTree(first_manager)
    old_root_page_id = tree.root_page_id
    root = tree.fetch_node(old_root_page_id)
    for index in range(2):
        assert root.insert_leaf_entry(_mbr(index), RID(40 + index, index))
    first_manager.unpin_page(old_root_page_id, is_dirty=True)

    pending = LeafEntry(_mbr(2), RID(42, 2))
    sibling_page_id = tree.split_node(old_root_page_id, pending)
    persisted_root_page_id = tree.root_page_id
    assert persisted_root_page_id != old_root_page_id
    first_manager.flush_all()

    second_manager = _buffer_manager(path)
    reopened_tree = RTree(second_manager, root_page_id=persisted_root_page_id)
    reopened_root = reopened_tree.fetch_node(persisted_root_page_id)
    assert not reopened_root.is_leaf
    root_entries = reopened_root.entries
    child_page_ids = {entry.child_page_id for entry in root_entries}
    assert sibling_page_id in child_page_ids

    child_nodes = [reopened_tree.fetch_node(page_id) for page_id in child_page_ids]
    stored_rids = {
        entry.rid
        for child in child_nodes
        for entry in child.entries
        if isinstance(entry, LeafEntry)
    }
    assert stored_rids == {RID(40, 0), RID(41, 1), RID(42, 2)}

    for child_page_id in child_page_ids:
        second_manager.unpin_page(child_page_id, is_dirty=False)
    second_manager.unpin_page(persisted_root_page_id, is_dirty=False)


def test_corruption_is_detected_after_disk_reload(tmp_path):
    path = tmp_path / "corrupt-rtree.idx"
    first_manager = _buffer_manager(path)
    tree = RTree(first_manager)
    root_page_id = tree.root_page_id
    page = first_manager.fetch_page(root_page_id)
    page.write_bytes(0, b"BAD!")
    first_manager.unpin_page(root_page_id, is_dirty=True)
    first_manager.flush_all()

    second_manager = _buffer_manager(path)
    with pytest.raises(RTreePageCorruptionError):
        RTree(second_manager, root_page_id=root_page_id)