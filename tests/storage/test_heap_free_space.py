from common.record import Record
from common.schema import Column, Schema
from common.value import DataType, Value
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.heap_file import HeapFile


def _heap(tmp_path, pool_size=8):
    schema = Schema("items", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("value", DataType.VARCHAR, size=1000),
    ])
    return HeapFile(
        schema,
        BufferManager(FileManager(str(tmp_path / "items.tbl")), pool_size=pool_size),
    )


def _record(identifier, value):
    return Record([
        Value(DataType.INTEGER, identifier),
        Value(DataType.VARCHAR, value),
    ])


def test_deleted_slot_is_reused(tmp_path):
    heap = _heap(tmp_path)
    first = heap.insert(_record(1, "a"))
    second = heap.insert(_record(2, "b"))
    heap.delete(first)

    reused = heap.insert(_record(3, "c"))

    assert reused == first
    assert heap.get(second)[0].data == 2


def test_compaction_reuses_fragmented_page_before_allocating_page(tmp_path):
    heap = _heap(tmp_path)
    rids = [heap.insert(_record(index, "x" * 200)) for index in range(12)]
    page_count_before = heap.bm.file_manager.page_count()
    for rid in rids[::2]:
        heap.delete(rid)

    heap.insert(_record(99, "y" * 700))

    assert heap.bm.file_manager.page_count() == page_count_before


def test_delete_original_rid_removes_forwarded_update_target(tmp_path):
    heap = _heap(tmp_path)
    original = heap.insert(_record(1, "short"))
    heap.update(original, _record(1, "x" * 700))
    heap.delete(original)

    replacement = heap.insert(_record(2, "replacement"))

    assert heap.get(original) is None
    assert heap.get(replacement)[0].data == 2


def test_free_space_and_forwarding_survive_reboot(tmp_path):
    heap = _heap(tmp_path)
    original = heap.insert(_record(1, "short"))
    heap.update(original, _record(1, "x" * 700))
    heap.delete(original)
    heap.bm.flush_all()

    reopened = _heap(tmp_path)

    assert reopened.get(original) is None
    replacement = reopened.insert(_record(2, "reused"))
    reopened.bm.flush_all()

    assert reopened.get(replacement)[0].data == 2