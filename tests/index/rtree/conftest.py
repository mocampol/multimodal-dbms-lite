import pytest

from index.rtree import RTree
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager


@pytest.fixture
def rtree(tmp_path):
    file_path = tmp_path / "rtree.idx"
    buffer_manager = BufferManager(
        FileManager(str(file_path), page_size=128),
        pool_size=8,
    )
    return RTree(buffer_manager)