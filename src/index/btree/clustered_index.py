from common.value import DataType, Value
from storage.buffer_manager import BufferManager
from storage.heap.rid import RID

from .btree import BTree


class ClusteredIndex:
    """Incrementally maintained B+ index over SequentialFile page minima."""

    def __init__(self, key_type: DataType, buffer_manager: BufferManager, sequential_file):
        self.key_type = key_type
        self.bm = buffer_manager
        self.sequential_file = sequential_file
        self._tree = None
        self._page_minima: dict[int, object] = {}
        self._syncing = False
        self.sync()

    @property
    def root_page_id(self):
        return self._tree.root_page_id if self._tree is not None else None

    @property
    def height(self):
        return 0 if self._tree is None else 1

    def build(self):
        self.sync()

    def rebuild(self):
        self.sync()

    def sync(self):
        if self._syncing:
            return
        self._syncing = True
        try:
            current = {
                page_id: key
                for page_id, key in self.sequential_file.page_min_keys()
                if key is not None
            }
            if self._tree is None and current:
                if self.bm.file_manager.page_count() > 0:
                    self.bm.reset()
                self._tree = BTree(self.key_type, self.bm)

            for page_id, old_key in list(self._page_minima.items()):
                if page_id not in current:
                    self._tree.delete(Value(self.key_type, old_key), RID(page_id, 0))
                    del self._page_minima[page_id]

            for page_id, new_key in current.items():
                old_key = self._page_minima.get(page_id)
                if old_key == new_key:
                    continue
                if old_key is not None:
                    self._tree.delete(Value(self.key_type, old_key), RID(page_id, 0))
                self._tree.insert(Value(self.key_type, new_key), RID(page_id, 0))
                self._page_minima[page_id] = new_key
        finally:
            self._syncing = False

    def search(self, key: Value) -> list:
        if self._tree is None:
            return self.sequential_file.search(key.data)
        entry = self._tree.predecessor(key)
        if entry is None:
            return []
        return self.sequential_file.search_in_page(entry.page_id, key.data)

    def range(self, start_key: Value, end_key: Value) -> list:
        return [
            record
            for record in self.sequential_file.scan()
            if start_key.data <= record[self.sequential_file._key_index].data <= end_key.data
        ]
