from storage.buffer_manager import BufferManager
from common.value import DataType, Value

from .node import BTreeNode


class ClusteredIndex:
    def __init__(self, key_type: DataType, buffer_manager: BufferManager, sequential_file):
        self.key_type = key_type
        self.bm = buffer_manager
        self.sequential_file = sequential_file

        self._root_page_id: int | None = None
        self._height: int = 0

        self.build()

    def build(self):
        entries = list(self.sequential_file.page_min_keys())
        if not entries:
            raise ValueError(
                "No se puede construir el índice agrupado: "
                "el Sequential File no tiene páginas"
            )

        level = entries
        height = 0
        while True:
            level = self._build_level(level)
            height += 1
            if len(level) == 1:
                break

        self._root_page_id = level[0][1]
        self._height = height

    def rebuild(self):
        self.build()

    def search(self, key: Value) -> list:
        page_id = self._find_sequential_page(key)
        return self.sequential_file.search_in_page(page_id, key.data)

    def range(self, start_key: Value, end_key: Value) -> list:
        start_page_id = self._find_sequential_page(start_key)
        results = []
        for record in self.sequential_file.scan_from(start_page_id):
            key = record[self.sequential_file._key_index].data
            if key < start_key.data:
                continue
            if key > end_key.data:
                break
            results.append(record)
        return results

    @property
    def root_page_id(self) -> int:
        return self._root_page_id

    @property
    def height(self) -> int:
        return self._height

    def _find_sequential_page(self, key: Value) -> int:
        page_id = self._root_page_id
        for _ in range(self._height):
            page = self.bm.fetch_page(page_id)
            node = BTreeNode(page, self.key_type)
            child_idx = node.find_child_index(key)
            next_page_id = node.children()[child_idx]
            self.bm.unpin_page(page_id, is_dirty=False)
            page_id = next_page_id
        return page_id

    def _build_level(self, entries: list) -> list:
        result = []
        i, n = 0, len(entries)

        while i < n:
            page_id = self.bm.allocate_page()
            page = self.bm.fetch_page(page_id)
            node = BTreeNode.init_internal(page, self.key_type)

            node.set_child(0, entries[i][1])
            representative_key = entries[i][0]

            j = i + 1
            while j < n and node.has_room_for(entries[j][0]):
                node.insert_internal_entry(entries[j][0], entries[j][1])
                j += 1

            self.bm.unpin_page(page_id, is_dirty=True)
            result.append((representative_key, page_id))
            i = j

        return result
