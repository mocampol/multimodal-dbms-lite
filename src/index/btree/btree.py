import bisect

from storage.buffer_manager.buffer_manager import BufferManager
from storage.heap.rid import RID
from common.value import DataType, Value

from .node import (
    BTreeNode, NodeType, NO_SIBLING,
    HEADER_SIZE, SLOT_SIZE, CHILD_SIZE, RID_SIZE,
)
from .key_codec import key_size
from .exceptions import DuplicateKeyError, KeyNotFoundError, IndexEntryTooLargeError, IndexCorruptionError


class BTree:
    def __init__(
        self,
        key_type: DataType,
        buffer_manager: BufferManager,
        root_page_id: int = None,
        unique: bool = False,
    ):
        self.key_type = key_type
        self.bm = buffer_manager
        self.unique = unique

        if root_page_id is not None:
            self._root_page_id = root_page_id
        elif self.bm.file_manager.page_count() == 0:
            self._root_page_id = self._allocate_and_init_root()
        else:
            raise ValueError(
                "El archivo del índice ya existe pero no se indicó root_page_id "
                "— léelo desde sys_indexes.root_page_id antes de reabrir este índice"
            )

    @property
    def root_page_id(self) -> int:
        return self._root_page_id

    def search(self, key: Value) -> RID | None:
        leaf = self._find_leaf(key)
        try:
            entries = leaf.entries()
            idx = leaf.find_entry_index(key)
            if idx < len(entries) and entries[idx][0].data == key.data:
                return entries[idx][1]
            return None
        finally:
            self.bm.unpin_page(leaf.page.page_id, is_dirty=False)

    def range(self, start_key: Value, end_key: Value) -> list[tuple]:
        results = []
        leaf = self._find_leaf(start_key)
        page_id = leaf.page.page_id
        idx = leaf.find_entry_index(start_key)

        while True:
            entries = leaf.entries()
            for key, rid in entries[idx:]:
                if key.data > end_key.data:
                    self.bm.unpin_page(page_id, is_dirty=False)
                    return results
                results.append((key, rid))

            next_page_id = leaf.next_leaf_page_id()
            self.bm.unpin_page(page_id, is_dirty=False)
            if next_page_id is None:
                return results

            page_id = next_page_id
            page = self.bm.fetch_page(page_id)
            leaf = BTreeNode(page, self.key_type)
            idx = 0

    def insert(self, key: Value, rid: RID):
        entry_size = key_size(key) + RID_SIZE + SLOT_SIZE
        max_capacity = self.bm.file_manager.page_size - HEADER_SIZE
        if entry_size > max_capacity:
            raise IndexEntryTooLargeError(
                f"La clave {key.data!r} ({entry_size} bytes) no cabe en una "
                "sola página de índice; claves de este tamaño no están "
                "soportadas por este B+ Tree"
            )

        old_root_id = self._root_page_id
        result = self._insert_recursive(old_root_id, key, rid)
        if result is not None:
            split_key, new_right_page_id = result
            self._create_new_root(split_key, old_root_id, new_right_page_id)

    def delete(self, key: Value):
        self._delete_recursive(self._root_page_id, key)
        self._shrink_root_if_needed()

    def _find_leaf(self, key: Value) -> BTreeNode:
        page_id = self._root_page_id
        while True:
            page = self.bm.fetch_page(page_id)
            node = BTreeNode(page, self.key_type)
            if node.is_leaf():
                return node
            child_idx = node.find_child_index(key)
            next_page_id = node.children()[child_idx]
            self.bm.unpin_page(page_id, is_dirty=False)
            page_id = next_page_id

    def _insert_recursive(self, page_id: int, key: Value, rid: RID):
        page = self.bm.fetch_page(page_id)
        node = BTreeNode(page, self.key_type)

        if node.is_leaf():
            if self.unique:
                entries = node.entries()
                idx = node.find_entry_index(key)
                if idx < len(entries) and entries[idx][0].data == key.data:
                    self.bm.unpin_page(page_id, is_dirty=False)
                    raise DuplicateKeyError(
                        f"La clave {key.data!r} ya existe en el índice"
                    )

            if node.has_room_for(key):
                node.insert_leaf_entry(key, rid)
                self.bm.unpin_page(page_id, is_dirty=True)
                return None

            result = self._split_leaf(node, key, rid)
            self.bm.unpin_page(page_id, is_dirty=True)
            return result

        child_idx = node.find_child_index(key)
        child_page_id = node.children()[child_idx]
        self.bm.unpin_page(page_id, is_dirty=False)

        child_result = self._insert_recursive(child_page_id, key, rid)
        if child_result is None:
            return None

        child_split_key, new_child_page_id = child_result

        page = self.bm.fetch_page(page_id)
        node = BTreeNode(page, self.key_type)
        if node.has_room_for(child_split_key):
            node.insert_internal_entry(child_split_key, new_child_page_id)
            self.bm.unpin_page(page_id, is_dirty=True)
            return None

        result = self._split_internal(node, child_split_key, new_child_page_id)
        self.bm.unpin_page(page_id, is_dirty=True)
        return result

    def _split_leaf(self, node: BTreeNode, new_key: Value, new_rid: RID) -> tuple:
        existing = node.entries()
        combined = sorted(existing + [(new_key, new_rid)], key=lambda e: e[0].data)

        split_idx = self._find_leaf_split_point(combined)
        left_entries, right_entries = combined[:split_idx], combined[split_idx:]

        old_next = self._raw_next(node)
        right_page_id = self.bm.allocate_page()

        self._rebuild_leaf(node.page, left_entries, next_leaf_page_id=right_page_id)

        right_page = self.bm.fetch_page(right_page_id)
        self._rebuild_leaf(right_page, right_entries, next_leaf_page_id=old_next)
        self.bm.unpin_page(right_page_id, is_dirty=True)

        return right_entries[0][0], right_page_id

    def _split_internal(self, node: BTreeNode, new_key: Value, new_right_child: int) -> tuple:
        keys = node.keys()
        children = node.children()

        idx = bisect.bisect_right([k.data for k in keys], new_key.data)
        keys.insert(idx, new_key)
        children.insert(idx + 1, new_right_child)

        mid = self._find_internal_split_point(keys)
        push_up_key = keys[mid]

        left_keys, left_children = keys[:mid], children[:mid + 1]
        right_keys, right_children = keys[mid + 1:], children[mid + 1:]

        right_page_id = self.bm.allocate_page()

        self._rebuild_internal(node.page, left_keys, left_children)

        right_page = self.bm.fetch_page(right_page_id)
        self._rebuild_internal(right_page, right_keys, right_children)
        self.bm.unpin_page(right_page_id, is_dirty=True)

        return push_up_key, right_page_id

    def _create_new_root(self, split_key: Value, left_page_id: int, right_page_id: int):
        new_root_page_id = self.bm.allocate_page()
        page = self.bm.fetch_page(new_root_page_id)
        root_node = BTreeNode.init_internal(page, self.key_type)
        root_node.set_child(0, left_page_id)
        root_node.insert_internal_entry(split_key, right_page_id)
        self.bm.unpin_page(new_root_page_id, is_dirty=True)
        self._root_page_id = new_root_page_id

    def _delete_recursive(self, page_id: int, key: Value) -> bool:
        page = self.bm.fetch_page(page_id)
        node = BTreeNode(page, self.key_type)

        if node.is_leaf():
            removed = node.remove_leaf_entry(key)
            if not removed:
                self.bm.unpin_page(page_id, is_dirty=False)
                raise KeyNotFoundError(f"La clave {key.data!r} no existe en el índice")

            underflow = node.is_underflow() and page_id != self._root_page_id
            self.bm.unpin_page(page_id, is_dirty=True)
            return underflow

        child_idx = node.find_child_index(key)
        child_page_id = node.children()[child_idx]
        self.bm.unpin_page(page_id, is_dirty=False)

        child_underflow = self._delete_recursive(child_page_id, key)
        if not child_underflow:
            return False

        page = self.bm.fetch_page(page_id)
        node = BTreeNode(page, self.key_type)
        self._rebalance_child(node, child_idx)
        underflow = node.is_underflow() and page_id != self._root_page_id
        self.bm.unpin_page(page_id, is_dirty=True)
        return underflow

    def _rebalance_child(self, parent: BTreeNode, child_idx: int):
        children = parent.children()
        child_page_id = children[child_idx]
        child_page = self.bm.fetch_page(child_page_id)
        child = BTreeNode(child_page, self.key_type)
        is_leaf = child.is_leaf()

        if child_idx > 0:
            left_page_id = children[child_idx - 1]
            left_page = self.bm.fetch_page(left_page_id)
            left_sibling = BTreeNode(left_page, self.key_type)
            if not left_sibling.is_underflow() and left_sibling.num_entries() > 1:
                if is_leaf:
                    self._borrow_from_left_leaf(parent, child_idx, left_sibling, child)
                else:
                    self._borrow_from_left_internal(parent, child_idx, left_sibling, child)
                self.bm.unpin_page(left_page_id, is_dirty=True)
                self.bm.unpin_page(child_page_id, is_dirty=True)
                return
            self.bm.unpin_page(left_page_id, is_dirty=False)

        if child_idx < len(children) - 1:
            right_page_id = children[child_idx + 1]
            right_page = self.bm.fetch_page(right_page_id)
            right_sibling = BTreeNode(right_page, self.key_type)
            if not right_sibling.is_underflow() and right_sibling.num_entries() > 1:
                if is_leaf:
                    self._borrow_from_right_leaf(parent, child_idx, child, right_sibling)
                else:
                    self._borrow_from_right_internal(parent, child_idx, child, right_sibling)
                self.bm.unpin_page(right_page_id, is_dirty=True)
                self.bm.unpin_page(child_page_id, is_dirty=True)
                return
            self.bm.unpin_page(right_page_id, is_dirty=False)

        
        if child_idx > 0:
            left_page_id = children[child_idx - 1]
            left_page = self.bm.fetch_page(left_page_id)
            left_sibling = BTreeNode(left_page, self.key_type)
            if is_leaf:
                self._merge_leaves(parent, child_idx - 1, left_sibling, child)
            else:
                self._merge_internal(parent, child_idx - 1, left_sibling, child)
            self.bm.unpin_page(left_page_id, is_dirty=True)
        else:
            right_page_id = children[child_idx + 1]
            right_page = self.bm.fetch_page(right_page_id)
            right_sibling = BTreeNode(right_page, self.key_type)
            if is_leaf:
                self._merge_leaves(parent, child_idx, child, right_sibling)
            else:
                self._merge_internal(parent, child_idx, child, right_sibling)
            self.bm.unpin_page(right_page_id, is_dirty=True)

        self.bm.unpin_page(child_page_id, is_dirty=True)

    def _borrow_from_left_leaf(self, parent, child_idx, left, child):
        left_entries, child_entries = left.entries(), child.entries()
        borrowed = left_entries[-1]

        left_next = self._raw_next(left)
        self._rebuild_leaf(left.page, left_entries[:-1], next_leaf_page_id=left_next)

        child_next = self._raw_next(child)
        new_child_entries = sorted([borrowed] + child_entries, key=lambda e: e[0].data)
        self._rebuild_leaf(child.page, new_child_entries, next_leaf_page_id=child_next)

        self._replace_key(parent, child_idx - 1, new_child_entries[0][0])

    def _borrow_from_right_leaf(self, parent, child_idx, child, right):
        child_entries, right_entries = child.entries(), right.entries()
        borrowed = right_entries[0]

        child_next = self._raw_next(child)
        new_child_entries = child_entries + [borrowed]
        self._rebuild_leaf(child.page, new_child_entries, next_leaf_page_id=child_next)

        right_next = self._raw_next(right)
        self._rebuild_leaf(right.page, right_entries[1:], next_leaf_page_id=right_next)

        self._replace_key(parent, child_idx, right_entries[1][0] if len(right_entries) > 1 else borrowed)

    def _merge_leaves(self, parent, left_idx, left, right):
        left_entries, right_entries = left.entries(), right.entries()
        right_next = self._raw_next(right)

        merged = left_entries + right_entries
        self._rebuild_leaf(left.page, merged, next_leaf_page_id=right_next)

        self._remove_key_and_right_child(parent, left_idx)

    def _borrow_from_left_internal(self, parent, child_idx, left, child):
        left_keys, left_children = left.keys(), left.children()
        child_keys, child_children = child.keys(), child.children()

        separator = parent.keys()[child_idx - 1]
        borrowed_key, borrowed_child = left_keys[-1], left_children[-1]

        self._rebuild_internal(left.page, left_keys[:-1], left_children[:-1])
        self._rebuild_internal(
            child.page, [separator] + child_keys, [borrowed_child] + child_children
        )
        self._replace_key(parent, child_idx - 1, borrowed_key)

    def _borrow_from_right_internal(self, parent, child_idx, child, right):
        right_keys, right_children = right.keys(), right.children()
        child_keys, child_children = child.keys(), child.children()

        separator = parent.keys()[child_idx]
        borrowed_key, borrowed_child = right_keys[0], right_children[0]

        self._rebuild_internal(
            child.page, child_keys + [separator], child_children + [borrowed_child]
        )
        self._rebuild_internal(right.page, right_keys[1:], right_children[1:])
        self._replace_key(parent, child_idx, borrowed_key)

    def _merge_internal(self, parent, left_idx, left, right):
        separator = parent.keys()[left_idx]
        merged_keys = left.keys() + [separator] + right.keys()
        merged_children = left.children() + right.children()

        self._rebuild_internal(left.page, merged_keys, merged_children)
        self._remove_key_and_right_child(parent, left_idx)

    def _shrink_root_if_needed(self):
        page = self.bm.fetch_page(self._root_page_id)
        node = BTreeNode(page, self.key_type)
        if not node.is_leaf() and node.num_entries() == 0:
            new_root_id = node.children()[0]
            self.bm.unpin_page(self._root_page_id, is_dirty=False)
            self._root_page_id = new_root_id
        else:
            self.bm.unpin_page(self._root_page_id, is_dirty=False)

    def _raw_next(self, node: BTreeNode) -> int:
        n = node.next_leaf_page_id()
        return n if n is not None else NO_SIBLING

    def _rebuild_leaf(self, page, entries: list, next_leaf_page_id: int) -> BTreeNode:
        page.clear()
        node = BTreeNode.init_leaf(page, self.key_type, next_leaf_page_id=next_leaf_page_id)
        for key, rid in entries:
            if not node.insert_leaf_entry(key, rid):
                raise IndexCorruptionError(
                    f"No se pudo insertar la clave {key.data!r} al reconstruir "
                    "una hoja — el split/merge dejó más datos de los que la "
                    "página puede contener"
                )
        return node

    def _rebuild_internal(self, page, keys: list, children: list) -> BTreeNode:
        page.clear()
        node = BTreeNode.init_internal(page, self.key_type)
        node.set_child(0, children[0])
        for key, child_page_id in zip(keys, children[1:]):
            if not node.insert_internal_entry(key, child_page_id):
                raise IndexCorruptionError(
                    f"No se pudo insertar la clave {key.data!r} al reconstruir "
                    "un nodo interno — el split/merge dejó más datos de los "
                    "que la página puede contener"
                )
        return node

    def _replace_key(self, node: BTreeNode, index: int, new_key: Value):
        keys = node.keys()
        keys[index] = new_key
        self._rebuild_internal(node.page, keys, node.children())

    def _remove_key_and_right_child(self, node: BTreeNode, index: int):
        keys = node.keys()
        children = node.children()
        del keys[index]
        del children[index + 1]
        self._rebuild_internal(node.page, keys, children)

    def _allocate_and_init_root(self) -> int:
        page_id = self.bm.allocate_page()
        page = self.bm.fetch_page(page_id)
        BTreeNode.init_leaf(page, self.key_type)
        self.bm.unpin_page(page_id, is_dirty=True)
        return page_id

    def _find_leaf_split_point(self, entries: list) -> int:
        usable = self.bm.file_manager.page_size - HEADER_SIZE
        sizes = [key_size(key) + RID_SIZE + SLOT_SIZE for key, _ in entries]
        total = sum(sizes)

        prefix = [0] * (len(sizes) + 1)
        for i, s in enumerate(sizes):
            prefix[i + 1] = prefix[i] + s

        best_idx, best_diff = None, None
        for i in range(1, len(entries)):
            left, right = prefix[i], total - prefix[i]
            if left <= usable and right <= usable:
                diff = abs(left - right)
                if best_diff is None or diff < best_diff:
                    best_idx, best_diff = i, diff

        if best_idx is None:
            raise IndexEntryTooLargeError(
                "No es posible dividir la hoja: al menos una entrada es "
                "demasiado grande para convivir con otras en una sola página"
            )
        return best_idx

    def _find_internal_split_point(self, keys: list) -> int:
        usable = self.bm.file_manager.page_size - HEADER_SIZE - CHILD_SIZE
        sizes = [key_size(k) + SLOT_SIZE + CHILD_SIZE for k in keys]
        total = sum(sizes)

        prefix = [0] * (len(sizes) + 1)
        for i, s in enumerate(sizes):
            prefix[i + 1] = prefix[i] + s

        best_m, best_diff = None, None
        for m in range(1, len(keys)):
            left = prefix[m]
            right = total - prefix[m + 1] if m + 1 <= len(keys) else 0
            if left <= usable and right <= usable:
                diff = abs(left - right)
                if best_diff is None or diff < best_diff:
                    best_m, best_diff = m, diff

        if best_m is None:
            raise IndexEntryTooLargeError(
                "No es posible dividir el nodo interno: alguna clave es "
                "demasiado grande para convivir con otras en una sola página"
            )
        return best_m