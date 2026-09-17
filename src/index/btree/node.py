import struct
import bisect
from enum import Enum

from storage.page import Page
from storage.heap.rid import RID
from common.value import DataType, Value

from .key_codec import encode_key, decode_key, key_size


class NodeType(Enum):
    LEAF = 1
    INTERNAL = 2


HEADER_FORMAT = ">BHiI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

SLOT_FORMAT = ">II"
SLOT_SIZE = struct.calcsize(SLOT_FORMAT)

CHILD_FORMAT = ">I"
CHILD_SIZE = struct.calcsize(CHILD_FORMAT)

RID_FORMAT = ">II"
RID_SIZE = struct.calcsize(RID_FORMAT)

NO_SIBLING = -1


class BTreeNode:
    def __init__(self, page: Page, key_type: DataType):
        self.page = page
        self.key_type = key_type

    @classmethod
    def init_leaf(cls, page: Page, key_type: DataType, next_leaf_page_id: int = NO_SIBLING) -> "BTreeNode":
        node = cls(page, key_type)
        node._write_header(NodeType.LEAF, num_entries=0, next_page_id=next_leaf_page_id, free_space_offset=page.size)
        return node

    @classmethod
    def init_internal(cls, page: Page, key_type: DataType) -> "BTreeNode":
        node = cls(page, key_type)
        node._write_header(NodeType.INTERNAL, num_entries=0, next_page_id=NO_SIBLING, free_space_offset=page.size)
        return node

    def is_leaf(self) -> bool:
        return self._read_header()[0] == NodeType.LEAF

    def num_entries(self) -> int:
        return self._read_header()[1]

    def next_leaf_page_id(self) -> int | None:
        node_type, _, next_page_id, _ = self._read_header()
        if node_type != NodeType.LEAF:
            raise ValueError("next_leaf_page_id() solo aplica a nodos hoja")
        return None if next_page_id == NO_SIBLING else next_page_id

    def set_next_leaf_page_id(self, page_id: int):
        node_type, num_entries, _, free_space_offset = self._read_header()
        if node_type != NodeType.LEAF:
            raise ValueError("set_next_leaf_page_id() solo aplica a nodos hoja")
        self._write_header(node_type, num_entries, page_id, free_space_offset)

    def _read_header(self) -> tuple:
        node_type_val, num_entries, next_page_id, free_space_offset = struct.unpack(
            HEADER_FORMAT, self.page.read_bytes(0, HEADER_SIZE)
        )
        return NodeType(node_type_val), num_entries, next_page_id, free_space_offset

    def _write_header(self, node_type: NodeType, num_entries: int, next_page_id: int, free_space_offset: int):
        self.page.write_bytes(
            0, struct.pack(HEADER_FORMAT, node_type.value, num_entries, next_page_id, free_space_offset)
        )

    def _child_offset(self, index: int) -> int:
        return HEADER_SIZE + index * CHILD_SIZE

    def children(self) -> list[int]:
        node_type, num_entries, _, _ = self._read_header()
        if node_type != NodeType.INTERNAL:
            raise ValueError("children() solo aplica a nodos internos")
        return [
            struct.unpack(CHILD_FORMAT, self.page.read_bytes(self._child_offset(i), CHILD_SIZE))[0]
            for i in range(num_entries + 1)
        ]

    def _write_child(self, index: int, page_id: int):
        self.page.write_bytes(self._child_offset(index), struct.pack(CHILD_FORMAT, page_id))

    def _slot_array_offset(self) -> int:
        node_type, num_entries, _, _ = self._read_header()
        if node_type == NodeType.LEAF:
            return HEADER_SIZE
        return HEADER_SIZE + (num_entries + 1) * CHILD_SIZE

    def _slot_offset(self, index: int) -> int:
        return self._slot_array_offset() + index * SLOT_SIZE

    def _read_slot(self, index: int) -> tuple:
        return struct.unpack(SLOT_FORMAT, self.page.read_bytes(self._slot_offset(index), SLOT_SIZE))

    def _write_slot(self, index: int, offset: int, length: int):
        self.page.write_bytes(self._slot_offset(index), struct.pack(SLOT_FORMAT, offset, length))

    def keys(self) -> list[Value]:
        _, num_entries, _, _ = self._read_header()
        result = []
        for i in range(num_entries):
            offset, length = self._read_slot(i)
            buf = self.page.read_bytes(offset, length)
            result.append(decode_key(self.key_type, buf))
        return result

    def entries(self) -> list[tuple]:
        if not self.is_leaf():
            raise ValueError("entries() solo aplica a nodos hoja")
        _, num_entries, _, _ = self._read_header()
        result = []
        for i in range(num_entries):
            offset, length = self._read_slot(i)
            buf = self.page.read_bytes(offset, length)
            key = decode_key(self.key_type, buf)
            page_id, slot = struct.unpack(RID_FORMAT, buf[-RID_SIZE:])
            result.append((key, RID(page_id, slot)))
        return result

    def find_entry_index(self, search_key: Value) -> int:
        if not self.is_leaf():
            raise ValueError("find_entry_index() solo aplica a nodos hoja")
        key_values = [k.data for k in self.keys()]
        return bisect.bisect_left(key_values, search_key.data)

    def find_child_index(self, search_key: Value) -> int:
        if self.is_leaf():
            raise ValueError("find_child_index() solo aplica a nodos internos")
        key_values = [k.data for k in self.keys()]
        return bisect.bisect_right(key_values, search_key.data)

    def insert_leaf_entry(self, key: Value, rid: RID) -> bool:
        if not self.is_leaf():
            raise ValueError("insert_leaf_entry() solo aplica a nodos hoja")

        node_type, num_entries, next_page_id, free_space_offset = self._read_header()

        payload = encode_key(key) + struct.pack(RID_FORMAT, rid.page_id, rid.slot)

        header_end = HEADER_SIZE + (num_entries + 1) * SLOT_SIZE
        if free_space_offset - header_end < len(payload):
            return False

        insert_at = self.find_entry_index(key)
        for i in range(num_entries, insert_at, -1):
            offset, length = self._read_slot(i - 1)
            self._write_slot(i, offset, length)

        new_offset = free_space_offset - len(payload)
        self.page.write_bytes(new_offset, payload)
        self._write_slot(insert_at, new_offset, len(payload))
        self._write_header(node_type, num_entries + 1, next_page_id, new_offset)
        return True

    def remove_leaf_entry(self, key: Value, rid=None) -> bool:
        if not self.is_leaf():
            raise ValueError("remove_leaf_entry() solo aplica a nodos hoja")

        node_type, num_entries, next_page_id, free_space_offset = self._read_header()
        idx = self.find_entry_index(key)
        if idx >= num_entries or self.keys()[idx].data != key.data:
            return False

        if rid is not None:
            while idx < num_entries:
                stored_key, stored_rid = self.entries()[idx]
                if stored_key.data != key.data:
                    return False
                if stored_rid == rid:
                    break
                idx += 1
            if idx >= num_entries:
                return False

        for i in range(idx, num_entries - 1):
            offset, length = self._read_slot(i + 1)
            self._write_slot(i, offset, length)

        self._write_header(node_type, num_entries - 1, next_page_id, free_space_offset)
        return True

    def insert_internal_entry(self, key: Value, right_child_page_id: int) -> bool:
        if self.is_leaf():
            raise ValueError("insert_internal_entry() solo aplica a nodos internos")

        node_type, num_entries, next_page_id, _ = self._read_header()

        # Snapshot of the logical contents BEFORE touching the page —
        # independent of any subsequent writes.
        old_keys = self.keys()
        old_children = self.children()

        insert_at = bisect.bisect_right([k.data for k in old_keys], key.data)
        new_keys = old_keys[:insert_at] + [key] + old_keys[insert_at:]
        new_children = old_children[:insert_at + 1] + [right_child_page_id] + old_children[insert_at + 1:]
        new_num_entries = num_entries + 1

        encoded_keys = [encode_key(k) for k in new_keys]
        children_area_size = (new_num_entries + 1) * CHILD_SIZE
        slot_area_size = new_num_entries * SLOT_SIZE
        header_end = HEADER_SIZE + children_area_size + slot_area_size
        needed_data_bytes = sum(len(e) for e in encoded_keys)

        if self.page.size - header_end < needed_data_bytes:
            return False  # btree.py must split this node before retrying

        # Update the header FIRST so _child_offset()/_slot_offset(),
        # called below, calculate against the FINAL layout (new_num_entries),
        # avoiding the old/new offset mismatch that caused corruption.
        self._write_header(node_type, new_num_entries, next_page_id, self.page.size)

        for i, child_page_id in enumerate(new_children):
            self._write_child(i, child_page_id)

        cursor = self.page.size
        for i, raw in enumerate(encoded_keys):
            cursor -= len(raw)
            self.page.write_bytes(cursor, raw)
            self._write_slot(i, cursor, len(raw))

        self._write_header(node_type, new_num_entries, next_page_id, cursor)
        return True

    def has_room_for(self, key: Value) -> bool:
        node_type, num_entries, _, free_space_offset = self._read_header()
        needed = key_size(key) + (RID_SIZE if node_type == NodeType.LEAF else 0)

        slot_area_size = (num_entries + 1) * SLOT_SIZE
        if node_type == NodeType.INTERNAL:
            children_area_size = (num_entries + 2) * CHILD_SIZE
        else:
            children_area_size = 0

        header_end = HEADER_SIZE + children_area_size + slot_area_size
        return free_space_offset - header_end >= needed

    def __repr__(self):
        node_type, num_entries, next_page_id, _ = self._read_header()
        return f"BTreeNode({node_type.name}, page_id={self.page.page_id}, entries={num_entries})"

    def set_child(self, index: int, page_id: int):
        if self.is_leaf():
            raise ValueError("set_child() solo aplica a nodos internos")
        self._write_child(index, page_id)

    def occupancy_bytes(self) -> int:
        node_type, num_entries, _, free_space_offset = self._read_header()
        data_bytes = self.page.size - free_space_offset
        slot_bytes = num_entries * SLOT_SIZE
        children_bytes = (num_entries + 1) * CHILD_SIZE if node_type == NodeType.INTERNAL else 0
        return data_bytes + slot_bytes + children_bytes

    def is_underflow(self, min_occupancy: float = 0.5) -> bool:
        usable_capacity = self.page.size - HEADER_SIZE
        return self.occupancy_bytes() < usable_capacity * min_occupancy
