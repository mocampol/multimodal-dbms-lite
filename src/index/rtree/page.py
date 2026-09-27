"""Fixed-size serialized pages for persistent R-Tree nodes."""

from dataclasses import dataclass
from enum import Enum
import struct
import zlib

from spatial.geometry import BoundingBox
from storage.heap.rid import RID
from storage.page import Page

from .exceptions import RTreeNodeCapacityError, RTreePageCorruptionError


PAGE_MAGIC = b"RTRE"
FORMAT_VERSION = 1
HEADER_FORMAT = ">4sBBH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CHECKSUM_FORMAT = ">I"
CHECKSUM_SIZE = struct.calcsize(CHECKSUM_FORMAT)
MBR_FORMAT = ">dddd"
MBR_SIZE = struct.calcsize(MBR_FORMAT)
LEAF_ENTRY_FORMAT = ">ddddII"
LEAF_ENTRY_SIZE = struct.calcsize(LEAF_ENTRY_FORMAT)
INTERNAL_ENTRY_FORMAT = ">ddddI"
INTERNAL_ENTRY_SIZE = struct.calcsize(INTERNAL_ENTRY_FORMAT)
UINT32_MAX = (1 << 32) - 1


class NodeType(Enum):
    LEAF = 1
    INTERNAL = 2


@dataclass(frozen=True)
class LeafEntry:
    mbr: BoundingBox
    rid: RID

    def __post_init__(self):
        if not isinstance(self.mbr, BoundingBox):
            raise TypeError("mbr debe ser un BoundingBox")
        if not isinstance(self.rid, RID):
            raise TypeError("rid debe ser un RID")
        _validate_uint32(self.rid.page_id, "rid.page_id")
        _validate_uint32(self.rid.slot, "rid.slot")


@dataclass(frozen=True)
class InternalEntry:
    mbr: BoundingBox
    child_page_id: int

    def __post_init__(self):
        if not isinstance(self.mbr, BoundingBox):
            raise TypeError("mbr debe ser un BoundingBox")
        _validate_uint32(self.child_page_id, "child_page_id")


def _validate_uint32(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT32_MAX:
        raise ValueError(f"{name} debe ser un entero entre 0 y {UINT32_MAX}")


class RTreePage:
    """Codec and capacity rules for one fixed-size R-Tree node page.

    The page stores an 8-byte magic/version/type/count header, fixed-width
    entries, zero-filled unused space, and a 4-byte CRC32 at the end.
    """

    @staticmethod
    def entry_size(node_type: NodeType) -> int:
        if node_type == NodeType.LEAF:
            return LEAF_ENTRY_SIZE
        if node_type == NodeType.INTERNAL:
            return INTERNAL_ENTRY_SIZE
        raise ValueError(f"Tipo de nodo desconocido: {node_type!r}")

    @classmethod
    def max_entries(cls, page_size: int, node_type: NodeType) -> int:
        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise ValueError("page_size debe ser un entero")
        usable_bytes = page_size - HEADER_SIZE - CHECKSUM_SIZE
        if usable_bytes < 0:
            return 0
        return min(0xFFFF, usable_bytes // cls.entry_size(node_type))

    @classmethod
    def initialize(cls, page: Page, node_type: NodeType) -> None:
        cls.serialize(page, node_type, ())

    @classmethod
    def serialize(cls, page: Page, node_type: NodeType, entries) -> None:
        if not isinstance(page, Page):
            raise TypeError("page debe ser un Page")
        if not isinstance(node_type, NodeType):
            raise ValueError(f"Tipo de nodo desconocido: {node_type!r}")
        entries = tuple(entries)
        capacity = cls.max_entries(page.size, node_type)
        if capacity == 0:
            raise RTreeNodeCapacityError(
                f"Una página de {page.size} bytes no puede contener un nodo R-Tree"
            )
        if len(entries) > capacity:
            raise RTreeNodeCapacityError(
                f"El nodo tiene {len(entries)} entradas; capacidad máxima: {capacity}"
            )

        encoded_entries = bytearray()
        for entry in entries:
            if node_type == NodeType.LEAF:
                if not isinstance(entry, LeafEntry):
                    raise TypeError("una hoja solo puede contener LeafEntry")
                encoded_entries.extend(
                    struct.pack(
                        LEAF_ENTRY_FORMAT,
                        entry.mbr.min_x,
                        entry.mbr.min_y,
                        entry.mbr.max_x,
                        entry.mbr.max_y,
                        entry.rid.page_id,
                        entry.rid.slot,
                    )
                )
            else:
                if not isinstance(entry, InternalEntry):
                    raise TypeError("un nodo interno solo puede contener InternalEntry")
                encoded_entries.extend(
                    struct.pack(
                        INTERNAL_ENTRY_FORMAT,
                        entry.mbr.min_x,
                        entry.mbr.min_y,
                        entry.mbr.max_x,
                        entry.mbr.max_y,
                        entry.child_page_id,
                    )
                )

        raw_page = bytearray(page.size)
        struct.pack_into(
            HEADER_FORMAT,
            raw_page,
            0,
            PAGE_MAGIC,
            FORMAT_VERSION,
            node_type.value,
            len(entries),
        )
        entries_end = HEADER_SIZE + len(encoded_entries)
        raw_page[HEADER_SIZE:entries_end] = encoded_entries
        checksum = zlib.crc32(raw_page[:-CHECKSUM_SIZE]) & UINT32_MAX
        struct.pack_into(CHECKSUM_FORMAT, raw_page, page.size - CHECKSUM_SIZE, checksum)
        page.write_bytes(0, raw_page)

    @classmethod
    def deserialize(cls, page: Page) -> tuple[NodeType, list[LeafEntry | InternalEntry]]:
        if not isinstance(page, Page):
            raise TypeError("page debe ser un Page")
        if page.size < HEADER_SIZE + CHECKSUM_SIZE + INTERNAL_ENTRY_SIZE:
            raise RTreePageCorruptionError(
                f"Página {page.page_id} demasiado pequeña para un nodo R-Tree"
            )

        raw_page = page.read_bytes(0, page.size)
        stored_checksum = struct.unpack_from(
            CHECKSUM_FORMAT, raw_page, page.size - CHECKSUM_SIZE
        )[0]
        computed_checksum = zlib.crc32(raw_page[:-CHECKSUM_SIZE]) & UINT32_MAX
        if stored_checksum != computed_checksum:
            raise RTreePageCorruptionError(
                f"Checksum inválido en la página R-Tree {page.page_id}"
            )

        magic, version, node_type_value, entry_count = struct.unpack_from(
            HEADER_FORMAT, raw_page, 0
        )
        if magic != PAGE_MAGIC:
            raise RTreePageCorruptionError(
                f"Magic inválido en la página R-Tree {page.page_id}"
            )
        if version != FORMAT_VERSION:
            raise RTreePageCorruptionError(
                f"Versión no soportada ({version}) en la página R-Tree {page.page_id}"
            )
        try:
            node_type = NodeType(node_type_value)
        except ValueError:
            raise RTreePageCorruptionError(
                f"Tipo de nodo inválido en la página R-Tree {page.page_id}"
            ) from None

        entry_size = cls.entry_size(node_type)
        capacity = cls.max_entries(page.size, node_type)
        if entry_count > capacity:
            raise RTreePageCorruptionError(
                f"Conteo de entradas inválido ({entry_count}) en la página R-Tree {page.page_id}"
            )

        entries_end = HEADER_SIZE + entry_count * entry_size
        padding_end = page.size - CHECKSUM_SIZE
        if any(raw_page[entries_end:padding_end]):
            raise RTreePageCorruptionError(
                f"Padding no nulo en la página R-Tree {page.page_id}"
            )

        entries = []
        try:
            for entry_index in range(entry_count):
                offset = HEADER_SIZE + entry_index * entry_size
                values = struct.unpack_from(
                    LEAF_ENTRY_FORMAT if node_type == NodeType.LEAF else INTERNAL_ENTRY_FORMAT,
                    raw_page,
                    offset,
                )
                mbr = BoundingBox(*values[:4])
                if node_type == NodeType.LEAF:
                    entries.append(LeafEntry(mbr, RID(values[4], values[5])))
                else:
                    entries.append(InternalEntry(mbr, values[4]))
        except (TypeError, ValueError, struct.error) as error:
            raise RTreePageCorruptionError(
                f"Entrada inválida en la página R-Tree {page.page_id}: {error}"
            ) from error
        return node_type, entries