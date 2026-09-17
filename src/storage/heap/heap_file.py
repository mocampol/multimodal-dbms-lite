"""
Heap File implementation using the Slotted Page layout, on top of the
Buffer Manager.

Physical layout of one page:

    Header:
        num_slots           (2 bytes, unsigned)
        free_space_offset   (4 bytes, unsigned), start of the data area

    Each ItemId Array entry (9 bytes) is one of:
        status = EMPTY:      (a, b) = (offset, length) of the LAST record
                              that occupied this slot, kept around so a
                              future insert can reuse the space if the new
                              record fits (free-space reuse strategy).
        status = VALID:      (a, b) = (offset, length) of the live record.
        status = FORWARDED:  (a, b) = (page_id, slot) of where this record
                              now actually lives, on a DIFFERENT page.

Classes:
    HeapFile: insert/get/delete/update/scan over slotted pages.
"""

import struct

from storage.page import Page, PAGE_SIZE
from storage.buffer_manager import BufferManager
from common.schema import Schema
from common.record import Record

from .rid import RID
from .record_codec import encode_record, decode_record


HEADER_FORMAT = ">HI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

SLOT_FORMAT = ">BII"
SLOT_SIZE = struct.calcsize(SLOT_FORMAT)

STATUS_EMPTY = 0
STATUS_VALID = 1
STATUS_FORWARDED = 2


class HeapFile:
    """
    Unordered collection of pages holding records for one table.

    Attributes:
        schema (Schema): Structure of the records stored in this file.
        bm (BufferManager): Buffer Manager for this table's physical file.
    """

    def __init__(self, schema: Schema, buffer_manager: BufferManager):
        self.schema = schema
        self.bm = buffer_manager

        # In-memory list of page_ids belonging to this file, in creation
        # order. Rebuilt from disk on startup (existing tables), or
        # started fresh with a single page (brand-new tables).
        self._known_pages: list[int] = list(range(self.bm.file_manager.page_count()))
        if not self._known_pages:
            self._known_pages.append(self._allocate_and_init_page())

        # First-fit cursor: remembers the last page where an insert
        # succeeded, so repeated inserts don't rescan from page 0 every time.
        self._page_hint = 0

    # ---------- public API ----------

    def insert(self, record: Record) -> RID:
        """
        Inserts `record`, returning its new RID. Tries to reuse space
        left by a lazy delete or a shrinking update first; falls back to
        appending in a page's free area, and finally to allocating a new
        page if no existing page has room.
        """
        if not record.validate(self.schema):
            raise ValueError(
                f"El registro no es válido para el schema de "
                f"'{self.schema.table_name}': {record}"
            )

        payload = encode_record(record, self.schema)
        if len(payload) + SLOT_SIZE > PAGE_SIZE - HEADER_SIZE:
            raise ValueError(
                f"El registro serializado ({len(payload)} bytes) no cabe "
                "en una sola página; registros que abarcan varias páginas "
                "no están soportados en esta implementación"
            )

        n = len(self._known_pages)
        for offset in range(n):
            idx = (self._page_hint + offset) % n
            page_id = self._known_pages[idx]
            page = self.bm.fetch_page(page_id)
            slot = self._try_insert_into_page(page, payload)
            self.bm.unpin_page(page_id, is_dirty=(slot is not None))
            if slot is not None:
                self._page_hint = idx
                return RID(page_id, slot)

        # No existing page had room: allocate a new one.
        page_id = self._allocate_and_init_page()
        self._known_pages.append(page_id)
        page = self.bm.fetch_page(page_id)
        slot = self._try_insert_into_page(page, payload)
        self.bm.unpin_page(page_id, is_dirty=True)
        self._page_hint = len(self._known_pages) - 1
        return RID(page_id, slot)

    def get(self, rid: RID) -> Record | None:
        """
        Returns the record at `rid`, transparently following a forwarding
        pointer if the record was relocated to another page by update().
        Returns None if the slot is empty (deleted or never used).
        """
        page = self.bm.fetch_page(rid.page_id)
        try:
            status, a, b = self._read_slot(page, rid.slot)
            if status == STATUS_EMPTY:
                return None
            if status == STATUS_FORWARDED:
                return self.get(RID(a, b))
            return decode_record(page.read_bytes(a, b), self.schema)
        finally:
            self.bm.unpin_page(rid.page_id, is_dirty=False)

    def delete(self, rid: RID):
        """
        Lazily deletes the record at `rid`: marks the slot EMPTY but
        keeps its (offset, length) so a future insert can reuse that
        exact space. Does not compact the page immediately, per spec.
        """
        page = self.bm.fetch_page(rid.page_id)
        try:
            status, a, b = self._read_slot(page, rid.slot)
            if status == STATUS_FORWARDED:
                self.bm.unpin_page(rid.page_id, is_dirty=False)
                self.delete(RID(a, b))
                return
            self._write_slot(page, rid.slot, STATUS_EMPTY, a, b)
        finally:
            self.bm.unpin_page(rid.page_id, is_dirty=True)

    def update(self, rid: RID, new_record: Record) -> RID:
        """
        Updates the record at `rid` with `new_record`. If the new,
        serialized record fits within the original slot's capacity, it
        is overwritten in place. Otherwise, the new version is inserted
        elsewhere (possibly on a different page) and the original slot
        becomes a FORWARDED pointer to it — so external references to
        the original RID (e.g. a secondary B+ Tree index) keep working
        without needing to be updated themselves.

        Always returns the ORIGINAL rid: from the caller's perspective,
        the record's identity never changes.
        """
        if not new_record.validate(self.schema):
            raise ValueError(
                f"El registro no es válido para el schema de "
                f"'{self.schema.table_name}': {new_record}"
            )

        page = self.bm.fetch_page(rid.page_id)
        status, a, b = self._read_slot(page, rid.slot)

        if status == STATUS_FORWARDED:
            self.bm.unpin_page(rid.page_id, is_dirty=False)
            self.update(RID(a, b), new_record)
            return rid

        payload = encode_record(new_record, self.schema)

        if len(payload) <= b:
            page.write_bytes(a, payload)
            self._write_slot(page, rid.slot, STATUS_VALID, a, len(payload))
            self.bm.unpin_page(rid.page_id, is_dirty=True)
            return rid

        # Doesn't fit in the original slot: relocate and leave a
        # forwarding pointer. Note this is the ONLY situation in this
        # file that produces a forwarding pointer to a DIFFERENT page —
        # intra-page compaction (compact_page) never needs one, since
        # slot numbers (and therefore RIDs) don't change during it.
        self.bm.unpin_page(rid.page_id, is_dirty=False)
        new_rid = self.insert(new_record)

        page = self.bm.fetch_page(rid.page_id)
        self._write_slot(page, rid.slot, STATUS_FORWARDED, new_rid.page_id, new_rid.slot)
        self.bm.unpin_page(rid.page_id, is_dirty=True)

        return rid

    def compact_page(self, page_id: int):
        """
        Defragments a single page in place: repacks all VALID records
        contiguously in the data area, reclaiming space left behind by
        lazy deletes and by in-place updates that shrank a record.

        Slot NUMBERS never change during this process — only the
        (offset, length) each slot points to — so every RID pointing
        into this page stays correct without any forwarding pointer.
        Forwarding is only needed when a record moves to a DIFFERENT
        page (see update()).
        """
        page = self.bm.fetch_page(page_id)
        try:
            num_slots, _ = self._read_header(page)

            live_data = {}  # slot_index -> raw bytes (captured before any overwrite)
            for i in range(num_slots):
                status, a, b = self._read_slot(page, i)
                if status == STATUS_VALID:
                    live_data[i] = page.read_bytes(a, b)

            cursor = page.size
            for i, data in live_data.items():
                cursor -= len(data)
                page.write_bytes(cursor, data)
                self._write_slot(page, i, STATUS_VALID, cursor, len(data))

            for i in range(num_slots):
                status, _, _ = self._read_slot(page, i)
                if status == STATUS_EMPTY:
                    # Nothing behind it anymore — reset so it doesn't
                    # falsely advertise leftover reusable capacity.
                    self._write_slot(page, i, STATUS_EMPTY, 0, 0)

            self._write_header(page, num_slots, cursor)
        finally:
            self.bm.unpin_page(page_id, is_dirty=True)

    def scan(self):
        """
        Yields every currently VALID record in the file, across all
        pages, in page then slot order. FORWARDED slots are skipped:
        the record they point to is a VALID slot somewhere else in the
        file and will be yielded when that slot is reached instead.
        """
        for page_id in self._known_pages:
            page = self.bm.fetch_page(page_id)
            try:
                num_slots, _ = self._read_header(page)
                for i in range(num_slots):
                    status, a, b = self._read_slot(page, i)
                    if status == STATUS_VALID:
                        yield decode_record(page.read_bytes(a, b), self.schema)
            finally:
                self.bm.unpin_page(page_id, is_dirty=False)


    def _allocate_and_init_page(self) -> int:
        page_id = self.bm.allocate_page()
        page = self.bm.fetch_page(page_id)
        self._write_header(page, num_slots=0, free_space_offset=page.size)
        self.bm.unpin_page(page_id, is_dirty=True)
        return page_id

    def _try_insert_into_page(self, page: Page, payload: bytes) -> int | None:
        """
        Attempts to insert `payload` into `page`. Returns the slot index
        on success, or None if this page has no room for it.
        """
        num_slots, free_space_offset = self._read_header(page)

        # 1) Free-space reuse: look for a deleted (EMPTY) slot whose
        #    leftover capacity can hold the new payload.
        for i in range(num_slots):
            status, a, b = self._read_slot(page, i)
            if status == STATUS_EMPTY and b >= len(payload):
                page.write_bytes(a, payload)
                self._write_slot(page, i, STATUS_VALID, a, len(payload))
                return i

        # 2) No reusable slot: append in the contiguous free area.
        header_end = HEADER_SIZE + num_slots * SLOT_SIZE
        contiguous_free = free_space_offset - header_end
        if contiguous_free < SLOT_SIZE + len(payload):
            return None

        new_offset = free_space_offset - len(payload)
        page.write_bytes(new_offset, payload)
        self._write_slot(page, num_slots, STATUS_VALID, new_offset, len(payload))
        self._write_header(page, num_slots + 1, new_offset)
        return num_slots


    def _read_header(self, page: Page) -> tuple:
        return struct.unpack(HEADER_FORMAT, page.read_bytes(0, HEADER_SIZE))

    def _write_header(self, page: Page, num_slots: int, free_space_offset: int):
        page.write_bytes(0, struct.pack(HEADER_FORMAT, num_slots, free_space_offset))

    def _read_slot(self, page: Page, index: int) -> tuple:
        offset = HEADER_SIZE + index * SLOT_SIZE
        return struct.unpack(SLOT_FORMAT, page.read_bytes(offset, SLOT_SIZE))

    def _write_slot(self, page: Page, index: int, status: int, a: int, b: int):
        offset = HEADER_SIZE + index * SLOT_SIZE
        page.write_bytes(offset, struct.pack(SLOT_FORMAT, status, a, b))

    @property
    def root_page_id(self) -> int:
        """
        Returns the first page_id of this heap file. Used by Catalog to
        persist the real starting page into sys_tables.root_page_id once
        physical storage has actually been created for a table.
        """
        return self._known_pages[0]
