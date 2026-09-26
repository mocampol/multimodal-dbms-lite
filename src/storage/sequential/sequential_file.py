"""
Sequential File implementation: records are kept in physical order by a
search key, spread across a linked chain of pages. When a page has no
room for a new record in its correct sorted position, the record is
routed to an auxiliary Overflow File and logically linked to the page
it belongs to. Periodically, reorganize() merges the overflow back into
the main chain once wasted space crosses a configurable threshold.

Physical layout of one main-chain page:

    Header (14 bytes):
        num_slots           (2 bytes, unsigned)
        free_space_offset   (4 bytes, unsigned) — start of the data area
        next_page_id        (4 bytes, signed)   — -1 if this is the last page
        (reserved, unused)  (4 bytes)           — kept for future use / alignment

    ItemId Array: one 9-byte entry per slot, kept in ASCENDING KEY ORDER —
    i.e. slot i's key <= slot i+1's key among VALID slots. Entries:
        status = EMPTY: (offset, length) of the record that used to be
                         there (kept so the space stays measurable for
                         the reorganization threshold, but is NOT reused
                         for new inserts — reclaiming space is reorganize()'s
                         job for this structure, per spec).
        status = VALID: (offset, length) of the live record.

    Data Area: same append-from-the-end strategy as heap_file's slotted
    page. A record's physical position in the data area does NOT need to
    match its logical (sorted) position — only the ItemId Array's slot
    ORDER encodes the sort order, so inserting in the middle only means
    shifting small 9-byte slot entries, never the record bytes themselves.

Algorithms:

    ORDERED INSERTION (_try_insert_sorted):
        1. Find the correct insertion position by scanning the ItemId Array
           left-to-right and comparing only VALID slots' keys. EMPTY slots
           are skipped — they do not affect key order.
        2. Shift all slot entries to the right of insert_at by one position
           (only 9-byte metadata entries are moved, never record bytes).
        3. Append the serialized record at the top of the Data Area
           (free_space_offset grows downward) and write the new slot entry.
        4. If there is not enough contiguous free space (slot header + payload),
           return None → the caller routes the record to the Overflow File.
        INVARIANT: After every insert, all VALID slots in the ItemId Array
        are in non-decreasing key order.

    LAZY DELETION (delete):
        - By RID: locate the slot, flip status VALID → EMPTY. The record
          bytes remain on disk; only the slot's status bit changes. Space
          is reclaimed only during reorganize().
        - By key: scan all slots on the target page for matching keys;
          mark each as EMPTY. Also removes matching entries from the
          in-memory overflow chain (actually deletes them from the HeapFile).
        After deletion, maybe_reorganize() is called automatically: if the
        wasted-space ratio has crossed the threshold it triggers a full
        reorganization immediately.

    REORGANIZATION (reorganize / maybe_reorganize):
        Trigger condition: wasted_space_ratio() > reorganize_threshold (default 30%).
        The wasted ratio is:
            wasted = Σ(length + SLOT_SIZE for each EMPTY slot) +
                     (overflow_file_pages × PAGE_SIZE)
            total  = main_chain_pages × PAGE_SIZE + overflow_file_pages × PAGE_SIZE
        Algorithm:
            1. Collect all live records via scan() — already yields them in
               ascending key order by merging main-chain + overflow per page.
            2. Clear the overflow chain and reset the overflow HeapFile.
            3. Re-distribute records across pages, leaving 20% free space
               per page for future inserts before the next reorg is needed.
            4. Reuse already-allocated pages before allocating new ones.
               (Note: leftover pages are reset but not physically truncated.)

Classes:
    SeqRID: Identifies a record's location, either in the main chain or
            in the overflow file (delete() needs this distinction).
    SequentialFile: insert/delete/search/scan/reorganize over the above.
"""

import struct
import bisect

from storage.page import Page, PAGE_SIZE
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile
from storage.heap.rid import RID
from storage.heap.record_codec import encode_record, decode_record
from common.schema import Schema
from common.record import Record


HEADER_FORMAT = ">HIiI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

SLOT_FORMAT = ">BII"
SLOT_SIZE = struct.calcsize(SLOT_FORMAT)

STATUS_EMPTY = 0
STATUS_VALID = 1

NO_NEXT_PAGE = -1


class SeqRID(RID):
    """
    Identifies where a record physically lives: either a slot in the
    main sorted chain, or a record in the overflow HeapFile.

    Inherits from RID so it can be serialized by BTree nodes, which
    expect a standard RID. We use the highest bit of page_id (1 << 31)
    to encode whether this record is in the overflow file.
    """

    def __init__(self, in_overflow: bool, page_id: int = None, slot: int = None, overflow_rid: RID = None):
        if in_overflow:
            super().__init__(overflow_rid.page_id | (1 << 31), overflow_rid.slot)
        else:
            super().__init__(page_id, slot)
        
        self.in_overflow = in_overflow
        self.overflow_rid = overflow_rid

    @property
    def real_page_id(self):
        return self.page_id & ~(1 << 31)

    def __repr__(self):
        if self.in_overflow:
            return f"SeqRID(overflow={self.overflow_rid})"
        return f"SeqRID(page={self.real_page_id}, slot={self.slot})"


class SequentialFile:
    """
    Physically orders records by key_column, using a linked chain of
    pages plus an Overflow File for records that don't fit in their
    correct page at insertion time.

    Attributes:
        schema (Schema): Structure of the records stored in this file.
        key_column (str): Name of the column records are sorted by.
        bm (BufferManager): Buffer Manager for the main chain's file.
        overflow (HeapFile): Auxiliary unordered storage for records
            that overflowed their target page.
        reorganize_threshold (float): Wasted-space ratio (0.0-1.0) above
            which reorganize() should be triggered.
    """

    def __init__(
        self,
        schema: Schema,
        key_column: str,
        buffer_manager: BufferManager,
        overflow_buffer_manager: BufferManager,
        reorganize_threshold: float = 0.30,
    ):
        self.schema = schema
        self.key_column = key_column
        self._key_index = schema.column_index(key_column)
        self.bm = buffer_manager
        self.overflow = HeapFile(schema, overflow_buffer_manager)
        self.reorganize_threshold = reorganize_threshold

        # NOTE: this chain is rebuilt from scratch on cold start (empty
        # file -> single fresh page). It is NOT currently reconstructed
        # by re-scanning an existing overflow file across engine restarts
        # — persisting/recovering this chain robustly is left as a
        # follow-up, same category of limitation as Catalog's in-memory
        # _unique_values set.
        self._overflow_chain: dict[int, list[tuple]] = {}  # page_id -> [(key, RID), ...] sorted

        if self.bm.file_manager.page_count() == 0:
            self._head_page_id = self._allocate_and_init_page(next_page_id=NO_NEXT_PAGE)
        else:
            self._head_page_id = 0


    def insert(self, record: Record):
        """
        Inserts record in its correct sorted position. If the target
        page has room, it's placed there directly (shifting slot entries
        to keep the ItemId Array sorted by key). Otherwise, it's routed
        to the Overflow File and linked to that page.
        """
        if not record.validate(self.schema):
            raise ValueError(
                f"El registro no es válido para el schema de "
                f"'{self.schema.table_name}': {record}"
            )

        key = record[self._key_index].data
        payload = encode_record(record, self.schema)
        if len(payload) + SLOT_SIZE > PAGE_SIZE - HEADER_SIZE:
            raise ValueError(
                f"El registro serializado ({len(payload)} bytes) no cabe "
                "en una sola página"
            )

        page_id = self._find_target_page(key)
        page = self.bm.fetch_page(page_id)
        slot = self._try_insert_sorted(page, key, payload)
        self.bm.unpin_page(page_id, is_dirty=(slot is not None))

        if slot is not None:
            return SeqRID(in_overflow=False, page_id=page_id, slot=slot)

        overflow_rid = self.overflow.insert(record)
        chain = self._overflow_chain.setdefault(page_id, [])
        bisect.insort(chain, (key, overflow_rid))
        return SeqRID(in_overflow=True, overflow_rid=overflow_rid)

    def search(self, key) -> list[Record]:
        """
        Returns every record whose key_column equals key (duplicates
        are supported), looking both in the target page and in its
        overflow chain.
        """
        return self.search_in_page(self._find_target_page(key), key)

    def _is_overflow(self, rid: RID) -> bool:
        return getattr(rid, 'in_overflow', False) or bool(rid.page_id & (1 << 31))

    def _get_overflow_rid(self, rid: RID) -> RID:
        if getattr(rid, 'in_overflow', False) and getattr(rid, 'overflow_rid', None):
            return rid.overflow_rid
        return RID(rid.page_id & ~(1 << 31), rid.slot)

    def _get_real_page_id(self, rid: RID) -> int:
        return rid.page_id & ~(1 << 31)

    def delete(self, key_or_rid) -> int:
        """
        Lazily deletes every record matching key, or deletes a specific SeqRID.
        """
        if isinstance(key_or_rid, RID):
            rid = key_or_rid
            if self._is_overflow(rid):
                self.overflow.delete(self._get_overflow_rid(rid))
                return 1
            else:
                real_page_id = self._get_real_page_id(rid)
                page = self.bm.fetch_page(real_page_id)
                try:
                    status, a, b = self._read_slot(page, rid.slot)
                    if status == STATUS_VALID:
                        self._write_slot(page, rid.slot, STATUS_EMPTY, a, b)
                        return 1
                    return 0
                finally:
                    self.bm.unpin_page(real_page_id, is_dirty=True)
        
        key = key_or_rid
        page_id = self._find_target_page(key)
        deleted = 0

        page = self.bm.fetch_page(page_id)
        try:
            num_slots, _, _ = self._read_header(page)
            for i in range(num_slots):
                status, a, b = self._read_slot(page, i)
                if status != STATUS_VALID:
                    continue
                record = decode_record(page.read_bytes(a, b), self.schema)
                if record[self._key_index].data == key:
                    self._write_slot(page, i, STATUS_EMPTY, a, b)
                    deleted += 1
        finally:
            self.bm.unpin_page(page_id, is_dirty=(deleted > 0))

        chain = self._overflow_chain.get(page_id, [])
        remaining = []
        for chain_key, overflow_rid in chain:
            if chain_key == key:
                self.overflow.delete(overflow_rid)
                deleted += 1
            else:
                remaining.append((chain_key, overflow_rid))
        if chain:
            self._overflow_chain[page_id] = remaining

        # Automatically trigger reorganization if wasted space has grown
        # beyond the configured threshold (default 30%). This enforces the
        # acceptance criterion: "reorganize when the gap exceeds the threshold".
        if deleted > 0:
            self.maybe_reorganize()

        return deleted

    def update(self, rid: RID, new_record: Record) -> SeqRID:
        """
        Updates a record by its SeqRID (or plain RID). If it fits in place and the key is
        unchanged, updates in place. Otherwise, deletes and re-inserts.
        """
        if not new_record.validate(self.schema):
            raise ValueError(
                f"El registro no es válido para el schema de "
                f"'{self.schema.table_name}': {new_record}"
            )
        
        if self._is_overflow(rid):
            self.overflow.update(self._get_overflow_rid(rid), new_record)
            # Retain the SeqRID wrapper if it had one, otherwise reconstruct it
            if getattr(rid, 'in_overflow', False):
                return rid
            return SeqRID(in_overflow=True, overflow_rid=self._get_overflow_rid(rid))
        
        real_page_id = self._get_real_page_id(rid)
        page = self.bm.fetch_page(real_page_id)
        try:
            status, a, b = self._read_slot(page, rid.slot)
            payload = encode_record(new_record, self.schema)
            old_record = decode_record(page.read_bytes(a, b), self.schema)
            
            if len(payload) <= b and old_record[self._key_index].data == new_record[self._key_index].data:
                page.write_bytes(a, payload)
                self._write_slot(page, rid.slot, STATUS_VALID, a, len(payload))
                if getattr(rid, 'in_overflow', None) is not None:
                    return rid
                return SeqRID(in_overflow=False, page_id=real_page_id, slot=rid.slot)
        finally:
            self.bm.unpin_page(real_page_id, is_dirty=True)

        self.delete(rid)
        return self.insert(new_record)

    def get(self, rid: RID) -> Record | None:
        """
        Retrieves a record by its RID. Used by secondary indexes which deserialize
        plain RIDs instead of SeqRIDs.
        """
        if self._is_overflow(rid):
            return self.overflow.get(self._get_overflow_rid(rid))

        real_page_id = self._get_real_page_id(rid)
        page = self.bm.fetch_page(real_page_id)
        try:
            status, a, b = self._read_slot(page, rid.slot)
            if status == STATUS_VALID:
                return decode_record(page.read_bytes(a, b), self.schema)
            return None
        finally:
            self.bm.unpin_page(real_page_id, is_dirty=False)

    def scan(self):
        """
        Yields every live record across the whole file, in ascending
        key order. Merges each page's live slots with that page's
        overflow chain (both already individually sorted by key) using
        a standard two-pointer merge, then moves to the next page.
        """
        yield from self.scan_from(self._head_page_id)

    def scan_with_rid(self):
        """
        Yields (SeqRID, Record) for every live record in the file in ascending key order.
        """
        page_id = self._head_page_id
        while page_id != NO_NEXT_PAGE:
            page = self.bm.fetch_page(page_id)
            try:
                num_slots, _, next_page_id = self._read_header(page)
                main_entries = []
                for i in range(num_slots):
                    status, a, b = self._read_slot(page, i)
                    if status == STATUS_VALID:
                        record = decode_record(page.read_bytes(a, b), self.schema)
                        main_entries.append((record[self._key_index].data, SeqRID(in_overflow=False, page_id=page_id, slot=i), record))
            finally:
                self.bm.unpin_page(page_id, is_dirty=False)

            overflow_entries = []
            for k, overflow_rid in self._overflow_chain.get(page_id, []):
                record = self.overflow.get(overflow_rid)
                if record is not None:
                    overflow_entries.append((k, SeqRID(in_overflow=True, overflow_rid=overflow_rid), record))

            yield from self._merge_by_key_with_rid(main_entries, overflow_entries)
            page_id = next_page_id

    def _merge_by_key_with_rid(self, left: list, right: list):
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                yield left[i][1], left[i][2]
                i += 1
            else:
                yield right[j][1], right[j][2]
                j += 1
        while i < len(left):
            yield left[i][1], left[i][2]
            i += 1
        while j < len(right):
            yield right[j][1], right[j][2]
            j += 1

    def reorganize(self):
        """
        Rebuilds the sequential file from scratch: gathers every live
        record (main chain + overflow) via scan(), which already yields
        them in sorted order, then redistributes them evenly across
        pages, leaving ~20% free space per page for future inserts
        before the next reorganization is needed.

        LIMITATION: reuses the already-allocated pages by overwriting
        them and only allocates new ones if more are needed. It cannot
        physically shrink the file if fewer pages are needed than
        before — the leftover pages are simply reset to empty and left
        unused. True file truncation is out of scope for this entry.
        """
        all_records = list(self.scan())

        # Clear the overflow chain and file: everything is being
        # re-inserted directly into main pages below.
        self._overflow_chain.clear()
        self.overflow = HeapFile(self.schema, self.overflow.bm)

        usable_bytes_per_page = int((PAGE_SIZE - HEADER_SIZE) * 0.80)  # 20% slack reserved

        pages_needed = 1
        page_id = self._head_page_id
        self._reset_page(page_id, next_page_id=NO_NEXT_PAGE)

        current_page_id = page_id
        current_page = self.bm.fetch_page(current_page_id)
        used_in_page = 0

        for record in all_records:
            key = record[self._key_index].data
            payload = encode_record(record, self.schema)
            needed = payload_size = len(payload) + SLOT_SIZE

            if used_in_page + needed > usable_bytes_per_page:
                # Move to the next page, allocating a new one only if
                # we've run out of previously existing pages to reuse.
                pages_needed += 1
                next_id = self._next_reusable_page_id(pages_needed)
                self._write_header_next_page(current_page, next_id)
                self.bm.unpin_page(current_page_id, is_dirty=True)

                self._reset_page(next_id, next_page_id=NO_NEXT_PAGE)
                current_page_id = next_id
                current_page = self.bm.fetch_page(current_page_id)
                used_in_page = 0

            self._try_insert_sorted(current_page, key, payload)
            used_in_page += payload_size

        self.bm.unpin_page(current_page_id, is_dirty=True)

    def wasted_space_ratio(self) -> float:
        """
        Returns the fraction of total allocated space currently wasted:
        bytes held by EMPTY main-chain slots, plus everything sitting
        in the overflow file (since, ideally, none of it would exist
        after a reorganize()). Used to decide when to call reorganize().
        """
        wasted = 0
        total = 0

        page_id = self._head_page_id
        while page_id != NO_NEXT_PAGE:
            page = self.bm.fetch_page(page_id)
            try:
                num_slots, _, next_page_id = self._read_header(page)
                total += PAGE_SIZE
                for i in range(num_slots):
                    status, _, length = self._read_slot(page, i)
                    if status == STATUS_EMPTY:
                        wasted += length + SLOT_SIZE
            finally:
                self.bm.unpin_page(page_id, is_dirty=False)
            page_id = next_page_id

        # Use the actual number of pages allocated by the overflow HeapFile
        # instead of a per-record estimate. This is exact and avoids under/over
        # counting when record sizes vary significantly across the schema.
        overflow_pages = self.overflow.bm.file_manager.page_count()
        overflow_bytes = overflow_pages * PAGE_SIZE
        wasted += overflow_bytes
        total += overflow_bytes

        return wasted / total if total > 0 else 0.0

    def maybe_reorganize(self):
        """
        Convenience hook: checks wasted_space_ratio() against the
        configured threshold and reorganizes if it's exceeded. Intended
        to be called periodically (e.g. after a batch of inserts, or on
        a schedule) rather than after every single insert.
        """
        if self.wasted_space_ratio() > self.reorganize_threshold:
            self.reorganize()

    def _find_target_page(self, key) -> int:
        """
        Walks the main chain starting at the head, comparing key
        against each page's successor's minimum key, stopping at the
        page whose range should contain key.
        """
        page_id = self._head_page_id
        while True:
            page = self.bm.fetch_page(page_id)
            next_page_id = self._read_header(page)[2]
            self.bm.unpin_page(page_id, is_dirty=False)

            if next_page_id == NO_NEXT_PAGE:
                return page_id

            next_page = self.bm.fetch_page(next_page_id)
            next_min_key = self._min_key(next_page)
            self.bm.unpin_page(next_page_id, is_dirty=False)

            if next_min_key is None or key < next_min_key:
                return page_id

            page_id = next_page_id

    def _min_key(self, page: Page):
        """
        Returns the smallest key among this page's VALID slots (the
        first one found, since slots are kept sorted), or None if the
        page has no live records.
        """
        num_slots, _, _ = self._read_header(page)
        for i in range(num_slots):
            status, a, b = self._read_slot(page, i)
            if status == STATUS_VALID:
                record = decode_record(page.read_bytes(a, b), self.schema)
                return record[self._key_index].data
        return None

    def _live_entries(self, page: Page):
        """
        Yields (key, Record) for every VALID slot in page, in slot
        order (already sorted by key by construction).
        """
        num_slots, _, _ = self._read_header(page)
        for i in range(num_slots):
            status, a, b = self._read_slot(page, i)
            if status == STATUS_VALID:
                record = decode_record(page.read_bytes(a, b), self.schema)
                yield record[self._key_index].data, record

    def _merge_by_key(self, left: list, right: list):
        """
        Standard two-pointer merge of two (key, Record) lists, each
        already individually sorted by key, yielding Records in overall
        sorted order (classic building block of Sort-Merge Join / the
        merge phase of External Sorting).
        """
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                yield left[i][1]
                i += 1
            else:
                yield right[j][1]
                j += 1
        while i < len(left):
            yield left[i][1]
            i += 1
        while j < len(right):
            yield right[j][1]
            j += 1

    def _try_insert_sorted(self, page: Page, key, payload: bytes) -> int | None:
        """
        Attempts to insert payload (whose key is key) into page,
        keeping the ItemId Array sorted by key. Returns slot index on success,
        None if there isn't enough contiguous free space.
        """
        num_slots, free_space_offset, next_page_id = self._read_header(page)

        header_end = HEADER_SIZE + num_slots * SLOT_SIZE
        contiguous_free = free_space_offset - header_end
        if contiguous_free < SLOT_SIZE + len(payload):
            return None

        insert_at = 0
        for i in range(num_slots):
            status, a, b = self._read_slot(page, i)
            if status == STATUS_VALID:
                # Only VALID slots carry a meaningful key for ordering.
                # EMPTY slots are skipped: advancing insert_at past them
                # would break the invariant that VALID slots stay sorted.
                existing_key = decode_record(page.read_bytes(a, b), self.schema)[self._key_index].data
                if existing_key <= key:
                    insert_at = i + 1
                else:
                    break
            # EMPTY slots: do NOT advance insert_at; they are invisible
            # to the ordering logic and will be reclaimed by reorganize().

        for i in range(num_slots, insert_at, -1):
            status, a, b = self._read_slot(page, i - 1)
            self._write_slot(page, i, status, a, b)

        new_offset = free_space_offset - len(payload)
        page.write_bytes(new_offset, payload)
        self._write_slot(page, insert_at, STATUS_VALID, new_offset, len(payload))
        self._write_header(page, num_slots + 1, new_offset, next_page_id)
        return insert_at

    def _next_reusable_page_id(self, ordinal: int) -> int:
        """
        Returns the page_id to use for the ordinal-th page (1-based)
        of a fresh reorganize() pass: reuses an existing page_id if the
        file already has one at that position, otherwise allocates a
        brand-new page.
        """
        if ordinal - 1 < self.bm.file_manager.page_count():
            return ordinal - 1
        return self.bm.allocate_page()

    def _reset_page(self, page_id: int, next_page_id: int):
        page = self.bm.fetch_page(page_id)
        self._write_header(page, num_slots=0, free_space_offset=page.size, next_page_id=next_page_id)
        self.bm.unpin_page(page_id, is_dirty=True)

    def _write_header_next_page(self, page: Page, next_page_id: int):
        num_slots, free_space_offset, _ = self._read_header(page)
        self._write_header(page, num_slots, free_space_offset, next_page_id)

    def _allocate_and_init_page(self, next_page_id: int) -> int:
        page_id = self.bm.allocate_page()
        page = self.bm.fetch_page(page_id)
        self._write_header(page, num_slots=0, free_space_offset=page.size, next_page_id=next_page_id)
        self.bm.unpin_page(page_id, is_dirty=True)
        return page_id

    def _read_header(self, page: Page) -> tuple:
        num_slots, free_space_offset, next_page_id, _reserved = struct.unpack(
            HEADER_FORMAT, page.read_bytes(0, HEADER_SIZE)
        )
        return num_slots, free_space_offset, next_page_id

    def _write_header(self, page: Page, num_slots: int, free_space_offset: int, next_page_id: int):
        page.write_bytes(
            0, struct.pack(HEADER_FORMAT, num_slots, free_space_offset, next_page_id, 0)
        )

    def _read_slot(self, page: Page, index: int) -> tuple:
        offset = HEADER_SIZE + index * SLOT_SIZE
        return struct.unpack(SLOT_FORMAT, page.read_bytes(offset, SLOT_SIZE))

    def _write_slot(self, page: Page, index: int, status: int, a: int, b: int):
        offset = HEADER_SIZE + index * SLOT_SIZE
        page.write_bytes(offset, struct.pack(SLOT_FORMAT, status, a, b))

    def page_min_keys(self):
        page_id = self._head_page_id
        while page_id != NO_NEXT_PAGE:
            page = self.bm.fetch_page(page_id)
            try:
                min_key = self._min_key(page)
                next_page_id = self._read_header(page)[2]
            finally:
                self.bm.unpin_page(page_id, is_dirty=False)
            yield page_id, min_key
            page_id = next_page_id

    def search_in_page(self, page_id: int, key) -> list:
        results = []
        page = self.bm.fetch_page(page_id)
        try:
            for slot_key, record in self._live_entries(page):
                if slot_key == key:
                    results.append(record)
        finally:
            self.bm.unpin_page(page_id, is_dirty=False)

        for chain_key, overflow_rid in self._overflow_chain.get(page_id, []):
            if chain_key == key:
                record = self.overflow.get(overflow_rid)
                if record is not None:
                    results.append(record)
        return results

    def scan_from(self, page_id: int):
        while page_id != NO_NEXT_PAGE:
            page = self.bm.fetch_page(page_id)
            try:
                main_entries = list(self._live_entries(page))
                next_page_id = self._read_header(page)[2]
            finally:
                self.bm.unpin_page(page_id, is_dirty=False)

            overflow_entries = [
                (k, self.overflow.get(rid)) for k, rid in self._overflow_chain.get(page_id, [])
            ]
            overflow_entries = [(k, r) for k, r in overflow_entries if r is not None]

            yield from self._merge_by_key(main_entries, overflow_entries)
            page_id = next_page_id