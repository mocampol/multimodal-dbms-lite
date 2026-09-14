"""
Buffer Manager: mediates all page access between the storage layer
(heap_file, sequential_file) and disk (via FileManager), minimizing I/O.

Implements, following Ramakrishnan's structure directly:
    - Buffer Pool: a fixed-size array of frames.
    - Page Table: page_id -> frame_number mapping, with pin_count
      and dirty_bit tracked per frame.
    - Dirty Pages: flushed to disk before being reused or on demand.
    - Replacement Policy: pluggable; starts as a simple "first unpinned
      frame" policy so heap_file/sequential_file can be built without
      waiting on LRU-K, and is swapped later without touching callers.

Classes:
    Frame: One slot of the Buffer Pool, holding a Page plus its metadata.
    BufferManager: Public interface used by heap_file and sequential_file.
"""

from storage.page import Page
from storage.file_manager.file_manager import FileManager


class Frame:
    """
    One slot of the Buffer Pool.

    Attributes:
        page (Page | None): The page currently occupying this frame,
            or None if the frame is empty.
        pin_count (int): Number of active consumers using this page.
            While pin_count > 0, this frame can NEVER be chosen as a
            replacement victim, regardless of what the replacement
            policy says.
        dirty_bit (bool): True if the page was modified in memory
            since it was last written to disk.
    """

    def __init__(self):
        self.page: Page | None = None
        self.pin_count: int = 0
        self.dirty_bit: bool = False

    def is_empty(self) -> bool:
        return self.page is None

    def reset(self):
        """
        Clears this frame so it can hold a different page.
        Only safe to call when pin_count == 0.
        """
        self.page = None
        self.pin_count = 0
        self.dirty_bit = False

    def __repr__(self):
        pid = self.page.page_id if self.page else None
        return f"Frame(page_id={pid}, pin_count={self.pin_count}, dirty={self.dirty_bit})"


class BufferManager:
    """
    Manages a fixed-size Buffer Pool of frames for a single table's
    FileManager, satisfying page requests from heap_file/sequential_file
    while minimizing disk I/O.

    Attributes:
        file_manager (FileManager): Underlying disk access for this table.
        pool_size (int): Number of frames in the Buffer Pool.
        frames (list[Frame]): The Buffer Pool itself.
        page_table (dict[int, int]): Maps page_id -> frame index, for O(1) lookup.
    """

    def __init__(self, file_manager: FileManager, pool_size: int = 64):
        self.file_manager = file_manager
        self.pool_size = pool_size
        self.frames: list[Frame] = [Frame() for _ in range(pool_size)]
        self.page_table: dict[int, int] = {}

    # ---------- core API used by heap_file / sequential_file ----------

    def fetch_page(self, page_id: int) -> Page:
        """
        Returns the requested page, pinning it in the Buffer Pool.

        Cache hit: page_id is already in the Page Table -> increments
        pin_count and returns the cached Page directly (no disk I/O).

        Cache miss: finds a free frame or a replacement victim, evicting
        it (flushing first if dirty), then loads the page from disk via
        FileManager, registers it in the Page Table with pin_count=1
        and dirty_bit=False.

        Every successful fetch_page() MUST be matched by exactly one
        unpin_page() call once the caller is done with the page —
        otherwise the frame stays pinned forever and the effective
        pool size shrinks permanently.
        """
        if page_id in self.page_table:
            frame_idx = self.page_table[page_id]
            frame = self.frames[frame_idx]
            frame.pin_count += 1
            return frame.page

        frame_idx = self._find_free_or_victim_frame()
        frame = self.frames[frame_idx]

        if not frame.is_empty():
            self._evict(frame_idx)

        page = self.file_manager.read_page(page_id)
        frame.page = page
        frame.pin_count = 1
        frame.dirty_bit = False
        self.page_table[page_id] = frame_idx

        return page

    def unpin_page(self, page_id: int, is_dirty: bool = False):
        """
        Signals that the caller is done using page_id for now.

        Decrements pin_count (never below 0). If is_dirty is True, marks
        the frame as dirty — once a frame is dirty it stays dirty until
        the next flush, even if a later unpin passes is_dirty=False.
        """
        frame_idx = self._require_frame(page_id)
        frame = self.frames[frame_idx]

        if frame.pin_count > 0:
            frame.pin_count -= 1

        if is_dirty:
            frame.dirty_bit = True

    def flush_page(self, page_id: int):
        """
        Writes page_id to disk if it is dirty, then clears its dirty_bit.
        Does nothing if the page isn't currently cached or isn't dirty.
        """
        if page_id not in self.page_table:
            return

        frame = self.frames[self.page_table[page_id]]
        if frame.dirty_bit:
            self.file_manager.write_page(frame.page)
            frame.dirty_bit = False

    def flush_all(self):
        """
        Flushes every dirty frame currently in the Buffer Pool to disk.
        Used for checkpoints or an orderly engine shutdown.
        """
        for frame in self.frames:
            if frame.page is not None and frame.dirty_bit:
                self.file_manager.write_page(frame.page)
                frame.dirty_bit = False

    def allocate_page(self) -> int:
        """
        Allocates a brand-new page on disk (via FileManager) and returns
        its page_id. The page is NOT automatically pinned into the
        Buffer Pool — callers that need to write to it right away should
        follow up with fetch_page(page_id).
        """
        return self.file_manager.allocate_page()

    def _require_frame(self, page_id: int) -> int:
        if page_id not in self.page_table:
            raise ValueError(
                f"page_id={page_id} no está actualmente en el Buffer Pool "
                "(¿olvidaste hacer fetch_page primero?)"
            )
        return self.page_table[page_id]

    def _find_free_or_victim_frame(self) -> int:
        """
        Returns the index of a frame to use for a new page: an empty
        frame if one exists, otherwise the victim chosen by the
        replacement policy. Raises RuntimeError if the pool is full
        and every frame is currently pinned (nothing can be evicted).
        """
        for i, frame in enumerate(self.frames):
            if frame.is_empty():
                return i

        victim_idx = self._choose_victim()
        if victim_idx is None:
            raise RuntimeError(
                "Buffer Pool lleno: todos los frames están pinneados, "
                "no hay ninguna víctima disponible para reemplazo"
            )
        return victim_idx

    def _choose_victim(self) -> int | None:
        """
        Replacement policy hook. Current implementation: returns the
        index of the first frame with pin_count == 0 (arbitrary, but
        correct — never picks a pinned frame). This is a deliberate
        placeholder so heap_file/sequential_file can be developed
        without waiting on LRU-K; replace ONLY this method's body with
        the LRU-K logic later, no other code in this class needs to change.

        Returns None if no unpinned frame exists (pool exhausted).
        """
        for i, frame in enumerate(self.frames):
            if not frame.is_empty() and frame.pin_count == 0:
                return i
        return None

    def _evict(self, frame_idx: int):
        """
        Removes the current occupant of frame_idx from the Page Table,
        flushing it to disk first if it's dirty.
        """
        frame = self.frames[frame_idx]
        if frame.dirty_bit:
            self.file_manager.write_page(frame.page)

        del self.page_table[frame.page.page_id]
        frame.reset()

    def __repr__(self):
        used = sum(1 for f in self.frames if not f.is_empty())
        return f"BufferManager(pool_size={self.pool_size}, used={used})"
