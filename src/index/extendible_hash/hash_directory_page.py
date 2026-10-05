"""
Extendible Hashing Directory Page representation.

Classes:
    HashDirectoryPage: Represents the directory page in the extendible hashing index,
                       storing the global depth, local depths, and bucket pointers.
"""

import struct
from storage.page import PAGE_SIZE, Page

# The root page stores 512 entries and references additional pages for larger
# directories. Each extension page stores another 512 entries.
# Offset 0: global_depth (4 bytes)
MAX_BUCKETS = 512
GLOBAL_DEPTH_OFFSET = 0
LOCAL_DEPTHS_OFFSET = 4
BUCKET_PAGE_IDS_OFFSET = LOCAL_DEPTHS_OFFSET + MAX_BUCKETS
EXTENSION_PAGE_IDS_OFFSET = BUCKET_PAGE_IDS_OFFSET + MAX_BUCKETS * 4
MAX_EXTENSION_PAGES = (PAGE_SIZE - EXTENSION_PAGE_IDS_OFFSET) // 4
MAX_DIRECTORY_PAGES = MAX_EXTENSION_PAGES + 1
MAX_DIRECTORY_DEPTH = (MAX_DIRECTORY_PAGES * MAX_BUCKETS).bit_length() - 1

INVALID_PAGE_ID = 0xFFFFFFFF


class HashDirectoryPage:
    """
    Manages the physical layout of a hash directory page.
    """
    def __init__(self, page: Page, buffer_manager=None):
        self.page = page
        self.buffer_manager = buffer_manager

    def format_page(self):
        """
        Initializes a new directory page with default values.
        """
        self.set_global_depth(0)
        
        for i in range(MAX_BUCKETS):
            self.set_local_depth(i, 0)
            
        for i in range(MAX_BUCKETS):
            self.set_bucket_page_id(i, INVALID_PAGE_ID)

        for i in range(MAX_EXTENSION_PAGES):
            offset = EXTENSION_PAGE_IDS_OFFSET + i * 4
            self.page.write_bytes(offset, struct.pack(">I", INVALID_PAGE_ID))

    def get_global_depth(self) -> int:
        """
        Returns the global depth of the directory.
        """
        data = self.page.read_bytes(GLOBAL_DEPTH_OFFSET, 4)
        global_depth, = struct.unpack(">I", data)
        return global_depth

    def set_global_depth(self, global_depth: int):
        """
        Sets the global depth of the directory.
        """
        data = struct.pack(">I", global_depth)
        self.page.write_bytes(GLOBAL_DEPTH_OFFSET, data)

    def get_local_depth(self, bucket_idx: int) -> int:
        """
        Returns the local depth for a given bucket index.
        """
        page, local_idx, is_extension = self._entry_page(bucket_idx)
        offset = (0 if is_extension else LOCAL_DEPTHS_OFFSET) + local_idx
        data = page.read_bytes(offset, 1)
        local_depth, = struct.unpack(">B", data)
        if is_extension:
            self.buffer_manager.unpin_page(page.page_id)
        return local_depth

    def set_local_depth(self, bucket_idx: int, local_depth: int):
        """
        Sets the local depth for a given bucket index.
        """
        page, local_idx, is_extension = self._entry_page(bucket_idx)
        offset = (0 if is_extension else LOCAL_DEPTHS_OFFSET) + local_idx
        data = struct.pack(">B", local_depth)
        page.write_bytes(offset, data)
        if is_extension:
            self.buffer_manager.unpin_page(page.page_id, is_dirty=True)

    def get_bucket_page_id(self, bucket_idx: int) -> int:
        """
        Returns the page ID for a given bucket index.
        """
        page, local_idx, is_extension = self._entry_page(bucket_idx)
        offset = (MAX_BUCKETS if is_extension else BUCKET_PAGE_IDS_OFFSET) + local_idx * 4
        data = page.read_bytes(offset, 4)
        page_id, = struct.unpack(">I", data)
        if is_extension:
            self.buffer_manager.unpin_page(page.page_id)
        return page_id

    def set_bucket_page_id(self, bucket_idx: int, page_id: int):
        """
        Sets the page ID for a given bucket index.
        """
        page, local_idx, is_extension = self._entry_page(bucket_idx)
        offset = (MAX_BUCKETS if is_extension else BUCKET_PAGE_IDS_OFFSET) + local_idx * 4
        data = struct.pack(">I", page_id)
        page.write_bytes(offset, data)
        if is_extension:
            self.buffer_manager.unpin_page(page.page_id, is_dirty=True)

    def ensure_capacity(self, entry_count: int):
        """Allocate extension pages until the directory can hold entry_count slots."""
        max_entries = MAX_DIRECTORY_PAGES * MAX_BUCKETS
        if entry_count < 0 or entry_count > max_entries:
            raise ValueError(f"El directorio admite como máximo {max_entries} entradas")
        if self.buffer_manager is None:
            if entry_count > MAX_BUCKETS:
                raise ValueError("Se requiere BufferManager para extender el directorio")
            return

        required_pages = (entry_count + MAX_BUCKETS - 1) // MAX_BUCKETS
        for page_index in range(1, required_pages):
            offset = EXTENSION_PAGE_IDS_OFFSET + (page_index - 1) * 4
            page_id, = struct.unpack(">I", self.page.read_bytes(offset, 4))
            if page_id not in (0, INVALID_PAGE_ID):
                continue

            page_id = self.buffer_manager.allocate_page()
            page = self.buffer_manager.fetch_page(page_id)
            self._format_entries_page(page)
            self.buffer_manager.unpin_page(page_id, is_dirty=True)
            self.page.write_bytes(offset, struct.pack(">I", page_id))

    def _entry_page(self, bucket_idx: int) -> tuple[Page, int, bool]:
        max_entries = MAX_DIRECTORY_PAGES * MAX_BUCKETS
        if bucket_idx < 0 or bucket_idx >= max_entries:
            raise ValueError(f"Índice de directorio fuera de rango: {bucket_idx}")
        if bucket_idx < MAX_BUCKETS:
            return self.page, bucket_idx, False
        if self.buffer_manager is None:
            raise ValueError("Se requiere BufferManager para acceder al directorio extendido")

        page_index, local_idx = divmod(bucket_idx, MAX_BUCKETS)
        offset = EXTENSION_PAGE_IDS_OFFSET + (page_index - 1) * 4
        page_id, = struct.unpack(">I", self.page.read_bytes(offset, 4))
        if page_id in (0, INVALID_PAGE_ID):
            raise ValueError(f"La página de directorio {page_index} no está asignada")
        return self.buffer_manager.fetch_page(page_id), local_idx, True

    @staticmethod
    def _format_entries_page(page: Page):
        for i in range(MAX_BUCKETS):
            page.write_bytes(i, b"\x00")
            page.write_bytes(MAX_BUCKETS + i * 4, struct.pack(">I", INVALID_PAGE_ID))

    def get_split_image_index(self, bucket_idx: int) -> int:
        """
        Returns the split image index for a given bucket index.
        """
        local_depth = self.get_local_depth(bucket_idx)
        split_mask = 1 << (local_depth - 1)
        return bucket_idx ^ split_mask
