"""
Extendible Hashing Directory Page representation.

Classes:
    HashDirectoryPage: Represents the directory page in the extendible hashing index,
                       storing the global depth, local depths, and bucket pointers.
"""

import struct
from storage.page import Page

# Directory Page Layout for 4096-byte pages:
# Supports up to a global_depth of 9 (512 buckets).
# Offset 0: global_depth (4 bytes)
# Offset 4: local_depths (512 bytes, 1 byte per bucket)
# Offset 516: bucket_page_ids (2048 bytes, 4 bytes per bucket)
MAX_BUCKETS = 512
GLOBAL_DEPTH_OFFSET = 0
LOCAL_DEPTHS_OFFSET = 4
BUCKET_PAGE_IDS_OFFSET = LOCAL_DEPTHS_OFFSET + MAX_BUCKETS

INVALID_PAGE_ID = 0xFFFFFFFF


class HashDirectoryPage:
    """
    Manages the physical layout of a hash directory page.
    """
    def __init__(self, page: Page):
        self.page = page

    def format_page(self):
        """
        Initializes a new directory page with default values.
        """
        self.set_global_depth(0)
        
        for i in range(MAX_BUCKETS):
            self.set_local_depth(i, 0)
            
        for i in range(MAX_BUCKETS):
            self.set_bucket_page_id(i, INVALID_PAGE_ID)

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
        offset = LOCAL_DEPTHS_OFFSET + bucket_idx
        data = self.page.read_bytes(offset, 1)
        local_depth, = struct.unpack(">B", data)
        return local_depth

    def set_local_depth(self, bucket_idx: int, local_depth: int):
        """
        Sets the local depth for a given bucket index.
        """
        offset = LOCAL_DEPTHS_OFFSET + bucket_idx
        data = struct.pack(">B", local_depth)
        self.page.write_bytes(offset, data)

    def get_bucket_page_id(self, bucket_idx: int) -> int:
        """
        Returns the page ID for a given bucket index.
        """
        offset = BUCKET_PAGE_IDS_OFFSET + (bucket_idx * 4)
        data = self.page.read_bytes(offset, 4)
        page_id, = struct.unpack(">I", data)
        return page_id

    def set_bucket_page_id(self, bucket_idx: int, page_id: int):
        """
        Sets the page ID for a given bucket index.
        """
        offset = BUCKET_PAGE_IDS_OFFSET + (bucket_idx * 4)
        data = struct.pack(">I", page_id)
        self.page.write_bytes(offset, data)

    def get_split_image_index(self, bucket_idx: int) -> int:
        """
        Returns the split image index for a given bucket index.
        """
        local_depth = self.get_local_depth(bucket_idx)
        split_mask = 1 << (local_depth - 1)
        return bucket_idx ^ split_mask
