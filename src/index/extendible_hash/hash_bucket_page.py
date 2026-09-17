"""
Extendible Hashing Bucket Page representation.

Classes:
    HashBucketPage: Represents a bucket page in the extendible hashing index,
                    handling insertion, search, and deletion of key-RID pairs.
"""

import struct

from storage.page import Page
from storage.heap.rid import RID
from common.value import DataType, Value
from storage.heap.record_codec import encode_scalar, decode_scalar

# Header Layout: local_depth (2 bytes), num_slots (2 bytes), free_space_offset (4 bytes)
HEADER_FORMAT = ">HHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Slot Layout: status (1 byte), offset (4 bytes), length (4 bytes)
SLOT_FORMAT = ">BII"
SLOT_SIZE = struct.calcsize(SLOT_FORMAT)

STATUS_EMPTY = 0
STATUS_VALID = 1


class HashBucketPage:
    """
    Manages the physical layout of a hash bucket page.
    """
    def __init__(self, page: Page, key_type: DataType):
        self.page = page
        self.key_type = key_type

    def format_page(self, local_depth: int):
        """
        Formats a new bucket page with the given local depth.
        """
        self._write_header(local_depth, 0, self.page.size)

    def get_local_depth(self) -> int:
        """
        Returns the local depth of this bucket.
        """
        local_depth, _, _ = self._read_header()
        return local_depth

    def set_local_depth(self, local_depth: int):
        """
        Sets the local depth of this bucket.
        """
        _, num_slots, free_space = self._read_header()
        self._write_header(local_depth, num_slots, free_space)

    def insert(self, key: Value, rid: RID) -> bool:
        """
        Inserts a key-RID pair into the bucket.
        Returns True if successful, False if the bucket is full (overflow).
        """
        local_depth, num_slots, free_space_offset = self._read_header()
        
        key_bytes = encode_scalar(key)
        rid_bytes = struct.pack(">II", rid.page_id, rid.slot)
        
        payload = key_bytes + rid_bytes
        payload_len = len(payload)
        
        header_end = HEADER_SIZE + num_slots * SLOT_SIZE
        contiguous_free = free_space_offset - header_end 
        
        # Try to recycle an empty slot that is large enough
        for slot in range(num_slots):
            status, offset, length = self._read_slot(slot)
            if status == STATUS_EMPTY and length >= payload_len:
                self.page.write_bytes(offset, payload)
                self._write_slot(slot, STATUS_VALID, offset, payload_len)
                return True
                
        # If no slot is recycled, check for enough contiguous free space
        if contiguous_free < SLOT_SIZE + payload_len:
            return False

        new_offset = free_space_offset - payload_len
        self.page.write_bytes(new_offset, payload)
        self._write_slot(num_slots, STATUS_VALID, new_offset, payload_len)
        self._write_header(local_depth, num_slots + 1, new_offset)

        return True

    def search(self, key: Value) -> list[RID]:
        """
        Searches for a key and returns a list of matching RIDs.
        """
        results = []
        _, num_slots, _ = self._read_header()
        
        for slot in range(num_slots):
            status, offset, length = self._read_slot(slot)
            if status == STATUS_VALID:
                payload = self.page.read_bytes(offset, length)
                key_len = length - 8
                key_bytes = payload[:key_len]

                stored_key_data = decode_scalar(self.key_type, key_bytes)
                if stored_key_data == key.data:
                    rid_bytes = payload[key_len:]
                    rid_page, rid_slot = struct.unpack(">II", rid_bytes)
                    results.append(RID(rid_page, rid_slot))
                    
        return results

    def remove(self, key: Value, rid: RID) -> bool:
        """
        Removes a specific key-RID pair from the bucket.
        Returns True if removed, False if not found.
        """
        _, num_slots, _ = self._read_header()

        for slot in range(num_slots):
            status, offset, length = self._read_slot(slot)
            if status == STATUS_VALID:
                payload = self.page.read_bytes(offset, length)
                key_len = length - 8
                key_bytes = payload[:key_len]
                stored_key_data = decode_scalar(self.key_type, key_bytes)
                
                if stored_key_data == key.data:
                    rid_bytes = payload[key_len:]
                    rid_page, rid_slot = struct.unpack(">II", rid_bytes)
                    
                    if rid_page == rid.page_id and rid_slot == rid.slot:
                        # Mark the slot as empty
                        self._write_slot(slot, STATUS_EMPTY, offset, length)
                        return True
                        
        return False

    def get_all_entries(self) -> list[tuple[Value, RID]]:
        """
        Retrieves all valid key-RID pairs in the bucket.
        """
        results = []
        _, num_slots, _ = self._read_header()
        
        for slot in range(num_slots):
            status, offset, length = self._read_slot(slot)
            
            if status == STATUS_VALID:
                payload = self.page.read_bytes(offset, length)
                key_len = length - 8
                key_bytes = payload[:key_len]
                
                stored_key_data = decode_scalar(self.key_type, key_bytes)
                stored_key = Value(self.key_type, stored_key_data)
                
                rid_bytes = payload[key_len:]
                rid_page, rid_slot = struct.unpack(">II", rid_bytes)
                
                results.append((stored_key, RID(rid_page, rid_slot)))
                
        return results

    # ==========================================
    # PRIVATE METHODS (Byte Helpers)
    # ==========================================

    def _read_header(self) -> tuple[int, int, int]:
        return struct.unpack(HEADER_FORMAT, self.page.read_bytes(0, HEADER_SIZE))

    def _write_header(self, local_depth: int, num_slots: int, free_space_offset: int):
        self.page.write_bytes(0, struct.pack(
            HEADER_FORMAT, local_depth, num_slots, free_space_offset))

    def _read_slot(self, index: int) -> tuple[int, int, int]:
        offset = HEADER_SIZE + index * SLOT_SIZE
        return struct.unpack(SLOT_FORMAT, self.page.read_bytes(offset, SLOT_SIZE))

    def _write_slot(self, index: int, status: int, offset_val: int, length: int):
        offset = HEADER_SIZE + index * SLOT_SIZE
        self.page.write_bytes(offset, struct.pack(
            SLOT_FORMAT, status, offset_val, length))
