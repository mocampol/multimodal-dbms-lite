"""
Extendible Hashing Index implementation.

Classes:
    ExtendibleHashIndex: The main index class that coordinates the Directory Page 
                         and the Bucket Pages using the Buffer Manager.
"""

from storage.buffer_manager import BufferManager
from common.value import DataType, Value
from storage.heap.rid import RID
from index.extendible_hash.hash_directory_page import HashDirectoryPage
from index.extendible_hash.hash_bucket_page import HashBucketPage

class ExtendibleHashIndex:
    """
    Implements an Extendible Hashing index.
    """
    def __init__(self, buffer_manager: BufferManager, directory_page_id: int, key_type: DataType):
        self.buffer_manager = buffer_manager
        self.directory_page_id = directory_page_id
        self.key_type = key_type

    def insert(self, key: Value, rid: RID) -> bool:
        """
        Inserts a key-RID pair into the index.
        """
        hash_value = self._hash_key(key)
        
        # 1. Fetch directory
        dir_page = self.buffer_manager.fetch_page(self.directory_page_id)
        directory = HashDirectoryPage(dir_page)
        
        # 2. Get target bucket page ID
        bucket_idx = self._get_bucket_idx(hash_value, directory.get_global_depth())
        bucket_page_id = directory.get_bucket_page_id(bucket_idx)
        
        # 3. Fetch bucket
        bucket_page = self.buffer_manager.fetch_page(bucket_page_id)
        bucket = HashBucketPage(bucket_page, self.key_type)
        
        # 4. Try to insert
        success = bucket.insert(key, rid)
        
        if success:
            self.buffer_manager.unpin_page(bucket_page_id, True)
            self.buffer_manager.unpin_page(self.directory_page_id, False)
            return True
            
        # 5. Overflow handling: bucket is full, we must split
        local_depth = bucket.get_local_depth()
        global_depth = directory.get_global_depth()
        
        # Grow directory if local_depth == global_depth
        if local_depth == global_depth:
            self._grow_directory(directory)
            
        # Allocate a new bucket page
        new_bucket_page = self.buffer_manager.new_page()
        new_bucket_page_id = new_bucket_page.page_id
        new_bucket = HashBucketPage(new_bucket_page, self.key_type)
        
        new_local_depth = local_depth + 1
        new_bucket.format_page(new_local_depth)
        
        # Fetch existing entries to redistribute
        entries = bucket.get_all_entries()
        
        # Clear the old bucket, update its local depth
        bucket.format_page(new_local_depth)
        
        # Update directory pointers
        split_bit = 1 << local_depth
        dir_size = 1 << directory.get_global_depth()
        
        for i in range(dir_size):
            if directory.get_bucket_page_id(i) == bucket_page_id:
                # If the split bit is 1, it points to the new bucket
                if (i & split_bit) != 0:
                    directory.set_bucket_page_id(i, new_bucket_page_id)
                # Update the depth in the directory
                directory.set_local_depth(i, new_local_depth)
                
        # Redistribute the entries between the old and new bucket
        for e_key, e_rid in entries:
            e_hash = self._hash_key(e_key)
            e_idx = self._get_bucket_idx(e_hash, directory.get_global_depth())
            e_target_page_id = directory.get_bucket_page_id(e_idx)
            
            if e_target_page_id == bucket_page_id:
                bucket.insert(e_key, e_rid)
            else:
                new_bucket.insert(e_key, e_rid)
                
        # Unpin all pages used during split
        self.buffer_manager.unpin_page(bucket_page_id, True)
        self.buffer_manager.unpin_page(new_bucket_page_id, True)
        self.buffer_manager.unpin_page(self.directory_page_id, True)
        
        # Retry inserting the new key recursively
        return self.insert(key, rid)

    def search(self, key: Value) -> list[RID]:
        """
        Searches for a key and returns a list of its RIDs.
        """
        hash_value = self._hash_key(key)
        
        dir_page = self.buffer_manager.fetch_page(self.directory_page_id)
        directory = HashDirectoryPage(dir_page)
        
        bucket_idx = self._get_bucket_idx(hash_value, directory.get_global_depth())
        bucket_page_id = directory.get_bucket_page_id(bucket_idx)
        
        # Unpin directory early since we only needed to read the page id
        self.buffer_manager.unpin_page(self.directory_page_id, False)
        
        bucket_page = self.buffer_manager.fetch_page(bucket_page_id)
        bucket = HashBucketPage(bucket_page, self.key_type)
        
        results = bucket.search(key)
        
        self.buffer_manager.unpin_page(bucket_page_id, False)
        
        return results

    def remove(self, key: Value, rid: RID) -> bool:
        """
        Removes a key-RID pair from the index.
        """
        hash_value = self._hash_key(key)
        
        dir_page = self.buffer_manager.fetch_page(self.directory_page_id)
        directory = HashDirectoryPage(dir_page)
        
        bucket_idx = self._get_bucket_idx(hash_value, directory.get_global_depth())
        bucket_page_id = directory.get_bucket_page_id(bucket_idx)
        
        self.buffer_manager.unpin_page(self.directory_page_id, False)
        
        bucket_page = self.buffer_manager.fetch_page(bucket_page_id)
        bucket = HashBucketPage(bucket_page, self.key_type)
        
        success = bucket.remove(key, rid)
        
        self.buffer_manager.unpin_page(bucket_page_id, success)
        
        return success

    # ==========================================
    # PRIVATE HELPER METHODS
    # ==========================================

    def _hash_key(self, key: Value) -> int:
        """
        Returns a 32-bit positive integer hash of the key's data.
        """
        return abs(hash(key.data)) & 0xFFFFFFFF

    def _get_bucket_idx(self, hash_value: int, global_depth: int) -> int:
        """
        Calculates the bucket index for a given hash value and global depth.
        Keeps only the lowest `global_depth` bits of the hash value.
        """
        mask = (1 << global_depth) - 1
        return hash_value & mask

    def _grow_directory(self, directory: HashDirectoryPage):
        """
        Doubles the size of the directory.
        """
        global_depth = directory.get_global_depth()
        current_size = 1 << global_depth
        
        for i in range(current_size):
            page_id = directory.get_bucket_page_id(i)
            local_depth = directory.get_local_depth(i)
            
            directory.set_bucket_page_id(i + current_size, page_id)
            directory.set_local_depth(i + current_size, local_depth)
            
        directory.set_global_depth(global_depth + 1)
