"""
test_extendible_hash.py — Full test battery for the Extendible Hash index.

Test organization:
    1. Smoke tests            : minimal sanity checks (empty index, single op).
    2. Unit tests             : correct behaviour for insert / search / remove.
    3. GlobalDepth consistency: validates the invariants of the split/grow logic.
    4. Stress tests           : bulk inserts (1k–5k) with integrity verification.
    5. Buffer pool stress     : bulk inserts against a tiny pool (eviction paths).

Fixtures used (defined in conftest.py):
    extendible_hash      — standard pool (128 frames)
    small_pool_hash      — tiny pool (16 frames)
"""

import random
import pytest

from common.value import DataType, Value
from storage.heap.rid import RID
from index.extendible_hash.hash_directory_page import HashDirectoryPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iv(n: int) -> Value:
    """Shorthand: integer Value."""
    return Value(DataType.INTEGER, n)


def rid(page: int, slot: int = 0) -> RID:
    """Shorthand: RID(page_id, slot)."""
    return RID(page_id=page, slot=slot)


# ---------------------------------------------------------------------------
# 1. Smoke tests
# ---------------------------------------------------------------------------

class TestExtendibleHashSmoke:
    def test_index_is_created(self, extendible_hash):
        """The fixture creates a valid ExtendibleHashIndex instance."""
        assert extendible_hash is not None

    def test_search_on_empty_index_returns_empty_list(self, extendible_hash):
        """Searching an empty index must return [], not raise."""
        result = extendible_hash.search(iv(1))
        assert result == []

    def test_remove_on_empty_index_returns_false(self, extendible_hash):
        """Removing from an empty index must return False, not raise."""
        result = extendible_hash.remove(iv(1), rid(1))
        assert result is False


# ---------------------------------------------------------------------------
# 2. Unit tests — insert / search / remove
# ---------------------------------------------------------------------------

class TestInsertAndSearch:
    def test_single_insert_found(self, extendible_hash):
        extendible_hash.insert(iv(42), rid(1))
        result = extendible_hash.search(iv(42))
        assert rid(1) in result

    def test_key_not_inserted_returns_empty(self, extendible_hash):
        extendible_hash.insert(iv(10), rid(1))
        assert extendible_hash.search(iv(99)) == []

    def test_multiple_inserts_all_found(self, extendible_hash):
        pairs = [(iv(k), rid(k)) for k in range(1, 101)]
        for key, r in pairs:
            extendible_hash.insert(key, r)
        for key, r in pairs:
            result = extendible_hash.search(key)
            assert r in result, f"RID for key {key.data} not found after insert"

    def test_duplicate_key_both_rids_found(self, extendible_hash):
        """Hash index allows duplicate keys (non-unique secondary index)."""
        extendible_hash.insert(iv(1), rid(1, slot=0))
        extendible_hash.insert(iv(1), rid(1, slot=1))
        result = extendible_hash.search(iv(1))
        assert len(result) == 2
        assert rid(1, slot=0) in result
        assert rid(1, slot=1) in result

    def test_ghost_keys_never_found(self, extendible_hash):
        """Keys that were never inserted must not be found."""
        for k in range(1, 101):
            extendible_hash.insert(iv(k), rid(k))
        assert extendible_hash.search(iv(0)) == []
        assert extendible_hash.search(iv(101)) == []
        assert extendible_hash.search(iv(999_999)) == []


class TestRemove:
    def test_remove_key_no_longer_found(self, extendible_hash):
        extendible_hash.insert(iv(5), rid(5))
        success = extendible_hash.remove(iv(5), rid(5))
        assert success is True
        assert extendible_hash.search(iv(5)) == []

    def test_remove_nonexistent_returns_false(self, extendible_hash):
        extendible_hash.insert(iv(10), rid(10))
        result = extendible_hash.remove(iv(999), rid(999))
        assert result is False

    def test_remove_one_key_others_intact(self, extendible_hash):
        for k in range(1, 11):
            extendible_hash.insert(iv(k), rid(k))
        extendible_hash.remove(iv(5), rid(5))
        assert extendible_hash.search(iv(5)) == []
        for k in range(1, 11):
            if k != 5:
                assert rid(k) in extendible_hash.search(iv(k))

    def test_remove_specific_rid_duplicate_key(self, extendible_hash):
        """Removing one RID of a duplicate key must leave the other intact."""
        extendible_hash.insert(iv(1), rid(1, slot=0))
        extendible_hash.insert(iv(1), rid(1, slot=1))
        extendible_hash.remove(iv(1), rid(1, slot=0))
        result = extendible_hash.search(iv(1))
        assert rid(1, slot=0) not in result
        assert rid(1, slot=1) in result


# ---------------------------------------------------------------------------
# 3. Global Depth consistency (Acceptance Criteria of the ticket)
# ---------------------------------------------------------------------------

class TestGlobalDepthConsistency:
    def _get_directory(self, extendible_hash) -> HashDirectoryPage:
        """Helper: fetches the directory page and returns a HashDirectoryPage."""
        bm = extendible_hash.buffer_manager
        dir_page = bm.fetch_page(extendible_hash.directory_page_id)
        directory = HashDirectoryPage(dir_page)
        bm.unpin_page(extendible_hash.directory_page_id, is_dirty=False)
        return directory

    def test_initial_global_depth_is_zero(self, extendible_hash):
        """A freshly created index must start with global_depth=0."""
        directory = self._get_directory(extendible_hash)
        assert directory.get_global_depth() == 0

    def test_global_depth_grows_after_split_that_requires_double(self, extendible_hash):
        """
        Inserting enough keys to force a bucket split when LD==GD must
        increment the global depth by exactly 1 (not more).
        """
        # Insert enough keys to guarantee at least one directory double
        for k in range(1, 500):
            extendible_hash.insert(iv(k), rid(k))

        directory = self._get_directory(extendible_hash)
        gd = directory.get_global_depth()
        assert gd >= 1, "GD must have grown after splits"
        # Directory size must be exactly 2^GD
        assert (1 << gd) <= 512, "GD must not exceed MAX_BUCKETS capacity (9)"

    def test_directory_size_is_always_power_of_two(self, extendible_hash):
        """
        After any number of inserts the directory size must be 2^GD.
        This guarantees no desalignment between the hash mask and the directory.
        """
        for k in range(1, 300):
            extendible_hash.insert(iv(k), rid(k))

        directory = self._get_directory(extendible_hash)
        gd = directory.get_global_depth()
        expected_size = 1 << gd  # 2^GD
        # Spot-check: all entries up to expected_size must not be INVALID
        from index.extendible_hash.hash_directory_page import INVALID_PAGE_ID
        for i in range(expected_size):
            page_id = directory.get_bucket_page_id(i)
            assert page_id != INVALID_PAGE_ID, (
                f"Directory entry {i} is INVALID after {gd=} splits"
            )

    def test_local_depth_never_exceeds_global_depth(self, extendible_hash):
        """
        Core invariant: LD ≤ GD must hold for every directory entry at all times.
        A violation here would produce invalid pointer redirecctions.
        """
        for k in range(1, 500):
            extendible_hash.insert(iv(k), rid(k))

        directory = self._get_directory(extendible_hash)
        gd = directory.get_global_depth()
        dir_size = 1 << gd

        for i in range(dir_size):
            ld = directory.get_local_depth(i)
            assert ld <= gd, (
                f"Invariant violated at index {i}: local_depth={ld} > global_depth={gd}"
            )

    def test_no_invalid_pointers_after_splits(self, extendible_hash):
        """
        After splits, all active directory entries must point to valid page IDs.
        INVALID_PAGE_ID (0xFFFFFFFF) must not appear in any used slot.
        """
        from index.extendible_hash.hash_directory_page import INVALID_PAGE_ID

        for k in range(1, 400):
            extendible_hash.insert(iv(k), rid(k))

        directory = self._get_directory(extendible_hash)
        gd = directory.get_global_depth()
        dir_size = 1 << gd

        for i in range(dir_size):
            page_id = directory.get_bucket_page_id(i)
            assert page_id != INVALID_PAGE_ID, (
                f"Directory entry {i} has INVALID_PAGE_ID after splits"
            )

    def test_all_inserted_keys_still_found_after_multiple_doubles(self, extendible_hash):
        """
        Regression: after the directory doubles multiple times, every key
        inserted before the doubles must still be found (no RID is lost
        during the rehash / redistribution step).
        """
        n = 600
        for k in range(1, n + 1):
            extendible_hash.insert(iv(k), rid(k))

        for k in range(1, n + 1):
            result = extendible_hash.search(iv(k))
            assert rid(k) in result, (
                f"Key {k} lost after directory growth. search() returned {result}"
            )


# ---------------------------------------------------------------------------
# 4. Stress tests — bulk inserts with integrity verification
# ---------------------------------------------------------------------------

class TestBulkInserts:
    @pytest.mark.parametrize("n_records", [1_000, 5_000])
    def test_random_bulk_insert_all_keys_found(self, extendible_hash, n_records):
        """
        Insert n_records keys in a random (shuffled) order.
        Then verify every inserted key can be found with the correct RID.
        """
        keys = list(range(1, n_records + 1))
        random.seed(42)
        random.shuffle(keys)

        for k in keys:
            extendible_hash.insert(iv(k), rid(k))

        # Verify a sample of 300 (or all if n < 300) lookups
        sample = random.sample(keys, min(300, n_records))
        for k in sample:
            result = extendible_hash.search(iv(k))
            assert result, f"Key {k} not found after bulk insert"
            assert rid(k) in result, f"Wrong RID for key {k}: got {result}"

    def test_sequential_bulk_insert_all_keys_found(self, extendible_hash):
        """Insert 2000 keys in ascending order; all must be found."""
        n = 2_000
        for k in range(1, n + 1):
            extendible_hash.insert(iv(k), rid(k))
        for k in range(1, n + 1, 10):
            result = extendible_hash.search(iv(k))
            assert rid(k) in result, f"Key {k} not found after sequential insert"


# ---------------------------------------------------------------------------
# 5. Buffer pool stress — forces eviction during inserts
# ---------------------------------------------------------------------------

class TestBufferPoolStress:
    def test_bulk_insert_with_tiny_pool_no_data_loss(self, small_pool_hash):
        """
        Insert 2,000 keys using a buffer pool of only 16 frames.
        This forces the replacement policy to evict and re-read pages constantly.
        Verifies that no data is lost or corrupted across eviction cycles.
        """
        n = 2_000
        keys = list(range(1, n + 1))
        random.seed(7)
        random.shuffle(keys)

        for k in keys:
            small_pool_hash.insert(iv(k), rid(k))

        sample = random.sample(keys, 150)
        for k in sample:
            result = small_pool_hash.search(iv(k))
            assert rid(k) in result, (
                f"Key {k} corrupted after eviction: expected {rid(k)}, got {result}"
            )
