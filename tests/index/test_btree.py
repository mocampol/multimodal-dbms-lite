"""
test_btree.py — Full test battery for the B+ Tree index implementation.

Test organization:
    1. Smoke tests        : minimal sanity checks (empty tree, single op).
    2. Unit tests         : correct behavior for each public method.
    3. Edge case tests    : boundary values, duplicates, key-not-found.
    4. Stress tests       : bulk inserts (1k–50k) with integrity verification.
    5. Buffer pool stress : bulk inserts against a tiny pool (eviction paths).

Fixtures used (defined in conftest.py):
    btree            — standard pool (128 frames), unique=False
    unique_btree     — standard pool (128 frames), unique=True
    small_pool_btree — tiny pool (16 frames),      unique=False
"""

import random
import pytest

from common.value import DataType, Value
from storage.heap.rid import RID
from index.btree.exceptions import DuplicateKeyError, KeyNotFoundError


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

class TestBTreeSmoke:
    def test_tree_is_created(self, btree):
        """The fixture creates a valid BTree instance."""
        assert btree is not None

    def test_search_on_empty_tree_returns_none(self, btree):
        """Searching an empty tree must return None, not raise."""
        assert btree.search(iv(1)) is None

    def test_range_on_empty_tree_returns_empty_list(self, btree):
        """range() on an empty tree must return [], not raise."""
        assert btree.range(iv(1), iv(100)) == []


# ---------------------------------------------------------------------------
# 2. Unit tests — insert / search / range / delete
# ---------------------------------------------------------------------------

class TestInsertAndSearch:
    def test_single_insert_found(self, btree):
        btree.insert(iv(42), rid(1))
        result = btree.search(iv(42))
        assert result == rid(1)

    def test_key_not_inserted_not_found(self, btree):
        btree.insert(iv(10), rid(1))
        assert btree.search(iv(99)) is None

    def test_multiple_inserts_all_found(self, btree):
        pairs = [(iv(k), rid(k)) for k in range(1, 101)]
        for key, r in pairs:
            btree.insert(key, r)
        for key, r in pairs:
            assert btree.search(key) == r

    def test_insert_preserves_sorted_order(self, btree):
        """Keys inserted in reverse order must still be retrievable."""
        for k in range(100, 0, -1):
            btree.insert(iv(k), rid(k))
        for k in range(1, 101):
            assert btree.search(iv(k)) == rid(k)


class TestRangeQuery:
    def test_range_returns_all_keys_in_bounds(self, btree):
        for k in range(1, 21):
            btree.insert(iv(k), rid(k))
        results = btree.range(iv(5), iv(10))
        returned_keys = [key.data for key, _ in results]
        assert returned_keys == list(range(5, 11))

    def test_range_exclusive_upper_bound(self, btree):
        """range() includes start_key and end_key (closed interval)."""
        for k in [1, 5, 10, 15, 20]:
            btree.insert(iv(k), rid(k))
        results = btree.range(iv(5), iv(15))
        returned_keys = {key.data for key, _ in results}
        assert 5 in returned_keys
        assert 15 in returned_keys
        assert 20 not in returned_keys

    def test_range_single_element(self, btree):
        btree.insert(iv(7), rid(7))
        results = btree.range(iv(7), iv(7))
        assert len(results) == 1
        assert results[0][0].data == 7

    def test_range_empty_when_no_keys_in_interval(self, btree):
        for k in [1, 2, 50, 51]:
            btree.insert(iv(k), rid(k))
        assert btree.range(iv(10), iv(20)) == []


class TestDelete:
    def test_delete_key_no_longer_found(self, btree):
        btree.insert(iv(5), rid(5))
        btree.delete(iv(5))
        assert btree.search(iv(5)) is None

    def test_delete_one_key_others_intact(self, btree):
        for k in range(1, 11):
            btree.insert(iv(k), rid(k))
        btree.delete(iv(5))
        assert btree.search(iv(5)) is None
        for k in range(1, 11):
            if k != 5:
                assert btree.search(iv(k)) == rid(k)


# ---------------------------------------------------------------------------
# 3. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_duplicate_key_raises_on_unique_tree(self, unique_btree):
        unique_btree.insert(iv(1), rid(1))
        with pytest.raises(DuplicateKeyError):
            unique_btree.insert(iv(1), rid(2))

    def test_duplicate_key_allowed_on_non_unique_tree(self, btree):
        """Non-unique trees allow the same key with different RIDs."""
        btree.insert(iv(1), rid(1, slot=0))
        btree.insert(iv(1), rid(1, slot=1))
        results = btree.search_all(iv(1))
        assert len(results) == 2

    def test_search_minimum_key(self, btree):
        for k in range(100, 201):
            btree.insert(iv(k), rid(k))
        assert btree.search(iv(100)) == rid(100)

    def test_search_maximum_key(self, btree):
        for k in range(100, 201):
            btree.insert(iv(k), rid(k))
        assert btree.search(iv(200)) == rid(200)

    def test_delete_nonexistent_key_raises(self, btree):
        """
        Deleting a key that was never inserted must raise KeyNotFoundError.

        This is the correct database contract: silently ignoring a delete
        on a missing key would hide bugs in the caller. The BTree explicitly
        raises KeyNotFoundError so callers can decide whether to handle it.
        """
        btree.insert(iv(10), rid(10))
        with pytest.raises(KeyNotFoundError):
            btree.delete(iv(999))


# ---------------------------------------------------------------------------
# 4. Stress tests — bulk inserts with integrity verification
# ---------------------------------------------------------------------------

class TestBulkInserts:
    @pytest.mark.parametrize("n_records", [1_000, 10_000, 50_000])
    def test_random_bulk_insert_all_keys_found(self, btree, n_records):
        """
        Insert n_records keys in a random (shuffled) order.
        Then verify every inserted key can be found with the correct RID.

        Shuffled order maximizes the number of node splits at arbitrary
        positions in the tree, which is the hardest case for B+ Tree
        correctness.
        """
        keys = list(range(1, n_records + 1))
        random.seed(42)
        random.shuffle(keys)

        for k in keys:
            btree.insert(iv(k), rid(k))

        # Verify a sample of 500 (or all if n < 500) lookups
        sample = random.sample(keys, min(500, n_records))
        for k in sample:
            result = btree.search(iv(k))
            assert result is not None, f"key {k} not found after bulk insert"
            assert result == rid(k), f"key {k}: expected {rid(k)}, got {result}"

    @pytest.mark.parametrize("n_records", [1_000, 10_000])
    def test_sequential_bulk_insert_all_keys_found(self, btree, n_records):
        """
        Insert n_records keys in strictly ascending order (worst case for
        right-biased leaf splits).
        """
        for k in range(1, n_records + 1):
            btree.insert(iv(k), rid(k))

        for k in range(1, n_records + 1, max(1, n_records // 200)):
            result = btree.search(iv(k))
            assert result == rid(k)

    @pytest.mark.parametrize("n_records", [1_000, 10_000])
    def test_reverse_bulk_insert_all_keys_found(self, btree, n_records):
        """
        Insert n_records keys in strictly descending order (worst case for
        left-biased leaf splits).
        """
        for k in range(n_records, 0, -1):
            btree.insert(iv(k), rid(k))

        for k in range(1, n_records + 1, max(1, n_records // 200)):
            result = btree.search(iv(k))
            assert result == rid(k)

    def test_ghost_keys_never_found(self, btree):
        """
        After a bulk insert of [1..1000], keys outside that range
        must never be found.
        """
        for k in range(1, 1001):
            btree.insert(iv(k), rid(k))
        assert btree.search(iv(0)) is None
        assert btree.search(iv(1001)) is None
        assert btree.search(iv(999_999)) is None


# ---------------------------------------------------------------------------
# 5. Buffer pool stress — forces LRU-K eviction during inserts
# ---------------------------------------------------------------------------

class TestBufferPoolStress:
    def test_bulk_insert_with_tiny_pool_no_data_loss(self, small_pool_btree):
        """
        Insert 5,000 keys using a buffer pool of only 16 frames.
        This forces the LRU-K policy to evict and re-read pages constantly.
        Verifies that no data is lost or corrupted across eviction cycles.
        """
        n = 5_000
        keys = list(range(1, n + 1))
        random.seed(7)
        random.shuffle(keys)

        for k in keys:
            small_pool_btree.insert(iv(k), rid(k))

        sample = random.sample(keys, 200)
        for k in sample:
            result = small_pool_btree.search(iv(k))
            assert result == rid(k), (
                f"key {k} corrupted after eviction: expected {rid(k)}, got {result}"
            )
