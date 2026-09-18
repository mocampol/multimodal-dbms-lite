"""
conftest.py — Shared pytest fixtures for the multimodal-dbms-lite test suite.

Pytest automatically discovers this file and makes every fixture defined
here available to ALL test files in this directory and subdirectories
without needing to import it manually.

Fixture scope reference:
    - "function"  (default): fixture is created & destroyed for EACH test.
    - "class"    : one fixture instance shared across all tests in a class.
    - "module"   : one fixture instance shared across all tests in a file.
    - "session"  : one fixture instance shared for the entire test run.

Use tmp_path (a built-in pytest fixture of scope="function") to get a
unique temporary directory per test; it is cleaned up automatically after
each test finishes, so no index files ever leak between tests.
"""

import pytest

from common.value import DataType
from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from index.btree.btree import BTree


# ---------------------------------------------------------------------------
# B-Tree fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def btree(tmp_path):
    """
    Scope: function (default) — a fresh, empty BTree for EVERY test.

    Wiring:
        tmp_path  →  FileManager  →  BufferManager  →  BTree

    tmp_path is a built-in pytest fixture that provides a unique
    temporary directory (e.g. /tmp/pytest-of-user/test_foo0/) that is
    automatically removed after the test completes.

    key_type=DataType.INTEGER:
        All keys inserted through this fixture must be Value(INTEGER, int).
        Change to DataType.VARCHAR or DataType.TEXT for string-key tests.

    unique=False:
        Allows duplicate keys (non-unique secondary index semantics).
        Set to True to enforce a primary key / unique constraint.

    pool_size=128:
        Number of frames in the Buffer Pool. Large enough to hold a
        moderate B-Tree without thrashing to disk; reduce to 10-32 to
        stress-test the buffer replacement (LRU-K) policy.
    """
    index_file = tmp_path / "btree.idx"
    fm = FileManager(str(index_file))
    bm = BufferManager(file_manager=fm, pool_size=128)
    return BTree(key_type=DataType.INTEGER, buffer_manager=bm, unique=False)


@pytest.fixture
def unique_btree(tmp_path):
    """
    Scope: function — same as `btree` but with unique=True.

    Use this fixture when testing DuplicateKeyError or primary-key
    constraint enforcement.
    """
    index_file = tmp_path / "unique_btree.idx"
    fm = FileManager(str(index_file))
    bm = BufferManager(file_manager=fm, pool_size=128)
    return BTree(key_type=DataType.INTEGER, buffer_manager=bm, unique=True)


@pytest.fixture
def small_pool_btree(tmp_path):
    """
    Scope: function — BTree backed by a tiny Buffer Pool (16 frames).

    Purpose: forces the LRU-K replacement policy to evict and re-read
    pages from disk during bulk inserts, validating that no data is lost
    or corrupted across eviction cycles.

    Use this fixture for stress / persistence tests, NOT for fast unit tests.
    """
    index_file = tmp_path / "small_pool_btree.idx"
    fm = FileManager(str(index_file))
    bm = BufferManager(file_manager=fm, pool_size=16)
    return BTree(key_type=DataType.INTEGER, buffer_manager=bm, unique=False)
