"""
benchmark_btree.py — Performance benchmarks for the B+ Tree index.

Run with:
    uv run pytest tests/benchmark_btree.py
"""

import tempfile
import random
import pytest

from common.value import DataType, Value
from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from index.btree.btree import BTree
from storage.heap.rid import RID


def iv(n: int) -> Value:
    return Value(DataType.INTEGER, n)


def rid(page: int, slot: int = 0) -> RID:
    return RID(page_id=page, slot=slot)


# ---------------------------------------------------------------------------
# Insert Benchmarks
# ---------------------------------------------------------------------------

def test_benchmark_sequential_inserts(benchmark):
    """50,000 keys in ascending order — right-biased splits."""
    keys = list(range(50_000))

    def run():
        with tempfile.TemporaryDirectory() as tmp:
            fm = FileManager(f"{tmp}/btree.idx")
            bm = BufferManager(file_manager=fm, pool_size=128)
            tree = BTree(key_type=DataType.INTEGER, buffer_manager=bm)
            for k in keys:
                tree.insert(iv(k), rid(k))

    benchmark.pedantic(run, rounds=5, iterations=1)


def test_benchmark_random_inserts(benchmark):
    """50,000 keys in random order — arbitrary splits."""
    keys = list(range(50_000))
    random.seed(42)
    random.shuffle(keys)

    def run():
        with tempfile.TemporaryDirectory() as tmp:
            fm = FileManager(f"{tmp}/btree.idx")
            bm = BufferManager(file_manager=fm, pool_size=128)
            tree = BTree(key_type=DataType.INTEGER, buffer_manager=bm)
            for k in keys:
                tree.insert(iv(k), rid(k))

    benchmark.pedantic(run, rounds=5, iterations=1)


# ---------------------------------------------------------------------------
# Search Benchmark
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prefilled_tree(tmp_path_factory):
    """Tree pre-loaded with 100,000 keys."""
    tmp = tmp_path_factory.mktemp("bench")
    fm = FileManager(str(tmp / "bench.idx"))
    bm = BufferManager(file_manager=fm, pool_size=128)
    tree = BTree(key_type=DataType.INTEGER, buffer_manager=bm)
    for k in range(100_000):
        tree.insert(iv(k), rid(k))
    return tree


def test_benchmark_random_reads(benchmark, prefilled_tree):
    """50,000 random reads on a tree with 100,000 keys."""
    keys_to_search = [random.randint(0, 99_999) for _ in range(50_000)]

    def run():
        for k in keys_to_search:
            prefilled_tree.search(iv(k))

    benchmark.pedantic(run, rounds=10, iterations=1)
