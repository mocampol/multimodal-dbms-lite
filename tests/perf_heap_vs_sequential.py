"""
perf_heap_vs_sequential.py — Standalone benchmark comparing HeapFile vs
SequentialFile on: insertion time, primary-key lookup time, disk space
used, and reorganization time, for N = 1,000 / 10,000 / 100,000 records.

Run with (from the repository root):
    uv run python tests/perf_heap_vs_sequential.py
or, without uv:
    PYTHONPATH=src python tests/perf_heap_vs_sequential.py     (bash)
    $env:PYTHONPATH="src"; python tests/perf_heap_vs_sequential.py   (PowerShell)

Writes raw results to tests/results/heap_vs_sequential_results.csv (one
row per N, heap_* and seq_* columns side by side) so they can be charted
externally, and prints a formatted table + short analysis to stdout.

Not a pytest file on purpose (no test_ prefix): this is a data-collection
script you run once per machine/config, not a pass/fail suite.

Methodology
-----------
For each N, both structures are built fresh (own temp dir, own
FileManager/BufferManager, pool_size=256 frames) and loaded with the SAME
dataset: N unique integer ids 0..N-1, in a fixed-seed random (not sorted)
insertion order — the realistic case, since real data rarely arrives
pre-sorted by primary key.

  1. Insertion time: wall time of the raw insert loop only. Buffered
     writes are flushed once via flush_all() afterwards, NOT counted
     towards insertion time (matches how a write-back buffer pool is
     normally benchmarked — see tests/benchmark_btree.py in this repo).

  2. Primary-key lookup time:
       - Heap: HeapFile has no index, so a PK lookup is a linear scan
         (scan(), stopping at the first match) — the actual cost of a
         PK lookup with no secondary index.
       - Sequential: .search(key), walking the page chain to the target
         page then scanning that page + its overflow chain. Measured
         BOTH before and after reorganize(), because this structure is
         explicitly designed to degrade under overflow and be restored
         by reorganization — collapsing those two into one number would
         hide the whole point of the technique.
     Averaged over a sample of existing keys (sample size scaled down
     for large N to keep total runtime bounded — see SAMPLE_BUDGET).
     A single lookup for a key that doesn't exist (-1) is also timed,
     as a cheap, deterministic worst-case-scan indicator.

  3. Disk space used: bytes on disk (os.path.getsize) of every physical
     file backing the structure, read after flush_all() — for
     sequential, main chain file + overflow file, added together.
     Measured before AND after reorganize(), because this
     implementation's reorganize() does not truncate/shrink files (see
     its docstring) — a fact worth surfacing rather than assuming.

  4. Reorganization time:
       - Sequential: wall time of one reorganize() call.
       - Heap: heap files have no reorganize() operation by design (that
         IS the structural trade-off: O(1) insert, no reorg needed, at
         the cost of an O(n) search). Reported as N/A in the main
         metric. As an optional bonus data point, we also time a full
         compaction pass (compact_page() over every page) after
         deleting a random 20% of records — the closest heap-side
         analogue, included for completeness rather than as a required
         comparison point.
"""

import csv
import os
import random
import sys
import tempfile
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from common.schema import Schema, Column
from common.value import DataType, Value
from common.record import Record
from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile
from storage.sequential.sequential_file import SequentialFile


N_VALUES = [1_000, 10_000, 100_000]
SEED = 42
DELETE_FRACTION_FOR_HEAP_COMPACTION = 0.20
LOOKUP_DECODE_BUDGET = 250_000  # total records scanned per avg-lookup phase, ~constant across N
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "heap_vs_sequential_results.csv"


def pool_size_for(n: int) -> int:
    """Sized generously (~n/90 records-per-page, x1.5 margin) so the
    whole file stays resident in the buffer pool once inserted. This
    isolates the algorithmic cost being compared (records/pages touched
    per operation) from this teaching FileManager's own I/O overhead
    (it reopens the OS file handle on every single page read/write,
    which would otherwise dominate and swamp the comparison at N=100k)."""
    return max(256, int(n / 90 * 1.5) + 64)


def make_schema() -> Schema:
    return Schema("bench", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("name", DataType.VARCHAR, size=20),
        Column("value", DataType.DOUBLE_PRECISION),
    ])


def make_record(schema: Schema, key: int, rng: random.Random) -> Record:
    return Record([
        Value(DataType.INTEGER, key),
        Value(DataType.VARCHAR, f"name_{key}"[:20]),
        Value(DataType.DOUBLE_PRECISION, rng.uniform(0, 1_000_000)),
    ])


def lookup_sample_size(n: int) -> int:
    """Scales the lookup sample down for large N so the total number of
    records scanned across the whole avg-lookup phase (samples * ~n/2
    worst case) stays roughly constant (LOOKUP_DECODE_BUDGET) instead of
    exploding with N."""
    per_lookup_cost = max(1, n // 2)
    return min(n, max(5, min(100, LOOKUP_DECODE_BUDGET // per_lookup_cost)))


def heap_linear_search(heap: HeapFile, key_index: int, key):
    for record in heap.scan():
        if record[key_index].data == key:
            return record
    return None


def time_lookups(fn, keys) -> float:
    """Returns average seconds per call of fn(key) over keys."""
    start = time.perf_counter()
    for k in keys:
        fn(k)
    elapsed = time.perf_counter() - start
    return elapsed / len(keys)


def run_heap_experiment(n: int, ids: list[int], schema: Schema, rng: random.Random) -> dict:
    key_index = schema.column_index("id")

    with tempfile.TemporaryDirectory() as tmp:
        fm = FileManager(str(Path(tmp) / "heap.tbl"))
        bm = BufferManager(fm, pool_size=pool_size_for(n))
        heap = HeapFile(schema, bm)

        rid_by_key = {}
        t0 = time.perf_counter()
        for key in ids:
            rid_by_key[key] = heap.insert(make_record(schema, key, rng))
        insert_time = time.perf_counter() - t0

        bm.flush_all()
        disk_space = os.path.getsize(fm.file_path)

        sample = rng.sample(ids, k=lookup_sample_size(n))
        lookup_avg = time_lookups(lambda k: heap_linear_search(heap, key_index, k), sample)
        t0 = time.perf_counter()
        heap_linear_search(heap, key_index, -1)  # guaranteed-absent key: full scan
        lookup_worst_case = time.perf_counter() - t0

        # Optional bonus: closest heap-side analogue to "reorganization" —
        # compact every page after introducing fragmentation via deletes.
        to_delete = rng.sample(ids, k=int(n * DELETE_FRACTION_FOR_HEAP_COMPACTION))
        for key in to_delete:
            heap.delete(rid_by_key[key])

        t0 = time.perf_counter()
        for page_id in heap._known_pages:
            heap.compact_page(page_id)
        compaction_time = time.perf_counter() - t0

        return {
            "heap_insert_time_s": insert_time,
            "heap_insert_us_per_record": insert_time / n * 1e6,
            "heap_lookup_avg_us": lookup_avg * 1e6,
            "heap_lookup_worst_case_us": lookup_worst_case * 1e6,
            "heap_lookup_sample_size": len(sample),
            "heap_disk_space_bytes": disk_space,
            "heap_reorganize_time_s": "",  # N/A by design, see module docstring
            "heap_bonus_compaction_time_s": compaction_time,
        }


def run_sequential_experiment(n: int, ids: list[int], schema: Schema, rng: random.Random) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        main_fm = FileManager(str(Path(tmp) / "seq_main.tbl"))
        main_bm = BufferManager(main_fm, pool_size=pool_size_for(n))
        overflow_fm = FileManager(str(Path(tmp) / "seq_overflow.tbl"))
        overflow_bm = BufferManager(overflow_fm, pool_size=pool_size_for(n))
        seq = SequentialFile(schema, "id", main_bm, overflow_bm)

        t0 = time.perf_counter()
        for key in ids:
            seq.insert(make_record(schema, key, rng))
        insert_time = time.perf_counter() - t0

        main_bm.flush_all()
        overflow_bm.flush_all()
        disk_space_before = os.path.getsize(main_fm.file_path) + os.path.getsize(overflow_fm.file_path)
        overflow_count = sum(len(chain) for chain in seq._overflow_chain.values())

        sample = rng.sample(ids, k=lookup_sample_size(n))
        lookup_avg_before = time_lookups(lambda k: seq.search(k), sample)
        t0 = time.perf_counter()
        seq.search(-1)  # guaranteed-absent key
        lookup_worst_case_before = time.perf_counter() - t0

        t0 = time.perf_counter()
        seq.reorganize()
        reorganize_time = time.perf_counter() - t0

        main_bm.flush_all()
        overflow_bm.flush_all()
        disk_space_after = os.path.getsize(main_fm.file_path) + os.path.getsize(overflow_fm.file_path)

        lookup_avg_after = time_lookups(lambda k: seq.search(k), sample)
        t0 = time.perf_counter()
        seq.search(-1)
        lookup_worst_case_after = time.perf_counter() - t0

        return {
            "seq_insert_time_s": insert_time,
            "seq_insert_us_per_record": insert_time / n * 1e6,
            "seq_lookup_avg_us_before_reorg": lookup_avg_before * 1e6,
            "seq_lookup_worst_case_us_before_reorg": lookup_worst_case_before * 1e6,
            "seq_lookup_avg_us_after_reorg": lookup_avg_after * 1e6,
            "seq_lookup_worst_case_us_after_reorg": lookup_worst_case_after * 1e6,
            "seq_lookup_sample_size": len(sample),
            "seq_overflow_fraction_before_reorg": overflow_count / n,
            "seq_disk_space_bytes_before_reorg": disk_space_before,
            "seq_disk_space_bytes_after_reorg": disk_space_after,
            "seq_reorganize_time_s": reorganize_time,
        }


def main():
    schema = make_schema()
    rows = []

    for n in N_VALUES:
        print(f"\n=== N = {n:,} ===")
        base_rng = random.Random(SEED)
        ids = list(range(n))
        base_rng.shuffle(ids)  # same shuffled insertion order for both structures

        print("  heap...")
        heap_rng = random.Random(SEED)
        heap_metrics = run_heap_experiment(n, ids, schema, heap_rng)

        print("  sequential...")
        seq_rng = random.Random(SEED)
        seq_metrics = run_sequential_experiment(n, ids, schema, seq_rng)

        row = {"n": n, **heap_metrics, **seq_metrics}
        rows.append(row)

        print(f"  heap:  insert={heap_metrics['heap_insert_time_s']:.4f}s  "
              f"lookup_avg={heap_metrics['heap_lookup_avg_us']:.1f}us  "
              f"space={heap_metrics['heap_disk_space_bytes']:,}B")
        print(f"  seq:   insert={seq_metrics['seq_insert_time_s']:.4f}s  "
              f"lookup_avg(before)={seq_metrics['seq_lookup_avg_us_before_reorg']:.1f}us  "
              f"lookup_avg(after)={seq_metrics['seq_lookup_avg_us_after_reorg']:.1f}us  "
              f"space={seq_metrics['seq_disk_space_bytes_before_reorg']:,}B  "
              f"reorg={seq_metrics['seq_reorganize_time_s']:.4f}s")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n"] + [k for k in rows[0].keys() if k != "n"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRaw results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
