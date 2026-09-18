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
REPETITIONS = 3
DELETE_FRACTION_FOR_HEAP_COMPACTION = 0.20
LOOKUP_DECODE_BUDGET = 250_000
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "heap_vs_sequential_results.csv"


def pool_size_for(n: int) -> int:
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
    per_lookup_cost = max(1, n // 2)
    return min(n, max(5, min(100, LOOKUP_DECODE_BUDGET // per_lookup_cost)))


def heap_linear_search(heap: HeapFile, key_index: int, key):
    for record in heap.scan():
        if record[key_index].data == key:
            return record
    return None


def time_lookups(fn, keys) -> float:
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
        heap_linear_search(heap, key_index, -1)
        lookup_worst_case = time.perf_counter() - t0

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
            "heap_reorganize_time_s": "",
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
        seq.search(-1)
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


def average_metrics(dicts: list[dict]) -> dict:
    result = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if all(isinstance(v, (int, float)) for v in values):
            result[key] = sum(values) / len(values)
        else:
            result[key] = values[0]
    return result


def main():
    schema = make_schema()
    rows = []

    for n in N_VALUES:
        print(f"\n=== N = {n:,} ===")
        base_rng = random.Random(SEED)
        ids = list(range(n))
        base_rng.shuffle(ids)

        heap_runs = []
        seq_runs = []
        for rep in range(1, REPETITIONS + 1):
            print(f"  heap (run {rep}/{REPETITIONS})...")
            heap_runs.append(run_heap_experiment(n, ids, schema, random.Random(SEED)))

            print(f"  sequential (run {rep}/{REPETITIONS})...")
            seq_runs.append(run_sequential_experiment(n, ids, schema, random.Random(SEED)))

        heap_metrics = average_metrics(heap_runs)
        seq_metrics = average_metrics(seq_runs)

        row = {"n": n, "repetitions": REPETITIONS, **heap_metrics, **seq_metrics}
        rows.append(row)

        print(f"  heap:  insert={heap_metrics['heap_insert_time_s']:.4f}s  "
              f"lookup_avg={heap_metrics['heap_lookup_avg_us']:.1f}us  "
              f"space={heap_metrics['heap_disk_space_bytes']:,.0f}B")
        print(f"  seq:   insert={seq_metrics['seq_insert_time_s']:.4f}s  "
              f"lookup_avg(before)={seq_metrics['seq_lookup_avg_us_before_reorg']:.1f}us  "
              f"lookup_avg(after)={seq_metrics['seq_lookup_avg_us_after_reorg']:.1f}us  "
              f"space={seq_metrics['seq_disk_space_bytes_before_reorg']:,.0f}B  "
              f"reorg={seq_metrics['seq_reorganize_time_s']:.4f}s")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n", "repetitions"] + [k for k in rows[0].keys() if k not in ("n", "repetitions")]
    out_path = write_csv_with_retry(rows, fieldnames, RESULTS_CSV)

    print(f"\nRaw results written to {out_path}")


def write_csv_with_retry(rows, fieldnames, path: Path, attempts: int = 5) -> Path:
    last_error = None
    for i in range(attempts):
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return path
        except PermissionError as e:
            last_error = e
            time.sleep(2)

    fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
    with open(fallback, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WARNING: could not write to {path} ({last_error}); wrote to {fallback} instead")
    return fallback


if __name__ == "__main__":
    main()
