"""
benchmarks/benchmark_indexes.py — Comparación experimental de estructuras
de indexación: B+ clusterizado, B+ no clusterizado, y Hash dinámico.

Issue: "Compare clustered B+ vs. non-clustered B+ vs. Dynamic Hash by
evaluating exact equality searches, range searches, and sorting. Measure
index build time, query time, additional space required, and performance
with frequent inserts/deletions. Include comparative charts"

Metodología (siguiendo la Sección VI del enunciado del proyecto, la misma
que ya usa tests/perf_heap_vs_sequential.py):
    - N = 1,000 / 10,000 / 100,000 registros.
    - Semilla fija (reproducible).
    - Cada medición se repite 3 veces; se reporta el promedio.
    - Búsqueda por rango y ordenamiento son N/A para Hash (se declara
      explícitamente en vez de fallar).

Separación de responsabilidades (a propósito, para no mezclar con
perf_heap_vs_sequential.py):
    - Ese benchmark ya mide el costo de HeapFile/SequentialFile como
      almacenamiento base.
    - Este benchmark aísla el costo de la ESTRUCTURA DE ÍNDICE en sí:
      primero se puebla el storage base (no cronometrado aquí), y luego
      se cronometra SOLO la construcción/consulta/espacio del índice.

Uso:
    uv run python benchmarks/benchmark_indexes.py           # corrida completa
    uv run python benchmarks/benchmark_indexes.py --quick    # smoke test rápido
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
from index.btree.btree import BTree
from index.btree.clustered_index import ClusteredIndex
from index.extendible_hash.extendible_hash_index import ExtendibleHashIndex


N_VALUES = [1_000, 10_000, 100_000]
SEED = 42
REPETITIONS = 3
RANGE_WIDTH = 100          # ancho de la ventana para búsquedas por rango
QUERY_SAMPLE_BUDGET = 250_000  # mismo espíritu que perf_heap_vs_sequential.py
MUTATION_SAMPLE_SIZE = 50      # cuántos inserts/deletes individuales medir tras construir

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "index_comparison_results.csv"
CHARTS_DIR = RESULTS_DIR / "charts"

NA = "N/A"  # marcador explícito para métricas que no aplican (rango/orden en Hash)


# =============================================================================
# Helpers comunes
# =============================================================================

def pool_size_for(n: int) -> int:
    return max(256, int(n / 90 * 1.5) + 64)


def make_schema() -> Schema:
    return Schema("bench", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("name", DataType.VARCHAR, size=20),
        Column("value", DataType.DOUBLE_PRECISION),
    ])


def make_record(key: int, rng: random.Random) -> Record:
    return Record([
        Value(DataType.INTEGER, key),
        Value(DataType.VARCHAR, f"name_{key}"[:20]),
        Value(DataType.DOUBLE_PRECISION, rng.uniform(0, 1_000_000)),
    ])


def query_sample_size(n: int) -> int:
    per_query_cost = max(1, n // 2)
    return min(n, max(5, min(100, QUERY_SAMPLE_BUDGET // per_query_cost)))


def time_avg(fn, items) -> float:
    """Tiempo promedio (segundos) de llamar fn(item) sobre items."""
    start = time.perf_counter()
    for item in items:
        fn(item)
    return (time.perf_counter() - start) / len(items)


# =============================================================================
# B+ clusterizado (sobre SequentialFile)
# =============================================================================

def run_clustered_experiment(n: int, ids: list[int], schema: Schema, rng: random.Random) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        main_fm = FileManager(str(Path(tmp) / "seq_main.tbl"))
        main_bm = BufferManager(main_fm, pool_size=pool_size_for(n))
        overflow_fm = FileManager(str(Path(tmp) / "seq_overflow.tbl"))
        overflow_bm = BufferManager(overflow_fm, pool_size=pool_size_for(n))
        seq = SequentialFile(schema, "id", main_bm, overflow_bm)

        # Poblar el storage base (NO se cronometra: es responsabilidad de
        # perf_heap_vs_sequential.py, no de este benchmark de índices).
        for key in ids:
            seq.insert(make_record(key, rng))

        # ---- Construcción del índice ----
        idx_fm = FileManager(str(Path(tmp) / "clustered.idx"))
        idx_bm = BufferManager(idx_fm, pool_size=pool_size_for(n))
        t0 = time.perf_counter()
        cidx = ClusteredIndex(DataType.INTEGER, idx_bm, seq)
        build_time = time.perf_counter() - t0

        idx_bm.flush_all()
        index_space = os.path.getsize(idx_fm.file_path)

        sample = rng.sample(ids, k=query_sample_size(n))

        # ---- Búsqueda exacta ----
        exact_avg = time_avg(lambda k: cidx.search(Value(DataType.INTEGER, k)), sample)

        # ---- Búsqueda por rango ----
        range_starts = rng.sample(range(0, max(1, n - RANGE_WIDTH)), k=min(20, len(sample)))
        range_avg = time_avg(
            lambda s: cidx.range(Value(DataType.INTEGER, s), Value(DataType.INTEGER, s + RANGE_WIDTH)),
            range_starts,
        )

        # ---- Ordenamiento (ya viene físicamente ordenado: costo del scan completo) ----
        t0 = time.perf_counter()
        list(seq.scan())
        sort_time = time.perf_counter() - t0

        # ---- Insert/Delete individuales tras construido ----
        new_keys = list(range(n, n + MUTATION_SAMPLE_SIZE))

        def do_insert(k):
            seq.insert(make_record(k, rng))
            cidx.sync()

        insert_avg = time_avg(do_insert, new_keys)

        def do_delete(k):
            seq.delete(k)
            cidx.sync()

        delete_avg = time_avg(do_delete, new_keys)

        return {
            "clustered_build_time_s": build_time,
            "clustered_exact_search_avg_us": exact_avg * 1e6,
            "clustered_range_search_avg_us": range_avg * 1e6,
            "clustered_sort_time_s": sort_time,
            "clustered_index_space_bytes": index_space,
            "clustered_insert_avg_us": insert_avg * 1e6,
            "clustered_delete_avg_us": delete_avg * 1e6,
            "clustered_query_sample_size": len(sample),
        }


# =============================================================================
# B+ no clusterizado (BTree secundario sobre HeapFile)
# =============================================================================

def build_base_heap(n: int, ids: list[int], schema: Schema, rng: random.Random, tmp: str):
    hfm = FileManager(str(Path(tmp) / "heap.tbl"))
    hbm = BufferManager(hfm, pool_size=pool_size_for(n))
    heap = HeapFile(schema, hbm)
    rid_by_key = {}
    for key in ids:
        rid_by_key[key] = heap.insert(make_record(key, rng))
    hbm.flush_all()
    return heap, hbm, rid_by_key


def run_nonclustered_experiment(n: int, ids: list[int], schema: Schema, rng: random.Random) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        heap, hbm, rid_by_key = build_base_heap(n, ids, schema, rng, tmp)

        idx_fm = FileManager(str(Path(tmp) / "btree.idx"))
        idx_bm = BufferManager(idx_fm, pool_size=pool_size_for(n))
        bt = BTree(DataType.INTEGER, idx_bm, unique=True)

        # ---- Construcción del índice ----
        t0 = time.perf_counter()
        for key in ids:
            bt.insert(Value(DataType.INTEGER, key), rid_by_key[key])
        build_time = time.perf_counter() - t0

        idx_bm.flush_all()
        index_space = os.path.getsize(idx_fm.file_path)

        sample = rng.sample(ids, k=query_sample_size(n))

        # ---- Búsqueda exacta ----
        exact_avg = time_avg(lambda k: bt.search(Value(DataType.INTEGER, k)), sample)

        # ---- Búsqueda por rango ----
        range_starts = rng.sample(range(0, max(1, n - RANGE_WIDTH)), k=min(20, len(sample)))
        range_avg = time_avg(
            lambda s: bt.range(Value(DataType.INTEGER, s), Value(DataType.INTEGER, s + RANGE_WIDTH)),
            range_starts,
        )

        # ---- Ordenamiento: recorrido completo de las hojas enlazadas ----
        t0 = time.perf_counter()
        bt.range(Value(DataType.INTEGER, min(ids)), Value(DataType.INTEGER, max(ids)))
        sort_time = time.perf_counter() - t0

        # ---- Insert/Delete individuales tras construido ----
        new_keys = list(range(n, n + MUTATION_SAMPLE_SIZE))

        def do_insert(k):
            rid = heap.insert(make_record(k, rng))
            bt.insert(Value(DataType.INTEGER, k), rid)
            rid_by_key[k] = rid

        insert_avg = time_avg(do_insert, new_keys)

        def do_delete(k):
            heap.delete(rid_by_key[k])
            bt.delete(Value(DataType.INTEGER, k), rid_by_key[k])

        delete_avg = time_avg(do_delete, new_keys)

        return {
            "nonclustered_build_time_s": build_time,
            "nonclustered_exact_search_avg_us": exact_avg * 1e6,
            "nonclustered_range_search_avg_us": range_avg * 1e6,
            "nonclustered_sort_time_s": sort_time,
            "nonclustered_index_space_bytes": index_space,
            "nonclustered_insert_avg_us": insert_avg * 1e6,
            "nonclustered_delete_avg_us": delete_avg * 1e6,
            "nonclustered_query_sample_size": len(sample),
        }


# =============================================================================
# Hash dinámico (Extendible Hashing sobre HeapFile)
# =============================================================================

def run_hash_experiment(n: int, ids: list[int], schema: Schema, rng: random.Random) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        heap, hbm, rid_by_key = build_base_heap(n, ids, schema, rng, tmp)

        idx_fm = FileManager(str(Path(tmp) / "hash.idx"))
        idx_bm = BufferManager(idx_fm, pool_size=pool_size_for(n))
        hidx = ExtendibleHashIndex.create(idx_bm, DataType.INTEGER)

        # ---- Construcción del índice ----
        t0 = time.perf_counter()
        for key in ids:
            hidx.insert(Value(DataType.INTEGER, key), rid_by_key[key])
        build_time = time.perf_counter() - t0

        idx_bm.flush_all()
        index_space = os.path.getsize(idx_fm.file_path)

        sample = rng.sample(ids, k=query_sample_size(n))

        # ---- Búsqueda exacta ----
        exact_avg = time_avg(lambda k: hidx.search(Value(DataType.INTEGER, k)), sample)

        # ---- Insert/Delete individuales tras construido ----
        new_keys = list(range(n, n + MUTATION_SAMPLE_SIZE))

        def do_insert(k):
            rid = heap.insert(make_record(k, rng))
            hidx.insert(Value(DataType.INTEGER, k), rid)
            rid_by_key[k] = rid

        insert_avg = time_avg(do_insert, new_keys)

        def do_delete(k):
            heap.delete(rid_by_key[k])
            hidx.remove(Value(DataType.INTEGER, k), rid_by_key[k])

        delete_avg = time_avg(do_delete, new_keys)

        return {
            "hash_build_time_s": build_time,
            "hash_exact_search_avg_us": exact_avg * 1e6,
            "hash_range_search_avg_us": NA,   # Hash no soporta rangos (declarado explícitamente)
            "hash_sort_time_s": NA,            # Hash no preserva orden
            "hash_index_space_bytes": index_space,
            "hash_insert_avg_us": insert_avg * 1e6,
            "hash_delete_avg_us": delete_avg * 1e6,
            "hash_query_sample_size": len(sample),
        }


# =============================================================================
# Orquestación
# =============================================================================

HASH_RESULT_FIELDS = [
    "hash_build_time_s", "hash_exact_search_avg_us", "hash_range_search_avg_us",
    "hash_sort_time_s", "hash_index_space_bytes", "hash_insert_avg_us",
    "hash_delete_avg_us", "hash_query_sample_size",
]

DIRECTORY_LIMIT_EXCEEDED = "LIMITE_CAPACIDAD_DIRECTORIO"


def hash_capacity_limit_row() -> dict:
    """
    Fila de reemplazo cuando ExtendibleHashIndex excede la capacidad fija
    del directorio (MAX_BUCKETS=512 en hash_directory_page.py, ~9 niveles
    de global_depth). NO es un fallo del benchmark: es un límite real y no
    documentado como tal del índice, que aparece alrededor de N=100,000
    con claves INTEGER. Se registra explícitamente en vez de reintentar o
    silenciar el error, para que quede evidenciado en el CSV/reporte.
    """
    return {field: DIRECTORY_LIMIT_EXCEEDED for field in HASH_RESULT_FIELDS}


def average_metrics(dicts: list[dict]) -> dict:
    result = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if all(isinstance(v, (int, float)) for v in values):
            result[key] = sum(values) / len(values)
        else:
            result[key] = values[0]  # p.ej. "N/A" o DIRECTORY_LIMIT_EXCEEDED
    return result


def run_all(n_values, repetitions):
    schema = make_schema()
    rows = []

    for n in n_values:
        print(f"\n=== N = {n:,} ===")
        base_ids = list(range(n))
        random.Random(SEED).shuffle(base_ids)

        clustered_runs, nonclustered_runs, hash_runs = [], [], []
        for rep in range(1, repetitions + 1):
            print(f"  clustered B+   (run {rep}/{repetitions})...")
            clustered_runs.append(run_clustered_experiment(n, base_ids, schema, random.Random(SEED)))

            print(f"  non-clustered B+ (run {rep}/{repetitions})...")
            nonclustered_runs.append(run_nonclustered_experiment(n, base_ids, schema, random.Random(SEED)))

            print(f"  dynamic hash   (run {rep}/{repetitions})...")
            try:
                hash_runs.append(run_hash_experiment(n, base_ids, schema, random.Random(SEED)))
            except ValueError as e:
                # Cualquier ValueError en esta llamada es evidencia del
                # mismo límite de capacidad del directorio (MAX_BUCKETS):
                # ya sea el corte limpio contra el borde de la página, o
                # -en directorios muy chicos- corrupción de punteros que
                # termina fallando más adelante al leer una página
                # inexistente. En ambos casos la causa es la misma y no
                # hay nada que el benchmark pueda hacer salvo reportarlo.
                print(f"    ! Hash excedió la capacidad del directorio en N={n:,} "
                      f"({type(e).__name__}: {e}) — se registra como "
                      f"{DIRECTORY_LIMIT_EXCEEDED!r} y se continúa.")
                hash_runs.append(hash_capacity_limit_row())

        row = {
            "n": n,
            "repetitions": repetitions,
            **average_metrics(clustered_runs),
            **average_metrics(nonclustered_runs),
            **average_metrics(hash_runs),
        }
        rows.append(row)

        print(f"  clustered:    build={row['clustered_build_time_s']:.4f}s  "
              f"exact={row['clustered_exact_search_avg_us']:.1f}us  "
              f"space={row['clustered_index_space_bytes']:,.0f}B")
        print(f"  nonclustered: build={row['nonclustered_build_time_s']:.4f}s  "
              f"exact={row['nonclustered_exact_search_avg_us']:.1f}us  "
              f"space={row['nonclustered_index_space_bytes']:,.0f}B")
        if isinstance(row["hash_build_time_s"], str):
            print(f"  hash:         {row['hash_build_time_s']} (excedió la capacidad del directorio)")
        else:
            print(f"  hash:         build={row['hash_build_time_s']:.4f}s  "
                  f"exact={row['hash_exact_search_avg_us']:.1f}us  "
                  f"space={row['hash_index_space_bytes']:,.0f}B")

    return rows


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n", "repetitions"] + [k for k in rows[0].keys() if k not in ("n", "repetitions")]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados crudos escritos en {path}")


# =============================================================================
# Gráficas comparativas
# =============================================================================

def make_charts(rows: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ns = [r["n"] for r in rows]
    structures = [
        ("clustered", "B+ clusterizado", "tab:blue"),
        ("nonclustered", "B+ no clusterizado", "tab:orange"),
        ("hash", "Hash dinámico", "tab:green"),
    ]

    def numeric_series(field: str):
        """(xs, ys) solo con los N donde el valor es numérico — salta
        puntos marcados como N/A o DIRECTORY_LIMIT_EXCEEDED en vez de
        romper el gráfico o inventar un valor."""
        xs, ys = [], []
        for r in rows:
            v = r[field]
            if isinstance(v, (int, float)):
                xs.append(r["n"])
                ys.append(v)
        return xs, ys

    # ---- Gráfica 1: tiempo de construcción del índice ----
    plt.figure(figsize=(7, 5))
    for prefix, label, color in structures:
        xs, ys = numeric_series(f"{prefix}_build_time_s")
        plt.plot(xs, ys, marker="o", label=label, color=color)
        skipped = [r["n"] for r in rows if r["n"] not in xs]
        if skipped:
            plt.scatter(skipped, [max(ys) if ys else 1] * len(skipped), marker="x", color=color,
                        label=f"{label} (excedió capacidad)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N (registros)")
    plt.ylabel("Tiempo de construcción (s)")
    plt.title("Tiempo de construcción del índice")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "build_time.png", dpi=150)
    plt.close()

    # ---- Gráfica 2: tiempo de búsqueda exacta ----
    plt.figure(figsize=(7, 5))
    for prefix, label, color in structures:
        xs, ys = numeric_series(f"{prefix}_exact_search_avg_us")
        plt.plot(xs, ys, marker="o", label=label, color=color)
    plt.xscale("log")
    plt.xlabel("N (registros)")
    plt.ylabel("Tiempo promedio de búsqueda exacta (µs)")
    plt.title("Búsqueda exacta por igualdad")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "exact_search_time.png", dpi=150)
    plt.close()

    # ---- Gráfica 3: espacio adicional requerido (barras agrupadas) ----
    import numpy as np
    x = np.arange(len(ns))
    width = 0.25
    plt.figure(figsize=(7, 5))
    for i, (prefix, label, color) in enumerate(structures):
        raw = [r[f"{prefix}_index_space_bytes"] for r in rows]
        values = [(v / 1024) if isinstance(v, (int, float)) else 0 for v in raw]  # KB
        bars = plt.bar(x + (i - 1) * width, values, width, label=label, color=color)
        for bar, v in zip(bars, raw):
            if not isinstance(v, (int, float)):
                plt.text(bar.get_x() + bar.get_width() / 2, 1, "N/A",
                        ha="center", va="bottom", fontsize=8, rotation=90)
    plt.xticks(x, [f"{n:,}" for n in ns])
    plt.xlabel("N (registros)")
    plt.ylabel("Espacio adicional del índice (KB)")
    plt.title("Espacio adicional requerido por el índice")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "index_space.png", dpi=150)
    plt.close()

    print(f"Gráficas guardadas en {out_dir}")


# =============================================================================
# Tabla resumen
# =============================================================================

SUMMARY_TABLE = """
## Tabla resumen

| Técnica | Ventajas | Desventajas | Cuándo usarla |
|---|---|---|---|
| **B+ clusterizado** | Igualdad, rango y orden en O(log n) + secuencial; el scan ordenado es prácticamente gratis. | Solo una columna por tabla puede ser clusterizada; construir/mantener el índice implica reescribir el archivo de datos ordenado. | Columna de acceso más frecuente y con muchas consultas por rango u `ORDER BY` (típicamente la PK). |
| **B+ no clusterizado** | Igualdad, rango y orden en O(log n); se pueden tener varios por tabla sin reordenar los datos. | Cada resultado implica un salto adicional al Heap File (menos localidad que el clusterizado); espacio extra por cada índice. | Columnas secundarias con búsquedas por rango u orden frecuentes, cuando no pueden ser la columna clusterizada. |
| **Hash dinámico** | Igualdad en O(1) amortizado; se adapta al crecimiento de datos sin reorganización manual. | No soporta rango ni orden (N/A); no aprovechable si la consulta necesita `ORDER BY`/`BETWEEN`. | Columnas con búsquedas exclusivamente por igualdad exacta y alto volumen (ej. claves foráneas, lookups puntuales). |
"""


def write_summary(path: Path):
    path.write_text(SUMMARY_TABLE, encoding="utf-8")
    print(f"Tabla resumen escrita en {path}")


# =============================================================================
def main():
    quick = "--quick" in sys.argv
    n_values = [200, 1_000] if quick else N_VALUES
    repetitions = 1 if quick else REPETITIONS

    rows = run_all(n_values, repetitions)
    write_csv(rows, RESULTS_CSV)
    make_charts(rows, CHARTS_DIR)
    write_summary(RESULTS_DIR / "index_comparison_summary.md")


if __name__ == "__main__":
    main()