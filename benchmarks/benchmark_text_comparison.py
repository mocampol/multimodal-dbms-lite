"""
benchmarks/benchmark_text_comparison.py — Comparación de texto: SPIMI + BM25
y SPIMI + TF-IDF (coseno) del DBMS propio, contra PostgreSQL GIN + ts_rank
(benchmarks/postgres/load_text_data.py).

Para cada corpus y N, cada fase corre en un proceso nuevo sobre el mismo
directorio de datos (el catálogo se reabre desde disco):
    1. Carga: CREATE TABLE documentos e INSERT por lotes (no cronometrado).
    2. Construcción: CREATE INDEX ... USING SPIMI (contenido), cronometrado;
       tamaño del archivo .idx y vocabulario del índice.
    3. Consultas, una vez con USING BM25 y otra con USING TF_IDF:
           SELECT id, SCORE() AS relevancia FROM documentos
           WHERE MATCH(contenido, '...') USING <ranking>
           ORDER BY relevancia DESC LIMIT k;
       y, sin cronometrar, EXPLAIN (¿TextIndexScan?) y los candidatos sin LIMIT.

El dataset, las consultas, los relevantes y el formato del CSV vienen de
benchmarks/text_common.py, compartido con el lado PostgreSQL.

Metodología (misma que benchmarks/postgres/load_text_data.py):
    - WARMUP ejecuciones de calentamiento por consulta (descartadas) y luego
      REPETITIONS repeticiones cronometradas.
    - Tiempo cliente: perf_counter alrededor de execute(), desde el texto SQL
      hasta tener todas las filas (parseo, planificación y ejecución).
    - Memoria: pico de RSS del proceso de la fase menos su RSS al empezar
      (Windows: peak_wset de psutil; Linux: VmHWM de /proc/self/status). En
      la fase de consultas el RSS inicial se toma antes de abrir el catálogo,
      así que incluye cargar el índice en memoria; en PostgreSQL, el backend
      nuevo también parte sin páginas del GIN.

Uso:
    python benchmarks/benchmark_text_comparison.py                    # spanish, N = 1k/10k/100k
    python benchmarks/benchmark_text_comparison.py --quick            # smoke test (N=1,000)
    python benchmarks/benchmark_text_comparison.py --corpus all --sizes 1000,10000
"""

import argparse
import multiprocessing
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_common import (
    CORPORA,
    DEFAULT_CORPUS,
    ENVIRONMENT_JSON,
    N_VALUES,
    QUERY_GROUPS,
    QUICK_N_VALUES,
    REPETITIONS,
    RESULTS_CSV,
    TOP_K,
    WARMUP,
    base_row,
    generate_queries,
    host_environment,
    load_documents,
    query_reference,
    query_vocabulary,
    quick_path,
    relevance_metrics,
    term_postings,
    time_stats,
    tokenize,
    write_environment,
    write_results,
)

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from query.query_engine import execute
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.heap_file import HeapFile
from storage.page import PAGE_SIZE
from transaction import TransactionManager

ENGINES = {"bm25": "spimi_bm25", "tf_idf": "spimi_tfidf"}
USING = {"bm25": "BM25", "tf_idf": "TF_IDF"}
TABLE = "documentos"
COLUMN = "contenido"
INDEX_NAME = "documentos_contenido_spimi"
INSERT_BATCH = 500
POOL_PAGES = 64  # mismo pool por archivo que src/main.py


# =============================================================================
# Catálogo y memoria del proceso
# =============================================================================

def open_catalog(base_dir: str) -> Catalog:
    """Catálogo sobre base_dir con la misma configuración que src/main.py,
    sin abrir el catálogo de data/."""
    def heap_factory(schema):
        file_manager = FileManager(f"{base_dir}/{schema.table_name}.tbl")
        return HeapFile(schema, BufferManager(file_manager, pool_size=POOL_PAGES))

    def index_buffer_factory(table_name, column_name, index_id):
        file_manager = FileManager(f"{base_dir}/{table_name}.{column_name}.{index_id}.idx")
        return BufferManager(file_manager, pool_size=POOL_PAGES)

    catalog = Catalog(
        heap_factory=heap_factory,
        storage_factories={StorageType.HEAP: heap_factory},
        index_buffer_factory=index_buffer_factory,
    )
    catalog._transaction_manager = TransactionManager(log_path=f"{base_dir}/benchmark.wal")
    return catalog


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    """Pico de RSS del proceso desde que empezó (high-water mark del SO)."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):          # Windows
        return info.peak_wset
    with open("/proc/self/status", encoding="ascii") as f:  # Linux
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("No se puede leer el pico de RSS en este sistema")


def in_fresh_process(function, *args):
    """Ejecuta function(*args) en un proceso nuevo: su pico de RSS empieza limpio."""
    with multiprocessing.get_context("spawn").Pool(processes=1) as pool:
        return pool.apply(function, args)


# =============================================================================
# Fases (cada una corre en un proceso nuevo)
# =============================================================================

def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def load_phase(base_dir: str, corpus: str, n: int, terms: set[str]) -> dict:
    """Crea y llena la tabla; devuelve los postings de los términos de consulta."""
    catalog = open_catalog(base_dir)
    execute(f"CREATE TABLE {TABLE} (id INTEGER, {COLUMN} TEXT);", catalog)
    documents = load_documents(corpus, n)
    for start in range(0, n, INSERT_BATCH):
        values = ", ".join(
            f"({doc_id}, {sql_literal(text)})" for doc_id, text in documents[start:start + INSERT_BATCH]
        )
        execute(f"INSERT INTO {TABLE} VALUES {values};", catalog)
    catalog.flush_table(TABLE)
    catalog._flush_sys_tables()
    return term_postings(((doc_id, tokenize(text)) for doc_id, text in documents), terms)


def build_phase(base_dir: str) -> dict:
    catalog = open_catalog(base_dir)
    rss_before = rss_bytes()
    t0 = time.perf_counter()
    execute(f"CREATE INDEX {INDEX_NAME} ON {TABLE} USING SPIMI ({COLUMN});", catalog)
    build_time = time.perf_counter() - t0
    peak = peak_rss_bytes()

    index = catalog.get_physical_index(TABLE, COLUMN)
    return {
        "index_build_time_s": build_time,
        "index_space_bytes": os.path.getsize(index.file_path),
        "vocabulary_size": len(index.index),
        "build_peak_rss_delta_bytes": peak - rss_before,
    }


def topk_sql(query: dict, ranking: str) -> str:
    return (f"SELECT id, SCORE() AS relevancia FROM {TABLE} "
            f"WHERE MATCH({COLUMN}, '{query['text']}') USING {USING[ranking]} "
            f"ORDER BY relevancia DESC LIMIT {TOP_K};")


def query_phase(base_dir: str, ranking: str, queries: list[dict]) -> tuple[list[dict], int]:
    rss_before = rss_bytes()          # antes de abrir el catálogo: incluye cargar el índice
    catalog = open_catalog(base_dir)

    results = []
    for query in queries:
        sql = topk_sql(query, ranking)
        for _ in range(WARMUP):
            execute(sql, catalog)     # calentamiento (descartado)
        times = []
        for _ in range(REPETITIONS):
            t0 = time.perf_counter()
            rows = execute(sql, catalog)
            times.append((time.perf_counter() - t0) * 1e3)
        results.append({"query": query, "times_ms": times, "top_ids": [row[0].data for row in rows]})
    peak = peak_rss_bytes()

    # Verificación, fuera de la medición de memoria y tiempo.
    for result in results:
        query = result["query"]
        plan = execute("EXPLAIN " + topk_sql(query, ranking), catalog)
        result["used_index"] = "TextIndexScan" in str(plan.description)
        matches = execute(f"SELECT id FROM {TABLE} WHERE MATCH({COLUMN}, '{query['text']}');", catalog)
        result["matched_docs"] = len(matches)
    return results, peak - rss_before


# =============================================================================
# Main
# =============================================================================

def run_experiment(corpus: str, n: int, queries: list[dict]) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="text-bench-", ignore_cleanup_errors=True) as base_dir:
        print(f"  {corpus} N={n:,}: cargando documentos...")
        postings = in_fresh_process(load_phase, base_dir, corpus, n, query_vocabulary(queries))
        build = in_fresh_process(build_phase, base_dir)
        print(f"    SPIMI {build['index_build_time_s']:.3f}s, {build['index_space_bytes']:,} B, "
              f"vocabulario {build['vocabulary_size']:,}")

        for ranking, engine in ENGINES.items():
            results, query_memory = in_fresh_process(query_phase, base_dir, ranking, queries)
            engine_rows = []
            for result in results:
                query = result["query"]
                reference = query_reference(postings, query)
                engine_rows.append({
                    **base_row(engine, corpus, n, query, reference),
                    **time_stats(result["times_ms"]),
                    "returned": len(result["top_ids"]),
                    "matched_docs": result["matched_docs"],
                    **relevance_metrics(result["top_ids"], reference["_relevant"]),
                    "used_index": result["used_index"],
                    **build,
                    "query_peak_rss_delta_bytes": query_memory,
                })
            print_groups(engine, engine_rows)
            rows.extend(engine_rows)
    return rows


def print_groups(engine: str, rows: list[dict]) -> None:
    for group in QUERY_GROUPS:
        group_rows = [row for row in rows if row["query_group"] == group]
        avg_ms = statistics.fmean(row["avg_time_ms"] for row in group_rows)
        p_at_k = statistics.fmean(row["precision_at_k"] for row in group_rows)
        mismatches = sum(row["matched_docs"] != row["expected_matches"] for row in group_rows)
        warning = f"  ⚠ {mismatches} consultas con candidatos distintos" if mismatches else ""
        print(f"    {engine:<12} {group:<3} términos  avg={avg_ms:.3f}ms  P@{TOP_K}={p_at_k:.2f}{warning}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus", choices=CORPORA + ["all"], default=DEFAULT_CORPUS)
    parser.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=None,
                        help="lista de N separada por comas (por defecto 1000,10000,100000)")
    parser.add_argument("--quick", action="store_true", help="solo N=1,000; escribe *_quick.*")
    args = parser.parse_args()

    corpora = CORPORA if args.corpus == "all" else [args.corpus]
    n_values = args.sizes or (QUICK_N_VALUES if args.quick else N_VALUES)
    csv_path = quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV
    env_path = quick_path(ENVIRONMENT_JSON) if args.quick else ENVIRONMENT_JSON

    rows = []
    for corpus in corpora:
        queries = generate_queries(corpus)
        for n in n_values:
            rows.extend(run_experiment(corpus, n, queries))

    environment = {
        **host_environment(),
        "psutil_version": psutil.__version__,
        "page_size_bytes": PAGE_SIZE,
        "buffer_pool_pages": POOL_PAGES,
        "memory_metric": "peak_wset" if sys.platform == "win32" else "VmHWM",
    }
    out_path = write_results(rows, csv_path)
    write_environment("minidbms", environment, env_path)
    print(f"\nResultados escritos en {out_path}")
    print(f"Entorno escrito en {env_path}")

    if not all(row["used_index"] for row in rows):
        print("\nAdvertencia: alguna consulta no usó el índice SPIMI (ver columna used_index).")


if __name__ == "__main__":
    main()