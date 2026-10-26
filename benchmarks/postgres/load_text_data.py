"""
benchmarks/postgres/load_text_data.py — Comparación externa: PostgreSQL con
tsvector + índice GIN y ranking ts_rank, contra SPIMI (BM25 y TF-IDF) del
DBMS propio.

Para cada corpus y N:
    1. Recrea las tablas (text_schema.sql), tokeniza con las reglas del motor
       y carga los documentos con COPY; llena tsv con to_tsvector('simple', ...)
       (cronometrado aparte como tsvector_time_s).
    2. En una conexión nueva construye el índice GIN (cronometrado), ejecuta
       ANALYZE y mide su tamaño (pg_relation_size).
    3. En otra conexión nueva ejecuta las consultas de text_queries.sql,
       verificando con EXPLAIN que usen el índice GIN.
    4. Verifica que el vocabulario (ts_stat) y los candidatos de cada
       consulta coincidan con los de benchmarks/text_common.py.

El dataset, las consultas, los relevantes y el formato del CSV vienen de
benchmarks/text_common.py, compartido con el benchmark del DBMS propio.

Metodología (misma que benchmarks/postgres/load_data.py):
    - WARMUP ejecuciones de calentamiento por consulta (descartadas) y luego
      REPETITIONS repeticiones cronometradas.
    - Tiempo cliente: perf_counter alrededor de execute + fetchall.
    - Tiempo servidor: "Execution Time" de EXPLAIN (ANALYZE, FORMAT JSON).
    - Memoria: cada fase (construcción, consultas) corre en una conexión
      nueva, es decir, en un backend nuevo; se reporta su pico de RSS
      (VmHWM) menos su RSS al empezar (VmRSS), leídos de /proc/<pid>/status
      con docker exec. Es la misma medida que usa el lado del DBMS propio.

Requisitos:
    cd benchmarks/postgres && docker compose up -d
    pip install -r benchmarks/postgres/requirements.txt

Uso:
    python benchmarks/postgres/load_text_data.py                    # spanish, N = 1k/10k/100k
    python benchmarks/postgres/load_text_data.py --quick            # smoke test (N=1,000)
    python benchmarks/postgres/load_text_data.py --corpus all --sizes 1000,10000
    PG_DSN="host=... port=..." python benchmarks/postgres/load_text_data.py
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from load_data import (
    CONTAINER_NAME,
    DEFAULT_DSN,
    collect_environment,
    connect,
    load_named_queries,
    run_command,
)
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

ENGINE = "pg_gin"
INDEX_NAME = "documentos_tsv_gin"
TEXT_CONFIG = "simple"
RANKING = "ts_rank(tsv, q, 1)"

SCHEMA_SQL = HERE / "text_schema.sql"
QUERIES_SQL = HERE / "text_queries.sql"

# El DBMS propio es de un solo hilo: sin paralelismo ni JIT (igual que
# benchmarks/benchmark_spatial.py).
SESSION_SETTINGS = {
    "max_parallel_workers_per_gather": "0",
    "jit": "off",
}
REPORTED_SETTINGS = (
    "shared_buffers", "work_mem", "maintenance_work_mem", "effective_cache_size",
    "max_parallel_workers_per_gather", "jit",
)


# =============================================================================
# Conexión y memoria del backend
# =============================================================================

def apply_session_settings(conn) -> None:
    for name, value in SESSION_SETTINGS.items():
        conn.execute(f"SET {name} = {value}")


def backend_memory(pid: int):
    """{"VmRSS", "VmHWM"} en bytes del backend `pid`, o None sin docker."""
    status = run_command(["docker", "exec", CONTAINER_NAME, "cat", f"/proc/{pid}/status"])
    if not status:
        return None
    memory = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            memory[key] = int(value.split()[0]) * 1024  # /proc reporta kB
    return memory if len(memory) == 2 else None


def run_phase(dsn: str, phase):
    """Ejecuta phase(conn) en una conexión nueva (un backend nuevo) y devuelve
    (resultado, pico de RSS del backend - RSS al empezar)."""
    with connect(dsn) as conn:
        apply_session_settings(conn)
        pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
        before = backend_memory(pid)
        result = phase(conn)
        after = backend_memory(pid)
    if before is None or after is None:
        return result, None
    return result, after["VmHWM"] - before["VmRSS"]


# =============================================================================
# Carga e índice
# =============================================================================

def copy_rows(copy, documents, stats: dict):
    """Tokeniza con las reglas del motor y copia (id, contenido, tokens) a la
    tabla staging; devuelve (id, tokens) para calcular los relevantes en el
    mismo recorrido."""
    for doc_id, text in documents:
        t0 = time.perf_counter()
        tokens = tokenize(text)
        stats["preprocess_time_s"] += time.perf_counter() - t0
        stats["vocabulary"].update(tokens)
        copy.write_row((doc_id, text, " ".join(tokens)))
        yield doc_id, tokens


def load_table(conn, documents, terms: set[str]):
    """Carga los documentos; devuelve (postings de los términos de consulta, estadísticas)."""
    conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    stats = {"preprocess_time_s": 0.0, "vocabulary": set()}
    with conn.cursor() as cur:
        with cur.copy("COPY documentos_staging (id, contenido, tokens) FROM STDIN") as copy:
            postings = term_postings(copy_rows(copy, documents, stats), terms)

    t0 = time.perf_counter()
    conn.execute(
        "INSERT INTO documentos "
        f"SELECT id, contenido, to_tsvector('{TEXT_CONFIG}', tokens) FROM documentos_staging"
    )
    tsvector_time = time.perf_counter() - t0
    conn.execute("DROP TABLE documentos_staging")
    conn.execute("VACUUM (ANALYZE) documentos")

    return postings, {
        "preprocess_time_s": stats["preprocess_time_s"],
        "tsvector_time_s": tsvector_time,
        "vocabulary_size": len(stats["vocabulary"]),
    }


def build_index(conn) -> tuple[float, int]:
    """Construye el GIN; devuelve (tiempo de construcción s, tamaño bytes)."""
    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX {INDEX_NAME} ON documentos USING GIN (tsv)")
    build_time = time.perf_counter() - t0

    # Sin estadísticas el planner puede preferir un Seq Scan.
    conn.execute("ANALYZE documentos")

    size = conn.execute("SELECT pg_relation_size(%s::regclass)", (INDEX_NAME,)).fetchone()[0]
    return build_time, size


# =============================================================================
# Consultas
# =============================================================================

def tsquery(query: dict) -> str:
    return " | ".join(query["terms"])


def plan_uses_index(node: dict) -> bool:
    if node.get("Index Name") == INDEX_NAME:
        return True
    return any(plan_uses_index(child) for child in node.get("Plans", []))


def time_topk(conn, sql: str, params: dict) -> tuple[float, list[int]]:
    """Devuelve (tiempo cliente ms, ids del top-k)."""
    t0 = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    return (time.perf_counter() - t0) * 1e3, [row[0] for row in rows]


def run_queries(conn, sql: str, queries: list[dict]) -> list[dict]:
    results = []
    for query in queries:
        params = {"tsquery": tsquery(query), "k": TOP_K}

        for _ in range(WARMUP):
            time_topk(conn, sql, params)  # calentamiento (descartado)
        plan = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", params).fetchone()[0][0]

        times = []
        for _ in range(REPETITIONS):
            elapsed_ms, top_ids = time_topk(conn, sql, params)
            times.append(elapsed_ms)

        results.append({
            "query": query,
            "times_ms": times,
            "server_ms": plan["Execution Time"],
            "used_index": plan_uses_index(plan["Plan"]),
            "top_ids": top_ids,
        })
    return results


# =============================================================================
# Entorno
# =============================================================================

def postgres_settings(conn) -> dict:
    apply_session_settings(conn)
    settings = {
        name: conn.execute("SELECT current_setting(%s)", (name,)).fetchone()[0]
        for name in REPORTED_SETTINGS
    }
    settings["text_search_config"] = TEXT_CONFIG
    settings["ranking"] = RANKING
    return settings


# =============================================================================
# Main
# =============================================================================

def run_experiment(dsn, conn, corpus: str, n: int, named_queries: dict, queries: list[dict]) -> list[dict]:
    postings, load_stats = load_table(conn, load_documents(corpus, n), query_vocabulary(queries))
    (build_time, index_size), build_memory = run_phase(dsn, build_index)
    results, query_memory = run_phase(dsn, lambda c: run_queries(c, named_queries["topk"], queries))

    vocabulary = conn.execute(named_queries["vocabulary"]).fetchone()[0]
    if vocabulary != load_stats["vocabulary_size"]:
        print(f"  ⚠ vocabulario distinto: ts_stat={vocabulary:,} vs. tokens={load_stats['vocabulary_size']:,}")

    rows = []
    for result in results:
        query = result["query"]
        reference = query_reference(postings, query)
        matched = conn.execute(named_queries["matches"], {"tsquery": tsquery(query)}).fetchone()[0]
        rows.append({
            **base_row(ENGINE, corpus, n, query, reference),
            **time_stats(result["times_ms"]),
            "avg_server_time_ms": result["server_ms"],
            "returned": len(result["top_ids"]),
            "matched_docs": matched,
            **relevance_metrics(result["top_ids"], reference["_relevant"]),
            "used_index": result["used_index"],
            "vocabulary_size": vocabulary,
            "preprocess_time_s": load_stats["preprocess_time_s"],
            "tsvector_time_s": load_stats["tsvector_time_s"],
            "index_build_time_s": build_time,
            "index_space_bytes": index_size,
            "build_peak_rss_delta_bytes": build_memory,
            "query_peak_rss_delta_bytes": query_memory,
        })

    print(f"  {corpus} N={n:,}: GIN {build_time:.3f}s, {index_size:,} B, vocabulario {vocabulary:,}")
    for group in QUERY_GROUPS:
        group_rows = [row for row in rows if row["query_group"] == group]
        avg_ms = statistics.fmean(row["avg_time_ms"] for row in group_rows)
        p_at_k = statistics.fmean(row["precision_at_k"] for row in group_rows)
        mismatches = sum(row["matched_docs"] != row["expected_matches"] for row in group_rows)
        warning = f"  ⚠ {mismatches} consultas con candidatos distintos" if mismatches else ""
        print(f"    {group:<3} términos  avg={avg_ms:.3f}ms  P@{TOP_K}={p_at_k:.2f}{warning}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus", choices=CORPORA + ["all"], default=DEFAULT_CORPUS)
    parser.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=None,
                        help="lista de N separada por comas (por defecto 1000,10000,100000)")
    parser.add_argument("--quick", action="store_true", help="solo N=1,000; escribe *_quick.*")
    parser.add_argument("--dsn", default=os.environ.get("PG_DSN", DEFAULT_DSN))
    args = parser.parse_args()

    corpora = CORPORA if args.corpus == "all" else [args.corpus]
    n_values = args.sizes or (QUICK_N_VALUES if args.quick else N_VALUES)
    csv_path = quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV
    env_path = quick_path(ENVIRONMENT_JSON) if args.quick else ENVIRONMENT_JSON
    named_queries = load_named_queries(QUERIES_SQL)

    with connect(args.dsn) as conn:
        environment = {**collect_environment(conn), **postgres_settings(conn)}
        print(f"{environment['postgres_version']}\n")

        rows = []
        for corpus in corpora:
            queries = generate_queries(corpus)
            for n in n_values:
                rows.extend(run_experiment(args.dsn, conn, corpus, n, named_queries, queries))

    out_path = write_results(rows, csv_path)
    write_environment("postgres", environment, env_path)
    print(f"\nResultados escritos en {out_path}")
    print(f"Entorno escrito en {env_path}")

    if any(row["build_peak_rss_delta_bytes"] is None for row in rows):
        print(f"\nAdvertencia: no se pudo leer /proc del contenedor '{CONTAINER_NAME}' "
              "(memoria vacía en el CSV).")
    if not all(row["used_index"] for row in rows):
        print("\nAdvertencia: alguna consulta no usó el índice GIN (ver columna used_index).")


if __name__ == "__main__":
    main()