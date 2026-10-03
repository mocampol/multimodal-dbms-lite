"""
benchmarks/postgres/load_data.py — Comparación externa: PostgreSQL/PostGIS
con índice GiST, contra el R-tree del DBMS propio.

Para cada N:
    1. Recrea la tabla (schema.sql) y carga los puntos con COPY.
    2. Construye el índice GiST (cronometrado) y ejecuta ANALYZE.
    3. Mide el tamaño del índice (pg_relation_size).
    4. Ejecuta las consultas de queries.sql (radius, knn, intersects),
       verificando con EXPLAIN que usen el índice GiST.

El dataset, las consultas y el formato del CSV vienen de
benchmarks/spatial_common.py, compartido con el benchmark del DBMS propio.

Metodología (misma que benchmarks/benchmark_indexes.py):
    - N = 1,000 / 10,000 / 100,000 puntos, semilla fija.
    - 1 ejecución de calentamiento por consulta (descartada) y luego
      REPETITIONS repeticiones; se reporta el promedio por consulta.
    - Tiempo cliente: perf_counter alrededor de execute + fetchall.
    - Tiempo servidor: "Execution Time" de EXPLAIN (ANALYZE, FORMAT JSON).

Requisitos:
    cd benchmarks/postgres && docker compose up -d
    pip install -r benchmarks/postgres/requirements.txt

Uso:
    python benchmarks/postgres/load_data.py           # corrida completa
    python benchmarks/postgres/load_data.py --quick    # smoke test (N=1,000)
    PG_DSN="host=... port=..." python benchmarks/postgres/load_data.py
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from spatial_common import (
    N_VALUES,
    QUERY_PARAMS,
    REPETITIONS,
    RESULTS_DIR,
    generate_points,
    generate_queries,
    write_results,
)

ENGINE = "postgis"
DEFAULT_DSN = "host=localhost port=5433 dbname=spatial_bench user=bench password=bench"
CONTAINER_NAME = "mdbms-postgis-bench"
INDEX_NAME = "places_geom_gist"

SCHEMA_SQL = HERE / "schema.sql"
QUERIES_SQL = HERE / "queries.sql"
ENVIRONMENT_JSON = RESULTS_DIR / "postgis_environment.json"


# =============================================================================
# Conexión y archivos SQL
# =============================================================================

def connect(dsn: str, timeout_s: float = 60.0) -> psycopg.Connection:
    """Conecta reintentando mientras el contenedor termina de arrancar."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return psycopg.connect(dsn, autocommit=True)
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def load_named_queries(path: Path) -> dict[str, str]:
    """Separa queries.sql en bloques "-- name: <query_type>"."""
    queries: dict[str, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("-- name:"):
            current = stripped.removeprefix("-- name:").strip()
            queries[current] = ""
        elif current and stripped and not stripped.startswith("--"):
            queries[current] += line + "\n"
    return {name: sql.strip().rstrip(";") for name, sql in queries.items()}


def query_params(query_type: str, query: dict) -> dict:
    if query_type == "intersects":
        ring = query["polygon"] + query["polygon"][:1]  # WKT exige anillo cerrado
        coords = ", ".join(f"{x!r} {y!r}" for x, y in ring)
        return {"poly": f"POLYGON(({coords}))"}
    return query


# =============================================================================
# Carga e índice
# =============================================================================

def load_table(conn: psycopg.Connection, n: int) -> None:
    conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    with conn.cursor() as cur:
        with cur.copy("COPY places (id, name, geom) FROM STDIN") as copy:
            for pid, name, x, y in generate_points(n):
                copy.write_row((pid, name, f"POINT({x!r} {y!r})"))


def build_index(conn: psycopg.Connection) -> tuple[float, int]:
    """Construye el GiST; devuelve (tiempo de construcción s, tamaño bytes)."""
    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX {INDEX_NAME} ON places USING GIST (geom)")
    build_time = time.perf_counter() - t0

    # Sin estadísticas el planner puede preferir un Seq Scan.
    conn.execute("ANALYZE places")

    size = conn.execute("SELECT pg_relation_size(%s::regclass)", (INDEX_NAME,)).fetchone()[0]
    return build_time, size


# =============================================================================
# Consultas
# =============================================================================

def plan_uses_index(node: dict) -> bool:
    if node.get("Index Name") == INDEX_NAME:
        return True
    return any(plan_uses_index(child) for child in node.get("Plans", []))


def explain_analyze(conn: psycopg.Connection, sql: str, params: dict) -> tuple[float, bool]:
    """Devuelve (Execution Time ms, ¿usó el índice GiST?)."""
    row = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", params).fetchone()
    plan = row[0][0]
    return plan["Execution Time"], plan_uses_index(plan["Plan"])


def time_query(conn: psycopg.Connection, sql: str, params: dict) -> tuple[float, int]:
    """Devuelve (tiempo cliente ms, filas devueltas)."""
    t0 = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    return (time.perf_counter() - t0) * 1e3, len(rows)


def run_query_type(conn, query_type: str, sql: str, queries: list[dict]) -> dict:
    client_ms, server_ms = [], []
    total_rows = 0
    used_index = True

    for query in queries:
        params = query_params(query_type, query)

        time_query(conn, sql, params)  # calentamiento (descartado)
        exec_ms, uses_index = explain_analyze(conn, sql, params)
        used_index &= uses_index

        per_query_ms = []
        for _ in range(REPETITIONS):
            elapsed_ms, rows = time_query(conn, sql, params)
            per_query_ms.append(elapsed_ms)
        client_ms.append(sum(per_query_ms) / REPETITIONS)
        server_ms.append(exec_ms)
        total_rows += rows

    return {
        "num_queries": len(queries),
        "repetitions": REPETITIONS,
        "avg_time_ms": sum(client_ms) / len(client_ms),
        "min_time_ms": min(client_ms),
        "max_time_ms": max(client_ms),
        "avg_server_time_ms": sum(server_ms) / len(server_ms),
        "total_rows_returned": total_rows,
        "used_index": used_index,
    }


# =============================================================================
# Entorno (versiones y hardware)
# =============================================================================

def total_ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def run_command(args: list[str]):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment(conn: psycopg.Connection) -> dict:
    limits = run_command(["docker", "inspect", CONTAINER_NAME, "--format",
                          "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}"])
    container_cpus = container_memory = None
    if limits:
        nano_cpus, memory = (int(v) for v in limits.split())
        container_cpus = nano_cpus / 1e9 or None
        container_memory = memory or None

    return {
        "postgres_version": conn.execute("SELECT version()").fetchone()[0],
        "postgis_version": conn.execute("SELECT PostGIS_Full_Version()").fetchone()[0],
        "docker_version": run_command(["docker", "--version"]),
        "container_cpus": container_cpus,
        "container_memory_bytes": container_memory,
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
        "host_cpu_count": os.cpu_count(),
        "host_ram_bytes": total_ram_bytes(),
        "python_version": platform.python_version(),
        "psycopg_version": psycopg.__version__,
    }


# =============================================================================
# Main
# =============================================================================

def run_experiment(conn, n: int, named_queries: dict[str, str], workload: dict) -> list[dict]:
    load_table(conn, n)
    build_time, index_size = build_index(conn)
    print(f"  N={n:,}: índice GiST {build_time:.3f}s, {index_size:,} B")

    rows = []
    for query_type, queries in workload.items():
        stats = run_query_type(conn, query_type, named_queries[query_type], queries)
        rows.append({
            "engine": ENGINE,
            "n": n,
            "query_type": query_type,
            "param": QUERY_PARAMS[query_type],
            **stats,
            "index_build_time_s": build_time,
            "index_space_bytes": index_size,
        })
        warning = "" if stats["used_index"] else "  ⚠ NO usó el índice GiST"
        print(f"    {query_type:<10} avg={stats['avg_time_ms']:.3f}ms  "
              f"server={stats['avg_server_time_ms']:.3f}ms  "
              f"rows={stats['total_rows_returned']}{warning}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="solo N=1,000")
    parser.add_argument("--dsn", default=os.environ.get("PG_DSN", DEFAULT_DSN))
    args = parser.parse_args()

    n_values = [1_000] if args.quick else N_VALUES
    named_queries = load_named_queries(QUERIES_SQL)
    workload = generate_queries()

    with connect(args.dsn) as conn:
        environment = collect_environment(conn)
        print(f"{environment['postgres_version']}\nPostGIS: {environment['postgis_version']}\n")

        rows = []
        for n in n_values:
            rows.extend(run_experiment(conn, n, named_queries, workload))

    out_path = write_results(rows, ENGINE)
    ENVIRONMENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_JSON.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"\nResultados escritos en {out_path}")
    print(f"Entorno escrito en {ENVIRONMENT_JSON}")

    if not all(row["used_index"] for row in rows):
        print("\nAdvertencia: alguna consulta no usó el índice GiST (ver columna used_index).")


if __name__ == "__main__":
    main()
