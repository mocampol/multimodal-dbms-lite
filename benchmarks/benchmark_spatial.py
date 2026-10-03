"""
benchmarks/benchmark_spatial.py — Lado DBMS propio de la comparación
espacial R-tree vs. PostgreSQL/PostGIS GiST (benchmarks/postgres/).

Para cada N:
    1. Crea la tabla places (id, name, geom POINT) en un directorio temporal
       y carga los puntos con INSERT por lotes (no cronometrado).
    2. Construye el índice RTREE con CREATE INDEX (cronometrado).
    3. Mide el tamaño del archivo .idx del R-tree.
    4. Ejecuta las mismas consultas que benchmarks/postgres/queries.sql,
       verificando que el plan use SpatialIndexScan.

El dataset, las consultas y el formato del CSV vienen de
benchmarks/spatial_common.py, compartido con benchmarks/postgres/load_data.py.

Equivalencias (distancia euclidiana; la métrica por defecto de DISTANCIA es
HAVERSINE, por eso se indica EUCLIDEAN explícitamente):
    radius      DISTANCIA(geom, POINT(x, y), EUCLIDEAN) <= r   ~ ST_DWithin
    knn         ORDER BY DISTANCIA(geom, POINT(x, y), EUCLIDEAN)
                LIMIT k                                       ~ ORDER BY <-> LIMIT k
    intersects  DENTRO_DE(geom, POLYGON(...))                  ~ ST_Intersects

Metodología: la misma que load_data.py (1 calentamiento descartado y luego
REPETITIONS repeticiones por consulta; tiempo = execute() completo, desde el
texto SQL hasta las filas, medido con perf_counter).

Uso:
    python benchmarks/benchmark_spatial.py           # corrida completa
    python benchmarks/benchmark_spatial.py --quick    # smoke test (N=1,000)
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "src"))
sys.path.insert(0, str(BENCH_DIR))

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.planner.plan_builder import build_select_plan
from query.query_engine import execute
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.heap_file import HeapFile

from spatial_common import (
    N_VALUES,
    QUERY_PARAMS,
    REPETITIONS,
    generate_points,
    generate_queries,
    write_results,
)

ENGINE = "minidbms"
INSERT_BATCH_SIZE = 1_000


def pool_size_for(n: int) -> int:
    # Mismo criterio que benchmarks/benchmark_indexes.py.
    return max(256, int(n / 90 * 1.5) + 64)


# =============================================================================
# Carga e índice
# =============================================================================

def make_catalog(base_dir: str, n: int) -> Catalog:
    # Fábricas propias en vez de importar src/main.py, que al importarse abre
    # el catálogo de data/ y corre la recuperación de transacciones.
    pool_size = pool_size_for(n)

    def heap_factory(schema):
        file_manager = FileManager(f"{base_dir}/{schema.table_name}.tbl")
        return HeapFile(schema, BufferManager(file_manager, pool_size=pool_size))

    def index_buffer_factory(table_name, column_name, index_id):
        file_manager = FileManager(f"{base_dir}/{table_name}.{column_name}.{index_id}.idx")
        return BufferManager(file_manager, pool_size=pool_size)

    return Catalog(
        heap_factory=heap_factory,
        storage_factories={StorageType.HEAP: heap_factory},
        index_buffer_factory=index_buffer_factory,
    )


def load_table(catalog: Catalog, n: int) -> None:
    execute("CREATE TABLE places (id INTEGER PRIMARY KEY, name VARCHAR(50), geom POINT);", catalog)
    points = generate_points(n)
    for start in range(0, n, INSERT_BATCH_SIZE):
        values = ", ".join(
            f"({pid}, '{name}', POINT({x!r}, {y!r}))"
            for pid, name, x, y in points[start:start + INSERT_BATCH_SIZE]
        )
        execute(f"INSERT INTO places VALUES {values};", catalog)


def build_index(catalog: Catalog, base_dir: str) -> tuple[float, int]:
    """Construye el R-tree; devuelve (tiempo de construcción s, tamaño bytes)."""
    t0 = time.perf_counter()
    execute("CREATE INDEX places_geom_rtree ON places (geom) USING RTREE;", catalog)
    build_time = time.perf_counter() - t0

    catalog.flush_table("places")
    entry = next(e for e in catalog.get_indexes("places") if e["index_type"] == "rtree")
    index_file = Path(base_dir) / f"places.geom.{entry['index_id']}.idx"
    return build_time, index_file.stat().st_size


# =============================================================================
# Consultas
# =============================================================================

def query_sql(query_type: str, query: dict) -> str:
    if query_type == "radius":
        return (f"SELECT id FROM places WHERE DISTANCIA(geom, "
                f"POINT({query['x']!r}, {query['y']!r}), EUCLIDEAN) <= {query['r']!r};")
    if query_type == "knn":
        return (f"SELECT id FROM places ORDER BY DISTANCIA(geom, "
                f"POINT({query['x']!r}, {query['y']!r}), EUCLIDEAN) LIMIT {query['k']};")
    if query_type == "intersects":
        ring = query["polygon"] + query["polygon"][:1]  # el DBMS exige repetir el primer punto
        points = ", ".join(f"POINT({x!r}, {y!r})" for x, y in ring)
        return f"SELECT id FROM places WHERE DENTRO_DE(geom, POLYGON({points}));"
    raise ValueError(f"Tipo de consulta desconocido: {query_type}")


def plan_uses_index(catalog: Catalog, sql: str) -> bool:
    plan = build_select_plan(Parser(Scanner(sql)).parse_sql_statements()[0], catalog)
    pending = [plan]
    while pending:
        node = pending.pop()
        if type(node).__name__ == "SpatialIndexScan":
            return True
        pending.extend(node.children())
    return False


def time_query(catalog: Catalog, sql: str) -> tuple[float, int]:
    """Devuelve (tiempo ms, filas devueltas)."""
    t0 = time.perf_counter()
    rows = execute(sql, catalog)
    return (time.perf_counter() - t0) * 1e3, len(rows)


def run_query_type(catalog: Catalog, query_type: str, queries: list[dict]) -> dict:
    times_ms = []
    total_rows = 0
    used_index = True

    for query in queries:
        sql = query_sql(query_type, query)
        used_index &= plan_uses_index(catalog, sql)

        time_query(catalog, sql)  # calentamiento (descartado)
        per_query_ms = []
        for _ in range(REPETITIONS):
            elapsed_ms, rows = time_query(catalog, sql)
            per_query_ms.append(elapsed_ms)
        times_ms.append(sum(per_query_ms) / REPETITIONS)
        total_rows += rows

    return {
        "num_queries": len(queries),
        "repetitions": REPETITIONS,
        "avg_time_ms": sum(times_ms) / len(times_ms),
        "min_time_ms": min(times_ms),
        "max_time_ms": max(times_ms),
        "avg_server_time_ms": "",
        "total_rows_returned": total_rows,
        "used_index": used_index,
    }


# =============================================================================
# Main
# =============================================================================

def run_experiment(n: int, workload: dict) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = make_catalog(tmp, n)
        load_table(catalog, n)
        build_time, index_size = build_index(catalog, tmp)
        print(f"  N={n:,}: índice R-tree {build_time:.3f}s, {index_size:,} B")

        rows = []
        for query_type, queries in workload.items():
            stats = run_query_type(catalog, query_type, queries)
            rows.append({
                "engine": ENGINE,
                "n": n,
                "query_type": query_type,
                "param": QUERY_PARAMS[query_type],
                **stats,
                "index_build_time_s": build_time,
                "index_space_bytes": index_size,
            })
            warning = "" if stats["used_index"] else "  ⚠ NO usó el R-tree"
            print(f"    {query_type:<10} avg={stats['avg_time_ms']:.3f}ms  "
                  f"rows={stats['total_rows_returned']}{warning}")
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="solo N=1,000")
    args = parser.parse_args()

    n_values = [1_000] if args.quick else N_VALUES
    workload = generate_queries()

    rows = []
    for n in n_values:
        rows.extend(run_experiment(n, workload))

    out_path = write_results(rows, ENGINE)
    print(f"\nResultados escritos en {out_path}")

    if not all(row["used_index"] for row in rows):
        print("\nAdvertencia: alguna consulta no usó el R-tree (ver columna used_index).")


if __name__ == "__main__":
    main()
