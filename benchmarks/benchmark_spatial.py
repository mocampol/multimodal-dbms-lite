"""
benchmarks/benchmark_spatial.py — Comparación experimental de tres técnicas
de búsqueda espacial: búsqueda secuencial, R-Tree propio y PostgreSQL +
PostGIS (índice GiST).

Issue: "Crear el experimento exigido para comparar las tres técnicas"
(range search por radio y k-NN; build, query, memoria, disco, exactitud).

Metodología
-----------
    - N = 1,000 / 10,000 / 100,000 puntos.
    - Radios: 1, 5 y 10 km.            k-NN: k = 10, 50 y 100.
    - 100 consultas por configuración (mismos centros para las 3 técnicas).
    - Semilla fija (SEED): datos y centros de consulta son reproducibles.
    - Warm-up: WARMUP_QUERIES consultas NO medidas antes de cada celda.
    - Repeticiones: cada técnica se reconstruye y se mide REPETITIONS veces
      (build + warm-up + 100 consultas por configuración); se reporta media
      y desviación estándar entre repeticiones.
    - Memoria: pasada aparte con tracemalloc (no contamina los tiempos).
    - Exactitud: la búsqueda secuencial es la referencia. Cada resultado de
      las otras técnicas se compara contra ella (ver classify_*).

Métrica y datos
---------------
    Distancia = Haversine en metros, esfera de radio 6 371 008.8 m (la misma
    que spatial/distance.py y el README). Puntos sintéticos uniformes en una
    región de ~100 km x 100 km (REGION, alrededor de Lima). En PostGIS se usa
    `geography(Point, 4326)` con use_spheroid=false para que el modelo de
    la Tierra sea el mismo.

Qué mide cada técnica (todas devuelven ids de la tabla base)
-----------------------------------------------------------
    seq     : HeapFile.scan() completo + filtro Haversine (lo que hace hoy
              SeqScan + SpatialFilter en el motor).
    rtree   : RTree.range_query() sobre las cajas conservadoras de
              distance_bounding_boxes() -> HeapFile.get(rid) -> filtro
              Haversine exacto. k-NN: radio expansivo (1 km, x2, ...) igual
              que KNNScan._open_index_top_k().
    postgis : ST_DWithin(...) / ORDER BY geog <-> punto LIMIT k contra un
              índice GiST. El tiempo incluye el viaje cliente-servidor y
              fetchall() de los ids (el cliente también es Python).

Uso
---
    python benchmarks/benchmark_spatial.py --quick          # humo, ~1-2 min
    python benchmarks/benchmark_spatial.py                  # corrida completa
    PG_DSN="postgresql://postgres:postgres@localhost:5433/postgres" \\
        python benchmarks/benchmark_spatial.py
    python benchmarks/benchmark_spatial.py --no-postgres    # omite PostGIS
    python benchmarks/benchmark_spatial.py --help
"""

import argparse
import csv
import heapq
import io
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from common.schema import Schema, Column
from common.value import DataType, Value
from common.record import Record
from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile
from index.rtree import RTree
from spatial.geometry import Point2D
from spatial.distance import haversine_distance, EARTH_MEAN_RADIUS_METERS
from query.executor.access.spatial_scan import distance_bounding_boxes


# =============================================================================
# Configuración del experimento
# =============================================================================

SEED = 42
N_VALUES = [1_000, 10_000, 100_000]
RADII_KM = [1, 5, 10]
K_VALUES = [10, 50, 100]
N_QUERIES = 100
WARMUP_QUERIES = 10
REPETITIONS = 3
MEMORY_PROBE_QUERIES = 5

# (lon_min, lat_min, lon_max, lat_max) en grados. ~108 km x ~111 km.
REGION = (-78.0, -12.5, -77.0, -11.5)

TOLERANCE_M = 0.05          # diferencia admitida entre modelos de distancia (m)
CELL_BUDGET_S = 900.0       # presupuesto de tiempo por celda y repetición
PG_QUERY_TIMEOUT_S = 120.0  # statement_timeout en PostgreSQL

# Ajustes de sesión de PostgreSQL (documentados en el resumen): las técnicas
# Python son de un solo hilo; se desactiva el paralelismo y el JIT para que
# la comparación no mezcle varios núcleos / compilación JIT.
PG_SESSION_SETTINGS = {
    "max_parallel_workers_per_gather": "0",
    "jit": "off",
}
PG_SCHEMA = "bench_spatial"
PG_INDEX_NAME = "bench_points_geog_gist"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

NA = "N/A"
TECHNIQUES = {
    "seq": ("Búsqueda secuencial", "tab:gray"),
    "rtree": ("R-Tree propio", "tab:blue"),
    "postgis": ("PostgreSQL + PostGIS (GiST)", "tab:green"),
}
SKIPPED_NO_POSTGRES = "OMITIDO_SIN_POSTGRES"


@dataclass(frozen=True)
class QueryConfig:
    kind: str       # "range" | "knn"
    param: int      # km o k

    @property
    def unit(self) -> str:
        return "km" if self.kind == "range" else "k"

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.param}"


def all_configs() -> list[QueryConfig]:
    return ([QueryConfig("range", km) for km in RADII_KM]
            + [QueryConfig("knn", k) for k in K_VALUES])


# =============================================================================
# Datos y distancia
# =============================================================================

def gen_points(n: int, seed: int) -> list[tuple[int, float, float]]:
    rng = random.Random(seed * 1_000_003 + n)
    lon0, lat0, lon1, lat1 = REGION
    return [(i, rng.uniform(lon0, lon1), rng.uniform(lat0, lat1)) for i in range(n)]


def gen_centers(n: int, count: int, seed: int, salt: int) -> list[tuple[float, float]]:
    rng = random.Random(seed * 7_919 + n + salt)
    lon0, lat0, lon1, lat1 = REGION
    return [(rng.uniform(lon0, lon1), rng.uniform(lat0, lat1)) for _ in range(count)]


def make_dist(lon: float, lat: float):
    """Haversine (m) desde un centro fijo; misma aritmética que
    spatial.distance.haversine_distance, con el centro precalculado."""
    p1 = math.radians(lat)
    cos1 = math.cos(p1)
    radius = EARTH_MEAN_RADIUS_METERS

    def dist(lon2: float, lat2: float) -> float:
        p2 = math.radians(lat2)
        dl = math.radians(lon2 - lon)
        dl = (dl + math.pi) % (2 * math.pi) - math.pi
        h = math.sin((p2 - p1) / 2) ** 2 + cos1 * math.cos(p2) * math.sin(dl / 2) ** 2
        h = min(1.0, max(0.0, h))
        return radius * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))

    return dist


def self_check_distance() -> None:
    """Falla pronto si la versión rápida difiere de la del proyecto."""
    rng = random.Random(SEED)
    lon0, lat0, lon1, lat1 = REGION
    for _ in range(50):
        a = (rng.uniform(lon0, lon1), rng.uniform(lat0, lat1))
        b = (rng.uniform(lon0, lon1), rng.uniform(lat0, lat1))
        fast = make_dist(*a)(*b)
        ref = haversine_distance(Point2D(*a), Point2D(*b))
        if abs(fast - ref) > 1e-6:
            raise AssertionError(f"make_dist difiere de haversine_distance: {fast} vs {ref}")


# =============================================================================
# Técnicas
# =============================================================================

def make_schema() -> Schema:
    return Schema("spatial_bench", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("lon", DataType.DOUBLE_PRECISION),
        Column("lat", DataType.DOUBLE_PRECISION),
    ])


class HeapTable:
    """Tabla base compartida por la búsqueda secuencial y el R-Tree."""

    def __init__(self, points, tmp: str):
        n = len(points)
        self.fm = FileManager(str(Path(tmp) / "points.tbl"))
        self.bm = BufferManager(self.fm, pool_size=max(256, n // 80 + 128))
        self.heap = HeapFile(make_schema(), self.bm)
        t0 = time.perf_counter()
        self.rids = [
            self.heap.insert(Record([
                Value(DataType.INTEGER, pid),
                Value(DataType.DOUBLE_PRECISION, lon),
                Value(DataType.DOUBLE_PRECISION, lat),
            ]))
            for pid, lon, lat in points
        ]
        self.bm.flush_all()
        self.load_time = time.perf_counter() - t0
        self.points = points
        self.table_bytes = os.path.getsize(self.fm.file_path)


class SeqTechnique:
    name = "seq"

    def __init__(self, table: HeapTable, tmp: str):
        self.table = table

    def build(self) -> float:
        return 0.0

    def index_bytes(self) -> int:
        return 0

    def range_ids(self, center, radius_m):
        dist = make_dist(*center)
        return [rec[0].data for rec in self.table.heap.scan()
                if dist(rec[1].data, rec[2].data) <= radius_m]

    def knn_ids(self, center, k):
        dist = make_dist(*center)
        best = heapq.nsmallest(
            k, ((dist(rec[1].data, rec[2].data), rec[0].data)
                for rec in self.table.heap.scan()))
        return [pid for _, pid in best]

    def close(self):
        pass


class RTreeTechnique:
    name = "rtree"

    def __init__(self, table: HeapTable, tmp: str):
        n = len(table.points)
        self.table = table
        self.fm = FileManager(str(Path(tmp) / "rtree.idx"))
        self.bm = BufferManager(self.fm, pool_size=max(256, n // 40 + 128))
        self.tree = None

    def build(self) -> float:
        self.tree = RTree(self.bm)
        points, rids = self.table.points, self.table.rids
        t0 = time.perf_counter()
        for (_, lon, lat), rid in zip(points, rids):
            self.tree.insert(Point2D(lon, lat), rid)
        elapsed = time.perf_counter() - t0
        self.bm.flush_all()
        return elapsed

    def index_bytes(self) -> int:
        return os.path.getsize(self.fm.file_path)

    def _candidates(self, center, radius_m, dist):
        """(distancia, id) de los puntos a <= radius_m, vía R-Tree + heap."""
        out = []
        boxes = distance_bounding_boxes(Point2D(*center), radius_m, "HAVERSINE")
        rids = dict.fromkeys(r for box in boxes for r in self.tree.range_query(box))
        for rid in rids:
            rec = self.table.heap.get(rid)
            if rec is None:
                continue
            d = dist(rec[1].data, rec[2].data)
            if d <= radius_m:
                out.append((d, rec[0].data))
        return out

    def range_ids(self, center, radius_m):
        return [pid for _, pid in self._candidates(center, radius_m, make_dist(*center))]

    def knn_ids(self, center, k):
        dist = make_dist(*center)
        radius, max_radius = 1000.0, math.pi * EARTH_MEAN_RADIUS_METERS
        while True:
            cands = self._candidates(center, radius, dist)
            if len(cands) >= k or radius >= max_radius:
                cands.sort()
                return [pid for _, pid in cands[:k]]
            radius = min(radius * 2, max_radius)

    def close(self):
        pass


class PostGISTechnique:
    name = "postgis"

    def __init__(self, dsn: str, query_timeout_s: float):
        import psycopg2  # import perezoso: solo si se usa PostgreSQL
        self._psycopg2 = psycopg2
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        for name, value in PG_SESSION_SETTINGS.items():
            self.cur.execute(f"SET {name} = %s", (value,))
        self.cur.execute("SET statement_timeout = %s", (int(query_timeout_s * 1000),))
        self.load_time = 0.0
        self.table_bytes = 0

    # ---- entorno -------------------------------------------------------
    def environment(self) -> dict:
        cur = self.cur
        info = {}
        cur.execute("SELECT version()")
        info["postgres_version"] = cur.fetchone()[0]
        cur.execute("SELECT postgis_version()")
        info["postgis_version"] = cur.fetchone()[0]
        for setting in ("shared_buffers", "work_mem", "effective_cache_size",
                        "random_page_cost", "max_parallel_workers_per_gather", "jit"):
            cur.execute("SELECT current_setting(%s)", (setting,))
            info[setting] = cur.fetchone()[0]
        cur.execute(
            "SELECT ST_Distance(a, b, false), a <-> b FROM ("
            " SELECT ST_SetSRID(ST_MakePoint(-77.5,-12.0),4326)::geography a,"
            "        ST_SetSRID(ST_MakePoint(-77.4,-12.0),4326)::geography b) t")
        sphere, knn_op = cur.fetchone()
        info["knn_operator_model"] = (
            "esfera" if abs(sphere - knn_op) <= 0.01 else "esferoide (difiere de la esfera)")
        return info

    def memory_probe(self):
        try:
            self.cur.execute("SELECT sum(total_bytes) FROM pg_backend_memory_contexts")
            return int(self.cur.fetchone()[0])
        except Exception:
            return None

    # ---- carga e índice ------------------------------------------------
    def load(self, points) -> float:
        s = PG_SCHEMA
        cur = self.cur
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        cur.execute(f"CREATE SCHEMA {s}")
        cur.execute(f"CREATE TABLE {s}.staging (id integer, lon double precision, lat double precision)")
        cur.execute(
            f"CREATE TABLE {s}.bench_points (id integer PRIMARY KEY, "
            "lon double precision, lat double precision, geog geography(Point, 4326))")
        buf = io.StringIO()
        for pid, lon, lat in points:
            buf.write(f"{pid},{lon!r},{lat!r}\n")
        buf.seek(0)
        t0 = time.perf_counter()
        cur.copy_expert(f"COPY {s}.staging (id, lon, lat) FROM STDIN WITH (FORMAT csv)", buf)
        cur.execute(
            f"INSERT INTO {s}.bench_points SELECT id, lon, lat, "
            "ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography FROM " + f"{s}.staging")
        self.load_time = time.perf_counter() - t0
        cur.execute(f"DROP TABLE {s}.staging")
        cur.execute(f"VACUUM (ANALYZE) {s}.bench_points")
        cur.execute(f"SELECT pg_relation_size('{s}.bench_points')")
        self.table_bytes = int(cur.fetchone()[0])
        return self.load_time

    def build(self) -> float:
        s = PG_SCHEMA
        t0 = time.perf_counter()
        self.cur.execute(f"CREATE INDEX {PG_INDEX_NAME} ON {s}.bench_points USING GIST (geog)")
        elapsed = time.perf_counter() - t0
        self.cur.execute(f"ANALYZE {s}.bench_points")
        return elapsed

    def index_bytes(self) -> int:
        self.cur.execute(f"SELECT pg_relation_size('{PG_SCHEMA}.{PG_INDEX_NAME}')")
        return int(self.cur.fetchone()[0])

    # ---- consultas -----------------------------------------------------
    _POINT = "ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography"

    def _range_sql(self):
        return (f"SELECT id FROM {PG_SCHEMA}.bench_points "
                f"WHERE ST_DWithin(geog, {self._POINT}, %s, false)")

    def _knn_sql(self):
        return (f"SELECT id FROM {PG_SCHEMA}.bench_points "
                f"ORDER BY geog <-> {self._POINT} LIMIT %s")

    def range_ids(self, center, radius_m):
        self.cur.execute(self._range_sql(), (center[0], center[1], radius_m))
        return [row[0] for row in self.cur.fetchall()]

    def knn_ids(self, center, k):
        self.cur.execute(self._knn_sql(), (center[0], center[1], k))
        return [row[0] for row in self.cur.fetchall()]

    def explain(self, cfg: QueryConfig, center) -> str:
        if cfg.kind == "range":
            sql, params = self._range_sql(), (center[0], center[1], cfg.param * 1000.0)
        else:
            sql, params = self._knn_sql(), (center[0], center[1], cfg.param)
        self.cur.execute("EXPLAIN " + sql, params)
        return "\n".join(row[0] for row in self.cur.fetchall())

    def close(self, drop: bool = True):
        try:
            if drop:
                self.cur.execute(f"DROP SCHEMA IF EXISTS {PG_SCHEMA} CASCADE")
        except Exception:
            pass
        finally:
            self.conn.close()


def run_query(tech, cfg: QueryConfig, center):
    if cfg.kind == "range":
        return tech.range_ids(center, cfg.param * 1000.0)
    return tech.knn_ids(center, cfg.param)


# =============================================================================
# Medición
# =============================================================================

@dataclass
class CellAgg:
    rep_means: list = field(default_factory=list)
    times_ms: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    executed: int = 0
    discarded: int = 0
    errors: int = 0
    error_samples: list = field(default_factory=list)
    results: dict = field(default_factory=dict)   # solo repetición 0
    plan: str = ""


def run_cell(tech, cfg, centers, warm_centers, budget_s, keep_results, agg: CellAgg):
    """Warm-up + consultas medidas. Descarta (y cuenta) las consultas que
    fallan o que ya no caben en el presupuesto de tiempo de la celda."""
    for c in warm_centers:
        try:
            run_query(tech, cfg, c)
        except Exception as exc:
            agg.error_samples.append(f"warm-up: {type(exc).__name__}: {exc}"[:200])
            break

    times = []
    start = time.perf_counter()
    for qi, center in enumerate(centers):
        if time.perf_counter() - start > budget_s:
            agg.discarded += len(centers) - qi
            agg.error_samples.append(
                f"presupuesto de celda ({budget_s:.0f}s) agotado tras {qi} consultas")
            break
        t0 = time.perf_counter()
        try:
            ids = run_query(tech, cfg, center)
        except Exception as exc:
            agg.errors += 1
            agg.discarded += 1
            agg.error_samples.append(f"{type(exc).__name__}: {exc}"[:200])
            continue
        times.append((time.perf_counter() - t0) * 1000.0)
        agg.sizes.append(len(ids))
        if keep_results:
            agg.results[qi] = ids
    agg.executed += len(times)
    agg.times_ms.extend(times)
    if times:
        agg.rep_means.append(statistics.fmean(times))


def memory_pass(points, centers) -> dict:
    """Pico de memoria Python (tracemalloc) de construcción y consulta.
    Pasada separada: tracemalloc ralentiza todo, así que no se cronometra."""
    out = {"seq": {}, "rtree": {}}
    probe = centers[:MEMORY_PROBE_QUERIES]
    heavy = {"range": QueryConfig("range", max(RADII_KM)),
             "knn": QueryConfig("knn", max(K_VALUES))}
    tracemalloc.start()
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            table = HeapTable(points, tmp)
            base, _ = tracemalloc.get_traced_memory()
            seq = SeqTechnique(table, tmp)
            for kind, cfg in heavy.items():
                tracemalloc.reset_peak()
                cur0, _ = tracemalloc.get_traced_memory()
                for c in probe:
                    run_query(seq, cfg, c)
                out["seq"][f"query_{kind}_peak"] = tracemalloc.get_traced_memory()[1] - cur0

            tracemalloc.reset_peak()
            cur0, _ = tracemalloc.get_traced_memory()
            rt = RTreeTechnique(table, tmp)
            rt.build()
            cur1, peak = tracemalloc.get_traced_memory()
            out["rtree"]["build_peak"] = peak - cur0
            out["rtree"]["resident"] = cur1 - cur0
            for kind, cfg in heavy.items():
                tracemalloc.reset_peak()
                cur0, _ = tracemalloc.get_traced_memory()
                for c in probe:
                    run_query(rt, cfg, c)
                out["rtree"][f"query_{kind}_peak"] = tracemalloc.get_traced_memory()[1] - cur0
    finally:
        tracemalloc.stop()
    return out


# =============================================================================
# Equivalencia de resultados
# =============================================================================

def classify_range(ref, got, center, radius_m, coords, tol) -> str:
    if len(set(got)) != len(got):
        return "mismatch"
    a, b = set(ref), set(got)
    if a == b:
        return "exact"
    dist = make_dist(*center)
    for pid in a ^ b:
        if abs(dist(*coords[pid]) - radius_m) > tol:
            return "mismatch"
    return "tolerance"      # solo difieren puntos en el borde del radio


def classify_knn(ref, got, center, coords, tol) -> str:
    if len(set(got)) != len(got) or len(ref) != len(got):
        return "mismatch"
    if ref == got:
        return "exact"
    dist = make_dist(*center)
    for r, g in zip(ref, got):
        if abs(dist(*coords[r]) - dist(*coords[g])) > tol:
            return "mismatch"
    return "tolerance"      # mismos vecinos salvo empates / cuasi-empates


def compare_cell(ref: CellAgg, got: CellAgg, cfg, centers, coords, tol) -> dict:
    counts = {"exact": 0, "tolerance": 0, "mismatch": 0}
    for qi, ref_ids in ref.results.items():
        if qi not in got.results:
            continue
        if cfg.kind == "range":
            verdict = classify_range(ref_ids, got.results[qi], centers[qi],
                                     cfg.param * 1000.0, coords, tol)
        else:
            verdict = classify_knn(ref_ids, got.results[qi], centers[qi], coords, tol)
        counts[verdict] += 1
    compared = sum(counts.values())
    return {"compared": compared, **counts}


# =============================================================================
# Orquestación
# =============================================================================

def run_n(n, args, pg_state) -> list[dict]:
    points = gen_points(n, args.seed)
    coords = [(lon, lat) for _, lon, lat in points]      # índice == id
    centers = gen_centers(n, args.queries, args.seed, salt=1)
    warm = gen_centers(n, args.warmup, args.seed, salt=2)
    cfgs = all_configs()

    aggs = {(t, c.key): CellAgg() for t in TECHNIQUES for c in cfgs}
    status = {t: "OK" for t in TECHNIQUES}
    build_times = {t: [] for t in TECHNIQUES}
    load_times = {"seq": [], "rtree": [], "postgis": []}
    index_bytes = {t: None for t in TECHNIQUES}
    table_bytes = {t: None for t in TECHNIQUES}
    pg_memory = None

    # ---------- técnicas Python (comparten tabla base) ----------
    for rep in range(args.repetitions):
        print(f"  [rep {rep + 1}/{args.repetitions}] búsqueda secuencial + R-Tree")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            table = HeapTable(points, tmp)
            for cls in (SeqTechnique, RTreeTechnique):
                tech = cls(table, tmp)
                if status[tech.name] != "OK":
                    continue
                try:
                    build_times[tech.name].append(tech.build())
                    load_times[tech.name].append(table.load_time)
                    index_bytes[tech.name] = tech.index_bytes()
                    table_bytes[tech.name] = table.table_bytes
                    for cfg in cfgs:
                        run_cell(tech, cfg, centers, warm, args.cell_budget_s,
                                 rep == 0, aggs[(tech.name, cfg.key)])
                except Exception as exc:
                    status[tech.name] = f"ERROR: {type(exc).__name__}: {exc}"[:300]
                    print(f"    ! {tech.name}: {status[tech.name]}")
                finally:
                    tech.close()

    # ---------- PostgreSQL + PostGIS ----------
    if not pg_state["available"]:
        status["postgis"] = pg_state["reason"]
    else:
        for rep in range(args.repetitions):
            print(f"  [rep {rep + 1}/{args.repetitions}] PostgreSQL + PostGIS")
            try:
                pg = PostGISTechnique(args.pg_dsn, args.pg_timeout_s)
            except Exception as exc:
                status["postgis"] = f"ERROR_CONEXION: {type(exc).__name__}: {exc}"[:300]
                break
            try:
                load_times["postgis"].append(pg.load(points))
                build_times["postgis"].append(pg.build())
                index_bytes["postgis"] = pg.index_bytes()
                table_bytes["postgis"] = pg.table_bytes
                for cfg in cfgs:
                    agg = aggs[("postgis", cfg.key)]
                    if rep == 0:
                        agg.plan = pg.explain(cfg, centers[0])
                    run_cell(pg, cfg, centers, warm, args.cell_budget_s, rep == 0, agg)
                if rep == 0:
                    pg_memory = pg.memory_probe()
            except Exception as exc:
                status["postgis"] = f"ERROR: {type(exc).__name__}: {exc}"[:300]
                print(f"    ! postgis: {status['postgis']}")
            finally:
                pg.close(drop=not args.keep_pg_data)
            if status["postgis"] != "OK":
                break

    # ---------- memoria ----------
    print("  [memoria] pasada con tracemalloc")
    try:
        mem = memory_pass(points, centers)
    except Exception as exc:
        print(f"    ! memoria: {type(exc).__name__}: {exc}")
        mem = {"seq": {}, "rtree": {}}

    # ---------- filas ----------
    rows = []
    ref_cfg_aggs = {c.key: aggs[("seq", c.key)] for c in cfgs}
    for tech in TECHNIQUES:
        for cfg in cfgs:
            agg = aggs[(tech, cfg.key)]
            ok = status[tech] == "OK" and agg.rep_means
            row = {
                "n": n, "technique": tech, "technique_label": TECHNIQUES[tech][0],
                "query_type": cfg.kind, "param": cfg.param, "param_unit": cfg.unit,
                "status": status[tech] if status[tech] != "OK" else ("OK" if ok else "SIN_DATOS"),
                "repetitions": args.repetitions,
                "queries_requested": args.queries * args.repetitions,
                "queries_executed": agg.executed,
                "queries_discarded": agg.discarded,
                "query_errors": agg.errors,
                "error_samples": " | ".join(dict.fromkeys(agg.error_samples))[:400] or "",
            }
            blank = {k: NA for k in (
                "avg_query_ms", "std_query_ms", "median_query_ms", "p95_query_ms",
                "avg_result_size", "build_time_s", "build_time_std_s", "base_load_time_s",
                "index_disk_bytes", "table_disk_bytes", "mem_build_peak_bytes",
                "mem_index_resident_bytes", "mem_query_peak_bytes", "mem_server_backend_bytes",
                "compared_queries", "exact_matches", "tolerance_matches", "mismatches",
                "accuracy_pct", "plan_uses_gist")}
            row.update(blank)
            if ok:
                pooled = sorted(agg.times_ms)
                row["avg_query_ms"] = statistics.fmean(agg.rep_means)
                row["std_query_ms"] = statistics.pstdev(agg.rep_means) if len(agg.rep_means) > 1 else 0.0
                row["median_query_ms"] = statistics.median(pooled)
                row["p95_query_ms"] = pooled[min(len(pooled) - 1, int(0.95 * len(pooled)))]
                row["avg_result_size"] = statistics.fmean(agg.sizes) if agg.sizes else 0.0
                bt = build_times[tech]
                row["build_time_s"] = statistics.fmean(bt) if bt else NA
                row["build_time_std_s"] = statistics.pstdev(bt) if len(bt) > 1 else 0.0
                lt = load_times[tech]
                row["base_load_time_s"] = statistics.fmean(lt) if lt else NA
                row["index_disk_bytes"] = index_bytes[tech] if index_bytes[tech] is not None else NA
                row["table_disk_bytes"] = table_bytes[tech] if table_bytes[tech] is not None else NA
                m = mem.get(tech, {})
                if tech == "rtree":
                    row["mem_build_peak_bytes"] = m.get("build_peak", NA)
                    row["mem_index_resident_bytes"] = m.get("resident", NA)
                if tech in ("seq", "rtree"):
                    row["mem_query_peak_bytes"] = m.get(f"query_{cfg.kind}_peak", NA)
                if tech == "postgis":
                    row["mem_server_backend_bytes"] = pg_memory if pg_memory is not None else NA
                    row["plan_uses_gist"] = PG_INDEX_NAME in agg.plan
                if tech == "seq":
                    row.update({"compared_queries": len(agg.results), "exact_matches": len(agg.results),
                                "tolerance_matches": 0, "mismatches": 0, "accuracy_pct": 100.0})
                elif ref_cfg_aggs[cfg.key].results:
                    cmp = compare_cell(ref_cfg_aggs[cfg.key], agg, cfg, centers, coords, args.tolerance_m)
                    row["compared_queries"] = cmp["compared"]
                    row["exact_matches"] = cmp["exact"]
                    row["tolerance_matches"] = cmp["tolerance"]
                    row["mismatches"] = cmp["mismatch"]
                    row["accuracy_pct"] = (100.0 * (cmp["exact"] + cmp["tolerance"]) / cmp["compared"]
                                           if cmp["compared"] else NA)
            row["_plan"] = agg.plan
            rows.append(row)

        r0 = next(r for r in rows if r["technique"] == tech)
        b = r0["build_time_s"]
        print(f"  {tech:8s} status={r0['status'][:40]:40s} "
              f"build={b if isinstance(b, str) else format(b, '.3f') + 's'}")
    return rows


def run_all(args):
    self_check_distance()
    pg_state = {"available": False, "reason": SKIPPED_NO_POSTGRES, "env": {}}
    if args.no_postgres or not args.pg_dsn:
        pg_state["reason"] = (f"{SKIPPED_NO_POSTGRES}: " +
                              ("--no-postgres" if args.no_postgres else "PG_DSN no definido"))
    else:
        try:
            probe = PostGISTechnique(args.pg_dsn, args.pg_timeout_s)
            probe.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
            pg_state["env"] = probe.environment()
            probe.close(drop=False)
            pg_state["available"] = True
            print(f"PostgreSQL OK: {pg_state['env']['postgres_version']}")
        except Exception as exc:
            pg_state["reason"] = f"{SKIPPED_NO_POSTGRES}: {type(exc).__name__}: {exc}"[:300]
            print(f"! PostgreSQL no disponible, se omite: {pg_state['reason']}")

    rows = []
    for n in args.sizes:
        print(f"\n=== N = {n:,} ===")
        rows.extend(run_n(n, args, pg_state))
    return rows, pg_state


# =============================================================================
# Salidas: CSV, gráficas, resumen
# =============================================================================

CSV_FIELDS = [
    "n", "technique", "technique_label", "query_type", "param", "param_unit", "status",
    "repetitions", "queries_requested", "queries_executed", "queries_discarded",
    "query_errors", "error_samples", "avg_query_ms", "std_query_ms", "median_query_ms",
    "p95_query_ms", "avg_result_size", "build_time_s", "build_time_std_s",
    "base_load_time_s", "index_disk_bytes", "table_disk_bytes", "mem_build_peak_bytes",
    "mem_index_resident_bytes", "mem_query_peak_bytes", "mem_server_backend_bytes",
    "compared_queries", "exact_matches", "tolerance_matches", "mismatches",
    "accuracy_pct", "plan_uses_gist",
]


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados crudos escritos en {path}")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def make_charts(rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    ns = sorted({r["n"] for r in rows})

    def series(tech, field, **match):
        xs, ys = [], []
        for n in ns:
            for r in rows:
                if (r["n"] == n and r["technique"] == tech and _num(r[field])
                        and all(r[k] == v for k, v in match.items())):
                    xs.append(n)
                    ys.append(r[field])
                    break
        return xs, ys

    def finish(path, title, xlabel="N (puntos)", ylabel=""):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    # 1) tiempo de construcción del índice
    plt.figure(figsize=(7, 5))
    for tech in ("rtree", "postgis"):
        xs, ys = series(tech, "build_time_s", query_type="range", param=RADII_KM[0])
        if xs:
            plt.plot(xs, ys, marker="o", label=TECHNIQUES[tech][0], color=TECHNIQUES[tech][1])
    plt.xscale("log"); plt.yscale("log"); plt.legend()
    finish(out_dir / "spatial_build_time.png", "Tiempo de construcción del índice "
           "(secuencial: sin índice)", ylabel="Tiempo (s)")

    # 2) y 3) tiempo promedio de consulta
    for kind, params, unit, fname, title in (
        ("range", RADII_KM, "km", "spatial_range_query_time.png", "Búsqueda por radio"),
        ("knn", K_VALUES, "k", "spatial_knn_query_time.png", "k-NN"),
    ):
        fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4.5), sharey=True)
        for ax, p in zip(np.atleast_1d(axes), params):
            for tech in TECHNIQUES:
                xs, ys = series(tech, "avg_query_ms", query_type=kind, param=p)
                if xs:
                    ax.plot(xs, ys, marker="o", label=TECHNIQUES[tech][0], color=TECHNIQUES[tech][1])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{title}: {p} {unit}")
            ax.set_xlabel("N (puntos)")
            ax.grid(True, which="both", alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("Tiempo promedio por consulta (ms)")
        np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # 4) espacio en disco del índice
    def grouped_bars(path, title, ylabel, specs):
        """specs: [(label, color, {n: valor})]"""
        x = np.arange(len(ns))
        width = 0.8 / max(1, len(specs))
        plt.figure(figsize=(7.5, 5))
        for i, (label, color, values) in enumerate(specs):
            heights = [values.get(n, 0) for n in ns]
            bars = plt.bar(x + (i - (len(specs) - 1) / 2) * width, heights, width,
                           label=label, color=color)
            for bar, n in zip(bars, ns):
                if n not in values:
                    plt.text(bar.get_x() + bar.get_width() / 2, 0, "N/A", ha="center",
                             va="bottom", fontsize=8, rotation=90)
        plt.xticks(x, [f"{n:,}" for n in ns])
        plt.yscale("log"); plt.legend(fontsize=8)
        finish(path, title, ylabel=ylabel)

    def first(tech, field):
        out = {}
        for r in rows:
            if r["technique"] == tech and _num(r[field]) and r["n"] not in out:
                out[r["n"]] = r[field] / 1024
        return out

    grouped_bars(out_dir / "spatial_index_space.png", "Espacio en disco del índice",
                 "KB (escala log)",
                 [(TECHNIQUES["rtree"][0], TECHNIQUES["rtree"][1], first("rtree", "index_disk_bytes")),
                  (TECHNIQUES["postgis"][0], TECHNIQUES["postgis"][1], first("postgis", "index_disk_bytes"))])

    # 5) memoria
    def mem_by_n(tech, field, kind=None):
        out = {}
        for r in rows:
            if (r["technique"] == tech and _num(r[field]) and r["n"] not in out
                    and (kind is None or r["query_type"] == kind)):
                out[r["n"]] = r[field] / 1024
        return out

    grouped_bars(out_dir / "spatial_memory.png",
                 "Memoria Python (tracemalloc); PostgreSQL no es comparable", "KB (escala log)",
                 [("Secuencial: pico consulta k-NN", TECHNIQUES["seq"][1], mem_by_n("seq", "mem_query_peak_bytes", "knn")),
                  ("R-Tree: pico consulta k-NN", TECHNIQUES["rtree"][1], mem_by_n("rtree", "mem_query_peak_bytes", "knn")),
                  ("R-Tree: pico construcción", "tab:cyan", mem_by_n("rtree", "mem_build_peak_bytes"))])

    # 6) exactitud
    x = np.arange(len(ns))
    plt.figure(figsize=(7.5, 5))
    for i, tech in enumerate(("rtree", "postgis")):
        vals = []
        for n in ns:
            accs = [r["accuracy_pct"] for r in rows
                    if r["n"] == n and r["technique"] == tech and _num(r["accuracy_pct"])]
            vals.append(min(accs) if accs else 0)
        plt.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=TECHNIQUES[tech][0], color=TECHNIQUES[tech][1])
    plt.xticks(x, [f"{n:,}" for n in ns]); plt.ylim(0, 105); plt.legend(fontsize=8)
    finish(out_dir / "spatial_accuracy.png",
           "Exactitud vs. búsqueda secuencial (peor configuración)", ylabel="% consultas equivalentes")
    print(f"Gráficas guardadas en {out_dir}")


def _f(v, nd=2) -> str:
    return f"{v:,.{nd}f}" if _num(v) else str(v)


def write_summary(rows, pg_state, args, path: Path):
    ns = sorted({r["n"] for r in rows})
    cfgs = all_configs()
    L = []
    add = L.append
    add("# Resumen: comparación de técnicas de búsqueda espacial\n")
    add(f"_Generado por `benchmarks/benchmark_spatial.py` el {datetime.now():%Y-%m-%d %H:%M}"
        f"{' (modo --quick: NO es la corrida completa)' if args.quick else ''}._\n")

    add("## 1. Configuración del experimento\n")
    add("| Parámetro | Valor |\n|---|---|")
    add(f"| Tamaños N | {', '.join(f'{n:,}' for n in args.sizes)} |")
    add(f"| Radios | {', '.join(str(k) + ' km' for k in RADII_KM)} |")
    add(f"| k-NN | {', '.join(str(k) for k in K_VALUES)} |")
    add(f"| Consultas por configuración | {args.queries} (mismos centros para las 3 técnicas) |")
    add(f"| Warm-up | {args.warmup} consultas no medidas antes de cada celda |")
    add(f"| Repeticiones | {args.repetitions} (cada una reconstruye índices y repite todo) |")
    add(f"| Semilla | {args.seed} |")
    add(f"| Región de datos | lon [{REGION[0]}, {REGION[2]}], lat [{REGION[1]}, {REGION[3]}] "
        "(uniforme, ~108 km x 111 km) |")
    add("| Métrica | Haversine en metros, esfera R = 6 371 008.8 m |")
    add(f"| Tolerancia de equivalencia | {args.tolerance_m} m |")
    add(f"| Presupuesto por celda y repetición | {args.cell_budget_s:.0f} s |")
    add(f"| Python / SO | {platform.python_version()} / {platform.platform()} |")
    add("")
    add("Tabla base común: `HeapFile` con `(id INTEGER, lon DOUBLE, lat DOUBLE)`. "
        "El R-Tree indexa `(lon, lat) -> RID`; las tres técnicas devuelven ids.\n")

    add("## 2. Configuración de PostgreSQL / PostGIS\n")
    if pg_state["available"]:
        add("| Ajuste | Valor |\n|---|---|")
        for k, v in pg_state["env"].items():
            add(f"| {k} | {v} |")
        for k, v in PG_SESSION_SETTINGS.items():
            add(f"| `SET {k}` (sesión del benchmark) | {v} |")
        add(f"| `statement_timeout` | {args.pg_timeout_s:.0f} s |")
        add("")
        add("Esquema: `bench_spatial.bench_points(id integer PK, lon, lat, geog geography(Point,4326))`; "
            f"índice `CREATE INDEX {PG_INDEX_NAME} ... USING GIST (geog)`; `VACUUM ANALYZE` tras la carga. "
            "Rango: `ST_DWithin(geog, punto, metros, false)` (esfera). k-NN: `ORDER BY geog <-> punto LIMIT k`. "
            "El tiempo incluye ida y vuelta al servidor y `fetchall()`.\n")
        gist = [r for r in rows if r["technique"] == "postgis" and r["plan_uses_gist"] is False]
        if gist:
            add("**Atención:** en estas celdas el planificador NO usó el índice GiST "
                "(`EXPLAIN` del primer centro):\n")
            for r in gist:
                add(f"- N={r['n']:,}, {r['query_type']} {r['param']} {r['param_unit']}")
            add("")
    else:
        add(f"PostgreSQL **no se ejecutó**: `{pg_state['reason']}`. Las filas de PostGIS en el CSV "
            f"llevan `status` = `{SKIPPED_NO_POSTGRES}...`. Ver el README para levantar PostGIS con Docker.\n")

    def table(kind, params, unit):
        out = [f"| N | Técnica | " + " | ".join(f"{p} {unit}" for p in params) + " |",
               "|---|---|" + "---|" * len(params)]
        for n in ns:
            for tech in TECHNIQUES:
                cells = []
                for p in params:
                    r = next((x for x in rows if x["n"] == n and x["technique"] == tech
                              and x["query_type"] == kind and x["param"] == p), None)
                    cells.append(_f(r["avg_query_ms"], 3) if r and _num(r["avg_query_ms"])
                                 else (r["status"].split(":")[0][:24] if r else NA))
                out.append(f"| {n:,} | {TECHNIQUES[tech][0]} | " + " | ".join(cells) + " |")
        return out

    add("## 3. Tiempo promedio de consulta (ms)\n")
    add("### Búsqueda por radio\n")
    L.extend(table("range", RADII_KM, "km"))
    add("\n### k-NN\n")
    L.extend(table("knn", K_VALUES, "k"))

    add("\n## 4. Construcción, disco y memoria\n")
    add("| N | Técnica | Build índice (s) | Carga tabla base (s) | Índice en disco (KB) | "
        "Tabla en disco (KB) | Pico mem. build (KB) | Pico mem. consulta k-NN (KB) |")
    add("|---|---|---|---|---|---|---|---|")
    for n in ns:
        for tech in TECHNIQUES:
            r = next((x for x in rows if x["n"] == n and x["technique"] == tech
                      and x["query_type"] == "knn" and x["param"] == K_VALUES[-1]), None)
            if r is None:
                continue
            kb = lambda v: _f(v / 1024, 1) if _num(v) else NA
            add(f"| {n:,} | {TECHNIQUES[tech][0]} | {_f(r['build_time_s'], 3)} | "
                f"{_f(r['base_load_time_s'], 3)} | {kb(r['index_disk_bytes'])} | "
                f"{kb(r['table_disk_bytes'])} | {kb(r['mem_build_peak_bytes'])} | "
                f"{kb(r['mem_query_peak_bytes'])} |")
    add("\nNotas: `build` del secuencial es 0 (no hay índice). La memoria de Python se mide con "
        "`tracemalloc` en una pasada aparte; PostgreSQL corre en otro proceso y su memoria "
        "(`mem_server_backend_bytes` en el CSV, si PG >= 14) no es comparable con la de Python.\n")

    add("## 5. Equivalencia de resultados (referencia: búsqueda secuencial)\n")
    add("| N | Técnica | Configuración | Comparadas | Exactas | Con tolerancia | Diferentes | Exactitud |")
    add("|---|---|---|---|---|---|---|---|")
    for n in ns:
        for tech in ("rtree", "postgis"):
            for c in cfgs:
                r = next((x for x in rows if x["n"] == n and x["technique"] == tech
                          and x["query_type"] == c.kind and x["param"] == c.param), None)
                if r is None or not _num(r["compared_queries"]):
                    continue
                acc = _f(r["accuracy_pct"], 1) + " %" if _num(r["accuracy_pct"]) else NA
                add(f"| {n:,} | {TECHNIQUES[tech][0]} | {c.kind} {c.param} {c.unit} | "
                    f"{r['compared_queries']} | {r['exact_matches']} | {r['tolerance_matches']} | "
                    f"{r['mismatches']} | {acc} |")
    add(f"\n*Con tolerancia*: range = solo difieren puntos a menos de {args.tolerance_m} m del borde del radio; "
        "k-NN = mismas distancias posición a posición (empates / cuasi-empates con ids distintos). "
        "*Diferentes* es un resultado incorrecto y debe investigarse.\n")

    add("## 6. Errores y consultas descartadas\n")
    bad = [r for r in rows if r["queries_discarded"] or r["query_errors"]
           or (r["status"] not in ("OK",))]
    if not bad:
        add("Ninguna: todas las consultas se ejecutaron sin error y dentro del presupuesto.\n")
    else:
        add("| N | Técnica | Configuración | Estado | Descartadas | Errores | Detalle |")
        add("|---|---|---|---|---|---|---|")
        seen = set()
        for r in bad:
            key = (r["n"], r["technique"], r["status"]) if r["status"] != "OK" else None
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            conf = "todas" if key else f"{r['query_type']} {r['param']} {r['param_unit']}"
            add(f"| {r['n']:,} | {TECHNIQUES[r['technique']][0]} | {conf} | {r['status'][:80]} | "
                f"{r['queries_discarded']} | {r['query_errors']} | {r['error_samples'][:120]} |")
        add("")

    add("## 7. Alcance y limitaciones\n")
    add("- Los datos son sintéticos y uniformes; con datos agrupados (ciudades) los resultados cambiarían.")
    add("- Los índices propios usan un buffer pool dimensionado para caber en memoria (mismo criterio que "
        "`benchmark_indexes.py`): es un escenario de caché caliente.")
    add("- El R-Tree se construye con inserciones una a una (no hay bulk-load); PostgreSQL construye el "
        "GiST en una sola operación, por lo que el tiempo de build no es estrictamente análogo.")
    add("- El k-NN del R-Tree usa radio expansivo (1 km, x2, ...) y repite la consulta en cada ampliación; "
        "PostGIS usa búsqueda ordenada nativa del GiST (`<->`).")
    add("- Los tiempos de PostgreSQL incluyen red local y deserialización del cliente; los de las técnicas "
        "Python incluyen su propio overhead de intérprete.")
    add("- El `timeout` por consulta solo puede interrumpir a PostgreSQL; en Python se aplica el "
        "presupuesto por celda (las consultas restantes se descartan y se cuentan).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Resumen escrito en {path}")


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark espacial: secuencial vs R-Tree vs PostGIS")
    p.add_argument("--quick", action="store_true",
                   help="corrida corta (N=500/2000, 10 consultas, 1 repetición); no sobreescribe resultados completos")
    p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=None,
                   help="lista de N separada por comas (por defecto 1000,10000,100000)")
    p.add_argument("--queries", type=int, default=None, help=f"consultas por configuración ({N_QUERIES})")
    p.add_argument("--warmup", type=int, default=None, help=f"consultas de warm-up por celda ({WARMUP_QUERIES})")
    p.add_argument("--repetitions", type=int, default=None, help=f"repeticiones ({REPETITIONS})")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tolerance-m", type=float, default=TOLERANCE_M, help="tolerancia de equivalencia en metros")
    p.add_argument("--cell-budget-s", type=float, default=CELL_BUDGET_S)
    p.add_argument("--pg-dsn", default=os.environ.get("PG_DSN"),
                   help="DSN de PostgreSQL (por defecto, variable de entorno PG_DSN)")
    p.add_argument("--pg-timeout-s", type=float, default=PG_QUERY_TIMEOUT_S)
    p.add_argument("--no-postgres", action="store_true", help="omite PostgreSQL + PostGIS")
    p.add_argument("--keep-pg-data", action="store_true", help="no borra el esquema bench_spatial al terminar")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()
    if args.quick:
        args.sizes = args.sizes or [500, 2_000]
        args.queries = args.queries or 10
        args.warmup = args.warmup if args.warmup is not None else 2
        args.repetitions = args.repetitions or 1
    else:
        args.sizes = args.sizes or N_VALUES
        args.queries = args.queries or N_QUERIES
        args.warmup = args.warmup if args.warmup is not None else WARMUP_QUERIES
        args.repetitions = args.repetitions or REPETITIONS
    return args


def main():
    args = parse_args()
    rows, pg_state = run_all(args)
    out = args.output_dir
    suffix = "_quick" if args.quick else ""
    write_csv(rows, out / f"spatial_results{suffix}.csv")
    make_charts(rows, out / ("charts_quick" if args.quick else "charts"))
    write_summary(rows, pg_state, args, out / f"spatial_summary{suffix}.md")


if __name__ == "__main__":
    main()
