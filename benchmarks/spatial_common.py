"""
benchmarks/spatial_common.py — Dataset, carga de trabajo y formato de
resultados compartidos por la comparación espacial R-tree vs. PostGIS GiST.

Ambos lados (benchmarks/postgres/load_data.py y el benchmark espacial del
DBMS propio) DEBEN importar este módulo para garantizar que:
    - indexan exactamente los mismos puntos (misma semilla),
    - ejecutan exactamente las mismas consultas,
    - escriben sus resultados en el mismo CSV con las mismas columnas.

Coordenadas: x = longitud, y = latitud (grados), dentro de los rangos que
acepta spatial.geometry.Point2D. La distancia es euclidiana sobre esas
coordenadas (EUCLIDEAN en el DBMS propio, SRID 0 en PostGIS).
"""

import csv
import random
from pathlib import Path

N_VALUES = [1_000, 10_000, 100_000]
SEED = 42
REPETITIONS = 3

# Región de los puntos (≈ Lima Metropolitana): min_x, min_y, max_x, max_y.
BBOX = (-77.20, -12.30, -76.80, -11.80)

QUERIES_PER_TYPE = 20
RADIUS = 0.005        # radio de las consultas por rango, en grados
KNN_K = 10            # vecinos de las consultas k-NN
RECT_SIDE = 0.02      # lado de los polígonos (rectángulos) de intersección

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "spatial_comparison_results.csv"

# Una fila por (engine, n, query_type). Columnas vacías = no aplica al motor.
FIELDNAMES = [
    "engine",                 # "minidbms" | "postgis"
    "n",
    "query_type",             # "radius" | "knn" | "intersects"
    "param",                  # "r=0.005" | "k=10" | "side=0.02"
    "num_queries",
    "repetitions",
    "avg_time_ms",            # tiempo cliente (ejecución + fetch) por consulta
    "min_time_ms",
    "max_time_ms",
    "avg_server_time_ms",     # solo PostGIS: "Execution Time" de EXPLAIN ANALYZE
    "total_rows_returned",    # suma sobre las consultas: debe coincidir entre motores
    "index_build_time_s",
    "index_space_bytes",
    "used_index",             # True si todas las consultas usaron el índice espacial
]


# =============================================================================
# Dataset y carga de trabajo
# =============================================================================

def generate_points(n: int, seed: int = SEED) -> list[tuple[int, str, float, float]]:
    """Devuelve n filas (id, name, x, y) uniformes dentro de BBOX."""
    rng = random.Random(seed)
    min_x, min_y, max_x, max_y = BBOX
    return [
        (i, f"place_{i}", rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        for i in range(1, n + 1)
    ]


def generate_queries(seed: int = SEED + 1) -> dict[str, list[dict]]:
    """Consultas por tipo; independientes de n para comparar entre tamaños.

    - radius:     {"x", "y", "r"}
    - knn:        {"x", "y", "k"}
    - intersects: {"polygon": [(x, y), ...]}  (vértices sin repetir el primero)
    """
    rng = random.Random(seed)
    min_x, min_y, max_x, max_y = BBOX

    def random_point():
        return rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)

    radius = []
    for _ in range(QUERIES_PER_TYPE):
        x, y = random_point()
        radius.append({"x": x, "y": y, "r": RADIUS})

    knn = []
    for _ in range(QUERIES_PER_TYPE):
        x, y = random_point()
        knn.append({"x": x, "y": y, "k": KNN_K})

    intersects = []
    for _ in range(QUERIES_PER_TYPE):
        x0 = rng.uniform(min_x, max_x - RECT_SIDE)
        y0 = rng.uniform(min_y, max_y - RECT_SIDE)
        x1, y1 = x0 + RECT_SIDE, y0 + RECT_SIDE
        intersects.append({"polygon": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]})

    return {"radius": radius, "knn": knn, "intersects": intersects}


QUERY_PARAMS = {
    "radius": f"r={RADIUS}",
    "knn": f"k={KNN_K}",
    "intersects": f"side={RECT_SIDE}",
}


# =============================================================================
# Resultados
# =============================================================================

def write_results(rows: list[dict], engine: str, path: Path = RESULTS_CSV) -> Path:
    """Reemplaza las filas de `engine` en el CSV común, conservando las del otro motor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if path.exists():
        with open(path, newline="") as f:
            kept = [row for row in csv.DictReader(f) if row.get("engine") != engine]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    return path
