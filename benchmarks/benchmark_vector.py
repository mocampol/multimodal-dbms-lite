"""
benchmarks/benchmark_vector.py — Compara IVF y HNSW con coseno y euclidiana sobre
imágenes y audios (issue #85), usando los vectores de prepare_vector_data.py.

Mide IVFIndex y HNSWIndex directamente y no por SQL: VectorSearch vuelve a extraer
features del corpus en cada consulta y el catálogo crea los índices siempre en
coseno, así que por SQL se mediría sobre todo la extracción.

Qué se compara:
    - exact: fuerza bruta con numpy; línea base y verdad exacta del recall.
    - ivf: IVFIndex con n_lists = raíz de N y un barrido de n_probe, y con los
      valores por defecto del motor (n_lists=16, n_probe=4).
    - hnsw: HNSWIndex (hnswlib, en C++) con M=16, ef_construction=200 y un barrido
      de ef; ef=50 es el valor por defecto del motor.
    - ivf_numpy: control con las mismas listas y n_probe que ivf, pero con las
      distancias vectorizadas. IVFIndex calcula en Python y hnswlib en C++; este
      control indica cuánto de esa diferencia se debe al lenguaje.

Medidas:
    - Tiempo de construcción y de consulta con un hilo. HNSW reporta además la
      construcción con todos los núcleos, que es como la usa el motor.
    - Recall@10 contra la fuerza bruta de la MISMA métrica. Un resultado cuenta si
      su distancia real no supera la del décimo vecino exacto, porque los objetos
      con histograma idéntico empatan.
    - Precision@10: vecinos de la misma clase que la consulta. Es lo único que
      compara métricas entre sí, porque cada recall se mide contra su propia métrica.
    - Memoria: pico de RSS menos RSS inicial, con cada fase en un proceso nuevo
      (Windows: peak_wset; Linux: VmHWM). En consultas incluye cargar el índice.
    - Disco: tamaño de los archivos del índice guardado.

Es seguro interrumpirlo: las filas se guardan al terminar cada índice, y al volver a
ejecutarlo se saltan los ya medidos (salvo con --force).

Uso:
    python benchmarks/benchmark_vector.py --quick                # solo N = 1,000
    python benchmarks/benchmark_vector.py                        # N = 1k, 10k y 100k
    python benchmarks/benchmark_vector.py --sizes 10000 --modality image
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Un hilo: IVF (k-means) y HNSW se comparan en igualdad. Antes de importar numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vector_common import (
    DATASETS,
    FEATURE_DIM,
    MODALITIES,
    N_VALUES,
    QUICK_N_VALUES,
    RESULTS_DIR,
    VARIANTS,
    load_features,
)

METRICS = ("cosine", "euclidean")
TOP_K = 10
WARMUP_QUERIES = 10
REPETITIONS = 3

IVF_PROBES = (1, 2, 4, 8, 16, 32, 64)
ENGINE_IVF_LISTS = 16
ENGINE_IVF_PROBE = 4
HNSW_EFS = (10, 20, 50, 100, 200, 400)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
ENGINE_HNSW_EF = 50

RESULTS_CSV = RESULTS_DIR / "vector_comparison_results.csv"
ENVIRONMENT_JSON = RESULTS_DIR / "vector_comparison_environment.json"

FIELDNAMES = [
    "modality", "dataset", "n", "variant", "metric",
    "index_kind",                   # exact | ivf | ivf_numpy | hnsw
    "config",                       # exact | sqrt | engine | sweep
    "is_engine_default",            # True si son los parámetros que usa el motor
    "n_lists", "n_probe", "m", "ef_construction", "ef_search",
    "k", "num_queries", "repetitions",
    "avg_time_ms", "median_time_ms", "p95_time_ms", "min_time_ms", "max_time_ms",
    "recall_at_k", "precision_at_k",
    "avg_candidates",               # solo IVF: vectores comparados por consulta
    "index_build_time_s",
    "build_time_all_cores_s",       # solo HNSW
    "index_space_bytes",
    "build_peak_rss_delta_bytes",
    "query_peak_rss_delta_bytes",
]
KEY_FIELDS = ("modality", "n", "variant", "metric", "index_kind", "config")


# =============================================================================
# Distancias y métricas de calidad
# =============================================================================

def distances_to(base: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    """Distancia de `query` a cada fila de `base`, igual que compute_distance."""
    if metric == "euclidean":
        return np.linalg.norm(base - query, axis=1)
    if metric == "cosine":
        norms = np.linalg.norm(base, axis=1) * np.linalg.norm(query)
        valid = norms >= 1e-12
        similarity = np.divide(base @ query, norms, out=np.zeros(len(base), dtype=np.float32), where=valid)
        return np.where(valid, 1.0 - np.clip(similarity, -1.0, 1.0), 1.0)
    raise ValueError(f"Métrica no soportada: {metric}")


def top_k_ids(distances: np.ndarray, k: int) -> np.ndarray:
    """Los k más cercanos, por distancia y luego por id, para que sea determinista."""
    k = min(k, len(distances))
    candidates = np.argpartition(distances, k - 1)[:k]
    # Los empatados con el último también compiten, para desempatar por id.
    tied = np.flatnonzero(distances <= distances[candidates].max())
    return tied[np.lexsort((tied, distances[tied]))][:k]


def recall_at_k(returned_ids, query_distances: np.ndarray, kth_distance: float, k: int = TOP_K) -> float:
    """Fracción de k que está a la distancia del k-ésimo vecino exacto o menos."""
    ids = np.asarray(returned_ids, dtype=np.int64)
    if ids.size == 0:
        return 0.0
    tolerance = max(abs(kth_distance) * 1e-5, 1e-6)
    return float(np.count_nonzero(query_distances[ids] <= kth_distance + tolerance)) / k


def precision_at_k(returned_ids, base_labels: np.ndarray, query_label: str, k: int = TOP_K) -> float:
    ids = np.asarray(returned_ids, dtype=np.int64)
    if ids.size == 0:
        return 0.0
    return float(np.count_nonzero(base_labels[ids] == query_label)) / k


def time_summary(times_s: np.ndarray) -> dict:
    """Promedia las repeticiones de cada consulta y resume sobre las consultas."""
    per_query_ms = np.asarray(times_s).mean(axis=0) * 1000
    return {
        "avg_time_ms": round(float(per_query_ms.mean()), 6),
        "median_time_ms": round(float(np.median(per_query_ms)), 6),
        "p95_time_ms": round(float(np.percentile(per_query_ms, 95)), 6),
        "min_time_ms": round(float(per_query_ms.min()), 6),
        "max_time_ms": round(float(per_query_ms.max()), 6),
    }


# =============================================================================
# Control: IVF con distancias vectorizadas
# =============================================================================

class VectorizedIVF:
    """Busca sobre las listas de un IVFIndex construido, sin bucles de Python."""

    def __init__(self, index) -> None:
        self.centroids = np.asarray(index.centroids, dtype=np.float32)
        self.metric = index.metric.value
        self.lists = {}
        for cluster_id, items in index.inverted_lists.items():
            if items:
                self.lists[cluster_id] = (
                    np.array([rid for rid, _ in items], dtype=np.int64),
                    np.stack([vector for _, vector in items]),
                )

    def search(self, query: np.ndarray, n_probe: int, k: int = TOP_K) -> tuple[np.ndarray, int]:
        """(ids del top-k, vectores comparados)."""
        order = np.argsort(distances_to(self.centroids, query, self.metric))[:n_probe]
        probed = [self.lists[int(c)] for c in order if int(c) in self.lists]
        if not probed:
            return np.empty(0, dtype=np.int64), 0
        ids = np.concatenate([rids for rids, _ in probed])
        vectors = np.concatenate([stored for _, stored in probed])
        return ids[top_k_ids(distances_to(vectors, query, self.metric), k)], len(ids)


# =============================================================================
# Fases: cada una corre en un proceso nuevo, para que su pico de RSS empiece limpio
# =============================================================================

def rss_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    """Pico de RSS del proceso desde que empezó (high-water mark del SO)."""
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):          # Windows
        return info.peak_wset
    with open("/proc/self/status", encoding="ascii") as handle:  # Linux
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("No se puede leer el pico de RSS en este sistema")


def in_fresh_process(function, *args):
    with multiprocessing.get_context("spawn").Pool(processes=1) as pool:
        return pool.apply(function, args)


def _new_index(kind: str, metric: str, n: int, n_lists: int, num_threads: int = 1):
    from index.vector import HNSWIndex, IVFIndex

    if kind == "ivf":
        return IVFIndex(dim=FEATURE_DIM, metric=metric, n_lists=n_lists)
    if kind == "hnsw":
        return HNSWIndex(
            dim=FEATURE_DIM, metric=metric, max_elements=n,
            ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M, num_threads=num_threads,
        )
    raise ValueError(f"Índice desconocido: {kind}")


def _index_size(path: str) -> int:
    """El archivo del índice y, en HNSW, también el grafo en <ruta>.bin."""
    binary = path + ".bin"
    return os.path.getsize(path) + (os.path.getsize(binary) if os.path.exists(binary) else 0)


def build_phase(kind: str, metric: str, n_lists: int, vectors_path: str, index_path: str) -> dict:
    from index import vector  # noqa: F401  (las importaciones no cuentan en el pico)

    vectors = np.load(vectors_path)
    rids = list(range(len(vectors)))
    before = rss_bytes()  # antes de crear el índice: HNSW ya reserva su grafo al crearse
    index = _new_index(kind, metric, len(vectors), n_lists)
    start = time.perf_counter()
    index.build(vectors, rids)
    build_time = time.perf_counter() - start
    peak = peak_rss_bytes()
    index.save(index_path)
    result = {
        "index_build_time_s": round(build_time, 6),
        "index_space_bytes": _index_size(index_path),
        "build_peak_rss_delta_bytes": peak - before,
        "build_time_all_cores_s": "",
    }
    if kind == "hnsw":
        del index
        parallel = _new_index(kind, metric, len(vectors), n_lists, num_threads=-1)
        start = time.perf_counter()
        parallel.build(vectors, rids)
        result["build_time_all_cores_s"] = round(time.perf_counter() - start, 6)
    return result


def _timed_search(search, queries: np.ndarray, repetitions: int):
    """Calienta, y devuelve (tiempos [rep, consulta], resultados de la última pasada)."""
    for query in queries[:WARMUP_QUERIES]:
        search(query)
    times = np.zeros((repetitions, len(queries)))
    results = [None] * len(queries)
    # Sin recolector, como timeit: una recolección completa recorre todo lo que
    # tiene el proceso en memoria y caía siempre en el mismo punto del barrido.
    gc.collect()
    gc.disable()
    try:
        for repetition in range(repetitions):
            for position, query in enumerate(queries):
                start = time.perf_counter()
                results[position] = search(query)
                times[repetition, position] = time.perf_counter() - start
    finally:
        gc.enable()
    return times, results


def query_phase(kind: str, index_path: str, queries_path: str, values: list[int], repetitions: int) -> dict:
    """Barre n_probe (IVF) o ef (HNSW) sobre un índice recién cargado de disco."""
    from index.vector import HNSWIndex, IVFIndex

    queries = np.load(queries_path)
    before = rss_bytes()
    index = (IVFIndex if kind == "ivf" else HNSWIndex).load(index_path)

    runs = {}
    for value in values:
        if kind == "ivf":
            index.n_probe = value
        else:
            index.set_ef_search(value)
        times, results = _timed_search(lambda q: index.search(q, k=TOP_K), queries, repetitions)
        runs[value] = {"times": times, "ids": [[r.rid for r in result] for result in results]}
    peak_delta = peak_rss_bytes() - before

    control = {}
    if kind == "ivf":
        vectorized = VectorizedIVF(index)
        for value in values:
            times, results = _timed_search(lambda q, v=value: vectorized.search(q, v), queries, repetitions)
            control[value] = {
                "times": times,
                "ids": [ids.tolist() for ids, _ in results],
                "avg_candidates": float(np.mean([candidates for _, candidates in results])),
            }
    return {"runs": runs, "control": control, "query_peak_rss_delta_bytes": peak_delta}


# =============================================================================
# Experimento
# =============================================================================

@dataclass(frozen=True)
class IndexConfig:
    kind: str
    config: str
    n_lists: int | str
    values: tuple[int, ...]


def ivf_lists_for(n: int) -> int:
    return max(1, round(math.sqrt(n)))


def index_configs(n: int) -> list[IndexConfig]:
    lists = ivf_lists_for(n)
    return [
        IndexConfig("ivf", "sqrt", lists, tuple(p for p in IVF_PROBES if p <= lists)),
        IndexConfig("ivf", "engine", ENGINE_IVF_LISTS, (ENGINE_IVF_PROBE,)),
        IndexConfig("hnsw", "sweep", "", HNSW_EFS),
    ]


def row_key(row: dict) -> tuple:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(new_rows: list[dict], path: Path) -> None:
    """Reemplaza las filas de las mismas claves y conserva las demás."""
    keys = {row_key(row) for row in new_rows}
    rows = [row for row in read_rows(path) if row_key(row) not in keys] + new_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def quality(ids_per_query, truth: dict, labels: np.ndarray, query_labels: np.ndarray) -> dict:
    recalls, precisions = [], []
    for position, ids in enumerate(ids_per_query):
        recalls.append(recall_at_k(ids, truth["distances"][position], truth["kth"][position]))
        precisions.append(precision_at_k(ids, labels, query_labels[position]))
    return {
        "recall_at_k": round(float(np.mean(recalls)), 6),
        "precision_at_k": round(float(np.mean(precisions)), 6),
    }


def exact_baseline(base: np.ndarray, queries: np.ndarray, metric: str, repetitions: int):
    """Verdad exacta por consulta y su tiempo de búsqueda con numpy."""
    def search(query):
        return top_k_ids(distances_to(base, query, metric), TOP_K)

    times, ids = _timed_search(search, queries, repetitions)
    distances = np.stack([distances_to(base, query, metric) for query in queries])
    kth = np.array([distances[position, ids[position][-1]] for position in range(len(queries))])
    return {"distances": distances, "kth": kth}, times, ids


def run_combination(features, modality, n, variant, metric, out_path, repetitions, force, skip) -> None:
    base = features.base_vectors(n, variant)
    queries = features.query_vectors(variant)
    labels = features.base_labels[:n]
    common = {
        "modality": modality, "dataset": DATASETS[modality].name, "n": n,
        "variant": variant, "metric": metric, "k": TOP_K,
        "num_queries": len(queries), "repetitions": repetitions,
    }
    empty = {field: "" for field in FIELDNAMES}

    truth, times, ids = exact_baseline(base, queries, metric, repetitions)
    write_rows([{
        **empty, **common, "index_kind": "exact", "config": "exact", "is_engine_default": False,
        **time_summary(times), **quality(ids, truth, labels, features.query_labels),
    }], out_path)

    with tempfile.TemporaryDirectory(prefix="vector-bench-", ignore_cleanup_errors=True) as workdir:
        vectors_path = os.path.join(workdir, "base.npy")
        queries_path = os.path.join(workdir, "queries.npy")
        np.save(vectors_path, base)
        np.save(queries_path, queries)

        for index_config in index_configs(n):
            key = (modality, str(n), variant, metric, index_config.kind, index_config.config)
            if not force and key in skip:
                continue
            started = time.perf_counter()
            index_path = os.path.join(workdir, f"{index_config.kind}_{index_config.config}.idx")
            built = in_fresh_process(
                build_phase, index_config.kind, metric, index_config.n_lists, vectors_path, index_path,
            )
            measured = in_fresh_process(
                query_phase, index_config.kind, index_path, queries_path,
                list(index_config.values), repetitions,
            )
            write_rows(
                _index_rows(index_config, built, measured, common, empty, truth, labels, features.query_labels),
                out_path,
            )
            print(
                f"  {modality} n={n} {variant} {metric} {index_config.kind}/{index_config.config}: "
                f"{time.perf_counter() - started:.1f} s (construcción {built['index_build_time_s']:.2f} s)",
                flush=True,
            )


def _index_rows(index_config, built, measured, common, empty, truth, labels, query_labels) -> list[dict]:
    rows = []
    is_ivf = index_config.kind == "ivf"
    for value in index_config.values:
        run = measured["runs"][value]
        shared = {
            **empty, **common, **built,
            "config": index_config.config,
            "query_peak_rss_delta_bytes": measured["query_peak_rss_delta_bytes"],
        }
        if is_ivf:
            shared.update(n_lists=index_config.n_lists, n_probe=value)
            control = measured["control"][value]
            shared["avg_candidates"] = round(control["avg_candidates"], 3)
            engine = index_config.config == "engine"
            rows.append({
                **shared, "index_kind": "ivf", "is_engine_default": engine,
                **time_summary(run["times"]), **quality(run["ids"], truth, labels, query_labels),
            })
            rows.append({
                **shared, "index_kind": "ivf_numpy", "is_engine_default": False,
                **time_summary(control["times"]), **quality(control["ids"], truth, labels, query_labels),
            })
        else:
            rows.append({
                **shared, "index_kind": "hnsw", "is_engine_default": value == ENGINE_HNSW_EF,
                "m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": value,
                **time_summary(run["times"]), **quality(run["ids"], truth, labels, query_labels),
            })
    return rows


def write_environment(path: Path, sizes: list[int]) -> None:
    import hnswlib
    import psutil
    import sklearn

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "hnswlib": getattr(hnswlib, "__version__", "desconocida"),
        "sizes": sizes,
        "memory_metric": "peak_wset" if sys.platform == "win32" else "VmHWM",
        "threads": 1,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="solo N = 1,000; escribe *_quick.*")
    parser.add_argument("--sizes", default=None, help="tamaños separados por coma, p. ej. 1000,10000")
    parser.add_argument("--modality", choices=[*MODALITIES, "all"], default="all")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--force", action="store_true", help="repite lo que ya está en el CSV")
    args = parser.parse_args()

    sizes = [int(size) for size in args.sizes.split(",")] if args.sizes else (
        QUICK_N_VALUES if args.quick else N_VALUES
    )
    suffix = "_quick" if args.quick else ""
    out_path = RESULTS_CSV.with_name(f"{RESULTS_CSV.stem}{suffix}{RESULTS_CSV.suffix}")
    env_path = ENVIRONMENT_JSON.with_name(f"{ENVIRONMENT_JSON.stem}{suffix}{ENVIRONMENT_JSON.suffix}")
    write_environment(env_path, sizes)

    started = time.perf_counter()
    for modality in (MODALITIES if args.modality == "all" else [args.modality]):
        features = load_features(modality, quick=args.quick)
        for n in sizes:
            for variant in VARIANTS:
                for metric in METRICS:
                    skip = {row_key(row) for row in read_rows(out_path)}
                    run_combination(features, modality, n, variant, metric, out_path, args.repetitions, args.force, skip)
    print(f"\nTotal: {time.perf_counter() - started:.1f} s\nResultados: {out_path}\nEntorno: {env_path}", flush=True)


if __name__ == "__main__":
    main()
