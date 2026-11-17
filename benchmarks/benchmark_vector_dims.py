"""
benchmarks/benchmark_vector_dims.py — Compara IVF y HNSW según la dimensión de los
vectores (issue #108), con los vectores de prepare_vector_data.py --dims.

Complementa benchmark_vector.py (#85), que fija la dimensión en 64 y compara métricas.
Aquí la métrica es siempre coseno sobre vectores normalizados, que es lo que usa el
motor, y lo que cambia es el tamaño del codebook. Los objetos y las consultas son los
mismos en todas las dimensiones.

Qué se compara, en cada dimensión:
    - exact: fuerza bruta con numpy; es la verdad del recall.
    - hnsw: HNSWIndex con M=16, ef_construction=200 y un barrido de ef.
    - ivf_numpy: las listas de IVFIndex (n_lists = raíz de N) con distancias
      vectorizadas y un barrido de n_probe.
    - ivf: IVFIndex tal cual, que calcula en Python, solo con n_probe=4 (el del motor).
      Barrerlo entero multiplicaría el tiempo, y #85 ya mostró cómo crece con n_probe.

Medidas:
    - Construcción con un hilo, latencia y memoria pico, con cada fase en un proceso
      nuevo, como en #85.
    - Recall@10 contra la fuerza bruta de la MISMA dimensión, y Precision@10 por clase,
      que sí se puede comparar entre dimensiones.
    - Guardado y carga: tamaño en disco, tiempo de save y de load, y si el índice
      cargado devuelve exactamente los mismos vecinos y distancias que el original.
    - avg_nonzero: palabras distintas por vector. Con codebooks grandes y pocos
      descriptores por objeto, los histogramas quedan dispersos.

Es seguro interrumpirlo: las filas se guardan al terminar cada índice, y al volver a
ejecutarlo se saltan los ya medidos (salvo con --force).

Uso:
    python benchmarks/benchmark_vector_dims.py --quick              # N = 1,000
    python benchmarks/benchmark_vector_dims.py --sizes 1000,10000
    python benchmarks/benchmark_vector_dims.py --dims 64,512 --modality image
    python benchmarks/benchmark_vector_dims.py --sizes 10000 --cpu 2   # tiempos comparables

En CPUs híbridas conviene --cpu con un núcleo rápido (P): sin fijarlo, cada fase
puede caer en un núcleo de distinta velocidad y los tiempos no son comparables.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Un hilo, como en #85. Antes de importar numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_vector import (
    ENGINE_HNSW_EF,
    ENGINE_IVF_PROBE,
    HNSW_EF_CONSTRUCTION,
    HNSW_EFS,
    HNSW_M,
    IVF_PROBES,
    REPETITIONS,
    TOP_K,
    VectorizedIVF,
    _index_size,
    _new_index,
    _timed_search,
    exact_baseline,
    in_fresh_process,
    ivf_lists_for,
    peak_rss_bytes,
    quality,
    read_rows,
    rss_bytes,
    time_summary,
    write_environment,
)
from vector_common import DATASETS, DIMENSIONS, MODALITIES, QUICK_N_VALUES, RESULTS_DIR, features_path, load_features

METRIC = "cosine"
VARIANT = "normalized"
DEFAULT_SIZES = (1_000, 10_000)
PYTHON_IVF_PROBES = (ENGINE_IVF_PROBE,)

RESULTS_CSV = RESULTS_DIR / "vector_dims_results.csv"
ENVIRONMENT_JSON = RESULTS_DIR / "vector_dims_environment.json"

FIELDNAMES = [
    "modality", "dataset", "dim", "n", "variant", "metric",
    "index_kind",                   # exact | ivf | ivf_numpy | hnsw
    "is_engine_default",
    "n_lists", "n_probe", "m", "ef_construction", "ef_search",
    "k", "num_queries", "repetitions",
    "avg_time_ms", "median_time_ms", "p95_time_ms", "min_time_ms", "max_time_ms",
    "recall_at_k", "precision_at_k",
    "avg_candidates",               # solo IVF: vectores comparados por consulta
    "avg_nonzero",                  # palabras distintas por vector de la base
    "index_build_time_s", "index_space_bytes",
    "save_time_s", "load_time_s", "save_load_identical",
    "build_peak_rss_delta_bytes", "query_peak_rss_delta_bytes",
]
KEY_FIELDS = ("modality", "dim", "n", "index_kind", "n_probe", "ef_search")


# =============================================================================
# Fases: cada una corre en un proceso nuevo, para que su pico de RSS empiece limpio
# =============================================================================

def _neighbors(index, queries: np.ndarray) -> list[list[tuple]]:
    return [[(r.rid, r.distance) for r in index.search(query, k=TOP_K)] for query in queries]


def same_neighbors(expected: list[list[tuple]], found: list[list[tuple]]) -> bool:
    """Mismos RID en el mismo orden y las mismas distancias (salvo redondeo float32)."""
    if len(expected) != len(found):
        return False
    for before, after in zip(expected, found):
        if [rid for rid, _ in before] != [rid for rid, _ in after]:
            return False
        if not np.allclose([d for _, d in before], [d for _, d in after], rtol=1e-5, atol=1e-6):
            return False
    return True


def build_phase(kind: str, n_lists: int, vectors_path: str, queries_path: str, index_path: str) -> dict:
    """Construye, guarda, carga y comprueba que el índice cargado responda igual."""
    from index.vector import HNSWIndex, IVFIndex

    vectors = np.load(vectors_path)
    queries = np.load(queries_path)
    before = rss_bytes()  # antes de crear el índice: HNSW ya reserva su grafo al crearse
    index = _new_index(kind, METRIC, len(vectors), n_lists, dim=vectors.shape[1])
    start = time.perf_counter()
    index.build(vectors, list(range(len(vectors))))
    build_time = time.perf_counter() - start
    peak = peak_rss_bytes()

    expected = _neighbors(index, queries)
    start = time.perf_counter()
    index.save(index_path)
    save_time = time.perf_counter() - start
    del index
    gc.collect()

    start = time.perf_counter()
    loaded = (IVFIndex if kind == "ivf" else HNSWIndex).load(index_path)
    load_time = time.perf_counter() - start
    return {
        "index_build_time_s": round(build_time, 6),
        "index_space_bytes": _index_size(index_path),
        "save_time_s": round(save_time, 6),
        "load_time_s": round(load_time, 6),
        "save_load_identical": same_neighbors(expected, _neighbors(loaded, queries)),
        "build_peak_rss_delta_bytes": peak - before,
    }


def query_phase(kind: str, index_path: str, queries_path: str, values: list[int],
                python_values: list[int], repetitions: int) -> dict:
    """Barre ef (HNSW) o n_probe (IVF) sobre el índice cargado de disco. En IVF, el
    índice en Python solo se mide con python_values y el control con todos."""
    from index.vector import HNSWIndex, IVFIndex

    queries = np.load(queries_path)
    before = rss_bytes()
    index = (IVFIndex if kind == "ivf" else HNSWIndex).load(index_path)

    runs = {}
    for value in (values if kind == "hnsw" else python_values):
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
# Filas
# =============================================================================

def ivf_probes_for(n: int) -> tuple[int, ...]:
    lists = ivf_lists_for(n)
    return tuple(probe for probe in IVF_PROBES if probe <= lists)


def index_rows(kind: str, n_lists, built: dict, measured: dict, common: dict,
               truth: dict, labels: np.ndarray, query_labels: np.ndarray) -> list[dict]:
    empty = {field: "" for field in FIELDNAMES}
    shared = {**empty, **common, **built, "query_peak_rss_delta_bytes": measured["query_peak_rss_delta_bytes"]}
    rows = []
    if kind == "hnsw":
        for ef, run in measured["runs"].items():
            rows.append({
                **shared, "index_kind": "hnsw", "is_engine_default": ef == ENGINE_HNSW_EF,
                "m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": ef,
                **time_summary(run["times"]), **quality(run["ids"], truth, labels, query_labels),
            })
        return rows
    for probe, control in measured["control"].items():
        ivf_shared = {**shared, "n_lists": n_lists, "n_probe": probe,
                      "avg_candidates": round(control["avg_candidates"], 3)}
        if probe in measured["runs"]:
            run = measured["runs"][probe]
            rows.append({
                **ivf_shared, "index_kind": "ivf", "is_engine_default": False,
                **time_summary(run["times"]), **quality(run["ids"], truth, labels, query_labels),
            })
        rows.append({
            **ivf_shared, "index_kind": "ivf_numpy", "is_engine_default": False,
            **time_summary(control["times"]), **quality(control["ids"], truth, labels, query_labels),
        })
    return rows


def row_key(row: dict) -> tuple:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def write_rows(new_rows: list[dict], path: Path) -> None:
    """Reemplaza las filas de las mismas claves y conserva las demás."""
    keys = {row_key(row) for row in new_rows}
    rows = [row for row in read_rows(path) if row_key(row) not in keys] + new_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def measured_groups(path: Path) -> set[tuple]:
    """(modalidad, dim, n, índice) ya medidos, para no repetirlos."""
    return {(row["modality"], row["dim"], row["n"], row["index_kind"]) for row in read_rows(path)}


# =============================================================================
# Experimento
# =============================================================================

def run_dimension(features, modality: str, dim: int, n: int, out_path: Path,
                  repetitions: int, force: bool) -> None:
    base = features.base_vectors(n, VARIANT)
    queries = features.query_vectors(VARIANT)
    labels = features.base_labels[:n]
    common = {
        "modality": modality, "dataset": DATASETS[modality].name, "dim": dim, "n": n,
        "variant": VARIANT, "metric": METRIC, "k": TOP_K,
        "num_queries": len(queries), "repetitions": repetitions,
        "avg_nonzero": round(float(np.count_nonzero(base, axis=1).mean()), 3),
    }
    done = set() if force else measured_groups(out_path)
    group = (modality, str(dim), str(n))

    truth, times, ids = exact_baseline(base, queries, METRIC, repetitions)
    write_rows([{
        **{field: "" for field in FIELDNAMES}, **common, "index_kind": "exact", "is_engine_default": False,
        **time_summary(times), **quality(ids, truth, labels, features.query_labels),
    }], out_path)

    plans = [("hnsw", "", list(HNSW_EFS)), ("ivf", ivf_lists_for(n), list(ivf_probes_for(n)))]
    with tempfile.TemporaryDirectory(prefix="vector-dims-", ignore_cleanup_errors=True) as workdir:
        vectors_path = os.path.join(workdir, "base.npy")
        queries_path = os.path.join(workdir, "queries.npy")
        np.save(vectors_path, base)
        np.save(queries_path, queries)
        for kind, n_lists, values in plans:
            if (*group, "ivf_numpy" if kind == "ivf" else kind) in done:
                continue
            started = time.perf_counter()
            index_path = os.path.join(workdir, f"{kind}.idx")
            built = in_fresh_process(build_phase, kind, n_lists or 0, vectors_path, queries_path, index_path)
            measured = in_fresh_process(
                query_phase, kind, index_path, queries_path, values, list(PYTHON_IVF_PROBES), repetitions,
            )
            write_rows(index_rows(kind, n_lists, built, measured, common, truth, labels, features.query_labels),
                       out_path)
            print(
                f"  {modality} k={dim} n={n} {kind}: {time.perf_counter() - started:.1f} s "
                f"(construcción {built['index_build_time_s']:.2f} s, "
                f"guardar/cargar idéntico: {built['save_load_identical']})",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="N = 1,000 con los vectores de --quick; escribe *_quick.*")
    parser.add_argument("--sizes", default=None, help="tamaños separados por coma, p. ej. 1000,10000")
    parser.add_argument("--dims", default=",".join(map(str, DIMENSIONS)))
    parser.add_argument("--modality", choices=[*MODALITIES, "all"], default="all")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--force", action="store_true", help="repite lo que ya está en el CSV")
    parser.add_argument("--cpu", type=int, default=None,
                        help="fija el benchmark a este núcleo lógico (las fases lo heredan)")
    args = parser.parse_args()

    if args.cpu is not None:
        # En CPUs híbridas (núcleos P y E) cada proceso nuevo puede caer en un núcleo
        # de distinta velocidad; fijarlo hace comparables los tiempos entre fases.
        import psutil

        psutil.Process().cpu_affinity([args.cpu])

    sizes =[int(size) for size in args.sizes.split(",")] if args.sizes else (
        QUICK_N_VALUES if args.quick else list(DEFAULT_SIZES)
    )
    dims = [int(dim) for dim in args.dims.split(",")]
    modalities = MODALITIES if args.modality == "all" else [args.modality]
    missing = [features_path(m, args.quick, d) for m in modalities for d in dims
               if not features_path(m, args.quick, d).exists()]
    if missing:
        quick = " --quick" if args.quick else ""
        raise SystemExit(
            "Faltan vectores:\n  " + "\n  ".join(map(str, missing)) +
            f"\nPrepáralos con: python benchmarks/prepare_vector_data.py{quick} --dims {args.dims} --n {max(sizes)}"
        )

    suffix = "_quick" if args.quick else ""
    out_path = RESULTS_CSV.with_name(f"{RESULTS_CSV.stem}{suffix}{RESULTS_CSV.suffix}")
    env_path = ENVIRONMENT_JSON.with_name(f"{ENVIRONMENT_JSON.stem}{suffix}{ENVIRONMENT_JSON.suffix}")
    write_environment(env_path, sizes)
    environment = json.loads(env_path.read_text(encoding="utf-8"))
    environment.update(dims=dims, cpu_affinity=args.cpu)
    env_path.write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    started = time.perf_counter()
    for modality in modalities:
        for dim in dims:
            features = load_features(modality, quick=args.quick, dim=dim)
            for n in sizes:
                run_dimension(features, modality, dim, n, out_path, args.repetitions, args.force)
    print(f"\nTotal: {time.perf_counter() - started:.1f} s\nResultados: {out_path}\nEntorno: {env_path}", flush=True)


if __name__ == "__main__":
    main()
