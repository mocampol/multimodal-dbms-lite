"""
benchmarks/compare_vector_results.py — Verifica que la comparación IVF vs. HNSW y
coseno vs. euclidiana sea válida y resume el CSV común.

Verificaciones:
    - mismo k y mismo número de consultas en todas las filas,
    - recall y precisión entre 0 y 1, y recall 1 en la fuerza bruta,
    - IVFIndex y su control en numpy devuelven lo mismo: solo cambia el tiempo,
    - en IVF, sondear más listas nunca baja el recall,
    - coseno y euclidiana sobre vectores normalizados, y coseno sobre vectores sin
      normalizar, encuentran los mismos vecinos,
    - HNSW construye más rápido con todos los núcleos que con uno (N >= 10,000).

Sale con código 1 si alguna verificación falla.

Uso:
    python benchmarks/compare_vector_results.py            # results/vector_comparison_results.csv
    python benchmarks/compare_vector_results.py --quick    # results/vector_comparison_results_quick.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_vector import METRICS, RESULTS_CSV, TOP_K, index_configs, read_rows
from vector_common import MODALITIES, NUM_QUERIES, VARIANTS

MIN_RECALL = 0.95
MIN_N_FOR_THREADS = 10_000
TOLERANCE = 1e-9
SWEEPS = ("sqrt", "sweep")          # barridos; "engine" es un solo punto con otras listas
SWEEP_KINDS = ("ivf", "ivf_numpy", "hnsw")
MB = 1024 * 1024


def quick_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_quick{path.suffix}")


def number(row: dict, column: str) -> float | None:
    return None if row[column] in ("", None) else float(row[column])


def select(rows: list[dict], **filters) -> list[dict]:
    """Filas cuyas columnas coinciden con los valores indicados."""
    return [row for row in rows if all(row[column] == str(value) for column, value in filters.items())]


def sizes_of(rows: list[dict]) -> list[int]:
    return sorted({int(row["n"]) for row in rows})


def modalities_of(rows: list[dict]) -> list[str]:
    return [modality for modality in MODALITIES if select(rows, modality=modality)]


# =============================================================================
# Verificación
# =============================================================================

def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= TOLERANCE


def _check_ivf_control(rows: list[dict]) -> list[str]:
    errors = []
    pairs = defaultdict(dict)
    for row in rows:
        if row["index_kind"] in ("ivf", "ivf_numpy"):
            key = (row["modality"], row["n"], row["variant"], row["metric"], row["config"], row["n_probe"])
            pairs[key][row["index_kind"]] = row
    for key, pair in pairs.items():
        label = " ".join(key)
        if len(pair) < 2:
            errors.append(f"{label}: falta {'ivf_numpy' if 'ivf' in pair else 'ivf'}")
            continue
        for column in ("recall_at_k", "precision_at_k", "avg_candidates"):
            if not _same(number(pair["ivf"], column), number(pair["ivf_numpy"], column)):
                errors.append(f"{label}: {column} distinto entre ivf e ivf_numpy "
                              f"({pair['ivf'][column]} vs. {pair['ivf_numpy'][column]})")
    return errors


def _check_ivf_recall_grows(rows: list[dict]) -> list[str]:
    errors = []
    sweeps = defaultdict(list)
    for row in select(rows, index_kind="ivf", config="sqrt"):
        sweeps[(row["modality"], row["n"], row["variant"], row["metric"])].append(row)
    for key, sweep in sweeps.items():
        sweep.sort(key=lambda row: int(row["n_probe"]))
        for before, after in zip(sweep, sweep[1:]):
            if number(after, "recall_at_k") < number(before, "recall_at_k") - TOLERANCE:
                errors.append(f"{' '.join(key)}: el recall baja de n_probe={before['n_probe']} "
                              f"a n_probe={after['n_probe']}")
    return errors


def _check_same_neighbors(rows: list[dict]) -> list[str]:
    errors = []
    exact = defaultdict(dict)
    for row in select(rows, index_kind="exact"):
        exact[(row["modality"], row["n"])][(row["variant"], row["metric"])] = number(row, "precision_at_k")
    for (modality, n), values in exact.items():
        reference = values.get(("normalized", "cosine"))
        for other in (("normalized", "euclidean"), ("raw", "cosine")):
            if other in values and not _same(reference, values[other]):
                errors.append(f"{modality} N={n}: precisión exacta de {' '.join(other)} ({values[other]}) "
                              f"distinta de normalized cosine ({reference})")
    return errors


def _check_hnsw_threads(rows: list[dict]) -> list[str]:
    """Una vez un valor mal pasado dejó a HNSW en un hilo sin que nada fallara."""
    errors = []
    seen = set()
    for row in select(rows, index_kind="hnsw"):
        key = (row["modality"], row["n"], row["variant"], row["metric"])
        if key in seen or int(row["n"]) < MIN_N_FOR_THREADS:
            continue
        seen.add(key)
        single, every_core = number(row, "index_build_time_s"), number(row, "build_time_all_cores_s")
        if single is None or every_core is None or every_core >= single:
            errors.append(f"{' '.join(key)}: HNSW no construye más rápido con todos los núcleos "
                          f"({every_core} s vs. {single} s con uno)")
    return errors


def check(rows: list[dict]) -> list[str]:
    errors = []
    for row in rows:
        label = f"{row['modality']} N={row['n']} {row['variant']} {row['metric']} {row['index_kind']}"
        if row["k"] != str(TOP_K):
            errors.append(f"{label}: k={row['k']}, se esperaba {TOP_K}")
        if row["num_queries"] != str(NUM_QUERIES):
            errors.append(f"{label}: {row['num_queries']} consultas, se esperaban {NUM_QUERIES}")
        for column in ("recall_at_k", "precision_at_k"):
            value = number(row, column)
            if value is None or not 0.0 <= value <= 1.0:
                errors.append(f"{label}: {column}={row[column]} fuera de [0, 1]")
        if row["index_kind"] == "exact" and not _same(number(row, "recall_at_k"), 1.0):
            errors.append(f"{label}: la fuerza bruta tiene recall {row['recall_at_k']}")
    return (errors + _check_ivf_control(rows) + _check_ivf_recall_grows(rows)
            + _check_same_neighbors(rows) + _check_hnsw_threads(rows))


def expected_rows_per_combination(n: int) -> int:
    """Fuerza bruta + cada valor del barrido; IVF se mide dos veces (motor y control)."""
    return 1 + sum(len(config.values) * (2 if config.kind == "ivf" else 1) for config in index_configs(n))


def missing_combinations(rows: list[dict]) -> list[str]:
    counts = defaultdict(int)
    for row in rows:
        counts[(row["modality"], int(row["n"]), row["variant"], row["metric"])] += 1
    warnings = []
    for modality in modalities_of(rows):
        for n in sizes_of(rows):
            for variant in VARIANTS:
                for metric in METRICS:
                    found, expected = counts[(modality, n, variant, metric)], expected_rows_per_combination(n)
                    if found != expected:
                        warnings.append(f"{modality} N={n} {variant} {metric}: {found} filas de {expected}")
    return warnings


# =============================================================================
# Consultas sobre los resultados
# =============================================================================

def fastest_at_recall(rows: list[dict], kind: str, min_recall: float = MIN_RECALL, **filters) -> dict | None:
    """La configuración más rápida del barrido de `kind` que llega a `min_recall`."""
    candidates = [row for row in select(rows, index_kind=kind, **filters)
                  if row["config"] in SWEEPS and number(row, "recall_at_k") >= min_recall]
    return min(candidates, key=lambda row: number(row, "avg_time_ms"), default=None)


def parameter(row: dict) -> str:
    if row["index_kind"] == "hnsw":
        return f"ef={row['ef_search']}"
    if row["index_kind"] == "exact":
        return "—"
    return f"listas={row['n_lists']} p={row['n_probe']}"


# =============================================================================
# Resumen en consola
# =============================================================================

def fmt(value: float | None, spec: str, scale: float = 1.0) -> str:
    return "—" if value is None else format(value / scale, spec)


def print_fastest_table(rows: list[dict]) -> None:
    print(f"\nLo más rápido con Recall@{TOP_K} >= {MIN_RECALL} (normalized, cosine)\n")
    print(f"{'modalidad':<9} {'N':>8} {'índice':<10} {'parámetro':<18} {'recall':>7} {'ms':>10}")
    for modality in modalities_of(rows):
        for n in sizes_of(rows):
            common = dict(modality=modality, n=n, variant="normalized", metric="cosine")
            for kind in SWEEP_KINDS:
                row = fastest_at_recall(rows, kind, **common)
                print(f"{modality:<9} {n:>8,} {kind:<10} "
                      + (f"{parameter(row):<18} {row['recall_at_k']:>7} {float(row['avg_time_ms']):>10.3f}"
                         if row else f"{'no llega':<18}"))


def print_defaults_table(rows: list[dict]) -> None:
    print("\nValores por defecto del motor frente a la fuerza bruta (normalized, cosine)\n")
    print(f"{'modalidad':<9} {'N':>8} {'índice':<10} {'parámetro':<18} {'recall':>7} {'P@k':>7} {'ms':>10}")
    for modality in modalities_of(rows):
        for n in sizes_of(rows):
            common = dict(modality=modality, n=n, variant="normalized", metric="cosine")
            chosen = select(rows, index_kind="exact", **common) + select(rows, is_engine_default=True, **common)
            for row in chosen:
                print(f"{modality:<9} {n:>8,} {row['index_kind']:<10} {parameter(row):<18} "
                      f"{float(row['recall_at_k']):>7.3f} {float(row['precision_at_k']):>7.4f} "
                      f"{float(row['avg_time_ms']):>10.3f}")


def print_cost_table(rows: list[dict]) -> None:
    print("\nConstrucción, disco y memoria (normalized, cosine; pico de RSS - RSS inicial de cada fase)\n")
    print(f"{'modalidad':<9} {'N':>8} {'índice':<12} {'build s':>9} {'núcleos s':>10} "
          f"{'disco MB':>9} {'mem build MB':>13} {'mem consulta MB':>16}")
    for modality in modalities_of(rows):
        for n in sizes_of(rows):
            common = dict(modality=modality, n=n, variant="normalized", metric="cosine")
            for kind, config in (("ivf", "sqrt"), ("ivf", "engine"), ("hnsw", "sweep")):
                found = select(rows, index_kind=kind, config=config, **common)
                if found:
                    row = found[0]
                    print(f"{modality:<9} {n:>8,} {kind + '/' + config:<12} "
                          f"{fmt(number(row, 'index_build_time_s'), '.3f'):>9} "
                          f"{fmt(number(row, 'build_time_all_cores_s'), '.3f'):>10} "
                          f"{fmt(number(row, 'index_space_bytes'), '.1f', MB):>9} "
                          f"{fmt(number(row, 'build_peak_rss_delta_bytes'), '.1f', MB):>13} "
                          f"{fmt(number(row, 'query_peak_rss_delta_bytes'), '.1f', MB):>16}")


def print_metric_table(rows: list[dict]) -> None:
    print(f"\nCoseno vs. euclidiana, N = {max(sizes_of(rows)):,}: recall, precisión y tiempo con el mismo parámetro\n")
    print(f"{'modalidad':<9} {'variante':<11} {'índice':<10} {'parámetro':<18} "
          f"{'recall cos':>10} {'recall euc':>10} {'P@k cos':>8} {'P@k euc':>8} {'ms cos':>9} {'ms euc':>9}")
    probes = {"exact": {}, "ivf": {"n_probe": 16, "config": "sqrt"}, "ivf_numpy": {"n_probe": 16, "config": "sqrt"},
              "hnsw": {"ef_search": 200}}
    for modality in modalities_of(rows):
        for variant in VARIANTS:
            for kind, extra in probes.items():
                common = dict(modality=modality, n=max(sizes_of(rows)), variant=variant, index_kind=kind, **extra)
                cosine = select(rows, metric="cosine", **common)
                euclidean = select(rows, metric="euclidean", **common)
                if cosine and euclidean:
                    a, b = cosine[0], euclidean[0]
                    print(f"{modality:<9} {variant:<11} {kind:<10} {parameter(a):<18} "
                          f"{float(a['recall_at_k']):>10.3f} {float(b['recall_at_k']):>10.3f} "
                          f"{float(a['precision_at_k']):>8.4f} {float(b['precision_at_k']):>8.4f} "
                          f"{float(a['avg_time_ms']):>9.3f} {float(b['avg_time_ms']):>9.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="lee el CSV de --quick")
    parser.add_argument("--csv", type=Path, default=None, help="CSV a verificar")
    args = parser.parse_args()

    path = args.csv or (quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV)
    rows = read_rows(path)
    print(f"{len(rows)} filas en {path}")
    if not rows:
        sys.exit(1)

    for warning in missing_combinations(rows):
        print(f"Aviso: {warning}")
    print_fastest_table(rows)
    print_defaults_table(rows)
    print_cost_table(rows)
    print_metric_table(rows)

    errors = check(rows)
    if errors:
        print(f"\n✗ {len(errors)} verificaciones fallaron:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print(f"\n✓ Mismo k ({TOP_K}) y {NUM_QUERIES} consultas; IVF y su control coinciden; el recall de IVF no baja "
          "con n_probe; las métricas equivalentes encuentran los mismos vecinos; HNSW aprovecha los núcleos.")


if __name__ == "__main__":
    main()
