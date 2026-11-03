"""
benchmarks/compare_text_results.py — Verifica que la comparación SPIMI vs.
PostgreSQL GIN sea válida y resume el CSV común por corpus, N y grupo de
longitud de consulta.

Verificaciones, por (corpus, n, query_id), entre todos los motores presentes:
    - misma consulta y mismo k (= TOP_K),
    - mismos documentos relevantes (relevant_docs y relevant_digest),
    - mismos candidatos: matched_docs == expected_matches en cada motor;
y, por (corpus, n), el mismo vocabulario en todos los índices.

Sale con código 1 si alguna verificación falla.

Uso:
    python benchmarks/compare_text_results.py            # results/text_comparison_results.csv
    python benchmarks/compare_text_results.py --quick    # results/text_comparison_results_quick.csv
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_common import ENGINES, QUERY_GROUPS, RESULTS_CSV, TOP_K, quick_path

SHARED_COLUMNS = ("query", "k", "relevant_docs", "relevant_digest", "expected_matches")
ENGINE_ORDER = list(ENGINES)
GROUP_ORDER = list(QUERY_GROUPS)


# =============================================================================
# Verificación
# =============================================================================

def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(rows: list[dict]) -> list[str]:
    errors = []
    by_query = defaultdict(list)
    vocabulary = defaultdict(dict)

    for row in rows:
        label = f"{row['engine']} {row['corpus']} N={row['n']} {row['query_id']}"
        if row["k"] != str(TOP_K):
            errors.append(f"{label}: k={row['k']}, se esperaba {TOP_K}")
        if row["matched_docs"] != row["expected_matches"]:
            errors.append(f"{label}: {row['matched_docs']} candidatos, "
                          f"se esperaban {row['expected_matches']}")
        by_query[(row["corpus"], row["n"], row["query_id"])].append(row)
        vocabulary[(row["corpus"], row["n"])][row["engine"]] = row["vocabulary_size"]

    for (corpus, n, query_id), group in by_query.items():
        for column in SHARED_COLUMNS:
            if len({row[column] for row in group}) > 1:
                values = ", ".join(f"{row['engine']}={row[column]}" for row in group)
                errors.append(f"{corpus} N={n} {query_id}: {column} distinto entre motores ({values})")

    for (corpus, n), by_engine in vocabulary.items():
        if len(set(by_engine.values())) > 1:
            values = ", ".join(f"{engine}={size}" for engine, size in by_engine.items())
            errors.append(f"{corpus} N={n}: vocabulario distinto ({values})")
    return errors


def missing_engines(rows: list[dict]) -> list[str]:
    present = defaultdict(set)
    for row in rows:
        present[(row["corpus"], row["n"])].add(row["engine"])
    return [f"{corpus} N={n}: falta {', '.join(sorted(set(ENGINES) - engines))}"
            for (corpus, n), engines in sorted(present.items()) if set(ENGINES) - engines]


# =============================================================================
# Resumen en consola
# =============================================================================

def mean(values):
    numbers = [float(v) for v in values if v not in ("", None)]
    return statistics.fmean(numbers) if numbers else None


def fmt(value, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def sort_key(corpus, n, engine, group=None):
    return (corpus, int(n), GROUP_ORDER.index(group) if group else 0, ENGINE_ORDER.index(engine))


def print_query_table(rows: list[dict]) -> None:
    cells = defaultdict(list)
    for row in rows:
        cells[(row["corpus"], row["n"], row["engine"], row["query_group"])].append(row)

    print(f"\nConsultas (k = {TOP_K}; promedio sobre las consultas de cada grupo)\n")
    print(f"{'corpus':<10} {'N':>8} {'grupo':<5} {'motor':<12} {'avg ms':>10} "
          f"{'P@k':>6} {'R@k':>6} {'índice':>7}")
    for key in sorted(cells, key=lambda k: sort_key(*k)):
        corpus, n, engine, group = key
        group_rows = cells[key]
        used = all(row["used_index"] == "True" for row in group_rows)
        print(f"{corpus:<10} {int(n):>8,} {group:<5} {engine:<12} "
              f"{fmt(mean(r['avg_time_ms'] for r in group_rows), '.3f'):>10} "
              f"{fmt(mean(r['precision_at_k'] for r in group_rows), '.2f'):>6} "
              f"{fmt(mean(r['recall_at_k'] for r in group_rows), '.2f'):>6} "
              f"{'sí' if used else 'no':>7}")


def print_build_table(rows: list[dict]) -> None:
    first = {}
    for row in rows:
        first.setdefault((row["corpus"], row["n"], row["engine"]), row)

    print("\nConstrucción, disco y memoria (pico de RSS - RSS inicial del proceso de cada fase)\n")
    print(f"{'corpus':<10} {'N':>8} {'motor':<12} {'build s':>9} {'prep s':>8} "
          f"{'índice KB':>11} {'mem build KB':>13} {'mem consulta KB':>16}")
    for key in sorted(first, key=lambda k: sort_key(*k)):
        corpus, n, engine = key
        row = first[key]
        prep = mean([row["preprocess_time_s"]]) or 0.0
        prep += mean([row["tsvector_time_s"]]) or 0.0
        kb = lambda column: fmt(mean([row[column]]) / 1024 if mean([row[column]]) is not None else None, ",.0f")
        print(f"{corpus:<10} {int(n):>8,} {engine:<12} "
              f"{fmt(mean([row['index_build_time_s']]), '.3f'):>9} "
              f"{fmt(prep if engine == 'pg_gin' else None, '.3f'):>8} "
              f"{kb('index_space_bytes'):>11} {kb('build_peak_rss_delta_bytes'):>13} "
              f"{kb('query_peak_rss_delta_bytes'):>16}")
    print("\nprep s (solo PostgreSQL): tokenización compartida en Python + to_tsvector. "
          "En SPIMI ese trabajo está dentro de build s.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="lee el CSV de --quick")
    parser.add_argument("--csv", type=Path, default=None, help="CSV a verificar")
    args = parser.parse_args()

    path = args.csv or (quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV)
    rows = read_rows(path)
    print(f"{len(rows)} filas en {path}")

    for warning in missing_engines(rows):
        print(f"Aviso: {warning}")
    print_query_table(rows)
    print_build_table(rows)

    errors = check(rows)
    if errors:
        print(f"\n✗ {len(errors)} verificaciones fallaron:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print(f"\n✓ Mismo k ({TOP_K}), mismos relevantes, mismos candidatos y mismo vocabulario "
          "en todos los motores.")


if __name__ == "__main__":
    main()