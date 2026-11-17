"""
benchmarks/compare_vector_dims.py — Verifica la comparación por dimensión (#108),
resume el CSV y dibuja sus gráficas.

Verificaciones:
    - mismo k y mismo número de consultas en todas las filas,
    - recall y precisión entre 0 y 1, y recall 1 en la fuerza bruta,
    - cada índice cargado de disco devolvió los mismos vecinos que el original,
    - IVFIndex y su control en numpy devuelven lo mismo: solo cambia el tiempo,
    - en IVF, sondear más listas nunca baja el recall,
    - el tamaño en disco crece con la dimensión (si no, se construyó con otra).

Sale con código 1 si alguna falla, y en ese caso no dibuja nada.

Uso:
    python benchmarks/compare_vector_dims.py            # results/vector_dims_results.csv
    python benchmarks/compare_vector_dims.py --quick    # results/vector_dims_results_quick.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_vector import ENGINE_HNSW_EF, ENGINE_IVF_PROBE, HNSW_EFS, TOP_K, read_rows
from benchmark_vector_dims import PYTHON_IVF_PROBES, RESULTS_CSV, ivf_probes_for
from compare_vector_results import MB, MIN_RECALL, TOLERANCE, fmt, modalities_of, number, quick_path, select
from vector_common import NUM_QUERIES, RESULTS_DIR

INDEX_KINDS = ("hnsw", "ivf_numpy", "ivf")


def dims_of(rows: list[dict]) -> list[int]:
    return sorted({int(row["dim"]) for row in rows})


def sizes_of(rows: list[dict]) -> list[int]:
    return sorted({int(row["n"]) for row in rows})


# =============================================================================
# Verificación
# =============================================================================

def _label(row: dict) -> str:
    return f"{row['modality']} k={row['dim']} N={row['n']} {row['index_kind']}"


def _check_rows(rows: list[dict]) -> list[str]:
    errors = []
    for row in rows:
        if row["k"] != str(TOP_K):
            errors.append(f"{_label(row)}: k={row['k']}, se esperaba {TOP_K}")
        if row["num_queries"] != str(NUM_QUERIES):
            errors.append(f"{_label(row)}: {row['num_queries']} consultas, se esperaban {NUM_QUERIES}")
        for column in ("recall_at_k", "precision_at_k"):
            value = number(row, column)
            if value is None or not 0.0 <= value <= 1.0:
                errors.append(f"{_label(row)}: {column}={row[column]} fuera de [0, 1]")
        if row["index_kind"] == "exact":
            if number(row, "recall_at_k") != 1.0:
                errors.append(f"{_label(row)}: la fuerza bruta tiene recall {row['recall_at_k']}")
        elif row["save_load_identical"] != "True":
            errors.append(f"{_label(row)}: el índice cargado no devolvió lo mismo que el original")
    return errors


def _check_ivf_control(rows: list[dict]) -> list[str]:
    errors = []
    pairs = defaultdict(dict)
    for row in rows:
        if row["index_kind"] in ("ivf", "ivf_numpy"):
            pairs[(row["modality"], row["dim"], row["n"], row["n_probe"])][row["index_kind"]] = row
    for key, pair in pairs.items():
        if "ivf" not in pair:      # el índice en Python solo se mide con algunos n_probe
            continue
        if "ivf_numpy" not in pair:
            errors.append(f"{' '.join(key)}: falta ivf_numpy")
            continue
        for column in ("recall_at_k", "precision_at_k", "avg_candidates"):
            a, b = number(pair["ivf"], column), number(pair["ivf_numpy"], column)
            if a is None or b is None or abs(a - b) > TOLERANCE:
                errors.append(f"{' '.join(key)}: {column} distinto entre ivf e ivf_numpy ({a} vs. {b})")
    return errors


def _check_ivf_recall_grows(rows: list[dict]) -> list[str]:
    errors = []
    sweeps = defaultdict(list)
    for row in select(rows, index_kind="ivf_numpy"):
        sweeps[(row["modality"], row["dim"], row["n"])].append(row)
    for key, sweep in sweeps.items():
        sweep.sort(key=lambda row: int(row["n_probe"]))
        for before, after in zip(sweep, sweep[1:]):
            if number(after, "recall_at_k") < number(before, "recall_at_k") - TOLERANCE:
                errors.append(f"{' '.join(key)}: el recall baja de n_probe={before['n_probe']} "
                              f"a n_probe={after['n_probe']}")
    return errors


def _check_space_grows_with_dim(rows: list[dict]) -> list[str]:
    errors = []
    spaces = defaultdict(dict)
    for row in rows:
        if row["index_kind"] in ("hnsw", "ivf_numpy"):
            spaces[(row["modality"], row["n"], row["index_kind"])][int(row["dim"])] = number(row, "index_space_bytes")
    for key, by_dim in spaces.items():
        ordered = [by_dim[dim] for dim in sorted(by_dim)]
        if any(after <= before for before, after in zip(ordered, ordered[1:])):
            errors.append(f"{' '.join(key)}: el tamaño en disco no crece con la dimensión ({ordered})")
    return errors


def check(rows: list[dict]) -> list[str]:
    return (_check_rows(rows) + _check_ivf_control(rows) + _check_ivf_recall_grows(rows)
            + _check_space_grows_with_dim(rows))


def missing_combinations(rows: list[dict]) -> list[str]:
    counts = defaultdict(int)
    for row in rows:
        counts[(row["modality"], int(row["dim"]), int(row["n"]))] += 1
    warnings = []
    for modality in modalities_of(rows):
        for dim in dims_of(rows):
            for n in sizes_of(rows):
                expected = 1 + len(HNSW_EFS) + len(ivf_probes_for(n)) + len(PYTHON_IVF_PROBES)
                found = counts[(modality, dim, n)]
                if found != expected:
                    warnings.append(f"{modality} k={dim} N={n}: {found} filas de {expected}")
    return warnings


# =============================================================================
# Consultas sobre los resultados
# =============================================================================

def fastest_at_recall(rows: list[dict], kind: str, min_recall: float = MIN_RECALL, **filters) -> dict | None:
    candidates = [row for row in select(rows, index_kind=kind, **filters)
                  if number(row, "recall_at_k") >= min_recall]
    return min(candidates, key=lambda row: number(row, "avg_time_ms"), default=None)


def engine_default(rows: list[dict], kind: str, **filters) -> dict | None:
    """HNSW con el ef del motor, o IVF en Python con su n_probe (con raíz de N listas)."""
    found = (select(rows, index_kind="hnsw", ef_search=ENGINE_HNSW_EF, **filters) if kind == "hnsw"
             else select(rows, index_kind="ivf", n_probe=ENGINE_IVF_PROBE, **filters))
    return found[0] if found else None


def first(rows: list[dict], **filters) -> dict | None:
    found = select(rows, **filters)
    return found[0] if found else None


# =============================================================================
# Resumen en consola
# =============================================================================

def print_quality_table(rows: list[dict], n: int) -> None:
    print(f"\nCalidad, N = {n:,}: palabras por vector, Precision@{TOP_K} exacta y recall con los valores del motor\n")
    print(f"{'modalidad':<9} {'dim':>5} {'palabras':>9} {'P@k':>7} {'HNSW ef=50':>11} {'IVF p=4':>8}")
    for modality in modalities_of(rows):
        for dim in dims_of(rows):
            common = dict(modality=modality, dim=dim, n=n)
            exact = first(rows, index_kind="exact", **common)
            hnsw, ivf = engine_default(rows, "hnsw", **common), engine_default(rows, "ivf", **common)
            print(f"{modality:<9} {dim:>5} {fmt(number(exact, 'avg_nonzero'), '.1f'):>9} "
                  f"{fmt(number(exact, 'precision_at_k'), '.4f'):>7} "
                  f"{fmt(number(hnsw, 'recall_at_k') if hnsw else None, '.3f'):>11} "
                  f"{fmt(number(ivf, 'recall_at_k') if ivf else None, '.3f'):>8}")


def print_speed_table(rows: list[dict], n: int) -> None:
    print(f"\nTiempo por consulta (ms), N = {n:,}: fuerza bruta, lo más rápido con Recall@{TOP_K} >= {MIN_RECALL} "
          f"e IVF del motor\n")
    print(f"{'modalidad':<9} {'dim':>5} {'exacta':>8} {'HNSW':>8} {'ef':>4} {'IVF numpy':>10} {'p':>3} {'IVF Python p=4':>15}")
    for modality in modalities_of(rows):
        for dim in dims_of(rows):
            common = dict(modality=modality, dim=dim, n=n)
            exact = first(rows, index_kind="exact", **common)
            hnsw, numpy_ivf = fastest_at_recall(rows, "hnsw", **common), fastest_at_recall(rows, "ivf_numpy", **common)
            python_ivf = engine_default(rows, "ivf", **common)
            print(f"{modality:<9} {dim:>5} {fmt(number(exact, 'avg_time_ms'), '.3f'):>8} "
                  f"{fmt(number(hnsw, 'avg_time_ms') if hnsw else None, '.3f'):>8} {hnsw['ef_search'] if hnsw else '—':>4} "
                  f"{fmt(number(numpy_ivf, 'avg_time_ms') if numpy_ivf else None, '.3f'):>10} "
                  f"{numpy_ivf['n_probe'] if numpy_ivf else '—':>3} "
                  f"{fmt(number(python_ivf, 'avg_time_ms') if python_ivf else None, '.3f'):>15}")


def print_cost_table(rows: list[dict], n: int) -> None:
    print(f"\nConstrucción, disco, memoria y persistencia, N = {n:,} (memoria: pico de RSS - RSS inicial)\n")
    print(f"{'modalidad':<9} {'dim':>5} {'índice':<6} {'build s':>8} {'disco MB':>9} {'mem build MB':>13} "
          f"{'mem consulta MB':>16} {'save ms':>8} {'load ms':>8}")
    for modality in modalities_of(rows):
        for dim in dims_of(rows):
            for kind, label in (("hnsw", "HNSW"), ("ivf_numpy", "IVF")):
                row = first(rows, index_kind=kind, modality=modality, dim=dim, n=n)
                if row:
                    print(f"{modality:<9} {dim:>5} {label:<6} {fmt(number(row, 'index_build_time_s'), '.2f'):>8} "
                          f"{fmt(number(row, 'index_space_bytes'), '.1f', MB):>9} "
                          f"{fmt(number(row, 'build_peak_rss_delta_bytes'), '.1f', MB):>13} "
                          f"{fmt(number(row, 'query_peak_rss_delta_bytes'), '.1f', MB):>16} "
                          f"{fmt(number(row, 'save_time_s'), '.0f', 1e-3):>8} "
                          f"{fmt(number(row, 'load_time_s'), '.0f', 1e-3):>8}")


# =============================================================================
# Gráficas
# =============================================================================

# Una sola tonalidad de más clara a más oscura: la dimensión es una magnitud ordenada.
DIM_COLORS = {64: "#9ec5f0", 128: "#5a9be6", 256: "#2a78d6", 512: "#174a8c"}
DIM_LABEL = "Dimensión (tamaño del codebook)"


def _style_dims(ax, dims, log_y=True, **labels):
    from plot_vector_results import style_axes

    style_axes(ax, log_y=log_y, **labels)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, [str(dim) for dim in dims])
    ax.minorticks_off()


def plot_recall_latency(plt, rows, n, path):
    from plot_vector_results import MODALITY_NAMES, QUERY_TIME_LABEL, RECALL_LABEL, draw, save, style_axes

    modalities, dims = modalities_of(rows), dims_of(rows)
    panels = (("hnsw", "HNSW (ef de 10 a 400)", "^"), ("ivf_numpy", "IVF en numpy (n_probe de 1 a 64)", "s"))
    fig, axes = plt.subplots(len(modalities), len(panels), figsize=(10, 3.9 * len(modalities)),
                             sharex=True, sharey=True, squeeze=False, facecolor="#fcfcfb")
    for i, modality in enumerate(modalities):
        for j, (kind, title, marker) in enumerate(panels):
            ax = axes[i][j]
            for dim in dims:
                effort = "ef_search" if kind == "hnsw" else "n_probe"
                sweep = sorted(select(rows, index_kind=kind, modality=modality, dim=dim, n=n),
                               key=lambda row: int(row[effort]))
                points = [(number(row, "recall_at_k"), number(row, "avg_time_ms")) for row in sweep]
                draw(ax, points, (f"{dim} dimensiones", DIM_COLORS.get(dim, "#2a78d6"), marker),
                     label=f"{dim} dimensiones")
            style_axes(ax, title=f"{MODALITY_NAMES[modality]}: {title}",
                       xlabel=RECALL_LABEL if i == len(modalities) - 1 else None,
                       ylabel=QUERY_TIME_LABEL if j == 0 else None)
    return save(fig, axes, path, f"Recall frente a tiempo de consulta según la dimensión, N = {n:,}")


def plot_query_time(plt, rows, n, path):
    from plot_vector_results import HNSW, IVF, IVF_NUMPY, MODALITY_NAMES, QUERY_TIME_LABEL, draw, save

    modalities, dims = modalities_of(rows), dims_of(rows)
    exact_style = ("Fuerza bruta (numpy)", "#52514e", "D")
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.5 * len(modalities), 4.6), sharey=True,
                             squeeze=False, facecolor="#fcfcfb")
    for ax, modality in zip(axes[0], modalities):
        series = {exact_style: [], HNSW: [], IVF_NUMPY: [], IVF: []}
        for dim in dims:
            common = dict(modality=modality, dim=dim, n=n)
            for style, row in ((exact_style, first(rows, index_kind="exact", **common)),
                               (HNSW, fastest_at_recall(rows, "hnsw", **common)),
                               (IVF_NUMPY, fastest_at_recall(rows, "ivf_numpy", **common)),
                               (IVF, engine_default(rows, "ivf", **common))):
                if row:
                    series[style].append((dim, number(row, "avg_time_ms")))
        for style, points in series.items():
            label = {HNSW: f"HNSW (recall >= {MIN_RECALL})",
                     IVF_NUMPY: f"IVF numpy (recall >= {MIN_RECALL})",
                     IVF: f"IVF del motor, n_probe={ENGINE_IVF_PROBE}"}.get(style)
            draw(ax, points, style, label=label)
        _style_dims(ax, dims, title=MODALITY_NAMES[modality], xlabel=DIM_LABEL,
                    ylabel=QUERY_TIME_LABEL if ax is axes[0][0] else None)
    return save(fig, axes, path, f"Tiempo por consulta según la dimensión, N = {n:,}")


def plot_cost(plt, rows, n, path):
    from plot_vector_results import HNSW, IVF_NUMPY, MODALITY_NAMES, draw, save

    dims = dims_of(rows)
    panels = (("index_build_time_s", 1.0, "Construcción (s, un hilo)"),
              ("index_space_bytes", MB, "Tamaño en disco (MB)"),
              ("query_peak_rss_delta_bytes", MB, "Memoria al cargar y consultar (MB)"))
    styles = ((("hnsw", HNSW), "HNSW"), (("ivf_numpy", ("IVF", IVF_NUMPY[1], IVF_NUMPY[2])), "IVF"))
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.4), squeeze=False, facecolor="#fcfcfb")
    for ax, (column, scale, title) in zip(axes[0], panels):
        for (kind, style), name in styles:
            for modality, linestyle in zip(modalities_of(rows), ("-", "--")):
                points = []
                for dim in dims:
                    row = first(rows, index_kind=kind, modality=modality, dim=dim, n=n)
                    if row and number(row, column) is not None:
                        points.append((dim, number(row, column) / scale))
                draw(ax, points, style, linestyle=linestyle,
                     label=f"{name}, {MODALITY_NAMES[modality].split(' (')[0].lower()}")
        _style_dims(ax, dims, log_y=False, title=title, xlabel=DIM_LABEL)
        ax.set_ylim(bottom=0)
    return save(fig, axes, path, f"Costo de los índices según la dimensión, N = {n:,}")


def plot_quality(plt, rows, n, path):
    from plot_vector_results import HNSW, IVF, MODALITY_NAMES, NUM_CLASSES, RECALL_LABEL, draw, save

    modalities, dims = modalities_of(rows), dims_of(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), squeeze=False, facecolor="#fcfcfb")
    recall_ax, words_ax, precision_ax = axes[0]
    for modality, linestyle in zip(modalities, ("-", "--")):
        name = MODALITY_NAMES[modality].split(" (")[0].lower()
        exact = [first(rows, index_kind="exact", modality=modality, dim=dim, n=n) for dim in dims]
        for kind, style, label in (("hnsw", HNSW, f"HNSW ef={ENGINE_HNSW_EF}"),
                                   ("ivf", IVF, f"IVF n_probe={ENGINE_IVF_PROBE}")):
            points = [(dim, number(row, "recall_at_k")) for dim in dims
                      if (row := engine_default(rows, kind, modality=modality, dim=dim, n=n))]
            draw(recall_ax, points, style, linestyle=linestyle, label=f"{label}, {name}")
        gray = ("", "#52514e", "D")
        draw(words_ax, [(dim, number(row, "avg_nonzero")) for dim, row in zip(dims, exact) if row],
             gray, linestyle=linestyle, label=f"Fuerza bruta, {name}")
        draw(precision_ax, [(dim, number(row, "precision_at_k")) for dim, row in zip(dims, exact) if row],
             gray, linestyle=linestyle, label=f"Fuerza bruta, {name}")
        precision_ax.axhline(1 / NUM_CLASSES[modality], color="#c3c2b7", linestyle=linestyle, linewidth=1)
    _style_dims(recall_ax, dims, log_y=False, title=f"{RECALL_LABEL} con los valores del motor",
                xlabel=DIM_LABEL)
    _style_dims(words_ax, dims, log_y=False, title="Palabras distintas por vector", xlabel=DIM_LABEL)
    _style_dims(precision_ax, dims, log_y=False,
                title=f"Precision@{TOP_K} exacta (gris claro: azar)", xlabel=DIM_LABEL)
    return save(fig, axes, path, f"Calidad según la dimensión, N = {n:,}")


def make_charts(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(sizes_of(rows))
    charts = (
        ("vector_dims_recall_latency.png", plot_recall_latency),
        ("vector_dims_query_time.png", plot_query_time),
        ("vector_dims_cost.png", plot_cost),
        ("vector_dims_quality.png", plot_quality),
    )
    written = [plot(plt, rows, n, out_dir / name) for name, plot in charts]
    plt.close("all")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="lee el CSV de --quick y escribe en charts_quick/")
    parser.add_argument("--csv", type=Path, default=None, help="CSV a verificar")
    parser.add_argument("--no-charts", action="store_true", help="solo verifica y resume")
    args = parser.parse_args()

    path = args.csv or (quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV)
    rows = read_rows(path)
    print(f"{len(rows)} filas en {path}")
    if not rows:
        sys.exit(1)

    for warning in missing_combinations(rows):
        print(f"Aviso: {warning}")
    for n in sizes_of(rows):
        print_quality_table(rows, n)
        print_speed_table(rows, n)
        print_cost_table(rows, n)

    errors = check(rows)
    if errors:
        print(f"\n✗ {len(errors)} verificaciones fallaron:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print(f"\n✓ Mismo k ({TOP_K}) y {NUM_QUERIES} consultas; los índices cargados responden igual que los "
          "originales; IVF y su control coinciden; el recall de IVF no baja con n_probe; el tamaño crece "
          "con la dimensión.")

    if not args.no_charts:
        for chart in make_charts(rows, RESULTS_DIR / ("charts_quick" if args.quick else "charts")):
            print(f"Gráfica: {chart}")


if __name__ == "__main__":
    main()
