"""
benchmarks/plot_vector_results.py — Gráficas de la comparación IVF vs. HNSW y
coseno vs. euclidiana a partir del CSV común: recall frente a latencia, tiempo de
consulta y de construcción, espacio en disco, memoria y efecto de la métrica.

Antes de graficar aplica las verificaciones de compare_vector_results.py y sale con
código 1 si alguna falla: no se publican gráficas de una comparación inválida.

Uso:
    python benchmarks/plot_vector_results.py            # results/charts/vector_*.png
    python benchmarks/plot_vector_results.py --quick    # results/charts_quick/vector_*.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_vector import RESULTS_CSV, TOP_K, read_rows
from compare_vector_results import (
    MB,
    MIN_RECALL,
    check,
    fastest_at_recall,
    missing_combinations,
    modalities_of,
    number,
    quick_path,
    select,
    sizes_of,
)
from vector_common import RESULTS_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# (etiqueta, color, marcador). El color sigue al índice en todas las gráficas.
IVF = ("IVF del motor (Python)", "#2a78d6", "o")
IVF_NUMPY = ("IVF en numpy (control)", "#eb6834", "s")
HNSW = ("HNSW (hnswlib, C++)", "#1baf7a", "^")
SWEEP_STYLES = (("ivf", IVF), ("ivf_numpy", IVF_NUMPY), ("hnsw", HNSW))

BASELINE = dict(variant="normalized", metric="cosine")
MODALITY_NAMES = {"image": "Imágenes (Tiny ImageNet)", "audio": "Audio (Speech Commands)"}
VARIANT_NAMES = {"normalized": "Vectores normalizados (motor)", "raw": "Vectores sin normalizar"}
NUM_CLASSES = {"image": 200, "audio": 35}
N_LABEL = "N (objetos, escala log)"
RECALL_LABEL = f"Recall@{TOP_K}"
QUERY_TIME_LABEL = "Tiempo por consulta (ms, escala log)"


# =============================================================================
# Datos
# =============================================================================

def sweep_points(rows, kind, **filters):
    """[(recall, ms)] del barrido de `kind`, en orden de esfuerzo creciente."""
    effort = "ef_search" if kind == "hnsw" else "n_probe"
    found = [row for row in select(rows, index_kind=kind, **filters) if row["config"] != "engine"]
    found.sort(key=lambda row: int(row[effort]))
    return [(number(row, "recall_at_k"), number(row, "avg_time_ms")) for row in found]


def by_n(rows, column, scale=1.0, **filters):
    """{n: valor} de una columna de construcción (igual en todo el barrido de un índice)."""
    values = {}
    for row in select(rows, **filters):
        value = number(row, column)
        if value is not None:
            values.setdefault(int(row["n"]), value / scale)
    return values


def fastest_by_n(rows, kind, **filters):
    """{n: ms} de lo más rápido con recall suficiente; falta el N donde no se alcanza."""
    values = {}
    for n in sizes_of(rows):
        row = fastest_at_recall(rows, kind, n=n, **filters)
        if row:
            values[n] = number(row, "avg_time_ms")
    return values


# =============================================================================
# Gráficas
# =============================================================================

def draw(ax, points, style, linestyle="-", label=None):
    """Línea sobre ejes log; un delta de RSS <= 0 no se puede dibujar en ellos."""
    points = sorted((x, y) for x, y in points if y > 0)
    if points:
        name, color, marker = style
        ax.plot(*zip(*points), label=label or name, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.5, markersize=6)


def style_axes(ax, title=None, xlabel=None, ylabel=None, ns=None, log_y=True):
    from matplotlib.ticker import FuncFormatter, NullLocator

    if ns:
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n:,}" for n in ns])
        ax.xaxis.set_minor_locator(NullLocator())
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.10g}"))
    if title:
        ax.set_title(title, color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED_INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED_INK)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_INK, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def save(fig, axes, path, title):
    """Título y leyenda comunes (una entrada por serie) y guarda el PNG."""
    import numpy as np

    entries = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    fig.suptitle(title, color=INK, fontsize=13)
    fig.legend(entries.values(), entries.keys(), loc="lower center", ncol=min(len(entries), 4),
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.09 if len(entries) > 4 else 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def plot_recall_latency(plt, rows, path):
    modalities, ns = modalities_of(rows), sizes_of(rows)
    fig, axes = plt.subplots(len(modalities), len(ns), figsize=(4.3 * len(ns), 3.9 * len(modalities)),
                             sharex=True, sharey=True, squeeze=False, facecolor=SURFACE)
    for i, modality in enumerate(modalities):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            common = dict(modality=modality, n=n, **BASELINE)
            for kind, style in SWEEP_STYLES:
                draw(ax, sweep_points(rows, kind, **common), style)
            for row in select(rows, index_kind="exact", **common):
                ax.axhline(number(row, "avg_time_ms"), color=MUTED_INK, linestyle=":", linewidth=1.5,
                           label="Fuerza bruta (numpy, recall 1)")
            for row in select(rows, is_engine_default=True, **common):
                recall, ms = number(row, "recall_at_k"), number(row, "avg_time_ms")
                if row["index_kind"] == "ivf":     # otras listas: no está sobre la curva
                    ax.plot(recall, ms, marker=IVF[2], color=IVF[1], markersize=6, linestyle="none")
                ax.plot(recall, ms, marker="o", markersize=12, markerfacecolor="none", markeredgecolor=INK,
                        linestyle="none", label="Valores por defecto del motor")
            style_axes(ax, title=f"{MODALITY_NAMES[modality]}, N = {n:,}",
                       xlabel=RECALL_LABEL if i == len(modalities) - 1 else None,
                       ylabel=QUERY_TIME_LABEL if j == 0 else None)
    return save(fig, axes, path, f"Recall frente a tiempo de consulta, coseno, k = {TOP_K}")


def plot_query_time(plt, rows, path):
    modalities, ns = modalities_of(rows), sizes_of(rows)
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.5 * len(modalities), 4.8), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for j, (ax, modality) in enumerate(zip(axes[0], modalities)):
        common = dict(modality=modality, **BASELINE)
        for kind, style in SWEEP_STYLES:
            draw(ax, fastest_by_n(rows, kind, **common).items(), style)
        exact = by_n(rows, "avg_time_ms", index_kind="exact", **common)
        draw(ax, exact.items(), ("Fuerza bruta (numpy)", MUTED_INK, "D"), linestyle=":")
        style_axes(ax, title=MODALITY_NAMES[modality], xlabel=N_LABEL, ns=ns,
                   ylabel=QUERY_TIME_LABEL if j == 0 else None)
    return save(fig, axes, path, f"Lo más rápido con Recall@{TOP_K} ≥ {MIN_RECALL}, coseno")


def plot_build_time(plt, rows, path):
    modalities, ns = modalities_of(rows), sizes_of(rows)
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.5 * len(modalities), 4.8), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for j, (ax, modality) in enumerate(zip(axes[0], modalities)):
        common = dict(modality=modality, **BASELINE)
        draw(ax, by_n(rows, "index_build_time_s", index_kind="ivf", config="sqrt", **common).items(), IVF,
             label="IVF (√N listas), un hilo")
        hnsw = dict(index_kind="hnsw", **common)
        draw(ax, by_n(rows, "index_build_time_s", **hnsw).items(), HNSW, label="HNSW, un hilo")
        draw(ax, by_n(rows, "build_time_all_cores_s", **hnsw).items(), HNSW, linestyle="--",
             label="HNSW, todos los núcleos")
        style_axes(ax, title=MODALITY_NAMES[modality], xlabel=N_LABEL, ns=ns,
                   ylabel="Tiempo de construcción (s, escala log)" if j == 0 else None)
    return save(fig, axes, path, "Tiempo de construcción del índice")


def plot_index_space(plt, rows, path):
    modalities, ns = modalities_of(rows), sizes_of(rows)
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.5 * len(modalities), 4.8), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for j, (ax, modality) in enumerate(zip(axes[0], modalities)):
        common = dict(modality=modality, **BASELINE)
        draw(ax, by_n(rows, "index_space_bytes", MB, index_kind="ivf", config="sqrt", **common).items(), IVF,
             label="IVF (√N listas)")
        draw(ax, by_n(rows, "index_space_bytes", MB, index_kind="hnsw", **common).items(), HNSW, label="HNSW")
        style_axes(ax, title=MODALITY_NAMES[modality], xlabel=N_LABEL, ns=ns,
                   ylabel="Tamaño del índice (MB, escala log)" if j == 0 else None)
    return save(fig, axes, path, "Espacio en disco del índice")


def plot_memory(plt, rows, path):
    modalities, ns = modalities_of(rows), sizes_of(rows)
    phases = (("build_peak_rss_delta_bytes", "construcción"), ("query_peak_rss_delta_bytes", "consultas"))
    fig, axes = plt.subplots(len(modalities), 2, figsize=(10.5, 3.9 * len(modalities)), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for i, modality in enumerate(modalities):
        common = dict(modality=modality, **BASELINE)
        for j, (column, phase) in enumerate(phases):
            ax = axes[i][j]
            draw(ax, by_n(rows, column, MB, index_kind="ivf", config="sqrt", **common).items(), IVF,
                 label="IVF (√N listas)")
            draw(ax, by_n(rows, column, MB, index_kind="hnsw", **common).items(), HNSW, label="HNSW")
            style_axes(ax, title=f"{MODALITY_NAMES[modality]}, {phase}", ns=ns,
                       xlabel=N_LABEL if i == len(modalities) - 1 else None,
                       ylabel="Pico de RSS - RSS inicial (MB, escala log)" if j == 0 else None)
    return save(fig, axes, path, "Memoria por fase")


def plot_metric_quality(plt, rows, path):
    """Precisión exacta: no depende del índice, solo de los vectores y la métrica."""
    modalities, ns = modalities_of(rows), sizes_of(rows)
    # Eje común: las imágenes son unos 20 aciertos de 2,000; a su propia escala ese
    # ruido parecería una diferencia entre métricas.
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.5 * len(modalities), 4.8), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    series = (
        (dict(variant="normalized", metric="cosine"), ("Coseno y euclidiana normalizada (mismos vecinos)", INK, "o"), "-"),
        (dict(variant="raw", metric="euclidean"), ("Euclidiana sin normalizar", MUTED_INK, "s"), "--"),
    )
    for j, (ax, modality) in enumerate(zip(axes[0], modalities)):
        for filters, style, linestyle in series:
            values = by_n(rows, "precision_at_k", 0.01, modality=modality, index_kind="exact", **filters)
            draw(ax, values.items(), style, linestyle=linestyle)
        ax.axhline(100 / NUM_CLASSES[modality], color=MUTED_INK, linestyle=":", linewidth=1.5,
                   label="Al azar (1 / clases)")
        style_axes(ax, title=MODALITY_NAMES[modality], xlabel=N_LABEL, ns=ns, log_y=False,
                   ylabel=f"Precision@{TOP_K} exacta (%)" if j == 0 else None)
    axes[0][0].set_ylim(bottom=0)   # después de dibujar: fijar antes congela el eje común
    return save(fig, axes, path, "Vecinos de la misma clase según la métrica")


def plot_metric_tradeoff(plt, rows, path):
    modalities, n = modalities_of(rows), max(sizes_of(rows))
    variants = ("normalized", "raw")
    metrics = (("cosine", "coseno", "-"), ("euclidean", "euclidiana", "--"))
    fig, axes = plt.subplots(len(modalities), 2, figsize=(10.5, 3.9 * len(modalities)), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for i, modality in enumerate(modalities):
        for j, variant in enumerate(variants):
            ax = axes[i][j]
            for kind, style in (("ivf", IVF), ("hnsw", HNSW)):
                for metric, name, linestyle in metrics:
                    points = sweep_points(rows, kind, modality=modality, n=n, variant=variant, metric=metric)
                    draw(ax, points, style, linestyle=linestyle, label=f"{style[0].split(' (')[0]}, {name}")
            style_axes(ax, title=f"{MODALITY_NAMES[modality]}, {VARIANT_NAMES[variant].lower()}",
                       xlabel=RECALL_LABEL if i == len(modalities) - 1 else None,
                       ylabel=QUERY_TIME_LABEL if j == 0 else None)
    return save(fig, axes, path, f"Coseno frente a euclidiana, N = {n:,}, k = {TOP_K}")


def make_charts(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    charts = (
        ("vector_recall_latency.png", plot_recall_latency),
        ("vector_query_time.png", plot_query_time),
        ("vector_build_time.png", plot_build_time),
        ("vector_index_space.png", plot_index_space),
        ("vector_memory.png", plot_memory),
        ("vector_metric_quality.png", plot_metric_quality),
        ("vector_metric_tradeoff.png", plot_metric_tradeoff),
    )
    written = [plot(plt, rows, out_dir / name) for name, plot in charts]
    plt.close("all")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="lee el CSV de --quick y escribe en charts_quick/")
    parser.add_argument("--csv", type=Path, default=None, help="CSV a graficar")
    args = parser.parse_args()

    path = args.csv or (quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV)
    rows = read_rows(path)
    if not rows:
        print(f"No hay filas en {path}")
        sys.exit(1)

    errors = check(rows)
    if errors:
        print(f"✗ {len(errors)} verificaciones fallaron; ejecuta compare_vector_results.py para ver el detalle.")
        sys.exit(1)
    for warning in missing_combinations(rows):
        print(f"Aviso: {warning}")

    out_dir = RESULTS_DIR / ("charts_quick" if args.quick else "charts")
    for chart in make_charts(rows, out_dir):
        print(f"Gráfica: {chart}")


if __name__ == "__main__":
    main()
