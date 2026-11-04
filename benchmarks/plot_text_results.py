"""
benchmarks/plot_text_results.py — Gráficas de la comparación de texto SPIMI vs.
PostgreSQL GIN a partir del CSV común: tiempo de construcción, tiempo de consulta
por longitud de consulta, espacio en disco y memoria.

Antes de graficar aplica las verificaciones de compare_text_results.py y sale con
código 1 si alguna falla: no se publican gráficas de una comparación inválida.

Uso:
    python benchmarks/plot_text_results.py            # results/charts/text_*.png
    python benchmarks/plot_text_results.py --quick    # results/charts_quick/text_*.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_text_results import check, mean, missing_engines, read_rows
from text_common import DEFAULT_CORPUS, ENGINES, QUERY_GROUPS, RESULTS_CSV, RESULTS_DIR, quick_path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MB = 1024 * 1024

# (etiqueta, color, marcador). El color sigue al motor en todas las gráficas.
BM25 = (ENGINES["spimi_bm25"], "#2a78d6", "o")
TFIDF = (ENGINES["spimi_tfidf"], "#eb6834", "s")
GIN_RANK = (ENGINES["pg_gin"], "#1baf7a", "^")
RANKED_ENGINES = (("spimi_bm25", BM25), ("spimi_tfidf", TFIDF), ("pg_gin", GIN_RANK))

# Construcción, disco y memoria de construcción dependen del índice, no del
# ranking: BM25 y TF-IDF comparten el índice SPIMI y ts_rank no interviene en GIN.
SPIMI_INDEX = ("SPIMI (BM25 y TF-IDF)", BM25[1], BM25[2])
GIN_INDEX = ("PostgreSQL GIN", GIN_RANK[1], GIN_RANK[2])
GIN_WITH_PREP = ("GIN + tokenización y tsvector", GIN_RANK[1], "v")


# =============================================================================
# Datos
# =============================================================================

def series(rows, engine, columns, group=None, scale=1.0):
    """{n: valor}: promedio por N de la suma de `columns` en cada fila del motor
    (y del grupo de consultas, si se indica). Omite las filas sin alguna columna."""
    by_n = {}
    for row in rows:
        if row["engine"] != engine or (group is not None and row["query_group"] != group):
            continue
        if all(row[column] not in ("", None) for column in columns):
            by_n.setdefault(int(row["n"]), []).append(sum(float(row[column]) for column in columns))
    return {n: mean(values) / scale for n, values in by_n.items()}


# =============================================================================
# Gráficas
# =============================================================================

def draw(ax, values, style, linestyle="-"):
    """Línea sobre ejes log-log; un delta de RSS <= 0 no se puede dibujar en ellos."""
    points = sorted((n, value) for n, value in values.items() if value > 0)
    if points:
        label, color, marker = style
        ax.plot(*zip(*points), label=label, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.5, markersize=6)


def style_axes(ax, ns, title=None, ylabel=None):
    from matplotlib.ticker import FuncFormatter, LogFormatter, NullLocator

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.10g}"))
    ax.yaxis.set_minor_formatter(LogFormatter())
    ax.set_xlabel("N (documentos, escala log)", color=MUTED_INK)
    if title:
        ax.set_title(title, color=INK, fontsize=11)
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
    entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    fig.suptitle(title, color=INK, fontsize=13)
    fig.legend(entries.values(), entries.keys(), loc="lower center", ncol=len(entries),
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def make_charts(rows, out_dir: Path, corpus: str = DEFAULT_CORPUS) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ns = sorted({int(row["n"]) for row in rows})
    suffix = "" if corpus == DEFAULT_CORPUS else f"_{corpus}"
    written = []

    # 1) Tiempo de construcción. SPIMI tokeniza dentro de su tiempo; en PostgreSQL
    # esa preparación se mide aparte, así que se muestra con y sin ella.
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    draw(ax, series(rows, "spimi_bm25", ["index_build_time_s"]), SPIMI_INDEX)
    draw(ax, series(rows, "pg_gin", ["index_build_time_s"]), GIN_INDEX)
    draw(ax, series(rows, "pg_gin", ["index_build_time_s", "preprocess_time_s", "tsvector_time_s"]),
         GIN_WITH_PREP, linestyle="--")
    style_axes(ax, ns, ylabel="Tiempo (s, escala log)")
    written.append(save(fig, [ax], out_dir / f"text_build_time{suffix}.png",
                        f"Tiempo de construcción del índice ({corpus})"))

    # 2) Tiempo de consulta por longitud de consulta.
    groups = list(QUERY_GROUPS)
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.8), sharey=True, facecolor=SURFACE)
    for i, (ax, group) in enumerate(zip(axes, groups)):
        for engine, style in RANKED_ENGINES:
            draw(ax, series(rows, engine, ["avg_time_ms"], group=group), style)
        style_axes(ax, ns, title="1 término" if group == "1" else f"{group} términos",
                   ylabel="Tiempo promedio por consulta (ms, escala log)" if i == 0 else None)
    written.append(save(fig, axes, out_dir / f"text_query_time{suffix}.png",
                        f"Tiempo de consulta, k = 10 ({corpus})"))

    # 3) Espacio en disco del índice.
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    draw(ax, series(rows, "spimi_bm25", ["index_space_bytes"], scale=MB), SPIMI_INDEX)
    draw(ax, series(rows, "pg_gin", ["index_space_bytes"], scale=MB), GIN_INDEX)
    style_axes(ax, ns, ylabel="Tamaño del índice (MB, escala log)")
    written.append(save(fig, [ax], out_dir / f"text_index_space{suffix}.png",
                        f"Espacio en disco del índice ({corpus})"))

    # 4) Memoria: pico de RSS menos RSS inicial del proceso de cada fase.
    fig, (build_ax, query_ax) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    draw(build_ax, series(rows, "spimi_bm25", ["build_peak_rss_delta_bytes"], scale=MB), SPIMI_INDEX)
    draw(build_ax, series(rows, "pg_gin", ["build_peak_rss_delta_bytes"], scale=MB), GIN_INDEX)
    style_axes(build_ax, ns, title="Construcción", ylabel="Pico de RSS - RSS inicial (MB, escala log)")
    for engine, style in RANKED_ENGINES:
        draw(query_ax, series(rows, engine, ["query_peak_rss_delta_bytes"], scale=MB), style)
    style_axes(query_ax, ns, title="Consultas")
    written.append(save(fig, [build_ax, query_ax], out_dir / f"text_memory{suffix}.png",
                        f"Memoria por fase ({corpus})"))

    plt.close("all")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="lee el CSV de --quick y escribe en charts_quick/")
    parser.add_argument("--csv", type=Path, default=None, help="CSV a graficar")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help=f"corpus a graficar (por defecto: {DEFAULT_CORPUS})")
    args = parser.parse_args()

    path = args.csv or (quick_path(RESULTS_CSV) if args.quick else RESULTS_CSV)
    rows = read_rows(path)

    errors = check(rows)
    if errors:
        print(f"✗ {len(errors)} verificaciones fallaron; ejecuta compare_text_results.py para ver el detalle.")
        sys.exit(1)
    for warning in missing_engines(rows):
        print(f"Aviso: {warning}")

    rows = [row for row in rows if row["corpus"] == args.corpus]
    if not rows:
        print(f"No hay filas del corpus '{args.corpus}' en {path}")
        sys.exit(1)

    out_dir = RESULTS_DIR / ("charts_quick" if args.quick else "charts")
    for chart in make_charts(rows, out_dir, args.corpus):
        print(f"Gráfica: {chart}")


if __name__ == "__main__":
    main()