"""
benchmarks/plot_vector_gallery.py — Galería de aciertos y errores de la búsqueda por
similitud: cada fila es una consulta con sus 10 vecinos exactos, marcados según sean
o no de su misma clase. El audio se muestra como espectrograma.

Usa la fuerza bruta porque la galería muestra lo que pueden lograr los vectores,
sin los errores propios de cada índice (esos los mide el recall). Los casos son los
mejores y los peores de las consultas, no una muestra representativa: el título de
la figura indica la precisión promedio.

Las figuras contienen imágenes del dataset, así que se quedan en local: results/gallery/
está en .gitignore. Necesita el corpus de prepare_vector_data.py.

Uso:
    python benchmarks/plot_vector_gallery.py                  # N = 100,000
    python benchmarks/plot_vector_gallery.py --n 1000 --modality audio
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_vector import TOP_K, distances_to, top_k_ids
from vector_common import DATASETS, MODALITIES, RESULTS_DIR, data_dir, load_features

GALLERY_DIR = RESULTS_DIR / "gallery"
CASES_PER_GROUP = 3
AUDIO_SAMPLE_RATE = 16_000
AUDIO_MELS = 64

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
SAME_CLASS = "#1baf7a"
OTHER_CLASS = "#eb6834"
MODALITY_NAMES = {"image": "Imágenes (Tiny ImageNet)", "audio": "Audio (Speech Commands)"}


# =============================================================================
# Selección de casos
# =============================================================================

@dataclass(frozen=True)
class Case:
    query: int
    neighbors: tuple[int, ...]
    hits: int


def exact_neighbors(base: np.ndarray, queries: np.ndarray, metric: str = "cosine", k: int = TOP_K) -> np.ndarray:
    return np.array([top_k_ids(distances_to(base, query, metric), k) for query in queries])


def count_hits(neighbors: np.ndarray, base_labels: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    """Por consulta, cuántos vecinos son de su misma clase."""
    return (base_labels[neighbors] == query_labels[:, None]).sum(axis=1)


def select_cases(neighbors: np.ndarray, hits: np.ndarray, per_group: int = CASES_PER_GROUP) -> list[Case]:
    """Las consultas con más aciertos y luego las de menos, sin repetir ninguna.
    Los empates se resuelven por posición, así que la selección es reproducible."""
    order = sorted(range(len(hits)), key=lambda query: (-int(hits[query]), query))
    best = order[:per_group]
    worst = [query for query in reversed(order) if query not in best][:per_group]
    worst.sort(key=lambda query: (int(hits[query]), query))
    return [Case(query, tuple(int(i) for i in neighbors[query]), int(hits[query])) for query in best + worst]


# =============================================================================
# Figuras
# =============================================================================

def read_picture(path: Path, modality: str) -> np.ndarray:
    if modality == "image":
        import cv2

        picture = cv2.imread(str(path))
        if picture is None:
            raise ValueError(f"No se pudo leer {path}")
        return cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)

    import librosa

    signal, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=signal, sr=AUDIO_SAMPLE_RATE, n_mels=AUDIO_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def draw_cell(ax, picture: np.ndarray, title: str, frame: str, modality: str) -> None:
    if modality == "image":
        ax.imshow(picture)
    else:
        ax.imshow(picture, origin="lower", aspect="auto", cmap="Greys")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7, color=INK, pad=3)
    for spine in ax.spines.values():
        spine.set_color(frame)
        spine.set_linewidth(2.5)


def make_figure(features, cases: list[Case], modality: str, n: int, mean_precision: float, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = data_dir(DATASETS[modality])
    columns = TOP_K + 1
    fig, axes = plt.subplots(len(cases), columns, figsize=(1.45 * columns, 1.85 * len(cases)),
                             squeeze=False, facecolor=SURFACE)
    for row, case in zip(axes, cases):
        query_label = str(features.query_labels[case.query])
        draw_cell(row[0], read_picture(root / str(features.query_paths[case.query]), modality),
                  f"Consulta: {query_label}", INK, modality)
        row[0].set_xlabel(f"{case.hits}/{TOP_K} de su clase", fontsize=8, color=INK)
        for ax, neighbor in zip(row[1:], case.neighbors):
            label = str(features.base_labels[neighbor])
            same = label == query_label
            draw_cell(ax, read_picture(root / str(features.base_paths[neighbor]), modality),
                      f"{'✓' if same else '✗'} {label}", SAME_CLASS if same else OTHER_CLASS, modality)
    fig.suptitle(f"{MODALITY_NAMES[modality]}, N = {n:,}: las {len(cases) // 2} mejores y las {len(cases) // 2} "
                 f"peores consultas de {len(features.query_labels)}. Precision@{TOP_K} promedio: {mean_precision:.2%}",
                 color=INK, fontsize=11)
    fig.text(0.5, 0.01, "Vecinos exactos por coseno, de más a menos parecido. "
             "✓ misma clase que la consulta, ✗ otra clase.", ha="center", fontsize=8, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_gallery(modality: str, n: int, quick: bool = False) -> Path:
    features = load_features(modality, quick=quick)
    base = features.base_vectors(n, "normalized")
    neighbors = exact_neighbors(base, features.query_vectors("normalized"))
    hits = count_hits(neighbors, features.base_labels[:n], features.query_labels)
    mean_precision = float(hits.mean()) / TOP_K
    return make_figure(features, select_cases(neighbors, hits), modality, n, mean_precision,
                       GALLERY_DIR / f"vector_gallery_{modality}_{n}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--modality", choices=[*MODALITIES, "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="usa los vectores de --quick")
    args = parser.parse_args()

    for modality in (MODALITIES if args.modality == "all" else [args.modality]):
        print(f"Galería: {make_gallery(modality, args.n, args.quick)}", flush=True)


if __name__ == "__main__":
    main()
