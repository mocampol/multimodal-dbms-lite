"""
apps/audio-copyright/evaluation/compare_features.py — Compara formas de convertir las
tramas MFCC de un segmento en un vector (issue #115), sin tocar la base de datos.

Todas las variantes usan MFCCExtractor, Quantizer y BoWTFIDF del motor sin modificarlos;
solo cambia qué se hace con las tramas antes y después del codebook:
    actual            histograma de 64 palabras de las tramas MFCC (lo que usa la app)
    cmvn              sin el coeficiente 0 (volumen) y normalizando cada coeficiente en el
                      segmento antes de cuantizar
    cmvn_deltas       cmvn más los deltas (cómo cambia cada coeficiente entre tramas)
    sub_bloques       un histograma por cada mitad del segmento, concatenados
    cmvn_sub_bloques  cmvn y sub_bloques juntos
    actual_128        histograma de 128 palabras
    cmvn_deltas_128   cmvn_deltas con 128 palabras

Los MFCC y sus deltas se calculan una sola vez por canción y por consulta (con los 4
corrimientos de la app), y cada variante entrena su codebook y su IDF a partir de ellos.
Las consultas son las de benchmark_detection.py y se buscan con fuerza bruta en numpy
(coseno), con la misma agrupación de la app (detection.py).

Cada variante produce otra distribución de similitudes, así que a cada una se le busca su
umbral y su mínimo de segmentos: el mayor recall con falsos positivos <= 5%. Como se
eligen y se miden con las mismas consultas, los números son optimistas por igual para
todas: sirven para compararlas, y la ganadora se valida después con la calibración y el
benchmark reales.

Corre dentro de un contenedor con el corpus y la API:
    docker compose --profile audio run --rm \\
        -v "$PWD/apps/audio-copyright/evaluation/results:/out" \\
        audio-api python apps/audio-copyright/evaluation/compare_features.py --out /out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for path in (APP_DIR.parents[1] / "src", APP_DIR / "dataset", APP_DIR / "features", APP_DIR / "backend",
             APP_DIR / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from benchmark_detection import CASES, build_queries, score  # noqa: E402
from calibrate_threshold import MAX_FALSE_POSITIVE_RATE, decode  # noqa: E402
from detection import DetectionSettings, detect  # noqa: E402
from index.multimedia import BoWTFIDF, MFCCExtractor, Quantizer  # noqa: E402
from load_songs import HttpDBMSClient  # noqa: E402
from segment_extractor import (  # noqa: E402
    CODEBOOK_SIZE, HOP_LENGTH, HOP_S, MAX_FRAMES_PER_SONG, N_MFCC, SAMPLE_RATE, WINDOW_S, segment_intervals,
)
from segment_search import SearchSettings  # noqa: E402

K = 10
THRESHOLDS = [round(0.80 + 0.005 * step, 3) for step in range(40)]
MIN_SEGMENTS = (2, 3, 4)


@dataclass(frozen=True)
class Variant:
    name: str
    cmvn: bool = False       # sin c0 y normalizado por segmento
    deltas: bool = False
    blocks: int = 1
    k: int = CODEBOOK_SIZE


VARIANTS = (
    Variant("actual"),
    Variant("cmvn", cmvn=True),
    Variant("cmvn_deltas", cmvn=True, deltas=True),
    Variant("sub_bloques", blocks=2),
    Variant("cmvn_sub_bloques", cmvn=True, blocks=2),
    Variant("actual_128", k=128),
    Variant("cmvn_deltas_128", cmvn=True, deltas=True, k=128),
)


# =============================================================================
# Tramas y segmentos
# =============================================================================

MFCC = MFCCExtractor(n_mfcc=N_MFCC, sr=SAMPLE_RATE)


def frames_of(signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Tramas MFCC con sus deltas (T, 2 * N_MFCC) y la duración del audio."""
    import librosa

    mfcc = MFCC.extract(signal)
    width = min(9, len(mfcc) - (1 - len(mfcc) % 2))   # delta pide un ancho impar <= tramas
    deltas = librosa.feature.delta(mfcc.T, width=max(3, width)).T if len(mfcc) >= 3 else np.zeros_like(mfcc)
    return np.hstack([mfcc, deltas]).astype(np.float32), len(signal) / SAMPLE_RATE


def segment_slices(n_frames: int, duration_s: float) -> list[tuple[float, float, slice]]:
    """Los mismos segmentos y tramas que SegmentMFCCExtractor._segments."""
    result = []
    for start, end in segment_intervals(duration_s, WINDOW_S, HOP_S):
        first = min(int(round(start * SAMPLE_RATE / HOP_LENGTH)), n_frames - 1)
        last = max(first + 1, min(int(round(end * SAMPLE_RATE / HOP_LENGTH)), n_frames))
        result.append((start, end, slice(first, last)))
    return result


def segment_features(frames: np.ndarray, variant: Variant) -> np.ndarray:
    """Las columnas que usa la variante, normalizadas en el segmento si corresponde."""
    columns = list(range(1 if variant.cmvn else 0, N_MFCC))
    if variant.deltas:
        columns += list(range(N_MFCC + (1 if variant.cmvn else 0), 2 * N_MFCC))
    features = frames[:, columns]
    if variant.cmvn:
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    return features


def blocks_of(n: int, blocks: int) -> list[slice]:
    edges = np.linspace(0, n, blocks + 1).round().astype(int)
    return [slice(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(blocks)]


class VariantModel:
    """Codebook e IDF de una variante, entrenados con los segmentos de la base."""

    def __init__(self, variant: Variant, songs: list[tuple[np.ndarray, float]], seed: int = 42) -> None:
        self.variant = variant
        rng = np.random.default_rng(seed)
        per_song = []
        sample = []
        for frames, duration in songs:
            segments = [segment_features(frames[s], variant) for _, _, s in segment_slices(len(frames), duration)]
            per_song.append(segments)
            stacked = np.vstack(segments)
            take = min(len(stacked), MAX_FRAMES_PER_SONG)
            sample.append(stacked[rng.choice(len(stacked), take, replace=False)])
        self.quantizer = Quantizer(k=variant.k, random_state=seed).fit(np.vstack(sample))
        self.bow = BoWTFIDF(k=variant.k).fit(
            [words for segments in per_song for words in self._words(segments)])

    def _words(self, segments: list[np.ndarray]) -> list[np.ndarray]:
        lengths = [len(s) for s in segments]
        words = self.quantizer.predict(np.vstack(segments))
        return np.split(words, np.cumsum(lengths)[:-1])

    def vectors(self, frames: np.ndarray, duration_s: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Un vector por segmento, con norma 1, y sus intervalos."""
        slices = segment_slices(len(frames), duration_s)
        segments = [segment_features(frames[s], self.variant) for _, _, s in slices]
        result = []
        for words in self._words(segments):
            parts = [self.bow.compute(words[block]) for block in blocks_of(len(words), self.variant.blocks)]
            vector = np.concatenate(parts)
            result.append(vector / (np.linalg.norm(vector) or 1.0))
        return np.vstack(result).astype(np.float32), [(start, end) for start, end, _ in slices]


# =============================================================================
# Búsqueda y evaluación
# =============================================================================

def search(model: VariantModel, base: dict, query_frames: list[tuple[float, np.ndarray, float]]) -> dict:
    """Como SegmentSearcher.search_audio, con fuerza bruta: el resultado que espera detect()."""
    grid = None
    per_shift = []
    for shift_s, frames, duration in query_frames:
        vectors, intervals = model.vectors(frames, duration)
        scores = vectors @ base["matrix"].T
        per_shift.append((shift_s, scores))
        if grid is None:
            grid = intervals
    segments = []
    for position, (start, end) in enumerate(grid):
        best: dict[int, dict] = {}
        for shift_s, scores in per_shift:
            if position >= len(scores):
                continue
            row = scores[position]
            for column in np.argpartition(-row, min(K, len(row) - 1))[:K]:
                value = float(row[column])
                if column not in best or value > best[column]["score"]:
                    best[column] = {"song_id": base["song_ids"][column], "score": value,
                                    "start_s": base["starts"][column] - shift_s,
                                    "end_s": base["starts"][column] - shift_s + WINDOW_S}
        candidates = sorted(best.values(), key=lambda c: -c["score"])[:K]
        segments.append({"segment_index": position, "start_s": start, "end_s": end, "candidates": candidates})
    return {"duration_s": query_frames[0][2], "segments": segments}


def evaluate(searches: list[dict], threshold: float, min_segments: int) -> dict:
    settings = DetectionSettings(threshold=threshold, min_segments=min_segments)
    rows = [{"case": item["case"], **score(item, detect(item["search"], settings))} for item in searches]
    positives = [r for r in rows if r["expected"]]
    negatives = [r for r in rows if not r["expected"]]
    by_case = {}
    for case in CASES:
        items = [r for r in rows if r["case"] == case]
        if items:
            expected = sum(r["expected"] for r in items)
            by_case[case] = (round(sum(r["found"] for r in items) / expected, 3) if expected
                             else round(sum(r["false_positive"] for r in items) / len(items), 3))
    return {
        "threshold": threshold, "min_segments": min_segments,
        "recall": round(sum(r["found"] for r in positives) / sum(r["expected"] for r in positives), 4),
        "false_positive_rate": round(sum(r["false_positive"] for r in negatives) / len(negatives), 4),
        "by_case": by_case,
    }


def best_configuration(searches: list[dict]) -> dict:
    results = [evaluate(searches, t, m) for m in MIN_SEGMENTS for t in THRESHOLDS]
    allowed = [r for r in results if r["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE]
    return max(allowed or results, key=lambda r: (r["recall"], -r["false_positive_rate"], -r["threshold"]))


class _Searcher:
    """Lo que build_queries necesita del buscador: un cliente con query()."""

    def __init__(self, url: str) -> None:
        self.client = HttpDBMSClient(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--per-genre", type=int, default=5, help="consultas por género (como el benchmark)")
    parser.add_argument("--base-songs", type=int, default=0, help="canciones de la base; 0 = todo el corpus")
    parser.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    args = parser.parse_args()

    settings = SearchSettings()
    corpus_dir = settings.codebook_path.parent
    searcher = _Searcher(settings.dbms_api_url)
    timings = {}
    started = time.perf_counter()

    queries = build_queries(searcher, corpus_dir, args.per_genre)
    timings["consultas_s"] = round(time.perf_counter() - started, 1)

    expected = sorted({song_id for q in queries for song_id in q["targets"]})
    all_songs = sorted(int(row[0]) for row in searcher.client.query("SELECT song_id FROM audio_songs;")["rows"])
    others = [s for s in all_songs if s not in set(expected)]
    base_ids = expected + (others if args.base_songs <= 0 else others[: max(0, args.base_songs - len(expected))])

    mark = time.perf_counter()
    base_frames = {song_id: frames_of(decode(corpus_dir / f"{song_id:06d}.mp3")) for song_id in base_ids}
    timings["mfcc_base_s"] = round(time.perf_counter() - mark, 1)

    mark = time.perf_counter()
    for item in queries:
        signal = item.pop("signal")
        item["frames"] = []
        for shift_s in settings.query_shifts_s:
            start = int(round(shift_s * SAMPLE_RATE))
            if shift_s > 0 and len(signal) - start < WINDOW_S * SAMPLE_RATE:
                continue
            item["frames"].append((shift_s, *frames_of(signal[start:])))
    timings["mfcc_consultas_s"] = round(time.perf_counter() - mark, 1)
    print(f"{len(queries)} consultas y {len(base_ids)} canciones de base; {timings}", flush=True)

    report = {"base_songs": len(base_ids), "queries": len(queries), "per_genre": args.per_genre,
              "rule": f"mayor recall con falsos positivos <= {MAX_FALSE_POSITIVE_RATE:.0%}", "variants": {}}
    chosen = [v for v in VARIANTS if v.name in args.variants.split(",")]
    for variant in chosen:
        mark = time.perf_counter()
        model = VariantModel(variant, list(base_frames.values()))
        matrices, song_ids, starts = [], [], []
        for song_id, (frames, duration) in base_frames.items():
            vectors, intervals = model.vectors(frames, duration)
            matrices.append(vectors)
            song_ids += [song_id] * len(vectors)
            starts += [start for start, _ in intervals]
        base = {"matrix": np.vstack(matrices), "song_ids": song_ids, "starts": starts}
        searches = [{**{k: v for k, v in item.items() if k != "frames"}, "search": search(model, base, item["frames"])}
                    for item in queries]
        best = best_configuration(searches)
        best["dimension"] = int(base["matrix"].shape[1])
        best["seconds"] = round(time.perf_counter() - mark, 1)
        report["variants"][variant.name] = best
        print(f"  {variant.name}: umbral {best['threshold']}, mín {best['min_segments']}, recall {best['recall']:.3f}, "
              f"falsos + {best['false_positive_rate']:.3f} ({best['seconds']:.0f} s)", flush=True)

    report["timings"] = {**timings, "total_s": round(time.perf_counter() - started, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "feature_comparison.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'variante':<18} {'dim':>4} {'umbral':>7} {'mín':>4} {'recall':>7} {'falsos +':>9}  por caso")
    for name, values in report["variants"].items():
        cases = ", ".join(f"{case}={value}" for case, value in values["by_case"].items())
        print(f"{name:<18} {values['dimension']:>4} {values['threshold']:>7} {values['min_segments']:>4} "
              f"{values['recall']:>7.3f} {values['false_positive_rate']:>9.3f}  {cases}")
    print(f"\nResultados: {path} ({report['timings']['total_s']:.0f} s)")


if __name__ == "__main__":
    main()
