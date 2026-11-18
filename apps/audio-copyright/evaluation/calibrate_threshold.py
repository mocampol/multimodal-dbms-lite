"""
apps/audio-copyright/evaluation/calibrate_threshold.py — Elige el umbral por defecto de
la detección (issue #113) con el corpus real.

Positivos: fragmentos de canciones del corpus. Negativos: fragmentos de canciones de
fma_small con licencia aceptada que NO están en el corpus. Cada fragmento dura 15 s y
empieza en 7.3 s, fuera de la grilla de 2.5 s de los segmentos, y se prueba limpio y con
ruido blanco a 20 dB.

Cada fragmento se busca una sola vez (el umbral no cambia la búsqueda) y después se
evalúa la agrupación con cada umbral y cada mínimo de segmentos:
    - recall: positivos en los que aparece su canción,
    - falsos positivos: negativos con alguna coincidencia,
    - coincidencias ajenas: positivos que además reportan otra canción.
El umbral elegido es el de mayor recall con a lo sumo MAX_FALSE_POSITIVE_RATE de falsos
positivos. La validación completa (más casos y latencia) es la del issue #115.

Corre dentro del contenedor audio-api, que tiene el codebook, el corpus y la API:
    docker compose --profile audio run --rm \\
        -v "$PWD/apps/audio-copyright/evaluation/results:/out" \\
        audio-api python apps/audio-copyright/evaluation/calibrate_threshold.py --out /out
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for path in (APP_DIR.parents[1] / "src", APP_DIR / "dataset", APP_DIR / "features", APP_DIR / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from detection import DetectionSettings, detect  # noqa: E402
from fma_corpus import (  # noqa: E402
    METADATA_BUFFER, METADATA_URL, MIN_DURATION_S, SILENCE_PEAK, SMALL_URL, candidates_by_genre, member_name,
    open_zip, read_excluded, read_small_tracks,
)
from segment_extractor import SAMPLE_RATE  # noqa: E402
from segment_search import SegmentSearcher  # noqa: E402

PER_GENRE = 10
EXCERPT_START_S = 7.3
EXCERPT_S = 15.0
NOISE_SNR_DB = 20.0
SEED = 113
THRESHOLDS = [round(0.80 + 0.01 * step, 2) for step in range(20)]
MIN_SEGMENTS = (2, 3, 4)
MAX_FALSE_POSITIVE_RATE = 0.05


# =============================================================================
# Fragmentos
# =============================================================================

def excerpt(signal: np.ndarray) -> np.ndarray:
    start = int(EXCERPT_START_S * SAMPLE_RATE)
    return signal[start:start + int(EXCERPT_S * SAMPLE_RATE)]


def with_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(signal ** 2))
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), len(signal))
    return (signal + noise).astype(np.float32)


def decode(data_or_path) -> np.ndarray:
    import librosa

    source = io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else str(data_or_path)
    signal, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    return signal


def pick_queries(searcher: SegmentSearcher, corpus_dir: Path, per_genre: int) -> list[dict]:
    """per_genre positivos y per_genre negativos por género, siempre los mismos."""
    rows = searcher.client.query("SELECT song_id, genre FROM audio_songs;")["rows"]
    in_corpus = {int(song_id): genre for song_id, genre in rows}

    metadata, _ = open_zip(METADATA_URL, METADATA_BUFFER)
    candidates = candidates_by_genre(read_small_tracks(metadata), read_excluded())
    archive, _ = open_zip(SMALL_URL)
    queries = []
    for genre, tracks in sorted(candidates.items()):
        for track in [t for t in tracks if t.song_id in in_corpus][:per_genre]:
            queries.append({"kind": "positivo", "song_id": track.song_id, "genre": genre,
                            "signal": decode(corpus_dir / f"{track.song_id:06d}.mp3")})
        # Los negativos se bajan de fma_small y pasan los mismos filtros que el corpus:
        # FMA tiene algunos clips dañados, cortos o en silencio.
        negatives = 0
        for track in (t for t in tracks if t.song_id not in in_corpus):
            if negatives == per_genre:
                break
            try:
                signal = decode(archive.read(member_name(track.song_id)))
            except Exception:
                continue
            if len(signal) < MIN_DURATION_S * SAMPLE_RATE or float(np.max(np.abs(signal))) < SILENCE_PEAK:
                continue
            queries.append({"kind": "negativo", "song_id": track.song_id, "genre": genre, "signal": signal})
            negatives += 1
    return queries


# =============================================================================
# Evaluación
# =============================================================================

def evaluate(searches: list[dict], threshold: float, min_segments: int) -> dict:
    settings = DetectionSettings(threshold=threshold, min_segments=min_segments)
    found = wrong = false_positive = positives = negatives = 0
    for item in searches:
        songs = {match["song_id"] for match in detect(item["search"], settings)["matches"]}
        if item["kind"] == "positivo":
            positives += 1
            found += item["song_id"] in songs
            wrong += bool(songs - {item["song_id"]})
        else:
            negatives += 1
            false_positive += bool(songs)
    return {
        "threshold": threshold,
        "min_segments": min_segments,
        "recall": round(found / positives, 4),
        "false_positive_rate": round(false_positive / negatives, 4),
        "foreign_match_rate": round(wrong / positives, 4),
    }


def choose(results: list[dict]) -> dict:
    allowed = [r for r in results if r["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE]
    if not allowed:
        raise RuntimeError("Ningún umbral deja los falsos positivos bajo el límite")
    # Mayor recall; a igual recall, menos coincidencias ajenas y el umbral más bajo.
    return max(allowed, key=lambda r: (r["recall"], -r["foreign_match_rate"], -r["threshold"], -r["min_segments"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=Path("."), help="carpeta de los resultados")
    parser.add_argument("--per-genre", type=int, default=PER_GENRE)
    args = parser.parse_args()

    started = time.perf_counter()
    searcher = SegmentSearcher.from_settings()
    corpus_dir = searcher.settings.codebook_path.parent
    queries = pick_queries(searcher, corpus_dir, args.per_genre)
    print(f"{len(queries)} fragmentos en {time.perf_counter() - started:.0f} s", flush=True)

    rng = np.random.default_rng(SEED)
    searches, latencies = [], []
    for query in queries:
        clean = excerpt(query["signal"])
        for variant, signal in (("limpio", clean), ("ruido", with_noise(clean, NOISE_SNR_DB, rng))):
            search = searcher.search_audio(signal)
            latencies.append(search["timings_ms"]["total_ms"])
            searches.append({"kind": query["kind"], "song_id": query["song_id"], "variant": variant, "search": search})

    results = [evaluate(searches, threshold, minimum) for minimum in MIN_SEGMENTS for threshold in THRESHOLDS]
    by_variant = {}
    best = choose(results)
    for variant in ("limpio", "ruido"):
        subset = [item for item in searches if item["variant"] == variant]
        by_variant[variant] = evaluate(subset, best["threshold"], best["min_segments"])

    top_scores = defaultdict(list)
    for item in searches:
        best_scores = [max((c["score"] for c in s["candidates"]
                            if item["kind"] == "negativo" or c["song_id"] == item["song_id"]), default=0.0)
                       for s in item["search"]["segments"]]
        top_scores[item["kind"]].append(float(np.median(best_scores)))

    report = {
        "queries": {"positives": sum(q["kind"] == "positivo" for q in queries),
                    "negatives": sum(q["kind"] == "negativo" for q in queries),
                    "variants": ["limpio", f"ruido {NOISE_SNR_DB:g} dB"],
                    "excerpt": f"{EXCERPT_S:g} s desde {EXCERPT_START_S:g} s"},
        "rule": f"mayor recall con falsos positivos <= {MAX_FALSE_POSITIVE_RATE:.0%}",
        "chosen": best,
        "chosen_by_variant": by_variant,
        "median_segment_score": {kind: round(float(np.median(values)), 4) for kind, values in top_scores.items()},
        "latency_ms": {"median": round(float(np.median(latencies)), 1), "p95": round(float(np.percentile(latencies, 95)), 1)},
        "search": {key: searches[0]["search"][key] for key in ("index_type", "metric", "k", "model_name")},
        "grid": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "threshold_calibration.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'umbral':>6} {'mín':>4} {'recall':>7} {'falsos +':>9} {'ajenas':>7}")
    for row in results:
        if row["threshold"] in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
            print(f"{row['threshold']:>6.2f} {row['min_segments']:>4} {row['recall']:>7.3f} "
                  f"{row['false_positive_rate']:>9.3f} {row['foreign_match_rate']:>7.3f}")
    print(f"\nElegido: {best}\nPor variante: {by_variant}\nScore mediano por segmento: "
          f"{report['median_segment_score']}\nLatencia: {report['latency_ms']}\nResultados: {path}  "
          f"({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
