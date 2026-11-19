"""
apps/audio-copyright/evaluation/calibrate_threshold.py — Elige los valores por defecto de
la detección con el corpus real: umbral, mínimo de segmentos seguidos y margen frente a
otras canciones (issues #113 y #115).

Positivos: fragmentos de canciones del corpus, de 15 y 10 s desde 7.3 s (fuera de la
grilla de 2.5 s de los segmentos), y el de 15 s con ruido blanco a 20 dB. Negativos:
fragmentos de 15 s y clips completos de 30 s de canciones de fma_small con licencia
aceptada que NO están en el corpus.

Las canciones son otras que las del benchmark (benchmark_detection.py usa las primeras
BENCHMARK_PER_GENRE de cada género): si se eligieran los valores con las mismas consultas
con que después se miden, el benchmark saldría mejor de lo que es.

Cada fragmento se busca una sola vez (estos valores no cambian la búsqueda) y después se
evalúa la agrupación con cada combinación:
    - recall: positivos en los que aparece su canción,
    - falsos positivos: negativos con alguna coincidencia,
    - coincidencias ajenas: positivos que además reportan otra canción.
Se elige la de mayor recall con a lo sumo MAX_FALSE_POSITIVE_RATE de falsos positivos.

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
BENCHMARK_PER_GENRE = 5          # benchmark_detection.PER_GENRE: esas canciones se saltan
EXCERPT_START_S = 7.3
NOISE_SNR_DB = 20.0
SEED = 113
THRESHOLDS = [round(0.70 + 0.01 * step, 2) for step in range(30)]
MIN_SEGMENTS = (2, 3, 4)
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05)
MAX_FALSE_POSITIVE_RATE = 0.05


# =============================================================================
# Fragmentos
# =============================================================================

def excerpt(signal: np.ndarray, seconds: float, start_s: float = EXCERPT_START_S) -> np.ndarray:
    start = int(start_s * SAMPLE_RATE)
    return signal[start:start + int(seconds * SAMPLE_RATE)]


def with_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(signal ** 2))
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), len(signal))
    return (signal + noise).astype(np.float32)


def decode(data_or_path) -> np.ndarray:
    import librosa

    source = io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else str(data_or_path)
    signal, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    return signal


def variants(kind: str, signal: np.ndarray, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    if kind == "positivo":
        fifteen = excerpt(signal, 15.0)
        return [("15s", fifteen), ("10s", excerpt(signal, 10.0)), ("ruido_15s", with_noise(fifteen, NOISE_SNR_DB, rng))]
    return [("15s", excerpt(signal, 15.0)), ("30s", signal)]


def pick_queries(searcher: SegmentSearcher, corpus_dir: Path, per_genre: int) -> list[dict]:
    """per_genre positivos y per_genre negativos por género, después de los del benchmark."""
    rows = searcher.client.query("SELECT song_id, genre FROM audio_songs;")["rows"]
    in_corpus = {int(song_id): genre for song_id, genre in rows}

    metadata, _ = open_zip(METADATA_URL, METADATA_BUFFER)
    candidates = candidates_by_genre(read_small_tracks(metadata), read_excluded())
    archive, _ = open_zip(SMALL_URL)
    window = slice(BENCHMARK_PER_GENRE, BENCHMARK_PER_GENRE + per_genre)
    queries = []
    for genre, tracks in sorted(candidates.items()):
        for track in [t for t in tracks if t.song_id in in_corpus][window]:
            queries.append({"kind": "positivo", "song_id": track.song_id, "genre": genre,
                            "signal": decode(corpus_dir / f"{track.song_id:06d}.mp3")})
        # Los negativos se bajan de fma_small y pasan los mismos filtros que el corpus:
        # FMA tiene algunos clips dañados, cortos o en silencio.
        valid = []
        for track in (t for t in tracks if t.song_id not in in_corpus):
            if len(valid) == window.stop:
                break
            try:
                signal = decode(archive.read(member_name(track.song_id)))
            except Exception:
                continue
            if len(signal) < MIN_DURATION_S * SAMPLE_RATE or float(np.max(np.abs(signal))) < SILENCE_PEAK:
                continue
            valid.append({"kind": "negativo", "song_id": track.song_id, "genre": genre, "signal": signal})
        queries += valid[window]
    return queries


# =============================================================================
# Evaluación
# =============================================================================

def evaluate(searches: list[dict], threshold: float, min_segments: int, margin: float) -> dict:
    settings = DetectionSettings(threshold=threshold, min_segments=min_segments, margin=margin)
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
        "margin": margin,
        "recall": round(found / positives, 4) if positives else None,
        "false_positive_rate": round(false_positive / negatives, 4) if negatives else None,
        "foreign_match_rate": round(wrong / positives, 4) if positives else None,
    }


def choose(results: list[dict]) -> dict:
    allowed = [r for r in results if r["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE]
    if not allowed:
        raise RuntimeError("Ninguna combinación deja los falsos positivos bajo el límite")
    # Mayor recall; a igual recall, menos coincidencias ajenas y las reglas menos estrictas.
    return max(allowed, key=lambda r: (r["recall"], -r["foreign_match_rate"], -r["threshold"],
                                       -r["min_segments"], -r["margin"]))


def per_variant(searches: list[dict], settings: dict) -> dict:
    """La combinación elegida, por variante de positivos y de negativos."""
    values = {key: settings[key] for key in ("threshold", "min_segments", "margin")}
    result = {}
    for kind in ("positivo", "negativo"):
        for variant in sorted({item["variant"] for item in searches if item["kind"] == kind}):
            subset = [item for item in searches if item["kind"] == kind and item["variant"] == variant]
            scores = evaluate(subset, **values)
            result[f"{kind}_{variant}"] = (scores["recall"] if kind == "positivo" else scores["false_positive_rate"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=Path("."), help="carpeta de los resultados")
    parser.add_argument("--per-genre", type=int, default=PER_GENRE)
    args = parser.parse_args()

    started = time.perf_counter()
    searcher = SegmentSearcher.from_settings()
    corpus_dir = searcher.settings.codebook_path.parent
    queries = pick_queries(searcher, corpus_dir, args.per_genre)
    print(f"{len(queries)} canciones en {time.perf_counter() - started:.0f} s", flush=True)

    rng = np.random.default_rng(SEED)
    searches, latencies = [], []
    for query in queries:
        for variant, signal in variants(query["kind"], query["signal"], rng):
            search = searcher.search_audio(signal)
            latencies.append(search["timings_ms"]["total_ms"])
            searches.append({"kind": query["kind"], "song_id": query["song_id"], "variant": variant, "search": search})
    print(f"{len(searches)} búsquedas en {time.perf_counter() - started:.0f} s", flush=True)

    results = [evaluate(searches, threshold, minimum, margin)
               for margin in MARGINS for minimum in MIN_SEGMENTS for threshold in THRESHOLDS]
    best = choose(results)
    without_margin = choose([r for r in results if r["margin"] == 0.0])

    top_scores = defaultdict(list)
    for item in searches:
        best_scores = [max((c["score"] for c in s["candidates"]
                            if item["kind"] == "negativo" or c["song_id"] == item["song_id"]), default=0.0)
                       for s in item["search"]["segments"]]
        top_scores[item["kind"]].append(float(np.median(best_scores)))

    report = {
        "queries": {"positives": sum(q["kind"] == "positivo" for q in queries),
                    "negatives": sum(q["kind"] == "negativo" for q in queries),
                    "positive_variants": ["15s", "10s", f"ruido_15s ({NOISE_SNR_DB:g} dB)"],
                    "negative_variants": ["15s", "30s"],
                    "excerpt_start_s": EXCERPT_START_S,
                    "skipped_per_genre": BENCHMARK_PER_GENRE},
        "rule": f"mayor recall con falsos positivos <= {MAX_FALSE_POSITIVE_RATE:.0%}",
        "chosen": best,
        "chosen_by_variant": per_variant(searches, best),
        "best_without_margin": without_margin,
        "best_without_margin_by_variant": per_variant(searches, without_margin),
        "median_segment_score": {kind: round(float(np.median(values)), 4) for kind, values in top_scores.items()},
        "latency_ms": {"median": round(float(np.median(latencies)), 1), "p95": round(float(np.percentile(latencies, 95)), 1)},
        "search": {key: searches[0]["search"][key] for key in ("index_type", "metric", "k", "query_shifts_s", "model_name")
                   if key in searches[0]["search"]},
        "grid": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "threshold_calibration.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'margen':>6} {'umbral':>6} {'mín':>4} {'recall':>7} {'falsos +':>9} {'ajenas':>7}")
    for margin in MARGINS:
        allowed = [r for r in results if r["margin"] == margin and r["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE]
        row = max(allowed, key=lambda r: (r["recall"], -r["threshold"]), default=None)
        if row:
            print(f"{margin:>6.2f} {row['threshold']:>6.2f} {row['min_segments']:>4} {row['recall']:>7.3f} "
                  f"{row['false_positive_rate']:>9.3f} {row['foreign_match_rate']:>7.3f}")
    print(f"\nElegido: {best}\nPor variante: {report['chosen_by_variant']}\n"
          f"Mejor sin margen: {without_margin}\nPor variante: {report['best_without_margin_by_variant']}\n"
          f"Score mediano por segmento: {report['median_segment_score']}\nLatencia: {report['latency_ms']}\n"
          f"Resultados: {path} ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
