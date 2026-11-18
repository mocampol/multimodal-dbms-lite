"""
apps/audio-copyright/evaluation/benchmark_detection.py — Valida la detección de copyright
(issue #115): coincidencias, falsos positivos y latencia.

Casos, con canciones del corpus (positivos) y canciones de fma_small con licencia
aceptada que no están en el corpus (negativos):
    exacto              el clip completo de 30 s
    recortado_6s/10s/15s un fragmento que empieza en una frontera de trama MFCC (~5 s)
    desplazado_trama    el fragmento de 15 s, corrido media trama (256 muestras)
    desplazado_grilla   15 s desde 6.3 s: fuera de la grilla de 2.5 s y de las tramas
    remix               15 s de una canción seguidos de 15 s de otra
    ruido               el fragmento de 15 s con ruido blanco a 20 dB
    ajena_15s/30s       negativos: no deberían dar ninguna coincidencia

Los desplazados, el ruido y los negativos duran 15 s: con menos, el mínimo de segmentos
por defecto no se puede alcanzar, y el caso mediría la duración y no lo que dice su
nombre. Los recortes de 6 y 10 s miden justamente ese límite.

Métricas por caso y por índice (HNSW e IVF):
    recall              canciones esperadas que aparecen entre las coincidencias
    falsos_positivos    consultas con alguna coincidencia de una canción no esperada
    error_desfase_s     en las detecciones correctas, cuánto se aleja el desfase detectado
                        (inicio en la original - inicio en el audio) del real
    latencia            extracción + búsquedas en el DBMS, en el proceso; y de punta a
                        punta por POST /detections para una muestra

La búsqueda y la agrupación son las de la app (segment_search.py y detection.py), con su
configuración por defecto. --label nombra la corrida, para comparar antes y después.

Corre dentro de un contenedor con el corpus, con audio-api levantado:
    docker compose --profile audio run --rm \\
        -v "$PWD/apps/audio-copyright/evaluation/results:/out" \\
        audio-api python apps/audio-copyright/evaluation/benchmark_detection.py --out /out --label base
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for path in (APP_DIR.parents[1] / "src", APP_DIR / "dataset", APP_DIR / "features", APP_DIR / "backend",
             APP_DIR / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from calibrate_threshold import decode, with_noise  # noqa: E402
from detection import DetectionSettings, detect  # noqa: E402
from fma_corpus import (  # noqa: E402
    METADATA_BUFFER, METADATA_URL, MIN_DURATION_S, SILENCE_PEAK, SMALL_URL, candidates_by_genre, member_name,
    open_zip, read_excluded, read_small_tracks,
)
from segment_extractor import HOP_LENGTH, SAMPLE_RATE  # noqa: E402
from segment_search import INDEX_TYPES, SegmentSearcher  # noqa: E402

PER_GENRE = 5
SEED = 115
ALIGNED_START_S = 5.0           # se redondea a la frontera de trama más cercana
SHIFTED_START_S = 6.3
NOISE_SNR_DB = 20.0
HTTP_SAMPLE = 20                # consultas por índice medidas por la API HTTP
CASES = ("exacto", "recortado_6s", "recortado_10s", "recortado_15s", "desplazado_trama",
         "desplazado_grilla", "remix", "ruido", "ajena_15s", "ajena_30s")


# =============================================================================
# Consultas
# =============================================================================

def frame_aligned(seconds: float) -> int:
    """La muestra de la frontera de trama MFCC más cercana a `seconds`."""
    return int(round(seconds * SAMPLE_RATE / HOP_LENGTH)) * HOP_LENGTH


def clip(signal: np.ndarray, start_sample: int, seconds: float) -> np.ndarray:
    return signal[start_sample:start_sample + int(seconds * SAMPLE_RATE)]


def query(case: str, signal: np.ndarray, targets: dict[int, float]) -> dict:
    """targets: song_id esperado -> desfase real (inicio en la original - inicio en el audio)."""
    return {"case": case, "signal": signal.astype(np.float32), "targets": targets}


def positive_queries(song_id: int, signal: np.ndarray, rng: np.random.Generator) -> list[dict]:
    aligned = frame_aligned(ALIGNED_START_S)
    offset = aligned / SAMPLE_RATE
    fifteen = clip(signal, aligned, 15.0)
    shifted = int(SHIFTED_START_S * SAMPLE_RATE)
    return [
        query("exacto", signal, {song_id: 0.0}),
        query("recortado_6s", clip(signal, aligned, 6.0), {song_id: offset}),
        query("recortado_10s", clip(signal, aligned, 10.0), {song_id: offset}),
        query("recortado_15s", fifteen, {song_id: offset}),
        query("desplazado_trama", clip(signal, aligned + HOP_LENGTH // 2, 15.0),
              {song_id: (aligned + HOP_LENGTH // 2) / SAMPLE_RATE}),
        query("desplazado_grilla", clip(signal, shifted, 15.0), {song_id: shifted / SAMPLE_RATE}),
        query("ruido", with_noise(fifteen, NOISE_SNR_DB, rng), {song_id: offset}),
    ]


def remix_query(first: tuple[int, np.ndarray], second: tuple[int, np.ndarray]) -> dict:
    """0-15 s de la primera canción y luego 10-25 s de la segunda, como en las pruebas manuales."""
    (first_id, first_signal), (second_id, second_signal) = first, second
    signal = np.concatenate([first_signal[: 15 * SAMPLE_RATE], second_signal[10 * SAMPLE_RATE: 25 * SAMPLE_RATE]])
    return query("remix", signal, {first_id: 0.0, second_id: -5.0})


def build_queries(searcher: SegmentSearcher, corpus_dir: Path, per_genre: int) -> list[dict]:
    rows = searcher.client.query("SELECT song_id, genre FROM audio_songs;")["rows"]
    in_corpus = {int(song_id) for song_id, _ in rows}
    metadata, _ = open_zip(METADATA_URL, METADATA_BUFFER)
    candidates = candidates_by_genre(read_small_tracks(metadata), read_excluded())
    archive, _ = open_zip(SMALL_URL)
    rng = np.random.default_rng(SEED)

    queries, positives = [], []
    for genre, tracks in sorted(candidates.items()):
        for track in [t for t in tracks if t.song_id in in_corpus][:per_genre]:
            signal = decode(corpus_dir / f"{track.song_id:06d}.mp3")
            positives.append((track.song_id, signal))
            queries += positive_queries(track.song_id, signal, rng)
        negatives = 0
        for track in (t for t in tracks if t.song_id not in in_corpus):
            if negatives == per_genre:
                break
            try:
                signal = decode(archive.read(member_name(track.song_id)))
            except Exception:
                continue        # FMA tiene algunos clips dañados, cortos o en silencio
            if len(signal) < MIN_DURATION_S * SAMPLE_RATE or float(np.max(np.abs(signal))) < SILENCE_PEAK:
                continue
            queries += [query("ajena_15s", clip(signal, frame_aligned(ALIGNED_START_S), 15.0), {}),
                        query("ajena_30s", signal, {})]
            negatives += 1
    # Cada canción con la siguiente de otro género: un remix por canción.
    for position in range(len(positives)):
        queries.append(remix_query(positives[position], positives[(position + per_genre) % len(positives)]))
    return queries


# =============================================================================
# Evaluación
# =============================================================================

def score(item: dict, result: dict) -> dict:
    """Aciertos, falsos positivos y errores de desfase de una consulta."""
    targets = item["targets"]
    matches = result["matches"]
    found = {m["song_id"] for m in matches} & set(targets)
    offset_errors = []
    for song_id in found:
        best = max((m for m in matches if m["song_id"] == song_id), key=lambda m: m["segments"])
        detected = best["reference_start_s"] - best["query_start_s"]
        offset_errors.append(abs(detected - targets[song_id]))
    return {
        "expected": len(targets),
        "found": len(found),
        "false_positive": any(m["song_id"] not in targets for m in matches),
        "offset_errors": offset_errors,
    }


def summarize(rows: list[dict]) -> dict:
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    summary = {}
    for case in CASES:
        items = by_case.get(case, [])
        if not items:
            continue
        expected = sum(r["expected"] for r in items)
        errors = [e for r in items for e in r["offset_errors"]]
        latencies = [r["latency_ms"] for r in items]
        summary[case] = {
            "consultas": len(items),
            "recall": round(sum(r["found"] for r in items) / expected, 4) if expected else None,
            "falsos_positivos": round(sum(r["false_positive"] for r in items) / len(items), 4),
            "error_desfase_s_mediana": round(float(np.median(errors)), 3) if errors else None,
            "latencia_ms_mediana": round(float(np.median(latencies)), 1),
            "latencia_ms_p95": round(float(np.percentile(latencies, 95)), 1),
        }
    return summary


def wav_bytes(signal: np.ndarray) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, signal, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def http_latency(url: str, queries: list[dict], index_type: str) -> dict:
    """De punta a punta: subir el WAV, decodificar, extraer, buscar y agrupar."""
    times = []
    for item in queries:
        request = urllib.request.Request(f"{url}/detections?filename=consulta.wav&index={index_type}",
                                         data=wav_bytes(item["signal"]), method="POST",
                                         headers={"Content-Type": "audio/wav"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
        times.append((time.perf_counter() - started) * 1000.0)
    return {"consultas": len(times), "mediana_ms": round(float(np.median(times)), 1),
            "p95_ms": round(float(np.percentile(times, 95)), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--label", default="base", help="nombre de la corrida, p. ej. base o desplazamientos")
    parser.add_argument("--per-genre", type=int, default=PER_GENRE)
    parser.add_argument("--api-url", default="http://audio-api:8001", help="API de detección para la latencia HTTP")
    args = parser.parse_args()

    started = time.perf_counter()
    searcher = SegmentSearcher.from_settings()
    settings = DetectionSettings()
    queries = build_queries(searcher, searcher.settings.codebook_path.parent, args.per_genre)
    print(f"{len(queries)} consultas armadas en {time.perf_counter() - started:.0f} s", flush=True)

    report = {"label": args.label, "settings": settings.__dict__, "per_genre": args.per_genre,
              "search_k": searcher.settings.k, "indices": {}}
    for index_type in INDEX_TYPES:
        rows = []
        for item in queries:
            search = searcher.search_audio(item["signal"], index_type=index_type)
            rows.append({"case": item["case"], "latency_ms": search["timings_ms"]["total_ms"],
                         **score(item, detect(search, settings))})
        report["indices"][index_type] = {"casos": summarize(rows)}
        try:
            sample = [q for q in queries if q["case"] == "recortado_15s"][:HTTP_SAMPLE]
            report["indices"][index_type]["http_15s"] = http_latency(args.api_url, sample, index_type)
        except OSError as error:
            report["indices"][index_type]["http_15s"] = {"error": str(error)}
        print(f"  {index_type}: {time.perf_counter() - started:.0f} s", flush=True)

    report["duration_s"] = round(time.perf_counter() - started, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"detection_benchmark_{args.label}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for index_type, result in report["indices"].items():
        print(f"\n{index_type.upper()}  (umbral {settings.threshold}, mín {settings.min_segments} segmentos)")
        print(f"{'caso':<18} {'n':>3} {'recall':>7} {'falsos +':>9} {'err desfase':>12} {'ms p50':>7} {'ms p95':>7}")
        for case, values in result["casos"].items():
            recall = "—" if values["recall"] is None else f"{values['recall']:.2f}"
            error = "—" if values["error_desfase_s_mediana"] is None else f"{values['error_desfase_s_mediana']:.2f} s"
            print(f"{case:<18} {values['consultas']:>3} {recall:>7} {values['falsos_positivos']:>9.2f} "
                  f"{error:>12} {values['latencia_ms_mediana']:>7.0f} {values['latencia_ms_p95']:>7.0f}")
        print(f"HTTP, 15 s: {result['http_15s']}")
    print(f"\nResultados: {path} ({report['duration_s']:.0f} s)")


if __name__ == "__main__":
    main()
