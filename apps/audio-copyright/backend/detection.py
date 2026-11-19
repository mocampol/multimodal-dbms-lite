"""
apps/audio-copyright/backend/detection.py — Convierte los candidatos por segmento de
segment_search.py en coincidencias: rangos del audio subido que se parecen a un rango de
una canción de referencia (issue #113).

Cómo se agrupa:
    1. De cada segmento del audio subido se toman los candidatos con score >= umbral que,
       además, superan al mejor candidato de otra canción por al menos `margin`. Como
       cualquier canción se parece bastante a cualquier otra con estos vectores, lo que
       distingue a una copia es que su candidato les gana claramente a los demás.
    2. Dos segmentos seguidos se unen si apuntan a la misma canción y el desfase entre
       ambos audios (inicio en la original - inicio en el subido) se mantiene, con un
       margen de un paso. Una copia avanza igual en los dos; dos segmentos parecidos por
       casualidad casi nunca lo hacen.
    3. Se puede saltar un segmento débil dentro de una cadena (max_gap).
    4. Una cadena con al menos min_segments segmentos es una coincidencia. Como los
       segmentos se solapan a la mitad, una copia también activa los segmentos vecinos de
       la original y deja cadenas paralelas a ±1 paso: de las cadenas de una misma canción
       que se solapan en el audio subido se queda la más fuerte.

El resultado es una posible coincidencia, no una determinación legal.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for path in (APP_DIR.parents[1] / "src", APP_DIR / "features"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from segment_extractor import HOP_S  # noqa: E402

POSSIBLE_MATCH = "posible_coincidencia"
NO_MATCH = "sin_coincidencias"
DISCLAIMER = ("Son posibles coincidencias de audio según su similitud; no son una determinación legal "
              "de infracción de copyright.")


@dataclass(frozen=True)
class DetectionSettings:
    """Reglas de la agrupación; cada una se puede cambiar por variable de entorno.

    El umbral, el mínimo de segmentos y el margen salen de evaluation/calibrate_threshold.py:
    son los que dan más recall sin pasar del 5% de falsos positivos con el corpus real."""

    threshold: float = float(os.getenv("AUDIO_MATCH_THRESHOLD", "0.70"))
    min_segments: int = int(os.getenv("AUDIO_MIN_SEGMENTS", "2"))
    margin: float = float(os.getenv("AUDIO_MATCH_MARGIN", "0.03"))
    max_gap: int = int(os.getenv("AUDIO_MAX_GAP", "1"))
    offset_tolerance_s: float = float(os.getenv("AUDIO_OFFSET_TOLERANCE_S", str(HOP_S)))

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"El umbral debe estar entre 0 y 1, recibido {self.threshold}")
        if self.min_segments < 1 or self.max_gap < 0 or self.offset_tolerance_s < 0 or self.margin < 0:
            raise ValueError("min_segments debe ser >= 1, y max_gap, offset_tolerance_s y margin, >= 0")


@dataclass(frozen=True)
class Hit:
    """Un candidato sobre el umbral: un segmento del audio subido parecido a uno de la base."""

    segment_index: int
    query_start_s: float
    query_end_s: float
    song_id: int
    reference_start_s: float
    reference_end_s: float
    score: float

    @property
    def offset_s(self) -> float:
        return self.reference_start_s - self.query_start_s


@dataclass
class Chain:
    song_id: int
    hits: list[Hit] = field(default_factory=list)

    @property
    def last(self) -> Hit:
        return self.hits[-1]

    @property
    def similarity(self) -> float:
        return sum(hit.score for hit in self.hits) / len(self.hits)

    def overlaps(self, other: Chain) -> bool:
        return (self.hits[0].query_start_s < other.last.query_end_s
                and other.hits[0].query_start_s < self.last.query_end_s)


# =============================================================================
# Agrupación
# =============================================================================

def _beats_other_songs(candidate: dict, candidates: list[dict], margin: float) -> bool:
    """Si el candidato supera por margin al mejor de otra canción. Si entre los candidatos
    no hay ninguna otra canción, no hay con qué compararlo y cuenta como que sí."""
    others = [other["score"] for other in candidates if other["song_id"] != candidate["song_id"]]
    return not others or candidate["score"] - max(others) >= margin - 1e-9


def hits_above(segments: list[dict], threshold: float, margin: float = 0.0) -> list[list[Hit]]:
    """Por segmento del audio subido, en orden, sus candidatos con score >= umbral que
    superan por margin al mejor candidato de otra canción."""
    by_segment = []
    for segment in sorted(segments, key=lambda s: s["segment_index"]):
        candidates = segment["candidates"]
        by_segment.append([
            Hit(segment["segment_index"], segment["start_s"], segment["end_s"], candidate["song_id"],
                candidate["start_s"], candidate["end_s"], candidate["score"])
            for candidate in candidates
            if candidate["score"] >= threshold and _beats_other_songs(candidate, candidates, margin)
        ])
    return by_segment


def build_chains(hits_by_segment: list[list[Hit]], settings: DetectionSettings) -> list[Chain]:
    """Une los aciertos de segmentos seguidos con la misma canción y el mismo desfase."""
    open_chains: list[Chain] = []
    closed: list[Chain] = []
    for hits in hits_by_segment:
        if not hits:
            continue
        index = hits[0].segment_index
        # Las cadenas que ya saltaron más de max_gap segmentos no pueden seguir.
        still_open = []
        for chain in open_chains:
            (still_open if index - chain.last.segment_index <= settings.max_gap + 1 else closed).append(chain)
        open_chains = still_open

        unused = list(hits)
        for chain in open_chains:
            fitting = [hit for hit in unused if hit.song_id == chain.song_id
                       and abs(hit.offset_s - chain.last.offset_s) <= settings.offset_tolerance_s + 1e-9]
            if fitting:
                best = min(fitting, key=lambda hit: (abs(hit.offset_s - chain.last.offset_s), -hit.score))
                chain.hits.append(best)
                unused.remove(best)
        open_chains += [Chain(hit.song_id, [hit]) for hit in unused]
    return closed + open_chains


def select_chains(chains: list[Chain], settings: DetectionSettings) -> list[Chain]:
    """Las cadenas largas, sin duplicados paralelos de una misma canción."""
    long_enough = [chain for chain in chains if len(chain.hits) >= settings.min_segments]
    long_enough.sort(key=lambda chain: (-len(chain.hits), -chain.similarity, chain.hits[0].segment_index))
    selected: list[Chain] = []
    for chain in long_enough:
        if not any(kept.song_id == chain.song_id and kept.overlaps(chain) for kept in selected):
            selected.append(chain)
    return sorted(selected, key=lambda chain: (chain.hits[0].query_start_s, -chain.similarity))


# =============================================================================
# Resultado
# =============================================================================

def _song_info(segments: list[dict]) -> dict[int, dict]:
    """Título, artista, licencia y URL de cada canción, tomados de los candidatos."""
    info = {}
    for segment in segments:
        for candidate in segment["candidates"]:
            info.setdefault(candidate["song_id"], {
                key: candidate.get(key) for key in ("title", "artist", "license_family", "track_url")
            })
    return info


def _coverage(chains: list[Chain], duration_s: float) -> float:
    """Fracción del audio subido dentro de alguna coincidencia, sin contar dos veces los solapes."""
    intervals = sorted((chain.hits[0].query_start_s, chain.last.query_end_s) for chain in chains)
    covered, end = 0.0, float("-inf")
    for start, stop in intervals:
        if stop > end:
            covered += stop - max(start, end)
            end = stop
    return covered / duration_s if duration_s > 0 else 0.0


def detect(search: dict, settings: DetectionSettings | None = None) -> dict:
    """Agrupa el resultado de SegmentSearcher.search_audio y decide si hay coincidencias."""
    settings = settings or DetectionSettings()
    segments = search["segments"]
    songs = _song_info(segments)
    chains = select_chains(build_chains(hits_above(segments, settings.threshold, settings.margin), settings), settings)

    matches, match_of_segment = [], {}
    for number, chain in enumerate(chains, start=1):
        scores = [hit.score for hit in chain.hits]
        matches.append({
            "match_id": number,
            "song_id": chain.song_id,
            **songs.get(chain.song_id, {}),
            "query_start_s": chain.hits[0].query_start_s,
            "query_end_s": chain.last.query_end_s,
            "reference_start_s": min(hit.reference_start_s for hit in chain.hits),
            "reference_end_s": max(hit.reference_end_s for hit in chain.hits),
            "similarity_pct": round(100.0 * sum(scores) / len(scores), 1),
            "max_similarity_pct": round(100.0 * max(scores), 1),
            "segments": len(chain.hits),
        })
        for hit in chain.hits:
            match_of_segment.setdefault(hit.segment_index, number)

    timeline = []
    for segment in sorted(segments, key=lambda s: s["segment_index"]):
        best = max(segment["candidates"], key=lambda candidate: candidate["score"], default=None)
        timeline.append({
            "segment_index": segment["segment_index"],
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
            "best_song_id": best["song_id"] if best else None,
            "best_title": best.get("title") if best else None,
            "best_artist": best.get("artist") if best else None,
            "best_similarity_pct": round(100.0 * best["score"], 1) if best else None,
            "above_threshold": bool(best and best["score"] >= settings.threshold),
            "match_id": match_of_segment.get(segment["segment_index"]),
        })

    verdict = POSSIBLE_MATCH if matches else NO_MATCH
    return {
        "verdict": verdict,
        "message": ("Se encontró 1 posible coincidencia." if len(matches) == 1 else
                    f"Se encontraron {len(matches)} posibles coincidencias." if matches else
                    f"Sin coincidencias: ningún tramo de al menos {settings.min_segments} segmentos seguidos "
                    f"supera el {settings.threshold:.0%} de similitud con una misma canción."),
        "disclaimer": DISCLAIMER,
        "duration_s": search["duration_s"],
        "coverage_pct": round(100.0 * _coverage(chains, search["duration_s"]), 1),
        "matches": matches,
        "timeline": timeline,
        "settings": asdict(settings),
        "search": {key: search[key] for key in ("index_type", "metric", "k", "query_shifts_s", "used_index", "model_name") if key in search},
        "timings_ms": search.get("timings_ms", {}),
    }
