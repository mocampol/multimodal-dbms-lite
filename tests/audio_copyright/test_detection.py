"""
Agrupación de candidatos en coincidencias (apps/audio-copyright/backend/detection.py).
Usa resultados de búsqueda sintéticos con la forma de SegmentSearcher.search_audio.
"""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright"
for folder in ("features", "backend"):
    sys.path.insert(0, str(APP_DIR / folder))

from detection import NO_MATCH, POSSIBLE_MATCH, DetectionSettings, detect  # noqa: E402

HOP, WINDOW = 2.5, 5.0
SETTINGS = DetectionSettings(threshold=0.9, min_segments=2, max_gap=1, offset_tolerance_s=HOP)


def candidate(song_id, reference_index, score):
    """Un segmento de la canción song_id que empieza en reference_index pasos."""
    start = reference_index * HOP
    return {"song_id": song_id, "title": f"Canción {song_id}", "artist": f"Artista {song_id}",
            "license_family": "CC BY", "track_url": f"http://fma/{song_id}",
            "start_s": start, "end_s": start + WINDOW, "score": score}


def search(*candidates_per_segment):
    """Un audio subido con un segmento por argumento, cada uno con sus candidatos."""
    segments = [{"segment_index": index, "start_s": index * HOP, "end_s": index * HOP + WINDOW,
                 "candidates": list(candidates)}
                for index, candidates in enumerate(candidates_per_segment)]
    duration = segments[-1]["end_s"] if segments else 0.0
    return {"duration_s": duration, "segments": segments, "index_type": "hnsw", "metric": "cosine", "k": 5,
            "used_index": True, "timings_ms": {"total_ms": 1.0}}


def noise(song_id=99, score=0.5):
    return [candidate(song_id, 0, score)]


# Coincidencias

def test_a_copy_becomes_one_match_with_both_intervals_and_its_similarity():
    # Los segmentos 1 a 4 del audio subido copian los segmentos 4 a 7 de la canción 2.
    result = detect(search(noise(), [candidate(2, 4, 0.95)], [candidate(2, 5, 0.93)],
                           [candidate(2, 6, 0.97)], [candidate(2, 7, 0.91)], noise()), SETTINGS)

    assert result["verdict"] == POSSIBLE_MATCH
    [match] = result["matches"]
    assert (match["song_id"], match["title"], match["artist"]) == (2, "Canción 2", "Artista 2")
    assert (match["query_start_s"], match["query_end_s"]) == (2.5, 15.0)
    assert (match["reference_start_s"], match["reference_end_s"]) == (10.0, 22.5)
    assert match["segments"] == 4
    assert match["similarity_pct"] == 94.0 and match["max_similarity_pct"] == 97.0


def test_the_timeline_marks_every_segment_and_its_match():
    result = detect(search(noise(), [candidate(2, 4, 0.95)], [candidate(2, 5, 0.93)], noise()), SETTINGS)

    timeline = result["timeline"]
    assert [segment["match_id"] for segment in timeline] == [None, 1, 1, None]
    assert [segment["above_threshold"] for segment in timeline] == [False, True, True, False]
    assert timeline[0]["best_similarity_pct"] == 50.0


def test_a_copy_shifted_half_a_step_still_counts():
    # Desplazado 1.25 s: el desfase varía entre segmentos, dentro de la tolerancia.
    shifted = search(*[[{**candidate(2, index + 3, 0.92), "start_s": (index + 3) * HOP + (1.25 if index % 2 else -1.25)}]
                       for index in range(4)])

    assert detect(shifted, SETTINGS)["matches"][0]["segments"] == 4


def test_one_weak_segment_inside_a_copy_does_not_split_it():
    result = detect(search([candidate(2, 4, 0.95)], [candidate(2, 5, 0.60)], [candidate(2, 6, 0.94)]), SETTINGS)

    [match] = result["matches"]
    assert (match["query_start_s"], match["query_end_s"], match["segments"]) == (0.0, 10.0, 2)


def test_two_weak_segments_in_a_row_split_the_copy():
    result = detect(search([candidate(2, 4, 0.95)], [candidate(2, 5, 0.94)], noise(), noise(),
                           [candidate(2, 8, 0.95)], [candidate(2, 9, 0.94)]), SETTINGS)

    assert [(m["query_start_s"], m["query_end_s"]) for m in result["matches"]] == [(0.0, 7.5), (10.0, 17.5)]


def test_neighbouring_segments_of_the_same_copy_do_not_duplicate_it():
    # Por el solape de las ventanas, cada segmento también se parece al siguiente de la original.
    result = detect(search(*[[candidate(2, index + 4, 0.95), candidate(2, index + 5, 0.91)] for index in range(4)]),
                    SETTINGS)

    assert len(result["matches"]) == 1
    assert result["matches"][0]["similarity_pct"] == 95.0


def test_two_songs_used_in_sequence_are_two_matches():
    result = detect(search([candidate(2, 0, 0.95)], [candidate(2, 1, 0.94)],
                           [candidate(7, 3, 0.92)], [candidate(7, 4, 0.93)]), SETTINGS)

    assert [m["song_id"] for m in result["matches"]] == [2, 7]
    assert result["coverage_pct"] == 100.0


# Sin coincidencias

def test_nothing_above_the_threshold_is_an_explicit_no_match():
    result = detect(search(noise(score=0.85), noise(score=0.89), noise(score=0.7)), SETTINGS)

    assert result["verdict"] == NO_MATCH
    assert result["matches"] == []
    assert "Sin coincidencias" in result["message"] and "90%" in result["message"]
    assert result["coverage_pct"] == 0.0
    assert "no son una determinación legal" in result["disclaimer"]


def test_a_single_similar_segment_is_not_enough():
    assert detect(search(noise(), [candidate(2, 4, 0.99)], noise()), SETTINGS)["verdict"] == NO_MATCH


def test_similar_segments_that_do_not_advance_together_are_not_a_copy():
    # Misma canción, pero cada segmento salta a otra parte de la original: no es una copia continua.
    result = detect(search([candidate(2, 0, 0.95)], [candidate(2, 5, 0.95)], [candidate(2, 9, 0.95)]), SETTINGS)

    assert result["verdict"] == NO_MATCH


def test_an_audio_without_candidates_has_no_matches():
    result = detect(search([], []), SETTINGS)

    assert result["verdict"] == NO_MATCH
    assert result["timeline"][0]["best_song_id"] is None


# Configuración

def test_settings_are_reported_with_the_result():
    result = detect(search([candidate(2, 0, 0.95)], [candidate(2, 1, 0.94)]), SETTINGS)

    assert result["settings"]["threshold"] == 0.9
    assert result["search"] == {"index_type": "hnsw", "metric": "cosine", "k": 5, "used_index": True}


@pytest.mark.parametrize("values", [{"threshold": 0.0}, {"threshold": 1.5}, {"min_segments": 0}, {"max_gap": -1}])
def test_invalid_settings_are_rejected(values):
    with pytest.raises(ValueError):
        DetectionSettings(**values)
