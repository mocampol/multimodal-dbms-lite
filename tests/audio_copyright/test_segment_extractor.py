"""
Vectores MFCC por segmento (apps/audio-copyright/features/segment_extractor.py).
Usa tonos sintéticos para no depender del corpus de FMA.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FEATURES_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from segment_extractor import SAMPLE_RATE, SegmentMFCCExtractor, segment_intervals  # noqa: E402


def tone(seconds, frequency, seed=0, noise=0.01):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    hiss = np.random.default_rng(seed).normal(0.0, noise, t.size)
    return (0.5 * np.sin(2 * np.pi * frequency * t) + hiss).astype(np.float32)


def melody(frequencies, seconds_each=4.0, seed=0, noise=0.01):
    """Una "canción" de 4 s por nota: cada una deja una huella MFCC distinta."""
    return np.concatenate([tone(seconds_each, f, seed + i, noise) for i, f in enumerate(frequencies)])


MELODIES = ([220, 440, 880], [330, 660, 990], [150, 300, 600])


@pytest.fixture(scope="module")
def trained():
    """Extractor entrenado con tres canciones de 12 s; devuelve también sus tramas."""
    extractor = SegmentMFCCExtractor(k=16)
    songs = [extractor.song_frames(melody(frequencies)) for frequencies in MELODIES]
    extractor.fit(songs)
    return extractor, songs


class TestSegmentIntervals:
    def test_a_30_second_clip_gives_11_overlapping_segments(self):
        intervals = segment_intervals(30.0)

        assert len(intervals) == 11
        assert intervals[0] == (0.0, 5.0) and intervals[1] == (2.5, 7.5) and intervals[-1] == (25.0, 30.0)
        assert all(end - start == 5.0 for start, end in intervals)

    def test_the_last_segment_ends_exactly_at_the_end_of_the_clip(self):
        intervals = segment_intervals(29.98)

        assert len(intervals) == 11
        assert intervals[-1] == (24.98, 29.98)

    def test_every_moment_of_the_song_is_covered(self):
        intervals = segment_intervals(17.3)

        for t in np.arange(0.0, 17.3, 0.1):
            assert any(start <= t < end for start, end in intervals)

    def test_a_clip_shorter_than_the_window_is_a_single_segment(self):
        assert segment_intervals(3.2) == [(0.0, 3.2)]

    @pytest.mark.parametrize("duration, window, hop, message", [
        (0.0, 5.0, 2.5, "duración"),
        (10.0, 0.0, 1.0, "positivos"),
        (10.0, 5.0, 6.0, "salto no puede"),
    ])
    def test_invalid_arguments(self, duration, window, hop, message):
        with pytest.raises(ValueError, match=message):
            segment_intervals(duration, window, hop)


class TestSegmentMFCCExtractor:
    def test_song_frames_keeps_the_real_duration(self):
        song = SegmentMFCCExtractor().song_frames(tone(12.0, 440))

        assert song.duration_s == pytest.approx(12.0)
        assert song.frames.shape[1] == 20

    def test_song_frames_reads_an_audio_file(self, tmp_path):
        path = tmp_path / "clip.wav"
        sf.write(str(path), tone(6.0, 440), SAMPLE_RATE)

        song = SegmentMFCCExtractor().song_frames(path)

        assert song.duration_s == pytest.approx(6.0, abs=0.01)

    def test_silent_or_unreadable_audio_is_an_error(self, tmp_path):
        extractor = SegmentMFCCExtractor()

        with pytest.raises(ValueError, match="silencio"):
            extractor.song_frames(np.zeros(6 * SAMPLE_RATE, dtype=np.float32))
        with pytest.raises(ValueError, match="No se pudo cargar"):
            extractor.song_frames(tmp_path / "no_existe.mp3")

    def test_transform_gives_one_unit_vector_per_segment_with_its_interval(self, trained):
        extractor, songs = trained

        segments = extractor.transform(songs[0])

        assert [(s.start_s, s.end_s) for s in segments] == segment_intervals(12.0)
        assert [s.segment_index for s in segments] == list(range(len(segments)))
        for segment in segments:
            assert segment.vector.shape == (16,)
            assert np.linalg.norm(segment.vector) == pytest.approx(1.0, abs=1e-5)

    def test_a_noisy_copy_matches_the_original_song_segment_by_segment(self, trained):
        extractor, songs = trained
        references = [(song_index, segment) for song_index, song in enumerate(songs)
                      for segment in extractor.transform(song)]
        copy = extractor.song_frames(melody(MELODIES[1], seed=99, noise=0.03))

        for query in extractor.transform(copy):
            best_song, best = max(references, key=lambda item: float(item[1].vector @ query.vector))
            assert best_song == 1
            assert best.start_s == query.start_s

    def test_transform_before_fit_is_an_error(self, trained):
        _, songs = trained

        with pytest.raises(ValueError, match="no está entrenado"):
            SegmentMFCCExtractor(k=16).transform(songs[0])

    def test_fit_without_songs_is_an_error(self):
        with pytest.raises(ValueError, match="al menos una canción"):
            SegmentMFCCExtractor().fit([])

    def test_save_and_load_give_the_same_vectors(self, trained, tmp_path):
        extractor, songs = trained
        path = tmp_path / "codebook" / "segments.pkl"

        extractor.save(path)
        loaded = SegmentMFCCExtractor.load(path)

        assert loaded.model_name == extractor.model_name == "mfcc20_boaw16_w5_h2.5"
        for original, restored in zip(extractor.transform(songs[2]), loaded.transform(songs[2])):
            np.testing.assert_allclose(original.vector, restored.vector)
