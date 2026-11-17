"""
apps/audio-copyright/features/segment_extractor.py — Vectores MFCC por segmento temporal.

Sigue el mismo pipeline que ExtractFeatures (src/index/multimedia): MFCC por trama,
K-Means para las palabras de audio, y un histograma con TF-IDF y norma L2. La
diferencia es que no produce un vector por canción sino uno por cada segmento, para
poder decir en qué intervalo de la canción está una coincidencia.

Cada canción se decodifica, se pasa por MFCC y se cuantiza una sola vez: los
segmentos son rebanadas de esas matrices, así que solaparlos no repite trabajo.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import librosa
import numpy as np

from index.multimedia import BoWTFIDF, MFCCExtractor, Quantizer

SAMPLE_RATE = 22050
N_MFCC = 20
HOP_LENGTH = 512  # salto entre tramas de librosa.feature.mfcc, que MFCCExtractor no cambia
WINDOW_S = 5.0
HOP_S = 2.5
CODEBOOK_SIZE = 64
MAX_FRAMES_PER_SONG = 400  # tramas por canción para entrenar K-Means


@dataclass(frozen=True)
class SongFrames:
    """Tramas MFCC de una canción completa, de forma (T, N_MFCC), y su duración real."""

    frames: np.ndarray
    duration_s: float


@dataclass(frozen=True)
class SegmentVector:
    """Vector de un segmento y el intervalo [start_s, end_s) de la canción que cubre."""

    segment_index: int
    start_s: float
    end_s: float
    vector: np.ndarray


def segment_intervals(duration_s: float, window_s: float = WINDOW_S, hop_s: float = HOP_S) -> list[tuple[float, float]]:
    """
    Intervalos de window_s segundos que avanzan hop_s hasta cubrir la canción.

    Una canción más corta que la ventana es un solo segmento. Si la última ventana no
    llega al final, se agrega una que termina justo en el final, para que ningún tramo
    quede sin vector.
    """
    if duration_s <= 0:
        raise ValueError(f"La duración debe ser positiva, recibida {duration_s}")
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("La ventana y el salto deben ser positivos")
    if hop_s > window_s:
        raise ValueError("El salto no puede ser mayor que la ventana: quedarían tramos sin cubrir")

    if duration_s <= window_s:
        return [(0.0, round(duration_s, 3))]

    starts = np.arange(0.0, duration_s - window_s + 1e-9, hop_s)
    intervals = [(round(float(start), 3), round(float(start) + window_s, 3)) for start in starts]
    if intervals[-1][1] < duration_s - 1e-9:
        intervals.append((round(duration_s - window_s, 3), round(duration_s, 3)))
    return intervals


class SegmentMFCCExtractor:
    """
    Convierte una canción en un vector de dimensión k por segmento.

    Uso: song_frames() por canción, fit() una vez con todo el corpus y transform() por
    canción. El codebook y el IDF se guardan con save() para que los audios de consulta
    se proyecten en el mismo espacio que los de referencia.
    """

    def __init__(
        self,
        k: int = CODEBOOK_SIZE,
        window_s: float = WINDOW_S,
        hop_s: float = HOP_S,
        random_state: int = 42,
    ) -> None:
        segment_intervals(window_s, window_s, hop_s)  # valida ventana y salto
        self.k = k
        self.window_s = window_s
        self.hop_s = hop_s
        self.random_state = random_state
        self.mfcc = MFCCExtractor(n_mfcc=N_MFCC, sr=SAMPLE_RATE)
        self.quantizer = Quantizer(k=k, random_state=random_state)
        self.bow_tfidf = BoWTFIDF(k=k)

    @property
    def model_name(self) -> str:
        """Identifica cómo se calcularon los vectores; si cambia, no son comparables."""
        return f"mfcc{N_MFCC}_boaw{self.k}_w{self.window_s:g}_h{self.hop_s:g}"

    @property
    def is_fitted(self) -> bool:
        return self.quantizer.is_fitted and self.bow_tfidf.idf_ is not None

    def song_frames(self, source: str | Path | np.ndarray) -> SongFrames:
        """Decodifica el audio (o toma la señal mono ya cargada) y calcula sus tramas MFCC."""
        if isinstance(source, np.ndarray):
            signal = np.asarray(source, dtype=np.float32).reshape(-1)
        else:
            try:
                signal, _ = librosa.load(str(source), sr=SAMPLE_RATE, mono=True)
            except Exception as error:
                raise ValueError(f"No se pudo cargar el audio {source}: {error}") from error
        frames = self.mfcc.extract(signal)  # rechaza audio vacío o en silencio total
        return SongFrames(frames=frames, duration_s=len(signal) / SAMPLE_RATE)

    def _segments(self, song: SongFrames) -> Iterator[tuple[int, float, float, slice]]:
        """Cada segmento y las tramas que le tocan; siempre al menos una."""
        n_frames = len(song.frames)
        for index, (start, end) in enumerate(segment_intervals(song.duration_s, self.window_s, self.hop_s)):
            first = min(int(round(start * SAMPLE_RATE / HOP_LENGTH)), n_frames - 1)
            last = max(first + 1, min(int(round(end * SAMPLE_RATE / HOP_LENGTH)), n_frames))
            yield index, start, end, slice(first, last)

    def fit(self, songs: Sequence[SongFrames], max_frames_per_song: int = MAX_FRAMES_PER_SONG) -> SegmentMFCCExtractor:
        """
        Entrena el codebook con una muestra de tramas de cada canción, para acotar
        el costo de K-Means, y el IDF con todos los segmentos, que son los
        "documentos" que después se comparan.
        """
        if not songs:
            raise ValueError("Se necesita al menos una canción para entrenar el codebook")
        if max_frames_per_song <= 0:
            raise ValueError("max_frames_per_song debe ser positivo")

        rng = np.random.default_rng(self.random_state)
        sample = []
        for song in songs:
            frames = song.frames
            if len(frames) > max_frames_per_song:
                frames = frames[rng.choice(len(frames), max_frames_per_song, replace=False)]
            sample.append(frames)
        self.quantizer.fit(np.vstack(sample))

        documents = []
        for song in songs:
            words = self.quantizer.predict(song.frames)
            documents.extend(words[frames] for _, _, _, frames in self._segments(song))
        self.bow_tfidf.fit(documents)
        return self

    def transform(self, song: SongFrames) -> list[SegmentVector]:
        """Un vector L2-normalizado de dimensión k por segmento, con su intervalo."""
        if not self.is_fitted:
            raise ValueError("El extractor no está entrenado: llama a fit() o carga uno con load()")
        words = self.quantizer.predict(song.frames)  # una sola vez por canción
        return [
            SegmentVector(index, start, end, self.bow_tfidf.compute(words[frames]))
            for index, start, end, frames in self._segments(song)
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump(self, handle)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> SegmentMFCCExtractor:
        with open(path, "rb") as handle:
            extractor = pickle.load(handle)
        if not isinstance(extractor, cls):
            raise ValueError(f"El archivo {path} no contiene un SegmentMFCCExtractor")
        return extractor
