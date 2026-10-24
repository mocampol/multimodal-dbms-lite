from __future__ import annotations

import io
import os
from typing import Union
import librosa
import numpy as np
import soundfile as sf


class MFCCExtractor:
    """Extrae coeficientes MFCC a partir de señales de audio."""

    def __init__(self, n_mfcc: int = 20, sr: int = 22050) -> None:
        self.n_mfcc = n_mfcc
        self.sr = sr

    def extract(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Extrae tramas de coeficientes MFCC del audio provisto.

        Returns:
            np.ndarray: matriz de forma (T, n_mfcc) con coeficientes float32 por trama.

        Raises:
            ValueError: Si el audio está vacío, silenciado o es inválido.
        """
        y, sr = self._load_audio(source)
        if y is None or len(y) == 0:
            raise ValueError("La señal de audio está vacía")

        # Verificar si es completamente silencioso
        if np.max(np.abs(y)) < 1e-6:
            raise ValueError("La señal de audio no contiene energía acústica (silencio total)")

        # Calcular MFCC -> forma (n_mfcc, T)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        # Transponer para obtener (T, n_mfcc) donde cada fila es una observación temporal
        mfcc_frames = mfcc.T.astype(np.float32)

        if len(mfcc_frames) == 0:
            raise ValueError("No se pudieron generar tramas MFCC suficientes para el audio")

        return mfcc_frames

    def _load_audio(self, source: Union[str, bytes, np.ndarray]) -> tuple[np.ndarray, int]:
        if isinstance(source, np.ndarray):
            y = source.astype(np.float32)
            if y.ndim > 1:
                # Convertir a mono promediando canales
                y = np.mean(y, axis=0) if y.shape[0] < y.shape[1] else np.mean(y, axis=1)
            return y, self.sr

        if isinstance(source, str):
            if not os.path.exists(source):
                raise ValueError(f"El archivo de audio no existe: {source}")
            try:
                y, sr = librosa.load(source, sr=self.sr, mono=True)
                return y, sr
            except Exception as e:
                raise ValueError(f"No se pudo cargar el audio desde el archivo: {source}. Detalle: {e}")

        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise ValueError("Buffer de bytes de audio vacío")
            try:
                buf = io.BytesIO(source)
                data, sr = sf.read(buf, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != self.sr:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.sr)
                return data, self.sr
            except Exception as e:
                raise ValueError(f"No se pudo decodificar el audio desde bytes. Detalle: {e}")

        raise TypeError(f"Tipo de entrada no soportado para audio: {type(source)}")
