from __future__ import annotations

import json
import os
import pickle
from typing import Any, Sequence

import numpy as np

from .feature_extractor import ExtractFeatures, FeatureVector


class RawVectorExtractor:
    """
    Extractor directo para columnas que ya almacenan vectores numéricos
    (por ejemplo embeddings faciales en BYTEA como float32 empaquetados,
    listas JSON o arreglos numpy) de dimensión arbitraria `dim`.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim debe ser un entero positivo, recibido {dim}")
        self.dim = int(dim)
        self.k = self.dim
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def modalities(self) -> tuple[str, ...]:
        return ("vector",)

    def fit(self, corpus: Sequence[Any]) -> RawVectorExtractor:
        return self

    def extract(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        vec = decode_raw_vector(obj, expected_dim=self.dim)
        return FeatureVector(data=vec, source_type=source_type or "vector", k=self.dim)

    def __call__(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        return self.extract(obj, source_type=source_type)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> RawVectorExtractor:
        with open(path, "rb") as f:
            extractor = pickle.load(f)
        if not isinstance(extractor, cls):
            raise ValueError(f"El archivo {path} no contiene un RawVectorExtractor")
        return extractor


def decode_raw_vector(obj: Any, expected_dim: int | None = None) -> np.ndarray:
    """Decodifica un vector float32 desde bytes/BYTEA, lista, ndarray o JSON string."""
    if obj is None:
        raise ValueError("El vector de entrada no puede ser None")

    if isinstance(obj, (bytes, bytearray, memoryview)):
        raw = bytes(obj)
        if len(raw) == 0 or len(raw) % 4 != 0:
            raise ValueError(
                f"Buffer BYTEA inválido para vector float32: longitud {len(raw)} bytes"
            )
        vec = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    elif isinstance(obj, str):
        text = obj.strip()
        if not text:
            raise ValueError("Cadena de vector vacía")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"No se pudo interpretar el vector JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("El vector en texto debe ser una lista JSON de números")
        vec = np.asarray(parsed, dtype=np.float32).reshape(-1)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        vec = np.asarray(obj, dtype=np.float32).reshape(-1)
    else:
        raise TypeError(f"Tipo no soportado como vector numérico: {type(obj)}")

    if vec.size == 0:
        raise ValueError("El vector numérico está vacío")
    if not np.all(np.isfinite(vec)):
        raise ValueError("El vector numérico contiene valores no finitos")
    if expected_dim is not None and vec.size != expected_dim:
        raise ValueError(
            f"La dimensión del vector ({vec.size}) no coincide con la esperada ({expected_dim})"
        )
    return vec


def encode_raw_vector(vector: Sequence[float] | np.ndarray) -> bytes:
    """Empaqueta un vector 1D como bytes float32 little-endian para columnas BYTEA."""
    vec = decode_raw_vector(vector)
    return vec.astype("<f4", copy=False).tobytes()


def load_vector_extractor(path: str) -> ExtractFeatures | RawVectorExtractor:
    """Carga un extractor vectorial guardado (.idx.extractor)."""
    with open(path, "rb") as f:
        extractor = pickle.load(f)
    if not isinstance(extractor, (ExtractFeatures, RawVectorExtractor)):
        raise ValueError(f"El archivo {path} no contiene un extractor vectorial válido")
    return extractor
