from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence, Union
import numpy as np

from .sift_extractor import SIFTExtractor
from .mfcc_extractor import MFCCExtractor
from .quantizer import Quantizer
from .bow_tfidf import BoWTFIDF


@dataclass
class FeatureVector:
    """Representación vectorial normalizada de dimensión K para búsqueda de similitud."""
    data: np.ndarray
    source_type: str
    k: int

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 1:
            self.data = self.data.flatten()
        if len(self.data) != self.k:
            raise ValueError(f"La dimensión de los datos ({len(self.data)}) no coincide con k ({self.k})")

    def to_list(self) -> list[float]:
        return self.data.tolist()

    def __len__(self) -> int:
        return self.k

    def __getitem__(self, idx: int) -> float:
        return float(self.data[idx])


class ExtractFeatures:
    """
    Extractor de características multimedia unificado para imágenes, audio y vectores numéricos.
    Produce representaciones comparables de dimensión K ponderadas por TF-IDF.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # Permite usar ExtractFeatures(objeto) directamente o ExtractFeatures(k=64)
        if len(args) == 1 and not isinstance(args[0], int) and not kwargs:
            instance = super().__new__(cls)
            instance.__init__(k=64)
            return instance.extract(args[0])
        return super().__new__(cls)

    def __init__(self, k: int = 64, quantizer: str = "kmeans") -> None:
        # Prevenir re-inicialización en invocaciones directas de __new__
        if hasattr(self, "_initialized") and self._initialized:
            return
        if k <= 0:
            raise ValueError(f"k debe ser un entero positivo, recibido {k}")

        self.k = k
        self.quantizer_type = quantizer
        self.sift = SIFTExtractor()
        self.mfcc = MFCCExtractor()
        self.quantizer = Quantizer(k=k)
        self.bow_tfidf = BoWTFIDF(k=k)
        self._is_fitted = False
        self._initialized = True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def detect_type(self, obj: Any) -> str:
        """Detecta el tipo de entrada: 'image', 'audio' o 'numeric'."""
        if obj is None:
            raise ValueError("El objeto de entrada no puede ser None")

        if isinstance(obj, str):
            if not obj.strip():
                raise ValueError("Ruta de archivo vacía")
            ext = os.path.splitext(obj.lower())[1]
            if ext in self.IMAGE_EXTENSIONS:
                return "image"
            if ext in self.AUDIO_EXTENSIONS:
                return "audio"
            raise ValueError(f"Extensión de archivo no reconocida para multimedia: '{ext}'")

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise ValueError("Lista/tupla numérica vacía")
            # Verificar si los elementos son numéricos
            if all(isinstance(x, (int, float, np.number)) for x in obj):
                return "numeric"
            raise ValueError("La lista o tupla contiene elementos no numéricos")

        if isinstance(obj, np.ndarray):
            if obj.size == 0:
                raise ValueError("Array numpy vacío")
            if obj.ndim in (2, 3) and obj.dtype == np.uint8:
                return "image"
            if obj.ndim == 1:
                return "numeric"
            if obj.ndim in (2, 3):
                # Matriz bidimensional o tridimensional numérica o imagen float
                return "image"

        raise TypeError(f"Tipo de objeto no soportado: {type(obj)}")

    def fit(self, corpus: Sequence[Any]) -> ExtractFeatures:
        """
        Entrena el cuantizador y las frecuencias de documentos TF-IDF sobre una colección de objetos.
        """
        if not corpus:
            raise ValueError("El corpus para fit() no puede estar vacío")

        all_descriptors: list[np.ndarray] = []
        assignments_per_doc: list[np.ndarray] = []
        docs_descriptors: list[np.ndarray] = []

        for item in corpus:
            modality = self.detect_type(item)
            if modality == "image":
                desc = self.sift.extract(item)
            elif modality == "audio":
                desc = self.mfcc.extract(item)
            elif modality == "numeric":
                vec = np.asarray(item, dtype=np.float32).flatten()
                # Expandir como descriptor 2D para cuantización
                desc = vec.reshape(-1, 1)
            else:
                continue

            all_descriptors.append(desc)
            docs_descriptors.append(desc)

        if not all_descriptors:
            raise ValueError("No se pudieron extraer descriptores válidos del corpus")

        combined = np.vstack(all_descriptors)
        self.quantizer.fit(combined)

        for desc in docs_descriptors:
            assigned = self.quantizer.predict(desc)
            assignments_per_doc.append(assigned)

        self.bow_tfidf.fit(assignments_per_doc)
        self._is_fitted = True
        return self

    def extract(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        """
        Extrae y cuantiza las características de un objeto multimedia, generando un FeatureVector de dimensión K.
        """
        if obj is None:
            raise ValueError("El objeto de entrada no puede ser None")

        modality = source_type or self.detect_type(obj)

        if modality == "image":
            descriptors = self.sift.extract(obj)
            if not self.quantizer.is_fitted:
                self.quantizer.fit(descriptors)
                self.bow_tfidf.fit([self.quantizer.predict(descriptors)])
                self._is_fitted = True

            assignments = self.quantizer.predict(descriptors)
            vec = self.bow_tfidf.compute(assignments)
            return FeatureVector(data=vec, source_type="image", k=self.k)

        elif modality == "audio":
            frames = self.mfcc.extract(obj)
            if not self.quantizer.is_fitted:
                self.quantizer.fit(frames)
                self.bow_tfidf.fit([self.quantizer.predict(frames)])
                self._is_fitted = True

            assignments = self.quantizer.predict(frames)
            vec = self.bow_tfidf.compute(assignments)
            return FeatureVector(data=vec, source_type="audio", k=self.k)

        elif modality == "numeric":
            raw = np.asarray(obj, dtype=np.float32).flatten()
            if len(raw) == 0:
                raise ValueError("El vector numérico está vacío")

            # Adaptar a dimensión K
            if len(raw) == self.k:
                vec = raw.copy()
            elif len(raw) > 1:
                # Interpolación suave para preservar la curva de frecuencias
                x_old = np.linspace(0.0, 1.0, len(raw))
                x_new = np.linspace(0.0, 1.0, self.k)
                vec = np.interp(x_new, x_old, raw).astype(np.float32)
            else:
                # Vector de 1 solo elemento -> expandir
                vec = np.full(self.k, raw[0], dtype=np.float32)

            # Normalización L2
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.zeros(self.k, dtype=np.float32)

            return FeatureVector(data=vec.astype(np.float32), source_type="numeric", k=self.k)

        else:
            raise ValueError(f"Modalidad desconocida: {modality}")

    def __call__(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        return self.extract(obj, source_type=source_type)

    def transform(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        return self.extract(obj, source_type=source_type)

    def fit_transform(self, corpus: Sequence[Any], obj: Any) -> FeatureVector:
        self.fit(corpus)
        return self.extract(obj)
