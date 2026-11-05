from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
from typing import Any, Sequence
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
    Extractor multimedia con codebooks e IDF independientes para cada modalidad.
    Produce vectores de dimensión K compatibles con índices de similitud.
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
        self._quantizers: dict[str, Quantizer] = {}
        self._bow_tfidf_by_modality: dict[str, BoWTFIDF] = {}
        self._is_fitted = False
        self._initialized = True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def modalities(self) -> tuple[str, ...]:
        """Modalidades con codebook propio. extract() agrega una al ver una nueva."""
        return tuple(self._quantizers)

    def _extract_descriptors(self, obj: Any, modality: str) -> np.ndarray:
        if modality == "image":
            return self.sift.extract(obj)
        if modality == "audio":
            return self.mfcc.extract(obj)
        if modality == "numeric":
            values = np.asarray(obj, dtype=np.float32).reshape(-1)
            if values.size == 0:
                raise ValueError("El vector numérico está vacío")
            if not np.all(np.isfinite(values)):
                raise ValueError("El vector numérico contiene valores no finitos")
            return values.reshape(-1, 1)
        raise ValueError(f"Modalidad desconocida: {modality}")

    def _train_modality(self, documents: Sequence[np.ndarray]) -> tuple[Quantizer, BoWTFIDF]:
        quantizer = Quantizer(k=self.k)
        quantizer.fit(np.vstack(documents))
        assignments = [quantizer.predict(descriptors) for descriptors in documents]
        bow_tfidf = BoWTFIDF(k=self.k).fit(assignments)
        return quantizer, bow_tfidf

    def _fit_modality(self, modality: str, documents: Sequence[np.ndarray]) -> None:
        quantizer, bow_tfidf = self._train_modality(documents)
        self._quantizers[modality] = quantizer
        self._bow_tfidf_by_modality[modality] = bow_tfidf
        self.quantizer = quantizer
        self.bow_tfidf = bow_tfidf

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
        if len(corpus) == 0:
            raise ValueError("El corpus para fit() no puede estar vacío")

        descriptors_by_modality: dict[str, list[np.ndarray]] = {}

        for item in corpus:
            modality = self.detect_type(item)
            descriptors = self._extract_descriptors(item, modality)
            descriptors_by_modality.setdefault(modality, []).append(descriptors)

        if not descriptors_by_modality:
            raise ValueError("No se pudieron extraer descriptores válidos del corpus")

        quantizers: dict[str, Quantizer] = {}
        bow_tfidf_by_modality: dict[str, BoWTFIDF] = {}
        for modality, documents in descriptors_by_modality.items():
            quantizer, bow_tfidf = self._train_modality(documents)
            quantizers[modality] = quantizer
            bow_tfidf_by_modality[modality] = bow_tfidf
        self._quantizers = quantizers
        self._bow_tfidf_by_modality = bow_tfidf_by_modality
        first_modality = next(iter(quantizers))
        self.quantizer = quantizers[first_modality]
        self.bow_tfidf = bow_tfidf_by_modality[first_modality]
        self._is_fitted = True
        return self

    def extract(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        """
        Extrae y cuantiza las características de un objeto multimedia, generando un FeatureVector de dimensión K.
        """
        modality = source_type or self.detect_type(obj)
        descriptors = self._extract_descriptors(obj, modality)
        if modality not in self._quantizers:
            self._fit_modality(modality, [descriptors])
            self._is_fitted = True

        assignments = self._quantizers[modality].predict(descriptors)
        vec = self._bow_tfidf_by_modality[modality].compute(assignments)
        return FeatureVector(data=vec, source_type=modality, k=self.k)

    def __call__(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        return self.extract(obj, source_type=source_type)

    def transform(self, obj: Any, source_type: str | None = None) -> FeatureVector:
        return self.extract(obj, source_type=source_type)

    def fit_transform(self, corpus: Sequence[Any], obj: Any) -> FeatureVector:
        self.fit(corpus)
        return self.extract(obj)

    def save(self, path: str) -> None:
        """Guarda el extractor con sus codebooks e IDF, para extraer después sin reentrenar."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> ExtractFeatures:
        with open(path, "rb") as f:
            extractor = pickle.load(f)
        if not isinstance(extractor, cls):
            raise ValueError(f"El archivo {path} no contiene un ExtractFeatures")
        return extractor
