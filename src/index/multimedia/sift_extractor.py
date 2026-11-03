from __future__ import annotations

import io
import os
from typing import Union
import cv2
import numpy as np


class SIFTExtractor:
    """Extrae descriptores SIFT (128-d) a partir de imágenes."""

    def __init__(self, n_features: int = 500) -> None:
        self.n_features = n_features
        self._sift = cv2.SIFT_create(nfeatures=n_features)

    def extract(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Extrae descriptores locales SIFT de la imagen provista.

        Returns:
            np.ndarray: matriz de forma (N, 128) con descriptores float32.

        Raises:
            ValueError: Si la imagen está vacía, no es válida o no produce puntos clave.
        """
        img = self._load_image(source)
        if img is None or img.size == 0:
            raise ValueError("No se pudo cargar la imagen o está vacía")

        # Convertir a escala de grises si es necesario
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            gray = img
        else:
            raise ValueError(f"Formato de imagen no soportado con dimensiones: {img.shape}")

        keypoints, descriptors = self._sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            raise ValueError("No se encontraron puntos clave SIFT en la imagen")

        return descriptors.astype(np.float32)

    def _load_image(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            if not os.path.exists(source):
                raise ValueError(f"El archivo de imagen no existe: {source}")
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"No se pudo decodificar la imagen desde el archivo: {source}")
            return img

        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise ValueError("Buffer de bytes de imagen vacío")
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("No se pudo decodificar la imagen desde los bytes proporcionados")
            return img

        raise TypeError(f"Tipo de entrada no soportado para imagen: {type(source)}")
