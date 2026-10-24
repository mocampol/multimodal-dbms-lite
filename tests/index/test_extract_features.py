import os
import tempfile
import cv2
import numpy as np
import pytest
import soundfile as sf

from index.multimedia import (
    SIFTExtractor,
    MFCCExtractor,
    Quantizer,
    BoWTFIDF,
    ExtractFeatures,
    FeatureVector,
)


def _create_synthetic_image(path: str) -> str:
    """Crea una imagen sintética con patrones y texturas para generar puntos clave SIFT."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(0, 200, 20):
        cv2.line(img, (i, 0), (200 - i, 200), (255, 255, 255), 2)
        cv2.circle(img, (i, i), 15, (128, 200, 50), -1)
        cv2.rectangle(img, (i, 50), (i + 15, 100), (0, 255, 100), -1)
    cv2.imwrite(path, img)
    return path


def _create_synthetic_audio(path: str, sr: int = 22050, duration: float = 1.0) -> str:
    """Crea un archivo de audio sintético con tonos sinusoidales."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Combinación de frecuencias armónicas
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    sf.write(path, audio.astype(np.float32), sr)
    return path


class TestSIFTExtractor:
    def test_extract_valid_image(self, tmp_path):
        img_path = str(tmp_path / "test_pattern.png")
        _create_synthetic_image(img_path)

        extractor = SIFTExtractor(n_features=100)
        descriptors = extractor.extract(img_path)

        assert isinstance(descriptors, np.ndarray)
        assert descriptors.ndim == 2
        assert descriptors.shape[1] == 128
        assert len(descriptors) > 0

    def test_extract_from_numpy_array(self):
        img = np.zeros((150, 150), dtype=np.uint8)
        cv2.circle(img, (75, 75), 40, 255, -1)
        cv2.rectangle(img, (20, 20), (60, 60), 180, 2)

        extractor = SIFTExtractor()
        descriptors = extractor.extract(img)
        assert descriptors.shape[1] == 128
        assert len(descriptors) > 0

    def test_extract_invalid_path(self):
        extractor = SIFTExtractor()
        with pytest.raises(ValueError, match="no existe"):
            extractor.extract("/ruta/inexistente/imagen.png")

    def test_extract_blank_image(self):
        # Imagen completamente negra sin esquinas ni contraste
        extractor = SIFTExtractor()
        blank = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="No se encontraron puntos clave SIFT"):
            extractor.extract(blank)


class TestMFCCExtractor:
    def test_extract_valid_audio(self, tmp_path):
        audio_path = str(tmp_path / "test_tone.wav")
        _create_synthetic_audio(audio_path)

        extractor = MFCCExtractor(n_mfcc=20)
        frames = extractor.extract(audio_path)

        assert isinstance(frames, np.ndarray)
        assert frames.ndim == 2
        assert frames.shape[1] == 20
        assert len(frames) > 0

    def test_extract_from_waveform_array(self):
        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5))
        y = 0.5 * np.sin(2 * np.pi * 300 * t)

        extractor = MFCCExtractor(n_mfcc=13)
        frames = extractor.extract(y)
        assert frames.shape[1] == 13
        assert len(frames) > 0

    def test_extract_silent_audio(self):
        extractor = MFCCExtractor()
        silent = np.zeros(22050, dtype=np.float32)
        with pytest.raises(ValueError, match="no contiene energía acústica"):
            extractor.extract(silent)

    def test_extract_invalid_path(self):
        extractor = MFCCExtractor()
        with pytest.raises(ValueError, match="no existe"):
            extractor.extract("/ruta/inexistente/audio.wav")


class TestQuantizerAndBoW:
    def test_quantizer_fit_predict(self):
        rng = np.random.default_rng(42)
        descriptors = rng.standard_normal((200, 128)).astype(np.float32)

        quantizer = Quantizer(k=16, random_state=42)
        assert not quantizer.is_fitted
        quantizer.fit(descriptors)
        assert quantizer.is_fitted
        assert quantizer.centroids.shape == (16, 128)

        assignments = quantizer.predict(descriptors[:10])
        assert len(assignments) == 10
        assert np.all(assignments >= 0) and np.all(assignments < 16)

    def test_bow_tfidf_normalization(self):
        bow = BoWTFIDF(k=32)
        assignments = np.array([0, 1, 1, 5, 10, 10, 10, 31], dtype=np.int64)

        # Sin fit previo (IDF uniforme)
        vec = bow.compute(assignments)
        assert vec.shape == (32,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
        assert vec[10] > vec[0]  # La palabra 10 apareció más veces

        # Con fit sobre corpus
        bow.fit([assignments, np.array([0, 2]), np.array([5, 15])])
        vec2 = bow.compute(assignments)
        assert vec2.shape == (32,)
        assert np.isclose(np.linalg.norm(vec2), 1.0, atol=1e-5)


class TestExtractFeatures:
    def test_extract_numeric_vector(self):
        extractor = ExtractFeatures(k=64)
        raw_vec = [1.0, 2.0, 3.0, 4.0, 5.0]

        fv = extractor.extract(raw_vec)
        assert isinstance(fv, FeatureVector)
        assert fv.source_type == "numeric"
        assert fv.k == 64
        assert len(fv.data) == 64
        assert np.isclose(np.linalg.norm(fv.data), 1.0, atol=1e-5)

    def test_extract_image_end_to_end(self, tmp_path):
        img_path = str(tmp_path / "img1.png")
        _create_synthetic_image(img_path)

        extractor = ExtractFeatures(k=32)
        fv = extractor.extract(img_path)

        assert isinstance(fv, FeatureVector)
        assert fv.source_type == "image"
        assert fv.k == 32
        assert len(fv.data) == 32
        assert np.isclose(np.linalg.norm(fv.data), 1.0, atol=1e-5)

    def test_extract_audio_end_to_end(self, tmp_path):
        audio_path = str(tmp_path / "audio1.wav")
        _create_synthetic_audio(audio_path)

        extractor = ExtractFeatures(k=32)
        fv = extractor.extract(audio_path)

        assert isinstance(fv, FeatureVector)
        assert fv.source_type == "audio"
        assert fv.k == 32
        assert len(fv.data) == 32
        assert np.isclose(np.linalg.norm(fv.data), 1.0, atol=1e-5)

    def test_direct_call_syntax(self, tmp_path):
        raw_vec = [10.0, 20.0, 30.0]
        # Invocación funcional directa ExtractFeatures(objeto)
        fv = ExtractFeatures(raw_vec)
        assert isinstance(fv, FeatureVector)
        assert fv.k == 64
        assert np.isclose(np.linalg.norm(fv.data), 1.0, atol=1e-5)

    def test_fit_corpus_multimodal(self, tmp_path):
        img1 = str(tmp_path / "c_img1.png")
        img2 = str(tmp_path / "c_img2.png")
        _create_synthetic_image(img1)
        _create_synthetic_image(img2)

        extractor = ExtractFeatures(k=16)
        extractor.fit([img1, img2])
        assert extractor.is_fitted

        fv1 = extractor.extract(img1)
        assert fv1.k == 16
        assert np.isclose(np.linalg.norm(fv1.data), 1.0, atol=1e-5)

    def test_invalid_and_empty_inputs(self):
        extractor = ExtractFeatures(k=32)

        with pytest.raises(ValueError):
            extractor.extract(None)

        with pytest.raises(ValueError):
            extractor.extract([])

        with pytest.raises(ValueError):
            extractor.extract("")

        with pytest.raises(ValueError, match="Extensión de archivo no reconocida"):
            extractor.extract("archivo_desconocido.xyz")
