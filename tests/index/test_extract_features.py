import os
import pickle
import tempfile
import cv2
import numpy as np
import pytest
import soundfile as sf
from sklearn.datasets import load_sample_image

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

    def test_quantizer_rejects_invalid_descriptors(self):
        quantizer = Quantizer(k=2)
        with pytest.raises(ValueError, match="valores no finitos"):
            quantizer.fit(np.array([[1.0], [np.nan]], dtype=np.float32))

        quantizer.fit(np.array([[0.0], [1.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="espera 1"):
            quantizer.predict(np.ones((1, 2), dtype=np.float32))

    def test_bow_rejects_invalid_assignments(self):
        bow = BoWTFIDF(k=4)
        with pytest.raises(ValueError, match="entre 0 y k - 1"):
            bow.compute(np.array([-1, 1]))
        with pytest.raises(ValueError, match="unidimensional de enteros"):
            bow.fit([np.array([[0, 1]])])


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
        assert extractor.is_fitted
        assert extractor.quantizer.centroids.shape[1] == 1
        assert extractor.bow_tfidf.idf_.shape == (64,)

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

    def test_extract_real_sample_image(self):
        image = load_sample_image("china.jpg")
        extractor = ExtractFeatures(k=32)

        feature_vector = extractor.extract(image)

        assert feature_vector.source_type == "image"
        assert feature_vector.data.shape == (32,)
        assert extractor.quantizer.centroids.shape[1] == 128

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
        audio1 = str(tmp_path / "c_audio1.wav")
        audio2 = str(tmp_path / "c_audio2.wav")
        _create_synthetic_image(img1)
        _create_synthetic_image(img2)
        _create_synthetic_audio(audio1)
        _create_synthetic_audio(audio2)

        extractor = ExtractFeatures(k=16)
        extractor.fit([img1, img2, audio1, audio2, [1.0, 2.0], [3.0, 4.0]])
        assert extractor.is_fitted

        image_vector = extractor.extract(img1)
        audio_vector = extractor.extract(audio1)
        numeric_vector = extractor.extract([1.0, 2.0])
        assert image_vector.data.shape == audio_vector.data.shape == numeric_vector.data.shape == (16,)
        assert extractor._quantizers["image"].centroids.shape[1] == 128
        assert extractor._quantizers["audio"].centroids.shape[1] == 20
        assert extractor._quantizers["numeric"].centroids.shape[1] == 1
        for feature_vector in (image_vector, audio_vector, numeric_vector):
            assert np.isclose(np.linalg.norm(feature_vector.data), 1.0, atol=1e-5)
        assert extractor._bow_tfidf_by_modality["numeric"].idf_.shape == (16,)

    def test_numeric_feature_vector_uses_corpus_idf(self):
        extractor = ExtractFeatures(k=2)
        extractor.fit([[0.0, 1.0], [0.0, 0.0]])

        assignments = extractor._quantizers["numeric"].predict(np.array([[0.0], [1.0]]))
        common_word, rare_word = assignments
        idf = extractor._bow_tfidf_by_modality["numeric"].idf_
        assert idf[rare_word] > idf[common_word]
        feature_vector = extractor.extract([0.0, 1.0])
        assert feature_vector.data[rare_word] > feature_vector.data[common_word]

    def test_numeric_vector_rejects_non_finite_values(self):
        extractor = ExtractFeatures(k=8)
        with pytest.raises(ValueError, match="valores no finitos"):
            extractor.extract([1.0, np.inf])

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



class TestQuantizerValidation:
    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k):
        with pytest.raises(ValueError, match="entero positivo"):
            Quantizer(k=k)

    def test_requires_fit_before_use(self):
        quantizer = Quantizer(k=2)
        with pytest.raises(ValueError, match="no está entrenado"):
            quantizer.predict(np.ones((1, 2)))
        with pytest.raises(ValueError, match="aún no ha sido entrenado"):
            quantizer.centroids

    @pytest.mark.parametrize(
        "descriptors, error",
        [
            (None, "No se proporcionaron"),
            (np.empty((0, 2)), "No se proporcionaron"),
            (np.ones(3), "2 dimensiones"),
            (np.empty((3, 0)), "al menos una característica"),
        ],
    )
    def test_fit_rejects_invalid_descriptors(self, descriptors, error):
        with pytest.raises(ValueError, match=error):
            Quantizer(k=2).fit(descriptors)

    @pytest.mark.parametrize("descriptors", [None, np.empty((0, 1))])
    def test_predict_rejects_empty_descriptors(self, descriptors):
        quantizer = Quantizer(k=2).fit(np.array([[0.0], [1.0]]))
        with pytest.raises(ValueError, match="para cuantizar"):
            quantizer.predict(descriptors)

    def test_uses_fewer_centroids_when_there_are_fewer_samples_than_k(self):
        quantizer = Quantizer(k=10).fit(np.array([[0.0], [1.0], [2.0]]))
        assert quantizer.centroids.shape == (3, 1)
        assert quantizer.codebook_size() == 10

    def test_assigns_separated_clusters_to_different_words(self):
        rng = np.random.default_rng(0)
        low = rng.normal(0.0, 0.01, (50, 2))
        high = rng.normal(10.0, 0.01, (50, 2))

        quantizer = Quantizer(k=2).fit(np.vstack([low, high]))
        low_words = set(quantizer.predict(low).tolist())
        high_words = set(quantizer.predict(high).tolist())

        assert len(low_words) == 1
        assert len(high_words) == 1
        assert low_words != high_words

    def test_same_seed_produces_same_assignments(self):
        descriptors = np.random.default_rng(1).standard_normal((200, 4))
        first = Quantizer(k=8, random_state=1).fit(descriptors).predict(descriptors)
        second = Quantizer(k=8, random_state=1).fit(descriptors).predict(descriptors)
        assert np.array_equal(first, second)

    def test_large_dataset_keeps_requested_codebook_size(self):
        descriptors = np.random.default_rng(2).standard_normal((600, 4)).astype(np.float32)
        quantizer = Quantizer(k=8).fit(descriptors)
        assignments = quantizer.predict(descriptors)

        assert quantizer.centroids.shape == (8, 4)
        assert assignments.min() >= 0
        assert assignments.max() < 8


class TestBoWTFIDFValidation:
    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError, match="entero positivo"):
            BoWTFIDF(k=0)

    @pytest.mark.parametrize(
        "assignments, error",
        [
            (None, "no pueden estar vacías"),
            (np.array([], dtype=np.int64), "no pueden estar vacías"),
            (np.array([0.5]), "unidimensional"),
            (np.array([2]), "entre 0 y k - 1"),
        ],
    )
    def test_compute_rejects_invalid_assignments(self, assignments, error):
        with pytest.raises(ValueError, match=error):
            BoWTFIDF(k=2).compute(assignments)

    def test_fit_with_empty_corpus_uses_uniform_idf(self):
        bow = BoWTFIDF(k=2).fit([])
        assert np.array_equal(bow.idf_, np.ones(2, dtype=np.float32))

    def test_idf_follows_smooth_formula(self):
        bow = BoWTFIDF(k=3).fit([np.array([0]), np.array([0, 1])])
        assert np.allclose(bow.idf_, [1.0, np.log(1.5) + 1.0, np.log(3.0) + 1.0])

    def test_compute_applies_idf_and_l2_normalization(self):
        bow = BoWTFIDF(k=3).fit([np.array([0]), np.array([0, 1])])

        vector = bow.compute(np.array([0, 1]))

        expected = np.array([1.0, np.log(1.5) + 1.0, 0.0])
        expected /= np.linalg.norm(expected)
        assert np.allclose(vector, expected)


class TestExtractFeaturesValidation:
    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k):
        with pytest.raises(ValueError, match="entero positivo"):
            ExtractFeatures(k=k)

    def test_fit_rejects_empty_corpus(self):
        with pytest.raises(ValueError, match="no puede estar vacío"):
            ExtractFeatures(k=2).fit([])

    @pytest.mark.parametrize(
        "item, error",
        [
            ("archivo.xyz", "Extensión de archivo no reconocida"),
            (None, "no puede ser None"),
            ("", "Ruta de archivo vacía"),
        ],
    )
    def test_fit_rejects_invalid_items_and_stays_unfitted(self, item, error):
        extractor = ExtractFeatures(k=2)
        with pytest.raises(ValueError, match=error):
            extractor.fit([item])
        assert not extractor.is_fitted

    @pytest.mark.parametrize("obj", [{}, {1}, np.zeros((2, 2, 2, 2))])
    def test_extract_rejects_unsupported_types(self, obj):
        with pytest.raises(TypeError, match="no soportado"):
            ExtractFeatures(k=2).extract(obj)

    @pytest.mark.parametrize(
        "obj, error",
        [
            (["a"], "elementos no numéricos"),
            (np.array([]), "Array numpy vacío"),
            (np.array([np.nan]), "valores no finitos"),
            ("archivo", "Extensión de archivo no reconocida"),
            ("no_existe.png", "no existe"),
            ("no_existe.wav", "no existe"),
        ],
    )
    def test_extract_rejects_invalid_values(self, obj, error):
        with pytest.raises(ValueError, match=error):
            ExtractFeatures(k=2).extract(obj)

    def test_extract_rejects_unknown_source_type(self):
        with pytest.raises(ValueError, match="Modalidad desconocida"):
            ExtractFeatures(k=2).extract([1.0], source_type="video")

    @pytest.mark.parametrize(
        "obj, expected",
        [
            ("FOTO.PNG", "image"),
            ("clip.WAV", "audio"),
            ([1, 2.0], "numeric"),
            ((1, 2), "numeric"),
            (np.zeros(3), "numeric"),
            (np.zeros((4, 4), dtype=np.uint8), "image"),
        ],
    )
    def test_detect_type(self, obj, expected):
        assert ExtractFeatures(k=2).detect_type(obj) == expected

    def test_same_object_is_more_similar_than_a_different_one(self):
        first = [0.0, 1.0, 2.0, 3.0]
        different = [10.0, 11.0, 12.0, 13.0]
        extractor = ExtractFeatures(k=4).fit([first, different, [0.0, 1.0, 2.0, 2.5]])

        first_vector = extractor.extract(first).data
        different_vector = extractor.extract(different).data

        assert np.allclose(first_vector, extractor.extract(first).data)
        assert np.dot(first_vector, first_vector) > np.dot(first_vector, different_vector)

    def test_same_corpus_produces_same_vector(self):
        corpus = [[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]
        first = ExtractFeatures(k=4).fit(corpus).extract(corpus[0]).data
        second = ExtractFeatures(k=4).fit(corpus).extract(corpus[0]).data
        assert np.array_equal(first, second)


class TestExtractFeaturesPersistence:
    def test_sift_extractor_survives_pickle(self, tmp_path):
        path = _create_synthetic_image(str(tmp_path / "sift.png"))
        extractor = SIFTExtractor(n_features=100)

        loaded = pickle.loads(pickle.dumps(extractor))

        assert loaded.n_features == 100
        assert np.array_equal(loaded.extract(path), extractor.extract(path))

    def test_loaded_extractor_produces_the_same_vectors(self, tmp_path):
        image = _create_synthetic_image(str(tmp_path / "img.png"))
        audio = _create_synthetic_audio(str(tmp_path / "audio.wav"))
        corpus = [image, audio, [1.0, 2.0], [3.0, 4.0]]
        extractor = ExtractFeatures(k=8).fit(corpus)
        path = str(tmp_path / "nested" / "extractor.pkl")

        extractor.save(path)
        loaded = ExtractFeatures.load(path)

        assert loaded.is_fitted
        assert loaded.k == 8
        for item in corpus:
            assert np.array_equal(loaded.extract(item).data, extractor.extract(item).data)

    def test_load_rejects_a_file_that_is_not_an_extractor(self, tmp_path):
        path = tmp_path / "other.pkl"
        path.write_bytes(pickle.dumps({"k": 8}))

        with pytest.raises(ValueError, match="no contiene un ExtractFeatures"):
            ExtractFeatures.load(str(path))


class TestFeatureVector:
    def test_rejects_dimension_different_from_k(self):
        with pytest.raises(ValueError, match="no coincide con k"):
            FeatureVector(np.zeros(3), "numeric", 2)

    def test_flattens_data_and_exposes_values(self):
        vector = FeatureVector(np.array([[1.0, 2.0], [3.0, 4.0]]), "numeric", 4)

        assert vector.data.shape == (4,)
        assert vector.data.dtype == np.float32
        assert len(vector) == 4
        assert vector[2] == 3.0
        assert vector.to_list() == [1.0, 2.0, 3.0, 4.0]