import os
import tempfile
import numpy as np
import pytest

from index.vector import (
    VectorMetric,
    SearchResult,
    compute_distance,
    distance_to_score,
    IVFIndex,
    HNSWIndex,
)


class TestMetrics:
    def test_euclidean_distance(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        dist = compute_distance(v1, v2, VectorMetric.EUCLIDEAN)
        assert np.isclose(dist, 3.0)
        score = distance_to_score(dist, VectorMetric.EUCLIDEAN)
        assert np.isclose(score, 1.0 / (1.0 + 3.0))

    def test_cosine_distance(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Ortogonales -> cos_sim = 0 -> dist = 1
        dist_ortho = compute_distance(v1, v2, VectorMetric.COSINE)
        assert np.isclose(dist_ortho, 1.0)
        assert np.isclose(distance_to_score(dist_ortho, VectorMetric.COSINE), 0.0)

        # Idénticos -> cos_sim = 1 -> dist = 0
        dist_same = compute_distance(v1, v3, VectorMetric.COSINE)
        assert np.isclose(dist_same, 0.0)
        assert np.isclose(distance_to_score(dist_same, VectorMetric.COSINE), 1.0)

    def test_dot_product_distance(self):
        v1 = np.array([2.0, 3.0], dtype=np.float32)
        v2 = np.array([4.0, 1.0], dtype=np.float32)
        # dot = 2*4 + 3*1 = 11 -> distance = -11
        dist = compute_distance(v1, v2, VectorMetric.DOT_PRODUCT)
        assert np.isclose(dist, -11.0)
        score = distance_to_score(dist, VectorMetric.DOT_PRODUCT)
        assert np.isclose(score, 11.0)


class TestIVFIndex:
    @pytest.mark.parametrize("metric", [VectorMetric.EUCLIDEAN, VectorMetric.COSINE, VectorMetric.DOT_PRODUCT])
    def test_ivf_build_and_search_order(self, metric):
        dim = 8
        n_samples = 50
        rng = np.random.default_rng(123)
        vectors = rng.standard_normal((n_samples, dim)).astype(np.float32)
        if metric in (VectorMetric.COSINE, VectorMetric.DOT_PRODUCT):
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        rids = [(1, i) for i in range(n_samples)]

        ivf = IVFIndex(dim=dim, metric=metric, n_lists=5, n_probe=5)
        ivf.build(vectors, rids)
        assert len(ivf) == n_samples

        query = vectors[0]  # El elemento idéntico debe ser el primero
        results = ivf.search(query, k=5)
        assert len(results) == 5
        assert results[0].rid == (1, 0)

        # Verificar orden ascendente de distancia
        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_ivf_add_incrementally(self):
        ivf = IVFIndex(dim=4, metric=VectorMetric.EUCLIDEAN, n_lists=2, n_probe=2)
        ivf.add(np.array([1, 0, 0, 0]), rid="doc1")
        ivf.add(np.array([0, 1, 0, 0]), rid="doc2")
        ivf.add(np.array([1.1, 0, 0, 0]), rid="doc3")

        assert len(ivf) == 3
        res = ivf.search(np.array([1, 0, 0, 0]), k=2)
        assert len(res) == 2
        assert res[0].rid == "doc1"
        assert res[1].rid == "doc3"

    def test_ivf_save_load(self, tmp_path):
        fpath = str(tmp_path / "test.ivf")
        ivf = IVFIndex(dim=4, metric=VectorMetric.COSINE, n_lists=2, n_probe=2)
        ivf.build(
            [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])],
            ["rid1", "rid2"],
        )
        ivf.save(fpath)

        loaded = IVFIndex.load(fpath)
        assert len(loaded) == 2
        assert loaded.dim == 4
        assert loaded.metric == VectorMetric.COSINE

        res = loaded.search(np.array([1, 0, 0, 0]), k=1)
        assert len(res) == 1
        assert res[0].rid == "rid1"


class TestHNSWIndex:
    @pytest.mark.parametrize("metric", [VectorMetric.EUCLIDEAN, VectorMetric.COSINE, VectorMetric.DOT_PRODUCT])
    def test_hnsw_build_and_search_order(self, metric):
        dim = 8
        n_samples = 40
        rng = np.random.default_rng(456)
        vectors = rng.standard_normal((n_samples, dim)).astype(np.float32)
        if metric in (VectorMetric.COSINE, VectorMetric.DOT_PRODUCT):
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        rids = [f"rec_{i}" for i in range(n_samples)]

        hnsw = HNSWIndex(dim=dim, metric=metric, max_elements=100)
        hnsw.build(vectors, rids)
        assert len(hnsw) == n_samples

        query = vectors[5]
        results = hnsw.search(query, k=5)
        assert len(results) == 5
        assert results[0].rid == "rec_5"

        # Verificar orden ascendente de distancia
        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_hnsw_add_incrementally(self):
        hnsw = HNSWIndex(dim=3, metric=VectorMetric.EUCLIDEAN, max_elements=10)
        hnsw.add(np.array([0.0, 0.0, 0.0]), "origin")
        hnsw.add(np.array([10.0, 10.0, 10.0]), "far")
        hnsw.add(np.array([0.1, 0.1, 0.0]), "near")

        res = hnsw.search(np.array([0.0, 0.0, 0.0]), k=2)
        assert len(res) == 2
        assert res[0].rid == "origin"
        assert res[1].rid == "near"

    def test_hnsw_save_load(self, tmp_path):
        fpath = str(tmp_path / "test_hnsw.index")
        hnsw = HNSWIndex(dim=4, metric=VectorMetric.EUCLIDEAN, max_elements=20)
        hnsw.build(
            [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])],
            ["a", "b"],
        )
        hnsw.save(fpath)

        loaded = HNSWIndex.load(fpath)
        assert len(loaded) == 2
        assert loaded.dim == 4

        res = loaded.search(np.array([1, 2, 3, 4]), k=1)
        assert len(res) == 1
        assert res[0].rid == "a"


class TestIndexUniformAPI:
    def test_metric_independent_api_unpacking(self):
        # Desempaquetado de resultados: (rid, score)
        res = SearchResult(rid=(1, 5), distance=0.25, score=0.8)
        rid, score = res
        assert rid == (1, 5)
        assert score == 0.8
        assert res.distance == 0.25

    def test_empty_search(self):
        ivf = IVFIndex(dim=4)
        assert ivf.search(np.array([1, 0, 0, 0]), k=5) == []

        hnsw = HNSWIndex(dim=4)
        assert hnsw.search(np.array([1, 0, 0, 0]), k=5) == []
