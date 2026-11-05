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

    @pytest.mark.parametrize(
        "metric",
        [VectorMetric.EUCLIDEAN, VectorMetric.DOT_PRODUCT, VectorMetric.COSINE],
    )
    def test_ivf_search_can_override_metric(self, metric):
        vectors = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([4.0, 1.0], dtype=np.float32),
            np.array([-2.0, 3.0], dtype=np.float32),
            np.array([0.5, -1.0], dtype=np.float32),
        ]
        rids = ["a", "b", "c", "d"]
        index = IVFIndex(
            dim=2,
            metric=VectorMetric.EUCLIDEAN,
            n_lists=2,
            n_probe=2,
        )
        index.build(vectors, rids)

        results = index.search(np.array([1.0, 1.0]), k=len(vectors), metric=metric)

        expected = sorted(
            zip(rids, vectors),
            key=lambda item: compute_distance(
                np.array([1.0, 1.0]), item[1], metric
            ),
        )
        assert [result.rid for result in results] == [rid for rid, _ in expected]
        assert [result.distance for result in results] == sorted(
            result.distance for result in results
        )

    def test_ivf_probes_lists_using_overridden_metric(self):
        index = IVFIndex(dim=2, metric=VectorMetric.EUCLIDEAN, n_lists=2, n_probe=1)
        index.centroids = np.array([[10.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        index.inverted_lists = {
            0: [("high-dot", np.array([10.0, 0.0], dtype=np.float32))],
            1: [("near-euclidean", np.array([1.0, 0.0], dtype=np.float32))],
        }
        index._count = 2

        results = index.search(
            np.array([1.0, 0.0]),
            k=1,
            metric=VectorMetric.DOT_PRODUCT,
        )

        assert results[0].rid == "high-dot"

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

    @pytest.mark.parametrize(
        "metric",
        [VectorMetric.EUCLIDEAN, VectorMetric.COSINE],
    )
    def test_hnsw_search_with_overridden_metric_ranks_all_vectors(self, metric):
        vectors = [
            np.array([float(x), 0.0], dtype=np.float32)
            for x in range(1, 31)
        ]
        rids = [str(x) for x in range(1, 31)]
        index = HNSWIndex(dim=2, metric=VectorMetric.DOT_PRODUCT, max_elements=64)
        index.build(vectors, rids)
        query = np.array([10.0, 0.0], dtype=np.float32)

        results = index.search(query, k=5, metric=metric)

        expected = sorted(
            zip(rids, vectors),
            key=lambda item: compute_distance(query, item[1], metric),
        )[:5]
        assert [result.rid for result in results] == [rid for rid, _ in expected]
        assert [result.distance for result in results] == sorted(
            result.distance for result in results
        )

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



VECTOR_INDEXES = [IVFIndex, HNSWIndex]


def _index_with_two_vectors(index_class, metric=VectorMetric.EUCLIDEAN):
    index = index_class(dim=2, metric=metric)
    index.build(
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        ["a", "b"],
    )
    return index


class TestMetricValidation:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("euclidean", VectorMetric.EUCLIDEAN),
            (" L2 ", VectorMetric.EUCLIDEAN),
            ("IP", VectorMetric.DOT_PRODUCT),
            ("dot", VectorMetric.DOT_PRODUCT),
            ("inner_product", VectorMetric.DOT_PRODUCT),
            ("Cos", VectorMetric.COSINE),
        ],
    )
    def test_from_str_accepts_aliases(self, name, expected):
        assert VectorMetric.from_str(name) == expected

    def test_from_str_returns_existing_metric(self):
        assert VectorMetric.from_str(VectorMetric.COSINE) is VectorMetric.COSINE

    def test_from_str_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="Métrica desconocida"):
            VectorMetric.from_str("manhattan")

    def test_cosine_of_opposite_vectors(self):
        distance = compute_distance(
            np.array([1.0, 0.0]), np.array([-1.0, 0.0]), VectorMetric.COSINE
        )
        assert np.isclose(distance, 2.0)
        assert np.isclose(distance_to_score(distance, VectorMetric.COSINE), -1.0)

    def test_cosine_with_zero_vector_has_neutral_distance(self):
        distance = compute_distance(
            np.zeros(2), np.array([1.0, 0.0]), VectorMetric.COSINE
        )
        assert distance == 1.0

    @pytest.mark.parametrize("metric", list(VectorMetric))
    def test_distance_rejects_different_dimensions(self, metric):
        with pytest.raises(ValueError):
            compute_distance(np.ones(2), np.ones(3), metric)

    def test_rejects_unknown_metric_in_distance_and_score(self):
        with pytest.raises(ValueError, match="no soportada"):
            compute_distance(np.ones(2), np.ones(2), "manhattan")
        with pytest.raises(ValueError, match="no soportada"):
            distance_to_score(1.0, "manhattan")

    @pytest.mark.parametrize("distance", [0.0, -5.0])
    def test_euclidean_score_never_exceeds_one(self, distance):
        assert distance_to_score(distance, VectorMetric.EUCLIDEAN) == 1.0

    def test_search_result_access_by_position(self):
        result = SearchResult(rid="a", distance=0.25, score=0.8)

        assert (result[0], result[1], result[2]) == ("a", 0.8, 0.25)
        with pytest.raises(IndexError):
            result[3]


@pytest.mark.parametrize("index_class", VECTOR_INDEXES)
class TestVectorIndexValidation:
    @pytest.mark.parametrize("dim", [0, -1])
    def test_rejects_non_positive_dimension(self, index_class, dim):
        with pytest.raises(ValueError, match="entero positivo"):
            index_class(dim=dim)

    def test_rejects_unknown_metric(self, index_class):
        with pytest.raises(ValueError, match="Métrica desconocida"):
            index_class(dim=2, metric="manhattan")

    def test_build_rejects_different_vector_and_rid_counts(self, index_class):
        with pytest.raises(ValueError, match="cantidad de vectores"):
            index_class(dim=2).build([np.zeros(2)], ["a", "b"])

    def test_build_rejects_wrong_dimension(self, index_class):
        with pytest.raises(ValueError, match="forma"):
            index_class(dim=2).build([np.zeros(3)], ["a"])

    def test_search_rejects_wrong_query_dimension(self, index_class):
        index = _index_with_two_vectors(index_class)
        with pytest.raises(ValueError, match="Dimensión del query"):
            index.search(np.zeros(3), k=1)

    def test_add_rejects_wrong_dimension_without_changing_index(self, index_class):
        index = _index_with_two_vectors(index_class)
        with pytest.raises(ValueError, match="Dimensión del vector"):
            index.add(np.zeros(3), "c")
        assert len(index) == 2

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_returns_no_results(self, index_class, k):
        index = _index_with_two_vectors(index_class)
        assert index.search(np.array([1.0, 0.0]), k=k) == []

    def test_k_larger_than_index_returns_all_vectors_in_order(self, index_class):
        index = _index_with_two_vectors(index_class)
        results = index.search(np.array([1.0, 0.0]), k=9)
        assert [result.rid for result in results] == ["a", "b"]

    def test_search_rejects_unknown_metric(self, index_class):
        index = _index_with_two_vectors(index_class)
        with pytest.raises(ValueError, match="Métrica desconocida"):
            index.search(np.array([1.0, 0.0]), k=1, metric="manhattan")

    def test_save_requires_path(self, index_class):
        index = _index_with_two_vectors(index_class)
        with pytest.raises(ValueError, match="Ruta de archivo requerida"):
            index.save()

    def test_load_rejects_missing_file(self, index_class, tmp_path):
        with pytest.raises(FileNotFoundError):
            index_class.load(str(tmp_path / "no_existe.idx"))

    def test_build_with_empty_dataset_produces_empty_index(self, index_class):
        index = index_class(dim=2)
        index.build([], [])

        assert len(index) == 0
        assert index.search(np.zeros(2), k=1) == []

    def test_rebuild_with_empty_dataset_discards_previous_vectors(self, index_class):
        index = _index_with_two_vectors(index_class)
        index.build([], [])

        assert len(index) == 0
        assert index.search(np.array([1.0, 0.0]), k=1) == []

    def test_zero_vector_is_ranked_after_similar_vector_with_cosine(self, index_class):
        index = index_class(dim=2, metric=VectorMetric.COSINE)
        index.build([np.zeros(2), np.array([1.0, 0.0])], ["zero", "a"])

        results = index.search(np.array([1.0, 0.0]), k=2)

        assert [result.rid for result in results] == ["a", "zero"]
        assert np.isclose(results[1].distance, 1.0)

    def test_dot_product_score_is_the_product(self, index_class):
        index = _index_with_two_vectors(index_class, VectorMetric.DOT_PRODUCT)

        results = index.search(np.array([2.0, 1.0]), k=2)

        assert [result.rid for result in results] == ["a", "b"]
        assert np.isclose(results[0].score, 2.0)
        assert all(np.isclose(r.score, -r.distance) for r in results)

    def test_add_after_build_makes_vector_searchable(self, index_class):
        index = _index_with_two_vectors(index_class)
        index.add(np.array([1.0, 1.0]), "c")

        assert len(index) == 3
        assert index.search(np.array([1.0, 1.0]), k=1)[0].rid == "c"


class TestIVFValidation:
    @pytest.mark.parametrize("params", [{"n_lists": 0}, {"n_probe": 0}])
    def test_rejects_non_positive_parameters(self, params):
        with pytest.raises(ValueError, match="mayor a 0"):
            IVFIndex(dim=2, **params)

    def test_uses_fewer_lists_when_there_are_fewer_vectors(self):
        index = IVFIndex(dim=2, n_lists=8, n_probe=8)
        index.build([np.array([1.0, 0.0]), np.array([0.0, 1.0])], ["a", "b"])

        assert len(index.centroids) == 2
        assert sum(len(items) for items in index.inverted_lists.values()) == 2

    def test_build_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            IVFIndex(dim=2).build(
                [np.array([np.nan, 1.0]), np.array([1.0, 1.0])],
                ["a", "b"],
            )

    def test_add_without_build_creates_first_list(self):
        index = IVFIndex(dim=2, metric=VectorMetric.EUCLIDEAN)
        index.add(np.array([1.0, 0.0]), "a")
        index.add(np.array([0.0, 1.0]), "b")

        assert len(index) == 2
        assert len(index.centroids) == 1
        assert index.search(np.array([0.0, 1.0]), k=1)[0].rid == "b"


class TestHNSWValidation:
    def test_add_grows_beyond_initial_capacity(self):
        index = HNSWIndex(dim=2, metric=VectorMetric.EUCLIDEAN, max_elements=100)
        for value in range(150):
            index.add(np.array([float(value), 0.0]), value)

        assert len(index) == 150
        assert index.max_elements == 200
        assert index.search(np.array([149.0, 0.0]), k=1)[0].rid == 149

    def test_capacity_has_minimum(self):
        assert HNSWIndex(dim=2, max_elements=1).max_elements == 100

    def test_build_grows_capacity_for_dataset(self):
        index = HNSWIndex(dim=2, max_elements=100)
        index.build(
            [np.array([float(value + 1), 1.0]) for value in range(120)],
            list(range(120)),
        )
        assert index.max_elements == 240


def _clustered_dataset(n_clusters=20, per_cluster=50, dim=16, seed=11):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10.0, 10.0, (n_clusters, dim))
    vectors = np.vstack(
        [center + rng.normal(0.0, 0.5, (per_cluster, dim)) for center in centers]
    ).astype(np.float32)
    queries = (centers + rng.normal(0.0, 0.5, centers.shape)).astype(np.float32)
    return vectors, queries


def _exact_top_k(vectors, query, k, metric):
    distances = [compute_distance(query, vector, metric) for vector in vectors]
    return sorted(range(len(vectors)), key=lambda i: (distances[i], i))[:k]


def _recall_at_k(index, vectors, queries, k, metric):
    hits = 0
    for query in queries:
        found = {result.rid for result in index.search(query, k=k, metric=metric)}
        hits += len(found & set(_exact_top_k(vectors, query, k, metric)))
    return hits / (k * len(queries))


@pytest.fixture(scope="module")
def clustered_dataset():
    return _clustered_dataset()


@pytest.mark.parametrize("metric", list(VectorMetric))
class TestRecallAndOrder:
    def test_hnsw_recall_on_clustered_data(self, clustered_dataset, metric):
        vectors, queries = clustered_dataset
        index = HNSWIndex(dim=vectors.shape[1], metric=metric)
        index.build(vectors, list(range(len(vectors))))

        assert _recall_at_k(index, vectors, queries, 10, metric) >= 0.90

    def test_ivf_recall_on_clustered_data(self, clustered_dataset, metric):
        vectors, queries = clustered_dataset
        index = IVFIndex(dim=vectors.shape[1], metric=metric, n_lists=20, n_probe=4)
        index.build(vectors, list(range(len(vectors))))

        assert _recall_at_k(index, vectors, queries, 10, metric) >= 0.95

    def test_ivf_recall_does_not_drop_when_probing_more_lists(self, metric):
        rng = np.random.default_rng(23)
        vectors = rng.standard_normal((400, 16)).astype(np.float32)
        queries = rng.standard_normal((20, 16)).astype(np.float32)

        recalls = []
        for n_probe in (1, 4, 16):
            index = IVFIndex(dim=16, metric=metric, n_lists=16, n_probe=n_probe)
            index.build(vectors, list(range(len(vectors))))
            recalls.append(_recall_at_k(index, vectors, queries, 10, metric))

        assert recalls == sorted(recalls)
        assert recalls[0] < recalls[-1]
        assert recalls[-1] == 1.0

    @pytest.mark.parametrize(
        "index_class, params",
        [
            (IVFIndex, {"n_lists": 20, "n_probe": 20}),
            (HNSWIndex, {}),
        ],
    )
    def test_results_are_ordered_and_match_the_metric(
        self, clustered_dataset, metric, index_class, params
    ):
        vectors, queries = clustered_dataset
        index = index_class(dim=vectors.shape[1], metric=metric, **params)
        index.build(vectors, list(range(len(vectors))))

        for query in queries:
            results = index.search(query, k=10)
            distances = [result.distance for result in results]
            scores = [result.score for result in results]

            assert len(results) == 10
            assert len({result.rid for result in results}) == 10
            assert distances == sorted(distances)
            assert scores == sorted(scores, reverse=True)
            for result in results:
                expected = compute_distance(query, vectors[result.rid], metric)
                assert np.isclose(result.distance, expected, atol=1e-3)

    def test_ivf_probing_all_lists_returns_exact_neighbors(
        self, clustered_dataset, metric
    ):
        vectors, queries = clustered_dataset
        index = IVFIndex(dim=vectors.shape[1], metric=metric, n_lists=20, n_probe=20)
        index.build(vectors, list(range(len(vectors))))

        for k in (1, 5, 10, 50):
            assert _recall_at_k(index, vectors, queries, k, metric) == 1.0

    @pytest.mark.parametrize("index_class", VECTOR_INDEXES)
    def test_saved_index_returns_same_results(
        self, clustered_dataset, metric, index_class, tmp_path
    ):
        vectors, queries = clustered_dataset
        index = index_class(dim=vectors.shape[1], metric=metric)
        index.build(vectors, list(range(len(vectors))))
        path = str(tmp_path / "vectores.idx")
        index.save(path)

        loaded = index_class.load(path)

        assert loaded.metric == metric
        assert [r.rid for r in loaded.search(queries[0], k=10)] == [
            r.rid for r in index.search(queries[0], k=10)
        ]


@pytest.mark.parametrize(
    "metric", [VectorMetric.COSINE, VectorMetric.DOT_PRODUCT]
)
def test_hnsw_with_other_metric_ranks_exactly(clustered_dataset, metric):
    vectors, queries = clustered_dataset
    index = HNSWIndex(dim=vectors.shape[1], metric=VectorMetric.EUCLIDEAN)
    index.build(vectors, list(range(len(vectors))))

    assert _recall_at_k(index, vectors, queries, 10, metric) == 1.0
