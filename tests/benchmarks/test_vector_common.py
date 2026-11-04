"""
Partición y vectores de la comparación de similitud (benchmarks/vector_common.py).
No necesita los datasets: usa árboles de archivos falsos e histogramas pequeños.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from index.multimedia.bow_tfidf import BoWTFIDF  # noqa: E402
from vector_common import (  # noqa: E402
    NUM_QUERIES,
    FeatureSet,
    Item,
    build_vectors,
    compute_idf,
    list_audio_items,
    list_image_items,
    plan_items,
    shuffled,
)


COUNTS = np.array([[3, 0, 1, 0], [0, 2, 2, 0], [1, 1, 1, 1]], dtype=np.uint16)


def touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def image_tree(tmp_path):
    base = tmp_path / "tiny-imagenet-200"
    for label in ("n01", "n02"):
        for i in range(400):
            touch(base / "train" / label / "images" / f"{label}_{i}.JPEG")
    lines = [f"val_{i}.JPEG\tn0{i % 2 + 1}\t0\t0\t1\t1\n" for i in range(NUM_QUERIES * 3)]
    touch(base / "val" / "val_annotations.txt", "".join(lines))
    return tmp_path


@pytest.fixture
def audio_tree(tmp_path):
    for label in ("yes", "no"):
        for i in range(NUM_QUERIES * 2):
            touch(tmp_path / label / f"{i}.wav")
    return tmp_path


# Vectores

def test_normalized_vectors_are_what_the_engine_computes():
    idf = compute_idf(COUNTS)
    bow = BoWTFIDF(k=4).fit([np.repeat(np.arange(4), row) for row in COUNTS])
    assert idf == pytest.approx(bow.idf_)

    vectors = build_vectors(COUNTS, idf, "normalized")
    for row, vector in zip(COUNTS, vectors):
        assert vector == pytest.approx(bow.compute(np.repeat(np.arange(4), row)))
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0)


def test_empty_histogram_gives_a_zero_vector():
    vector = build_vectors(np.zeros((1, 4), dtype=np.uint16), np.ones(4, dtype=np.float32), "normalized")
    assert not vector.any()


def test_raw_vectors_keep_the_magnitude():
    idf = compute_idf(COUNTS)
    raw = build_vectors(COUNTS, idf, "raw")
    assert raw == pytest.approx(COUNTS * idf)
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0)


def test_metrics_agree_on_normalized_vectors_but_not_on_raw_ones():
    """Por esto se miden las dos variantes: con norma 1 ordenan igual."""
    counts = np.array([[8, 1], [1, 2], [80, 60]], dtype=np.uint16)
    query = np.array([[5, 4]], dtype=np.uint16)
    idf = np.ones(2, dtype=np.float32)

    def orders(variant):
        base, q = build_vectors(counts, idf, variant), build_vectors(query, idf, variant)[0]
        euclidean = np.argsort(np.linalg.norm(base - q, axis=1))
        cosine = np.argsort(-(base @ q) / (np.linalg.norm(base, axis=1) * np.linalg.norm(q)))
        return euclidean.tolist(), cosine.tolist()

    euclidean, cosine = orders("normalized")
    assert euclidean == cosine
    euclidean, cosine = orders("raw")
    assert euclidean != cosine


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Variante"):
        build_vectors(COUNTS, compute_idf(COUNTS), "otra")


def test_vectors_of_a_smaller_n_are_a_prefix_of_a_larger_one():
    features = FeatureSet(
        modality="image", base_counts=COUNTS, query_counts=COUNTS[:1],
        base_labels=np.array(["a", "b", "c"]), query_labels=np.array(["a"]),
        base_paths=np.array(["x", "y", "z"]), query_paths=np.array(["q"]),
        idf=compute_idf(COUNTS),
    )
    assert features.base_vectors(2, "raw") == pytest.approx(features.base_vectors(3, "raw")[:2])
    with pytest.raises(ValueError, match="Se pidieron 4"):
        features.base_vectors(4, "raw")


# Partición

def test_shuffled_does_not_depend_on_the_listing_order():
    items = [Item(f"{i}.wav", "x") for i in range(50)]
    assert shuffled(items, 42) == shuffled(list(reversed(items)), 42)
    assert shuffled(items, 42) != shuffled(items, 43)


def test_image_items_come_with_their_labels(image_tree):
    train, val = list_image_items(image_tree)
    assert len(train) == 800 and len(val) == NUM_QUERIES * 3
    assert all(item.path.split("/")[2] == item.label for item in train)
    assert val[0] == Item("tiny-imagenet-200/val/images/val_0.JPEG", "n01")


def test_audio_items_use_the_folder_as_label(audio_tree):
    items = list_audio_items(audio_tree)
    assert len(items) == NUM_QUERIES * 4
    assert all(item.path.startswith(item.label + "/") for item in items)


@pytest.mark.parametrize("modality, fixture", [("image", "image_tree"), ("audio", "audio_tree")])
def test_queries_and_base_are_disjoint_and_reproducible(modality, fixture, request):
    root = request.getfixturevalue(fixture)
    queries, base = plan_items(modality, root)
    assert not {item.path for item in queries} & {item.path for item in base}
    assert len(queries) == NUM_QUERIES * 2
    assert (queries, base) == plan_items(modality, root)


def test_image_queries_come_from_validation_and_the_rest_tops_up_the_base(image_tree):
    queries, base = plan_items("image", image_tree)
    assert all("/val/" in item.path for item in queries)
    assert all("/train/" in item.path for item in base[:800])
    assert all("/val/" in item.path for item in base[800:])
    assert len(base) == 800 + NUM_QUERIES


def test_unknown_modality_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Modalidad"):
        plan_items("video", tmp_path)
