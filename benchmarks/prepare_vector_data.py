"""
benchmarks/prepare_vector_data.py — Descarga Tiny ImageNet y Speech Commands,
entrena el codebook con una muestra, extrae SIFT / MFCC en paralelo y guarda el
histograma de palabras de cada objeto en benchmarks/corpus/vector/ (ignorado por
git). El resumen reproducible queda en results/vector_dataset_<modalidad>.json.

Uso:
    python benchmarks/prepare_vector_data.py                     # 100,000 + consultas
    python benchmarks/prepare_vector_data.py --quick             # 1,000 + consultas
    python benchmarks/prepare_vector_data.py --modality image
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

# Un hilo por proceso: el paralelismo lo da el Pool. Antes de importar numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vector_common import (
    CODEBOOK_SAMPLES,
    DATASETS,
    FEATURE_DIM,
    N_VALUES,
    NUM_QUERIES,
    QUICK_CODEBOOK_SAMPLES,
    QUICK_N_VALUES,
    Item,
    compute_idf,
    ensure_dataset,
    features_path,
    plan_items,
    write_manifest,
)

MAX_FAILURE_EXAMPLES = 10

_root: Path | None = None
_extractor = None
_quantizer = None


def _init_worker(modality: str, root: str, quantizer) -> None:
    """Extractor por proceso; SIFT de OpenCV no se puede enviar entre procesos."""
    global _root, _extractor, _quantizer
    _root = Path(root)
    _quantizer = quantizer
    if modality == "image":
        import cv2

        from index.multimedia.sift_extractor import SIFTExtractor

        cv2.setNumThreads(1)
        _extractor = SIFTExtractor()
    else:
        from index.multimedia.mfcc_extractor import MFCCExtractor

        _extractor = MFCCExtractor()


def _descriptors_of(path: str) -> tuple[np.ndarray | None, str | None]:
    """Descriptores del objeto, o el motivo por el que el extractor lo rechaza."""
    try:
        return _extractor.extract(str(_root / path)), None
    except ValueError as error:
        return None, str(error).split(":")[0]


def _histogram_of(path: str) -> tuple[np.ndarray | None, str | None]:
    descriptors, failure = _descriptors_of(path)
    if descriptors is None:
        return None, failure
    assignments = _quantizer.predict(descriptors)
    return np.bincount(assignments, minlength=FEATURE_DIM).astype(np.uint16), None


def _collect(worker, items: list[Item], target: int, initargs: tuple, processes: int):
    """Los primeros `target` objetos válidos, en orden. Devuelve (ítems, resultados,
    motivos de rechazo con conteo, ejemplos rechazados)."""
    valid_items, results = [], []
    failures: Counter = Counter()
    examples: list[str] = []
    with multiprocessing.get_context("spawn").Pool(processes, _init_worker, initargs) as pool:
        for item, (result, failure) in zip(
            items, pool.imap(worker, [item.path for item in items], chunksize=32)
        ):
            if result is None:
                failures[failure] += 1
                if len(examples) < MAX_FAILURE_EXAMPLES:
                    examples.append(item.path)
                continue
            valid_items.append(item)
            results.append(result)
            if len(results) == target:
                break
    if len(results) < target:
        raise RuntimeError(f"Solo hay {len(results)} objetos válidos de los {target} pedidos")
    return valid_items, results, failures, examples


def prepare(modality: str, quick: bool, processes: int) -> None:
    from index.multimedia.quantizer import Quantizer

    dataset = DATASETS[modality]
    root, sha256 = ensure_dataset(dataset)
    query_candidates, base_candidates = plan_items(modality, root)
    target = max(QUICK_N_VALUES if quick else N_VALUES)
    samples = QUICK_CODEBOOK_SAMPLES if quick else CODEBOOK_SAMPLES

    # 1) Codebook e IDF: una muestra fija, independiente de N.
    start = time.perf_counter()
    _, documents, _, _ = _collect(
        _descriptors_of, base_candidates[: samples * 2], samples,
        (modality, str(root), None), processes,
    )
    quantizer = Quantizer(k=FEATURE_DIM).fit(np.vstack(documents))
    centroids = np.asarray(quantizer.centroids, dtype=np.float32)
    idf = compute_idf(np.stack([
        np.bincount(quantizer.predict(document), minlength=FEATURE_DIM) for document in documents
    ]))
    codebook_time = time.perf_counter() - start
    print(f"[{modality}] codebook con {samples} objetos: {codebook_time:.1f} s", flush=True)

    # 2) Histogramas de consultas y base.
    initargs = (modality, str(root), quantizer)
    queries, query_counts, query_failures, _ = _collect(
        _histogram_of, query_candidates, NUM_QUERIES, initargs, processes,
    )
    start = time.perf_counter()
    base, base_counts, base_failures, failure_examples = _collect(
        _histogram_of, base_candidates, target, initargs, processes,
    )
    extract_time = time.perf_counter() - start
    print(f"[{modality}] base de {target}: {extract_time:.1f} s", flush=True)

    # 3) Resultado: numpy sin pickle, con rutas relativas al dataset.
    path = features_path(modality, quick)
    np.savez(
        path,
        base_counts=np.stack(base_counts),
        query_counts=np.stack(query_counts),
        base_labels=np.array([item.label for item in base]),
        query_labels=np.array([item.label for item in queries]),
        base_paths=np.array([item.path for item in base]),
        query_paths=np.array([item.path for item in queries]),
        idf=idf,
        centroids=centroids,
    )
    manifest = write_manifest(modality, quick, {
        "modality": modality,
        "dataset": dataset.name,
        "url": dataset.url,
        "archive_sha256": sha256,
        "feature_dim": FEATURE_DIM,
        "base_objects": target,
        "queries": NUM_QUERIES,
        "classes": len({item.label for item in base}),
        "codebook_samples": samples,
        "codebook_time_s": round(codebook_time, 3),
        "extract_time_s": round(extract_time, 3),
        "processes": processes,
        "rejected": dict(base_failures + query_failures),
        "rejected_examples": failure_examples,
        "python": platform.python_version(),
        "platform": platform.platform(),
    })
    print(f"[{modality}] vectores: {path}\n[{modality}] resumen: {manifest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="solo 1,000 objetos de base")
    parser.add_argument("--modality", choices=["image", "audio", "all"], default="all")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    for modality in (DATASETS if args.modality == "all" else [args.modality]):
        prepare(modality, args.quick, args.processes)


if __name__ == "__main__":
    main()
