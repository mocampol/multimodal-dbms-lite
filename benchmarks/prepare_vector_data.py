"""
benchmarks/prepare_vector_data.py — Descarga Tiny ImageNet y Speech Commands,
entrena el codebook con una muestra, extrae SIFT / MFCC en paralelo y guarda el
histograma de palabras de cada objeto en benchmarks/corpus/vector/ (ignorado por
git). El resumen reproducible queda en results/vector_dataset_<modalidad>.json.

Con --dims arma varios codebooks (64, 128, ...) en una sola pasada: cada archivo se
extrae una vez y se cuantiza con todos. La muestra del codebook, los objetos y las
consultas son los mismos para todas las dimensiones. Las dimensiones cuyo archivo ya
existe se saltan, salvo con --force, para no pisar los vectores de un benchmark previo.

Uso:
    python benchmarks/prepare_vector_data.py                     # 100,000 + consultas
    python benchmarks/prepare_vector_data.py --quick             # 1,000 + consultas
    python benchmarks/prepare_vector_data.py --modality image
    python benchmarks/prepare_vector_data.py --dims 128,256,512 --n 10000
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
_quantizers: list = []


def _init_worker(modality: str, root: str, quantizers) -> None:
    """Extractor por proceso; SIFT de OpenCV no se puede enviar entre procesos."""
    global _root, _extractor, _quantizers
    _root = Path(root)
    _quantizers = quantizers or []
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


def _histogram_of(path: str) -> tuple[list[np.ndarray] | None, str | None]:
    """Un histograma por codebook, con los mismos descriptores."""
    descriptors, failure = _descriptors_of(path)
    if descriptors is None:
        return None, failure
    return [
        np.bincount(quantizer.predict(descriptors), minlength=quantizer.k).astype(np.uint16)
        for quantizer in _quantizers
    ], None


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


def pending_dims(modality: str, quick: bool, dims: list[int], force: bool) -> list[int]:
    """Las dimensiones que hay que preparar: sin archivo todavía, o todas con force."""
    return [dim for dim in dims if force or not features_path(modality, quick, dim).exists()]


def prepare(modality: str, quick: bool, processes: int, dims: list[int] | None = None,
            target: int | None = None, force: bool = False) -> None:
    from index.multimedia.quantizer import Quantizer

    requested = dims or [FEATURE_DIM]
    dims = pending_dims(modality, quick, requested, force)
    for dim in sorted(set(requested) - set(dims)):
        print(f"[{modality}] k={dim}: ya existe {features_path(modality, quick, dim)} (usa --force para rehacerlo)")
    if not dims:
        return

    dataset = DATASETS[modality]
    root, sha256 = ensure_dataset(dataset)
    query_candidates, base_candidates = plan_items(modality, root)
    target = target or max(QUICK_N_VALUES if quick else N_VALUES)
    samples = QUICK_CODEBOOK_SAMPLES if quick else CODEBOOK_SAMPLES

    # 1) Codebook e IDF: una muestra fija, independiente de N y de la dimensión.
    start = time.perf_counter()
    _, documents, _, _ = _collect(
        _descriptors_of, base_candidates[: samples * 2], samples,
        (modality, str(root), None), processes,
    )
    sample_time = time.perf_counter() - start
    stacked = np.vstack(documents)
    quantizers, idfs, codebook_times = [], [], []
    for dim in dims:
        start = time.perf_counter()
        quantizer = Quantizer(k=dim).fit(stacked)
        idfs.append(compute_idf(np.stack([
            np.bincount(quantizer.predict(document), minlength=dim) for document in documents
        ])))
        quantizers.append(quantizer)
        codebook_times.append(sample_time + time.perf_counter() - start)
        print(f"[{modality}] codebook k={dim} con {samples} objetos: {codebook_times[-1]:.1f} s", flush=True)

    # 2) Histogramas de consultas y base, con todos los codebooks a la vez.
    initargs = (modality, str(root), quantizers)
    queries, query_counts, query_failures, _ = _collect(
        _histogram_of, query_candidates, NUM_QUERIES, initargs, processes,
    )
    start = time.perf_counter()
    base, base_counts, base_failures, failure_examples = _collect(
        _histogram_of, base_candidates, target, initargs, processes,
    )
    extract_time = time.perf_counter() - start
    print(f"[{modality}] base de {target}: {extract_time:.1f} s", flush=True)

    # 3) Resultado por dimensión: numpy sin pickle, con rutas relativas al dataset.
    for position, (dim, quantizer, idf) in enumerate(zip(dims, quantizers, idfs)):
        path = features_path(modality, quick, dim)
        np.savez(
            path,
            base_counts=np.stack([counts[position] for counts in base_counts]),
            query_counts=np.stack([counts[position] for counts in query_counts]),
            base_labels=np.array([item.label for item in base]),
            query_labels=np.array([item.label for item in queries]),
            base_paths=np.array([item.path for item in base]),
            query_paths=np.array([item.path for item in queries]),
            idf=idf,
            centroids=np.asarray(quantizer.centroids, dtype=np.float32),
        )
        manifest = write_manifest(modality, quick, {
            "modality": modality,
            "dataset": dataset.name,
            "url": dataset.url,
            "archive_sha256": sha256,
            "feature_dim": dim,
            "base_objects": target,
            "queries": NUM_QUERIES,
            "classes": len({item.label for item in base}),
            "codebook_samples": samples,
            "codebook_time_s": round(codebook_times[position], 3),
            "extract_time_s": round(extract_time, 3),
            "dims_in_pass": dims,     # extract_time_s cubre todas estas dimensiones
            "processes": processes,
            "rejected": dict(base_failures + query_failures),
            "rejected_examples": failure_examples,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }, dim)
        print(f"[{modality}] k={dim} vectores: {path}\n[{modality}] k={dim} resumen: {manifest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="solo 1,000 objetos de base")
    parser.add_argument("--modality", choices=["image", "audio", "all"], default="all")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--dims", default=str(FEATURE_DIM), help="tamaños de codebook, p. ej. 128,256,512")
    parser.add_argument("--n", type=int, default=None, help="objetos de base (por defecto el N más grande)")
    parser.add_argument("--force", action="store_true", help="rehace las dimensiones que ya tienen archivo")
    args = parser.parse_args()

    dims = [int(dim) for dim in args.dims.split(",")]
    for modality in (DATASETS if args.modality == "all" else [args.modality]):
        prepare(modality, args.quick, args.processes, dims, args.n, args.force)


if __name__ == "__main__":
    main()
