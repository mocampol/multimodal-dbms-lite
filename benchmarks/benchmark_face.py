"""
benchmarks/benchmark_face.py — Evalúa reconocimiento facial sobre LFW y CelebA
comparando HNSW vs. IVF, agregación y barrido de umbrales (0.25..0.65 en pasos
de 0.05) para calibrar el umbral por defecto del sistema.

Uso:
    python benchmarks/benchmark_face.py --quick          # muestra pequeña (LFW + CelebA)
    python benchmarks/benchmark_face.py                  # escala completa (1,000+ muestras)
    python benchmarks/benchmark_face.py --dataset lfw --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from face_common import (
    DEFAULT_THRESHOLDS,
    load_manifest_into_dbms,
    results_paths_for,
)
from prepare_face_data import prepare_dataset_manifest

from backend.config import FaceAppSettings
from backend.dbms_client import DBMSClient
from backend.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from backend.extractor import build_extractor
from backend.schema_setup import SAMPLES_TABLE
from backend.service import FaceRecognitionService

CSV_FIELDNAMES = [
    "dataset",
    "model",
    "dim",
    "index_type",
    "metric",
    "k",
    "aggregation",
    "threshold",
    "ef_search",
    "n_probe",
    "known_probes",
    "unknown_probes",
    "skipped_no_face",
    "top1_accuracy",
    "true_accept_rate",
    "false_reject_rate",
    "false_id_swap_rate",
    "unknown_reject_rate",
    "false_accept_rate",
    "precision",
    "f1_score",
    "eer_gap",
    "dbms_search_ms_mean",
    "dbms_search_ms_p95",
    "extraction_ms_mean",
    "total_ms_p95",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _extract_probe_cache(
    probes: list[dict[str, Any]],
    service: FaceRecognitionService,
    *,
    subject_to_iid: dict[str, int] | None,
    face_policy: str,
) -> tuple[list[dict[str, Any]], int]:
    """Extrae una sola vez el embedding de cada imagen probe válida."""
    cached: list[dict[str, Any]] = []
    skipped = 0
    for probe in probes:
        expected_iid: int | None = None
        if subject_to_iid is not None:
            expected_iid = subject_to_iid.get(probe["source_subject_id"])
            if expected_iid is None:
                continue
        t0 = time.perf_counter()
        try:
            extracted, _ = service.extractor.extract_primary_face(
                probe["image_path"], policy=face_policy
            )
        except (NoFaceDetectedError, MultipleFacesDetectedError):
            skipped += 1
            continue
        extract_ms = (time.perf_counter() - t0) * 1000.0
        cached.append(
            {
                "image_path": probe["image_path"],
                "expected_iid": expected_iid,
                "embedding": extracted.embedding,
                "extraction_ms": extract_ms,
            }
        )
    return cached, skipped


def _search_cached_probes(
    cached_probes: list[dict[str, Any]],
    service: FaceRecognitionService,
    *,
    k: int,
    index_type: str,
    metric: str,
    ef_search: int,
    n_probe: int,
) -> list[dict[str, Any]]:
    """Ejecuta una sola búsqueda vectorial por probe para una combinación (index_type, k)."""
    searched: list[dict[str, Any]] = []
    for probe in cached_probes:
        t0 = time.perf_counter()
        search_res = service.client.vector_search(
            SAMPLES_TABLE,
            probe["embedding"],
            column="embedding",
            k=k,
            index_type=index_type,
            metric=metric,
            ef_search=ef_search,
            n_probe=n_probe,
        )
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        searched.append(
            {
                "expected_iid": probe["expected_iid"],
                "extraction_ms": probe["extraction_ms"],
                "dbms_search_ms": float(search_res.get("search_ms", 0.0)),
                "total_ms": probe["extraction_ms"] + roundtrip_ms,
                "raw_results": search_res.get("results", []),
            }
        )
    return searched


def _evaluate_threshold_slice(
    *,
    manifest: dict[str, Any],
    service: FaceRecognitionService,
    index_type: str,
    metric: str,
    k: int,
    aggregation: str,
    threshold: float,
    ef_search: int | None,
    n_probe: int | None,
    known_aggregated: list[dict[str, Any]],
    unknown_aggregated: list[dict[str, Any]],
    skipped_no_face: int,
) -> dict[str, Any]:
    known_total = len(known_aggregated)
    unknown_total = len(unknown_aggregated)
    known_matched_correct = 0
    known_wrong_identity = 0
    known_top1_unthresholded = 0
    known_false_unknown = 0
    unknown_rejected = 0
    unknown_false_accept = 0

    dbms_latencies: list[float] = []
    extract_latencies: list[float] = []
    total_latencies: list[float] = []

    for item in known_aggregated:
        dbms_latencies.append(item["dbms_search_ms"])
        extract_latencies.append(item["extraction_ms"])
        total_latencies.append(item["total_ms"])
        candidates = item["candidates"]
        best = candidates[0] if candidates else None
        if best is not None and best["identity_id"] == item["expected_iid"]:
            known_top1_unthresholded += 1
        if best is not None and best["score"] >= threshold:
            if best["identity_id"] == item["expected_iid"]:
                known_matched_correct += 1
            else:
                known_wrong_identity += 1
        else:
            known_false_unknown += 1

    for item in unknown_aggregated:
        dbms_latencies.append(item["dbms_search_ms"])
        extract_latencies.append(item["extraction_ms"])
        total_latencies.append(item["total_ms"])
        candidates = item["candidates"]
        best = candidates[0] if candidates else None
        if best is not None and best["score"] >= threshold:
            unknown_false_accept += 1
        else:
            unknown_rejected += 1

    tar = (known_matched_correct / known_total) if known_total else 0.0
    frr = (known_false_unknown / known_total) if known_total else 0.0
    swap_rate = (known_wrong_identity / known_total) if known_total else 0.0
    far = (unknown_false_accept / unknown_total) if unknown_total else 0.0
    unknown_rej = (unknown_rejected / unknown_total) if unknown_total else 0.0

    tp = known_matched_correct
    fp = unknown_false_accept + known_wrong_identity
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = (2.0 * precision * tar / (precision + tar)) if (precision + tar) else 0.0

    return {
        "dataset": manifest.get("dataset"),
        "model": service.extractor.model_name,
        "dim": service.extractor.embedding_dim,
        "index_type": index_type,
        "metric": metric,
        "k": k,
        "aggregation": aggregation,
        "threshold": float(threshold),
        "ef_search": ef_search,
        "n_probe": n_probe,
        "known_probes": known_total,
        "unknown_probes": unknown_total,
        "skipped_no_face": skipped_no_face,
        "top1_accuracy": round(known_top1_unthresholded / known_total, 4) if known_total else 0.0,
        "true_accept_rate": round(tar, 4),
        "false_reject_rate": round(frr, 4),
        "false_id_swap_rate": round(swap_rate, 4),
        "unknown_reject_rate": round(unknown_rej, 4),
        "false_accept_rate": round(far, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "eer_gap": round(abs(far - frr), 4),
        "dbms_search_ms_mean": round(statistics.fmean(dbms_latencies), 3) if dbms_latencies else 0.0,
        "dbms_search_ms_p95": round(_percentile(dbms_latencies, 95), 3),
        "extraction_ms_mean": round(statistics.fmean(extract_latencies), 3) if extract_latencies else 0.0,
        "total_ms_p95": round(_percentile(total_latencies, 95), 3),
    }


def run_benchmark(
    manifest: dict[str, Any],
    service: FaceRecognitionService,
    *,
    index_types: tuple[str, ...] = ("hnsw", "ivf"),
    k_values: tuple[int, ...] = (1, 3, 5),
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    aggregations: tuple[str, ...] = ("max",),
    ef_search: int | None = None,
    n_probe: int | None = None,
    metric: str = "cosine",
    allow_dataset_identities: bool = True,
    face_policy: str = "largest",
) -> list[dict[str, Any]]:
    """Evalúa latencia y calidad (TAR, FAR, rechazo desconocido, F1, Top-1) sobre probe_known y probe_unknown."""
    service._ensure_ready_once()
    for agg in aggregations:
        if agg not in ("max", "mean_top"):
            raise ValueError("aggregation debe ser 'max' o 'mean_top'")

    entries = manifest.get("entries", [])
    known_probes = [e for e in entries if e.get("split") == "probe_known"]
    unknown_probes = [e for e in entries if e.get("split") == "probe_unknown"]

    identities = service.list_identities()
    subject_to_iid = {
        str(item["source_subject_id"]): int(item["identity_id"])
        for item in identities
    }

    effective_ef = ef_search if ef_search is not None else service.settings.hnsw_ef_search
    effective_probe = n_probe if n_probe is not None else service.settings.ivf_n_probe

    known_cache, skipped_known = _extract_probe_cache(
        known_probes, service, subject_to_iid=subject_to_iid, face_policy=face_policy
    )
    unknown_cache, skipped_unknown = _extract_probe_cache(
        unknown_probes, service, subject_to_iid=None, face_policy=face_policy
    )
    skipped_total = skipped_known + skipped_unknown

    rows: list[dict[str, Any]] = []
    for index_type in index_types:
        for k in k_values:
            known_searched = _search_cached_probes(
                known_cache,
                service,
                k=k,
                index_type=index_type,
                metric=metric,
                ef_search=effective_ef,
                n_probe=effective_probe,
            )
            unknown_searched = _search_cached_probes(
                unknown_cache,
                service,
                k=k,
                index_type=index_type,
                metric=metric,
                ef_search=effective_ef,
                n_probe=effective_probe,
            )

            for aggregation in aggregations:
                known_agg = [
                    {
                        **item,
                        "candidates": service._aggregate_candidates(
                            item["raw_results"],
                            aggregation=aggregation,
                            allow_dataset_identities=allow_dataset_identities,
                        ),
                    }
                    for item in known_searched
                ]
                unknown_agg = [
                    {
                        **item,
                        "candidates": service._aggregate_candidates(
                            item["raw_results"],
                            aggregation=aggregation,
                            allow_dataset_identities=allow_dataset_identities,
                        ),
                    }
                    for item in unknown_searched
                ]

                for threshold in thresholds:
                    rows.append(
                        _evaluate_threshold_slice(
                            manifest=manifest,
                            service=service,
                            index_type=index_type,
                            metric=metric,
                            k=k,
                            aggregation=aggregation,
                            threshold=threshold,
                            ef_search=ef_search,
                            n_probe=n_probe,
                            known_aggregated=known_agg,
                            unknown_aggregated=unknown_agg,
                            skipped_no_face=skipped_total,
                        )
                    )
    return rows


def recommend_threshold(
    rows: list[dict[str, Any]],
    *,
    max_far: float = 0.02,
) -> dict[str, Any]:
    """Selecciona el umbral óptimo en el conjunto global maximizando TAR/F1 bajo FAR <= max_far."""
    if not rows:
        return {"recommended": None, "strict_1pct": None, "eer_point": None}

    by_config: dict[tuple[str, int, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["index_type"]),
            int(row["k"]),
            str(row["aggregation"]),
            float(row["threshold"]),
        )
        by_config.setdefault(key, []).append(row)

    aggregated_configs: list[dict[str, Any]] = []
    for (idx_type, k, agg, thr), group in by_config.items():
        worst_far = max(float(r["false_accept_rate"]) for r in group)
        min_tar = min(float(r["true_accept_rate"]) for r in group)
        mean_tar = statistics.fmean(float(r["true_accept_rate"]) for r in group)
        mean_f1 = statistics.fmean(float(r["f1_score"]) for r in group)
        worst_swap = max(float(r["false_id_swap_rate"]) for r in group)
        mean_eer_gap = statistics.fmean(float(r["eer_gap"]) for r in group)
        aggregated_configs.append(
            {
                "datasets": sorted({str(r["dataset"]) for r in group}),
                "index_type": idx_type,
                "k": k,
                "aggregation": agg,
                "threshold": thr,
                "worst_far": round(worst_far, 4),
                "min_tar": round(min_tar, 4),
                "mean_tar": round(mean_tar, 4),
                "mean_f1": round(mean_f1, 4),
                "worst_swap": round(worst_swap, 4),
                "mean_eer_gap": round(mean_eer_gap, 4),
                # Campos compatibles cuando hay 1 único dataset
                "true_accept_rate": round(mean_tar, 4),
                "false_accept_rate": round(worst_far, 4),
                "f1_score": round(mean_f1, 4),
            }
        )

    def _rank_key(cfg: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(cfg["mean_tar"]),
            float(cfg["mean_f1"]),
            -float(cfg["worst_far"]),
            -float(cfg["worst_swap"]),
            -float(cfg["threshold"]),
        )

    within_budget = [c for c in aggregated_configs if float(c["worst_far"]) <= max_far]
    strict_rows = [c for c in aggregated_configs if float(c["worst_far"]) <= 0.01]

    best = (
        max(within_budget, key=_rank_key)
        if within_budget
        else min(aggregated_configs, key=lambda c: (float(c["worst_far"]), -float(c["mean_f1"])))
    )
    strict_best = max(strict_rows, key=_rank_key) if strict_rows else None
    eer_best = min(aggregated_configs, key=lambda c: (float(c["mean_eer_gap"]), -float(c["mean_f1"])))

    return {
        "max_far_target": max_far,
        "met_far_target": bool(within_budget),
        "recommended": best,
        "strict_1pct": strict_best,
        "eer_point": eer_best,
    }


def write_csv_results(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDNAMES})


def format_markdown_report(rows: list[dict[str, Any]], recommendation: dict[str, Any]) -> str:
    """Formatea una tabla Markdown con métricas de calidad y latencia."""
    headers = (
        "| Dataset | Index | k | Agg | Threshold | TAR | FAR | FRR | Swap | F1 | DBMS p95 (ms) | Total p95 (ms) |\n"
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {r['dataset']} | {r['index_type']} | {r['k']} | {r['aggregation']} | "
        f"{r['threshold']:.2f} | {r['true_accept_rate']:.4f} | {r['false_accept_rate']:.4f} | "
        f"{r['false_reject_rate']:.4f} | {r['false_id_swap_rate']:.4f} | {r['f1_score']:.4f} | "
        f"{r['dbms_search_ms_p95']:.2f} | {r['total_ms_p95']:.2f} |"
        for r in rows
    ]
    rec = recommendation.get("recommended")
    footer = ""
    if rec:
        footer = (
            f"\n**Umbral recomendado (`FAR <= {recommendation['max_far_target']:.2%}`):** "
            f"`{rec['threshold']:.2f}` (`{rec['index_type']}`, `k={rec['k']}`, `agg={rec['aggregation']}`, "
            f"`TAR_mean={rec['mean_tar']:.4f}`, `FAR_max={rec['worst_far']:.4f}`, `F1_mean={rec['mean_f1']:.4f}`)\n"
        )
    return headers + "\n".join(lines) + footer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark facial en multimodal-dbms-lite (LFW / CelebA, HNSW vs IVF, calibración de threshold)."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset", choices=("lfw", "celeba", "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="Muestra pequeña (25+25 identidades)")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Descarga automáticamente LFW/CelebA en benchmarks/corpus/face/ si faltan.",
    )
    parser.add_argument("--min-images", type=int, default=None)
    parser.add_argument("--max-known", type=int, default=None)
    parser.add_argument("--max-unknown", type=int, default=None)
    parser.add_argument("--enroll-per-identity", type=int, default=None)
    parser.add_argument("--probes-per-identity", type=int, default=None)
    parser.add_argument("--max-unknown-probes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reset-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limpia identidades/muestras previas antes de cargar cada dataset.",
    )
    parser.add_argument("--dbms-url", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", choices=("sface", "facenet"), default="sface")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--index-types", nargs="+", default=["hnsw", "ivf"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--aggregations", nargs="+", choices=("max", "mean_top"), default=["max", "mean_top"])
    parser.add_argument("--ef-search", type=int, default=None)
    parser.add_argument("--n-probe", type=int, default=None)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--max-far", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    args = parser.parse_args(argv)

    default_paths = results_paths_for(quick=args.quick)
    json_out = args.output or default_paths["json"]
    csv_out = args.csv_out or default_paths["csv"]
    md_out = args.summary_md or default_paths["md"]

    settings = FaceAppSettings.from_env(
        dbms_api_url=args.dbms_url,
        extractor_backend=args.backend,
        embedding_dim=args.dim,
    )

    all_rows: list[dict[str, Any]] = []
    client = DBMSClient(base_url=settings.dbms_api_url)
    try:
        extractor = build_extractor(settings)
        service = FaceRecognitionService(settings, client, extractor)

        if args.manifest is not None:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            all_rows.extend(
                run_benchmark(
                    manifest,
                    service,
                    index_types=tuple(args.index_types),
                    k_values=tuple(args.k_values),
                    thresholds=tuple(args.thresholds),
                    aggregations=tuple(args.aggregations),
                    ef_search=args.ef_search,
                    n_probe=args.n_probe,
                    metric=args.metric,
                )
            )
        else:
            datasets = ("lfw", "celeba") if args.dataset == "all" else (args.dataset,)
            if len(datasets) > 1 and args.input_dir is not None:
                parser.error("--input-dir solo aplica cuando se selecciona un único --dataset.")
            for ds in datasets:
                _, manifest = prepare_dataset_manifest(
                    ds,
                    input_dir=args.input_dir,
                    quick=args.quick,
                    auto_download=args.auto_download,
                    min_images=args.min_images,
                    max_known=args.max_known,
                    max_unknown=args.max_unknown,
                    enroll_per_identity=args.enroll_per_identity,
                    probes_per_identity=args.probes_per_identity,
                    max_unknown_probes=args.max_unknown_probes,
                    seed=args.seed,
                )
                if args.reset_db:
                    service.purge_all_faces(samples_only=False)
                load_manifest_into_dbms(manifest, service)
                all_rows.extend(
                    run_benchmark(
                        manifest,
                        service,
                        index_types=tuple(args.index_types),
                        k_values=tuple(args.k_values),
                        thresholds=tuple(args.thresholds),
                        aggregations=tuple(args.aggregations),
                        ef_search=args.ef_search,
                        n_probe=args.n_probe,
                        metric=args.metric,
                    )
                )
    finally:
        client.close()

    recommendation = recommend_threshold(all_rows, max_far=args.max_far)
    payload = {"rows": all_rows, "recommendation": recommendation}

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv_results(all_rows, csv_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_markdown_report(all_rows, recommendation), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows_count": len(all_rows),
                "outputs": {
                    "csv": str(csv_out),
                    "json": str(json_out),
                    "summary_md": str(md_out),
                },
                "recommendation": recommendation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
