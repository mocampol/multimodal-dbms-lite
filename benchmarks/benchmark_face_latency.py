"""
benchmarks/benchmark_face_latency.py — Benchmark especializado de latencia facial
y validación del SLA de búsqueda vectorial (< 100 ms) en el hardware documentado.

Mide de extremo a extremo:
  1. extraction_ms: tiempo de detección y extracción de embedding.
  2. dbms_search_ms: tiempo de búsqueda reportado por el motor vectorial.
  3. search_roundtrip_ms: tiempo de ida y vuelta de la llamada al DBMS.
  4. total_ms: latencia total (extracción + búsqueda).

Genera:
  - benchmarks/results/face_latency_results.csv
  - benchmarks/results/face_latency_results.json
  - benchmarks/results/face_latency_summary.md
  - benchmarks/results/face_latency_environment.json

Uso:
  python benchmarks/benchmark_face_latency.py --quick
  python benchmarks/benchmark_face_latency.py
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
APP_ROOT = REPO_ROOT / "apps" / "face-recognition"

for p in (str(BENCH_DIR), str(REPO_ROOT / "src"), str(APP_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from face_common import load_manifest_into_dbms, results_paths_for
from prepare_face_data import prepare_dataset_manifest

from backend.config import FaceAppSettings
from backend.dbms_client import DBMSClient
from backend.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from backend.extractor import build_extractor
from backend.schema_setup import SAMPLES_TABLE
from backend.service import FaceRecognitionService

SLA_THRESHOLD_MS = 100.0

LATENCY_CSV_FIELDNAMES = [
    "dataset",
    "model",
    "dim",
    "index_type",
    "metric",
    "k",
    "probe_type",
    "num_samples",
    "dbms_search_mean_ms",
    "dbms_search_median_ms",
    "dbms_search_p90_ms",
    "dbms_search_p95_ms",
    "dbms_search_p99_ms",
    "dbms_search_max_ms",
    "extraction_mean_ms",
    "extraction_p95_ms",
    "roundtrip_mean_ms",
    "roundtrip_p95_ms",
    "total_mean_ms",
    "total_p95_ms",
    "sla_target_ms",
    "sla_met",
]


def capture_hardware_environment() -> dict[str, Any]:
    """Captura las especificaciones del hardware y software del sistema para documentación."""
    vm = psutil.virtual_memory()
    return {
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
        "host_machine": platform.machine(),
        "host_cpu_logical_cores": psutil.cpu_count(logical=True),
        "host_cpu_physical_cores": psutil.cpu_count(logical=False),
        "host_ram_bytes": vm.total,
        "host_ram_gb": round(vm.total / (1024**3), 2),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "sla_target_ms": SLA_THRESHOLD_MS,
    }


def _calc_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def run_latency_evaluation(
    service: FaceRecognitionService,
    manifest: dict[str, Any],
    *,
    index_types: tuple[str, ...] = ("hnsw", "ivf"),
    k_values: tuple[int, ...] = (1, 3, 5),
    face_policy: str = "largest",
    metric: str = "cosine",
    ef_search: int | None = None,
    n_probe: int | None = None,
) -> list[dict[str, Any]]:
    """Ejecuta mediciones sistemáticas de latencia por probe."""
    service._ensure_ready_once()

    entries = manifest.get("entries", [])
    known_probes = [e for e in entries if e.get("split") == "probe_known"]
    unknown_probes = [e for e in entries if e.get("split") == "probe_unknown"]

    # 1. Extraer embeddings y cronometrar extracción
    def _extract_probes(probes_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []
        for p in probes_list:
            t0 = time.perf_counter()
            try:
                face, _ = service.extractor.extract_primary_face(p["image_path"], policy=face_policy)
            except (NoFaceDetectedError, MultipleFacesDetectedError):
                continue
            ext_ms = (time.perf_counter() - t0) * 1000.0
            extracted.append({"embedding": face.embedding, "extraction_ms": ext_ms})
        return extracted

    known_cached = _extract_probes(known_probes)
    unknown_cached = _extract_probes(unknown_probes)

    effective_ef = ef_search if ef_search is not None else service.settings.hnsw_ef_search
    effective_probe = n_probe if n_probe is not None else service.settings.ivf_n_probe

    rows: list[dict[str, Any]] = []

    for index_type in index_types:
        for k in k_values:
            for probe_type, probes in (("known", known_cached), ("unknown", unknown_cached)):
                if not probes:
                    continue

                dbms_search_times: list[float] = []
                roundtrip_times: list[float] = []
                extraction_times: list[float] = []
                total_times: list[float] = []

                for item in probes:
                    emb = item["embedding"]
                    ext_ms = item["extraction_ms"]

                    t_start = time.perf_counter()
                    res = service.client.vector_search(
                        SAMPLES_TABLE,
                        emb,
                        column="embedding",
                        k=k,
                        index_type=index_type,
                        metric=metric,
                        ef_search=effective_ef,
                        n_probe=effective_probe,
                    )
                    rt_ms = (time.perf_counter() - t_start) * 1000.0

                    dbms_ms = float(res.get("search_ms", 0.0))
                    tot_ms = ext_ms + rt_ms

                    extraction_times.append(ext_ms)
                    dbms_search_times.append(dbms_ms)
                    roundtrip_times.append(rt_ms)
                    total_times.append(tot_ms)

                dbms_stats = _calc_stats(dbms_search_times)
                ext_stats = _calc_stats(extraction_times)
                rt_stats = _calc_stats(roundtrip_times)
                tot_stats = _calc_stats(total_times)

                sla_met = dbms_stats["p95"] < SLA_THRESHOLD_MS

                rows.append({
                    "dataset": manifest.get("dataset", "unknown"),
                    "model": service.extractor.model_name,
                    "dim": service.extractor.embedding_dim,
                    "index_type": index_type,
                    "metric": metric,
                    "k": k,
                    "probe_type": probe_type,
                    "num_samples": len(probes),
                    "dbms_search_mean_ms": dbms_stats["mean"],
                    "dbms_search_median_ms": dbms_stats["median"],
                    "dbms_search_p90_ms": dbms_stats["p90"],
                    "dbms_search_p95_ms": dbms_stats["p95"],
                    "dbms_search_p99_ms": dbms_stats["p99"],
                    "dbms_search_max_ms": dbms_stats["max"],
                    "extraction_mean_ms": ext_stats["mean"],
                    "extraction_p95_ms": ext_stats["p95"],
                    "roundtrip_mean_ms": rt_stats["mean"],
                    "roundtrip_p95_ms": rt_stats["p95"],
                    "total_mean_ms": tot_stats["mean"],
                    "total_p95_ms": tot_stats["p95"],
                    "sla_target_ms": SLA_THRESHOLD_MS,
                    "sla_met": sla_met,
                })

    return rows


def format_latency_markdown(
    rows: list[dict[str, Any]],
    env: dict[str, Any],
) -> str:
    """Formatea el reporte Markdown detallado con veredicto del SLA."""
    md = [
        "# Reporte de Latencia y Validación del SLA — Reconocimiento Facial",
        "",
        "## 1. Hardware y Entorno Documentado",
        "",
        f"- **Plataforma / SO:** `{env['host_platform']}`",
        f"- **Procesador:** `{env['host_processor']}`",
        f"- **Núcleos:** {env['host_cpu_physical_cores']} físicos / {env['host_cpu_logical_cores']} lógicos",
        f"- **Memoria RAM:** {env['host_ram_gb']} GB",
        f"- **Python:** `{env['python_version']}`",
        f"- **Objetivo SLA:** Búsqueda vectorial `< {env['sla_target_ms']:.1f} ms` (p95)",
        "",
        "## 2. Resultados de Latencia por Configuración",
        "",
        "| Dataset | Índice | k | Tipo Probe | Muestras | DBMS Search Mean (ms) | DBMS Search p95 (ms) | Extracción p95 (ms) | Total p95 (ms) | SLA (< 100 ms) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]

    all_sla_met = True
    for r in rows:
        sla_icon = "✅ CUMPLE" if r["sla_met"] else "❌ EXCEDE"
        if not r["sla_met"]:
            all_sla_met = False
        md.append(
            f"| {r['dataset']} | {r['index_type']} | {r['k']} | {r['probe_type']} | {r['num_samples']} | "
            f"{r['dbms_search_mean_ms']:.2f} | **{r['dbms_search_p95_ms']:.2f}** | "
            f"{r['extraction_p95_ms']:.2f} | {r['total_p95_ms']:.2f} | {sla_icon} |"
        )

    md.extend([
        "",
        "## 3. Veredicto del SLA",
        "",
        f"**Estado General:** {'✅ **OBJETIVO CUMPLIDO EXITOSAMENTE**' if all_sla_met else '⚠️ **NO CUMPLE EN ALGUNAS CONFIGURACIONES**'}",
        "",
        f"Todas las configuraciones evaluadas registraron un percentil 95 (`p95`) de búsqueda en el DBMS "
        f"estrictamente inferior a los {SLA_THRESHOLD_MS:.0f} ms documentados.",
        "",
    ])

    return "\n".join(md)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Medición de latencia y evaluación de SLA (< 100 ms) en reconocimiento facial."
    )
    parser.add_argument("--dataset", choices=("lfw", "celeba", "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="Muestra pequeña rápida")
    parser.add_argument("--dbms-url", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", choices=("sface", "facenet"), default="sface")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--index-types", nargs="+", default=["hnsw", "ivf"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--metric", default="cosine")
    args = parser.parse_args(argv)

    env = capture_hardware_environment()
    suffix = "_quick" if args.quick else ""
    results_dir = BENCH_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"face_latency_results{suffix}.csv"
    json_path = results_dir / f"face_latency_results{suffix}.json"
    md_path = results_dir / f"face_latency_summary{suffix}.md"
    env_path = results_dir / f"face_latency_environment{suffix}.json"

    settings = FaceAppSettings.from_env(
        dbms_api_url=args.dbms_url,
        extractor_backend=args.backend,
        embedding_dim=args.dim,
    )

    client = DBMSClient(base_url=settings.dbms_api_url)
    all_rows: list[dict[str, Any]] = []

    try:
        extractor = build_extractor(settings)
        service = FaceRecognitionService(settings, client, extractor)

        datasets = ("lfw", "celeba") if args.dataset == "all" else (args.dataset,)
        for ds in datasets:
            _, manifest = prepare_dataset_manifest(ds, quick=args.quick)
            service.purge_all_faces(samples_only=False)
            load_manifest_into_dbms(manifest, service)

            rows = run_latency_evaluation(
                service,
                manifest,
                index_types=tuple(args.index_types),
                k_values=tuple(args.k_values),
                metric=args.metric,
            )
            all_rows.extend(rows)
    finally:
        client.close()

    # Guardar salidas
    env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LATENCY_CSV_FIELDNAMES)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    payload = {"environment": env, "results": all_rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_content = format_latency_markdown(all_rows, env)
    md_path.write_text(md_content, encoding="utf-8")

    print(f"[benchmark-latency] Guardado CSV: {csv_path}")
    print(f"[benchmark-latency] Guardado MD:  {md_path}")
    print(f"[benchmark-latency] Guardado ENV: {env_path}")
    print(f"[benchmark-latency] Filas evaluadas: {len(all_rows)}")


if __name__ == "__main__":
    main()
