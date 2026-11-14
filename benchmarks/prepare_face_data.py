"""
benchmarks/prepare_face_data.py — Verifica o descarga LFW y CelebA en
benchmarks/corpus/face/ (ignorado por Git) y genera manifiestos reproducibles.

Uso:
    python benchmarks/prepare_face_data.py --quick          # 25 conocidas + 25 desconocidas
    python benchmarks/prepare_face_data.py                  # escala completa (1,000+ muestras)
    python benchmarks/prepare_face_data.py --dataset lfw --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from face_common import (
    FULL_PROFILE,
    QUICK_PROFILE,
    REPO_ROOT,
    build_benchmark_manifest,
    collect_celeba_subjects,
    collect_lfw_subjects,
    ensure_face_dataset,
    manifest_path_for,
)


def prepare_dataset_manifest(
    dataset: str,
    *,
    input_dir: Path | None = None,
    output_path: Path | None = None,
    quick: bool = False,
    auto_download: bool = True,
    min_images: int | None = None,
    max_known: int | None = None,
    max_unknown: int | None = None,
    enroll_per_identity: int | None = None,
    probes_per_identity: int | None = None,
    max_unknown_probes: int | None = None,
    seed: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Prepara y guarda el manifiesto reproducible de LFW o CelebA."""
    profile = QUICK_PROFILE if quick else FULL_PROFILE
    resolved_dir = ensure_face_dataset(dataset, input_dir, auto_download=auto_download)
    if dataset == "lfw":
        subjects = collect_lfw_subjects(resolved_dir)
        source = "public_lfw"
    elif dataset == "celeba":
        subjects = collect_celeba_subjects(resolved_dir)
        source = "public_celeba"
    else:
        raise ValueError(f"Dataset no soportado: {dataset!r}")

    manifest = build_benchmark_manifest(
        subjects,
        dataset_name=dataset,
        source=source,
        min_images=min_images if min_images is not None else profile["min_images"],
        max_known_identities=max_known if max_known is not None else profile["max_known"],
        max_unknown_identities=max_unknown if max_unknown is not None else profile["max_unknown"],
        enroll_per_identity=(
            enroll_per_identity
            if enroll_per_identity is not None
            else profile["enroll_per_identity"]
        ),
        max_probes_per_identity=(
            probes_per_identity
            if probes_per_identity is not None
            else profile["probes_per_identity"]
        ),
        max_unknown_probes=(
            max_unknown_probes
            if max_unknown_probes is not None
            else profile["max_unknown_probes"]
        ),
        seed=seed if seed is not None else profile["seed"],
        repo_root=REPO_ROOT,
    )
    target = output_path or manifest_path_for(dataset, quick=quick)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepara manifiestos reproducibles de LFW y CelebA en benchmarks/corpus/face/."
    )
    parser.add_argument("--dataset", choices=("lfw", "celeba", "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="Muestra pequeña (25+25 identidades)")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-images", type=int, default=None)
    parser.add_argument("--max-known", type=int, default=None)
    parser.add_argument("--max-unknown", type=int, default=None)
    parser.add_argument("--enroll-per-identity", type=int, default=None)
    parser.add_argument("--probes-per-identity", type=int, default=None)
    parser.add_argument("--max-unknown-probes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Descarga automáticamente LFW/CelebA en benchmarks/corpus/face/ si faltan.",
    )
    args = parser.parse_args(argv)

    datasets = ("lfw", "celeba") if args.dataset == "all" else (args.dataset,)
    if len(datasets) > 1 and (args.input_dir is not None or args.output is not None):
        parser.error("--input-dir y --output solo aplican cuando se selecciona un único --dataset.")

    summary: list[dict[str, Any]] = []
    for ds in datasets:
        target, manifest = prepare_dataset_manifest(
            ds,
            input_dir=args.input_dir,
            output_path=args.output,
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
        summary.append(
            {
                "dataset": ds,
                "output": str(target),
                "counts": manifest["counts"],
            }
        )

    print(json.dumps(summary if len(summary) > 1 else summary[0], indent=2))


if __name__ == "__main__":
    main()
