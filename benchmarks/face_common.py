"""
benchmarks/face_common.py — Datasets (LFW y CelebA), descarga/caché local,
manifiestos reproducibles y carga en el DBMS para el benchmark facial.
"""

from __future__ import annotations

import hashlib
import random
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import requests

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
FACE_APP_ROOT = REPO_ROOT / "apps" / "face-recognition"

for _path in (str(BENCH_DIR), str(REPO_ROOT / "src"), str(FACE_APP_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.exceptions import (
    ConsentRequiredError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from backend.service import FaceRecognitionService

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CORPUS_DIR = BENCH_DIR / "corpus" / "face"
MANIFESTS_DIR = CORPUS_DIR / "manifests"
RESULTS_DIR = BENCH_DIR / "results"

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
)

QUICK_PROFILE: dict[str, int] = {
    "min_images": 3,
    "max_known": 25,
    "max_unknown": 25,
    "enroll_per_identity": 2,
    "probes_per_identity": 2,
    "max_unknown_probes": 50,
    "seed": 42,
}

FULL_PROFILE: dict[str, int] = {
    "min_images": 3,
    "max_known": 500,
    "max_unknown": 150,
    "enroll_per_identity": 2,
    "probes_per_identity": 2,
    "max_unknown_probes": 200,
    "seed": 42,
}

DATASET_DOWNLOAD_SPECS: dict[str, dict[str, Any]] = {
    "lfw": {
        "archive": "lfw-funneled.tgz",
        "urls": (
            "https://ndownloader.figshare.com/files/5976015",
            "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
        ),
        "sha256": "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
    },
    "celeba": {
        "identity_file": "identity_CelebA.txt",
        "identity_urls": (
            "https://huggingface.co/datasets/Yuehao/celeba/resolve/main/identity_CelebA.txt",
        ),
        "archive": "img_align_celeba.zip",
        "archive_urls": (
            "https://huggingface.co/datasets/Yuehao/celeba/resolve/main/img_align_celeba.zip",
        ),
    },
}


def manifest_path_for(dataset: str, *, quick: bool = False) -> Path:
    suffix = "_quick" if quick else ""
    return MANIFESTS_DIR / f"face_{dataset}{suffix}.json"


def results_paths_for(*, quick: bool = False) -> dict[str, Path]:
    suffix = "_quick" if quick else ""
    return {
        "csv": RESULTS_DIR / f"face_comparison_results{suffix}.csv",
        "json": RESULTS_DIR / f"face_comparison_results{suffix}.json",
        "md": RESULTS_DIR / f"face_summary{suffix}.md",
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_from_mirrors(
    urls: tuple[str, ...],
    dest: Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Descarga un archivo de forma atómica (.part) probando los espejos configurados."""
    if dest.exists():
        if expected_sha256 is None or _sha256_file(dest) == expected_sha256:
            return dest
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    errors: list[str] = []

    for url in urls:
        try:
            print(f"[face-data] Descargando {url} -> {dest}", flush=True)
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
                resp.raise_for_status()
                with partial.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if expected_sha256 is not None:
                actual = _sha256_file(partial)
                if actual != expected_sha256:
                    partial.unlink(missing_ok=True)
                    raise ValueError(
                        f"Checksum SHA-256 inesperado para {dest.name}: {actual} != {expected_sha256}"
                    )
            partial.replace(dest)
            return dest
        except Exception as exc:
            partial.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")

    raise RuntimeError(f"No se pudo descargar {dest.name}: {'; '.join(errors)}")


def _extract_tar_gz_safe(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundle.extractall(target_dir, filter="data")


def _extract_celeba_zip_safe(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    root_resolved = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as bundle:
        names = [n for n in bundle.namelist() if not n.endswith("/") and not n.startswith("__MACOSX")]
        has_prefix = any(n.startswith("img_align_celeba/") for n in names)
        dest_base = target_dir if has_prefix else (target_dir / "img_align_celeba")
        dest_base.mkdir(parents=True, exist_ok=True)
        dest_resolved = dest_base.resolve()
        for member in bundle.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX"):
                continue
            target_path = (dest_resolved / member.filename).resolve()
            if root_resolved not in target_path.parents and target_path != root_resolved:
                raise ValueError(f"Entrada zip fuera de ruta permitida: {member.filename}")
            bundle.extract(member, dest_base)


def _scan_lfw_level(root_dir: Path) -> dict[str, list[Path]]:
    subjects: dict[str, list[Path]] = {}
    for person_dir in sorted(root_dir.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith("."):
            continue
        images = sorted(
            p
            for p in person_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            subjects[f"lfw:{person_dir.name}"] = images
    return subjects


def collect_lfw_subjects(root_dir: Path) -> dict[str, list[Path]]:
    """Escanea una carpeta LFW (<root>/<subject_id>/*.jpg o anidada un nivel)."""
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe el directorio LFW: {root_dir}")
    subjects = _scan_lfw_level(root_dir)
    if subjects:
        return subjects

    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            nested = _scan_lfw_level(child)
            if nested:
                return nested
    return {}


def collect_celeba_subjects(root_dir: Path) -> dict[str, list[Path]]:
    """Lee CelebA (identity_CelebA.txt + img_align_celeba/) y agrupa rutas por identidad."""
    identity_file = root_dir / "identity_CelebA.txt"
    img_dir = root_dir / "img_align_celeba"
    if not identity_file.exists():
        raise FileNotFoundError(f"Falta identity_CelebA.txt en {root_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Falta img_align_celeba/ en {root_dir}")

    subjects: dict[str, list[Path]] = defaultdict(list)
    with identity_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename, raw_id = parts
            img_path = img_dir / filename
            if img_path.is_file():
                subjects[f"celeba:{raw_id}"].append(img_path)

    return {sid: sorted(paths) for sid, paths in sorted(subjects.items())}


def ensure_face_dataset(
    dataset: str,
    target_dir: Path | None = None,
    *,
    auto_download: bool = True,
    downloader: Callable[[tuple[str, ...], Path, str | None], Path] | None = None,
) -> Path:
    """Verifica si LFW o CelebA existen localmente; si faltan y auto_download=True, los descarga."""
    root = target_dir or (CORPUS_DIR / dataset)
    fetch = downloader or (
        lambda urls, dest, sha=None: _download_from_mirrors(urls, dest, expected_sha256=sha)
    )

    if dataset == "lfw":
        if root.exists() and collect_lfw_subjects(root):
            print(f"[face-data] LFW ya disponible en {root}", flush=True)
            return root
        if not auto_download:
            raise FileNotFoundError(
                f"LFW no encontrado en {root}. Activa --auto-download o coloca las imágenes manualmente."
            )
        spec = DATASET_DOWNLOAD_SPECS["lfw"]
        archive = fetch(spec["urls"], root / spec["archive"], spec["sha256"])
        print(f"[face-data] Extrayendo {archive.name} en {root}...", flush=True)
        _extract_tar_gz_safe(archive, root)
        if not collect_lfw_subjects(root):
            raise RuntimeError(f"Se extrajo {archive} pero no se encontraron imágenes LFW en {root}")
        return root

    if dataset == "celeba":
        id_file = root / "identity_CelebA.txt"
        img_dir = root / "img_align_celeba"
        if id_file.is_file() and img_dir.is_dir() and any(img_dir.iterdir()):
            print(f"[face-data] CelebA ya disponible en {root}", flush=True)
            return root
        if not auto_download:
            raise FileNotFoundError(
                f"CelebA no encontrado en {root}. Activa --auto-download o coloca los archivos manualmente."
            )
        spec = DATASET_DOWNLOAD_SPECS["celeba"]
        if not id_file.is_file():
            fetch(spec["identity_urls"], id_file, None)
        if not (img_dir.is_dir() and any(img_dir.iterdir())):
            archive = fetch(spec["archive_urls"], root / spec["archive"], None)
            print(f"[face-data] Extrayendo {archive.name} en {root}...", flush=True)
            _extract_celeba_zip_safe(archive, root)
        return root

    raise ValueError(f"Dataset no soportado: {dataset!r}")


def _make_entry(
    path: Path,
    *,
    source: str,
    source_subject_id: str,
    display_label: str | None,
    consent_granted: bool,
    split: str,
    repo_root: Path | None,
) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        rel = str(resolved.relative_to(repo_root.resolve())) if repo_root else str(resolved)
    except ValueError:
        rel = str(resolved)
    return {
        "source": source,
        "source_subject_id": source_subject_id,
        "display_label": display_label,
        "consent_granted": consent_granted,
        "split": split,
        "image_path": str(resolved),
        "relative_path": rel,
        "sha256": _sha256_file(resolved),
    }


def build_benchmark_manifest(
    subjects: dict[str, list[Path]],
    *,
    dataset_name: str,
    source: str,
    min_images: int = 3,
    max_known_identities: int = 25,
    max_unknown_identities: int = 25,
    enroll_per_identity: int = 2,
    max_probes_per_identity: int = 3,
    max_unknown_probes: int = 50,
    seed: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Construye splits deterministas (enroll, probe_known, probe_unknown) con hashes SHA-256."""
    eligible = [
        (sid, paths)
        for sid, paths in sorted(subjects.items())
        if len(paths) >= max(2, min_images)
    ]
    if seed is not None:
        random.Random(seed).shuffle(eligible)
    known_slice = eligible[:max_known_identities]
    unknown_slice = eligible[
        max_known_identities : max_known_identities + max_unknown_identities
    ]

    entries: list[dict[str, Any]] = []
    for sid, paths in known_slice:
        n_enroll = min(enroll_per_identity, len(paths) - 1)
        enroll_paths = paths[:n_enroll]
        probe_paths = paths[n_enroll : n_enroll + max_probes_per_identity]
        for split_name, split_paths in (("enroll", enroll_paths), ("probe_known", probe_paths)):
            for p in split_paths:
                entries.append(
                    _make_entry(
                        p,
                        source=source,
                        source_subject_id=sid,
                        display_label=None,
                        consent_granted=False,
                        split=split_name,
                        repo_root=repo_root,
                    )
                )

    unknown_count = 0
    for sid, paths in unknown_slice:
        for p in paths[:max_probes_per_identity]:
            if unknown_count >= max_unknown_probes:
                break
            entries.append(
                _make_entry(
                    p,
                    source=source,
                    source_subject_id=sid,
                    display_label=None,
                    consent_granted=False,
                    split="probe_unknown",
                    repo_root=repo_root,
                )
            )
            unknown_count += 1
        if unknown_count >= max_unknown_probes:
            break

    return {
        "dataset": dataset_name,
        "source": source,
        "seed": seed,
        "known_identities": len(known_slice),
        "unknown_identities": len(unknown_slice),
        "counts": {
            "enroll": sum(1 for e in entries if e["split"] == "enroll"),
            "probe_known": sum(1 for e in entries if e["split"] == "probe_known"),
            "probe_unknown": sum(1 for e in entries if e["split"] == "probe_unknown"),
        },
        "entries": entries,
    }


def load_manifest_into_dbms(
    manifest: dict[str, Any],
    service: FaceRecognitionService,
    *,
    splits: tuple[str, ...] = ("enroll",),
    face_policy: str = "strict_single",
) -> dict[str, Any]:
    """Registra identidades y enrola muestras de un manifiesto vía FaceRecognitionService."""
    service.ensure_ready()
    subject_to_identity: dict[str, int] = {}
    enrolled = 0
    skipped: list[dict[str, str]] = []

    for entry in manifest.get("entries", []):
        if entry.get("split") not in splits:
            continue
        source = entry["source"]
        subject_id = entry["source_subject_id"]

        if subject_id not in subject_to_identity:
            created = service.register_identity(
                source=source,
                display_label=entry.get("display_label"),
                source_subject_id=subject_id,
                consent_granted=bool(entry.get("consent_granted", False)),
                purpose=entry.get("purpose", "facial_recognition_benchmark"),
                notice_version=entry.get("notice_version", "v1.0"),
            )
            subject_to_identity[subject_id] = int(created["identity_id"])

        identity_id = subject_to_identity[subject_id]
        try:
            service.enroll_face(
                identity_id=identity_id,
                image_path=entry["image_path"],
                relative_path=entry.get("relative_path"),
                split=entry["split"],
                face_policy=face_policy,
            )
            enrolled += 1
        except (NoFaceDetectedError, MultipleFacesDetectedError, ConsentRequiredError) as exc:
            skipped.append({"image_path": entry["image_path"], "reason": str(exc)})

    return {
        "identities_registered": len(subject_to_identity),
        "samples_enrolled": enrolled,
        "skipped": len(skipped),
        "skipped_details": skipped,
        "subject_to_identity": subject_to_identity,
    }
