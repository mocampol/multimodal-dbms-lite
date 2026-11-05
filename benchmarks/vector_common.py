"""
benchmarks/vector_common.py — Datasets, partición y vectores compartidos por la
comparación de búsqueda por similitud (issue #85).

Un objeto multimedia pasa por extracción (SIFT / MFCC) y cuantización una sola
vez, en prepare_vector_data.py, que guarda su histograma de palabras. Los
vectores se derivan de esos histogramas sin volver a abrir los archivos:
    - "normalized": TF-IDF con norma L2, lo que produce el motor
      (index/multimedia/bow_tfidf.py). Con norma 1, ||a-b||^2 = 2 - 2*cos(a, b),
      así que euclidiana y coseno ordenan igual.
    - "raw": conteos * IDF, sin normalizar, donde las dos métricas sí pueden
      discrepar.

Codebook e IDF se entrenan una vez con una muestra y se reutilizan para todo N,
por lo que los vectores de N = 1,000 son prefijo de los de N = 100,000. El motor,
en cambio, los entrena con el corpus completo de cada índice.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
SRC_DIR = BENCH_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

N_VALUES = [1_000, 10_000, 100_000]
QUICK_N_VALUES = [1_000]
SEED = 42
QUERY_SEED = 43
FEATURE_DIM = 64                    # dimensión fija de los índices del motor (catalog.py)
NUM_QUERIES = 200
CODEBOOK_SAMPLES = 5_000
QUICK_CODEBOOK_SAMPLES = 1_000
VARIANTS = ("normalized", "raw")
MODALITIES = ("image", "audio")

CORPUS_DIR = BENCH_DIR / "corpus" / "vector"
RESULTS_DIR = BENCH_DIR / "results"


@dataclass(frozen=True)
class Dataset:
    modality: str
    name: str
    url: str
    sha256: str
    archive: str


DATASETS = {
    "image": Dataset(
        modality="image",
        name="tiny_imagenet",
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        sha256="6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821",
        archive="tiny-imagenet-200.zip",
    ),
    "audio": Dataset(
        modality="audio",
        name="speech_commands",
        url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        sha256="af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
        archive="speech_commands_v0.02.tar.gz",
    ),
}


# =============================================================================
# Descarga y extracción
# =============================================================================

def dataset_dir(dataset: Dataset) -> Path:
    return CORPUS_DIR / dataset.name


def data_dir(dataset: Dataset) -> Path:
    return dataset_dir(dataset) / "data"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset: Dataset) -> Path:
    """Descarga el archivo si falta (vía .part, para no dejar uno a medias)."""
    target = dataset_dir(dataset) / dataset.archive
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"Descargando {dataset.url} ...", flush=True)
    urllib.request.urlretrieve(dataset.url, partial)
    partial.replace(target)
    return target


def _wanted_image_member(name: str) -> bool:
    return name.endswith((".JPEG", "wnids.txt", "val_annotations.txt")) and "/test/" not in name


def _wanted_audio_member(name: str) -> bool:
    return name.endswith(".wav") and "_background_noise_" not in name


def extract(dataset: Dataset, archive: Path) -> Path:
    """Extrae solo imágenes / audios y etiquetas; las rutas peligrosas se rechazan."""
    root = data_dir(dataset)
    marker = dataset_dir(dataset) / ".extracted"
    if marker.exists():
        return root
    print(f"Extrayendo {archive.name} ...", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    if dataset.modality == "image":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(root, [n for n in bundle.namelist() if _wanted_image_member(n)])
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            members = (m for m in bundle if m.isfile() and _wanted_audio_member(m.name))
            bundle.extractall(root, members=members, filter="data")
    marker.write_text("ok", encoding="utf-8")
    return root


def ensure_dataset(dataset: Dataset) -> tuple[Path, str]:
    """Deja el dataset listo y devuelve (raíz de datos, sha256 del archivo)."""
    archive = download(dataset)
    sha256 = file_sha256(archive)
    if sha256 != dataset.sha256:
        raise ValueError(
            f"{archive.name}: sha256 {sha256} no coincide con el esperado {dataset.sha256}"
        )
    return extract(dataset, archive), sha256


# =============================================================================
# Partición: objetos candidatos a consulta y a base
# =============================================================================

@dataclass(frozen=True)
class Item:
    path: str                       # relativa a data_dir(dataset), con '/'
    label: str


def list_image_items(root: Path) -> tuple[list[Item], list[Item]]:
    """Tiny ImageNet: (train, val), con la etiqueta de cada imagen."""
    base = root / "tiny-imagenet-200"
    train = [
        Item(path.relative_to(root).as_posix(), path.parent.parent.name)
        for path in (base / "train").glob("*/images/*.JPEG")
    ]
    val = []
    with open(base / "val" / "val_annotations.txt", encoding="utf-8") as handle:
        for line in handle:
            filename, label, *_ = line.split("\t")
            val.append(Item(f"tiny-imagenet-200/val/images/{filename}", label))
    return train, val


def list_audio_items(root: Path) -> list[Item]:
    """Speech Commands: la etiqueta de cada clip es su carpeta."""
    return [
        Item(path.relative_to(root).as_posix(), path.parent.name)
        for path in root.glob("*/*.wav")
    ]


def shuffled(items: list[Item], seed: int) -> list[Item]:
    """Orden que no depende de cómo liste los archivos el sistema operativo."""
    ordered = sorted(items, key=lambda item: item.path)
    random.Random(seed).shuffle(ordered)
    return ordered


def plan_items(modality: str, root: Path) -> tuple[list[Item], list[Item]]:
    """(candidatos a consulta, candidatos a base), ambos en orden reproducible y
    sin objetos en común. Son más que los necesarios porque SIFT y MFCC rechazan
    objetos sin puntos clave o en silencio; se usan los primeros válidos.

    Imágenes: consultas de la validación oficial, y lo que sobra de ella completa
    la base. Audio: se reservan objetos al azar, porque la partición oficial de
    entrenamiento no llega a 100,000 clips; un mismo hablante puede aparecer en
    consulta y base.
    """
    reserve = NUM_QUERIES * 2
    if modality == "image":
        train, val = list_image_items(root)
        val = shuffled(val, QUERY_SEED)
        return val[:reserve], shuffled(train, SEED) + val[reserve:]
    if modality == "audio":
        everything = shuffled(list_audio_items(root), SEED)
        return everything[:reserve], everything[reserve:]
    raise ValueError(f"Modalidad desconocida: {modality}")


# =============================================================================
# Vectores
# =============================================================================

def compute_idf(counts: np.ndarray) -> np.ndarray:
    """IDF suavizado del motor (BoWTFIDF.fit) a partir de histogramas."""
    df = (counts > 0).sum(axis=0)
    return (np.log((len(counts) + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def build_vectors(counts: np.ndarray, idf: np.ndarray, variant: str) -> np.ndarray:
    """Vectores (N, K) desde histogramas. "normalized" equivale a BoWTFIDF.compute."""
    counts = np.asarray(counts, dtype=np.float32)
    if variant == "raw":
        return counts * idf
    if variant == "normalized":
        tfidf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        return np.divide(tfidf, norms, out=np.zeros_like(tfidf), where=norms > 1e-12)
    raise ValueError(f"Variante desconocida: {variant}")


def features_path(modality: str, quick: bool = False) -> Path:
    suffix = "_quick" if quick else ""
    return dataset_dir(DATASETS[modality]) / f"features_k{FEATURE_DIM}{suffix}.npz"


def manifest_path(modality: str, quick: bool = False) -> Path:
    suffix = "_quick" if quick else ""
    return RESULTS_DIR / f"vector_dataset_{modality}{suffix}.json"


@dataclass(frozen=True)
class FeatureSet:
    modality: str
    base_counts: np.ndarray
    query_counts: np.ndarray
    base_labels: np.ndarray
    query_labels: np.ndarray
    base_paths: np.ndarray
    query_paths: np.ndarray
    idf: np.ndarray

    def base_vectors(self, n: int, variant: str) -> np.ndarray:
        if n > len(self.base_counts):
            raise ValueError(f"Se pidieron {n} objetos, pero hay {len(self.base_counts)}")
        return build_vectors(self.base_counts[:n], self.idf, variant)

    def query_vectors(self, variant: str) -> np.ndarray:
        return build_vectors(self.query_counts, self.idf, variant)


def load_features(modality: str, quick: bool = False) -> FeatureSet:
    with np.load(features_path(modality, quick), allow_pickle=False) as data:
        return FeatureSet(
            modality=modality,
            base_counts=data["base_counts"],
            query_counts=data["query_counts"],
            base_labels=data["base_labels"],
            query_labels=data["query_labels"],
            base_paths=data["base_paths"],
            query_paths=data["query_paths"],
            idf=data["idf"],
        )


def write_manifest(modality: str, quick: bool, manifest: dict) -> Path:
    path = manifest_path(modality, quick)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
