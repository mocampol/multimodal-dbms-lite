"""
benchmarks/text_manifest.py — Huellas y manifiesto de la carga de trabajo del
benchmark de búsqueda de texto.

La carga de trabajo (consultas + qrels) se congela en dos archivos por corpus:

    benchmarks/workload/text_queries_<corpus>.json   consultas y manifiesto
    benchmarks/workload/text_qrels_<corpus>.tsv      etiquetas de relevancia (TREC)

El manifiesto registra todo lo que la determina, para que un cambio en
cualquiera de esas piezas se detecte en lugar de dar métricas distintas sin
aviso:

    - semillas y parámetros de generación,
    - el archivo del corpus (tamaño y SHA-256) o, si es sintético, su semilla,
    - el tokenizador (reglas y lista de stopwords),
    - los documentos de cada N (SHA-256 y cantidad de documentos distintos),
    - los qrels (SHA-256 del archivo) y cuántos relevantes tiene cada consulta
      por N.

Solo usa la biblioteca estándar y text_qrels. No importa el motor ni el corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from text_qrels import read_qrels

WORKLOAD_DIR = Path(__file__).resolve().parent / "workload"
FORMAT_VERSION = 1


def queries_path(corpus: str, data_dir: Path = WORKLOAD_DIR) -> Path:
    return Path(data_dir) / f"text_queries_{corpus}.json"


def qrels_path(corpus: str, data_dir: Path = WORKLOAD_DIR) -> Path:
    return Path(data_dir) / f"text_qrels_{corpus}.tsv"


# =============================================================================
# Huellas
# =============================================================================

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_digest(documents: Iterable[tuple[int, str]]) -> str:
    """SHA-256 de los documentos (id, texto) en el orden dado.

    Cada id y cada texto van terminados en un byte NUL, que no puede aparecer
    en los textos (load_documents lo reemplaza por un espacio), así que dos
    conjuntos distintos nunca producen la misma cadena de bytes.
    """
    digest = hashlib.sha256()
    for doc_id, text in documents:
        digest.update(str(doc_id).encode("ascii"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def tokenizer_fingerprint(processor) -> dict:
    """Reglas del TextProcessor que cambian qué términos tiene un documento.
    Los relevantes dependen de ellas: si cambian (por ejemplo, la lista de
    stopwords), los qrels guardados dejan de corresponder al motor."""
    stopwords = sorted(getattr(processor, "stopwords", ()) or ())
    stemmer = getattr(processor, "stemmer", None)
    return {
        "strip_accents": bool(processor.strip_accents),
        "lowercase": bool(processor.lowercase),
        "stemmer": None if stemmer is None else getattr(stemmer, "__name__", type(stemmer).__name__),
        "stopwords": len(stopwords),
        "stopwords_sha256": hashlib.sha256("\n".join(stopwords).encode("utf-8")).hexdigest(),
    }


# =============================================================================
# Manifiesto
# =============================================================================

def write_manifest(path: Path, manifest: dict) -> Path:
    """JSON legible y determinista: UTF-8, sangría de 2, LF y salto de línea final."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        f.write("\n")
    return path


def read_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Validación de los archivos congelados (no necesita el corpus)
# =============================================================================

def validate_workload_files(queries_file: Path, qrels_file: Path) -> list[str]:
    """Comprueba que el manifiesto y los qrels sean coherentes entre sí y que
    permitan calcular Precision@k y Recall@k. Devuelve la lista de problemas
    (vacía si todo está bien)."""
    problems: list[str] = []
    for path in (queries_file, qrels_file):
        if not Path(path).exists():
            problems.append(f"falta el archivo {path}")
    if problems:
        return problems

    manifest = read_manifest(queries_file)
    if manifest.get("format_version") != FORMAT_VERSION:
        problems.append(f"format_version {manifest.get('format_version')!r}, se esperaba {FORMAT_VERSION}")

    qrels_info = manifest.get("qrels", {})
    if qrels_info.get("sha256") != file_sha256(qrels_file):
        problems.append("los qrels no corresponden al manifiesto: el SHA-256 del archivo es distinto")

    parameters = manifest.get("parameters", {})
    n_values = parameters.get("n_values", [])
    groups = parameters.get("query_groups", {})
    per_group = parameters.get("queries_per_group")
    queries = manifest.get("queries", [])

    ids = [query["query_id"] for query in queries]
    if len(ids) != len(set(ids)):
        problems.append("hay query_id repetidos")

    for group, bounds in groups.items():
        members = [query for query in queries if query["query_group"] == group]
        if len(members) != per_group:
            problems.append(f"el grupo '{group}' tiene {len(members)} consultas, se esperaban {per_group}")
        for query in members:
            low, high = bounds
            if not low <= query["query_terms"] <= high:
                problems.append(f"{query['query_id']}: {query['query_terms']} términos fuera del grupo '{group}' ({low}-{high})")
    for query in queries:
        if query["query_group"] not in groups:
            problems.append(f"{query['query_id']}: grupo desconocido '{query['query_group']}'")
        if len(query["terms"]) != query["query_terms"]:
            problems.append(f"{query['query_id']}: query_terms={query['query_terms']} pero tiene {len(query['terms'])} términos")

    qrels = read_qrels(qrels_file)
    for query in queries:
        query_id = query["query_id"]
        if query_id not in qrels:
            problems.append(f"{query_id}: sin etiquetas en los qrels (no se pueden calcular Precision@k ni Recall@k)")
            continue
        for n in n_values:
            relevant = [doc_id for doc_id, grade in qrels[query_id].items() if grade >= 1 and doc_id <= n]
            if not relevant:
                problems.append(f"{query_id}: ningún documento relevante con N={n:,} (Recall@k no se puede calcular)")
            declared = query.get("relevant_docs", {}).get(str(n))
            if declared != len(relevant):
                problems.append(f"{query_id}: el manifiesto declara {declared} relevantes con N={n:,} y los qrels tienen {len(relevant)}")
    for query_id in set(qrels) - set(ids):
        problems.append(f"los qrels etiquetan {query_id}, que no está en las consultas")
    return problems