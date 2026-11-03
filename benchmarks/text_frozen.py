"""
benchmarks/text_frozen.py — Lectura de la carga de trabajo congelada.

Los scripts de medición (SPIMI+BM25, SPIMI+TF-IDF y PostgreSQL GIN) toman las
consultas y los documentos relevantes de aquí, y no los recalculan: así los tres
enfoques usan exactamente las mismas etiquetas, las que se versionan en
benchmarks/workload/ (ver text_workload.py).
"""

from __future__ import annotations

from pathlib import Path

from text_manifest import WORKLOAD_DIR, qrels_path, queries_path, read_manifest, validate_workload_files
from text_qrels import read_qrels, relevant_ids

GENERATE = "python benchmarks/text_workload.py --corpus {corpus}"


def load_frozen_workload(
    corpus: str,
    n: int,
    data_dir: Path = WORKLOAD_DIR,
    validate: bool = True,
) -> tuple[list[dict], dict[str, set[int]]]:
    """Consultas congeladas y relevantes de cada una para un corpus de N documentos.

    Devuelve (consultas, relevantes): las consultas tal como están en el manifiesto
    y un dict query_id -> {doc_id relevantes con doc_id <= n}. Con validate=True
    comprueba antes que los archivos sean coherentes entre sí (hash de los qrels,
    grupos de consultas, relevantes por N) y falla si no lo son.
    """
    queries_file, qrels_file = queries_path(corpus, data_dir), qrels_path(corpus, data_dir)
    if validate:
        problems = validate_workload_files(queries_file, qrels_file)
        if problems:
            hint = f" Genera con: {GENERATE.format(corpus=corpus)}" if any("falta" in p for p in problems) else ""
            raise ValueError(f"La carga congelada de '{corpus}' no es válida:\n  - " + "\n  - ".join(problems) + hint)

    manifest = read_manifest(queries_file)
    n_values = manifest["parameters"]["n_values"]
    if n not in n_values:
        raise ValueError(f"N={n:,} no está congelado para '{corpus}': los N disponibles son {n_values}")

    qrels = read_qrels(qrels_file, max_doc_id=n)
    queries = manifest["queries"]
    relevant = {query["query_id"]: relevant_ids(qrels, query["query_id"]) for query in queries}
    return queries, relevant