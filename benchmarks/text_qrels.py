"""
benchmarks/text_qrels.py — Etiquetas de relevancia (qrels) del benchmark de
búsqueda de texto, en formato TREC.

Cada línea del archivo es

    <query_id> 0 <doc_id> <relevancia>

con la segunda columna fija en 0 (el formato TREC la reserva y no se usa). La
relevancia es un entero >= 0: hoy todas las etiquetas son binarias (1 = el
documento es relevante para la consulta), pero el formato admite relevancia
graduada sin cambios; relevant_ids(..., min_grade=) decide desde qué grado un
documento cuenta como relevante.

Un documento que no aparece para una consulta NO está etiquetado como
irrelevante por omisión de la consulta: una consulta que falta en el archivo
es un error (KeyError), no un conjunto vacío. Sin etiquetas no se pueden
calcular Precision@k ni Recall@k.

El archivo se escribe siempre igual: consultas y documentos ordenados (los
documentos numéricamente), codificación UTF-8 y saltos de línea LF también en
Windows, así que los mismos qrels producen los mismos bytes en cualquier
sistema y cualquier huella del archivo es comparable.

Este módulo solo usa la biblioteca estándar: no importa el motor ni el corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Union

ITERATION = "0"

Grades = Union[Mapping[int, int], Iterable[int]]


def _grades(query_id: str, labels: Grades) -> dict[int, int]:
    """Normaliza las etiquetas de una consulta a {doc_id: relevancia}.
    Un conjunto de ids (sin grados) equivale a relevancia 1 para cada uno."""
    if not query_id or any(ch.isspace() for ch in query_id):
        raise ValueError(f"query_id inválido {query_id!r}: no puede estar vacío ni contener espacios")
    grades = dict(labels) if isinstance(labels, Mapping) else {doc_id: 1 for doc_id in labels}
    for doc_id, grade in grades.items():
        if isinstance(doc_id, bool) or not isinstance(doc_id, int) or doc_id < 1:
            raise ValueError(f"{query_id}: doc_id inválido {doc_id!r}; debe ser un entero >= 1")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
            raise ValueError(f"{query_id} doc {doc_id}: relevancia inválida {grade!r}; debe ser un entero >= 0")
    return grades


def write_qrels(path: Path, qrels: Mapping[str, Grades]) -> Path:
    """Escribe los qrels en formato TREC, de forma determinista.

    qrels: {query_id: {doc_id: relevancia}} o {query_id: {doc_id, ...}}
    (conjunto = relevancia 1). Una consulta sin documentos no deja líneas.
    """
    lines = []
    for query_id in sorted(qrels):
        grades = _grades(query_id, qrels[query_id])
        for doc_id in sorted(grades):  # numérico: 2 < 10 < 100
            lines.append(f"{query_id} {ITERATION} {doc_id} {grades[doc_id]}\n")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    return path


def read_qrels(path: Path, max_doc_id: int | None = None) -> dict[str, dict[int, int]]:
    """Lee un archivo de qrels: {query_id: {doc_id: relevancia}}.

    max_doc_id: ignora los documentos con id mayor. Los documentos del
    benchmark son un prefijo coherente (el documento i es el mismo para
    cualquier N), así que los qrels del N más grande sirven para los menores:
    read_qrels(path, max_doc_id=N). Una consulta que se queda sin documentos
    sigue presente, con un diccionario vacío.
    """
    qrels: dict[str, dict[int, int]] = {}
    seen: set[tuple[str, int]] = set()
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{line_number}: se esperaban 4 campos "
                f"(query_id 0 doc_id relevancia), hay {len(fields)}: {line!r}"
            )
        query_id, _iteration, raw_doc_id, raw_grade = fields
        try:
            doc_id, grade = int(raw_doc_id), int(raw_grade)
        except ValueError:
            raise ValueError(f"{path}:{line_number}: doc_id y relevancia deben ser enteros: {line!r}") from None
        if doc_id < 1 or grade < 0:
            raise ValueError(f"{path}:{line_number}: doc_id debe ser >= 1 y relevancia >= 0: {line!r}")
        if (query_id, doc_id) in seen:
            raise ValueError(f"{path}:{line_number}: {query_id} etiqueta dos veces el documento {doc_id}")
        seen.add((query_id, doc_id))

        labels = qrels.setdefault(query_id, {})
        if max_doc_id is None or doc_id <= max_doc_id:
            labels[doc_id] = grade
    return qrels


def relevant_ids(qrels: Mapping[str, Mapping[int, int]], query_id: str, min_grade: int = 1) -> set[int]:
    """Ids de los documentos relevantes de la consulta: relevancia >= min_grade.

    Una consulta que no está en los qrels lanza KeyError: sin etiquetas no se
    puede decir que no tenga documentos relevantes.
    """
    if query_id not in qrels:
        raise KeyError(f"La consulta {query_id!r} no tiene etiquetas en los qrels")
    return {doc_id for doc_id, grade in qrels[query_id].items() if grade >= min_grade}