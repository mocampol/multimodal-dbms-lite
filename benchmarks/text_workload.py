"""
benchmarks/text_workload.py — Congela la carga de trabajo de la comparación de
texto (SPIMI + BM25, SPIMI + TF-IDF y PostgreSQL GIN): las consultas y sus
etiquetas de relevancia (qrels).

Para un corpus escribe, en benchmarks/workload/:

    text_queries_<corpus>.json   las consultas y el manifiesto (ver text_manifest.py)
    text_qrels_<corpus>.tsv      los qrels en formato TREC (ver text_qrels.py)

Relevancia (binaria): un documento es relevante para una consulta si contiene
TODOS sus términos, ya tokenizados con el TextProcessor del motor. Cada consulta
se genera a partir de un documento semilla de los primeros QUERY_POOL_DOCS, así
que ese documento es relevante en todos los N y ninguna consulta se queda sin
etiquetas.

Los qrels se calculan una sola vez, sobre el N más grande. Los documentos de
cualquier N son un prefijo de los del N más grande (el documento i es el mismo
para todo N; el script lo comprueba), así que los qrels de un N menor son los
mismos filtrados por doc_id <= N: read_qrels(path, max_doc_id=N).

Reproducibilidad: la generación usa las semillas fijas de text_common. Para los
corpus reales (spanish, 20news) la semilla no interviene, porque load_corpus
toma las primeras N filas del CSV; lo que fija los datos es el archivo, y por eso
el manifiesto guarda su SHA-256. --check regenera todo y falla si cualquier
pieza (corpus, tokenizador, semillas, documentos, consultas o qrels) difiere de
lo guardado.

Uso:
    python benchmarks/text_workload.py                      # spanish, N = 1k/10k/100k
    python benchmarks/text_workload.py --corpus synthetic   # sin descargas
    python benchmarks/text_workload.py --check              # verifica los archivos guardados
"""

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_text
from text_common import (
    CORPORA,
    DEFAULT_CORPUS,
    MAX_DOC_BYTES,
    N_VALUES,
    QUERIES_PER_GROUP,
    QUERY_GROUPS,
    QUERY_POOL_DOCS,
    SEED,
    TOP_K,
    generate_queries,
    load_documents,
    query_vocabulary,
    term_postings,
    tokenize,
)
from index.text.text_processor import default_text_processor
from text_manifest import (
    WORKLOAD_DIR,
    FORMAT_VERSION,
    dataset_digest,
    file_sha256,
    qrels_path,
    queries_path,
    read_manifest,
    tokenizer_fingerprint,
    validate_workload_files,
    write_manifest,
)
from text_qrels import write_qrels

# Semilla de las consultas: la que usa generate_queries por defecto.
QUERY_SEED = inspect.signature(generate_queries).parameters["seed"].default

# Archivos de los corpus reales dentro de benchmark_text.CORPUS_DIR.
CORPUS_FILES = {
    "spanish": ["data_larazon_publico_v2.csv"],
    "20news": ["train_20newsgroups.csv", "20_newsgroups.csv"],
}


# =============================================================================
# Construcción
# =============================================================================

def corpus_source(corpus: str) -> dict:
    """Qué fija los datos: la semilla (sintético) o el archivo del corpus (real).
    Solo el nombre del archivo, nunca rutas de la máquina, para que el manifiesto
    sea idéntico en cualquier equipo."""
    names = CORPUS_FILES.get(corpus)
    if names is None:
        return {"generator": corpus, "seed": SEED}
    for name in names:
        path = benchmark_text.CORPUS_DIR / name
        if path.exists():
            return {"file": name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    raise FileNotFoundError(f"No se encuentra el corpus '{corpus}' en {benchmark_text.CORPUS_DIR}")


def load_prefix_documents(corpus: str, n_values: list[int]) -> dict[int, list[tuple[int, str]]]:
    """Documentos de cada N. Exige que los de un N menor sean un prefijo de los
    del N más grande: de eso depende que los qrels del N más grande sirvan para
    todos los demás."""
    documents = {n: load_documents(corpus, n) for n in n_values}
    largest = documents[n_values[-1]]
    for n in n_values[:-1]:
        if documents[n] != largest[:n]:
            raise ValueError(
                f"Los {n:,} documentos de '{corpus}' no son un prefijo de los {n_values[-1]:,}: "
                "los qrels de un solo N no sirven para los demás"
            )
    return documents


def relevant_by_all_terms(postings: dict[str, set[int]], terms: list[str]) -> set[int]:
    """Documentos que contienen todos los términos de la consulta."""
    return set.intersection(*(postings[term] for term in terms))


def build_workload(corpus: str, out_dir: Path = WORKLOAD_DIR, n_values: list[int] = N_VALUES) -> dict:
    """Genera las consultas y los qrels de `corpus` en out_dir; devuelve el manifiesto."""
    n_values = sorted(n_values)
    if n_values[0] < QUERY_POOL_DOCS:
        raise ValueError(
            f"El N más chico ({n_values[0]:,}) debe ser >= {QUERY_POOL_DOCS:,}: las consultas salen de los "
            "primeros documentos y, con un N menor, su documento semilla podría quedar fuera"
        )
    largest = n_values[-1]

    documents = load_prefix_documents(corpus, n_values)
    queries = generate_queries(corpus)

    postings = term_postings(
        ((doc_id, tokenize(text)) for doc_id, text in documents[largest]),
        query_vocabulary(queries),
    )

    qrels, entries = {}, []
    for query in queries:
        relevant = relevant_by_all_terms(postings, query["terms"])
        ordered = sorted(relevant)
        counts = {str(n): bisect_right(ordered, n) for n in n_values}
        empty = [n for n in n_values if counts[str(n)] == 0]
        if empty or query["seed_doc_id"] not in relevant:
            raise ValueError(
                f"{query['query_id']} ('{query['text']}') no tiene relevantes con N={empty or n_values}; "
                "su documento semilla debería ser relevante"
            )
        qrels[query["query_id"]] = relevant
        entries.append({
            "query_id": query["query_id"],
            "query_group": query["query_group"],
            "query_terms": query["query_terms"],
            "terms": query["terms"],
            "text": query["text"],
            "seed_doc_id": query["seed_doc_id"],
            "relevant_docs": counts,
        })

    qrels_file = write_qrels(qrels_path(corpus, out_dir), qrels)
    manifest = {
        "format_version": FORMAT_VERSION,
        "corpus": corpus,
        "seeds": {"documents": SEED, "queries": QUERY_SEED},
        "parameters": {
            "n_values": n_values,
            "max_doc_bytes": MAX_DOC_BYTES,
            "top_k": TOP_K,
            "query_pool_docs": QUERY_POOL_DOCS,
            "queries_per_group": QUERIES_PER_GROUP,
            "query_groups": {group: list(bounds) for group, bounds in QUERY_GROUPS.items()},
            "relevance": "binaria: el documento contiene todos los términos de la consulta",
        },
        "corpus_source": corpus_source(corpus),
        "tokenizer": tokenizer_fingerprint(default_text_processor()),
        "datasets": {
            str(n): {
                "documents": len(documents[n]),
                "distinct_documents": len({text for _, text in documents[n]}),
                "sha256": dataset_digest(documents[n]),
            }
            for n in n_values
        },
        "qrels": {
            "file": qrels_file.name,
            "sha256": file_sha256(qrels_file),
            "max_doc_id": largest,
            "lines": sum(len(relevant) for relevant in qrels.values()),
        },
        "queries": entries,
    }
    write_manifest(queries_path(corpus, out_dir), manifest)
    return manifest


# =============================================================================
# Verificación
# =============================================================================

def describe_differences(saved: dict, fresh: dict) -> list[str]:
    """Qué pieza cambió entre el manifiesto guardado y el regenerado."""
    problems = []
    if saved.get("seeds") != fresh["seeds"]:
        problems.append(f"semillas distintas: guardadas {saved.get('seeds')}, actuales {fresh['seeds']}")
    if saved.get("parameters") != fresh["parameters"]:
        problems.append("parámetros de generación distintos (N, recorte, grupos de consultas o k)")
    if saved.get("corpus_source") != fresh["corpus_source"]:
        problems.append(f"el corpus cambió: guardado {saved.get('corpus_source')}, actual {fresh['corpus_source']}")
    if saved.get("tokenizer") != fresh["tokenizer"]:
        problems.append("el tokenizador cambió (reglas o lista de stopwords): los términos y los relevantes ya no coinciden")
    for n, info in fresh["datasets"].items():
        if saved.get("datasets", {}).get(n) != info:
            problems.append(f"los documentos de N={int(n):,} son distintos a los guardados")
    saved_queries = {query["query_id"]: query for query in saved.get("queries", [])}
    for query in fresh["queries"]:
        old = saved_queries.get(query["query_id"])
        if old is None:
            problems.append(f"{query['query_id']}: consulta nueva, no está en lo guardado")
        elif old != query:
            problems.append(f"{query['query_id']}: la consulta o su número de relevantes cambió "
                            f"('{old.get('text')}' -> '{query['text']}')")
    for query_id in set(saved_queries) - {query["query_id"] for query in fresh["queries"]}:
        problems.append(f"{query_id}: está en lo guardado y ya no se genera")
    if saved.get("qrels") != fresh["qrels"]:
        problems.append("los qrels regenerados son distintos a los guardados")
    return problems


def check_workload(corpus: str, data_dir: Path = WORKLOAD_DIR, n_values: list[int] = N_VALUES) -> list[str]:
    """Regenera la carga de trabajo y la compara con los archivos guardados.
    Devuelve la lista de problemas (vacía si coinciden byte a byte)."""
    saved_queries, saved_qrels = queries_path(corpus, data_dir), qrels_path(corpus, data_dir)
    problems = validate_workload_files(saved_queries, saved_qrels)
    if any(problem.startswith("falta el archivo") for problem in problems):
        return problems

    with tempfile.TemporaryDirectory(prefix="text-workload-") as tmp:
        fresh = build_workload(corpus, Path(tmp), n_values)
        problems += describe_differences(read_manifest(saved_queries), fresh)
        if saved_queries.read_bytes() != queries_path(corpus, tmp).read_bytes() and not problems:
            problems.append("el JSON de consultas es distinto al regenerado")
        if saved_qrels.read_bytes() != qrels_path(corpus, tmp).read_bytes() and not problems:
            problems.append("los qrels son distintos a los regenerados")
    return problems


# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus", choices=CORPORA + ["all"], default=DEFAULT_CORPUS)
    parser.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=N_VALUES,
                        help="lista de N separada por comas (por defecto 1000,10000,100000)")
    parser.add_argument("--out-dir", type=Path, default=WORKLOAD_DIR)
    parser.add_argument("--check", action="store_true",
                        help="no escribe: regenera y compara con los archivos guardados")
    args = parser.parse_args(argv)

    failed = False
    for corpus in (CORPORA if args.corpus == "all" else [args.corpus]):
        if args.check:
            problems = check_workload(corpus, args.out_dir, args.sizes)
            if problems:
                failed = True
                print(f"✗ {corpus}: {len(problems)} problemas")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                print(f"✓ {corpus}: las consultas y los qrels guardados coinciden con lo que se regenera")
            continue

        manifest = build_workload(corpus, args.out_dir, args.sizes)
        print(f"{corpus}: {len(manifest['queries'])} consultas, {manifest['qrels']['lines']:,} etiquetas")
        for n, info in manifest["datasets"].items():
            print(f"  N={int(n):>7,}: {info['distinct_documents']:,} documentos distintos de {info['documents']:,}")
        print(f"  escrito: {queries_path(corpus, args.out_dir)}")
        print(f"  escrito: {qrels_path(corpus, args.out_dir)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())