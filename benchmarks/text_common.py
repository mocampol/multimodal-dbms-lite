"""
benchmarks/text_common.py — Dataset, carga de trabajo, métricas y formato de
resultados compartidos por la comparación de texto SPIMI vs. PostgreSQL GIN.

Ambos lados (benchmarks/postgres/load_text_data.py y
benchmarks/benchmark_text_comparison.py) DEBEN importar este módulo para
garantizar que:
    - indexan exactamente los mismos documentos (mismo corpus, misma
      semilla, mismo recorte a MAX_DOC_BYTES),
    - tokenizan con el mismo TextProcessor del motor
      (default_text_processor: NFKD, sin tildes, minúsculas y stopwords en
      español de PostgreSQL; docs/text_search_policy.md §1-3),
    - ejecutan exactamente las mismas consultas con el mismo k,
    - evalúan contra los mismos documentos relevantes,
    - escriben sus resultados en el mismo CSV con las mismas columnas.

Consultas "known-item": los términos de cada consulta se toman de un mismo
documento semilla, así que toda consulta tiene al menos un documento que
contiene todos sus términos.

Relevancia: sin juicios humanos, un documento es relevante para q si
contiene TODOS los términos de q. Los candidatos (lo que deben devolver
MATCH y @@) son los documentos con AL MENOS UN término (semántica OR).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SRC_DIR = BENCH_DIR.parent / "src"
for _path in (SRC_DIR, BENCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmark_text import load_corpus  # descarga el corpus si falta
from index.text.text_processor import default_text_processor  

CORPORA = ["spanish", "synthetic", "20news"]
DEFAULT_CORPUS = "spanish"
N_VALUES = [1_000, 10_000, 100_000]
QUICK_N_VALUES = [1_000]
SEED = 42

# Un registro del heap debe caber en una página de 4 KB
# (storage/heap/heap_file.py); ambos motores reciben el mismo recorte.
MAX_DOC_BYTES = 3_500

TOP_K = 10
# Grupos de longitud del enunciado (1, 3 y 5+ palabras): (mínimo, máximo) términos.
QUERY_GROUPS = {"1": (1, 1), "3": (3, 3), "5+": (5, 8)}
QUERIES_PER_GROUP = 10
# Las consultas se toman de los primeros QUERY_POOL_DOCS documentos, que son
# prefijo de cualquier N: la carga de trabajo es la misma para todos los N.
QUERY_POOL_DOCS = 1_000
WARMUP = 1
REPETITIONS = 5

ENGINES = {
    "spimi_bm25": "SPIMI + BM25",
    "spimi_tfidf": "SPIMI + TF-IDF (coseno)",
    "pg_gin": "PostgreSQL GIN + ts_rank",
}

RESULTS_DIR = BENCH_DIR / "results"
RESULTS_CSV = RESULTS_DIR / "text_comparison_results.csv"
ENVIRONMENT_JSON = RESULTS_DIR / "text_comparison_environment.json"

# Una fila por (engine, corpus, n, query_id). Columnas vacías = no aplica al motor.
FIELDNAMES = [
    "engine",                       # clave de ENGINES
    "corpus",
    "n",
    "query_id",
    "query_group",                  # "1" | "3" | "5+"
    "query_terms",                  # cantidad exacta de términos
    "query",
    "k",
    "warmup",
    "repetitions",
    "avg_time_ms",                  # tiempo cliente: texto SQL -> todas las filas
    "median_time_ms",
    "min_time_ms",
    "max_time_ms",
    "avg_server_time_ms",           # solo PostgreSQL: "Execution Time" de EXPLAIN ANALYZE
    "returned",                     # filas devueltas con LIMIT k
    "matched_docs",                 # candidatos devueltos por el motor (sin LIMIT)
    "expected_matches",             # candidatos según text_common (deben coincidir)
    "relevant_docs",                # |R(q)|
    "relevant_digest",              # huella de R(q): debe coincidir entre motores
    "hits_at_k",
    "precision_at_k",
    "recall_at_k",
    "top_ids",                      # ids del top-k separados por espacio
    "used_index",                   # True si el plan usó el índice invertido / GIN
    "vocabulary_size",              # términos distintos en el índice
    "preprocess_time_s",            # solo PostgreSQL: tokenización compartida en Python
    "tsvector_time_s",              # solo PostgreSQL: to_tsvector al cargar
    "index_build_time_s",
    "index_space_bytes",
    "build_peak_rss_delta_bytes",   # pico de RSS al construir - RSS inicial
    "query_peak_rss_delta_bytes",   # pico de RSS al consultar - RSS inicial
]


# =============================================================================
# Dataset
# =============================================================================

def truncate_utf8(text: str, max_bytes: int = MAX_DOC_BYTES) -> str:
    """Recorta text a max_bytes en UTF-8 sin partir caracteres ni palabras."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    cut = data[:max_bytes].decode("utf-8", errors="ignore")
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


def load_documents(corpus: str, n: int) -> list[tuple[int, str]]:
    """n documentos (id, texto) con id = 1..n. Los primeros documentos son
    los mismos para cualquier n; si el corpus tiene menos filas que n, se
    repiten cíclicamente (como en benchmark_text.py)."""
    return [
        (doc_id, truncate_utf8(text.replace("\x00", " ")))  # PostgreSQL rechaza NUL en text
        for doc_id, text in load_corpus(source=corpus, max_docs=n, seed=SEED)
    ]


def tokenize(text: str) -> list[str]:
    """Tokens con las mismas reglas que usan el catálogo y MATCH."""
    return default_text_processor().process(text)


# =============================================================================
# Carga de trabajo
# =============================================================================

def generate_queries(corpus: str, seed: int = SEED + 1) -> list[dict]:
    """QUERIES_PER_GROUP consultas por cada grupo de QUERY_GROUPS.

    Cada consulta: {"query_id", "query_group", "query_terms", "terms",
    "text", "seed_doc_id"}. "terms" ya está normalizado, así que volver a
    tokenizar "text" (como hace MATCH en el motor) da los mismos términos.
    """
    rng = random.Random(seed)
    pool = [
        (doc_id, sorted(set(tokenize(text))))
        for doc_id, text in load_documents(corpus, QUERY_POOL_DOCS)
    ]

    queries = []
    for group, (min_terms, max_terms) in QUERY_GROUPS.items():
        for i in range(QUERIES_PER_GROUP):
            length = rng.randint(min_terms, max_terms)
            candidates = [(doc_id, terms) for doc_id, terms in pool if len(terms) >= length]
            if not candidates:
                raise ValueError(f"Ningún documento de '{corpus}' tiene {length} términos distintos")
            doc_id, terms = rng.choice(candidates)
            chosen = rng.sample(terms, length)
            queries.append({
                "query_id": f"G{group.rstrip('+')}-{i + 1:02d}",
                "query_group": group,
                "query_terms": length,
                "terms": chosen,
                "text": " ".join(chosen),
                "seed_doc_id": doc_id,
            })
    return queries


def query_vocabulary(queries: list[dict]) -> set[str]:
    return {term for query in queries for term in query["terms"]}


# =============================================================================
# Relevancia y métricas
# =============================================================================

def term_postings(tokenized_docs, terms: set[str]) -> dict[str, set[int]]:
    """term -> ids de documentos que lo contienen, solo para `terms`
    (los de las consultas): evita materializar el índice completo."""
    postings: dict[str, set[int]] = {term: set() for term in terms}
    for doc_id, tokens in tokenized_docs:
        for term in terms.intersection(tokens):
            postings[term].add(doc_id)
    return postings


def ids_digest(ids) -> str:
    return hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()[:16]


def query_reference(postings: dict[str, set[int]], query: dict) -> dict:
    """Columnas que dependen solo del dataset y la consulta, idénticas para
    todos los motores: k, candidatos esperados y documentos relevantes."""
    sets = [postings.get(term, set()) for term in query["terms"]]
    relevant = set.intersection(*sets) if sets else set()
    return {
        "k": TOP_K,
        "expected_matches": len(set().union(*sets)),
        "relevant_docs": len(relevant),
        "relevant_digest": ids_digest(relevant),
        "_relevant": relevant,
    }


def relevance_metrics(top_ids: list[int], relevant: set[int], k: int = TOP_K) -> dict:
    """Precision@k = aciertos / k; Recall@k = aciertos / |R(q)|."""
    hits = len(set(top_ids[:k]) & relevant)
    return {
        "hits_at_k": hits,
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant) if relevant else None,
        "top_ids": " ".join(map(str, top_ids[:k])),
    }


def time_stats(times_ms: list[float]) -> dict:
    return {
        "avg_time_ms": statistics.fmean(times_ms),
        "median_time_ms": statistics.median(times_ms),
        "min_time_ms": min(times_ms),
        "max_time_ms": max(times_ms),
    }


def base_row(engine: str, corpus: str, n: int, query: dict, reference: dict) -> dict:
    return {
        "engine": engine,
        "corpus": corpus,
        "n": n,
        "query_id": query["query_id"],
        "query_group": query["query_group"],
        "query_terms": query["query_terms"],
        "query": query["text"],
        "warmup": WARMUP,
        "repetitions": REPETITIONS,
        **{key: value for key, value in reference.items() if not key.startswith("_")},
    }


# =============================================================================
# Resultados y entorno
# =============================================================================

def quick_path(path: Path) -> Path:
    """Salida separada para --quick (benchmarks/results/*_quick.* está en .gitignore)."""
    return path.with_name(f"{path.stem}_quick{path.suffix}")


def write_results(rows: list[dict], path: Path = RESULTS_CSV) -> Path:
    """Reemplaza las filas de los mismos (engine, corpus, n) en el CSV común,
    conservando las de los demás motores, corpus y tamaños."""
    path.parent.mkdir(parents=True, exist_ok=True)
    replaced = {(row["engine"], row["corpus"], str(row["n"])) for row in rows}
    kept = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            kept = [row for row in csv.DictReader(f)
                    if (row["engine"], row["corpus"], row["n"]) not in replaced]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    return path


def host_environment() -> dict:
    try:
        import psutil
        ram = psutil.virtual_memory().total
    except ImportError:
        ram = None
    return {
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
        "host_cpu_count": os.cpu_count(),
        "host_ram_bytes": ram,
        "python_version": platform.python_version(),
    }


def write_environment(section: str, info: dict, path: Path = ENVIRONMENT_JSON) -> Path:
    """Guarda el entorno de un lado ("postgres" | "minidbms") sin borrar el otro."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = info
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path