"""
benchmarks/benchmark_text.py — Consolidated Performance Benchmarks for Text Indexing & Search.

Self-contained, unified experimental benchmark module for:
1. Index Creation (SPIMI): Build time, RAM tokenization, Spill I/O, K-way Merge, throughput (docs/sec, tokens/sec).
2. Search & Retrieval (Okapi BM25): Query latency with statistical rigor (20 passes + warm-up,
   Mean ± StdDev, Median, P95) compared against Brute-Force Sequential Scan (Speedup Factor).
3. Scale: Evaluated at N = 1,000 / 10,000 / 100,000 documents across 3 corpora:
   - 'synthetic': Reproducible text generation (SEED=42).
   - 'spanish': Real news articles from La Razón & Público (benchmarks/corpus/).
   - '20news': Real newsgroup posts in English (benchmarks/corpus/).

Usage:
    uv run python benchmarks/benchmark_text.py --quick                  # Fast verification (N=500, 1000)
    uv run python benchmarks/benchmark_text.py --corpus synthetic       # Synthetic corpus up to N=100k
    uv run python benchmarks/benchmark_text.py --corpus spanish         # Spanish news up to N=100k
    uv run python benchmarks/benchmark_text.py --corpus 20news          # 20 Newsgroups up to N=100k
    uv run python benchmarks/benchmark_text.py --corpus all             # Full comparison across all 3
    uv run pytest benchmarks/benchmark_text.py                          # Pytest-benchmark integration
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

# Ensure src/ and repo root are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from index.text.spimi import SPIMIIndexer
from index.text.text_processor import TextProcessor

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "text_benchmark_results.csv"
SUMMARY_MD = RESULTS_DIR / "text_summary.md"


# =============================================================================
# 1. Corpus Loader & Generator (Self-Contained)
# =============================================================================

SYNTHETIC_VOCABULARY = [
    "base", "datos", "relacional", "multimodal", "sistema", "archivo", "bloque",
    "disco", "memoria", "indice", "invertido", "arbol", "btree", "hash", "spimi",
    "algoritmo", "estructura", "optimizacion", "consulta", "proceso", "computacion",
    "distribuido", "redes", "servidor", "registro", "tabla", "columna", "grafo",
    "transaccion", "almacenamiento", "buffer", "cache", "ordenamiento", "busqueda",
    "relevancia", "ranking", "frecuencia", "documento", "termino", "lexema", "normalizacion",
    "tecnologia", "ciencia", "investigacion", "universidad", "estudiante", "proyecto",
    "desarrollo", "ingenieria", "programa", "codigo", "lenguaje", "python", "postgres",
    "innovacion", "digital", "comunicacion", "informacion", "seguridad", "rendimiento",
    "experimento", "analisis", "resultado", "velocidad", "medicion", "prueba", "diseno",
    "arquitectura", "plataforma", "metodo", "modelo", "aprendizaje", "inteligencia",
    "noticia", "articulo", "publicacion", "medio", "prensa", "editorial", "empresa",
    "mercado", "economia", "politica", "sociedad", "cultura", "historia", "futuro",
]


def generate_synthetic_document(rng: random.Random, min_words: int = 15, max_words: int = 45) -> str:
    """Generates a pseudo-realistic synthetic document with Zipf-like word repetition."""
    num_words = rng.randint(min_words, max_words)
    weights = [1.0 / (i + 1) ** 0.8 for i in range(len(SYNTHETIC_VOCABULARY))]
    chosen = rng.choices(SYNTHETIC_VOCABULARY, weights=weights, k=num_words)
    return " ".join(chosen)


def load_corpus(
    source: str = "synthetic",
    max_docs: int = 10000,
    seed: int = 42,
) -> list[tuple[int, str]]:
    """
    Loads up to max_docs from the specified corpus source.
    If max_docs exceeds available unique rows, cycles through them with unique doc_ids.
    """
    source = source.lower().strip()

    if source in ("synthetic", "sintetico"):
        rng = random.Random(seed)
        return [(i + 1, generate_synthetic_document(rng)) for i in range(max_docs)]

    elif source in ("spanish", "espanol", "noticias"):
        csv_path = CORPUS_DIR / "data_larazon_publico_v2.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Spanish news dataset not found at {csv_path}.")

        raw_texts: list[str] = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)  # header: ['', 'indi', 'cuerpo', 'titular']
            for row in reader:
                if len(row) >= 4:
                    cuerpo = row[2].strip()
                    titular = row[3].strip()
                    text = f"{titular}. {cuerpo}" if titular else cuerpo
                    if text:
                        raw_texts.append(text)
                        if len(raw_texts) >= max_docs:
                            break

        # Expand cyclically if requested max_docs exceeds available rows
        docs = []
        for i in range(max_docs):
            docs.append((i + 1, raw_texts[i % len(raw_texts)]))
        return docs

    elif source in ("20news", "english", "newsgroups"):
        csv_path = CORPUS_DIR / "train_20newsgroups.csv"
        if not csv_path.exists():
            csv_path = CORPUS_DIR / "20_newsgroups.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"20 Newsgroups dataset not found in {CORPUS_DIR}.")

        raw_texts = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    text = row[0].strip() if "train" in csv_path.name else (row[1].strip() if len(row) > 1 else row[0])
                    if text:
                        raw_texts.append(text)
                        if len(raw_texts) >= max_docs:
                            break

        docs = []
        for i in range(max_docs):
            docs.append((i + 1, raw_texts[i % len(raw_texts)]))
        return docs

    else:
        raise ValueError(f"Unsupported corpus source '{source}'. Options: 'synthetic', 'spanish', '20news'.")


# =============================================================================
# 2. Representative Query Workloads
# =============================================================================

QUERY_WORKLOADS: dict[str, list[dict]] = {
    "spanish": [
        {"type": "1-term (high freq)", "query": ["gobierno"]},
        {"type": "1-term (high freq)", "query": ["espana"]},
        {"type": "1-term (low freq)", "query": ["submarino"]},
        {"type": "1-term (med freq)", "query": ["tribunal"]},
        {"type": "2-terms", "query": ["economia", "mercado"]},
        {"type": "2-terms", "query": ["partido", "politica"]},
        {"type": "3-terms", "query": ["crisis", "laboral", "empleo"]},
        {"type": "3-terms", "query": ["presidente", "ministro", "parlamento"]},
        {"type": "4-terms", "query": ["justicia", "juez", "audiencia", "nacional"]},
        {"type": "out of vocab", "query": ["terminoinexistente999"]},
    ],
    "20news": [
        {"type": "1-term (high freq)", "query": ["system"]},
        {"type": "1-term (high freq)", "query": ["people"]},
        {"type": "1-term (low freq)", "query": ["encryption"]},
        {"type": "1-term (med freq)", "query": ["orbit"]},
        {"type": "2-terms", "query": ["space", "mission"]},
        {"type": "2-terms", "query": ["computer", "graphics"]},
        {"type": "3-terms", "query": ["hardware", "memory", "driver"]},
        {"type": "3-terms", "query": ["government", "national", "security"]},
        {"type": "4-terms", "query": ["software", "program", "code", "development"]},
        {"type": "out of vocab", "query": ["unknownwordxyz99"]},
    ],
    "synthetic": [
        {"type": "1-term (high freq)", "query": ["datos"]},
        {"type": "1-term (high freq)", "query": ["base"]},
        {"type": "1-term (low freq)", "query": ["buffer"]},
        {"type": "1-term (med freq)", "query": ["arbol"]},
        {"type": "2-terms", "query": ["base", "datos"]},
        {"type": "2-terms", "query": ["algoritmo", "estructura"]},
        {"type": "3-terms", "query": ["sistema", "distribuido", "redes"]},
        {"type": "3-terms", "query": ["memoria", "disco", "almacenamiento"]},
        {"type": "4-terms", "query": ["indice", "invertido", "spimi", "ranking"]},
        {"type": "out of vocab", "query": ["terminoinexistente999"]},
    ],
}


# =============================================================================
# 3. Statistical Helpers & Sequential Search Baseline
# =============================================================================

def compute_statistics(values: list[float]) -> dict[str, float]:
    """Calculates Mean, StdDev, Median, and P95 from sample timings."""
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "median": 0.0, "p95": 0.0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    stddev = math.sqrt(variance)

    sorted_vals = sorted(values)
    median = sorted_vals[n // 2]
    p95_idx = min(n - 1, int(0.95 * n))
    p95 = sorted_vals[p95_idx]

    return {
        "mean": round(mean, 4),
        "stddev": round(stddev, 4),
        "median": round(median, 4),
        "p95": round(p95, 4),
    }


def sequential_text_search(
    corpus: list[tuple[int, str]],
    query_tokens: list[str],
    processor: TextProcessor,
    top_k: int = 10,
) -> list[tuple[int, int]]:
    """Brute-force linear scan over raw documents (Full Table Scan baseline)."""
    matches: list[tuple[int, int]] = []
    query_set = set(query_tokens)

    for doc_id, text in corpus:
        tokens = processor.process(text)
        score = sum(1 for t in tokens if t in query_set)
        if score > 0:
            matches.append((doc_id, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


# =============================================================================
# 4. Integrated Benchmark Execution Engine
# =============================================================================

def run_experiments_for_corpus(
    corpus_name: str,
    n_values: list[int],
    query_repetitions: int = 20,
    block_size: int = 5000,
) -> tuple[list[dict], list[dict]]:
    """
    Executes both:
    1. Index Build Time & SPIMI Breakdown for each N in n_values.
    2. Query Search Latency Evaluation (20 passes + warm-up) for each N.
    """
    print(f"\n{'='*78}")
    print(f" BENCHMARK DE TEXTO: {corpus_name.upper()} (N = {n_values})")
    print(f"{'='*78}")

    stops = "spanish" if corpus_name == "spanish" else ("english" if corpus_name == "20news" else None)
    processor = TextProcessor(stopwords=stops)
    workload = QUERY_WORKLOADS.get(corpus_name, QUERY_WORKLOADS["synthetic"])

    build_results: list[dict] = []
    search_results: list[dict] = []

    for n in n_values:
        print(f"\n[1/2] Cargando e Indexando N = {n:,} documentos ({corpus_name})...")
        corpus = load_corpus(source=corpus_name, max_docs=n)

        # -------------------------------------------------------------
        # Fase 1: Medición de Creación de Índice (SPIMI)
        # -------------------------------------------------------------
        indexer = SPIMIIndexer(processor=processor, max_records_per_block=block_size)
        t_start = time.perf_counter()
        index = indexer.index_corpus(corpus)
        build_elapsed = time.perf_counter() - t_start
        stats = indexer.stats

        docs_per_sec = n / build_elapsed if build_elapsed > 0 else 0
        tokens_per_sec = stats.total_tokens_processed / build_elapsed if build_elapsed > 0 else 0
        vocab_size = len(index.get_terms())

        build_row = {
            "corpus": corpus_name,
            "N": n,
            "build_time_sec": round(build_elapsed, 4),
            "ram_time_sec": round(stats.ram_time_sec, 4),
            "spill_time_sec": round(stats.spill_time_sec, 4),
            "merge_time_sec": round(stats.merge_time_sec, 4),
            "blocks_count": stats.blocks_created,
            "vocab_size": vocab_size,
            "tokens_total": stats.total_tokens_processed,
            "docs_per_sec": round(docs_per_sec, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
        }
        build_results.append(build_row)

        print(f"  > Indexacion completada en {build_elapsed:.4f}s ({docs_per_sec:,.0f} docs/s)")
        print(f"    RAM: {stats.ram_time_sec:.4f}s | Spill: {stats.spill_time_sec:.4f}s | Merge: {stats.merge_time_sec:.4f}s")
        print(f"    Bloques volcados: {stats.blocks_created} | Vocabulario unico: {vocab_size:,} terminos")

        # -------------------------------------------------------------
        # Fase 2: Medición de Consultas (BM25 & TF-IDF Cosine vs Secuencial)
        # -------------------------------------------------------------
        print(f"\n[2/2] Evaluando Banco de Consultas (20 pasadas + warm-up) para N = {n:,}...")
        print(f"{'Tipo':<16} | {'Consulta':<24} | {'BM25 (ms)':<13} | {'Cosine (ms)':<13} | {'Seq (ms)':<10} | {'Speedup':<9}")
        print("-" * 100)

        for q_item in workload:
            q_type = q_item["type"]
            q_tokens = q_item["query"]
            q_str = " ".join(q_tokens)

            # Warm-up (5 pasadas no cronometradas)
            for _ in range(5):
                index.score_bm25(q_tokens, top_k=10)
                index.score_tfidf_cosine(q_tokens, top_k=10)

            # 20 pasadas cronometradas BM25
            bm25_timings: list[float] = []
            for _ in range(query_repetitions):
                t0 = time.perf_counter()
                index.score_bm25(q_tokens, k1=1.2, b=0.75, top_k=10)
                bm25_timings.append((time.perf_counter() - t0) * 1000.0)

            bm25_stats = compute_statistics(bm25_timings)

            # 20 pasadas cronometradas TF-IDF Cosine
            cosine_timings: list[float] = []
            for _ in range(query_repetitions):
                t_cos = time.perf_counter()
                index.score_tfidf_cosine(q_tokens, top_k=10)
                cosine_timings.append((time.perf_counter() - t_cos) * 1000.0)

            cosine_stats = compute_statistics(cosine_timings)

            # Escaneo Secuencial (Full Table Scan)
            # Para N=100k limitamos repeticiones secuenciales para evitar tiempo excesivo
            seq_reps = 3 if n >= 100000 else (5 if n >= 10000 else 10)
            seq_timings: list[float] = []
            for _ in range(seq_reps):
                t1 = time.perf_counter()
                sequential_text_search(corpus, q_tokens, processor, top_k=10)
                seq_timings.append((time.perf_counter() - t1) * 1000.0)

            seq_stats = compute_statistics(seq_timings)
            speedup_bm25 = (seq_stats["mean"] / bm25_stats["mean"]) if bm25_stats["mean"] > 0 else 1.0
            speedup_cosine = (seq_stats["mean"] / cosine_stats["mean"]) if cosine_stats["mean"] > 0 else 1.0

            search_row = {
                "corpus": corpus_name,
                "N": n,
                "query_type": q_type,
                "query": q_str,
                "bm25_mean_ms": bm25_stats["mean"],
                "bm25_stddev_ms": bm25_stats["stddev"],
                "bm25_median_ms": bm25_stats["median"],
                "bm25_p95_ms": bm25_stats["p95"],
                "cosine_mean_ms": cosine_stats["mean"],
                "cosine_stddev_ms": cosine_stats["stddev"],
                "cosine_median_ms": cosine_stats["median"],
                "cosine_p95_ms": cosine_stats["p95"],
                "seq_mean_ms": seq_stats["mean"],
                "speedup_bm25": round(speedup_bm25, 2),
                "speedup_cosine": round(speedup_cosine, 2),
            }
            search_results.append(search_row)

            print(f"{q_type:<16} | {q_str:<24} | {bm25_stats['mean']:>5.3f} ± {bm25_stats['stddev']:<5.3f} | "
                  f"{cosine_stats['mean']:>5.3f} ± {cosine_stats['stddev']:<5.3f} | {seq_stats['mean']:>7.2f} ms | "
                  f"{speedup_cosine:>7.1f}x")

    return build_results, search_results


# =============================================================================
# 5. Exportación de Resultados (CSV & Markdown)
# =============================================================================

def save_benchmark_outputs(build_rows: list[dict], search_rows: list[dict]):
    """Saves benchmark results to CSV and generates a complete Markdown summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Export CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        # Save search results CSV
        if search_rows:
            writer = csv.DictWriter(f, fieldnames=list(search_rows[0].keys()))
            writer.writeheader()
            writer.writerows(search_rows)

    # 2. Export Markdown Report
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Reporte Experimental: Indexación SPIMI, BM25 y Similitud del Coseno (TF-IDF)\n\n")
        f.write("Este informe resume los resultados experimentales de **tiempo de construcción de índice** ")
        f.write("y **latencia de consultas con 20 repeticiones estadísticas** para $N \\in [1,000, 10,000, 100,000]$.\n\n")

        # Table 1: Index Build Time
        f.write("## 1. Tiempo de Creación del Índice (Index Build Time & SPIMI Breakdown)\n\n")
        f.write("| Corpus | N (Docs) | Tiempo Total (s) | RAM (s) | Spill I/O (s) | Merge Heap (s) | Bloques | Vocabulario | Rendimiento (docs/s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for b in build_rows:
            f.write(f"| {b['corpus']} | {b['N']:,} | **{b['build_time_sec']:.4f}s** | {b['ram_time_sec']:.4f}s | "
                    f"{b['spill_time_sec']:.4f}s | {b['merge_time_sec']:.4f}s | {b['blocks_count']} | "
                    f"{b['vocab_size']:,} | {b['docs_per_sec']:,.0f} docs/s |\n")

        # Table 2: Search Latency & Speedup
        f.write("\n## 2. Latencia de Consulta: BM25 y Similitud Coseno vs Escaneo Secuencial (20 Pasadas)\n\n")
        f.write("| Corpus | N | Tipo | Consulta | BM25 Media (ms) | Coseno Media (ms) | Secuencial (ms) | Speedup Coseno |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for s in search_rows:
            f.write(f"| {s['corpus']} | {s['N']:,} | {s['query_type']} | `{s['query']}` | "
                    f"{s['bm25_mean_ms']:.3f} ms | {s['cosine_mean_ms']:.3f} ms | "
                    f"{s['seq_mean_ms']:.2f} ms | **{s['speedup_cosine']:.1f}x** |\n")

    print(f"\n[OK] Reportes guardados exitosamente:")
    print(f"  - CSV: {RESULTS_CSV}")
    print(f"  - Markdown: {SUMMARY_MD}")


# =============================================================================
# 6. Pytest-Benchmark Integration
# =============================================================================

def test_benchmark_spimi_synthetic(benchmark):
    """Benchmarks SPIMI indexing on 1,000 synthetic documents."""
    corpus = load_corpus(source="synthetic", max_docs=1000)
    processor = TextProcessor()

    def run():
        indexer = SPIMIIndexer(processor=processor, max_records_per_block=500)
        indexer.index_corpus(corpus)

    benchmark.pedantic(run, rounds=3, iterations=1)


def test_benchmark_bm25_query_synthetic(benchmark):
    """Benchmarks Okapi BM25 scoring over an indexed corpus."""
    corpus = load_corpus(source="synthetic", max_docs=1000)
    processor = TextProcessor()
    indexer = SPIMIIndexer(processor=processor, max_records_per_block=500)
    index = indexer.index_corpus(corpus)

    def run():
        index.score_bm25(["algoritmo", "estructura", "datos"], top_k=10)

    benchmark.pedantic(run, rounds=10, iterations=5)


def test_benchmark_cosine_query_synthetic(benchmark):
    """Benchmarks TF-IDF Cosine Similarity scoring over an indexed corpus."""
    corpus = load_corpus(source="synthetic", max_docs=1000)
    processor = TextProcessor()
    indexer = SPIMIIndexer(processor=processor, max_records_per_block=500)
    index = indexer.index_corpus(corpus)

    def run():
        index.score_tfidf_cosine(["algoritmo", "estructura", "datos"], top_k=10)

    benchmark.pedantic(run, rounds=10, iterations=5)


# =============================================================================
# 7. CLI Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SPIMI & BM25 Performance Benchmarks (Build Time & Query Latency)")
    parser.add_argument(
        "--corpus",
        choices=["synthetic", "spanish", "20news", "all"],
        default="synthetic",
        help="Corpus source to evaluate (default: synthetic)",
    )
    parser.add_argument(
        "-n",
        "--sizes",
        type=int,
        nargs="+",
        help="Explicit document counts N to benchmark (e.g., -n 10000 or -n 1000 10000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quickly with smaller sample sizes (N = 500, 1000)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="Number of query repetitions for statistical measurements (default: 20)",
    )
    args = parser.parse_args()

    corpora_to_run = (
        ["synthetic", "spanish", "20news"]
        if args.corpus == "all"
        else [args.corpus]
    )

    if args.sizes:
        n_values = args.sizes
    elif args.quick:
        n_values = [500, 1000]
    else:
        n_values = [1000, 10000, 100000]

    all_build_rows: list[dict] = []
    all_search_rows: list[dict] = []

    for corpus_name in corpora_to_run:
        build_rows, search_rows = run_experiments_for_corpus(
            corpus_name=corpus_name,
            n_values=n_values,
            query_repetitions=args.repetitions,
        )
        all_build_rows.extend(build_rows)
        all_search_rows.extend(search_rows)

    save_benchmark_outputs(all_build_rows, all_search_rows)


if __name__ == "__main__":
    main()
