"""
benchmarks/benchmark_vector_sql.py — Mide cuánto cuesta una búsqueda por similitud
de punta a punta por SQL, para ponerla junto a los tiempos de benchmark_vector.py,
que miden solo el índice.

Cada SELECT ... SIMILAR_TO vuelve a extraer y a cuantizar todo el corpus en
ExtractFeatures.fit (VectorSearch.open). El script separa ese tiempo del resto de
la consulta y comprueba que el índice se haya usado de verdad, porque VectorSearch
pasa a un escaneo secuencial si la búsqueda en el índice falla.

Corre con N = 1,000: el costo crece con N, así que para N mayores conviene estimarlo.

Uso:
    python benchmarks/benchmark_vector_sql.py
    python benchmarks/benchmark_vector_sql.py --n 100 --queries 1 --modality image
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_vector import TOP_K
from vector_common import DATASETS, MODALITIES, RESULTS_DIR, SRC_DIR, data_dir, load_features

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from index.multimedia import ExtractFeatures
from index.vector import HNSWIndex, IVFIndex
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute

RESULTS_JSON = RESULTS_DIR / "vector_sql_cost.json"
INDEX_KINDS = {"IVF": IVFIndex, "HNSW": HNSWIndex}
DEFAULT_N = 1_000
DEFAULT_QUERIES = 5
INSERT_BATCH = 200


class Probe:
    """Acumula el tiempo de fit y cuenta las búsquedas en cada clase de índice."""

    def __init__(self):
        self.fit_s = 0.0
        self.fits = 0
        self.searches = {name: 0 for name in INDEX_KINDS}

    def install(self):
        original_fit = ExtractFeatures.fit

        def timed_fit(extractor, corpus):
            started = time.perf_counter()
            try:
                return original_fit(extractor, corpus)
            finally:
                self.fit_s += time.perf_counter() - started
                self.fits += 1

        ExtractFeatures.fit = timed_fit
        for name, index_class in INDEX_KINDS.items():
            self._count_searches(name, index_class)

    def _count_searches(self, name, index_class):
        original_search = index_class.search

        def counted_search(index, *args, **kwargs):
            self.searches[name] += 1
            return original_search(index, *args, **kwargs)

        index_class.search = counted_search

    def reset(self):
        self.fit_s = 0.0
        self.fits = 0
        self.searches = {name: 0 for name in INDEX_KINDS}


def new_catalog(directory: str) -> Catalog:
    heap_factory = make_heap_factory(directory)
    return Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(directory),
        },
        index_buffer_factory=make_index_buffer_factory(directory),
    )


def insert_paths(catalog: Catalog, paths: list[str]) -> None:
    for start in range(0, len(paths), INSERT_BATCH):
        values = ", ".join(
            f"({start + offset}, '{path}')" for offset, path in enumerate(paths[start:start + INSERT_BATCH])
        )
        execute(f"INSERT INTO objetos VALUES {values};", catalog)


def similarity_sql(index_kind: str, query_path: str) -> str:
    return (f"SELECT id FROM objetos WHERE SIMILAR_TO(ruta, '{query_path}', k={TOP_K}) "
            f"USING {index_kind} WITH METRIC=cosine;")


def measure_modality(modality: str, n: int, num_queries: int, probe: Probe) -> list[dict]:
    features = load_features(modality)
    root = data_dir(DATASETS[modality])
    paths = [(root / str(path)).as_posix() for path in features.base_paths[:n]]
    queries = [(root / str(path)).as_posix() for path in features.query_paths[:num_queries]]

    rows = []
    with tempfile.TemporaryDirectory(prefix="vector_sql_") as directory:
        catalog = new_catalog(directory)
        execute("CREATE TABLE objetos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)
        insert_paths(catalog, paths)

        for index_kind in INDEX_KINDS:
            probe.reset()
            started = time.perf_counter()
            execute(f"CREATE INDEX idx_{index_kind.lower()} ON objetos USING {index_kind} (ruta);", catalog)
            create_s = time.perf_counter() - started
            print(f"{modality} {index_kind}: CREATE INDEX {create_s:.1f} s", flush=True)

            probe.reset()
            times = []
            for query in queries:
                started = time.perf_counter()
                result = execute(similarity_sql(index_kind, query), catalog)
                times.append(time.perf_counter() - started)
                if len(result) != TOP_K:
                    raise RuntimeError(f"Se esperaban {TOP_K} filas y llegaron {len(result)}")
                print(f"{modality} {index_kind}: consulta {times[-1]:.1f} s", flush=True)

            if probe.searches[index_kind] != num_queries:
                raise RuntimeError(f"{index_kind} se usó {probe.searches[index_kind]} veces en {num_queries} consultas")
            rows.append({
                "modality": modality,
                "n": n,
                "index_kind": index_kind,
                "num_queries": num_queries,
                "create_index_s": round(create_s, 3),
                "avg_query_s": round(sum(times) / num_queries, 3),
                "avg_fit_s": round(probe.fit_s / num_queries, 3),
                "fit_share": round(probe.fit_s / sum(times), 4),
                "fits_per_query": probe.fits / num_queries,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--modality", choices=[*MODALITIES, "all"], default="all")
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    args = parser.parse_args()

    probe = Probe()
    probe.install()
    rows = []
    for modality in (MODALITIES if args.modality == "all" else [args.modality]):
        rows.extend(measure_modality(modality, args.n, args.queries, probe))

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Resultados: {args.out}")


if __name__ == "__main__":
    main()
