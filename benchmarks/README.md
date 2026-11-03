# Benchmarks

| Script | Qué mide | Resultados |
|---|---|---|
| `perf_heap_vs_sequential.py` | HeapFile vs. SequentialFile como almacenamiento base | `results/heap_vs_sequential_results.csv` |
| `benchmark_indexes.py` | B+ clusterizado vs. B+ no clusterizado vs. Hash extensible | `results/index_comparison_results.csv` |
| `benchmark_btree.py` | B+ Tree (pytest-benchmark) | — |
| `benchmark_spatial.py` + `postgres/` | R-tree propio vs. PostgreSQL/PostGIS con GiST | `results/spatial_comparison_results.csv` |
| `benchmark_text_comparison.py` + `postgres/load_text_data.py` | SPIMI + BM25 y SPIMI + TF-IDF vs. PostgreSQL tsvector + GIN (`ts_rank`) | `results/text_comparison_results.csv` |


El resto de este documento describe las comparaciones externas con PostgreSQL: espacial (PostGIS) y de texto (GIN).


---

## Comparación espacial: R-tree propio vs. PostGIS (GiST)

### Archivos

| Archivo | Rol |
|---|---|
| `spatial_common.py` | Dataset, consultas y formato del CSV, compartidos por ambos motores |
| `benchmark_spatial.py` | Lado DBMS propio (R-tree) |
| `postgres/docker-compose.yml` | PostGIS fijado a `postgis/postgis:16-3.4`, puerto 5434, 2 CPU / 4 GB |
| `postgres/schema.sql` | Tabla `places (id, name, geom geometry(Point, 0))` |
| `postgres/queries.sql` | Consultas equivalentes (`ST_DWithin`, `<->`, `ST_Intersects`) |
| `postgres/load_data.py` | Carga, índice GiST, consultas, tiempos, tamaño del índice y entorno |
| `postgres/requirements.txt` | Dependencias del lado PostGIS (`psycopg`) |

### Requisitos

- Python 3.12+ (el `.venv` del proyecto).
- Docker Desktop (o Docker Engine con Compose v2) en ejecución.
- `pip install -r benchmarks/postgres/requirements.txt`

### Cómo reproducir

Desde la raíz del repositorio:

```bash
# 1. Levantar PostGIS desde cero (-v borra el volumen de datos anterior)
cd benchmarks/postgres
docker compose down -v
docker compose up -d
docker compose ps            # esperar a que el estado sea "healthy"
cd ../..

# 2. Lado PostGIS
python benchmarks/postgres/load_data.py

# 3. Lado DBMS propio
python benchmarks/benchmark_spatial.py

# 4. Apagar PostGIS al terminar
cd benchmarks/postgres && docker compose down -v
```

Ambos scripts aceptan `--quick` (solo N = 1,000) como prueba rápida.
`load_data.py` acepta `--dsn` o la variable `PG_DSN` para usar otro servidor.
El orden de los pasos 2 y 3 no importa: cada script reemplaza solo sus
propias filas en el CSV común.

### Dataset y consultas

Definidos en `spatial_common.py` y usados por ambos motores:

- **Puntos:** N = 1,000 / 10,000 / 100,000, uniformes dentro de
  `(-77.20, -12.30) – (-76.80, -11.80)` (lon/lat, ≈ Lima), semilla 42.
- **Consultas:** 20 por tipo, semilla 43, idénticas para todos los N.
- **Distancia:** euclidiana sobre las coordenadas (grados). En PostGIS se usa
  `geometry` con SRID 0; en el DBMS propio, `DISTANCIA(..., EUCLIDEAN)`
  (la métrica por defecto de `DISTANCIA` es HAVERSINE).

| Tipo | Parámetro | DBMS propio (R-tree) | PostGIS (GiST) |
|---|---|---|---|
| `radius` | r = 0.005° | `WHERE DISTANCIA(geom, POINT(x, y), EUCLIDEAN) <= r` | `WHERE ST_DWithin(geom, ST_MakePoint(x, y), r)` |
| `knn` | k = 10 | `ORDER BY DISTANCIA(geom, POINT(x, y), EUCLIDEAN) LIMIT k` | `ORDER BY geom <-> ST_MakePoint(x, y) LIMIT k` |
| `intersects` | rectángulo de 0.02° | `WHERE DENTRO_DE(geom, POLYGON(...))` | `WHERE ST_Intersects(geom, ST_GeomFromText('POLYGON(...)', 0))` |

### Metodología

Para cada N y cada motor:

1. Se crea la tabla y se cargan los puntos (no cronometrado). PostGIS usa
   `COPY`; el DBMS propio, `INSERT` por lotes de 1,000 filas.
2. Se construye el índice espacial **después** de la carga y se cronometra
   (`CREATE INDEX ... USING GIST` / `CREATE INDEX ... USING RTREE`). En
   PostGIS se ejecuta además `ANALYZE` (no cronometrado).
3. Se mide el tamaño del índice: `pg_relation_size('places_geom_gist')` en
   PostGIS y el tamaño del archivo `.idx` en el DBMS propio.
4. Cada consulta se ejecuta 1 vez de calentamiento (descartada) y luego
   3 veces; se promedia por consulta y se reportan el promedio, mínimo y
   máximo sobre las 20 consultas.
5. **Tiempo** (`avg_time_ms`): medido en el cliente con `perf_counter`,
   desde el texto SQL hasta tener todas las filas. En PostGIS incluye el
   viaje por red local al contenedor; en el DBMS propio, parseo, planificación
   y ejecución en el mismo proceso. Para PostGIS también se reporta
   `avg_server_time_ms`, el `Execution Time` de `EXPLAIN (ANALYZE)`.
6. **Uso del índice** (`used_index`): PostGIS revisa que el plan de
   `EXPLAIN` contenga `places_geom_gist`; el DBMS propio, que el plan físico
   contenga `SpatialIndexScan`. Ambos scripts avisan si alguna consulta no
   usó el índice.

`total_rows_returned` debe coincidir entre motores para el mismo N y tipo de
consulta; es la comprobación de que las consultas son equivalentes.

### Formato de resultados

`results/spatial_comparison_results.csv`, una fila por (motor, N, tipo de
consulta):

| Columna | Descripción |
|---|---|
| `engine` | `minidbms` o `postgis` |
| `n` | Número de puntos |
| `query_type` | `radius`, `knn` o `intersects` |
| `param` | Parámetro de la consulta (`r=0.005`, `k=10`, `side=0.02`) |
| `num_queries`, `repetitions` | 20 consultas × 3 repeticiones |
| `avg_time_ms`, `min_time_ms`, `max_time_ms` | Tiempo cliente por consulta |
| `avg_server_time_ms` | Solo PostGIS: `Execution Time` de `EXPLAIN ANALYZE` |
| `total_rows_returned` | Filas devueltas, sumadas sobre las 20 consultas |
| `index_build_time_s` | Tiempo de construcción del índice |
| `index_space_bytes` | Tamaño del índice en disco |
| `used_index` | `True` si todas las consultas usaron el índice espacial |

### Notas

- Con N = 1,000 el planner de PostgreSQL puede preferir un Seq Scan por ser
  más barato; en ese caso `used_index` queda en `False` y se documenta.
- La construcción del R-tree propio crece de forma superlineal, por lo que la
  corrida completa con 100,000 puntos puede tardar decenas de minutos.


---

## Comparación de texto: SPIMI vs. PostgreSQL GIN

### Archivos

| Archivo | Rol |
|---|---|
| `text_common.py` | Dataset, consultas, documentos relevantes, métricas y formato del CSV, compartidos por ambos motores |
| `benchmark_text_comparison.py` | Lado DBMS propio: `CREATE INDEX ... USING SPIMI` y `MATCH ... USING BM25 / TF_IDF` |
| `postgres/text_schema.sql` | Tablas `documentos (id, contenido, tsv tsvector)` y `documentos_staging` |
| `postgres/text_queries.sql` | Consultas equivalentes (`@@` + `ts_rank`), candidatos y vocabulario (`ts_stat`) |
| `postgres/load_text_data.py` | Carga, índice GIN, consultas, tiempos, tamaño, memoria y entorno |
| `compare_text_results.py` | Verifica que ambos lados usen el mismo k, los mismos relevantes y los mismos candidatos; resume por N y longitud |

Se usa el mismo contenedor que la comparación espacial (`postgres/docker-compose.yml`,
PostgreSQL 16, puerto 5434, 2 CPU / 4 GB).

### Requisitos

- Python 3.12+ (el `.venv` del proyecto).
- Docker Desktop (o Docker Engine con Compose v2) en ejecución.
- `pip install -r requirements.txt -r benchmarks/postgres/requirements.txt`

### Cómo reproducir

Desde la raíz del repositorio:

```bash
# 1. Levantar PostgreSQL desde cero
cd benchmarks/postgres
docker compose down -v
docker compose up -d
docker compose ps            # esperar a que el estado sea "healthy"
cd ../..

# 2. Lado PostgreSQL (GIN + ts_rank)
python benchmarks/postgres/load_text_data.py

# 3. Lado DBMS propio (SPIMI + BM25 y SPIMI + TF-IDF)
python benchmarks/benchmark_text_comparison.py

# 4. Verificar la comparación y ver el resumen
python benchmarks/compare_text_results.py

# 5. Apagar PostgreSQL al terminar
cd benchmarks/postgres && docker compose down -v
```

Los tres scripts aceptan `--quick` (solo N = 1,000, escribe `*_quick.*`, ignorado
por git). Los dos de medición aceptan `--corpus spanish|synthetic|20news|all` y
`--sizes 1000,10000`; `load_text_data.py` acepta además `--dsn` o `PG_DSN`. El
orden de los pasos 2 y 3 no importa: cada script reemplaza solo sus propias filas
(motor, corpus, N) en el CSV común. La primera corrida descarga el corpus a
`benchmarks/corpus/` (ignorado por git).

### Dataset y consultas

Definidos en `text_common.py` y usados por ambos motores:

- **Documentos:** N = 1,000 / 10,000 / 100,000 del corpus `spanish` (noticias de
  La Razón y Público) por defecto, semilla 42. Si el corpus tiene menos filas que
  N, se repiten cíclicamente. Cada documento se recorta a 3,500 bytes UTF-8 (sin
  partir palabras), porque un registro del heap debe caber en una página de 4 KB.
- **Tokenización:** la del motor (`default_text_processor`: NFKD, sin tildes,
  minúsculas y stopwords en español de PostgreSQL). PostgreSQL indexa
  `to_tsvector('simple', <tokens del motor>)`, porque su parser trata distinto
  guiones, URLs y decimales; así ambos índices tienen el mismo vocabulario.
- **Consultas:** 10 por grupo de longitud (1, 3 y 5+ términos; las de 5+ tienen
  entre 5 y 8), semilla 43, iguales para todos los N. Los términos de cada consulta
  se toman de un mismo documento semilla de los primeros 1,000.

| | DBMS propio (SPIMI) | PostgreSQL (GIN) |
|---|---|---|
| Índice | `CREATE INDEX ... ON documentos USING SPIMI (contenido)` | `CREATE INDEX ... ON documentos USING GIN (tsv)` |
| Candidatos (OR) | `WHERE MATCH(contenido, 'a b c')` | `WHERE tsv @@ to_tsquery('simple', 'a \| b \| c')` |
| Ranking | `USING BM25` / `USING TF_IDF` + `SCORE()` | `ts_rank(tsv, q, 1)` |
| Top-k | `ORDER BY relevancia DESC LIMIT 10` | `ORDER BY score DESC, id LIMIT 10` |

### Metodología

Para cada corpus, N y motor:

1. Se crea la tabla y se cargan los documentos (no cronometrado). PostgreSQL usa
   `COPY` y `to_tsvector` (este último se reporta como `tsvector_time_s`); el DBMS
   propio, `INSERT` por lotes de 500 filas.
2. Se construye el índice **después** de la carga y se cronometra. En SPIMI el
   tiempo incluye tokenizar; en PostgreSQL la tokenización compartida se reporta
   aparte (`preprocess_time_s`). Tamaño: el archivo `.idx` y `pg_relation_size`.
3. Cada consulta se ejecuta 1 vez de calentamiento (descartada) y luego 5 veces.
   **Tiempo** (`avg_time_ms`): medido en el cliente con `perf_counter`, desde el
   texto SQL hasta tener todas las filas. En PostgreSQL incluye la red local y se
   reporta también `avg_server_time_ms` (`EXPLAIN ANALYZE`).
4. **Uso del índice** (`used_index`): el plan del DBMS propio contiene
   `TextIndexScan`; el de PostgreSQL, el índice `documentos_tsv_gin`.
5. **Relevancia:** sin juicios humanos, un documento es relevante si contiene
   **todos** los términos de la consulta. Precision@10 = aciertos / 10;
   Recall@10 = aciertos / |relevantes|.

#### Memoria

Se mide igual en ambos motores: **pico de RSS del proceso que hace el trabajo
menos su RSS al empezar**, con cada fase (construcción, consultas) en un proceso
nuevo.

| | DBMS propio | PostgreSQL |
|---|---|---|
| Proceso | Proceso hijo nuevo por fase (`multiprocessing`, spawn) que reabre el catálogo desde disco | Conexión nueva por fase, es decir, un backend nuevo (`pg_backend_pid()`) |
| RSS inicial | `psutil` antes de la fase | `VmRSS` de `/proc/<pid>/status` (vía `docker exec`) |
| Pico | `peak_wset` (Windows) o `VmHWM` (Linux) | `VmHWM` de `/proc/<pid>/status` |

En la fase de consultas el RSS inicial se toma antes de abrir el catálogo, así que
el DBMS propio incluye cargar el índice completo en memoria; el backend de
PostgreSQL también parte sin páginas del GIN y cuenta las que lee a
`shared_buffers`. Si `docker exec` no está disponible, las columnas de memoria de
PostgreSQL quedan vacías.

#### Verificación

`compare_text_results.py` falla (código 1) si, para la misma consulta, los motores
difieren en k, en los relevantes (`relevant_docs`, `relevant_digest`) o en los
candidatos (`matched_docs` frente a `expected_matches`), o si el vocabulario de
los índices no coincide.

### Formato de resultados

`results/text_comparison_results.csv`, una fila por (motor, corpus, N, consulta);
el entorno de ambos lados (versiones, ajustes de PostgreSQL, CPU, RAM) va en
`results/text_comparison_environment.json`.

| Columna | Descripción |
|---|---|
| `engine` | `spimi_bm25`, `spimi_tfidf` o `pg_gin` |
| `corpus`, `n` | Corpus y número de documentos |
| `query_id`, `query_group`, `query_terms`, `query` | Consulta, grupo (`1`, `3`, `5+`), cantidad de términos y texto |
| `k`, `warmup`, `repetitions` | 10, 1 y 5 |
| `avg_time_ms`, `median_time_ms`, `min_time_ms`, `max_time_ms` | Tiempo cliente por consulta |
| `avg_server_time_ms` | Solo PostgreSQL: `Execution Time` de `EXPLAIN ANALYZE` |
| `returned`, `matched_docs`, `expected_matches` | Filas del top-k, candidatos del motor y candidatos esperados |
| `relevant_docs`, `relevant_digest` | Cantidad y huella de los relevantes (iguales entre motores) |
| `hits_at_k`, `precision_at_k`, `recall_at_k`, `top_ids` | Relevancia del top-k |
| `used_index` | `True` si el plan usó el índice invertido / GIN |
| `vocabulary_size` | Términos distintos del índice |
| `preprocess_time_s`, `tsvector_time_s` | Solo PostgreSQL: tokenización compartida y `to_tsvector` |
| `index_build_time_s`, `index_space_bytes` | Construcción y tamaño del índice |
| `build_peak_rss_delta_bytes`, `query_peak_rss_delta_bytes` | Memoria (ver arriba) |

### Notas

- Con N = 1,000 el planner de PostgreSQL suele preferir un Seq Scan; en ese caso
  `used_index` queda en `False` y se documenta.
- Las filas `spimi_bm25` y `spimi_tfidf` comparten el índice, así que repiten los
  valores de construcción, tamaño y memoria de construcción.
- `tsvector` guarda como máximo 256 posiciones por término; afecta solo a
  `ts_rank`, no a los candidatos.