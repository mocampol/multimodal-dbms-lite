# Benchmarks

| Script | Qué mide | Resultados |
|---|---|---|
| `perf_heap_vs_sequential.py` | HeapFile vs. SequentialFile como almacenamiento base | `results/heap_vs_sequential_results.csv` |
| `benchmark_indexes.py` | B+ clusterizado vs. B+ no clusterizado vs. Hash extensible | `results/index_comparison_results.csv` |
| `benchmark_btree.py` | B+ Tree (pytest-benchmark) | — |
| `benchmark_spatial.py` + `postgres/` | R-tree propio vs. PostgreSQL/PostGIS con GiST | `results/spatial_comparison_results.csv` |

El resto de este documento describe la comparación espacial externa.

---

## Comparación espacial: R-tree propio vs. PostGIS (GiST)

### Archivos

| Archivo | Rol |
|---|---|
| `spatial_common.py` | Dataset, consultas y formato del CSV, compartidos por ambos motores |
| `benchmark_spatial.py` | Lado DBMS propio (R-tree) |
| `postgres/docker-compose.yml` | PostGIS fijado a `postgis/postgis:16-3.4`, puerto 5433, 2 CPU / 4 GB |
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
