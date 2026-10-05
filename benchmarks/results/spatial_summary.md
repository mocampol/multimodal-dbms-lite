# Resumen: comparación de técnicas de búsqueda espacial

_Generado por `benchmarks/benchmark_spatial.py` el 2026-10-04 17:35._

## 1. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Tamaños N | 1,000, 10,000, 100,000 |
| Radios | 1 km, 5 km, 10 km |
| k-NN | 10, 50, 100 |
| Consultas por configuración | 100 (mismos centros para las 3 técnicas) |
| Warm-up | 10 consultas no medidas antes de cada celda |
| Repeticiones | 3 (cada una reconstruye índices y repite todo) |
| Semilla | 42 |
| Región de datos | lon [-78.0, -77.0], lat [-12.5, -11.5] (uniforme, ~108 km x 111 km) |
| Métrica | Haversine en metros, esfera R = 6 371 008.8 m |
| Tolerancia de equivalencia | 0.05 m |
| Presupuesto por celda y repetición | 900 s |
| Python / SO | 3.13.2 / Windows-11-10.0.22621-SP0 |

Tabla base común: `HeapFile` con `(id INTEGER, lon DOUBLE, lat DOUBLE)`. El R-Tree indexa `(lon, lat) -> RID`; las tres técnicas devuelven ids.

## 2. Configuración de PostgreSQL / PostGIS

| Ajuste | Valor |
|---|---|
| postgres_version | PostgreSQL 16.4 (Debian 16.4-1.pgdg110+2) on x86_64-pc-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit |
| postgis_version | 3.4 USE_GEOS=1 USE_PROJ=1 USE_STATS=1 |
| shared_buffers | 128MB |
| work_mem | 4MB |
| effective_cache_size | 4GB |
| random_page_cost | 4 |
| max_parallel_workers_per_gather | 0 |
| jit | off |
| knn_operator_model | esfera |
| `SET max_parallel_workers_per_gather` (sesión del benchmark) | 0 |
| `SET jit` (sesión del benchmark) | off |
| `statement_timeout` | 120 s |

Esquema: `bench_spatial.bench_points(id integer PK, lon, lat, geog geography(Point,4326))`; índice `CREATE INDEX bench_points_geog_gist ... USING GIST (geog)`; `VACUUM ANALYZE` tras la carga. Rango: `ST_DWithin(geog, punto, metros, false)` (esfera). k-NN: `ORDER BY geog <-> punto LIMIT k`. El tiempo incluye ida y vuelta al servidor y `fetchall()`.

## 3. Tiempo promedio de consulta (ms)

### Búsqueda por radio

| N | Técnica | 1 km | 5 km | 10 km |
|---|---|---|---|---|
| 1,000 | Búsqueda secuencial | 4.493 | 4.465 | 4.489 |
| 1,000 | R-Tree propio | 0.716 | 1.225 | 1.975 |
| 1,000 | PostgreSQL + PostGIS (GiST) | 0.512 | 0.494 | 0.523 |
| 10,000 | Búsqueda secuencial | 45.054 | 45.150 | 45.064 |
| 10,000 | R-Tree propio | 1.365 | 3.559 | 7.645 |
| 10,000 | PostgreSQL + PostGIS (GiST) | 0.628 | 0.719 | 1.049 |
| 100,000 | Búsqueda secuencial | 455.402 | 457.061 | 453.626 |
| 100,000 | R-Tree propio | 4.247 | 25.237 | 70.802 |
| 100,000 | PostgreSQL + PostGIS (GiST) | 0.887 | 2.113 | 4.481 |

### k-NN

| N | Técnica | 10 k | 50 k | 100 k |
|---|---|---|---|---|
| 1,000 | Búsqueda secuencial | 4.603 | 4.702 | 4.754 |
| 1,000 | R-Tree propio | 4.406 | 7.965 | 12.717 |
| 1,000 | PostgreSQL + PostGIS (GiST) | 0.502 | 0.564 | 0.645 |
| 10,000 | Búsqueda secuencial | 46.110 | 46.178 | 46.425 |
| 10,000 | R-Tree propio | 4.434 | 11.442 | 12.116 |
| 10,000 | PostgreSQL + PostGIS (GiST) | 0.552 | 0.622 | 0.718 |
| 100,000 | Búsqueda secuencial | 464.794 | 463.827 | 467.055 |
| 100,000 | R-Tree propio | 4.276 | 12.491 | 18.246 |
| 100,000 | PostgreSQL + PostGIS (GiST) | 0.670 | 0.853 | 1.025 |

## 4. Construcción, disco y memoria

| N | Técnica | Build índice (s) | Carga tabla base (s) | Índice en disco (KB) | Tabla en disco (KB) | Pico mem. build (KB) | Pico mem. consulta k-NN (KB) |
|---|---|---|---|---|---|---|---|
| 1,000 | Búsqueda secuencial | 0.000 | 0.116 | 0.0 | 32.0 | N/A | 11.9 |
| 1,000 | R-Tree propio | 2.208 | 0.116 | 56.0 | 32.0 | 262.1 | 154.3 |
| 1,000 | PostgreSQL + PostGIS (GiST) | 0.003 | 0.037 | 72.0 | 88.0 | N/A | N/A |
| 10,000 | Búsqueda secuencial | 0.000 | 1.384 | 0.0 | 296.0 | N/A | 27.4 |
| 10,000 | R-Tree propio | 36.938 | 1.384 | 596.0 | 296.0 | 2,105.3 | 118.4 |
| 10,000 | PostgreSQL + PostGIS (GiST) | 0.022 | 0.082 | 696.0 | 832.0 | N/A | N/A |
| 100,000 | Búsqueda secuencial | 0.000 | 40.381 | 0.0 | 2,944.0 | N/A | 176.6 |
| 100,000 | R-Tree propio | 448.604 | 40.381 | 5,772.0 | 2,944.0 | 25,927.6 | 117.4 |
| 100,000 | PostgreSQL + PostGIS (GiST) | 0.406 | 0.413 | 7,176.0 | 8,248.0 | N/A | N/A |

Notas: `build` del secuencial es 0 (no hay índice). La memoria de Python se mide con `tracemalloc` en una pasada aparte; PostgreSQL corre en otro proceso y su memoria (`mem_server_backend_bytes` en el CSV, si PG >= 14) no es comparable con la de Python.

## 5. Equivalencia de resultados (referencia: búsqueda secuencial)

| N | Técnica | Configuración | Comparadas | Exactas | Con tolerancia | Diferentes | Exactitud |
|---|---|---|---|---|---|---|---|
| 1,000 | R-Tree propio | range 1 km | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | R-Tree propio | range 5 km | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | R-Tree propio | range 10 km | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | R-Tree propio | knn 10 k | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | R-Tree propio | knn 50 k | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | R-Tree propio | knn 100 k | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | PostgreSQL + PostGIS (GiST) | range 1 km | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | PostgreSQL + PostGIS (GiST) | range 5 km | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | PostgreSQL + PostGIS (GiST) | range 10 km | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | PostgreSQL + PostGIS (GiST) | knn 10 k | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | PostgreSQL + PostGIS (GiST) | knn 50 k | 100 | 100 | 0 | 0 | 100.0 % |
| 1,000 | PostgreSQL + PostGIS (GiST) | knn 100 k | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | R-Tree propio | range 1 km | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | R-Tree propio | range 5 km | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | R-Tree propio | range 10 km | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | R-Tree propio | knn 10 k | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | R-Tree propio | knn 50 k | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | R-Tree propio | knn 100 k | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | PostgreSQL + PostGIS (GiST) | range 1 km | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | PostgreSQL + PostGIS (GiST) | range 5 km | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | PostgreSQL + PostGIS (GiST) | range 10 km | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | PostgreSQL + PostGIS (GiST) | knn 10 k | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | PostgreSQL + PostGIS (GiST) | knn 50 k | 100 | 100 | 0 | 0 | 100.0 % |
| 10,000 | PostgreSQL + PostGIS (GiST) | knn 100 k | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | R-Tree propio | range 1 km | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | R-Tree propio | range 5 km | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | R-Tree propio | range 10 km | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | R-Tree propio | knn 10 k | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | R-Tree propio | knn 50 k | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | R-Tree propio | knn 100 k | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | PostgreSQL + PostGIS (GiST) | range 1 km | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | PostgreSQL + PostGIS (GiST) | range 5 km | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | PostgreSQL + PostGIS (GiST) | range 10 km | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | PostgreSQL + PostGIS (GiST) | knn 10 k | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | PostgreSQL + PostGIS (GiST) | knn 50 k | 100 | 100 | 0 | 0 | 100.0 % |
| 100,000 | PostgreSQL + PostGIS (GiST) | knn 100 k | 100 | 100 | 0 | 0 | 100.0 % |

*Con tolerancia*: range = solo difieren puntos a menos de 0.05 m del borde del radio; k-NN = mismas distancias posición a posición (empates / cuasi-empates con ids distintos). *Diferentes* es un resultado incorrecto y debe investigarse.

## 6. Errores y consultas descartadas

Ninguna: todas las consultas se ejecutaron sin error y dentro del presupuesto.

## 7. Alcance y limitaciones

- Los datos son sintéticos y uniformes; con datos agrupados (ciudades) los resultados cambiarían.
- Los índices propios usan un buffer pool dimensionado para caber en memoria (mismo criterio que `benchmark_indexes.py`): es un escenario de caché caliente.
- El R-Tree se construye con inserciones una a una (no hay bulk-load); PostgreSQL construye el GiST en una sola operación, por lo que el tiempo de build no es estrictamente análogo.
- El k-NN del R-Tree usa radio expansivo (1 km, x2, ...) y repite la consulta en cada ampliación; PostGIS usa búsqueda ordenada nativa del GiST (`<->`).
- Los tiempos de PostgreSQL incluyen red local y deserialización del cliente; los de las técnicas Python incluyen su propio overhead de intérprete.
- El `timeout` por consulta solo puede interrumpir a PostgreSQL; en Python se aplica el presupuesto por celda (las consultas restantes se descartan y se cuentan).

## 8. Análisis y conclusiones

_Basado en la corrida completa descrita arriba (3 tamaños, 6 configuraciones, 100 consultas, 3 repeticiones, semilla 42). Las comparaciones entre técnicas Python (secuencial y R-Tree) son directas; las comparaciones contra PostgreSQL mezclan algoritmo e implementación (C vs Python), como se detalla en 8.4._

### 8.1 Hallazgos principales

1. **Los resultados son equivalentes.** Las 3,600 comparaciones (36 celdas x 100 consultas) del R-Tree y de PostGIS contra la búsqueda secuencial fueron exactas: ninguna "con tolerancia" ni "diferente". No hubo errores ni consultas descartadas, y el planificador de PostgreSQL usó el índice GiST en todas las celdas. Por tanto, las diferencias de tiempo comparan técnicas que devuelven lo mismo, y `<->` opera sobre la misma esfera que el Haversine del proyecto.

2. **La búsqueda secuencial escala linealmente y no depende del radio ni de k.** Tarda ~4.5 ms con 1,000 puntos, ~45 ms con 10,000 y ~455 ms con 100,000: cada x10 en N cuesta x10 en tiempo, porque decodifica todo el heap en cada consulta.

3. **El R-Tree propio supera a la secuencial, con matices.**
   - En búsqueda por radio gana en todos los tamaños, hasta 107x a N = 100,000 con 1 km. La ventaja se reduce al crecer el radio (6.4x con 10 km), porque ese radio devuelve ~2,370 puntos por consulta (2.4 % de la tabla) y cada uno exige una lectura del heap (~30 us por resultado).
   - En k-NN, con N = 1,000 la secuencial gana para k = 50 y k = 100 (4.7 vs 8.0 ms y 4.8 vs 12.7 ms) y hay empate para k = 10 (4.4 vs 4.6 ms). Desde N = 10,000 el R-Tree gana para todo k, entre 4x y 10x. El punto de cruce está entre 1,000 y 10,000 puntos.

4. **PostgreSQL + PostGIS es la técnica más rápida en todas las celdas.** Su k-NN es casi plano (0.50 ms con 1,000 puntos y 1.02 ms con 100,000 para k = 100), y su ventaja sobre el R-Tree crece con N y con el tamaño del resultado: de 4.8x (1 km) a 15.8x (10 km) en radio, y de 6.4x (k = 10) a 17.8x (k = 100) en k-NN con N = 100,000. Con N pequeño el piso de ~0.5 ms es sobre todo el viaje cliente-servidor.

5. **La construcción es la mayor debilidad del R-Tree propio.** Con N = 100,000 tarda 448.6 s frente a 0.41 s del GiST (~1,100x). El costo por inserción sube de 2.2 ms (N = 1,000) a 4.5 ms (N = 100,000). Es coherente con que cada inserción deserializa y reserializa páginas completas (con CRC) y con que no hay bulk-load, mientras que PostgreSQL construye el índice en bloque.

6. **Espacio en disco: parecido en el índice, muy distinto en la tabla.** Con N = 100,000 el índice del R-Tree ocupa 5.8 MB (~59 bytes por punto) y el GiST 7.2 MB (~73 bytes por punto), un ~20 % menos para el R-Tree. La tabla base pesa 2.9 MB en el heap propio y 8.2 MB en PostgreSQL (cabecera por tupla y columna `geography`; no incluye el índice de la clave primaria), así que ese segundo dato no es una comparación de índices.

7. **Memoria.** El pico durante una consulta es pequeño y no crece con N en las técnicas Python (~120-180 KB). Construir el R-Tree con N = 100,000 llega a ~26 MB, unas 4.5 veces el tamaño de su archivo (buffer pool más copias temporales de serialización). El backend de PostgreSQL reporta ~2.3 MB constantes, pero solo cuenta sus contextos de memoria (no los 128 MB de `shared_buffers`), por lo que no es comparable.

8. **Estabilidad de la medición.** La desviación estándar entre repeticiones es de ~3 % o menos de la media en las celdas con N = 100,000. En el R-Tree hay cola larga: el p95 llega a 2.7x la mediana (k-NN con k = 100: mediana 13.6 ms, media 18.2 ms, p95 36.2 ms), consistente con que algunas consultas necesitan una ampliación extra del radio.

### 8.2 Aceleración con N = 100,000

| Consulta (N = 100,000) | Secuencial (ms) | R-Tree (ms) | PostGIS (ms) | R-Tree vs secuencial | PostGIS vs secuencial | PostGIS vs R-Tree |
|---|---|---|---|---|---|---|
| Radio 1 km | 455.4 | 4.2 | 0.89 | 107x | 514x | 4.8x |
| Radio 5 km | 457.1 | 25.2 | 2.11 | 18x | 216x | 11.9x |
| Radio 10 km | 453.6 | 70.8 | 4.48 | 6x | 101x | 15.8x |
| k-NN k=10 | 464.8 | 4.3 | 0.67 | 109x | 694x | 6.4x |
| k-NN k=50 | 463.8 | 12.5 | 0.85 | 37x | 544x | 14.6x |
| k-NN k=100 | 467.1 | 18.2 | 1.02 | 26x | 456x | 17.8x |

### 8.3 Cuándo conviene cada técnica

- **Búsqueda secuencial:** tablas muy pequeñas (del orden de 1,000 puntos o menos) o consultas k-NN con k grande sobre pocos datos. No requiere construir ni mantener un índice, pero su costo crece linealmente con la tabla.
- **R-Tree propio:** vale la pena a partir de ~10,000 puntos y, para búsquedas por radio, desde tamaños menores. Es la opción correcta dentro del motor, porque está integrado con el catálogo, el heap y el resto del pipeline. Su costo de construcción es alto y su ventaja disminuye cuando el radio devuelve muchos puntos.
- **PostgreSQL + PostGIS:** la mejor opción de rendimiento en todos los escenarios medidos, a costa de depender de un servidor externo, que además queda fuera del alcance de un motor educativo autocontenido.

### 8.4 Limitaciones y trabajo futuro

- **Comparación con PostgreSQL:** parte de la brecha frente al R-Tree se debe a que PostgreSQL está implementado en C y el R-Tree en Python, no solo al algoritmo. Los tiempos de PostgreSQL incluyen red local y deserialización en el cliente.
- **Datos y caché:** los puntos son sintéticos y uniformes (con datos agrupados, como ciudades, los resultados cambiarían), y el buffer pool de los índices propios cabe en memoria (escenario de caché caliente).
- **Mejoras posibles para el R-Tree:**
  - un bulk-load (por ejemplo, ordenamiento STR) para reducir el tiempo de construcción;
  - un k-NN con cola de prioridad (best-first) en lugar de radio expansivo, que repite trabajo en cada ampliación;
  - devolver las coordenadas desde `range_query` para evitar una lectura del heap por candidato, que probablemente domina el costo con radios grandes;
  - evitar deserializar y reserializar la página completa en cada inserción.