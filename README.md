# Multimodal DBMS Lite

Multimodal DBMS Lite es un prototipo de gestor de bases de datos desarrollado como proyecto académico para comprender el funcionamiento interno de un motor relacional a pequeña escala. La arquitectura está dividida en tres capas principales:

- capa de almacenamiento y catálogo
- capa de consulta y planificación
- capa de acceso HTTP y frontend

El sistema soporta parte del flujo de un motor SQL clásico: análisis léxico/sintáctico, validación semántica, reescritura, planificación, ejecución y presentación de resultados a través de una interfaz web.

## Objetivo del proyecto

El proyecto busca implementar, de forma progresiva y modular, conceptos clave de los sistemas de bases de datos modernos, incluyendo:

- gestión de archivos y páginas en disco
- almacenamiento relacional mediante heap y archivos secuenciales
- registro de metadatos y esquemas
- índices B+ y hashing extensible
- parser SQL básico
- ejecución de consultas SELECT, INSERT, DELETE y UPDATE
- ejecución de planes lógicos y físicos
- API REST para acceso desde una interfaz frontend

## Arquitectura general

La solución sigue una organización por capas:

```text
multimodal-dbms-lite/
├── src/                 # lógica principal del motor de base de datos
│   ├── catalog/         # catálogo de tablas, columnas y metadatos
│   ├── common/          # estructuras comunes y valores
│   ├── index/           # índices (B+ y hash dinámico)
│   ├── query/           # parser, planner, rewriter y executor
│   ├── storage/         # heap files, buffer manager, file manager, páginas
│   ├── transaction/     # transacciones y locks
│   ├── main.py          # configuración del catálogo y factories
│   └── engine.py        # acceso desde la API
├── api/                 # API REST con FastAPI
│   ├── routers/         # endpoints de tablas y query
│   ├── engine.py        # wrapper para el catálogo del motor
│   ├── main.py          # aplicación FastAPI
│   ├── schemas.py       # modelos de request
│   ├── serialize.py     # serialización de registros
│   └── explain.py       # descripción del plan de ejecución
├── frontend/            # aplicación React + Vite
│   ├── src/             # componentes y lógica de presentación
│   ├── package.json     # dependencias del frontend
│   ├── vite.config.ts   # configuración de desarrollo
│   └── .env.example     # ejemplo de variables de entorno
├── tests/               # pruebas del motor y persistencia
├── requirements.txt     # dependencias del backend
├── pyproject.toml       # configuración PyProject
├── pytest.ini           # configuración de pytest
├── LICENSE              # licencia del proyecto
└── README.md            # documentación técnica
```

## Módulos principales

### 1. Storage layer

La capa `src/storage` implementa la gestión de archivos físicos y de páginas. Incluye:

- `file_manager.py`: acceso a archivos del sistema
- `buffer_manager.py`: manejo del buffer de páginas
- `page.py`: representación de páginas de disco
- `heap/`: almacenamiento en heap
- `sequential/`: almacenamiento secuencial paginado

Esta capa es la base del almacenamiento persistente y del manejo de registros.

### 2. Catalog layer

La carpeta `src/catalog` gestiona información estructural del sistema:

- tablas
- columnas
- tipos de datos
- restricciones
- almacenamiento asociado

La clase principal es `Catalog`, que administra los metadatos y conecta cada tabla con su storage físico.

### 3. Index layer

La carpeta `src/index` encapsula las estructuras de indexación, como:

- B+ tree
- extendible hashing
- acceso por columnas e índices secundarios

Estas estructuras permiten acelerar búsquedas y mejorar la ejecución de consultas.

### 4. Query engine

La carpeta `src/query` contiene la lógica del motor:

- `parser/`: tokenización, análisis sintáctico y AST
- `rewriter/`: optimizaciones básicas sobre expresiones
- `planner/`: construcción de planes físicos
- `executor/`: ejecución de nodos de plan

En esta capa se procesa el SQL y se transforma en un plan ejecutable.

### 5. Transactions

La carpeta `src/transaction` maneja:

- locks
- transacciones
- manejo de undo/log
- recuperación y demostración de concurrencia

### 6. API REST

La API se encuentra en `api/` y expone dos endpoints principales:

- `GET /tables`: lista las tablas del catálogo
- `GET /tables/{table_name}`: detalles de una tabla y sus índices
- `POST /query`: ejecuta una sentencia SQL
- `GET /health`: comprueba el estado del servicio

La API usa FastAPI y serializa los resultados para que el frontend pueda presentarlos en la UI.

### 7. Frontend

La aplicación frontend está en `frontend/` y usa React + Vite. Presenta 4 paneles principales:

- archivos/tablas
- editor de consultas SQL
- resultados
- plan de ejecución

La interfaz se comunica con la API mediante `fetch` utilizando la variable `VITE_API_BASE_URL`.

## Requisitos previos

Para ejecutar el backend y el frontend necesitas:

- Python 3.13+
- Node.js 18+
- npm
- entorno virtual de Python recomendado

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd multimodal-dbms-lite
```

### 2. Crear entorno virtual de Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias del backend

```bash
pip install -r api/requirements.txt
pip install -r requirements.txt
```

### 4. Instalar dependencias del frontend

```bash
cd frontend
npm install
```

## Ejecución

### Backend

Desde la raíz del proyecto:

```bash
source .venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

La API quedará disponible en:

```text
http://127.0.0.1:8000
```

### Frontend

En otra terminal:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

La UI quedará disponible en:

```text
http://localhost:5173
```

Si quieres apuntar a otra API, crea un archivo `.env` dentro de `frontend/` usando como referencia `.env.example`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Verificación

Se incluye una suite de pruebas con pytest. Para ejecutarlas:

```bash
source .venv/bin/activate
pytest -q
```

## Base de datos espacial

El motor soporta datos geográficos 2D con tipos SQL propios, un índice R-Tree
persistente y una extensión del parser para consultas por radio, k-NN y
polígonos. Los value objects `Point`, `Rectangle`, `Polygon` y `Distance` se
encuentran en `src/common/value.py`, y la geometría y las métricas en `src/spatial/`.
La sintaxis SQL completa está en
[`src/query/parser/GRAMMAR.md`](src/query/parser/GRAMMAR.md).

### Reglas de representación

- `POINT(longitud, latitud)` usa ese orden y almacena ambas coordenadas en grados. Longitud acepta `[-180, 180]` y latitud `[-90, 90]`, incluidos los extremos; no se convierten ni recortan valores fuera de rango.
- Un rectángulo se representa por oeste, sur, este, norte. `oeste > este` indica que envuelve el antimeridiano.
- Un polígono es una secuencia de puntos cerrada con al menos tres vértices distintos y el primer punto repetido al final. Sus lados son segmentos en el plano longitud/latitud; el anillo no puede cruzar el antimeridiano.
- `HAVERSINE` usa el arco longitudinal más corto y una esfera de radio `6 371 008.8 m`; distancia y radio de consulta se expresan en metros. `EUCLIDEAN` opera directamente sobre longitud/latitud y expresa distancia y radio en unidades de coordenada (grados).
- Los límites de rangos son inclusivos. Un punto sobre el borde pertenece al rango y dos geometrías que solo se tocan cuentan como intersección.
- Los polos se aceptan. Para Haversine, las longitudes representan el mismo punto en cada polo; para Euclidiana y rangos se conservan las coordenadas numéricas sin esa equivalencia.
- Coordenadas no numéricas, booleanas, no finitas o fuera de rango, rectángulos con `sur > norte`, anillos abiertos o mal formados y radios negativos son inválidos y se rechazan; no hay coerción ni normalización silenciosa.
- SQL `NULL` se representa en Python como `None`. Las columnas nullable pueden contenerlo; operaciones espaciales con un operando `NULL` producen `NULL` y no satisfacen un `WHERE`.

> **Orden de coordenadas:** Lima es `POINT(-77.0428, -12.0464)`. Escribir
> `POINT(-12.0464, -77.0428)` (latitud primero) no da error, porque ambos
> valores están en rango, pero ubica el punto en la Antártida.

### Tipos espaciales

| Tipo | Literal SQL | Uso |
|---|---|---|
| `POINT` | `POINT(longitud, latitud)` | ubicaciones (tiendas, restaurantes, etc.) |
| `RECTANGLE` | `RECTANGLE(oeste, sur, este, norte)` | zonas rectangulares |
| `POLYGON` | `POLYGON(POINT(...), POINT(...), ...)` | zonas irregulares (distritos) |
| `GEOMETRY` | cualquiera de los anteriores | columna genérica |

### Crear una tabla espacial

```sql
CREATE TABLE tiendas (
    id INTEGER PRIMARY KEY,
    nombre VARCHAR(100),
    ubicacion POINT
);
```

### Insertar puntos

```sql
INSERT INTO tiendas (id, nombre, ubicacion) VALUES
    (1, 'Tienda Miraflores', POINT(-77.0300, -12.1200)),
    (2, 'Tienda San Isidro', POINT(-77.0365, -12.0970)),
    (3, 'Tienda Centro',     POINT(-77.0428, -12.0464));
```

El archivo [`sql/querys/places_postgis_inserts.sql`](sql/querys/places_postgis_inserts.sql)
contiene un dataset de 500 lugares de Lima, generado con
[`sql/scripts/generate_places_postgis.py`](sql/scripts/generate_places_postgis.py).

### Crear un índice R-Tree

```sql
CREATE INDEX idx_tiendas_ubicacion ON tiendas(ubicacion) USING RTREE;
```

- Solo se admite sobre columnas `POINT`, `POLYGON`, `RECTANGLE` o `GEOMETRY`.
- Puede crearse antes o después de cargar los datos.
- El planner lo usa automáticamente: el plan muestra `SpatialIndexScan`
  (candidatos por MBR) seguido de `SpatialFilter` (verificación exacta).
  Sin índice, el plan usa `SeqScan` + `SpatialFilter`. Para verlo:
  `EXPLAIN SELECT ...;`

### Consultas por radio

```sql
-- Tiendas a menos de 5 km (Haversine, en metros: métrica por defecto)
SELECT * FROM tiendas
WHERE distancia(ubicacion, POINT(-77.0428, -12.0464)) < 5000;

-- Misma idea con distancia euclidiana (en grados)
SELECT * FROM tiendas
WHERE distancia(ubicacion, POINT(-77.0428, -12.0464), EUCLIDEAN) <= 0.05;
```

El R-Tree se consulta con el rectángulo que envuelve el círculo, y luego se
descartan los falsos positivos calculando la distancia exacta.

### Consultas k-NN

```sql
-- Las 10 tiendas más cercanas
SELECT * FROM tiendas
ORDER BY distancia(ubicacion, POINT(-77.0428, -12.0464)) LIMIT 10;
```

Con índice, el k-NN busca en un radio inicial (1 km en Haversine, 0.01° en
Euclidiana) y lo duplica hasta reunir al menos `k` puntos; después ordena por
distancia exacta.

### Consultas con polígonos

```sql
-- Tiendas dentro de un polígono (anillo cerrado: el primer punto se repite al final)
SELECT * FROM tiendas
WHERE dentro_de(ubicacion, POLYGON(
    POINT(-77.06, -12.13),
    POINT(-77.00, -12.13),
    POINT(-77.00, -12.08),
    POINT(-77.06, -12.08),
    POINT(-77.06, -12.13)
));
```

### Métricas de distancia

| Métrica | Sintaxis | Unidad de distancia y radio | Cuándo usarla |
|---|---|---|---|
| Haversine (por defecto) | `distancia(col, POINT(...))` o `distancia(col, POINT(...), HAVERSINE)` | metros | distancias reales sobre la superficie terrestre |
| Euclidiana | `distancia(col, POINT(...), EUCLIDEAN)` | grados de coordenada | comparaciones rápidas en zonas pequeñas o datos no geográficos |

### Panel de mapa

El panel **Spatial Search & Map** del frontend permite buscar sin escribir SQL:

1. Elegir la tabla y la columna espacial. El panel indica si la columna tiene
   R-Tree (`R-Tree #id`) o si se usará `Scan secuencial`.
2. Elegir el modo: **Radio** (presets o valor libre, en km para Haversine o en
   grados para Euclidiana) o **k-NN** (10, 50 o 100 vecinos).
3. Elegir la métrica e ingresar latitud y longitud del centro (por defecto, Lima).
4. Pulsar **Buscar**. El panel genera la consulta `distancia(...)`, la ejecuta y
   muestra en el mapa:
   - el punto de búsqueda y el círculo del radio;
   - los resultados resaltados;
   - en gris, los MBR de los nodos del R-Tree; en naranja punteado, las cajas
     de candidatos consultadas.

Las consultas con polígonos se escriben en el panel de consultas; sus resultados
también se dibujan en el mapa. El mapa base usa teselas de OpenStreetMap, por lo
que requiere conexión a internet.

### Reproducir los benchmarks espaciales

El benchmark espacial compara búsqueda secuencial, R-Tree propio y PostgreSQL +
PostGIS (GiST). Su configuración, requisitos y opciones están en la sección
[Benchmarks](#benchmarks). Corrida completa:

```bash
pip install -r requirements.txt
python benchmarks/benchmark_spatial.py
```

Los resultados se escriben en `benchmarks/results/spatial_results.csv`, el
reporte en [`benchmarks/results/spatial_summary.md`](benchmarks/results/spatial_summary.md)
y las gráficas en `benchmarks/results/charts/`.

### Resultados experimentales

Puntos uniformes en Lima, 100 consultas por configuración, 3 repeticiones,
semilla 42 y distancia Haversine. El R-Tree y PostGIS devolvieron exactamente
los mismos resultados que la búsqueda secuencial en las 3 600 comparaciones.

**Tiempo promedio de consulta con N = 100 000 (ms)**

| Consulta | Secuencial | R-Tree propio | PostGIS (GiST) | R-Tree vs. secuencial |
|---|---|---|---|---|
| Radio 1 km | 455.4 | 4.2 | 0.89 | 107x |
| Radio 5 km | 457.1 | 25.2 | 2.11 | 18x |
| Radio 10 km | 453.6 | 70.8 | 4.48 | 6x |
| k-NN k = 10 | 464.8 | 4.3 | 0.67 | 109x |
| k-NN k = 50 | 463.8 | 12.5 | 0.85 | 37x |
| k-NN k = 100 | 467.1 | 18.2 | 1.02 | 26x |

**Construcción y espacio del índice**

| N | Build R-Tree | Build GiST | Índice R-Tree | Índice GiST |
|---|---|---|---|---|
| 1 000 | 2.2 s | 0.003 s | 56 KB | 72 KB |
| 10 000 | 36.9 s | 0.022 s | 596 KB | 696 KB |
| 100 000 | 448.6 s | 0.406 s | 5 772 KB | 7 176 KB |

![Tiempo de búsqueda por radio](benchmarks/results/charts/spatial_range_query_time.png)
![Tiempo de k-NN](benchmarks/results/charts/spatial_knn_query_time.png)

**Conclusiones**

- La búsqueda secuencial escala linealmente (~4.5 ms, ~45 ms y ~455 ms para
  1 000, 10 000 y 100 000 puntos) y no depende del radio ni de k.
- El R-Tree propio gana en búsqueda por radio en todos los tamaños. Su ventaja
  baja al crecer el radio, porque cada resultado exige una lectura del heap.
  En k-NN el punto de cruce con la búsqueda secuencial está entre 1 000 y
  10 000 puntos.
- PostGIS es la técnica más rápida en todas las configuraciones. Su ventaja
  sobre el R-Tree crece con N y con el tamaño del resultado, y combina
  algoritmo e implementación (C frente a Python).
- La construcción es la mayor debilidad del R-Tree propio (~1 100x más lenta que
  GiST con 100 000 puntos), por la inserción registro por registro sin bulk-load.
  En cambio, su índice ocupa ~20 % menos que el GiST.

### Cuándo usar SeqScan, R-Tree o PostGIS GiST

| Técnica | Conviene cuando | Evitar cuando |
|---|---|---|
| SeqScan (sin índice) | tablas muy pequeñas (≈ 1 000 puntos o menos); k-NN con k grande sobre pocos datos; tablas con muchas escrituras y pocas consultas espaciales | tablas grandes: su costo crece linealmente con N |
| R-Tree propio | desde ≈ 10 000 puntos (antes en búsquedas por radio); consultas selectivas dentro del motor, integradas con el catálogo, el planner y el panel de mapa | cargas masivas frecuentes (construcción lenta) o radios que devuelven gran parte de la tabla |
| PostGIS GiST | máximo rendimiento en cualquier escenario medido; datasets grandes y carga masiva | cuando no se dispone de un servidor PostgreSQL externo |

### Limitaciones espaciales conocidas

- El R-Tree solo poda la búsqueda por radio cuando la consulta compara
  `distancia(col, POINT(...))` con un número usando `<`, `<=` o `=`. Con `>` o
  `>=` se recorre todo el índice.
- El k-NN usa el índice solo con `LIMIT` y un `POINT` literal. Si el segundo
  argumento es una columna, se ordena la tabla completa.
- El k-NN usa radio expansivo y repite la búsqueda en cada ampliación, en lugar
  de una búsqueda best-first con cola de prioridad.
- Los predicados espaciales sobre un `JOIN` no usan el índice (se filtra después del join).
- La distancia euclidiana está en grados, no en metros; en el mapa, su círculo
  se dibuja con una aproximación de 111 320 m por grado.
- El índice se construye insertando registro por registro (sin bulk-load), lo
  que hace lenta su creación con 100 000 puntos.


## Benchmarks

Los scripts de `benchmarks/` generan reportes y gráficas (no son pass/fail); la suite formal de pruebas vive en `tests/`.

| Script | Compara | Salidas |
|---|---|---|
| `benchmark_indexes.py` | B+ clusterizado vs. B+ no clusterizado vs. Hash dinámico | `results/index_comparison_*.csv/md`, `results/charts/` |
| `benchmark_spatial.py` | Búsqueda secuencial vs. R-Tree propio vs. PostgreSQL + PostGIS (GiST) | `results/spatial_results.csv`, `results/spatial_summary.md`, `results/charts/spatial_*.png` |

### Benchmark espacial (`benchmark_spatial.py`)

Configuración: N = 1 000 / 10 000 / 100 000 puntos; búsqueda por radio de 1, 5 y 10 km; k-NN con k = 10, 50 y 100; 100 consultas por configuración con los mismos centros para las tres técnicas; semilla fija (`SEED = 42`); warm-up de 10 consultas por celda; 3 repeticiones (cada una reconstruye los índices). Se mide tiempo de construcción, tiempo promedio de consulta (media, mediana, p95), memoria, espacio en disco y exactitud frente a la búsqueda secuencial. La distancia es Haversine en metros (esfera de 6 371 008.8 m), igual que `src/spatial/distance.py`.

```bash
pip install -r requirements.txt

python benchmarks/benchmark_spatial.py --quick       # corrida corta (humo); escribe *_quick.csv/md y charts_quick/
python benchmarks/benchmark_spatial.py               # corrida completa
python benchmarks/benchmark_spatial.py --no-postgres # sin PostgreSQL
python benchmarks/benchmark_spatial.py --help        # --sizes, --queries, --repetitions, --seed, ...
```

La corrida completa es lenta (el R-Tree se construye con inserciones una a una y la búsqueda secuencial recorre todo el heap en cada consulta). Para una primera prueba conviene `--sizes 1000,10000 --repetitions 1`.

#### PostgreSQL + PostGIS

Se necesita una instancia de PostgreSQL con la extensión PostGIS (local, en Docker o un servidor del curso). La conexión se lee de la variable de entorno `PG_DSN` (o `--pg-dsn`). Si no está definida o la conexión falla, **PostGIS se omite** y las filas del CSV quedan con `status = OMITIDO_SIN_POSTGRES...`; el resto del benchmark continúa.

Una forma rápida de tener PostGIS es con Docker:

```bash
docker run --name dbms-postgis-bench -e POSTGRES_PASSWORD=postgres -p 5433:5432 -d postgis/postgis:16-3.4
export PG_DSN="postgresql://postgres:postgres@localhost:5433/postgres"   # Windows (PowerShell): $env:PG_DSN="..."
python benchmarks/benchmark_spatial.py
```

(El puerto 5433 evita chocar con un PostgreSQL local en el 5432.)

El benchmark crea (y al terminar borra, salvo `--keep-pg-data`) el esquema `bench_spatial` con la tabla `bench_points(id, lon, lat, geog geography(Point,4326))` y un índice GiST sobre `geog`. Rango: `ST_DWithin(..., false)` (esfera); k-NN: `ORDER BY geog <-> punto LIMIT k`. Ajustes de sesión: `max_parallel_workers_per_gather = 0` y `jit = off` (las otras técnicas son de un solo hilo). La versión de PostgreSQL/PostGIS y los parámetros efectivos (`shared_buffers`, `work_mem`, ...) se registran automáticamente en `spatial_summary.md`, junto con si el planificador usó realmente el índice GiST (`EXPLAIN`).

#### Errores, descartes y equivalencia

Las consultas que fallan, o que ya no caben en el presupuesto de tiempo de la celda (`--cell-budget-s`, 900 s), se **descartan y se cuentan** (`queries_discarded`, `query_errors`, `error_samples` en el CSV y sección 6 del resumen). Los resultados de R-Tree y PostGIS se comparan consulta a consulta con la búsqueda secuencial: *exacto*, *con tolerancia* (`--tolerance-m`, solo puntos en el borde del radio o empates en k-NN) o *diferente* (error).

## Nota

Este sistema no pretende ser un motor SQL completo comparable con PostgreSQL o MySQL; su objetivo principal es servir como laboratorio de aprendizaje para entender los principios internos de un sistema de gestión de datos.