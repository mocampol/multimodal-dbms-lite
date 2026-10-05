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
- índices B+, hashing extensible y R-Tree espacial
- parser SQL básico
- ejecución de consultas SELECT, INSERT, DELETE y UPDATE
- ejecución de planes lógicos y físicos
- API REST para acceso desde una interfaz frontend

## Arquitectura general

La solución sigue una organización por capas:

```text
multimodal-dbms-lite/
├── api/                     # API REST con FastAPI
│   └── routers/             # endpoints de tablas, consultas e importación
├── benchmarks/              # mediciones de rendimiento
│   ├── postgres/            # comparación con PostgreSQL/PostGIS
│   └── results/             # resultados y gráficas versionados
├── frontend/                # aplicación React + Vite
│   └── src/                 # componentes y lógica de presentación
├── sql/
│   ├── examples/            # scripts SQL de ejemplo
│   ├── generated/           # SQL producido por generadores
│   └── scripts/             # generadores y utilidades SQL
├── src/                     # motor de base de datos
│   ├── catalog/             # tablas, columnas y metadatos
│   ├── common/              # tipos, esquemas y registros
│   ├── index/               # B+, hash extensible y R-Tree
│   ├── loader/              # lectura e importación de CSV
│   ├── query/               # parser, planner, rewriter y executor
│   ├── spatial/             # geometría y operaciones espaciales
│   ├── storage/             # páginas, buffer manager, heap y secuencial
│   ├── transaction/         # transacciones, locks y recuperación
│   └── main.py              # configuración del motor
├── tests/                   # pruebas organizadas por módulo
├── pyproject.toml           # configuración del proyecto y pytest
├── requirements.txt         # dependencias de benchmarks
├── pytest.ini               # configuración de pytest
├── uv.lock                  # versiones bloqueadas de dependencias
├── LICENSE
└── README.md
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
El R-Tree se encuentra en `src/index/rtree/` y se utiliza para consultas espaciales.

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
git clone https://github.com/mocampol/multimodal-dbms-lite
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

## Contrato espacial (preliminar)

La representación espacial normativa está definida antes de añadir soporte al
parser SQL, al catálogo o al almacenamiento. Los value objects `Point`,
`Rectangle`, `Polygon` y `Distance` se encuentran en `src/common/value.py`;
todavía no son tipos SQL declarables ni persistibles.

- `POINT(longitud, latitud)` usa ese orden y almacena ambas coordenadas en grados. Longitud acepta `[-180, 180]` y latitud `[-90, 90]`, incluidos los extremos; no se convierten ni recortan valores fuera de rango.
- Un rectángulo se representa por oeste, sur, este, norte. `oeste > este` indica que envuelve el antimeridiano.
- Un polígono es una secuencia de puntos cerrada con al menos tres vértices distintos y el primer punto repetido al final. Sus lados son segmentos en el plano longitud/latitud; el anillo no puede cruzar el antimeridiano.
- `HAVERSINE` usa el arco longitudinal más corto y una esfera de radio `6 371 008.8 m`; distancia y radio de consulta se expresan en metros. `EUCLIDEAN` opera directamente sobre longitud/latitud y expresa distancia y radio en unidades de coordenada (grados).
- Los límites de rangos son inclusivos. Un punto sobre el borde pertenece al rango y dos geometrías que solo se tocan cuentan como intersección.
- Los polos se aceptan. Para Haversine, las longitudes representan el mismo punto en cada polo; para Euclidiana y rangos se conservan las coordenadas numéricas sin esa equivalencia.
- Coordenadas no numéricas, booleanas, no finitas o fuera de rango, rectángulos con `sur > norte`, anillos abiertos o mal formados y radios negativos son inválidos y se rechazan; no hay coerción ni normalización silenciosa.
- SQL `NULL` se representa en Python como `None`. Las columnas nullable pueden contenerlo; operaciones espaciales con un operando `NULL` producen `NULL` y no satisfacen un `WHERE`.

La sintaxis SQL propuesta y los ejemplos normativos están en
[`src/query/parser/GRAMMAR.md`](src/query/parser/GRAMMAR.md).

## Benchmarks

Los scripts de `benchmarks/` generan reportes y gráficas (no son pass/fail); la suite formal de pruebas vive en `tests/`.

| Script                 | Compara                                                               | Salidas                                                                                     |
| ---------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `benchmark_indexes.py` | B+ clusterizado vs. B+ no clusterizado vs. Hash dinámico              | `results/index_comparison_*.csv/md`, `results/charts/`                                      |
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

Las consultas que fallan, o que ya no caben en el presupuesto de tiempo de la celda (`--cell-budget-s`, 900 s), se **descartan y se cuentan** (`queries_discarded`, `query_errors`, `error_samples` en el CSV y sección 6 del resumen). Los resultados de R-Tree y PostGIS se comparan consulta a consulta con la búsqueda secuencial: _exacto_, _con tolerancia_ (`--tolerance-m`, solo puntos en el borde del radio o empates en k-NN) o _diferente_ (error).

## Nota

Este sistema no pretende ser un motor SQL completo comparable con PostgreSQL o MySQL; su objetivo principal es servir como laboratorio de aprendizaje para entender los principios internos de un sistema de gestión de datos.
