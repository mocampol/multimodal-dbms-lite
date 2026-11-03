# Estructura actual del proyecto

**Proyecto:** Multimodal DBMS Lite  
**Actualizado:** 19 de octubre de 2026

Este documento resume la organización actual del repositorio. Se describen los módulos de código, las pruebas y los recursos auxiliares. Se omiten archivos generados de caché como `__pycache__` y `.pytest_cache`.

## Vista general

```text
multimodal-dbms-lite/
├── api/                 # API HTTP con FastAPI
├── benchmarks/          # mediciones, comparaciones y sus resultados
├── frontend/            # cliente web con React, TypeScript y Vite
├── sql/                 # ejemplos y generación de scripts SQL
├── src/                 # implementación del motor DBMS
├── tests/               # pruebas automatizadas por componente
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── uv.lock
```

## Árbol detallado

```text
multimodal-dbms-lite/
├── api/
│   ├── __init__.py
│   ├── engine.py                 # conexión de la API con el motor
│   ├── explain.py                # presentación de planes de ejecución
│   ├── main.py                   # aplicación FastAPI
│   ├── requirements.txt          # dependencias específicas de la API
│   ├── schemas.py                # modelos de entrada y salida
│   ├── serialize.py              # serialización de resultados
│   └── routers/
│       ├── __init__.py
│       ├── imports.py            # rutas de importación de datos
│       ├── query.py              # ejecución de consultas
│       └── tables.py             # acceso a tablas y metadatos
│
├── benchmarks/
│   ├── README.md                 # guía para ejecutar los benchmarks
│   ├── benchmark_btree.py
│   ├── benchmark_indexes.py
│   ├── benchmark_spatial.py
│   ├── perf_heap_vs_sequential.py
│   ├── spatial_common.py         # utilidades compartidas para pruebas espaciales
│   ├── postgres/                 # entorno de comparación PostgreSQL/PostGIS
│   │   ├── docker-compose.yml
│   │   ├── load_data.py
│   │   ├── queries.sql
│   │   ├── requirements.txt
│   │   └── schema.sql
│   └── results/                  # salidas guardadas de las mediciones
│       ├── heap_vs_sequential_results_*.csv
│       ├── postgis_environment.json
│       ├── spatial_comparison_results.csv
│       ├── spatial_results.csv
│       ├── spatial_results_quick.csv
│       ├── spatial_summary.md
│       ├── spatial_summary_quick.md
│       ├── charts/               # gráficos de resultados espaciales
│       └── charts_quick/         # gráficos de la ejecución rápida
│
├── frontend/
│   ├── .env.example              # ejemplo de configuración de la API
│   ├── .gitignore
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx               # composición principal de la interfaz
│       ├── CsvImport.tsx         # importación de CSV
│       ├── ExecutionPlanPanel.tsx
│       ├── FilesPanel.tsx
│       ├── QueryPanel.tsx
│       ├── ResultsPanel.tsx
│       ├── SpatialMapPanel.tsx   # visualización espacial
│       ├── api.ts                # comunicación con la API
│       ├── index.css
│       └── main.tsx              # punto de entrada de React
│
├── sql/
│   ├── examples/                 # scripts SQL de ejemplo y carga
│   │   ├── inserts_lima.sql
│   │   └── inserts_masivos.sql
│   ├── generated/                # scripts SQL generados
│   │   └── places_postgis_inserts.sql
│   └── scripts/                  # utilidades para generar SQL
│       └── generate_places_postgis.py
│
├── src/                          # motor de base de datos
│   ├── main.py                   # creación/configuración del catálogo y storage
│   ├── catalog/                  # catálogo y metadatos
│   │   ├── catalog.py
│   │   ├── column.py
│   │   ├── exceptions.py
│   │   └── table_metadata.py
│   ├── common/                   # tipos, esquemas y registros
│   │   ├── record.py
│   │   ├── schema.py
│   │   └── value.py
│   ├── index/                    # estructuras de indexación
│   │   ├── btree/                # árbol B/B+ y variante clustered
│   │   ├── extendible_hash/      # hashing extensible
│   │   ├── rtree/                # índice espacial R-tree
│   │   ├── text/                 # índices de texto invertido y SPIMI
│   │   ├── multimedia/           # extracción de características (SIFT, MFCC, BoW, TF-IDF)
│   │   └── vector/               # índices vectoriales de similitud (IVF, HNSW)
│   ├── loader/                   # lectura de CSV e inferencia de tipos
│   │   ├── csv_loader.py
│   │   ├── csv_reader.py
│   │   ├── exceptions.py
│   │   └── type_inference.py
│   ├── query/                    # procesamiento de consultas
│   │   ├── query_engine.py
│   │   ├── parser/               # scanner, parser, AST y gramática
│   │   │   ├── GRAMMAR.md
│   │   │   ├── ast_nodes.py
│   │   │   ├── exceptions.py
│   │   │   ├── parser.py
│   │   │   ├── scanner.py
│   │   │   ├── token_.py
│   │   │   └── visitor.py
│   │   ├── planner/              # construcción de planes
│   │   │   └── plan_builder.py
│   │   ├── rewriter/             # reescrituras y optimizaciones
│   │   │   ├── rewriter.py
│   │   │   └── rules/
│   │   │       ├── predicate_pushdown.py
│   │   │       └── simplify_expressions.py
│   │   └── executor/             # ejecución física de planes
│   │       ├── executor.py
│   │       ├── plan_node.py
│   │       ├── spatial_utils.py
│   │       ├── access/           # scans secuenciales, indexados y espaciales
│   │       ├── aggregate/        # agregación, ordenamiento y límite
│   │       ├── join/             # hash join, nested loop y sort-merge
│   │       └── processing/       # filtros, proyección y procesamiento
│   ├── spatial/                  # geometría y cálculos de distancia
│   │   ├── distance.py
│   │   └── geometry.py
│   ├── storage/                  # páginas, archivos y buffer pool
│   │   ├── buffer_manager.py
│   │   ├── file_manager.py
│   │   ├── latch.py
│   │   ├── page.py
│   │   ├── heap/                 # almacenamiento heap y codificación de filas
│   │   └── sequential/           # almacenamiento secuencial
│   └── transaction/              # transacciones, locks, WAL y recuperación
│       ├── demo.py
│       ├── demo_recovery.py
│       ├── lock_manager.py
│       ├── log_manager.py
│       ├── recovery.py
│       ├── row_undo.py
│       └── transaction_manager.py
│
├── tests/
│   ├── conftest.py
│   ├── catalog/                  # persistencia del catálogo e índices
│   ├── index/                    # B-tree, hash extensible y R-tree
│   ├── loader/                   # carga de CSV
│   ├── query/                    # parser, ejecución, API y consultas espaciales
│   ├── spatial/                  # geometría y distancias
│   ├── storage/                  # gestión de espacio en heap
│   └── transaction/              # locks, WAL, recovery y concurrencia
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml                # metadatos Python y configuración de pytest
├── pytest.ini                    # configuración adicional de pytest
├── requirements.txt              # dependencias Python del proyecto
└── uv.lock                       # versiones bloqueadas por uv
```

## Responsabilidades por área

### Motor (`src/`)

El motor reúne el catálogo, los tipos de datos, el almacenamiento persistente, los índices y el procesamiento SQL. Una consulta atraviesa el parser, las reglas de reescritura, el planificador y el ejecutor. El ejecutor compone operadores de acceso, procesamiento, joins y agregación.

### Datos espaciales

La funcionalidad espacial se reparte entre `src/spatial/` (geometría y distancias), `src/index/rtree/` (índice R-tree), los operadores espaciales de `src/query/executor/` y el panel `SpatialMapPanel.tsx` de la interfaz.

### API y frontend

`api/` expone el motor mediante FastAPI. `frontend/` contiene la aplicación web, los paneles de tablas, consultas, resultados y plan de ejecución, además de la importación CSV y la vista de mapa.

### Benchmarks y recursos SQL

`benchmarks/` contiene scripts de medición, un entorno para contrastar resultados con PostgreSQL/PostGIS y resultados exportados como CSV, JSON, Markdown e imágenes. `sql/` separa los ejemplos escritos a mano, los archivos generados y los scripts generadores.

### Pruebas

`tests/` organiza las pruebas según los subsistemas del motor: catálogo, índices, importación, consultas, geometría, almacenamiento y transacciones.

## Flujo general de una consulta

1. La aplicación web envía la solicitud a la API.
2. La API entrega la sentencia al motor de consultas.
3. El parser genera la representación sintáctica; el rewriter aplica reglas y el planner produce el plan.
4. El executor realiza las lecturas y operaciones necesarias usando el catálogo, el almacenamiento y los índices.
5. La API serializa el resultado y la interfaz lo presenta; para consultas espaciales también puede mostrarlo en el mapa.

## Archivos de configuración principales

- `pyproject.toml` y `uv.lock`: configuración y dependencias del proyecto Python.
- `requirements.txt` y `api/requirements.txt`: dependencias Python para el proyecto y la API.
- `pytest.ini`: ajustes del runner de pruebas.
- `frontend/package.json` y `frontend/package-lock.json`: dependencias y scripts del frontend.
- `frontend/.env.example`: plantilla de variables de entorno del cliente web.
