# Estructura actual del proyecto

**Proyecto:** Multimodal DBMS Lite  
**Actualizado:** Noviembre de 2026

Este documento resume la organización actual del repositorio. Se describen los módulos de código, las pruebas y los recursos auxiliares. Se omiten archivos generados de caché como `__pycache__` y `.pytest_cache`.

## Vista general

```text
multimodal-dbms-lite/
├── api/                 # API HTTP con FastAPI y streaming multimedia
├── benchmarks/          # mediciones, comparaciones y sus resultados
├── frontend/            # cliente web interactivo con React, TypeScript y Vite
├── sql/                 # ejemplos y generación de scripts SQL
├── src/                 # implementación del motor DBMS multimodal
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
│   ├── explain.py                # presentación y serialización de planes de ejecución
│   ├── main.py                   # aplicación FastAPI y configuración CORS
│   ├── requirements.txt          # dependencias específicas de la API
│   ├── schemas.py                # modelos Pydantic de entrada y salida
│   ├── serialize.py              # serialización de resultados para el frontend
│   └── routers/
│       ├── __init__.py
│       ├── imports.py            # rutas de importación de CSV
│       ├── media.py              # importación de carpetas multimedia y streaming (GET /files)
│       ├── query.py              # ejecución de consultas SQL
│       └── tables.py             # acceso a tablas y metadatos del catálogo
│
├── benchmarks/
│   ├── README.md                 # guía para ejecutar los benchmarks
│   ├── benchmark_btree.py
│   ├── benchmark_indexes.py
│   ├── benchmark_spatial.py
│   ├── benchmark_text.py
│   ├── benchmark_text_comparison.py # SPIMI vs PostgreSQL GIN
│   ├── benchmark_vector.py       # IVF vs HNSW sobre imágenes y audio
│   ├── benchmark_vector_sql.py   # costo de feature extraction en tiempo de consulta SQL
│   ├── compare_text_results.py   # validación de consistencia de resultados de texto
│   ├── compare_vector_results.py # validación de recall y consistencia k-NN
│   ├── perf_heap_vs_sequential.py
│   ├── plot_text_results.py      # generación de gráficos de texto
│   ├── plot_vector_results.py    # generación de gráficos vectoriales
│   ├── plot_vector_gallery.py    # galería visual de aciertos y errores
│   ├── prepare_vector_data.py    # descarga y preparación de Tiny ImageNet y Speech Commands
│   ├── spatial_common.py         # utilidades compartidas para pruebas espaciales
│   ├── text_common.py            # utilidades compartidas para pruebas de texto
│   ├── vector_common.py          # utilidades compartidas para pruebas vectoriales
│   ├── postgres/                 # entorno de comparación PostgreSQL (PostGIS y GIN)
│   │   ├── docker-compose.yml
│   │   ├── load_data.py
│   │   ├── load_text_data.py
│   │   ├── queries.sql
│   │   ├── requirements.txt
│   │   ├── schema.sql
│   │   ├── text_queries.sql
│   │   └── text_schema.sql
│   └── results/                  # salidas guardadas de las mediciones
│       ├── charts/               # gráficos generados (.png)
│       ├── heap_vs_sequential_results_*.csv
│       ├── spatial_comparison_results.csv
│       ├── spatial_summary.md
│       ├── text_comparison_results.csv
│       ├── text_comparison_summary.md
│       ├── vector_comparison_results.csv
│       └── vector_summary.md
│
├── frontend/
│   ├── .env.example              # ejemplo de configuración de la API
│   ├── .gitignore
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx               # composición principal de la interfaz y layout
│       ├── CsvImport.tsx         # importación de CSV con inferencia de tipos
│       ├── MediaImport.tsx       # importación masiva de carpetas multimedia
│       ├── ExecutionPlanPanel.tsx# visualización del plan de ejecución Volcano
│       ├── FilesPanel.tsx        # exploración de tablas, columnas e índices
│       ├── Panel.tsx             # componente contenedor con memoria de estado
│       ├── QueryPanel.tsx        # editor interactivo de consultas SQL
│       ├── ResultsPanel.tsx      # tabla de resultados con miniaturas, audio y "Buscar similares"
│       ├── SpatialMapPanel.tsx   # visualización espacial en mapa interactivo Leaflet
│       ├── api.ts                # cliente de comunicación con la API FastAPI
│       ├── index.css             # estilos del sistema
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
├── src/                          # motor de base de datos multimodal
│   ├── main.py                   # arranque, registro del catálogo y startup recovery
│   ├── catalog/                  # catálogo y metadatos persistidos en Heap
│   │   ├── catalog.py
│   │   ├── column.py
│   │   ├── exceptions.py
│   │   └── table_metadata.py
│   ├── common/                   # tipos, esquemas y registros
│   │   ├── record.py
│   │   ├── schema.py
│   │   └── value.py
│   ├── index/                    # estructuras de indexación
│   │   ├── btree/                # árbol B+ (clustered y non-clustered)
│   │   │   ├── btree.py
│   │   │   ├── clustered_index.py
│   │   │   ├── exceptions.py
│   │   │   ├── key_codec.py
│   │   │   └── node.py
│   │   ├── extendible_hash/      # hashing dinámico extensible
│   │   │   ├── extendible_hash_index.py
│   │   │   ├── hash_bucket_page.py
│   │   │   └── hash_directory_page.py
│   │   ├── rtree/                # índice espacial R-tree persistente
│   │   │   ├── exceptions.py
│   │   │   ├── node.py
│   │   │   ├── page.py
│   │   │   └── rtree.py
│   │   ├── text/                 # índices de texto invertido y SPIMI
│   │   │   ├── inverted_index.py
│   │   │   ├── spimi.py
│   │   │   ├── text_processor.py
│   │   │   └── stopwords/
│   │   ├── multimedia/           # extracción y cuantización de características
│   │   │   ├── bow_tfidf.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── mfcc_extractor.py
│   │   │   ├── quantizer.py
│   │   │   └── sift_extractor.py
│   │   └── vector/               # índices vectoriales k-NN de alta dimensionalidad
│   │       ├── base_vector_index.py
│   │       ├── hnsw_index.py
│   │       ├── ivf_index.py
│   │       └── metric.py
│   ├── loader/                   # lectura de CSV, medios e inferencia de tipos
│   │   ├── csv_loader.py
│   │   ├── csv_reader.py
│   │   ├── exceptions.py
│   │   ├── media_loader.py
│   │   └── type_inference.py
│   ├── query/                    # procesamiento de consultas SQL
│   │   ├── query_engine.py
│   │   ├── parser/               # scanner, parser, AST y gramática
│   │   │   ├── GRAMMAR.md
│   │   │   ├── ast_nodes.py
│   │   │   ├── exceptions.py
│   │   │   ├── parser.py
│   │   │   ├── scanner.py
│   │   │   ├── token_.py
│   │   │   └── visitor.py
│   │   ├── planner/              # construcción de planes físicos
│   │   │   └── plan_builder.py
│   │   ├── rewriter/             # reescrituras y optimizaciones
│   │   │   ├── rewriter.py
│   │   │   └── rules/
│   │   │       ├── predicate_pushdown.py
│   │   │       └── simplify_expressions.py
│   │   └── executor/             # ejecución física de planes (modelo Volcano)
│   │       ├── executor.py
│   │       ├── plan_node.py
│   │       ├── spatial_utils.py
│   │       ├── access/           # scans secuenciales, indexados, espaciales y de texto
│   │       │   ├── bitmap_scan.py
│   │       │   ├── clustered_scan.py
│   │       │   ├── index_scan.py
│   │       │   ├── seq_scan.py
│   │       │   ├── spatial_scan.py
│   │       │   └── text_index_scan.py
│   │       ├── aggregate/        # agregación y agrupamiento
│   │       │   ├── group_aggregate.py
│   │       │   └── hash_aggregate.py
│   │       ├── join/             # hash join, nested loop y sort-merge
│   │       │   ├── hash_join.py
│   │       │   ├── nested_loop_join.py
│   │       │   └── sort_merge_join.py
│   │       └── processing/       # filtros, proyección, ordenamiento, texto y vectores
│   │           ├── filter.py
│   │           ├── limit.py
│   │           ├── projection.py
│   │           ├── sort.py
│   │           ├── spatial_filter.py
│   │           ├── text_search.py
│   │           └── vector_search.py
│   ├── spatial/                  # geometría y cálculos de distancia
│   │   ├── distance.py
│   │   └── geometry.py
│   ├── storage/                  # páginas, archivos y buffer pool
│   │   ├── buffer_manager.py
│   │   ├── file_manager.py
│   │   ├── latch.py
│   │   ├── page.py
│   │   ├── heap/                 # almacenamiento heap y slotted page
│   │   │   ├── heap_file.py
│   │   │   ├── record_codec.py
│   │   │   └── rid.py
│   │   └── sequential/           # almacenamiento secuencial y overflow
│   │       └── sequential_file.py
│   └── transaction/              # transacciones, locks, WAL y recuperación
│       ├── demo.py
│       ├── demo_recovery.py
│       ├── lock_manager.py
│       ├── log_manager.py
│       ├── recovery.py
│       ├── row_undo.py
│       └── transaction_manager.py
│
├── tests/                        # pruebas automatizadas con pytest
│   ├── conftest.py
│   ├── catalog/                  # persistencia del catálogo e índices
│   ├── index/                    # B-tree, hash extensible, R-tree, IVF y HNSW
│   ├── loader/                   # carga de CSV y carpetas de medios
│   ├── query/                    # parser, ejecución, API, texto y consultas espaciales
│   ├── search/                   # nodos de búsqueda textual
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
