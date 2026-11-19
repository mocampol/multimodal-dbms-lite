# Multimodal DBMS Lite

Multimodal DBMS Lite es un prototipo académico de un gestor de bases de datos
relacional y multimodal. Implementa un motor SQL en Python, una API REST, una
interfaz web y dos aplicaciones independientes que consumen el motor: búsqueda
facial y detección de posibles coincidencias de audio.

El proyecto sirve como laboratorio para estudiar almacenamiento, catálogos,
índices, procesamiento de consultas, búsqueda multimodal, persistencia y
despliegue en contenedores. No es un sustituto de PostgreSQL, MySQL ni de un
sistema de reconocimiento biométrico de producción.

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

## Arquitectura y arquetipo del sistema

El arquetipo es un **motor de base de datos modular (monolito modular) detrás
de una API REST**, acompañado de clientes web independientes. No se conecta a
un servidor SQL externo en su ruta principal: el catálogo, los archivos de
tablas y los índices los administra el propio motor. PostgreSQL/PostGIS aparece
solo en benchmarks comparativos.

La ruta de una consulta del explorador principal es:

1. El usuario escribe SQL en la interfaz React.
2. Nginx entrega la aplicación web y reenvía `/api/` a FastAPI.
3. La API llama al motor Python, que valida, reescribe y planifica la consulta.
4. El ejecutor físico lee o modifica tablas e índices en el almacenamiento local.
5. La API serializa filas y planes; la interfaz muestra los resultados.

Las aplicaciones facial y de audio son servicios separados: no importan ni
modifican el catálogo directamente. Consumen cada instancia de DBMS por HTTP.
El stack Docker arranca tres instancias del motor, con datos aislados:
`api` para el explorador principal, `face-db` para reconocimiento facial y
`audio-db` para detección de audio.

```text
Navegador
  ├── UI principal :8080 ── /api/* ─────────────> api :8000 ──────> db-data
  ├── UI facial :8081 ── /face-api/* ──> face-api :8002 ─> face-db :8000 ─> face-db-data
  └── UI audio :8082 ── /api/* ────────> audio-api :8001 ─> audio-db :8000 ─> audio-db-data
```

Cada instancia (`api`, `face-db` y `audio-db`) ejecuta su propio catálogo,
motor SQL, archivos e índices. No se comparten tablas ni volúmenes entre las
tres demos.

En Docker, `db-data`, `face-db-data` y `audio-db-data` persisten por separado
`/app/data` de cada instancia. El corpus FMA y sus artefactos viven únicamente
en `audio-db-data`. La carpeta local `media/` se monta como solo lectura en
`/media` para importar archivos en la demo principal. Las imágenes faciales de
YuNet/SFace se descargan por separado y se montan desde
`benchmarks/corpus/face/models/`; no se guardan en Git.

## Organización del código fuente

El repositorio agrupa el código por responsabilidad y por aplicación:

```text
multimodal-dbms-lite/
├── api/                         # API DBMS: FastAPI, routers y serialización
├── apps/
│   ├── face-recognition/        # servicio y frontend de reconocimiento facial
│   └── audio-copyright/         # corpus, extracción, búsqueda y UI de audio
├── benchmarks/                  # mediciones, comparación PostgreSQL y resultados
│   ├── corpus/                  # datasets/modelos locales; ignorado por Git
│   ├── postgres/                # PostGIS y GIN para benchmarks
│   ├── results/                 # resultados y gráficas
│   └── workload/                # consultas/relevancia congeladas de texto
├── docs/                        # políticas de datos y documentación técnica
├── frontend/                    # UI principal (React + TypeScript + Vite)
├── media/                       # demo sintética e importaciones locales
├── sql/
│   ├── examples/                # SQL de ejemplo
│   ├── generated/               # salidas SQL generadas
│   └── scripts/                 # generadores y utilidades SQL
├── src/                         # motor DBMS modular
│   ├── catalog/                 # catálogo y metadatos
│   ├── common/                  # registros, esquemas y tipos
│   ├── index/                   # índices y extracción de características
│   ├── loader/                  # cargadores CSV y multimedia
│   ├── query/                   # parser, rewriter, planner y executor
│   ├── spatial/                 # geometría y distancias
│   ├── storage/                 # páginas, heap, secuencial y buffer
│   ├── transaction/             # WAL, locks y recuperación
│   └── main.py                  # inicialización del catálogo/motor
├── tests/                       # pruebas unitarias e integración
├── Dockerfile                   # imagen de la API DBMS
├── docker-compose.yml           # tres demos, cada una con su DBMS/volumen aislado
├── pyproject.toml / uv.lock     # metadatos y dependencias Python bloqueadas
├── requirements.txt             # dependencias de pruebas/benchmarks
└── README.md
```

### Componentes del motor

| Ruta | Responsabilidad |
|---|---|
| `src/catalog/` | Catálogo de tablas, columnas, metadatos e índices. |
| `src/common/` | Representación común de registros, esquemas y valores SQL/espaciales. |
| `src/storage/` | Páginas, administración de archivos y buffer; heap y almacenamiento secuencial. |
| `src/index/btree/`, `extendible_hash/`, `rtree/` | Índices B+, hash extensible y R-Tree persistente. |
| `src/index/text/` | Procesador de texto, índice invertido y SPIMI. |
| `src/index/vector/` | Índices k-NN IVF y HNSW y métricas de distancia. |
| `src/index/multimedia/` | Extracción de características SIFT/MFCC, BoW/TF-IDF, cuantización y embeddings faciales. |
| `src/loader/` | Importación de CSV y carpetas multimedia, con inferencia de tipos. |
| `src/query/parser/` | Scanner, parser SQL, AST y gramática. |
| `src/query/rewriter/`, `planner/`, `executor/` | Reescritura, creación de planes y operadores físicos (scans, joins, filtros, agregación y búsquedas). |
| `src/spatial/` | Tipos geométricos y cálculo de distancias. |
| `src/transaction/` | WAL, transacciones, locks, undo y recuperación. |
| `api/` | API HTTP para consultas, catálogo, importación de archivos y vectores. |
| `frontend/` | Explorador DBMS, editor SQL, resultados, plan y mapa espacial. |
| `apps/` | Aplicaciones desacopladas que usan la API DBMS. |
| `tests/` | Pruebas agrupadas por catálogo, índices, loaders, consultas, almacenamiento y transacciones. |
| `benchmarks/` | Workloads reproducibles y comparación de rendimiento; no forman parte del arranque normal. |

### Aplicaciones independientes

- **Reconocimiento facial (`apps/face-recognition/`)**: frontend y API propios
  para registrar identidades, gestionar consentimiento, enrolar muestras y
  buscar embeddings con IVF/HNSW en el DBMS. Usa YuNet + SFace por defecto
  (128 dimensiones); FaceNet es una alternativa opcional. No descarga ni carga
  automáticamente un dataset de rostros: las muestras se agregan en la app.
  La política del proyecto propone LFW, pero el corpus no forma parte del
  repositorio ni se importa durante el arranque.
- **Detección de copyright en audio (`apps/audio-copyright/`)**: preparación
  opcional de un corpus FMA Small, extracción MFCC por segmentos, índices
  vectoriales, API de detección y frontend con línea de tiempo. En Docker, su
  base y corpus están aislados en `audio-db-data`; el primer arranque prepara
  el corpus. El servicio no determina legalmente una infracción.
- **Demo multimodal (`media/`)**: generador de imágenes de patrones, sonidos y
  documentos sintéticos. No contiene rostros; sus datos pueden importarse en
  el explorador principal.

La estructura detallada de archivos está en
[`ESTRUCTURA_PROYECTO.md`](ESTRUCTURA_PROYECTO.md).

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
- R-Tree espacial
- SPIMI e índice invertido para texto
- índices vectoriales IVF y HNSW para búsquedas k-NN
- extracción de características para imágenes, audio y embeddings faciales

Estas estructuras permiten acelerar búsquedas y mejorar la ejecución de consultas.
IVF y HNSW indexan vectores de características generados a partir de rutas de
imagen/audio, embeddings faciales o vectores numéricos almacenados en la columna.

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

La API se encuentra en `api/`; entre sus rutas principales están:

- `GET /tables`: lista las tablas del catálogo
- `GET /tables/{table_name}`: detalles de una tabla y sus índices
- `POST /query`: ejecuta una sentencia SQL
- `POST /import/csv` y `POST /import/media`: importa datos
- `POST /tables/{table_name}/vector-search`: búsqueda k-NN por API
- `GET /files`: sirve archivos de carpetas previamente importadas
- `GET /health`: comprueba el estado del servicio

La API usa FastAPI y serializa resultados y planes para el frontend. La lista
de rutas y esquemas vigentes se consulta en <http://localhost:8000/docs> cuando
el servicio está levantado.

### 7. Frontend

La aplicación principal está en `frontend/` y usa React, TypeScript y Vite.
Permite:

- explorar archivos, tablas, columnas e índices;
- importar CSV o carpetas de medios;
- ejecutar SQL y revisar resultados y planes;
- realizar búsquedas espaciales y visualizar resultados en un mapa.

En desarrollo se comunica con la API mediante `VITE_API_BASE_URL`; detrás de
Nginx, `/api/` se reenvía al servicio `api`.

## Requisitos previos

Para ejecutar el DBMS principal localmente necesitas:

- Python 3.13+
- Node.js 22+
- npm
- entorno virtual de Python recomendado

Para ejecutar todas las aplicaciones también necesitas Docker Engine/Desktop
con Docker Compose v2, ffmpeg (solo para ciertos formatos de audio fuera de
Docker) y los modelos YuNet/SFace locales descritos en
[`docs/face_embeddings.md`](docs/face_embeddings.md). La aplicación de
reconocimiento facial no requiere un dataset precargado; las muestras se
registran desde la aplicación.

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

### 3. Instalar dependencias Python

Desde la raíz del proyecto:

```bash
python -m pip install --upgrade pip
python -m pip install -r api/requirements.txt -r requirements.txt
```

`api/requirements.txt` contiene las dependencias de la API DBMS; `requirements.txt`
añade herramientas de pruebas y benchmarks. Para usar la alternativa FaceNet
(opcional, más pesada que YuNet/SFace), instala además:

```bash
python -m pip install -r apps/face-recognition/requirements.txt
```

### 4. Instalar dependencias de cada frontend

Instala cada frontend en su propia carpeta:

```bash
cd frontend && npm ci
cd ../apps/face-recognition/frontend && npm ci
cd ../../audio-copyright/frontend && npm ci
```

## Ejecución

### DBMS principal: API y frontend

Terminal 1, desde la raíz:

```bash
source .venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

La UI queda en <http://localhost:5173>, la API en
<http://localhost:8000> y su documentación interactiva en
<http://localhost:8000/docs>. El frontend principal usa por defecto la API
local en el puerto 8000; `VITE_API_BASE_URL` permite cambiar esa dirección.

### Aplicación facial local (opcional)

Descarga primero los dos modelos locales siguiendo
[`docs/face_embeddings.md`](docs/face_embeddings.md). Luego inicia el backend
en el puerto 8002:

```bash
PYTHONPATH=src:apps/face-recognition uvicorn backend.main:app \
  --app-dir apps/face-recognition --host 127.0.0.1 --port 8002
```

En otra terminal, inicia su interfaz:

```bash
cd apps/face-recognition/frontend
FACE_API_TARGET=http://127.0.0.1:8002 npm run dev -- --host 0.0.0.0
```

La UI queda en <http://localhost:5174> y la API en
<http://localhost:8002>. Se pueden registrar participantes después de obtener
su consentimiento explícito; LFW no se descarga ni se carga automáticamente.
Consulta [`docs/face_data_policy.md`](docs/face_data_policy.md) antes de usar
imágenes de personas.

### Aplicación de audio local (opcional)

Con la API principal ejecutándose en el puerto 8000, prepara el corpus una vez:

```bash
python apps/audio-copyright/dataset/prepare_corpus.py
python apps/audio-copyright/dataset/load_songs.py
python apps/audio-copyright/features/extract_segments.py
```

Después inicia la API de audio y su frontend:

```bash
uvicorn detection_api:app --app-dir apps/audio-copyright/backend \
  --host 127.0.0.1 --port 8001
```

```bash
cd apps/audio-copyright/frontend
npm run dev -- --host 0.0.0.0
```

La UI queda en <http://localhost:5175>; la API de audio, en
<http://localhost:8001>. El corpus contiene audio de FMA Small; revisa
[`docs/audio_data_policy.md`](docs/audio_data_policy.md) y el manual de
[`apps/audio-copyright/`](apps/audio-copyright/) para las condiciones, tamaño
del corpus y opciones de prueba. En modo local, la preparación y extracción
pueden tardar y requieren conexión a Internet.

## Dockerización: levantar el stack

El stack levanta las **tres demos** en una sola red: explorador DBMS, aplicación
facial y aplicación de audio. Cada una tiene una instancia independiente del
motor DBMS y un volumen propio; ninguna usa la base de otra. No hay un
contenedor PostgreSQL en el stack principal.

### Requisitos y modelos faciales

Requiere Docker Desktop (o Docker Engine con el plugin de Docker Compose v2).
La app facial usa YuNet y SFace por defecto. Sus modelos no están en Git: antes
de `docker compose up`, descárgalos al directorio local esperado y verifica sus
hashes:

```bash
mkdir -p benchmarks/corpus/face/models
curl --fail --location --output benchmarks/corpus/face/models/face_detection_yunet_2026may.onnx \
  https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2026may.onnx
curl --fail --location --output benchmarks/corpus/face/models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
shasum -a 256 benchmarks/corpus/face/models/*.onnx
```

Los valores SHA-256 esperados y las licencias están en
[`docs/face_embeddings.md`](docs/face_embeddings.md). No subas esos modelos,
imágenes faciales, embeddings ni índices al repositorio.

### Arranque del stack completo

Desde la raíz, después de descargar los modelos faciales:

```bash
docker compose up --build
```

Compose construye y levanta los tres frontends, las tres instancias DBMS y las
dos APIs de aplicación. El cargador `audio-data` espera a que `audio-db` esté
saludable, prepara el corpus en su volumen y solo entonces permite iniciar la
API de audio. La primera corrida descarga alrededor de 1 GB de audio, además de
construir las imágenes, por lo que puede tardar varios minutos. La preparación
del corpus es idempotente.

Direcciones predeterminadas:

| Demo | Frontend | API de aplicación | API DBMS exclusiva |
|---|---|---|---|
| Explorador DBMS | <http://localhost:8080> | — | <http://localhost:8000> (`api`) |
| Reconocimiento facial | <http://localhost:8081> | <http://localhost:8002> (`face-api`) | <http://localhost:8003> (`face-db`) |
| Detección de audio | <http://localhost:8082> | <http://localhost:8001> (`audio-api`) | <http://localhost:8004> (`audio-db`) |

Las APIs DBMS faciales y de audio están publicadas para inspección, pero cada
aplicación se conecta por la red interna a `face-db:8000` o `audio-db:8000`.
La aplicación principal usa solo `api:8000`.

Para levantar únicamente el DBMS principal, sin la aplicación facial:

```bash
docker compose up --build api frontend
```

Para apagar los servicios:

```bash
docker compose down
```

Los datos se conservan en tres volúmenes independientes: `db-data`,
`face-db-data` y `audio-db-data`. **No** uses `docker compose down -v` salvo
que quieras borrar permanentemente las tres bases y el corpus de audio. Los
modelos faciales se montan desde el host; la carpeta `media/` también se monta
como solo lectura para importar imágenes, audio o texto en la demo principal.

Si ya habías ejecutado una versión anterior del Compose, su volumen `db-data`
puede contener datos de audio antiguos. La nueva configuración no los comparte:
el corpus se vuelve a preparar dentro de `audio-db-data`; los datos existentes
se conservan en el volumen viejo/main y no se copian automáticamente.

Para importar archivos multimedia en el explorador, colócalos en una
subcarpeta de `media/` en el host, por ejemplo `media/fotos`, y en el panel
**Importar carpeta** indica `/media/fotos`. Los archivos originales no se
copian a la base y permanecen en el host. La app facial gestiona sus propias
muestras por sus endpoints; no usa esta carpeta de importación.

### Puertos configurables

Los puertos del host se pueden cambiar con variables de entorno:

```bash
FRONTEND_PORT=8083 API_PORT=8010 \
FACE_FRONTEND_PORT=8084 FACE_API_PORT=8012 \
AUDIO_FRONTEND_PORT=8085 AUDIO_API_PORT=8011 \
FACE_DB_PORT=8013 AUDIO_DB_PORT=8014 \
docker compose up --build
```

En Windows PowerShell:

```powershell
$env:FRONTEND_PORT="8083"; $env:API_PORT="8010"
$env:FACE_FRONTEND_PORT="8084"; $env:FACE_API_PORT="8012"
$env:AUDIO_FRONTEND_PORT="8085"; $env:AUDIO_API_PORT="8011"
$env:FACE_DB_PORT="8013"; $env:AUDIO_DB_PORT="8014"
docker compose up --build
```

Dentro de la red Docker, los servicios siempre se comunican por sus nombres y
puertos internos (`api:8000`, `face-db:8000`, `audio-db:8000`,
`face-api:8002`, `audio-api:8001`); cambiar los puertos publicados del host no
cambia esos destinos.

## Verificación

Se incluye una suite de pruebas con pytest. Para ejecutarlas:

```bash
source .venv/bin/activate
pytest -q
```

## Búsqueda multimedia por similitud

`SIMILAR_TO` busca los `k` registros más parecidos al objeto indicado. La columna
consultada debe contener rutas de archivos multimedia o vectores numéricos, y
debe tener un índice IVF o HNSW creado sobre datos existentes. `USING` selecciona
el índice para esa consulta; `WITH METRIC` acepta `cosine`, `euclidean` o
`dot_product` (por defecto, `cosine`). `SIMILARITY_SCORE()` agrega el puntaje al
resultado y se puede ordenar con su alias.

```sql
CREATE TABLE fotos (
    id INTEGER PRIMARY KEY,
    nombre VARCHAR(100),
    ruta VARCHAR(255)
);

INSERT INTO fotos (id, nombre, ruta) VALUES
    (1, 'Costa', '/datos/fotos/costa.jpg'),
    (2, 'Montaña', '/datos/fotos/montana.jpg');

CREATE INDEX idx_fotos_ruta ON fotos USING HNSW (ruta);

SELECT nombre, SIMILARITY_SCORE() AS score
FROM fotos
WHERE SIMILAR_TO(ruta, '/datos/consultas/playa.jpg', k=10)
      USING HNSW WITH METRIC=cosine
ORDER BY score DESC;
```

Para audio se utiliza la misma sintaxis y se puede elegir IVF con distancia
euclidiana:

```sql
CREATE TABLE canciones (
    id INTEGER PRIMARY KEY,
    nombre VARCHAR(100),
    artista VARCHAR(100),
    archivo VARCHAR(255)
);

INSERT INTO canciones (id, nombre, artista, archivo) VALUES
    (1, 'Tema uno', 'Artista A', '/datos/audio/tema1.mp3'),
    (2, 'Tema dos', 'Artista B', '/datos/audio/tema2.mp3');

CREATE INDEX idx_canciones_archivo ON canciones USING IVF (archivo);

SELECT nombre, artista, SIMILARITY_SCORE() AS score
FROM canciones
WHERE SIMILAR_TO(archivo, '/datos/consultas/audio_query.mp3', k=5)
      USING IVF WITH METRIC=euclidean
ORDER BY score DESC;
```

Reemplaza las rutas de ejemplo por archivos accesibles desde el proceso del
motor. El índice se construye al ejecutar `CREATE INDEX`, por lo que la columna
debe contener al menos un objeto válido de la modalidad correspondiente.
`k` debe ser un entero positivo y los tipos de índice y métrica deben ser
`IVF`/`HNSW` y `cosine`/`euclidean`/`dot_product`, respectivamente.

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

| Script                 | Compara                                                               | Salidas                                                                                     |
| ---------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `benchmark_indexes.py` | B+ clusterizado vs. B+ no clusterizado vs. Hash dinámico              | `results/index_comparison_*.csv/md`, `results/charts/`                                      |
| `benchmark_spatial.py` | Búsqueda secuencial vs. R-Tree propio vs. PostgreSQL + PostGIS (GiST) | `results/spatial_results.csv`, `results/spatial_summary.md`, `results/charts/spatial_*.png` |
| `benchmark_text_comparison.py` | SPIMI + BM25 vs. SPIMI + TF-IDF vs. PostgreSQL GIN + `ts_rank`        | `results/text_comparison_results.csv`, `results/text_comparison_environment.json`           |


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

### Benchmark de texto: SPIMI vs. PostgreSQL GIN (`benchmark_text_comparison.py`)

Compara SPIMI + BM25, SPIMI + TF-IDF (coseno) y PostgreSQL tsvector + GIN con `ts_rank` sobre los mismos documentos (N = 1 000 / 10 000 / 100 000), las mismas consultas (1, 3 y 5+ términos) y el mismo k = 10. Mide tiempo de construcción, tamaño del índice, tiempo de consulta, Precision@10, Recall@10 y memoria, y verifica que todos los motores usen los mismos documentos relevantes.

```bash
pip install -r requirements.txt -r benchmarks/postgres/requirements.txt
cd benchmarks/postgres && docker compose up -d && cd ../..

python benchmarks/postgres/load_text_data.py       # PostgreSQL GIN
python benchmarks/benchmark_text_comparison.py     # SPIMI + BM25 / TF-IDF
python benchmarks/compare_text_results.py          # verificación y resumen
python benchmarks/plot_text_results.py             # gráficas
```

Todos los scripts de medición y verificación aceptan `--quick` (N = 1 000). El dataset, la metodología (incluida la medición de memoria) y el formato del CSV están en [`benchmarks/README.md`](benchmarks/README.md#comparación-de-texto-spimi-vs-postgresql-gin). Los resultados, las gráficas y la comparación entre GIN y `ts_rank` están en [`text_comparison_summary.md`](benchmarks/results/text_comparison_summary.md).

## Nota

Este sistema no pretende ser un motor SQL completo comparable con PostgreSQL o MySQL; su objetivo principal es servir como laboratorio de aprendizaje para entender los principios internos de un sistema de gestión de datos.
