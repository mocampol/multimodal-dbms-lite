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

## Nota

Este sistema no pretende ser un motor SQL completo comparable con PostgreSQL o MySQL; su objetivo principal es servir como laboratorio de aprendizaje para entender los principios internos de un sistema de gestión de datos.
