# Aplicación Independiente de Reconocimiento Facial (Parte 5 — Opción D)

Aplicación FastAPI independiente del explorador principal de base de datos.
Consume el motor `multimodal-dbms-lite` exclusivamente a través de su API REST
(`DBMS_API_URL`, por defecto `http://127.0.0.1:8000`).

## Características

- **Política de datos y consentimiento** ([`docs/face_data_policy.md`](../../docs/face_data_policy.md)):
  identificadores opacos, bloqueo sin consentimiento explícito para participantes
  y retiro completo (eliminación de muestras, embeddings y entradas en índices).
- **Detección y embeddings configurables**:
  - Por defecto reutiliza [`FaceEmbeddingExtractor`](../../src/index/multimedia/face_embedding_extractor.py)
    (OpenCV YuNet + SFace `128-d`).
  - Opcionalmente soporta `facenet-pytorch` (MTCNN + InceptionResnetV1 `512-d`) y
    proyección ortogonal a `dim` ajustable.
- **Búsqueda vectorial e identidades en el DBMS**:
  persiste embeddings en `face_samples` e indexa en `HNSW` / `IVF` con `dim`, `k`
  y parámetros de búsqueda dinámicos.
- **Resultado `"desconocido"`**:
  si ninguna muestra supera el umbral `threshold` (o la base está vacía),
  responde explícitamente `registered: false`, `status: "unknown"`, `label: "desconocido"`.

## Con Docker

La aplicación tiene una instancia DBMS aislada (`face-db`) y persiste sus tablas e
índices en el volumen `face-db-data`; no comparte datos con el explorador principal
ni con la aplicación de audio. Descarga primero los modelos YuNet/SFace siguiendo
[`docs/face_embeddings.md`](../../docs/face_embeddings.md). Desde la raíz del repositorio, para levantar solo la aplicación facial:

```bash
docker compose up --build face-frontend
```

La interfaz queda en <http://localhost:8081>, la API facial en
<http://localhost:8002> y la API DBMS exclusiva de esta aplicación en
<http://localhost:8003>. Para levantar las tres demos, incluyendo la interfaz y la
base del DBMS principal y la aplicación de audio, usa `docker compose up --build`
como se explica en [`README.md`](../../README.md#arranque-del-stack-completo).

## Ejecución local

1. Levantar primero el DBMS principal (puerto `8000`) desde la raíz del repositorio:

   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```

2. Levantar la app de reconocimiento facial (puerto `8001`):

   ```bash
   PYTHONPATH=src:apps/face-recognition uvicorn backend.main:app --host 127.0.0.1 --port 8001
   ```

3. Levantar el frontend independiente (puerto `5174`):

   ```bash
   cd apps/face-recognition/frontend
   npm install
   npm run dev
   ```

   Para probar desde un celular en la misma red local, ejecuta `npm run dev -- --host` y accede mediante `https://` (la API `getUserMedia` requiere origen seguro en dispositivos remotos; en `http://` desde móvil puedes usar el botón de subida/cámara nativa `<input type="file" capture>`).
