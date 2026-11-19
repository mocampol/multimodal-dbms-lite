# Detección de copyright en audio

Aplicación que busca, dentro de una canción subida, fragmentos parecidos a canciones de
una base de referencia (issues #110 a #115). Reporta **posibles coincidencias**, no una
determinación legal.

Por ahora contiene la preparación de esa base de referencia (#110), la extracción de un
vector MFCC por segmento de cada canción (#111), su indexación con HNSW e IVF para
buscar candidatos similares (#112), la API que agrupa esos candidatos en coincidencias
(#113) y la interfaz web que las muestra en una línea de tiempo (#114). Qué audio se usa
y por qué:
[`docs/audio_data_policy.md`](../../docs/audio_data_policy.md).

## Corpus de referencia

1,000 clips de 30 segundos de `fma_small` (Free Music Archive), 125 por cada uno de
sus 8 géneros, todos con licencias Creative Commons que permiten obras derivadas.

### Con Docker

```bash
docker compose up -d api
docker compose --profile audio run --rm audio-data
```

El servicio `audio-data` descarga los clips en el volumen `db-data`, el mismo de la
base de datos, y carga las canciones y los artistas por la API. La primera vez
descarga ~1 GB en unos 6 minutos, más lo que tarde construir la imagen (12 minutos en
total en la primera prueba). Después es idempotente: si el corpus y las tablas ya
están, termina en unos 15 segundos sin descargar nada, y reiniciar los contenedores no
borra nada.

### Sin Docker

Con la API corriendo en `http://127.0.0.1:8000` desde la raíz del repositorio:

```bash
python apps/audio-copyright/dataset/prepare_corpus.py      # data/audio/fma/
python apps/audio-copyright/dataset/load_songs.py
python apps/audio-copyright/features/extract_segments.py
```

Para probar el flujo con 16 clips: `prepare_corpus.py --per-genre 2`.

### Dónde queda cada cosa

| Qué | Dónde (en el contenedor) |
|---|---|
| Clips MP3, todos en la misma carpeta | `/app/data/audio/fma/<song_id>.mp3` |
| Manifiesto con los metadatos de cada clip | `/app/data/audio/fma/songs.csv` |
| Resumen por género y licencia | `/app/data/audio/fma/corpus_summary.json` |
| Tabla de canciones | `audio_songs` en el DBMS |
| Tabla de artistas | `audio_artists` en el DBMS |
| Codebook de los segmentos | `/app/data/audio/fma/segment_codebook.pkl` |
| Tabla de segmentos, con índices HNSW e IVF | `audio_segments` en el DBMS |

Fuera de Docker, las mismas rutas cuelgan de `data/` en la raíz del repositorio, que
Git ignora.

### Tablas

`audio_songs`: `song_id` (el `track_id` de FMA), `title`, `artist_id`, `album_title`,
`genre`, `duration_s`, `license_family`, `license_title`, `license_url`, `track_url`,
`source`, `path` y `sha256`. `path` es la ruta absoluta con la que la API abre el
archivo.

`audio_artists`: `artist_id`, `name` y `artist_url`.

La carga inserta solo las filas que faltan y borra las canciones que ya no están en el
manifiesto, junto con los artistas que se quedan sin canciones.

### Retirar una canción

Agregar su `song_id` a [`dataset/excluded_tracks.txt`](dataset/excluded_tracks.txt) y
volver a correr `audio-data`: se borra del corpus y de las tablas, y su lugar lo ocupa
la siguiente pista de su género. Las demás canciones no cambian.

La descarga también descarta los clips dañados, los de menos de 25 s y los que están en
silencio, y los reemplaza de la misma forma.

Si cambia la regla de selección, el atajo de idempotencia no lo nota: solo comprueba que
el corpus esté completo e intacto. Para recalcular la selección, reutilizando los archivos
que sigan sirviendo:

```bash
docker compose --profile audio run --rm audio-data sh -c "python apps/audio-copyright/dataset/prepare_corpus.py --refresh && python apps/audio-copyright/dataset/load_songs.py && python apps/audio-copyright/features/extract_segments.py"
```

### Segmentos MFCC

Cada canción se divide en segmentos de 5 segundos que avanzan cada 2.5 segundos: un
clip de 30 s da 11. Cada segmento se convierte en un vector con el mismo pipeline de
la Parte 4 (`src/index/multimedia`): MFCC por trama, K-Means con 64 palabras de audio
e histograma con TF-IDF y norma L2. Los audios que se suban para detectar coincidencias
deben pasar por el mismo codebook, que queda en `data/audio/fma/segment_codebook.pkl`.

`audio_segments`: `segment_id` (`song_id * 1000 + segment_index`), `song_id`,
`segment_index`, `start_s`, `end_s`, `embedding` (BYTEA, float32), `embedding_dim` y
`model_name`.

La extracción solo procesa las canciones que faltan y borra los segmentos de las
canciones retiradas. Si cambia el codebook (`--k`, `--window-s` o `--hop-s`), recalcula
todos los vectores.

### Índices y búsqueda de candidatos

Al terminar, `extract_segments.py` crea dos índices sobre `audio_segments.embedding`, si
faltan. Los dos usan coseno, porque los vectores tienen norma 1, y sus parámetros salen del
benchmark del #85 ([`vector_summary.md`](../../benchmarks/results/vector_summary.md)):

| Índice | Nombre | Parámetros |
|---|---|---|
| HNSW | `idx_audio_segments_hnsw` | `M` = 16, `ef_construction` = 200, `ef_search` = 100 |
| IVF | `idx_audio_segments_ivf` | `n_lists` = √N (unas 105 listas con 11,000 segmentos), `n_probe` = 8 |

Se crean después de insertar los segmentos, para construirlos una sola vez: si existieran
durante la carga, el motor los guardaría en disco con cada fila. Después, el motor los
mantiene solo en cada `INSERT` y `DELETE`.

`backend/segment_search.py` divide un audio en segmentos con el mismo codebook y, para cada
uno, busca los `k` segmentos más parecidos con `POST /tables/audio_segments/vector-search`.
Cada candidato trae la canción (título, artista, licencia y URL), su intervalo en la canción
original y su score, que con coseno es la similitud. Con la API corriendo:

```bash
python apps/audio-copyright/backend/segment_search.py remix.mp3 --k 5 --index hnsw
```

```json
{
  "duration_s": 40.0, "index_type": "hnsw", "metric": "cosine", "k": 5, "used_index": true,
  "segments": [
    {"segment_index": 0, "start_s": 0.0, "end_s": 5.0, "candidates": [
      {"song_id": 2, "title": "...", "artist": "...", "start_s": 10.0, "end_s": 15.0, "score": 0.93, ...}
    ]}
  ],
  "timings_ms": {"extraction_ms": ..., "dbms_search_ms": ..., "total_ms": ...}
}
```

Cada candidato tiene dos referencias temporales: el intervalo del audio buscado y el de la
canción original. Una copia real avanza igual en ambas, así que esos pares sirven para unir
segmentos seguidos en un rango y reportar el porcentaje de similitud, que corresponde a la
detección (#113).

Los valores por defecto se cambian con variables de entorno: `AUDIO_INDEX_TYPE` (`hnsw` o
`ivf`), `AUDIO_METRIC`, `AUDIO_DEFAULT_K`, `AUDIO_HNSW_EF_SEARCH`, `AUDIO_IVF_N_PROBE`,
`AUDIO_CODEBOOK` y `DBMS_API_URL`.

### API de detección

`backend/detection_api.py` recibe una canción, busca los candidatos de cada segmento y los
agrupa en coincidencias (`backend/detection.py`). Se levanta con Docker después de correr
`audio-data`:

```bash
docker compose --profile audio up -d audio-api        # http://localhost:8001
curl --data-binary @remix_2024.mp3 "http://localhost:8001/detections?filename=remix_2024.mp3"
```

El audio va como cuerpo crudo de la petición (MP3, WAV, FLAC, OGG o M4A; de 5 s a 10 min y
hasta 25 MB). Parámetros opcionales: `k`, `index` (`hnsw` o `ivf`), `threshold` y
`min_segments`. `GET /health` indica si el codebook está cargado.

Cómo se agrupa:

1. De cada segmento se toman los candidatos con similitud ≥ umbral.
2. Dos segmentos seguidos se unen si apuntan a la misma canción y el desfase entre ambos
   audios se mantiene (con un margen de un paso, 2.5 s): una copia avanza igual en los dos.
   Se puede saltar un segmento débil.
3. Una cadena de al menos `min_segments` segmentos es una coincidencia. Como los segmentos
   se solapan, una copia también activa a los vecinos de la original; de las cadenas de una
   misma canción que se solapan queda la más fuerte.

```json
{
  "filename": "remix_2024.mp3",
  "verdict": "posible_coincidencia",
  "message": "Se encontró 1 posible coincidencia.",
  "disclaimer": "Son posibles coincidencias de audio según su similitud; no son una determinación legal ...",
  "duration_s": 29.977, "coverage_pct": 100.0,
  "matches": [{"match_id": 1, "song_id": 2, "title": "Food", "artist": "AWOL", "license_family": "CC BY-NC-SA",
               "query_start_s": 0.0, "query_end_s": 29.977, "reference_start_s": 0.0, "reference_end_s": 29.977,
               "similarity_pct": 100.0, "max_similarity_pct": 100.0, "segments": 11}],
  "timeline": [{"segment_index": 0, "start_s": 0.0, "end_s": 5.0, "best_song_id": 2, "best_similarity_pct": 100.0,
                "above_threshold": true, "match_id": 1}, "..."],
  "settings": {"threshold": 0.98, "min_segments": 4, "max_gap": 1, "offset_tolerance_s": 2.5},
  "search": {"index_type": "hnsw", "metric": "cosine", "k": 5, "used_index": true},
  "timings_ms": {"extraction_ms": 295.0, "dbms_search_ms": 10.5, "total_ms": 583.7}
}
```

Sin coincidencias, `verdict` es `sin_coincidencias`, `matches` viene vacío y `message` dice
qué umbral y cuántos segmentos seguidos hacían falta. `timeline` trae siempre el mejor
candidato de cada segmento, para dibujar la línea de tiempo (#114).

### Umbral

El umbral por defecto, **0.98 con al menos 4 segmentos seguidos** (12.5 s), lo eligió
[`evaluation/calibrate_threshold.py`](evaluation/calibrate_threshold.py) con el corpus real:
es el de mayor recall que no detecta coincidencias en más del 5% de canciones ajenas.
Resultados completos en
[`evaluation/results/threshold_calibration.json`](evaluation/results/threshold_calibration.json).

Se probaron 80 fragmentos de canciones del corpus y 80 de canciones de FMA que no están en
él, de 15 s y desplazados fuera de la grilla de segmentos, limpios y con ruido a 20 dB:

| Variante | Recall | Falsos positivos | Otra canción además de la correcta |
|---|---:|---:|---:|
| Limpio | 88.8% | 2.5% | 0.0% |
| Con ruido a 20 dB | 2.5% | 7.5% | 6.3% |

**Limitación:** los vectores de #111 (histograma de 64 palabras de MFCC por segmento)
separan poco. La mediana de similitud por segmento es 0.977 con la canción correcta, pero
0.934 con canciones que no tienen nada que ver. Por eso el umbral tiene que ser tan alto,
y con él un ruido leve basta para perder la coincidencia. Detecta copias casi exactas, no
remixes ni covers. Mejorar esto requiere otros vectores, no otro umbral; la validación
completa es la del #115.

Se cambia con `AUDIO_MATCH_THRESHOLD`, `AUDIO_MIN_SEGMENTS`, `AUDIO_MAX_GAP` y
`AUDIO_OFFSET_TOLERANCE_S`, o por petición con `threshold` y `min_segments`.

### Interfaz web

`frontend/` es una app de React que sube una canción a la API de detección y muestra el
resultado.

Con Docker, después de correr `audio-data`, se levanta todo junto:

```bash
docker compose --profile audio up -d api audio-api audio-frontend   # http://localhost:8082
```

El contenedor `audio-frontend` sirve la app con nginx y reenvía `/api/` a `audio-api`, así
que la interfaz y la API comparten el origen y no hace falta configurar nada más. El puerto
se cambia con `AUDIO_FRONTEND_PORT`.

Para desarrollar, con la API corriendo en `http://localhost:8001`:

```bash
cd apps/audio-copyright/frontend
npm install
npm run dev        # http://localhost:5175
```

Si la API está en otra dirección, se indica con `VITE_AUDIO_API_URL` (ver `.env.example`).

Qué muestra:

- **Veredicto:** "Posible coincidencia" o "Sin coincidencias", con el mensaje y el aviso
  legal que devuelve la API. Un aviso fijo arriba recuerda que son posibles coincidencias y
  no una determinación legal: el veredicto se marca en ámbar, nunca como infracción.
- **Línea de tiempo:** una fila con los tramos de cada coincidencia, cada uno con su color,
  y otra con la similitud del mejor candidato de cada segmento y la línea del umbral. La
  escala va de 80 % a 100 %, porque casi todas las similitudes caen entre 93 % y 100 %.
- **Detección:** cada coincidencia como "0:00–0:15 → 92 % similar a "Canción" de Artista",
  con su intervalo en la canción original, la licencia y el enlace de origen.
- **Reproductor:** al hacer clic en la línea de tiempo o en un intervalo, el audio salta a
  ese instante, para escuchar el tramo antes de sacar conclusiones.

En "Opciones avanzadas" se elige el índice (HNSW o IVF) y se puede cambiar el umbral y el
mínimo de segmentos seguidos de esa búsqueda.

### Archivos

| Archivo | Qué hace |
|---|---|
| `dataset/fma_corpus.py` | Licencias, lectura de los zips de FMA por rangos HTTP, selección y manifiesto |
| `dataset/prepare_corpus.py` | Descarga los clips que faltan y escribe el manifiesto y el resumen |
| `dataset/load_songs.py` | Crea y sincroniza `audio_songs` y `audio_artists` por la API |
| `dataset/excluded_tracks.txt` | Canciones retiradas del corpus |
| `dataset/corpus_summary.json` | Resumen del corpus completo, versionado |
| `features/segment_extractor.py` | Segmentos, MFCC, codebook y un vector por segmento con su intervalo |
| `features/extract_segments.py` | Extrae los segmentos del corpus y los guarda en `audio_segments` por la API |
| `features/segment_index.py` | Crea los índices HNSW e IVF de `audio_segments`, si faltan |
| `backend/segment_search.py` | Busca los candidatos de cada segmento de un audio, con sus scores e intervalos |
| `backend/detection.py` | Agrupa los candidatos en coincidencias y da el veredicto |
| `backend/detection_api.py` | API de detección: `POST /detections` y `GET /health` |
| `evaluation/calibrate_threshold.py` | Elige el umbral por defecto con el corpus real |
| `frontend/` | Interfaz web: subida, veredicto con aviso legal, línea de tiempo y coincidencias |
| `frontend/Dockerfile`, `frontend/nginx.conf` | Imagen de la interfaz para `audio-frontend`: compila la app y reenvía `/api/` a `audio-api` |
