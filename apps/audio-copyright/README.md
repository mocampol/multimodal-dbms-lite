# Detección de copyright en audio

Aplicación que busca, dentro de una canción subida, fragmentos parecidos a canciones de
una base de referencia (issues #110 a #115). Reporta **posibles coincidencias**, no una
determinación legal.

Por ahora contiene la preparación de esa base de referencia (#110), la extracción de un
vector MFCC por segmento de cada canción (#111) y su indexación con HNSW e IVF para
buscar candidatos similares (#112). Qué audio se usa y por qué:
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
la siguiente pista de su género.

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
