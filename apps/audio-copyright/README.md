# Detección de copyright en audio

Aplicación que busca, dentro de una canción subida, fragmentos parecidos a canciones de
una base de referencia (issues #110 a #115). Reporta **posibles coincidencias**, no una
determinación legal.

Por ahora contiene la preparación de esa base de referencia (#110) y la extracción de
un vector MFCC por segmento de cada canción (#111). Qué audio se usa y por qué:
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
