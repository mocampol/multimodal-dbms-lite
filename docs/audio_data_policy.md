# Política de datos para la detección de copyright en audio

## Alcance y decisión

Esta política define qué audio puede usar la aplicación de detección de copyright
(issues #110 a #115): con qué canciones se arma la base de referencia, bajo qué
licencias y qué metadatos se guardan de cada una. La aplicación reporta **posibles
coincidencias** entre fragmentos de audio; no determina si hubo una infracción.

- **Dataset elegido: [FMA, Free Music Archive](https://github.com/mdeff/fma),
  subconjunto `fma_small`.** Tiene 8,000 clips de 30 segundos en MP3, balanceados en
  8 géneros, y metadatos por pista con título, artista, álbum, género, licencia y URL
  de origen. Se usan **1,000 clips**, 125 por género, todos con licencias aceptadas
  (sección siguiente).
- **Licencias del dataset**, según su README:
  - Metadatos: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Exigen
    atribución: citar el trabajo de Defferrard et al. (sección "Atribución").
  - Audio: los autores del dataset no tienen su copyright y lo distribuyen *"under
    the license chosen by the artist"*, es decir, cada pista con la licencia Creative
    Commons que eligió su artista. Por eso se revisa pista por pista.
  - Código del dataset: MIT. No se usa.
- **Limitación conocida:** los clips duran 30 segundos, no la canción completa. Las
  canciones completas están en `fma_full`, que pesa 879 GiB y queda fuera de
  alcance. Los intervalos que reporte la aplicación van de 0:00 a 0:30 de cada clip.

### Fuentes descartadas

| Fuente | Motivo |
|---|---|
| Spotify | Sus términos de uso no permiten descargar ni extraer el audio. |
| GTZAN | Se armó con grabaciones de CDs, radio y micrófono, y no documenta la licencia de ninguna pista. |
| YouTube y redes sociales | El contenido tiene derechos de terceros y la descarga va contra sus términos. |
| `fma_medium`, `fma_large`, `fma_full` | 22, 93 y 879 GiB. Para 1,000 clips basta `fma_small`. |

## Licencias aceptadas

La licencia de cada pista se clasifica por su `license_url` en `raw_tracks.csv` y no
por su nombre: en `fma_small` el mismo tipo de licencia aparece escrito de más de 60
formas distintas.

| Licencia | ¿Se acepta? | Motivo |
|---|---|---|
| CC BY, CC BY-SA | Sí | Permiten copiar y modificar con atribución. |
| CC BY-NC, CC BY-NC-SA | Sí | El proyecto es académico y no comercial. |
| CC0 y Public Domain Mark | Sí | Sin restricciones. |
| Cualquier variante **ND** (sin obras derivadas), incluida "Music Sharing" | No | Las pruebas de #115 crean remixes, recortes y fragmentos desplazados a partir de las canciones de referencia, y eso es una obra derivada. |
| Sampling Plus y Noncommercial Sampling Plus | No | Son licencias CC retiradas, con condiciones distintas a las actuales. |
| Art Libre, sin licencia o licencia desconocida | No | Para mantener una regla simple y verificable. |

Las condiciones **SA** (compartir igual) solo obligarían si se distribuyera una obra
derivada; los fragmentos modificados de las pruebas se quedan en local y no se
publican.

## Metadatos de cada canción

Cada canción del corpus guarda:

| Campo | Origen | Uso |
|---|---|---|
| `song_id` | `track_id` de FMA | Identificador estable de la canción |
| `title`, `artist_id`, `artist_name`, `album_title`, `genre` | `tracks.csv` | Lo que muestra la aplicación |
| `duration_s` | Duración del clip descargado | Límite de los intervalos (30 s) |
| `license_title`, `license_url` | `raw_tracks.csv` | Atribución y verificación de la licencia |
| `license_family` | Calculada a partir de `license_url` (CC BY, CC BY-NC-SA, ...) | Agrupar y filtrar por licencia |
| `track_url`, `artist_url` | `raw_tracks.csv` | Origen. Son URLs de 2017 y algunas pueden ya no resolver |
| `source` | Fijo: `fma_small` | Dataset de origen |
| `relative_path`, `sha256`, `size_bytes` | Archivo descargado | Ubicación en el volumen y verificación de integridad |

## Atribución

- **En la aplicación:** cada coincidencia muestra la canción de referencia como
  *"Título" de Artista, licencia, URL de origen*.
- **En la documentación y en cualquier presentación de resultados:** citar el
  dataset.

```bibtex
@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
```

## Descarga y almacenamiento

1. Descargar solo desde la ubicación oficial que indica el README de FMA
   (`os.unil.cloud.switch.ch/fma/`), sin mirrors. Los archivos `fma_metadata.zip` y
   `fma_small.zip` se comprueban contra los SHA1 que publica el README cuando se
   descargan completos; si solo se leen algunas pistas, cada una se verifica con el
   CRC del zip y se registra su SHA-256.
2. El corpus vive en el volumen de Docker `db-data`, en `/app/data/audio/fma/`, junto
   con la base de datos. Así las rutas que guarda la tabla de canciones existen dentro
   del contenedor, y ni el audio ni los índices se vuelven a calcular al reiniciar la
   aplicación. `./media` no sirve para esto: se monta en solo lectura. Todos los clips
   van en esa misma carpeta, sin las subcarpetas de FMA, porque `GET /files` solo
   sirve archivos que estén directamente dentro de una carpeta importada.
3. La preparación (`apps/audio-copyright/dataset/prepare_corpus.py`) es idempotente:
   si una pista ya está en el volumen con el SHA-256 esperado, no se vuelve a
   descargar, y si el corpus está completo termina sin tocar la red.
4. No se versionan audio, zips, metadatos de FMA, vectores ni índices. Sí se versiona
   el resumen del corpus (cantidad de pistas por género y por licencia), que no
   contiene audio.

## Canciones agregadas desde la aplicación

La aplicación permite agregar canciones a la base de referencia (`POST /songs`). Solo se
puede agregar audio que se tenga derecho a usar:

- **Propia:** quien la sube es su autor.
- **Una licencia aceptada** (sección "Licencias aceptadas"): CC BY, CC BY-SA, CC BY-NC,
  CC BY-NC-SA, CC0 o dominio público. Nunca una variante ND, por el mismo motivo que en FMA.

Se guardan el título, el artista, la licencia elegida y, si se da, la URL de origen, para
poder atribuirla igual que las de FMA. Los archivos quedan en el volumen, en
`data/audio/uploads/`, y no se versionan. Para retirar una canción agregada se quita su fila
de `uploads.csv`, se borra su archivo y se vuelve a correr `audio-data`, que borra su fila de
`audio_songs` y sus segmentos.

## Retiro de una canción

Si un artista cambia o retira la licencia de una pista, o si alguien del equipo
detecta que una pista no cumple esta política, se elimina del corpus, de la tabla de
canciones y de los índices, y se agrega su `song_id` a la lista de exclusión del
script de preparación para que no se vuelva a descargar.
