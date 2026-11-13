# Política de datos para reconocimiento facial

## Alcance y decisión

Esta política define qué datos puede usar el prototipo académico de
reconocimiento facial, cómo se relacionan las muestras con identidades y cómo
se registra y retira el consentimiento. No autoriza vigilancia ni identificación
de personas que no hayan aceptado participar.

- **Benchmark inicial propuesto: [LFW (Labeled Faces in the Wild)](https://vis-www.cs.umass.edu/lfw/)**,
  solo para evaluación offline de similitud y recuperación. Antes de descargar
  o usarlo, quien lo incorpore debe guardar la versión y verificar que sus
  condiciones vigentes permiten el uso académico previsto. La disponibilidad
  pública no se considera por sí sola una licencia ni consentimiento
  individual. No mostrar nombres de sus identidades en la aplicación.
- **CelebA y VGGFace2 quedan fuera del primer alcance.** Se reconsiderarán
  únicamente si se documentan sus condiciones de uso, redistribución y
  tratamiento compatibles con este proyecto
  ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),
  [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)). Su
  recomendación en clase no sustituye esa revisión.
- **Fotos de estudiantes o compañeros:** solo se aceptan tras consentimiento
  explícito, informado y voluntario, obtenido antes de capturar o importar
  imágenes. La falta de respuesta significa que no hay consentimiento. No se
  suben fotos de compañeros al repositorio.
- Si las condiciones de LFW no se pueden verificar o no permiten el uso
  previsto, no se incorporan sus imágenes; se usa un conjunto sintético o se
  detiene esa evaluación hasta resolverlo.

El proyecto es académico y local: no se despliegan los datos como servicio ni
se comparten fuera del equipo. Las imágenes y sus embeddings se tratan como
datos biométricos sensibles y reciben las mismas restricciones.

## Descarga y almacenamiento

1. Descargar manualmente desde la página oficial del dataset solo después de
   revisar y documentar sus condiciones. No descargar desde mirrors no
   verificados ni copiar imágenes desde redes sociales.
2. Guardar archivos locales bajo `benchmarks/corpus/face/` (por ejemplo,
   `benchmarks/corpus/face/lfw/`). `benchmarks/corpus/` ya está excluida de
   Git.
3. Mantener localmente un manifiesto del corpus con: dataset y versión,
   URL oficial, fecha de descarga, checksum del archivo fuente, URL/versión de
   las condiciones revisadas, propósito autorizado y responsable de la
   revisión. No incluir fotos, embeddings, nombres de participantes ni una
   copia de los términos que no se tenga derecho a redistribuir.
4. Antes de crear un commit, comprobar que no incluya imágenes, archivos
   comprimidos, embeddings, índices vectoriales, galerías, capturas ni
   manifiestos de participantes. No subirlos a issues, CI, logs o artefactos.
5. Si se autorizan fotos propias para una prueba, mantenerlas en un directorio
   local excluido de Git, separado del benchmark público y accesible solo al
   equipo que las necesita.

## Asociación entre muestras e identidades

Usar identificadores internos opacos (`identity_id` y `sample_id`), nunca el
nombre de una persona como clave. El esquema lógico mínimo es:

| Registro | Campos mínimos | Regla |
|---|---|---|
| Identidad | `identity_id`, `source`, `source_subject_id`, `display_label` opcional, `consent_status` | `source_subject_id` se limita al identificador que entrega el dataset; para participantes, se usa un código aleatorio. |
| Muestra | `sample_id`, `identity_id`, `relative_path`, `source`, `split`, `sha256` | Cada imagen debe apuntar a una identidad y tener procedencia verificable. |
| Consentimiento | `identity_id`, `purpose`, `granted_at`, `notice_version`, `status`, `withdrawn_at` opcional | Para fotos de participantes, `status` es `pending`, `granted` o `withdrawn`; solo `granted` permite procesarlas. Los registros de datasets públicos no representan consentimiento individual y no deben marcarse `granted`. |

Para LFW, la asociación imagen-identidad se toma exclusivamente de las
anotaciones oficiales de la misma versión descargada. El identificador de
origen se conserva como `source_subject_id`; no se infieren identidades por
similitud ni se fusionan identidades entre datasets. Para participantes, una
persona autorizada confirma qué imágenes le pertenecen y asigna el código
interno; no se usa reconocimiento automático para crear o corregir esas
asociaciones. La etiqueta visible, si se necesita en una demostración con
participantes, se mantiene separada de las imágenes y se muestra solo con
autorización explícita para ese uso.

El manifiesto versionable del código puede describir columnas y formato, pero
los registros de personas, rutas locales, nombres, consentimientos y hashes
de muestras individuales quedan fuera del repositorio.

## Consentimiento y retiro

Antes de una captura o importación de fotos de una persona, comunicar en
lenguaje claro: qué imágenes se usarán, que se generarán embeddings, el fin
académico concreto, quién tendrá acceso, cuánto tiempo se conservarán y cómo
pedir su eliminación. Obtener una respuesta afirmativa explícita y fechada
para ese propósito. La participación es voluntaria y retirarse no tiene
consecuencias. Un permiso para almacenar la imagen no implica permiso para
mostrar el nombre ni para otros usos.

Al retirar el consentimiento:

1. Marcar la identidad como retirada y excluirla inmediatamente de nuevas
   búsquedas y cargas.
2. Eliminar sus imágenes, embeddings y entradas del índice; reconstruir o
   actualizar los índices y limpiar copias de trabajo.
3. Eliminar copias de respaldo dentro del plazo de retención documentado y
   evitar que una restauración vuelva a incluirlas.
4. Conservar, si hace falta para auditar el retiro, solo el registro mínimo de
   que se procesó la solicitud, sin retener imágenes ni embeddings.

## Criterios para incorporar datos

Una fuente se puede incorporar únicamente cuando se cumplen todos estos
criterios:

- Hay procedencia y condiciones de uso documentadas, compatibles con el uso
  académico local.
- Las identidades se relacionan mediante etiquetas/anotaciones de la fuente o
  confirmación voluntaria de la persona; no por inferencia no verificada.
- Toda foto aportada por una persona tiene consentimiento explícito vigente y
  limitado al propósito registrado.
- Hay una ruta de retiro que cubre imágenes, embeddings, índices y copias.
- El conjunto y sus derivados permanecen fuera de Git y no se usan para
  identificar en vivo a personas que no participan.
