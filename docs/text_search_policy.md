# Política y Especificación de Búsqueda Textual y Normalización

Este documento establece el contrato formal y las especificaciones lingüísticas y algorítmicas adoptadas por `multimodal-dbms-lite` para el procesamiento de texto, indexación invertida y ranking.

---

## 1. Normalización Unicode, Tildes y Caracteres

### 1.1 Forma Canónica Unicode
Todo texto de entrada (sea documento o consulta) se normaliza a la forma **NFKD** (*Compatibility Decomposition*) utilizando el estándar Unicode.

### 1.2 Tratamiento de Tildes y Diacríticos
* **Decisión:** Se eliminan los diacríticos y acentos de los caracteres alfabéticos para maximizar el *recall* (exhaustividad) de la búsqueda.
* **Justificación:** Los usuarios frecuentemente buscan palabras sin tilde (ej. `"cancion"` en vez de `"canción"`). Normalizar ambos al mismo token base asegura que coincidan de manera simétrica.
* **Mapeo:** La letra `á` se descompone en `a` + `´`, y el acento combinatorio se descarta. La letra `ñ` se descompone en `n` + `~` y se normaliza a `n` (compatible con la extensión `unaccent` de PostgreSQL).

### 1.3 Case Folding (Mayúsculas y Minúsculas)
* Todas las palabras se transforman a minúsculas (`lowercase`).
* La búsqueda no distingue mayúsculas de minúsculas (*case-insensitive*).

### 1.4 Delimitadores y Puntuación
* Los signos de puntuación, símbolos especiales (`! ? , ; : . - _ ( ) [ ] { } " ' / \ @ # $ % ^ & * + = ~ `) y saltos de línea se consideran delimitadores.
* Solo se extraen secuencias de caracteres alfanuméricos (`\b\w+\b`).

---

## 2. Idiomas Admitidos y Ejemplos Contrastados

El componente de normalización es compatible con **español** e **inglés**:

| Idioma | Texto Original | Tokens Normalizados |
| :--- | :--- | :--- |
| **Español** | `"¡El rápido zorro marrón saltó sobre el perro perezoso!"` | `["rapido", "zorro", "marron", "salto", "sobre", "perro", "perezoso"]` |
| **Español** | `"Música, programación y comunicación en Bogotá"` | `["musica", "programacion", "comunicacion", "bogota"]` |
| **Inglés** | `"The QUICK brown foxes jumped over 10 lazy dogs!"` | `["quick", "brown", "foxes", "jumped", "over", "10", "lazy", "dogs"]` |
| **Inglés** | `"Information Retrieval & Natural Language Processing (NLP)"` | `["information", "retrieval", "natural", "language", "processing", "nlp"]` |

---

## 3. Stopwords (Palabras Vacías) Oficiales de PostgreSQL

* **Definición:** Palabras funcionales y gramaticales de altísima frecuencia que aportan poca discriminación semántica (artículos, preposiciones, pronombres, verbos auxiliares y conjunciones).
* **Fuente Oficial Canónica:** Los diccionarios adoptados son exactamente los que incluye el código fuente oficial de **PostgreSQL Full Text Search** (proyecto Snowball de Martin Porter):
  - **Español (`spanish.stop`):** [postgres/src/backend/snowball/stopwords/spanish.stop](https://github.com/postgres/postgres/blob/master/src/backend/snowball/stopwords/spanish.stop) (~313 palabras incluyendo pronombres y formas conjugadas de *ser, estar, haber, tener*).
  - **Inglés (`english.stop`):** [postgres/src/backend/snowball/stopwords/english.stop](https://github.com/postgres/postgres/blob/master/src/backend/snowball/stopwords/english.stop) (~127 palabras).
* **Almacenamiento e Integración:**
  - Los archivos `.stop` se almacenan en `src/index/text/stopwords/`.
  - La función `load_stopwords("spanish" | "english")` carga los archivos, normaliza las palabras (NFKD, minúsculas, sin tildes) y los cachea en memoria mediante `@functools.lru_cache` para búsquedas en tiempo $O(1)$.
* **Uso en `TextProcessor`:**
  - `TextProcessor(stopwords="spanish")`: Activa automáticamente el diccionario oficial de PostgreSQL en español.
  - `TextProcessor(stopwords="english")`: Activa el diccionario oficial en inglés.
  - `TextProcessor(stopwords=["palabra1", "palabra2"])`: Permite inyectar cualquier lista personalizada (que también es auto-normalizada).
  - `TextProcessor(stopwords=None)`: Desactiva el filtrado de stopwords.
* **Momento de filtrado:** Se filtran **después de la tokenización y remoción de acentos**, pero antes de cualquier proceso morfológico opcional (stemming), garantizando una correspondencia exacta y determinista.

---

## 4. Stemming (Conversión a Lexemas / Raíz)

* **Decisión:** El componente `TextProcessor` implementa una arquitectura modular con soporte para funciones de *stemming* opcionales.
* Por defecto, la normalización basada en remoción de diacríticos y minúsculas garantiza alta precisión sin sobre-generalización.
* Se provee la interfaz para inyectar algoritmos como Snowball Stemmer (Martin Porter) para español e inglés cuando se requiera reducir formas conjugadas a lexemas comunes:
  - `"caminando"`, `"caminó"`, `"caminante"` ➔ `"camin"`.

---

## 5. Semántica de Consultas Multi-Término

Cuando una consulta contiene múltiples términos (por ejemplo: `"base datos relacional"`):

1. **Modo Ranking BM25 (Disyuntivo Ponderado - OR):**
   * Es el modo por defecto para motores de búsqueda de texto completo.
   * Cualquier documento que contenga **al menos uno** de los términos de la consulta califica como candidato.
   * Aquellos documentos que contengan **más términos** y con mayor frecuencia reciben un puntaje acumulado significativamente superior mediante la suma de BM25.

2. **Modo Filtro Booleano Estricto (Conjuntivo - AND):**
   * Solo califican aquellos documentos que contengan **todos** los términos de la consulta simultáneamente ($D \cap t_i \neq \emptyset \quad \forall t_i \in Q$).

---

## 6. Tratamiento de Términos Desconocidos

* Si una consulta incluye un término que no existe en el vocabulario del corpus ($df(t) = 0$):
  - Su puntuación $\text{IDF}(t)$ se define como $0$.
  - El término no aporta puntaje y no penaliza a los documentos candidatos encontrados por los otros términos conocidos.
  - No se produce ningún error ni excepción en tiempo de ejecución.

---

## 7. Fórmulas y Parámetros de Ranking BM25

La función de puntuación adoptada para ordenar documentos es **Okapi BM25**:

$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{tf(t, D) \cdot (k_1 + 1)}{tf(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{avgdl}\right)}$$

### Parámetros Fijados:
* **$k_1 = 1.2$**: Factor de saturación de frecuencia de término. Controla qué tan rápido se satura la relevancia marginal al repetirse una palabra.
* **$b = 0.75$**: Factor de penalización por longitud de documento. Compara la longitud del documento $|D|$ contra la longitud promedio del corpus ($avgdl$).
* **Fórmula IDF (Lucene / No Negativa):**
  $$\text{IDF}(t) = \ln\left(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right)$$
  Donde:
  - $N$: Número total de documentos en la colección.
  - $df(t)$: Frecuencia de documento del término $t$ (cantidad de documentos que contienen a $t$).

---

## 8. Modelo de Espacio Vectorial (TF-IDF y Similitud del Coseno)

Además del ranking probabilístico BM25, el motor de búsqueda implementa el modelo clásico de espacio vectorial (**Vector Space Model - VSM**) con ponderación **TF-IDF** y normalización por **Similitud del Coseno**.

### 8.1 Fórmulas de Ponderación TF-IDF y Suavizado
Para cada término $t$ dentro de un documento $D$ o una consulta $Q$ en un corpus de $N$ documentos:

1. **Frecuencia de Término ($TF$):**
   $$tf(t, D) = \text{frecuencia absoluta de apariciones de } t \text{ en } D$$

2. **Frecuencia Inversa de Documento ($IDF$ Suavizada Estándar):**
   $$\text{idf}(t) = \ln\left(1 + \frac{N}{df(t)}\right)$$
   *donde $df(t)$ es la cantidad de documentos que contienen al término $t$. Si $df(t) = 0$, se define $\text{idf}(t) = 0$. Esta formulación garantiza que $\text{idf}(t) > 0$ para todo término presente en el corpus, evitando valores negativos o indeterminaciones.*

3. **Pesos de las Coordenadas:**
   * Peso del término en el documento: $w_{t, D} = tf(t, D) \cdot \text{idf}(t)$
   * Peso del término en la consulta: $w_{t, Q} = tf(t, Q) \cdot \text{idf}(t)$

### 8.2 Norma Euclidiana y Optimización de Documentos Candidatos
Para evitar evaluar los $N$ documentos del corpus en cada consulta, el cálculo opera exclusivamente sobre los **documentos candidatos** que contienen al menos un término de la consulta:

$$\text{Candidatos}(Q) = \bigcup_{t \in Q} \{doc\_id \mid (doc\_id, tf) \in \text{postings}(t)\}$$

La longitud o norma euclidiana de cada documento se precalcula o almacena en caché en memoria $O(1)$:

$$\|\vec{V}(D)\| = \sqrt{\sum_{t \in D} w_{t, D}^2} = \sqrt{\sum_{t \in D} (tf(t, D) \cdot \text{idf}(t))^2}$$

Y la norma de la consulta se calcula al momento de la búsqueda:

$$\|\vec{V}(Q)\| = \sqrt{\sum_{t \in Q} w_{t, Q}^2} = \sqrt{\sum_{t \in Q} (tf(t, Q) \cdot \text{idf}(t))^2}$$

### 8.3 Fórmula de Similitud del Coseno
La relevancia entre la consulta $Q$ y el documento candidato $D$ corresponde al coseno del ángulo entre sus respectivos vectores:

$$\text{sim}(Q, D) = \frac{\vec{V}(Q) \cdot \vec{V}(D)}{\|\vec{V}(Q)\| \cdot \|\vec{V}(D)\|} = \frac{\sum_{t \in Q \cap D} w_{t, Q} \cdot w_{t, D}}{\sqrt{\sum_{t \in Q} w_{t, Q}^2} \cdot \sqrt{\sum_{t \in D} w_{t, D}^2}}$$

El puntaje resultante está acotado en el intervalo $[0, 1]$, donde $1.0$ representa alineación vectorial exacta.

### 8.4 Tratamiento de Casos Límite y Términos No Indexables
1. **Consultas Vacías, Nulas o de Solo Stopwords:**
   * Si la consulta es `""`, `None`, o contiene únicamente stopwords o símbolos que se filtran por completo, el vector de consulta resultante es el vector nulo $\vec{0} \implies \|\vec{V}(Q)\| = 0$.
   * **Comportamiento:** La función retorna inmediatamente `[]` sin realizar operaciones innecesarias y evitando divisiones por cero.
2. **Consultas con Términos Desconocidos ($df = 0$):**
   * Los términos fuera del vocabulario reciben $\text{idf}(t) = 0$, aportando $0$ a las coordenadas y a la norma. Si ningún término es conocido, la consulta retorna `[]`.
3. **Documentos sin Términos Indexables:**
   * Si un documento quedó con longitud 0 tras la remoción de stopwords, su norma es $\|\vec{V}(D)\| = 0$.
   * **Comportamiento:** Se valida explícitamente `if norm_d == 0: continue`, previniendo `ZeroDivisionError` y asignando similitud $0.0$.

### 8.5 Política de Desempate Determinista
Para asegurar reproducibilidad absoluta entre ejecuciones independientes:
1. **Orden Primario:** Similitud del Coseno **descendente** (mayor puntaje primero).
2. **Orden Secundario (Desempate):** Identificador de documento (`doc_id`) **ascendente** (menor ID primero).
   $$\text{Clave de Ordenación: } (-score, doc\_id)$$

---

## 9. Coherencia con PostgreSQL Full-Text Search

La configuración de `multimodal-dbms-lite` es semánticamente coherente con PostgreSQL:
* **PostgreSQL:** `to_tsvector('spanish', 'Canción!')` produce `'cancion':1`.
* **multimodal-dbms-lite:** `TextProcessor.process('Canción!')` produce `['cancion']`.
* El uso de postings `(doc_id, tf)` equivale al vector de posiciones y pesos en `tsvector`, permitiendo benchmarks comparativos directos contra índices GIN en PostgreSQL.

---

## 10. Integración con el Catálogo del Motor (Sintaxis PostgreSQL y Ciclo de Vida)

Para no mantener los índices de texto completo como artefactos desconectados, `multimodal-dbms-lite` unifica la gestión de índices invertidos con la capa formal del catálogo (`Catalog`), siguiendo la sintaxis estándar y el modelo de metadatos de PostgreSQL.

### 10.1 Sintaxis Compatible con PostgreSQL
El motor soporta la definición y eliminación de índices textuales mediante sentencias SQL estándar:

1. **Creación de Índice:**
   ```sql
   CREATE INDEX [IF NOT EXISTS] <nombre_indice> ON <tabla> USING { gin | spimi } (<columna>);
   ```
   * *Compatibilidad:* `USING gin` y `USING spimi` son sinónimos soportados.
   * *Validación Semántica:* Solo se permite indexar columnas textuales (`TEXT`, `VARCHAR`, `CHAR`). Cualquier intento sobre columnas numéricas o espaciales genera `SemanticError`.
   * *Unicidad de Nombres:* El nombre del índice debe ser único en el catálogo.

2. **Eliminación de Índice:**
   ```sql
   DROP INDEX [IF EXISTS] <nombre_indice>;
   ```
   * En alineación con PostgreSQL, no es necesario especificar el nombre de la tabla para eliminar un índice.

### 10.2 Metadatos en el Catálogo del Sistema (`sys_indexes`)
La metadata del índice se almacena de forma persistente en la tabla del sistema `sys_indexes`:

| Columna | Tipo | Descripción |
| :--- | :--- | :--- |
| `index_id` | `INTEGER` | Identificador autoincremental único del índice (clave primaria). |
| `table_id` | `INTEGER` | Identificador de la tabla objetivo en `sys_tables`. |
| `column_name` | `VARCHAR(64)` | Columna indexada. |
| `index_type` | `VARCHAR(16)` | Tipo de índice (`"gin"` o `"spimi"`). |
| `root_page_id` | `INTEGER` | Fijado en `0` para índices invertidos (habilita reapertura automática). |
| `index_name` | `VARCHAR(64)` | Nombre formal del índice (ej. `idx_articulos_contenido`). |

### 10.3 Mapeo Biyectivo $O(1)$ entre RID y Identificador de Documento
Para enlazar los resultados de búsqueda invertida directamente con los registros del Heap File sin consumir memoria adicional en punteros:

$$\text{doc\_id} = (\text{page\_id} \ll 16) \mid (\text{slot} \ \& \ \text{0xFFFF})$$

$$\text{RID} = (\text{doc\_id} \gg 16, \ \text{doc\_id} \ \& \ \text{0xFFFF})$$

Esta transformación aritmética pura es estrictamente biyectiva y permite convertir los candidatos recuperados por BM25 o Similitud del Coseno directamente a RIDs para lectura instantánea de tuplas.

### 10.4 Persistencia, Reboot y Ciclo de Vida Físico
* **Ubicación en Disco:** Cada índice se almacena físicamente en `{table_name}.{column_name}.{index_id}.idx` a través de la fábrica de buffers del catálogo (`_index_buffer_factory`).
* **Reinicio del Motor (`_load`):** Al iniciar el catálogo, escanea `sys_indexes` y, al detectar entradas con `index_type in ("gin", "spimi")`, invoca `InvertedIndex.load(file_path)` restaurando automáticamente todo el vocabulario, listas de postings y longitudes de documentos en memoria.
* **Mantenimiento Dinámico (DML):**
  - `INSERT`: Inyecta los tokens procesados en el índice en memoria vía `add_document(doc_id, tokens)` y persiste a disco con `save()`.
  - `DELETE`: Remueve las entradas de postings vía `remove_document(doc_id)` y persiste a disco.
  - `UPDATE`: Remueve el `doc_id` anterior y registra el nuevo documento tokenizado.
* **Eliminación en Cascada:**
  - `DROP INDEX <nombre>`: Elimina el registro de `sys_indexes`, desaloca la estructura en memoria y borra físicamente el archivo `.idx` del disco (`os.remove`).
  - `DROP TABLE <tabla>`: Realiza la eliminación en cascada, eliminando de disco todos los índices secundarios e invertidos asociados a dicha tabla.

