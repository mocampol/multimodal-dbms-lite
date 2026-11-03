-- benchmarks/postgres/text_schema.sql — Tablas de texto para la comparación
-- externa contra el índice invertido SPIMI del DBMS propio.
--
-- documentos_staging recibe por COPY el texto y sus tokens ya normalizados
-- por el TextProcessor del motor (benchmarks/text_common.py); load_text_data.py
-- llena documentos.tsv con to_tsvector('simple', tokens), así ambos motores
-- indexan exactamente el mismo vocabulario.
--
-- El índice GIN NO se crea aquí: load_text_data.py lo construye después de
-- la carga para medir su tiempo de construcción, igual que CREATE INDEX ...
-- USING SPIMI en el DBMS propio.

DROP TABLE IF EXISTS documentos_staging;
DROP TABLE IF EXISTS documentos;

CREATE TABLE documentos (
    id        INTEGER PRIMARY KEY,
    contenido TEXT NOT NULL,
    tsv       tsvector NOT NULL
);

CREATE UNLOGGED TABLE documentos_staging (
    id        INTEGER PRIMARY KEY,
    contenido TEXT NOT NULL,
    tokens    TEXT NOT NULL
);