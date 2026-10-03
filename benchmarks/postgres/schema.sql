-- benchmarks/postgres/schema.sql — Tabla espacial para la comparación
-- externa contra el R-tree del DBMS propio.
--
-- Coordenadas cartesianas (SRID 0) con distancia euclidiana, equivalente a
-- DISTANCIA(col, POINT(x, y)) / DISTANCIA(col, POINT(x, y), EUCLIDEAN).
--
-- El índice GiST NO se crea aquí: load_data.py lo construye después de la
-- carga para medir su tiempo de construcción, igual que con el R-tree.

CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS places;

CREATE TABLE places (
    id   INTEGER PRIMARY KEY,
    name VARCHAR(50),
    geom geometry(Point, 0) NOT NULL
);
