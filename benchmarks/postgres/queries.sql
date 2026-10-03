-- benchmarks/postgres/queries.sql — Consultas PostGIS equivalentes a las
-- consultas espaciales del DBMS propio.
--
-- load_data.py separa los bloques por la línea "-- name: <query_type>" y
-- ejecuta cada uno con parámetros de psycopg (%(nombre)s), tomados de
-- benchmarks/spatial_common.py:generate_queries().
--
-- Equivalencias (distancia euclidiana, SRID 0):
--   radius      DISTANCIA(geom, POINT(x, y)) <= r     -> ST_DWithin
--   knn         ORDER BY DISTANCIA(geom, POINT(x, y))
--               LIMIT k                               -> ORDER BY <-> LIMIT k
--   intersects  DENTRO_DE(geom, POLYGON(...))         -> ST_Intersects
--
-- Las tres son "index-aware": el planner puede resolverlas con el índice
-- GiST places_geom_gist (load_data.py verifica el plan de cada una).

-- name: radius
SELECT id
FROM places
WHERE ST_DWithin(geom, ST_MakePoint(%(x)s, %(y)s), %(r)s);

-- name: knn
SELECT id
FROM places
ORDER BY geom <-> ST_MakePoint(%(x)s, %(y)s)
LIMIT %(k)s;

-- name: intersects
SELECT id
FROM places
WHERE ST_Intersects(geom, ST_GeomFromText(%(poly)s, 0));
