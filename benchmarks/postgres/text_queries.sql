-- benchmarks/postgres/text_queries.sql — Consultas PostgreSQL equivalentes a
-- las consultas full-text del DBMS propio.
--
-- load_text_data.py separa los bloques por la línea "-- name: <nombre>" y
-- los ejecuta con parámetros de psycopg (%(nombre)s). %(tsquery)s son los
-- términos de benchmarks/text_common.py:generate_queries() unidos con " | ".
--
-- Equivalencias:
--   MATCH(contenido, 'a b c')            -> tsv @@ to_tsquery('simple', 'a | b | c')
--                                           (semántica OR, como MATCH)
--   SCORE() ... USING BM25 | TF_IDF      -> ts_rank(tsv, q, 1)
--                                           (normalización 1: divide por 1 + log(longitud))
--   ORDER BY relevancia DESC LIMIT k     -> ORDER BY score DESC, id LIMIT k
--                                           (desempate por id, como el motor)
--
-- El índice GIN solo resuelve el filtro @@; ts_rank se calcula sobre cada
-- candidato leyendo su tsvector (load_text_data.py verifica el plan).

-- name: topk
SELECT id, ts_rank(tsv, q, 1) AS score
FROM documentos, to_tsquery('simple', %(tsquery)s) AS q
WHERE tsv @@ q
ORDER BY score DESC, id
LIMIT %(k)s;

-- name: matches
SELECT count(*)
FROM documentos
WHERE tsv @@ to_tsquery('simple', %(tsquery)s);

-- name: vocabulary
SELECT count(*)
FROM ts_stat('SELECT tsv FROM documentos');