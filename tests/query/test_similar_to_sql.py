import os

import cv2
import numpy as np
import pytest
import soundfile as sf

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType, IndexType
from index.multimedia import ExtractFeatures
from index.vector import VectorMetric
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.parser.ast_nodes import (
    SelectStm,
    SimilarToExp,
    SimilarityScoreSpec,
    VectorIndexKind,
    VectorMetricKind,
)
from query.parser.exceptions import SemanticError
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.parser.token_ import TokenType
from query.parser.visitor import SemanticVisitor
from query.query_engine import QueryError, execute


def parse(sql: str):
    return Parser(Scanner(sql)).parse_sql_statements()[0]


def token_types(source: str):
    scanner = Scanner(source)
    types = []
    while (token := scanner.next_token()).type != TokenType.END:
        types.append(token.type)
    return types


def _create_sample_image(path: str, color_val: int) -> str:
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(img, (75, 75), 30, (color_val, 255, color_val), -1)
    cv2.rectangle(img, (10, 10), (50, 50), (255, color_val, 100), 2)
    cv2.imwrite(path, img)
    return path


class TestScannerVector:
    def test_scanner_tokens(self):
        types = token_types("SIMILAR_TO SIMILARITY_SCORE IVF HNSW WITH METRIC EUCLIDEAN DOT_PRODUCT COSINE")
        assert types == [
            TokenType.SIMILAR_TO,
            TokenType.SIMILARITY_SCORE,
            TokenType.IVF,
            TokenType.HNSW,
            TokenType.WITH,
            TokenType.METRIC,
            TokenType.EUCLIDEAN,
            TokenType.DOT_PRODUCT,
            TokenType.COSINE,
        ]

    def test_scanner_compound_tokens(self):
        types = token_types("SIMILAR TO DOT PRODUCT")
        assert types == [
            TokenType.SIMILAR_TO,
            TokenType.DOT_PRODUCT,
        ]


class TestParserVector:
    def test_parse_similar_to_basic(self):
        sql = "SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg', k=5) USING HNSW WITH METRIC=cosine;"
        stm = parse(sql)
        assert isinstance(stm, SelectStm)
        assert isinstance(stm.where_cond, SimilarToExp)
        assert stm.where_cond.target == "query.jpg"
        assert stm.where_cond.k == 5
        assert stm.where_cond.index_kind == VectorIndexKind.HNSW
        assert stm.where_cond.metric == VectorMetricKind.COSINE

    def test_parse_similar_to_with_column_and_score(self):
        sql = (
            "SELECT id, SIMILARITY_SCORE() AS score FROM fotos "
            "WHERE SIMILAR_TO(archivo, 'query.png', k=10) USING IVF WITH METRIC=euclidean;"
        )
        stm = parse(sql)
        assert isinstance(stm, SelectStm)
        assert isinstance(stm.where_cond, SimilarToExp)
        assert stm.where_cond.column.value == "archivo"
        assert stm.where_cond.index_kind == VectorIndexKind.IVF
        assert stm.where_cond.metric == VectorMetricKind.EUCLIDEAN

        # ScoreSpec ligado al SimilarToExp
        score_spec = next(c for c in stm.columns if isinstance(c, SimilarityScoreSpec))
        assert score_spec.is_bound
        assert score_spec.alias == "score"
        assert score_spec.target == "query.png"
        assert score_spec.column == "archivo"

    def test_parse_syntax_errors(self):
        # Falta parámetro k
        with pytest.raises(RuntimeError):
            parse("SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg') USING HNSW;")

        # k inválido <= 0
        with pytest.raises(RuntimeError):
            parse("SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg', k=0) USING HNSW;")

        # Falta comilla en objeto
        with pytest.raises(RuntimeError):
            parse("SELECT * FROM fotos WHERE SIMILAR_TO(query.jpg, k=5) USING HNSW;")

    @pytest.mark.parametrize(
        "sql, error",
        [
            (
                "SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg', k=5) USING FOO;",
                "el tipo de índice para USING",
            ),
            (
                "SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg', k=5) WITH METRIC=manhattan;",
                "una métrica válida",
            ),
            (
                "SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg', k=-1);",
                "Error sintáctico",
            ),
            (
                "SELECT * FROM fotos WHERE SIMILAR_TO('query.jpg', k=5) USING;",
                "el tipo de índice para USING",
            ),
        ],
    )
    def test_invalid_similarity_queries_are_rejected(self, sql, error):
        with pytest.raises(QueryError, match=error):
            execute(sql, None)


class TestVisitorVector:
    def test_similarity_score_without_similar_to_rejected(self, tmp_path):
        heap_factory = make_heap_factory(str(tmp_path))
        cat = Catalog(
            heap_factory=heap_factory,
            storage_factories={
                StorageType.HEAP: heap_factory,
                StorageType.SEQUENTIAL: make_sequential_factory(str(tmp_path)),
            },
            index_buffer_factory=make_index_buffer_factory(str(tmp_path)),
        )
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, archivo VARCHAR(100));", cat)

        stm = parse("SELECT id, SIMILARITY_SCORE() FROM fotos;")
        with pytest.raises(SemanticError, match="requiere una condición SIMILAR_TO"):
            SemanticVisitor(cat).visit_select_stm(stm)

    def test_similar_to_with_join_rejected(self, tmp_path):
        heap_factory = make_heap_factory(str(tmp_path))
        cat = Catalog(
            heap_factory=heap_factory,
            storage_factories={
                StorageType.HEAP: heap_factory,
                StorageType.SEQUENTIAL: make_sequential_factory(str(tmp_path)),
            },
            index_buffer_factory=make_index_buffer_factory(str(tmp_path)),
        )
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, archivo VARCHAR(100));", cat)
        execute("CREATE TABLE tags (id INTEGER PRIMARY KEY, foto_id INTEGER);", cat)

        stm = parse(
            "SELECT * FROM fotos JOIN tags ON fotos.id = tags.foto_id "
            "WHERE SIMILAR_TO('busqueda.jpg', k=3);"
        )
        with pytest.raises(SemanticError, match="no se admite junto con JOIN"):
            SemanticVisitor(cat).visit_select_stm(stm)


class TestExecutionVector:
    def test_execute_similar_to_fallback(self, tmp_path):
        img1 = str(tmp_path / "img1.png")
        img2 = str(tmp_path / "img2.png")
        query_img = str(tmp_path / "query.png")

        _create_sample_image(img1, 50)
        _create_sample_image(img2, 200)
        # La consulta es idéntica a img1
        _create_sample_image(query_img, 50)

        heap_factory = make_heap_factory(str(tmp_path))
        cat = Catalog(
            heap_factory=heap_factory,
            storage_factories={
                StorageType.HEAP: heap_factory,
                StorageType.SEQUENTIAL: make_sequential_factory(str(tmp_path)),
            },
            index_buffer_factory=make_index_buffer_factory(str(tmp_path)),
        )
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(200));", cat)
        execute(f"INSERT INTO fotos VALUES (1, '{img1}'), (2, '{img2}');", cat)

        sql = (
            f"SELECT id, SIMILARITY_SCORE() AS score FROM fotos "
            f"WHERE SIMILAR_TO(ruta, '{query_img}', k=2) USING HNSW WITH METRIC=cosine;"
        )
        result = execute(sql, cat)
        assert len(result) == 2
        # img1 debe ser más similar a query_img que img2
        scores = [row.values[1].data for row in result]
        assert scores[0] >= scores[1]
        assert result[0].values[0].data == 1

    @pytest.mark.parametrize(
        "index_type,sql_index_kind",
        [
            (IndexType.IVF, "IVF"),
            (IndexType.HNSW, "HNSW"),
        ],
    )
    def test_execute_similar_to_uses_requested_index_kind(
        self, tmp_path, monkeypatch, index_type, sql_index_kind
    ):
        img1 = str(tmp_path / "idx_img1.png")
        img2 = str(tmp_path / "idx_img2.png")
        query_img = str(tmp_path / "idx_query.png")

        _create_sample_image(img1, 80)
        _create_sample_image(img2, 240)
        _create_sample_image(query_img, 80)

        heap_factory = make_heap_factory(str(tmp_path))
        cat = Catalog(
            heap_factory=heap_factory,
            storage_factories={
                StorageType.HEAP: heap_factory,
                StorageType.SEQUENTIAL: make_sequential_factory(str(tmp_path)),
            },
            index_buffer_factory=make_index_buffer_factory(str(tmp_path)),
        )
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(200));", cat)
        execute(f"INSERT INTO fotos VALUES (1, '{img1}'), (2, '{img2}');", cat)

        # Both index kinds can exist for one column; USING chooses which one runs.
        execute("CREATE INDEX idx_fotos_hnsw ON fotos USING HNSW (ruta);", cat)
        execute("CREATE INDEX idx_fotos_ivf ON fotos USING IVF (ruta);", cat)
        selected_index = cat.get_physical_index(
            "fotos", "ruta", index_type=index_type.value
        )
        calls = []
        original_search = selected_index.search

        def capture_search(query_vector, k=10, metric=None):
            calls.append(metric)
            return original_search(query_vector, k=k, metric=metric)

        monkeypatch.setattr(selected_index, "search", capture_search)

        sql = (
            f"SELECT id, SIMILARITY_SCORE() AS score FROM fotos "
            f"WHERE SIMILAR_TO(ruta, '{query_img}', k=2) "
            f"USING {sql_index_kind} WITH METRIC=euclidean "
            f"ORDER BY score DESC;"
        )
        result = execute(sql, cat)
        assert len(result) == 2
        assert result[0].values[0].data == 1
        assert result[0].values[1].data >= result[1].values[1].data
        assert calls == [VectorMetric.EUCLIDEAN]


def _make_catalog(base_dir):
    heap_factory = make_heap_factory(str(base_dir))
    return Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )


def _create_pattern_image(path, seed):
    """Imagen con figuras aleatorias: cada semilla produce descriptores distintos."""
    rng = np.random.default_rng(seed)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    for _ in range(30):
        x, y = rng.integers(0, 180, 2).tolist()
        size = int(rng.integers(5, 25))
        color = rng.integers(60, 255, 3).tolist()
        if rng.random() < 0.5:
            cv2.circle(image, (x, y), size, color, -1)
        else:
            cv2.rectangle(image, (x, y), (x + size, y + size), color, -1)
    cv2.imwrite(str(path), image)
    return str(path)


def _create_tone(path, frequency, sr=22050):
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)
    return str(path)


def _knn_sql(query, k, kind="HNSW", metric="cosine"):
    return (
        "SELECT id, SIMILARITY_SCORE() AS score FROM fotos "
        f"WHERE SIMILAR_TO(ruta, '{query}', k={k}) USING {kind} WITH METRIC={metric};"
    )


def _ids(rows):
    return [row.values[0].data for row in rows]


def _scores(rows):
    return [row.values[1].data for row in rows]


@pytest.fixture
def photos(tmp_path):
    """Tabla fotos con 4 imágenes; la consulta es idéntica a la imagen del id 3."""
    catalog = _make_catalog(tmp_path)
    execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)
    values = ", ".join(
        f"({i}, '{_create_pattern_image(tmp_path / f'foto{i}.png', seed=i)}')"
        for i in range(1, 5)
    )
    execute(f"INSERT INTO fotos VALUES {values};", catalog)
    query = _create_pattern_image(tmp_path / "consulta.png", seed=3)
    return catalog, query


class TestSimilarToKnn:
    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot_product"])
    @pytest.mark.parametrize(
        "create_index, kind",
        [(False, "HNSW"), (True, "HNSW"), (True, "IVF")],
    )
    def test_identical_object_is_first_and_scores_are_ordered(
        self, photos, create_index, kind, metric
    ):
        catalog, query = photos
        if create_index:
            execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)

        rows = execute(_knn_sql(query, 4, kind, metric), catalog)

        scores = _scores(rows)
        assert len(rows) == 4
        assert _ids(rows)[0] == 3
        assert np.isclose(scores[0], 1.0, atol=1e-3)
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[1]

    @pytest.mark.parametrize("k, expected", [(1, 1), (2, 2), (10, 4)])
    def test_k_limits_results_without_duplicates(self, photos, k, expected):
        catalog, query = photos
        ids = _ids(execute(_knn_sql(query, k), catalog))
        assert len(ids) == expected
        assert len(set(ids)) == expected

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot_product"])
    def test_indexes_return_same_order_as_sequential_search(self, photos, metric):
        catalog, query = photos
        sequential = _ids(execute(_knn_sql(query, 4, "HNSW", metric), catalog))

        execute("CREATE INDEX idx_hnsw ON fotos USING HNSW (ruta);", catalog)
        execute("CREATE INDEX idx_ivf ON fotos USING IVF (ruta);", catalog)

        assert _ids(execute(_knn_sql(query, 4, "HNSW", metric), catalog)) == sequential
        assert _ids(execute(_knn_sql(query, 4, "IVF", metric), catalog)) == sequential

    def test_other_index_kind_is_not_used(self, photos, monkeypatch):
        catalog, query = photos
        execute("CREATE INDEX idx_hnsw ON fotos USING HNSW (ruta);", catalog)
        hnsw = catalog.get_physical_index("fotos", "ruta", index_type=IndexType.HNSW.value)
        calls = []
        monkeypatch.setattr(hnsw, "search", lambda *args, **kwargs: calls.append(args))

        rows = execute(_knn_sql(query, 1, "IVF"), catalog)

        assert calls == []
        assert _ids(rows) == [3]

    def test_infix_syntax(self, photos):
        catalog, query = photos
        rows = execute(f"SELECT id FROM fotos WHERE ruta SIMILAR_TO('{query}', k=1);", catalog)
        assert _ids(rows) == [3]

    def test_column_is_optional(self, photos):
        catalog, query = photos
        rows = execute(f"SELECT id FROM fotos WHERE SIMILAR_TO('{query}', k=1);", catalog)
        assert _ids(rows) == [3]

    def test_null_values_are_ignored(self, photos):
        catalog, query = photos
        execute("INSERT INTO fotos VALUES (5, NULL);", catalog)

        ids = _ids(execute(_knn_sql(query, 10), catalog))

        assert 5 not in ids
        assert ids[0] == 3

    def test_audio_search(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)
        low = _create_tone(tmp_path / "bajo.wav", 220)
        high = _create_tone(tmp_path / "alto.wav", 3000)
        query = _create_tone(tmp_path / "consulta.wav", 3000)
        execute(f"INSERT INTO fotos VALUES (1, '{low}'), (2, '{high}');", catalog)

        assert _ids(execute(_knn_sql(query, 2), catalog))[0] == 2

    @pytest.mark.parametrize("kind", ["IVF", "HNSW"])
    def test_index_survives_catalog_restart(self, photos, tmp_path, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        expected = _ids(execute(_knn_sql(query, 4, kind), catalog))

        reopened = _reopen(catalog, tmp_path)

        index = reopened.get_physical_index("fotos", "ruta", index_type=kind.lower())
        assert len(index) == 4
        assert _ids(execute(_knn_sql(query, 4, kind), reopened)) == expected


def _reopen(catalog, base_dir):
    for storage in catalog._table_storage.values():
        storage.bm.flush_all()
    catalog._sys_tables.bm.flush_all()
    catalog._sys_columns.bm.flush_all()
    catalog._sys_indexes.bm.flush_all()
    return _make_catalog(base_dir)


def _vector_index(catalog, kind):
    return catalog.get_physical_index("fotos", "ruta", index_type=kind.lower())


def _forbid_fit(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("la consulta no debería reentrenar el extractor")

    monkeypatch.setattr(ExtractFeatures, "fit", fail)


@pytest.mark.parametrize("kind", ["IVF", "HNSW"])
class TestVectorIndexExtractor:
    """El índice guarda el extractor con que se construyó y las consultas lo reutilizan."""

    def test_search_uses_the_index_extractor_without_refitting(self, photos, monkeypatch, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        _forbid_fit(monkeypatch)

        assert _ids(execute(_knn_sql(query, 1, kind), catalog)) == [3]

    def test_extractor_survives_catalog_restart(self, photos, tmp_path, monkeypatch, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)

        reopened = _reopen(catalog, tmp_path)
        _forbid_fit(monkeypatch)

        assert reopened.get_vector_extractor(_vector_index(reopened, kind)) is not None
        assert _ids(execute(_knn_sql(query, 1, kind), reopened)) == [3]

    def test_index_without_saved_extractor_still_searches(self, photos, tmp_path, kind):
        # Índices creados antes de guardar el extractor: la consulta reentrena como antes.
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        os.remove(_vector_index(catalog, kind)._storage_path + ".extractor")

        reopened = _reopen(catalog, tmp_path)

        assert reopened.get_vector_extractor(_vector_index(reopened, kind)) is None
        assert _ids(execute(_knn_sql(query, 1, kind), reopened)) == [3]

    @pytest.mark.parametrize("statement", ["DROP INDEX idx_fotos;", "DROP TABLE fotos;"])
    def test_drop_removes_every_index_file(self, photos, kind, statement):
        catalog, _ = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        index = _vector_index(catalog, kind)
        path = index._storage_path
        files = [path, path + ".extractor"] + ([path + ".bin"] if kind == "HNSW" else [])
        assert all(os.path.exists(file) for file in files)

        execute(statement, catalog)

        assert not any(os.path.exists(file) for file in files)
        assert catalog.get_vector_extractor(index) is None


class TestSimilarToEmptyDatasets:
    @pytest.mark.parametrize("kind", ["IVF", "HNSW"])
    def test_empty_table_returns_no_rows(self, tmp_path, kind):
        catalog = _make_catalog(tmp_path)
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)
        query = _create_pattern_image(tmp_path / "consulta.png", seed=1)

        assert execute(_knn_sql(query, 3, kind), catalog) == []

    @pytest.mark.parametrize("kind", ["IVF", "HNSW"])
    def test_index_on_empty_table_is_rejected(self, tmp_path, kind):
        catalog = _make_catalog(tmp_path)
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)

        with pytest.raises((QueryError, ValueError), match="No hay valores"):
            execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)


class TestSimilarToInvalidInput:
    @pytest.mark.parametrize(
        "query, error",
        [
            ("no_existe.png", "no existe"),
            ("no_existe.wav", "no existe"),
            ("archivo.xyz", "Extensión de archivo no reconocida"),
        ],
    )
    def test_invalid_query_object_is_rejected(self, photos, query, error):
        catalog, _ = photos
        with pytest.raises(ValueError, match=error):
            execute(_knn_sql(query, 1), catalog)

    @pytest.mark.parametrize(
        "sql, error",
        [
            ("SELECT id FROM fotos WHERE SIMILAR_TO('a.png', k=0);", "entero positivo"),
            ("SELECT id FROM fotos WHERE SIMILAR_TO('   ', k=1);", "no vacía"),
            ("SELECT id, SIMILARITY_SCORE() FROM fotos;", "requiere una condición"),
            ("SELECT id FROM fotos WHERE SIMILAR_TO(no_existe, 'a.png', k=1);", "no_existe"),
            ("SELECT COUNT(*) FROM fotos WHERE SIMILAR_TO('a.png', k=1);", "agregados"),
            ("SELECT ruta FROM fotos WHERE SIMILAR_TO('a.png', k=1) GROUP BY ruta;", "GROUP BY"),
            (
                "SELECT id, SIMILARITY_SCORE() AS a, SIMILARITY_SCORE() AS b "
                "FROM fotos WHERE SIMILAR_TO('a.png', k=1);",
                "Solo se admite un",
            ),
            (
                "SELECT id, SIMILARITY_SCORE() AS ruta FROM fotos WHERE SIMILAR_TO('a.png', k=1);",
                "choca",
            ),
            ("DELETE FROM fotos WHERE SIMILAR_TO('a.png', k=1);", "solo se admite en el WHERE"),
            (
                "UPDATE fotos SET ruta = 'b.png' WHERE SIMILAR_TO('a.png', k=1);",
                "solo se admite en el WHERE",
            ),
        ],
    )
    def test_invalid_statement_is_rejected(self, tmp_path, sql, error):
        catalog = _make_catalog(tmp_path)
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)

        with pytest.raises(QueryError, match=error):
            execute(sql, catalog)


class TestVectorJoin:
    """El motor no implementa uniones por similitud: deben rechazarse con un error claro."""

    @pytest.mark.parametrize(
        "columns", ["*", "fotos.id, SIMILARITY_SCORE() AS score"]
    )
    def test_similar_to_with_join_is_rejected(self, tmp_path, columns):
        catalog = _make_catalog(tmp_path)
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY, ruta VARCHAR(300));", catalog)
        execute("CREATE TABLE etiquetas (id INTEGER PRIMARY KEY, foto_id INTEGER);", catalog)

        with pytest.raises(QueryError, match="no se admite junto con JOIN"):
            execute(
                f"SELECT {columns} FROM fotos JOIN etiquetas ON fotos.id = etiquetas.foto_id "
                "WHERE SIMILAR_TO('a.png', k=1);",
                catalog,
            )

    def test_regular_join_still_works_on_table_with_vector_index(self, photos):
        catalog, _ = photos
        execute("CREATE INDEX idx_fotos ON fotos USING HNSW (ruta);", catalog)
        execute("CREATE TABLE etiquetas (id INTEGER PRIMARY KEY, foto_id INTEGER);", catalog)
        execute("INSERT INTO etiquetas VALUES (1, 3);", catalog)

        rows = execute(
            "SELECT etiquetas.id, fotos.id FROM fotos "
            "JOIN etiquetas ON fotos.id = etiquetas.foto_id;",
            catalog,
        )

        assert [(row.values[0].data, row.values[1].data) for row in rows] == [(1, 3)]


def _all_ids(catalog):
    return sorted(_ids(execute("SELECT id FROM fotos;", catalog)))


@pytest.mark.parametrize("kind", ["IVF", "HNSW"])
class TestVectorIndexMaintenance:
    """INSERT, DELETE y UPDATE mantienen al día los índices IVF y HNSW."""

    def test_insert_after_creating_index(self, photos, tmp_path, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        path = _create_pattern_image(tmp_path / "foto5.png", seed=5)

        execute(f"INSERT INTO fotos VALUES (5, '{path}');", catalog)

        assert 5 in _ids(execute(_knn_sql(query, 5, kind), catalog))
        assert len(_vector_index(catalog, kind)) == 5
        assert _ids(execute(_knn_sql(path, 1, kind), catalog)) == [5]

    def test_delete_after_creating_index(self, photos, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)

        execute("DELETE FROM fotos WHERE id = 3;", catalog)

        assert 3 not in _ids(execute(_knn_sql(query, 4, kind), catalog))
        assert len(_vector_index(catalog, kind)) == 3

    def test_update_moves_the_row_to_its_new_vector(self, photos, tmp_path, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        path = _create_pattern_image(tmp_path / "nueva.png", seed=9)

        execute(f"UPDATE fotos SET ruta = '{path}' WHERE id = 3;", catalog)

        assert len(_vector_index(catalog, kind)) == 4
        assert _ids(execute(_knn_sql(path, 1, kind), catalog)) == [3]
        # el vector viejo de la fila 3 ya no está: nada coincide del todo con la consulta
        assert _scores(execute(_knn_sql(query, 1, kind), catalog))[0] < 0.99

    def test_null_value_is_inserted_but_not_indexed(self, photos, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)

        execute("INSERT INTO fotos VALUES (5, NULL);", catalog)
        execute("DELETE FROM fotos WHERE id = 5;", catalog)

        assert len(_vector_index(catalog, kind)) == 4
        assert _all_ids(catalog) == [1, 2, 3, 4]

    @pytest.mark.parametrize("value", ["no es una ruta", "C:/no/existe/foto.png"])
    def test_insert_without_vector_is_rejected_and_undone(self, photos, kind, value):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)

        with pytest.raises(ValueError, match="No se pudo calcular el vector"):
            execute(f"INSERT INTO fotos VALUES (5, '{value}');", catalog)

        assert _all_ids(catalog) == [1, 2, 3, 4]
        assert len(_vector_index(catalog, kind)) == 4

    def test_changes_survive_catalog_restart(self, photos, tmp_path, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        path = _create_pattern_image(tmp_path / "foto5.png", seed=5)
        execute(f"INSERT INTO fotos VALUES (5, '{path}');", catalog)
        execute("DELETE FROM fotos WHERE id = 3;", catalog)

        reopened = _reopen(catalog, tmp_path)

        assert len(_vector_index(reopened, kind)) == 4
        assert _ids(execute(_knn_sql(path, 1, kind), reopened)) == [5]
        assert 3 not in _ids(execute(_knn_sql(query, 4, kind), reopened))

    def test_index_without_saved_extractor_is_rebuilt_on_insert(self, photos, tmp_path, kind):
        catalog, _ = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        extractor_path = _vector_index(catalog, kind)._storage_path + ".extractor"
        os.remove(extractor_path)
        reopened = _reopen(catalog, tmp_path)
        path = _create_pattern_image(tmp_path / "foto5.png", seed=5)

        execute(f"INSERT INTO fotos VALUES (5, '{path}');", reopened)

        assert os.path.exists(extractor_path)
        assert len(_vector_index(reopened, kind)) == 5
        assert _ids(execute(_knn_sql(path, 1, kind), reopened)) == [5]

    def test_rebuild_derived_rebuilds_the_vector_index(self, photos, kind):
        catalog, query = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)

        catalog.rebuild_derived("fotos")

        assert len(_vector_index(catalog, kind)) == 4
        assert _ids(execute(_knn_sql(query, 1, kind), catalog)) == [3]

    def test_rebuild_derived_on_empty_table_waits_for_the_next_insert(self, photos, tmp_path, kind):
        catalog, _ = photos
        execute(f"CREATE INDEX idx_fotos ON fotos USING {kind} (ruta);", catalog)
        execute("DELETE FROM fotos;", catalog)

        catalog.rebuild_derived("fotos")
        assert len(_vector_index(catalog, kind)) == 0

        path = _create_pattern_image(tmp_path / "foto5.png", seed=5)
        execute(f"INSERT INTO fotos VALUES (5, '{path}');", catalog)
        assert _ids(execute(_knn_sql(path, 1, kind), catalog)) == [5]


class TestKnownSimilarityFailures:
    """Comportamiento esperado que el motor aún no cumple."""

    @pytest.mark.xfail(
        strict=True,
        reason="_validate_text_search solo admite SCORE(...) después de '*'",
    )
    def test_star_with_similarity_score(self, photos):
        catalog, query = photos

        rows = execute(
            f"SELECT *, SIMILARITY_SCORE() AS score FROM fotos "
            f"WHERE SIMILAR_TO(ruta, '{query}', k=1);",
            catalog,
        )

        assert rows[0].values[0].data == 3
        assert len(rows[0].values) == 3

    @pytest.mark.xfail(
        strict=True,
        reason="VectorSearch entrena con toda la columna y una fila no multimedia aborta la consulta",
    )
    def test_non_media_row_does_not_break_search(self, photos):
        catalog, query = photos
        execute("INSERT INTO fotos VALUES (5, 'no es una ruta');", catalog)

        assert _ids(execute(_knn_sql(query, 1), catalog)) == [3]