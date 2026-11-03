import cv2
import numpy as np
import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType, IndexType
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

    def test_execute_similar_to_with_physical_hnsw_index(self, tmp_path):
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

        # Crear índice vectorial HNSW en catálogo
        cat.create_index("fotos", "ruta", IndexType.HNSW, "idx_fotos_hnsw")

        sql = (
            f"SELECT id, SIMILARITY_SCORE() AS score FROM fotos "
            f"WHERE SIMILAR_TO(ruta, '{query_img}', k=1) USING HNSW WITH METRIC=cosine;"
        )
        result = execute(sql, cat)
        assert len(result) == 1
        assert result[0].values[0].data == 1
