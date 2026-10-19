"""
Unit tests for Full-Text Search (GIN/SPIMI) Catalog Integration:
Creation, registration, disk persistence, reboot recovery, dynamic DML updates,
and deletion via DROP INDEX / DROP TABLE following PostgreSQL syntax.
"""

import os
import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType, IndexType
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.parser.ast_nodes import CreateIndexStm, DropIndexStm, IndexType as AstIndexType
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.query_engine import execute, QueryError
from index.text import doc_id_to_rid, rid_to_doc_id


def make_catalog(base_dir):
    heap_factory = make_heap_factory(str(base_dir))
    return Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )


def flush_catalog(catalog):
    for storage in catalog._table_storage.values():
        storage.bm.flush_all()
        if hasattr(storage, "overflow"):
            storage.overflow.bm.flush_all()
    catalog._sys_tables.bm.flush_all()
    catalog._sys_columns.bm.flush_all()
    catalog._sys_indexes.bm.flush_all()
    for index in catalog._physical_indexes.values():
        if hasattr(index, "save") and callable(index.save):
            index.save()
        elif hasattr(index, "bm") and hasattr(index.bm, "flush_all"):
            index.bm.flush_all()


def test_parser_accepts_postgres_gin_and_spimi_syntax():
    """Validates PostgreSQL syntax: CREATE INDEX ... USING {gin | spimi} (...) and DROP INDEX."""
    # Standard PostgreSQL CREATE INDEX USING gin
    stmts1 = Parser(Scanner("CREATE INDEX idx_art_gin ON articulos USING gin (contenido);")).parse_sql_statements()
    assert len(stmts1) == 1
    assert isinstance(stmts1[0], CreateIndexStm)
    assert stmts1[0].index_name == "idx_art_gin"
    assert stmts1[0].table == "articulos"
    assert stmts1[0].column == "contenido"
    assert stmts1[0].index_type == AstIndexType.GIN

    # CREATE INDEX USING spimi alias
    stmts2 = Parser(Scanner("CREATE INDEX idx_art_spimi ON articulos USING spimi (contenido);")).parse_sql_statements()
    assert len(stmts2) == 1
    assert isinstance(stmts2[0], CreateIndexStm)
    assert stmts2[0].index_name == "idx_art_spimi"
    assert stmts2[0].index_type == AstIndexType.SPIMI

    # DROP INDEX
    stmts3 = Parser(Scanner("DROP INDEX idx_art_gin;")).parse_sql_statements()
    assert len(stmts3) == 1
    assert isinstance(stmts3[0], DropIndexStm)
    assert stmts3[0].index_name == "idx_art_gin"
    assert not stmts3[0].if_exists

    # DROP INDEX IF EXISTS
    stmts4 = Parser(Scanner("DROP INDEX IF EXISTS idx_art_gin;")).parse_sql_statements()
    assert len(stmts4) == 1
    assert isinstance(stmts4[0], DropIndexStm)
    assert stmts4[0].index_name == "idx_art_gin"
    assert stmts4[0].if_exists


def test_semantic_rejects_gin_spimi_on_non_text_columns(tmp_path):
    """GIN and SPIMI must fail when created on non-text columns (e.g. INTEGER, POINT)."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE articulos (id INTEGER PRIMARY KEY, vistas INTEGER, contenido TEXT);", catalog)

    with pytest.raises(QueryError, match="GIN y SPIMI requieren una columna de tipo texto"):
        execute("CREATE INDEX idx_art_num ON articulos USING gin (vistas);", catalog)

    with pytest.raises(QueryError, match="GIN y SPIMI requieren una columna de tipo texto"):
        execute("CREATE INDEX idx_art_spimi_num ON articulos USING spimi (vistas);", catalog)


def test_create_gin_index_builds_file_and_registers_metadata(tmp_path):
    """CREATE INDEX creates .idx file on disk and registers sys_indexes entry."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE noticias (id INTEGER PRIMARY KEY, cuerpo TEXT);", catalog)
    execute("INSERT INTO noticias VALUES (1, 'Base de datos relacionales y motores de búsqueda.');", catalog)
    execute("INSERT INTO noticias VALUES (2, 'Algoritmos de recuperación de información y SPIMI.');", catalog)

    execute("CREATE INDEX idx_noticias_cuerpo ON noticias USING gin (cuerpo);", catalog)

    # Check sys_indexes metadata
    indexes = catalog.get_indexes("noticias")
    assert len(indexes) == 1
    entry = indexes[0]
    assert entry["index_name"] == "idx_noticias_cuerpo"
    assert entry["column_name"] == "cuerpo"
    assert entry["index_type"] == "gin"

    # Check physical file creation
    expected_file = tmp_path / f"noticias.cuerpo.{entry['index_id']}.idx"
    assert os.path.exists(expected_file)
    assert os.path.getsize(expected_file) > 0

    # Query index directly from catalog
    physical_index = catalog.get_physical_index("noticias", "cuerpo")
    assert physical_index is not None
    # 'busqueda' stemmed should be in the index
    terms = physical_index.get_terms()
    assert any("busqued" in t or "busqueda" in t for t in terms)

    # Test BM25 retrieval
    query_tokens = catalog.text_processor.process("motores de busqueda")
    results = physical_index.bm25_search(query_tokens)
    assert len(results) > 0
    top_doc_id, top_score = results[0]
    page_id, slot = doc_id_to_rid(top_doc_id)
    assert top_score > 0.0


def test_index_persistence_after_catalog_reboot(tmp_path):
    """Verifies that GIN/SPIMI index survives reboot without loss of data or vocabulary."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE articulos (id INTEGER PRIMARY KEY, contenido TEXT);", catalog)
    execute("INSERT INTO articulos VALUES (1, 'Sistemas distribuidos y consistencia eventual.');", catalog)
    execute("INSERT INTO articulos VALUES (2, 'Indices invertidos y ranking BM25 en bases de datos.');", catalog)
    execute("CREATE INDEX idx_art_texto ON articulos USING gin (contenido);", catalog)
    flush_catalog(catalog)

    # Simulate shutdown and reboot
    del catalog
    rebooted = make_catalog(tmp_path)

    assert "articulos" in rebooted.tables
    indexes = rebooted.get_indexes("articulos")
    assert len(indexes) == 1
    assert indexes[0]["index_name"] == "idx_art_texto"
    assert indexes[0]["index_type"] == "gin"

    physical_index = rebooted.get_physical_index("articulos", "contenido")
    assert physical_index is not None
    assert physical_index.total_docs == 2

    # Query on rebooted catalog
    query_tokens = rebooted.text_processor.process("ranking BM25")
    results = physical_index.bm25_search(query_tokens)
    assert len(results) >= 1
    doc_id, score = results[0]
    page_id, slot = doc_id_to_rid(doc_id)
    assert score > 0.0

    # Verify cosine similarity works on rebooted index
    cosine_results = physical_index.cosine_similarity_search(query_tokens)
    assert len(cosine_results) >= 1
    assert cosine_results[0][1] > 0.0


def test_dynamic_dml_updates_inverted_index(tmp_path):
    """Verifies INSERT and DELETE dynamically maintain the inverted index and persist."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, mensaje TEXT);", catalog)
    execute("CREATE INDEX idx_logs_msg ON logs USING gin (mensaje);", catalog)

    physical_index = catalog.get_physical_index("logs", "mensaje")
    assert physical_index.total_docs == 0

    # Insert new record -> should update index
    execute("INSERT INTO logs VALUES (1, 'Error crítico en el servidor de base de datos');", catalog)
    assert physical_index.total_docs == 1
    query_tokens = catalog.text_processor.process("servidor critico")
    results = physical_index.bm25_search(query_tokens)
    assert len(results) == 1

    # Insert second record
    execute("INSERT INTO logs VALUES (2, 'Advertencia de memoria en el nodo secundario');", catalog)
    assert physical_index.total_docs == 2

    # Delete record 1 -> should remove document from postings
    execute("DELETE FROM logs WHERE id = 1;", catalog)
    assert physical_index.total_docs == 1
    results_after_delete = physical_index.bm25_search(query_tokens)
    assert len(results_after_delete) == 0

    # Verify persistence after reboot
    flush_catalog(catalog)
    del catalog
    rebooted = make_catalog(tmp_path)
    idx_rebooted = rebooted.get_physical_index("logs", "mensaje")
    assert idx_rebooted.total_docs == 1
    query_node = rebooted.text_processor.process("secundario")
    assert len(idx_rebooted.bm25_search(query_node)) == 1


def test_drop_index_removes_metadata_and_disk_file(tmp_path):
    """DROP INDEX cleans sys_indexes and deletes physical .idx file."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE articulos (id INTEGER PRIMARY KEY, cuerpo TEXT);", catalog)
    execute("INSERT INTO articulos VALUES (1, 'Texto de prueba.');", catalog)
    execute("CREATE INDEX idx_art_cuerpo ON articulos USING spimi (cuerpo);", catalog)

    idx_entry = catalog.get_indexes("articulos")[0]
    idx_file = tmp_path / f"articulos.cuerpo.{idx_entry['index_id']}.idx"
    assert os.path.exists(idx_file)

    # DROP INDEX
    execute("DROP INDEX idx_art_cuerpo;", catalog)

    # Metadata should be clean
    assert len(catalog.get_indexes("articulos")) == 0
    assert not catalog.index_exists("idx_art_cuerpo")
    assert catalog.get_physical_index("articulos", "cuerpo") is None

    # Disk file should be deleted
    assert not os.path.exists(idx_file)

    # Reboot should still show index does not exist
    flush_catalog(catalog)
    del catalog
    rebooted = make_catalog(tmp_path)
    assert len(rebooted.get_indexes("articulos")) == 0
    assert not rebooted.index_exists("idx_art_cuerpo")


def test_drop_index_if_exists_semantics(tmp_path):
    """DROP INDEX without IF EXISTS fails on unknown index, with IF EXISTS passes silently."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE articulos (id INTEGER PRIMARY KEY, cuerpo TEXT);", catalog)

    with pytest.raises(QueryError, match="El índice 'idx_fantasma' no existe"):
        execute("DROP INDEX idx_fantasma;", catalog)

    # Should succeed without error
    execute("DROP INDEX IF EXISTS idx_fantasma;", catalog)


def test_drop_table_cascades_and_removes_gin_spimi_file(tmp_path):
    """DROP TABLE cascades and deletes the .idx file of GIN/SPIMI indexes."""
    catalog = make_catalog(tmp_path)
    execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, texto TEXT);", catalog)
    execute("INSERT INTO docs VALUES (1, 'Documento para borrado de tabla.');", catalog)
    execute("CREATE INDEX idx_docs_texto ON docs USING gin (texto);", catalog)

    idx_entry = catalog.get_indexes("docs")[0]
    idx_file = tmp_path / f"docs.texto.{idx_entry['index_id']}.idx"
    assert os.path.exists(idx_file)

    # DROP TABLE
    execute("DROP TABLE docs;", catalog)

    # File should be removed
    assert not os.path.exists(idx_file)
    assert "docs" not in catalog.tables
