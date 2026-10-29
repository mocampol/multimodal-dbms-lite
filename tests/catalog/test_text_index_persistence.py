"""
Persistence and reconstruction of full-text (SPIMI / GIN) indexes, following
the lifecycle fixed in docs/text_search_policy.md section 10.4:

  * each index lives in {table}.{column}.{index_id}.idx and its metadata in
    sys_indexes (type 'gin' / 'spimi', root_page_id 0, unique name)
  * on startup the catalog reopens it with InvertedIndex.load (no rebuild)
  * INSERT / DELETE / UPDATE keep it current and save it before returning
  * DROP INDEX deletes its file; DROP TABLE cascades to every index file
  * catalog.rebuild_derived(table) reconstructs it from the table rows

A "reboot" is a new Catalog over the same directory. Search results are
compared with a reference BM25 computed from a plain Python model of the table,
so they prove the persisted/maintained index answers like the data says.
"""

import math
from pathlib import Path

import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from index.text import InvertedIndex, default_text_processor
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import QueryError, execute

PROCESSOR = default_text_processor()

# id -> contenido (None = SQL NULL). Id 4 has only stopwords (0 indexable
# terms) and id 5 is NULL: neither can match, and only id 4 counts as a document.
INITIAL = {
    1: "Los indices aceleran la busqueda de registros por clave y la busqueda por rango",
    2: "El optimizador elige el plan de ejecucion con menor costo",
    3: "Un indice invertido asocia el termino con la lista de documentos",
    4: "de la y",
    5: None,
    6: "El ranking BM25 ordena documentos por relevancia segun el termino y la frecuencia del termino",
}
QUERIES = [
    "indice busqueda",
    "documentos termino",
    "ranking relevancia frecuencia",
    "optimizador plan costo estadisticas",
    "lista invertida enlaza igualdad hash",
]

# The DML sequence of test_insert_delete_and_update_are_searchable_and_survive_reboots,
# as (statement, change to the model). Kept here so the model cannot drift from the SQL.
SESSION_1 = [
    ("INSERT INTO docs VALUES (7, 'Doc 7', 'el indice hash acelera la busqueda por igualdad');",
     lambda m: m.__setitem__(7, "el indice hash acelera la busqueda por igualdad")),
    ("DELETE FROM docs WHERE id = 2;", lambda m: m.pop(2)),
    ("UPDATE docs SET contenido = 'una lista invertida enlaza terminos con documentos' WHERE id = 3;",
     lambda m: m.__setitem__(3, "una lista invertida enlaza terminos con documentos")),
    ("UPDATE docs SET contenido = NULL WHERE id = 6;", lambda m: m.__setitem__(6, None)),
]
SESSION_2 = [
    ("INSERT INTO docs VALUES (8, 'Doc 8', 'el optimizador usa estadisticas del catalogo para el plan');",
     lambda m: m.__setitem__(8, "el optimizador usa estadisticas del catalogo para el plan")),
    ("DELETE FROM docs WHERE id = 7;", lambda m: m.pop(7)),
    ("UPDATE docs SET contenido = 'los indices B+ aceleran la busqueda' WHERE id = 1;",
     lambda m: m.__setitem__(1, "los indices B+ aceleran la busqueda")),
]


# --------------------------------------------------------------------------
# Engine helpers
# --------------------------------------------------------------------------

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


def reboot(catalog, base_dir):
    flush_catalog(catalog)
    return make_catalog(base_dir)


def literal(text):
    return "NULL" if text is None else f"'{text}'"


def create_table(catalog):
    execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, titulo VARCHAR(40), contenido TEXT);",
        catalog,
    )


def insert(catalog, rows):
    values = ", ".join(f"({i}, 'Doc {i}', {literal(text)})" for i, text in rows.items())
    execute(f"INSERT INTO docs VALUES {values};", catalog)


def seed(base_dir, index_type="SPIMI", index_first=False, model=None):
    """Catalog with table docs(id, titulo, contenido) and idx_contenido."""
    model = INITIAL if model is None else model
    catalog = make_catalog(base_dir)
    create_table(catalog)
    if index_first:
        execute(f"CREATE INDEX idx_contenido ON docs USING {index_type} (contenido);", catalog)
        insert(catalog, model)
    else:
        insert(catalog, model)
        execute(f"CREATE INDEX idx_contenido ON docs USING {index_type} (contenido);", catalog)
    return catalog


def index_files(base_dir, column=None):
    return sorted(p.name for p in Path(base_dir).glob(f"docs.{column or '*'}.*.idx"))


def index_path(base_dir):
    (path,) = Path(base_dir).glob("docs.contenido.*.idx")
    return path


def state(index):
    """Everything an InvertedIndex persists, comparable with =="""
    return {
        "index": {term: list(postings) for term, postings in index.index.items()},
        "doc_lengths": dict(index.doc_lengths),
        "total_docs": index.total_docs,
    }


def live_state(catalog):
    return state(catalog.get_text_index("docs", "contenido"))


# --------------------------------------------------------------------------
# Reference ranking over the Python model of the table
# --------------------------------------------------------------------------

def reference(model, query, k1=1.2, b=0.75):
    corpus = {i: PROCESSOR.process(t) for i, t in model.items() if t is not None}
    if not corpus:
        return []
    n = len(corpus)
    avgdl = sum(len(tokens) for tokens in corpus.values()) / n
    scores = {}
    for term in dict.fromkeys(PROCESSOR.process(query)):
        df = sum(1 for tokens in corpus.values() if term in tokens)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for doc_id, tokens in corpus.items():
            tf = tokens.count(term)
            if tf:
                norm = k1 * (1 - b + b * len(tokens) / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + norm)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def ranked(catalog, query):
    rows = execute(
        "SELECT id, SCORE() AS relevancia FROM docs "
        f"WHERE MATCH(contenido, '{query}') ORDER BY relevancia DESC;",
        catalog,
    )
    return [(row.values[0].data, row.values[1].data) for row in rows]


def check_searches(catalog, model):
    """Every query answers exactly as the model says (ids, order and scores)."""
    for query in QUERIES:
        expected = reference(model, query)
        scores = [s for _, s in expected]
        # tied scores would make the order depend on RID reuse, not on the index
        assert all(a - b > 1e-9 for a, b in zip(scores, scores[1:])), query
        actual = ranked(catalog, query)
        assert [i for i, _ in actual] == [i for i, _ in expected], query
        assert [s for _, s in actual] == pytest.approx(scores, rel=1e-9), query


# --------------------------------------------------------------------------
# Reopening the index after a reboot
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index_type", ["SPIMI", "GIN"])
def test_index_is_reopened_after_a_reboot_with_its_exact_content(tmp_path, index_type):
    before = seed(tmp_path, index_type)
    check_searches(before, INITIAL)
    expected_state = live_state(before)
    assert expected_state["total_docs"] == 5  # 6 rows minus the NULL; the stopwords-only row counts

    after = reboot(before, tmp_path)

    assert after.get_text_index("docs", "contenido") is not before.get_text_index("docs", "contenido")
    assert live_state(after) == expected_state
    check_searches(after, INITIAL)
    nodes = execute(
        "EXPLAIN SELECT id FROM docs WHERE MATCH(contenido, 'indice');", after
    ).description
    assert "TextIndexScan" in str(nodes)  # served by the reopened index, not by a scan


@pytest.mark.parametrize("index_type", ["SPIMI", "GIN"])
def test_documents_without_terms_and_null_values_persist_correctly(tmp_path, index_type):
    after = reboot(seed(tmp_path, index_type), tmp_path)
    index = after.get_text_index("docs", "contenido")

    texts = [t for t in INITIAL.values() if t is not None]
    assert index.total_docs == len(texts)  # NULL is not a document
    assert sorted(index.doc_lengths.values()) == sorted(len(PROCESSOR.process(t)) for t in texts)
    assert 0 in index.doc_lengths.values()  # the stopwords-only document is kept, length 0


def test_index_file_is_a_standalone_loadable_artifact(tmp_path):
    catalog = seed(tmp_path)
    path = index_path(tmp_path)

    assert path.name.startswith("docs.contenido.") and path.suffix == ".idx"
    assert state(InvertedIndex.load(str(path))) == live_state(catalog)


def test_index_metadata_is_persisted_in_the_catalog(tmp_path):
    before = seed(tmp_path)
    after = reboot(before, tmp_path)

    assert after.index_exists("idx_contenido")
    table, entry = after.get_index_by_name("idx_contenido")
    assert table == "docs"
    assert (entry["column_name"], entry["index_type"], entry["root_page_id"]) == ("contenido", "spimi", 0)
    assert entry["index_id"] == before.get_index_by_name("idx_contenido")[1]["index_id"]
    assert index_path(tmp_path).name == f"docs.contenido.{entry['index_id']}.idx"


def test_name_uniqueness_and_drop_work_after_a_reboot(tmp_path):
    after = reboot(seed(tmp_path), tmp_path)

    with pytest.raises((QueryError, ValueError), match="Ya existe un índice con nombre 'idx_contenido'"):
        execute("CREATE INDEX idx_contenido ON docs USING GIN (titulo);", after)

    execute("DROP INDEX idx_contenido;", after)
    assert not after.index_exists("idx_contenido") and index_files(tmp_path) == []


# --------------------------------------------------------------------------
# Maintenance by DML, persisted
# --------------------------------------------------------------------------

def apply(catalog, model, statements):
    for sql, change in statements:
        execute(sql, catalog)
        change(model)


def ids(catalog, query):
    return [doc_id for doc_id, _ in ranked(catalog, query)]


@pytest.mark.parametrize("index_type", ["SPIMI", "GIN"])
def test_insert_delete_and_update_are_searchable_and_survive_reboots(tmp_path, index_type):
    model = dict(INITIAL)
    catalog = seed(tmp_path, index_type)
    check_searches(catalog, model)
    assert ids(catalog, "optimizador plan costo estadisticas") == [2]
    assert ids(catalog, "ranking relevancia frecuencia") == [6]

    # --- first session: INSERT, DELETE, UPDATE of the text and UPDATE to NULL
    apply(catalog, model, SESSION_1)
    check_searches(catalog, model)
    assert ids(catalog, "lista invertida enlaza igualdad hash") == [3, 7]  # rewritten doc 3, new doc 7
    assert ids(catalog, "optimizador plan costo estadisticas") == []   # deleted row is gone
    assert ids(catalog, "ranking relevancia frecuencia") == []         # NULL leaves the index
    first_session = live_state(catalog)
    assert first_session["total_docs"] == sum(t is not None for t in model.values())

    # --- reboot: nothing lost, nothing resurrected
    second = reboot(catalog, tmp_path)
    assert live_state(second) == first_session
    check_searches(second, model)
    assert ids(second, "optimizador plan costo estadisticas") == []

    # --- second session on the reopened index, then a third boot
    apply(second, model, SESSION_2)
    check_searches(second, model)
    assert ids(second, "optimizador plan costo estadisticas") == [8]

    third = reboot(second, tmp_path)
    assert live_state(third) == live_state(second)
    check_searches(third, model)


def test_every_dml_statement_saves_the_index_before_returning(tmp_path):
    catalog = seed(tmp_path)
    path = index_path(tmp_path)

    for sql in (
        "INSERT INTO docs VALUES (7, 'Doc 7', 'el indice hash acelera la busqueda');",
        "UPDATE docs SET contenido = 'una lista invertida de terminos' WHERE id = 3;",
        "UPDATE docs SET contenido = NULL WHERE id = 1;",
        "DELETE FROM docs WHERE id = 2;",
    ):
        execute(sql, catalog)
        assert state(InvertedIndex.load(str(path))) == live_state(catalog), sql  # already on disk


def test_no_temporary_files_are_left_behind(tmp_path):
    catalog = seed(tmp_path)
    execute("INSERT INTO docs VALUES (7, 'Doc 7', 'el indice hash acelera la busqueda');", catalog)
    execute("DELETE FROM docs WHERE id = 2;", catalog)
    catalog.rebuild_derived("docs")

    assert list(Path(tmp_path).glob(".*.tmp")) == []  # atomic save replaces, never lingers


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index_type, index_first", [
    ("SPIMI", False), ("GIN", False), ("SPIMI", True), ("GIN", True),
])
def test_every_construction_path_builds_the_same_index(tmp_path, index_type, index_first):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    expected = live_state(seed(reference_dir, "SPIMI", index_first=False))  # bulk SPIMI

    built = seed(candidate_dir, index_type, index_first=index_first)  # bulk GIN / incremental inserts

    assert live_state(built) == expected
    check_searches(built, INITIAL)


@pytest.mark.parametrize("index_type", ["SPIMI", "GIN"])
def test_rebuild_derived_repairs_an_out_of_sync_index(tmp_path, index_type):
    catalog = reboot(seed(tmp_path, index_type), tmp_path)
    healthy = live_state(catalog)

    # simulate what a crash recovery can leave behind: the best document missing from the index
    index = catalog.get_text_index("docs", "contenido")
    best_doc_id = index.rank(PROCESSOR.process(QUERIES[0]), "bm25")[0][0]
    index.remove_document(best_doc_id)
    assert live_state(catalog) != healthy
    assert [i for i, _ in ranked(catalog, QUERIES[0])] != [i for i, _ in reference(INITIAL, QUERIES[0])]

    catalog.rebuild_derived("docs")

    assert live_state(catalog) == healthy
    check_searches(catalog, INITIAL)
    check_searches(reboot(catalog, tmp_path), INITIAL)  # and the repair was persisted


def test_rebuild_after_dml_equals_the_incrementally_maintained_index(tmp_path):
    catalog = seed(tmp_path)
    execute("INSERT INTO docs VALUES (7, 'Doc 7', 'el indice hash acelera la busqueda');", catalog)
    execute("DELETE FROM docs WHERE id = 2;", catalog)
    execute("UPDATE docs SET contenido = NULL WHERE id = 6;", catalog)
    maintained = live_state(catalog)

    catalog.rebuild_derived("docs")

    assert live_state(catalog) == maintained


# --------------------------------------------------------------------------
# Physical lifecycle: DROP INDEX and DROP TABLE
# --------------------------------------------------------------------------

def test_drop_index_removes_only_its_own_file(tmp_path):
    catalog = seed(tmp_path)
    execute("CREATE INDEX idx_titulo ON docs USING GIN (titulo);", catalog)
    assert [name.split(".")[1] for name in index_files(tmp_path)] == ["contenido", "titulo"]

    execute("DROP INDEX idx_titulo;", catalog)

    assert [name.split(".")[1] for name in index_files(tmp_path)] == ["contenido"]
    assert catalog.get_text_index("docs", "titulo") is None
    # titulo can still be searched, now by a full scan
    rows = execute("SELECT id FROM docs WHERE MATCH(titulo, 'doc');", catalog)
    assert sorted(row.values[0].data for row in rows) == sorted(INITIAL)

    after = reboot(catalog, tmp_path)
    assert after.index_exists("idx_contenido") and not after.index_exists("idx_titulo")
    check_searches(after, INITIAL)


def test_drop_table_cascades_to_every_index_file(tmp_path):
    catalog = seed(tmp_path)
    execute("CREATE INDEX idx_titulo ON docs USING GIN (titulo);", catalog)
    assert len(index_files(tmp_path)) == 2

    execute("DROP TABLE docs;", catalog)

    assert list(Path(tmp_path).glob("docs.*")) == []  # table file and both index files
    after = reboot(catalog, tmp_path)
    assert not after.table_exists("docs")
    assert not after.index_exists("idx_contenido") and not after.index_exists("idx_titulo")


def test_a_dropped_and_recreated_table_does_not_inherit_the_old_index(tmp_path):
    catalog = seed(tmp_path)
    execute("DROP TABLE docs;", catalog)

    create_table(catalog)
    insert(catalog, {1: "un solo documento sobre indices"})
    execute("CREATE INDEX idx_contenido ON docs USING SPIMI (contenido);", catalog)

    assert live_state(catalog)["total_docs"] == 1
    assert [i for i, _ in ranked(catalog, "indices")] == [1]
    assert [i for i, _ in ranked(reboot(catalog, tmp_path), "indices")] == [1]