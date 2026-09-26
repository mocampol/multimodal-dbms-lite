"""
Tests del flujo de importación CSV (prepare_import -> overrides -> load_csv)
usando los CSV mock de tests/loader/data/.
"""

from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest

from catalog.catalog import Catalog
from catalog.exceptions import TableAlreadyExistsError
from catalog.table_metadata import StorageType
from common.schema import Column, Schema
from common.value import DataType
from loader import (
    DuplicateColumnNameError,
    EmptyCSVError,
    InconsistentRowError,
    UnsupportedTypeError,
    infer_schema,
    load_csv,
    prepare_import,
)
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute


DATA = Path(__file__).parent / "data"


def csv(name: str) -> str:
    return str(DATA / name)


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


def types_of(schema: Schema) -> dict:
    return {c.name: c.data_type for c in schema.columns}


def rows_of(catalog, table: str) -> list[list]:
    rows = execute(f"SELECT * FROM {table};", catalog)
    return [[v.data for v in row.values] for row in rows]


def override(schema: Schema, **changes) -> Schema:
    """Simula lo que haría el frontend: reemplaza columnas por nombre."""
    columns = [changes.get(c.name, c) for c in schema.columns]
    return Schema(schema.table_name, columns)


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------

def test_infer_schema_basic_types():
    schema = infer_schema("alumnos", DATA / "alumnos.csv")

    assert schema.table_name == "alumnos"
    assert [c.name for c in schema.columns] == ["id", "nombre", "edad", "promedio", "activo"]
    assert types_of(schema) == {
        "id": DataType.INTEGER,
        "nombre": DataType.VARCHAR,
        "edad": DataType.INTEGER,
        "promedio": DataType.DOUBLE_PRECISION,
        "activo": DataType.BOOLEAN,
    }
    assert schema.get_column("nombre").size == 16
    assert all(c.nullable for c in schema.columns)


def test_infer_schema_ignores_empty_cells_mid_column():
    schema = infer_schema("productos", DATA / "productos_nulls.csv")

    assert types_of(schema) == {
        "codigo": DataType.INTEGER,
        "precio": DataType.DOUBLE_PRECISION,
        "stock": DataType.INTEGER,
        "descripcion": DataType.VARCHAR,
    }


def test_infer_schema_empty_first_cell_does_not_force_varchar():
    # edad: "", "30", "25"  -> debería ser INTEGER
    # nota: "", "15", ""    -> debería ser INTEGER
    schema = infer_schema("t", DATA / "primera_celda_vacia.csv")

    assert types_of(schema)["edad"] == DataType.INTEGER
    assert types_of(schema)["nota"] == DataType.INTEGER


def test_infer_schema_widening():
    schema = infer_schema("t", DATA / "widening.csv")

    assert types_of(schema) == {
        "id": DataType.INTEGER,
        "valor": DataType.DOUBLE_PRECISION,  # 10 -> 2.5
        "flag": DataType.BOOLEAN,            # true/false/T
        "mixto": DataType.VARCHAR,           # 10 -> abc
    }


def test_infer_schema_dates_fall_back_to_varchar():
    schema = infer_schema("t", DATA / "fechas_override.csv")

    assert types_of(schema)["fecha"] == DataType.VARCHAR
    assert types_of(schema)["hora"] == DataType.VARCHAR
    assert types_of(schema)["monto"] == DataType.DOUBLE_PRECISION


def test_infer_schema_quoted_fields():
    schema = infer_schema("t", DATA / "comillas.csv")

    assert types_of(schema) == {"id": DataType.INTEGER, "comentario": DataType.VARCHAR}


def test_infer_schema_varchar_size_buckets():
    schema = infer_schema("t", DATA / "grande.csv")

    # "item_4999" -> 9 chars -> bucket 16
    assert schema.get_column("label").size == 16
    assert types_of(schema)["x"] == DataType.INTEGER
    assert types_of(schema)["y"] == DataType.DOUBLE_PRECISION


def test_infer_schema_strips_utf8_bom_from_header():
    schema = infer_schema("t", DATA / "con_bom.csv")

    assert [c.name for c in schema.columns] == ["id", "nombre"]


def test_prepare_import_creates_nothing(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("alumnos.csv"), "tmp")

    assert isinstance(schema, Schema)
    assert not catalog.table_exists("tmp")


# ---------------------------------------------------------------------------
# infer_schema: errores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "file_name, error",
    [
        ("vacio.csv", EmptyCSVError),
        ("solo_cabecera.csv", EmptyCSVError),
        ("fila_inconsistente.csv", InconsistentRowError),
        ("columnas_duplicadas.csv", DuplicateColumnNameError),
    ],
)
def test_infer_schema_errors(file_name, error):
    with pytest.raises(error):
        infer_schema("t", DATA / file_name)


def test_inconsistent_row_error_reports_line_number():
    with pytest.raises(InconsistentRowError, match="Fila 3"):
        infer_schema("t", DATA / "fila_inconsistente.csv")


# ---------------------------------------------------------------------------
# load_csv: flujo completo
# ---------------------------------------------------------------------------

def test_load_csv_end_to_end(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("alumnos.csv"), "alumnos")

    count = load_csv(catalog, csv("alumnos.csv"), schema)

    assert count == 4
    assert catalog.table_exists("alumnos")
    assert rows_of(catalog, "alumnos") == [
        [1, "Daniela", 21, 17.5, True],
        [2, "Valentin", 22, 15.25, False],
        [3, "Mariana", 20, 18.0, True],
        [4, "Jose", 23, 12.75, False],
    ]


def test_load_csv_where_on_imported_table(tmp_path):
    catalog = make_catalog(tmp_path)
    load_csv(catalog, csv("alumnos.csv"), prepare_import(csv("alumnos.csv"), "alumnos"))

    rows = execute("SELECT nombre FROM alumnos WHERE edad > 21;", catalog)

    assert sorted(r.values[0].data for r in rows) == ["Jose", "Valentin"]


def test_load_csv_empty_cells_become_null(tmp_path):
    catalog = make_catalog(tmp_path)
    load_csv(catalog, csv("productos_nulls.csv"), prepare_import(csv("productos_nulls.csv"), "p"))

    assert rows_of(catalog, "p") == [
        [1, 10.0, 5, "Lapiz"],
        [2, 12.5, None, "Cuaderno A4"],
        [3, None, 0, None],
        [4, 7.0, 3, "Borrador blanco"],
    ]


def test_load_csv_quoted_fields(tmp_path):
    catalog = make_catalog(tmp_path)
    load_csv(catalog, csv("comillas.csv"), prepare_import(csv("comillas.csv"), "c"))

    assert rows_of(catalog, "c") == [
        [1, "Hola, mundo"],
        [2, 'Dijo "hola"'],
        [3, "linea1\nlinea2"],
    ]


def test_load_csv_renamed_table(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("alumnos.csv"), "placeholder")
    schema.table_name = "estudiantes"

    load_csv(catalog, csv("alumnos.csv"), schema)

    assert catalog.table_exists("estudiantes")
    assert not catalog.table_exists("placeholder")


def test_load_csv_large_file(tmp_path):
    catalog = make_catalog(tmp_path)
    count = load_csv(catalog, csv("grande.csv"), prepare_import(csv("grande.csv"), "g"))

    assert count == 5000
    rows = rows_of(catalog, "g")
    assert len(rows) == 5000
    assert rows[-1][0] == 4999 and rows[-1][3] == "item_4999"


def test_load_csv_existing_table_fails(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("alumnos.csv"), "alumnos")
    load_csv(catalog, csv("alumnos.csv"), schema)

    with pytest.raises(TableAlreadyExistsError):
        load_csv(catalog, csv("alumnos.csv"), schema)


# ---------------------------------------------------------------------------
# load_csv: overrides
# ---------------------------------------------------------------------------

def test_override_to_temporal_bytea_numeric(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("fechas_override.csv"), "eventos"),
        fecha=Column("fecha", DataType.DATE),
        hora=Column("hora", DataType.TIME),
        creado=Column("creado", DataType.TIMESTAMP),
        hex=Column("hex", DataType.BYTEA),
        monto=Column("monto", DataType.NUMERIC),
    )

    load_csv(catalog, csv("fechas_override.csv"), schema)

    assert types_of(catalog.get_schema("eventos"))["fecha"] == DataType.DATE
    assert rows_of(catalog, "eventos") == [
        [1, date(2024, 1, 15), time(8, 30), datetime(2024, 1, 15, 8, 30),
         bytes.fromhex("deadbeef"), Decimal("19.99")],
        [2, date(2025, 12, 31), time(23, 59, 59), datetime(2025, 12, 31, 23, 59, 59),
         bytes.fromhex("00ff"), Decimal("0.10")],
    ]


def test_override_integer_to_bigint_and_bool_to_varchar(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("alumnos.csv"), "alumnos"),
        id=Column("id", DataType.BIGINT),
        activo=Column("activo", DataType.VARCHAR, size=8),
    )

    load_csv(catalog, csv("alumnos.csv"), schema)

    assert [r[4] for r in rows_of(catalog, "alumnos")] == ["yes", "no", "yes", "no"]


def test_override_primary_key(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("alumnos.csv"), "alumnos"),
        id=Column("id", DataType.INTEGER, is_primary_key=True),
    )

    load_csv(catalog, csv("alumnos.csv"), schema)

    assert catalog.get_schema("alumnos").primary_key().name == "id"
    rows = execute("SELECT nombre FROM alumnos WHERE id = 3;", catalog)
    assert [r.values[0].data for r in rows] == ["Mariana"]


def test_override_primary_key_rejects_duplicates(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("pk_duplicada.csv"), "t"),
        id=Column("id", DataType.INTEGER, is_primary_key=True),
    )

    with pytest.raises(Exception):
        load_csv(catalog, csv("pk_duplicada.csv"), schema)


# ---------------------------------------------------------------------------
# load_csv: errores de conversión
# ---------------------------------------------------------------------------

def test_override_with_invalid_value_reports_row(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("fecha_invalida.csv"), "t"),
        fecha=Column("fecha", DataType.DATE),
    )

    with pytest.raises(UnsupportedTypeError, match="Fila 3.*no-es-fecha"):
        load_csv(catalog, csv("fecha_invalida.csv"), schema)


def test_override_varchar_to_integer_fails(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("alumnos.csv"), "alumnos"),
        nombre=Column("nombre", DataType.INTEGER),
    )

    with pytest.raises(UnsupportedTypeError, match="Fila 2"):
        load_csv(catalog, csv("alumnos.csv"), schema)


def test_override_varchar_size_too_small_is_rejected(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("alumnos.csv"), "alumnos"),
        nombre=Column("nombre", DataType.VARCHAR, size=3),
    )

    with pytest.raises(Exception):
        load_csv(catalog, csv("alumnos.csv"), schema)
