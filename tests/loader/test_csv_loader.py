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
from common.record import Record
from common.schema import Column, Schema
from common.value import DataType, Point, Polygon, Rectangle, Value
from loader import (
    DuplicateColumnNameError,
    EmptyCSVError,
    InconsistentRowError,
    UnsupportedTypeError,
    infer_schema,
    load_csv,
    prepare_import,
)
from loader.csv_loader import _run_batch, _transaction_manager
from loader.type_inference import convert_value
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute
from transaction import TransactionManager


DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module", autouse=True)
def csv_test_data(tmp_path_factory):
    global DATA
    DATA = tmp_path_factory.mktemp("csv-loader-data")
    fixtures = {
        "alumnos.csv": (
            "id,nombre,edad,promedio,activo\n"
            "1,Daniela,21,17.5,yes\n"
            "2,Valentin,22,15.25,no\n"
            "3,Mariana,20,18.0,yes\n"
            "4,Jose,23,12.75,no\n"
        ),
        "productos_nulls.csv": (
            "codigo,precio,stock,descripcion\n"
            "1,10,5,Lapiz\n"
            "2,12.5,,Cuaderno A4\n"
            "3,,0,\n"
            "4,7,3,Borrador blanco\n"
        ),
        "primera_celda_vacia.csv": "id,edad,nota\n1,,\n2,30,15\n3,25,\n",
        "widening.csv": "id,valor,flag,mixto\n1,10,true,10\n2,2.5,false,abc\n3,,T,\n",
        "fechas_override.csv": (
            "id,fecha,hora,creado,hex,monto\n"
            "1,2024-01-15,08:30:00,2024-01-15 08:30:00,deadbeef,19.99\n"
            "2,2025-12-31,23:59:59,2025-12-31 23:59:59,00ff,0.10\n"
        ),
        "comillas.csv": (
            'id,comentario\n1,"Hola, mundo"\n2,"Dijo ""hola"""\n'
            '3,"linea1\nlinea2"\n'
        ),
        "con_bom.csv": "\ufeffid,nombre\n1,Ana\n",
        "vacio.csv": "",
        "solo_cabecera.csv": "id,nombre\n",
        "fila_inconsistente.csv": "id,nombre\n1,Ana\n2\n",
        "columnas_duplicadas.csv": "id,id\n1,2\n",
        "pk_duplicada.csv": "id,nombre\n1,Ana\n1,Luis\n",
        "fecha_invalida.csv": "id,fecha\n1,2024-01-01\n2,no-es-fecha\n",
        "columna_vacia.csv": "id,vacia,nombre\n1,,Ana\n2,,Luis\n",
        "enteros_rango.csv": (
            "id,chico,grande\n1,32767,9223372036854775807\n"
            "2,-32768,-9223372036854775808\n"
        ),
        "smallint_overflow.csv": "id,chico\n1,10\n2,40000\n",
        "reales.csv": "id,r\n1,0.5\n2,0.1\n",
        "booleanos_override.csv": "id,flag\n1,1\n2,0\n3,yes\n4,no\n",
        "errores_mixtos.csv": (
            "id,nombre,fecha\n"
            "1,Ana,2024-01-01\n"
            "2,Luis,no-es-fecha\n"
            "3,Eva,2024-03-03\n"
            "4,Jose\n"
            "1,Duplicado,2024-04-04\n"
            "6,Este nombre excede dieciseis,2024-06-06\n"
            "6,Luisa,2024-06-06\n"
        ),
        "espaciales.csv": (
            "id,ubicacion,zona,caja,forma\n"
            '1,POINT(-77.03 -12.12),"POLYGON((-77.05 -12.14, -77.01 -12.14, -77.01 -12.10, -77.05 -12.14))",'
            "RECTANGLE(-77.05 -12.14 -77.01 -12.10),POINT(-77.03 -12.12)\n"
            '2,"POINT(-77.02, -12.13)","POLYGON(POINT(-77.0, -12.0), POINT(-76.9, -12.0), POINT(-76.9, -11.9))",'
            '"RECTANGLE(-77.0, -12.0, -76.9, -11.9)","POLYGON((-77.0 -12.0, -76.9 -12.0, -76.9 -11.9))"\n'
            "3,,,,\n"
        ),
    }
    for name, contents in fixtures.items():
        (DATA / name).write_text(contents, encoding="utf-8")

    with (DATA / "grande.csv").open("w", encoding="utf-8") as output:
        output.write("id,x,y,label\n")
        for row_id in range(5_000):
            output.write(f"{row_id},{row_id},{row_id / 10},item_{row_id}\n")


def csv(name: str) -> str:
    return str(DATA / name)


def make_catalog(base_dir):
    heap_factory = make_heap_factory(str(base_dir))
    catalog = Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )
    # WAL aislado por test: sin esto se usaría data/transactions.wal del repo.
    catalog._transaction_manager = TransactionManager(log_path=str(base_dir / "test.wal"))
    return catalog


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

    result = load_csv(catalog, csv("alumnos.csv"), schema)

    assert result == {"inserted": 4, "skipped": []}
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
    result = load_csv(
        catalog, csv("grande.csv"), prepare_import(csv("grande.csv"), "g"), batch_size=700
    )

    assert result["inserted"] == 5000
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

    with pytest.raises(ValueError, match="duplicado.*'id'"):
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

    with pytest.raises(UnsupportedTypeError, match="Fila 2"):
        load_csv(catalog, csv("alumnos.csv"), schema)


# ---------------------------------------------------------------------------
# inferencia: columna 100% vacía
# ---------------------------------------------------------------------------

def test_all_empty_column_defaults_to_varchar_and_loads_nulls(tmp_path):
    schema = infer_schema("t", DATA / "columna_vacia.csv")
    assert types_of(schema)["vacia"] == DataType.VARCHAR
    assert schema.get_column("vacia").size == 32

    catalog = make_catalog(tmp_path)
    load_csv(catalog, csv("columna_vacia.csv"), schema)
    assert rows_of(catalog, "t") == [[1, None, "Ana"], [2, None, "Luis"]]


# ---------------------------------------------------------------------------
# overrides: rangos enteros, REAL, BOOLEAN desde 1/0
# ---------------------------------------------------------------------------

def test_override_integer_ranges_at_limits(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("enteros_rango.csv"), "t"),
        chico=Column("chico", DataType.SMALLINT),
        grande=Column("grande", DataType.BIGINT),
    )

    load_csv(catalog, csv("enteros_rango.csv"), schema)

    assert rows_of(catalog, "t") == [
        [1, 32767, 9223372036854775807],
        [2, -32768, -9223372036854775808],
    ]


def test_override_smallint_overflow_reports_row(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("smallint_overflow.csv"), "t"),
        chico=Column("chico", DataType.SMALLINT),
    )

    with pytest.raises(UnsupportedTypeError, match="Fila 3.*40000 está fuera de rango para smallint"):
        load_csv(catalog, csv("smallint_overflow.csv"), schema)


def test_override_real_rejects_precision_loss(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("reales.csv"), "t"),
        r=Column("r", DataType.REAL),
    )

    # 0.5 es exacto en float32; 0.1 no
    with pytest.raises(UnsupportedTypeError, match="Fila 3.*pierde precisión"):
        load_csv(catalog, csv("reales.csv"), schema)


@pytest.mark.parametrize(
    "raw, data_type, message",
    [
        ("40000", DataType.SMALLINT, "fuera de rango"),
        ("9223372036854775808", DataType.BIGINT, "fuera de rango"),
        ("0.1", DataType.REAL, "pierde precisión"),
        ("1e40", DataType.REAL, "fuera de rango para REAL"),
        ("quizas", DataType.BOOLEAN, "no es un booleano reconocido"),
        ("abc", DataType.INTEGER, "no es válido para el tipo integer"),
        ("2024-13-01", DataType.DATE, "no es válido para el tipo date"),
        ("zz", DataType.BYTEA, "no es válido para el tipo bytea"),
        ("1,5", DataType.NUMERIC, "no es válido para el tipo numeric"),
    ],
)
def test_convert_value_error_messages(raw, data_type, message):
    with pytest.raises(UnsupportedTypeError, match=message):
        convert_value(raw, data_type)


def test_infer_schema_spatial_types():
    schema = infer_schema("t", DATA / "espaciales.csv")

    assert [c.data_type for c in schema.columns] == [
        DataType.INTEGER, DataType.POINT, DataType.POLYGON,
        DataType.RECTANGLE, DataType.GEOMETRY,
    ]


@pytest.mark.parametrize(
    "raw, data_type, expected",
    [
        ("POINT(-77.03 -12.12)", DataType.POINT, Point(-77.03, -12.12)),
        ("point(-77.03, -12.12)", DataType.POINT, Point(-77.03, -12.12)),
        ("-77.03 -12.12", DataType.POINT, Point(-77.03, -12.12)),
        ("-77.03,-12.12", DataType.POINT, Point(-77.03, -12.12)),
        ("RECTANGLE(-1 -2 3 4)", DataType.RECTANGLE, Rectangle(-1, -2, 3, 4)),
        ("POINT(1 2)", DataType.GEOMETRY, Point(1, 2)),
        (
            "POLYGON((0 0, 1 0, 1 1))", DataType.POLYGON,
            Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 0))),
        ),
        (
            "POLYGON(POINT(0, 0), POINT(1, 0), POINT(1, 1), POINT(0, 0))", DataType.GEOMETRY,
            Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 0))),
        ),
    ],
)
def test_convert_value_spatial_formats(raw, data_type, expected):
    assert convert_value(raw, data_type).data == expected


@pytest.mark.parametrize(
    "raw, data_type",
    [
        ("POINT(1)", DataType.POINT),
        ("POINT(1 2 3)", DataType.POINT),
        ("POINT(200 0)", DataType.POINT),
        ("POINT(a b)", DataType.POINT),
        ("POINT(1 2", DataType.POINT),
        ("POLYGON((0 0, 1 1))", DataType.POLYGON),
        ("RECTANGLE(0 0 1)", DataType.RECTANGLE),
        ("POLYGON((0 0, 1 0, 1 1))", DataType.POINT),
    ],
)
def test_convert_value_spatial_errors(raw, data_type):
    with pytest.raises(UnsupportedTypeError):
        convert_value(raw, data_type)


def test_load_csv_spatial_columns_and_rtree_query(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("espaciales.csv"), "lugares"),
        id=Column("id", DataType.INTEGER, is_primary_key=True),
    )

    assert load_csv(catalog, csv("espaciales.csv"), schema) == {"inserted": 3, "skipped": []}
    rows = rows_of(catalog, "lugares")
    assert rows[0][1] == Point(-77.03, -12.12)
    assert rows[1][2].points[-1] == Point(-77.0, -12.0)
    assert rows[2][1:] == [None, None, None, None]

    execute("CREATE INDEX idx_ubicacion ON lugares (ubicacion) USING RTREE;", catalog)
    near = execute(
        "SELECT id FROM lugares WHERE distancia(ubicacion, POINT(-77.03, -12.12), HAVERSINE) < 100;",
        catalog,
    )
    assert [r.values[0].data for r in near] == [1]


def test_override_boolean_accepts_1_0_and_words(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("booleanos_override.csv"), "t")
    assert types_of(schema)["flag"] == DataType.VARCHAR  # 1/0/yes mezclados

    schema = override(schema, flag=Column("flag", DataType.BOOLEAN))
    load_csv(catalog, csv("booleanos_override.csv"), schema)

    assert [r[1] for r in rows_of(catalog, "t")] == [True, False, True, False]


# ---------------------------------------------------------------------------
# on_error / batch_size / transacciones
# ---------------------------------------------------------------------------

def errores_mixtos_schema(table="t"):
    # No se puede usar prepare_import: la fila 5 tiene columnas de menos e
    # infer_schema la rechaza. Se arma el schema como lo haría el frontend.
    return Schema(table, [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("nombre", DataType.VARCHAR, size=16),
        Column("fecha", DataType.DATE),
    ])


def test_on_error_ignore_skips_bad_rows_and_reports_them(tmp_path):
    catalog = make_catalog(tmp_path)

    result = load_csv(catalog, csv("errores_mixtos.csv"), errores_mixtos_schema(), on_error="ignore")

    assert result["inserted"] == 3
    assert [s["row"] for s in result["skipped"]] == [3, 5, 6, 7]
    reasons = {s["row"]: s["reason"] for s in result["skipped"]}
    assert "no-es-fecha" in reasons[3]   # conversión
    assert "2 columnas" in reasons[5]    # fila inconsistente
    assert "duplicado" in reasons[6]     # PK repetida
    assert "VARCHAR" in reasons[7]       # tamaño excedido
    assert [r[0] for r in rows_of(catalog, "t")] == [1, 3, 6]


def test_on_error_ignore_with_batch_size_1(tmp_path):
    catalog = make_catalog(tmp_path)

    result = load_csv(
        catalog, csv("errores_mixtos.csv"), errores_mixtos_schema(),
        on_error="ignore", batch_size=1,
    )

    assert result["inserted"] == 3
    assert [r[0] for r in rows_of(catalog, "t")] == [1, 3, 6]


@pytest.mark.parametrize("batch_size", [1000, 1])
def test_on_error_stop_drops_the_table(tmp_path, batch_size):
    catalog = make_catalog(tmp_path)

    with pytest.raises(UnsupportedTypeError, match="Fila 3"):
        load_csv(
            catalog, csv("errores_mixtos.csv"), errores_mixtos_schema(), batch_size=batch_size
        )

    # Todo-o-nada: aunque con batch_size=1 la fila 2 ya se había
    # commiteado en su propio lote, la tabla desaparece entera.
    assert not catalog.table_exists("t")


def test_failed_import_allows_reimport_with_same_name(tmp_path):
    catalog = make_catalog(tmp_path)
    with pytest.raises(UnsupportedTypeError):
        load_csv(catalog, csv("errores_mixtos.csv"), errores_mixtos_schema("alumnos"))

    result = load_csv(catalog, csv("alumnos.csv"), prepare_import(csv("alumnos.csv"), "alumnos"))

    assert result["inserted"] == 4
    assert len(rows_of(catalog, "alumnos")) == 4


def test_existing_table_is_not_dropped_on_name_clash(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("alumnos.csv"), "alumnos")
    load_csv(catalog, csv("alumnos.csv"), schema)

    with pytest.raises(TableAlreadyExistsError):
        load_csv(catalog, csv("alumnos.csv"), schema)

    assert len(rows_of(catalog, "alumnos")) == 4


def test_failed_batch_is_undone_by_abort(tmp_path):
    # Prueba el undo del lote directamente, sin el drop_table de load_csv
    # que lo taparía.
    catalog = make_catalog(tmp_path)
    schema = errores_mixtos_schema()
    catalog.create_table(schema)
    manager = _transaction_manager(catalog)
    batch = [
        (2, ["1", "Ana", "2024-01-01"]),
        (3, ["2", "Luis", "2024-02-02"]),
        (4, ["3", "Eva", "no-es-fecha"]),
    ]

    with pytest.raises(UnsupportedTypeError, match="Fila 4"):
        _run_batch(catalog, manager, "t", schema, batch, "stop", [])

    assert rows_of(catalog, "t") == []
    assert manager.current() is None
    record = Record([
        Value(DataType.INTEGER, 1),
        Value(DataType.VARCHAR, "Ana"),
        Value(DataType.DATE, date(2024, 1, 1)),
    ])
    assert catalog.check_insert_uniques("t", record) is None


def test_imported_uniques_are_enforced_on_later_sql_insert(tmp_path):
    catalog = make_catalog(tmp_path)
    schema = override(
        prepare_import(csv("alumnos.csv"), "alumnos"),
        id=Column("id", DataType.INTEGER, is_primary_key=True),
    )
    load_csv(catalog, csv("alumnos.csv"), schema)

    with pytest.raises(Exception):
        execute("INSERT INTO alumnos VALUES (2, 'Otro', 30, 10.0, true);", catalog)


def test_load_csv_leaves_no_active_transaction(tmp_path):
    catalog = make_catalog(tmp_path)
    with pytest.raises(UnsupportedTypeError):
        load_csv(catalog, csv("errores_mixtos.csv"), errores_mixtos_schema())

    assert catalog._transaction_manager.current() is None


@pytest.mark.parametrize("kwargs", [{"on_error": "skip"}, {"batch_size": 0}, {"batch_size": -5}])
def test_load_csv_invalid_arguments(tmp_path, kwargs):
    catalog = make_catalog(tmp_path)
    schema = prepare_import(csv("alumnos.csv"), "alumnos")

    with pytest.raises(ValueError):
        load_csv(catalog, csv("alumnos.csv"), schema, **kwargs)

    assert not catalog.table_exists("alumnos")
