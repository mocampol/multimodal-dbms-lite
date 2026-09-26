import csv
import re
import tempfile
import uuid
from itertools import islice
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from engine import catalog
from schemas import CsvImportConfirm

from common.schema import Column, Schema
from common.value import DataType
from loader import (
    DuplicateColumnNameError,
    EmptyCSVError,
    InconsistentRowError,
    load_csv,
    prepare_import,
)

router = APIRouter(prefix="/import/csv")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "multimodal-dbms-lite-uploads"
PREVIEW_ROWS = 5

_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@router.post("/preview")
async def preview_csv(request: Request, filename: str = Query("tabla.csv")):
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    path = _upload_path(upload_id)
    path.write_bytes(content)

    table_name = _suggest_table_name(filename)
    try:
        schema = prepare_import(str(path), table_name)
        rows = _preview_rows(path)
    except (EmptyCSVError, InconsistentRowError, DuplicateColumnNameError) as exc:
        path.unlink(missing_ok=True)
        # El mensaje trae la ruta temporal interna: se muestra el nombre original.
        raise type(exc)(str(exc).replace(str(path), filename)) from exc
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return {
        "upload_id": upload_id,
        "table_name": table_name,
        "columns": [_column_to_json(c) for c in schema.columns],
        "preview_rows": rows,
        "data_types": [t.value for t in DataType],
    }


@router.post("/confirm")
def confirm_csv(payload: CsvImportConfirm):
    path = _existing_upload(payload.upload_id)

    header = _header(path)
    if len(payload.columns) != len(header):
        raise ValueError(
            f"El CSV tiene {len(header)} columnas pero se enviaron {len(payload.columns)}"
        )

    _check_identifier(payload.table_name, "tabla")
    columns = []
    for col in payload.columns:
        _check_identifier(col.name, "columna")
        columns.append(Column(
            col.name,
            _data_type(col.type),
            size=col.size,
            is_primary_key=col.is_primary_key,
            nullable=col.nullable,
            is_unique=col.is_unique,
        ))
    schema = Schema(payload.table_name, columns)

    result = load_csv(catalog, str(path), schema, on_error=payload.on_error)

    # Si falló, el archivo se conserva para que el usuario corrija el esquema
    # y reintente con el mismo upload_id.
    path.unlink(missing_ok=True)
    return {"table_name": schema.table_name, **result}


@router.delete("/{upload_id}")
def discard_csv(upload_id: str):
    _existing_upload(upload_id).unlink(missing_ok=True)
    return {"ok": True}


def _upload_path(upload_id: str) -> Path:
    return UPLOAD_DIR / f"{upload_id}.csv"


def _existing_upload(upload_id: str) -> Path:
    if not _UPLOAD_ID.match(upload_id):
        raise HTTPException(status_code=400, detail="upload_id inválido")
    path = _upload_path(upload_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="El upload no existe o ya fue importado")
    return path


def _header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f), [])


def _preview_rows(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        return list(islice(reader, PREVIEW_ROWS))


def _suggest_table_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def _data_type(value: str) -> DataType:
    try:
        return DataType(value)
    except ValueError:
        valid = ", ".join(t.value for t in DataType)
        raise ValueError(f"Tipo de dato desconocido: {value!r} (válidos: {valid})") from None


def _check_identifier(name: str, kind: str) -> None:
    if not _IDENTIFIER.match(name):
        raise ValueError(
            f"Nombre de {kind} inválido: {name!r} (solo letras, dígitos y _, "
            "sin empezar por dígito)"
        )


def _column_to_json(column: Column) -> dict:
    return {
        "name": column.name,
        "type": column.data_type.value,
        "size": column.size,
        "nullable": column.nullable,
        "is_unique": column.is_unique,
        "is_primary_key": column.is_primary_key,
    }
