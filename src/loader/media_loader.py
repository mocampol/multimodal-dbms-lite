"""
Carga una carpeta local como tabla multimodal: cada archivo es una fila y,
al terminar, se crea el índice pedido con el mismo camino que CREATE INDEX.

El índice se crea después de insertar todas las filas, así IVF y HNSW
entrenan su extractor con la carpeta completa.
"""

import re
from pathlib import Path

from catalog.exceptions import TableAlreadyExistsError
from common.record import Record
from common.schema import Column, Schema
from common.value import DataType, Value

from .csv_loader import _insert_one_record, _transaction_manager
from .exceptions import EmptyMediaDirectoryError, MediaDirectoryNotFoundError
from .type_inference import resolve_varchar_size

MEDIA_EXTENSIONS = {
    "image": (".jpg", ".jpeg", ".png"),
    "audio": (".wav", ".mp3"),
    "text": (".txt",),
}

MEDIA_INDEXES = {
    "image": ("IVF", "HNSW"),
    "audio": ("IVF", "HNSW"),
    "text": ("SPIMI",),
}

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_media_directory(catalog, directory: str, table_name: str, kind: str, index: str | None = None) -> dict:
    """
    Crea la tabla `table_name` con un archivo por fila:
        image/audio -> (id, nombre, ruta) con la ruta absoluta del archivo
        text        -> (id, titulo, contenido) con el texto del .txt

    Los archivos vacíos, ilegibles o que no caben en una fila se omiten y se
    reportan en "skipped". Si algo falla después de crear la tabla, la tabla
    se borra para que el usuario pueda reintentar con el mismo nombre.
    """
    index = _check_request(catalog, directory, table_name, kind, index)
    folder = _check_directory(directory)

    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS[kind]
    )
    rows, skipped = _read_files(files, kind)
    if not rows:
        raise EmptyMediaDirectoryError(_empty_message(folder, kind, skipped))

    schema = _media_schema(table_name, kind, rows)
    catalog.create_table(schema)
    try:
        inserted = _insert_rows(catalog, table_name, kind, rows, skipped)
        if inserted == 0:
            raise EmptyMediaDirectoryError(_empty_message(folder, kind, skipped))
        index_entry = _create_index(catalog, table_name, kind, index) if index else None
    except BaseException:
        catalog.drop_table(table_name)
        raise

    return {
        "table_name": table_name,
        "kind": kind,
        "directory": str(folder),
        "inserted": inserted,
        "skipped": skipped,
        "index": index_entry,
    }


def _check_request(catalog, directory, table_name, kind, index) -> str | None:
    if kind not in MEDIA_EXTENSIONS:
        valid = ", ".join(MEDIA_EXTENSIONS)
        raise ValueError(f"Tipo de archivo desconocido: {kind!r} (válidos: {valid})")

    if not _IDENTIFIER.match(table_name or ""):
        raise ValueError(
            f"Nombre de tabla inválido: {table_name!r} (solo letras, dígitos y _, "
            "sin empezar por dígito)"
        )
    if catalog.table_exists(table_name):
        raise TableAlreadyExistsError(f"La tabla '{table_name}' ya existe")

    if index is None:
        return None
    index = index.upper()
    if index not in MEDIA_INDEXES[kind]:
        valid = " o ".join(MEDIA_INDEXES[kind])
        raise ValueError(f"El índice {index} no sirve para archivos de tipo {kind}: usa {valid}")
    return index


def _check_directory(directory: str) -> Path:
    # Path("") es la carpeta actual: una ruta vacía se rechaza antes
    if not directory or not directory.strip():
        raise ValueError("Indica la carpeta que quieres importar")
    folder = Path(directory.strip()).expanduser()
    if not folder.is_dir():
        raise MediaDirectoryNotFoundError(f"La carpeta '{directory}' no existe o no es una carpeta")
    return folder.resolve()


def _read_files(files: list[Path], kind: str) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Devuelve las filas (archivo, nombre o título, ruta o contenido) y los archivos omitidos."""
    rows, skipped = [], []
    for path in files:
        try:
            content = _read_content(path, kind)
        except OSError as exc:
            skipped.append({"file": path.name, "reason": f"No se pudo leer el archivo: {exc.strerror or exc}"})
            continue
        if content is None:
            skipped.append({"file": path.name, "reason": "El archivo está vacío"})
            continue
        label = path.stem if kind == "text" else path.name
        rows.append((path.name, label, content))
    return rows, skipped


def _read_content(path: Path, kind: str) -> str | None:
    """El texto del .txt o la ruta del archivo multimedia; None si el archivo está vacío."""
    if kind == "text":
        data = path.read_bytes()
        if not data.strip():
            return None
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("latin-1")

    # solo se comprueba que se pueda leer: el contenido lo procesa el índice
    with open(path, "rb") as file:
        if not file.read(1):
            return None
    return str(path)


def _media_schema(table_name: str, kind: str, rows: list[tuple[str, str, str]]) -> Schema:
    label_size = resolve_varchar_size(max(len(label) for _, label, _ in rows))
    if kind == "text":
        columns = [
            Column("titulo", DataType.VARCHAR, size=label_size),
            Column("contenido", DataType.TEXT),
        ]
    else:
        path_size = resolve_varchar_size(max(len(content) for _, _, content in rows))
        columns = [
            Column("nombre", DataType.VARCHAR, size=label_size),
            Column("ruta", DataType.VARCHAR, size=path_size),
        ]
    return Schema(table_name, [Column("id", DataType.INTEGER, is_primary_key=True), *columns])


def _insert_rows(catalog, table_name: str, kind: str, rows, skipped: list[dict]) -> int:
    content_type = DataType.TEXT if kind == "text" else DataType.VARCHAR
    manager = _transaction_manager(catalog)
    manager.begin()
    manager.add_flush(("table", table_name), lambda: catalog.flush_table(table_name))
    inserted = 0
    try:
        for file_name, label, content in rows:
            record = Record([
                Value(DataType.INTEGER, inserted + 1),
                Value(DataType.VARCHAR, label),
                Value(content_type, content),
            ])
            try:
                _insert_one_record(catalog, manager, table_name, record)
            except ValueError as exc:
                # p. ej. un .txt que no cabe en una página del heap
                skipped.append({"file": file_name, "reason": str(exc)})
                continue
            inserted += 1
        manager.commit()
    except BaseException:
        manager.abort()
        raise
    return inserted


def _create_index(catalog, table_name: str, kind: str, index: str) -> dict:
    column = "contenido" if kind == "text" else "ruta"
    try:
        return catalog.create_index(table_name, column, index.lower())
    except Exception as exc:
        raise ValueError(f"No se pudo crear el índice {index} sobre '{column}': {exc}") from exc


def _empty_message(folder: Path, kind: str, skipped: list[dict]) -> str:
    extensions = ", ".join(MEDIA_EXTENSIONS[kind])
    message = f"La carpeta '{folder}' no tiene archivos válidos de tipo {kind} ({extensions})"
    if skipped:
        message += f": se omitieron {len(skipped)} archivo(s), p. ej. {skipped[0]['file']}: {skipped[0]['reason']}"
    return message
