"""
Importa una carpeta local del servidor como tabla multimodal y sirve sus
archivos al frontend. La lógica de carga vive en loader.media_loader.

GET /files solo entrega archivos que están directamente dentro de una
carpeta importada. La lista se guarda en data/ para que siga valiendo
después de reiniciar la API.
"""

import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from engine import catalog
from schemas import MediaImportRequest

from loader import load_media_directory

router = APIRouter()

ALLOWED_DIRS_FILE = Path("data") / "media_dirs.json"

# también es la lista de extensiones que se pueden servir
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".txt": "text/plain",
}

_allowed_dirs_lock = threading.Lock()


# def (no async): FastAPI la corre en un hilo aparte, así construir el
# índice de una carpeta grande no bloquea las demás peticiones.
@router.post("/import/media")
def import_media(payload: MediaImportRequest):
    result = load_media_directory(
        catalog,
        payload.directory,
        payload.table_name,
        payload.kind,
        payload.index,
    )
    _allow_directory(Path(result["directory"]))
    return result


@router.get("/files")
def get_file(path: str = Query(..., min_length=1)):
    # resolve() quita los ".." y sigue los enlaces simbólicos, así una ruta
    # como carpeta/../secreto.png se compara por su ubicación real
    file = Path(path).resolve()
    content_type = CONTENT_TYPES.get(file.suffix.lower())
    if content_type is None or file.parent not in _allowed_directories():
        raise HTTPException(status_code=403, detail="La ruta no pertenece a una carpeta importada")
    if not file.is_file():
        raise HTTPException(status_code=404, detail=f"El archivo '{file.name}' ya no existe")
    return FileResponse(file, media_type=content_type)


def _allowed_directories() -> set[Path]:
    try:
        return {Path(p) for p in json.loads(ALLOWED_DIRS_FILE.read_text(encoding="utf-8"))}
    except FileNotFoundError:
        return set()


def _allow_directory(directory: Path) -> None:
    with _allowed_dirs_lock:
        allowed = _allowed_directories()
        if directory in allowed:
            return
        allowed.add(directory)
        ALLOWED_DIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ALLOWED_DIRS_FILE.write_text(
            json.dumps(sorted(str(p) for p in allowed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
