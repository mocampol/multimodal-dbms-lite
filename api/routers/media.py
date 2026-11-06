"""
Importa una carpeta local del servidor como tabla multimodal. La lógica
vive en loader.media_loader; este router solo la expone por HTTP.
"""

from fastapi import APIRouter

from engine import catalog
from schemas import MediaImportRequest

from loader import load_media_directory

router = APIRouter()


# def (no async): FastAPI la corre en un hilo aparte, así construir el
# índice de una carpeta grande no bloquea las demás peticiones.
@router.post("/import/media")
def import_media(payload: MediaImportRequest):
    return load_media_directory(
        catalog,
        payload.directory,
        payload.table_name,
        payload.kind,
        payload.index,
    )
