"""
apps/audio-copyright/backend/detection_api.py — API de detección de copyright en audio
(issue #113).

POST /detections recibe una canción como cuerpo crudo de la petición, la divide en
segmentos, busca los candidatos de cada uno en el DBMS (segment_search.py) y los agrupa
en coincidencias (detection.py). Responde siempre con un veredicto explícito:
"posible_coincidencia" o "sin_coincidencias".

No se llama main.py porque los módulos de la app agregan src/ al sys.path, donde ya hay
un main.py: los dos chocarían al importar.

Uso:
    uvicorn detection_api:app --app-dir apps/audio-copyright/backend --port 8001
    curl --data-binary @remix.mp3 "http://127.0.0.1:8001/detections?filename=remix.mp3"
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for path in (APP_DIR.parents[1] / "src", APP_DIR / "features", APP_DIR / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from detection import DetectionSettings, detect  # noqa: E402
from segment_extractor import SAMPLE_RATE, WINDOW_S  # noqa: E402
from segment_search import INDEX_TYPES, SegmentSearcher  # noqa: E402

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
MAX_UPLOAD_MB = float(os.getenv("AUDIO_MAX_UPLOAD_MB", "25"))
MAX_DURATION_S = float(os.getenv("AUDIO_MAX_DURATION_S", "600"))
MAX_K = 50


def decode_audio(data: bytes, filename: str) -> np.ndarray:
    """Señal mono a SAMPLE_RATE. librosa usa soundfile y, para formatos que este no
    lee (como m4a), audioread con ffmpeg, que necesita un archivo en disco."""
    import librosa

    try:
        signal, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
        return signal
    except Exception:
        pass
    import tempfile

    descriptor, path = tempfile.mkstemp(suffix=Path(filename).suffix.lower())
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return signal
    finally:
        os.remove(path)


def create_app(searcher: SegmentSearcher | None = None, settings: DetectionSettings | None = None) -> FastAPI:
    """searcher y settings se pueden inyectar para probar sin la API del DBMS."""
    state = {"searcher": searcher, "error": None}
    default_settings = settings or DetectionSettings()

    def get_searcher() -> SegmentSearcher:
        if state["searcher"] is None:
            try:
                state["searcher"] = SegmentSearcher.from_settings()
                state["error"] = None
            except FileNotFoundError as error:
                state["error"] = str(error)
                raise HTTPException(status_code=503, detail=str(error)) from error
        return state["searcher"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["searcher"] is None:
            try:
                get_searcher()
            except HTTPException:
                pass  # /health lo informa; /detections responde 503 hasta que exista el codebook
        yield

    app = FastAPI(title="Detección de copyright en audio", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.getenv("AUDIO_CORS_ORIGINS", "*").split(",")],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        ready = state["searcher"] is not None
        return {"status": "ok" if ready else "sin_codebook", "ready": ready, "error": state["error"],
                "settings": default_settings.__dict__}

    @app.post("/detections")
    async def create_detection(
        request: Request,
        filename: str = Query("audio.mp3", min_length=1, max_length=255),
        k: int | None = Query(None, ge=1, le=MAX_K),
        index: str | None = Query(None, description=" o ".join(INDEX_TYPES)),
        threshold: float | None = Query(None, gt=0.0, le=1.0),
        min_segments: int | None = Query(None, ge=1),
    ):
        if Path(filename).suffix.lower() not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Formato no admitido; usa {', '.join(AUDIO_EXTENSIONS)}")
        if index is not None and index.lower() not in INDEX_TYPES:
            raise HTTPException(status_code=422, detail=f"Índice desconocido: {index!r} (usa {' o '.join(INDEX_TYPES)})")
        data = await _read_body(request)
        # Decodificar y buscar tarda: en un hilo aparte, para no bloquear las demás peticiones.
        return await run_in_threadpool(process, data, filename, k, index, threshold, min_segments)

    def process(data: bytes, filename: str, k, index, threshold, min_segments) -> dict:
        try:
            signal = decode_audio(data, filename)
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"No se pudo leer el audio: {type(error).__name__}") from error
        duration = len(signal) / SAMPLE_RATE
        if duration < WINDOW_S:
            raise HTTPException(status_code=422,
                                detail=f"El audio dura {duration:.1f} s; se necesitan al menos {WINDOW_S:g} s")
        if duration > MAX_DURATION_S:
            raise HTTPException(status_code=413,
                                detail=f"El audio dura {duration:.0f} s; el máximo es {MAX_DURATION_S:g} s")

        settings = DetectionSettings(
            threshold=default_settings.threshold if threshold is None else threshold,
            min_segments=default_settings.min_segments if min_segments is None else min_segments,
            max_gap=default_settings.max_gap,
            offset_tolerance_s=default_settings.offset_tolerance_s,
            margin=default_settings.margin,
        )
        try:
            search = get_searcher().search_audio(signal, k=k, index_type=index)
        except ValueError as error:   # audio en silencio, parámetros inválidos
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:  # el DBMS respondió con error
            raise HTTPException(status_code=502, detail=str(error)) from error
        return {"filename": filename, **detect(search, settings)}

    return app


async def _read_body(request: Request) -> bytes:
    """El cuerpo completo; corta apenas supera el límite, sin leer el resto."""
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"El archivo supera {MAX_UPLOAD_MB:g} MB")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="El cuerpo de la petición está vacío: envía el audio")
    return b"".join(chunks)


app = create_app()
