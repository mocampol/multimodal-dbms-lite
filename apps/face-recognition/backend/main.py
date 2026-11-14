from __future__ import annotations

import base64
import binascii
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import anyio
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import FaceAppSettings
from .dbms_client import DBMSClient
from .exceptions import (
    ConsentRequiredError,
    DBMSClientError,
    IdentityNotFoundError,
    IdentityWithdrawnError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from .extractor import ExtractedFace, FaceVectorExtractor, build_extractor
from .service import FaceRecognitionService


class _LazyExtractorProxy:
    """Inicializa el extractor real solo al procesar una imagen si no fue inyectado."""

    def __init__(self, settings: FaceAppSettings) -> None:
        self._settings = settings
        self._delegate: FaceVectorExtractor | None = None
        self.model_name = settings.extractor_backend
        self.embedding_dim = settings.resolved_dim()

    def _get_delegate(self) -> FaceVectorExtractor:
        if self._delegate is None:
            self._delegate = build_extractor(self._settings)
            self.model_name = self._delegate.model_name
            self.embedding_dim = self._delegate.embedding_dim
        return self._delegate

    def extract_primary_face(
        self,
        source: str | Path | bytes | bytearray,
        policy: str = "largest",
    ) -> tuple[ExtractedFace, int]:
        return self._get_delegate().extract_primary_face(source, policy=policy)


class IdentityCreateRequest(BaseModel):
    source: Literal["participant", "public_lfw", "public_celeba"] = "participant"
    display_label: str | None = Field(default=None, max_length=128)
    source_subject_id: str | None = Field(default=None, max_length=128)
    consent_granted: bool = False
    purpose: str = Field(default="facial_recognition_demo", max_length=128)
    notice_version: str = Field(default="v1.0", max_length=32)
    allow_pending: bool = False


class RebuildIndexRequest(BaseModel):
    index_type: Literal["hnsw", "ivf"] | None = None
    dim: int | None = Field(default=None, gt=0)
    n_lists: int | None = Field(default=None, gt=0)
    m: int | None = Field(default=None, gt=0)
    ef_construction: int | None = Field(default=None, gt=0)
    ef_search: int | None = Field(default=None, gt=0)


def _parse_multipart_form_bytes(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parser binario seguro para multipart/form-data sin alterar bytes de imágenes."""
    match = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Cabecera multipart/form-data sin boundary")
    boundary = (match.group(1) or match.group(2)).encode("latin-1")
    delimiter = b"--" + boundary

    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for chunk in body.split(delimiter)[1:]:
        if chunk.startswith(b"--"):
            break
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        header_bytes, sep, payload = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        header_text = header_bytes.decode("latin-1")
        name_m = re.search(r'name="([^"]+)"', header_text)
        if not name_m:
            continue
        field_name = name_m.group(1)
        file_m = re.search(r'filename="([^"]*)"', header_text)
        if file_m is not None:
            files[field_name] = (file_m.group(1), payload)
        else:
            fields[field_name] = payload.decode("utf-8")
    return fields, files


async def _extract_upload_payload(request: Request) -> tuple[bytes, str | None, dict[str, Any]]:
    """Acepta multipart/form-data ('file') o application/json ('image_b64')."""
    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()

    if "multipart/form-data" in content_type.lower():
        fields, files = _parse_multipart_form_bytes(content_type, raw_body)
        if "file" not in files:
            raise ValueError("Falta el campo de archivo 'file' en multipart/form-data")
        filename, image_bytes = files["file"]
        if not image_bytes:
            raise ValueError("El archivo de imagen está vacío")
        return image_bytes, filename or None, dict(fields)

    if "application/json" in content_type.lower():
        data = await request.json()
        if not isinstance(data, dict) or not data.get("image_b64"):
            raise ValueError("Debes enviar 'image_b64' en payload JSON")
        try:
            image_bytes = base64.b64decode(data["image_b64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image_b64 inválido: {exc}") from exc
        if not image_bytes:
            raise ValueError("El archivo de imagen está vacío")
        return image_bytes, data.get("filename"), data

    raise ValueError("Content-Type no soportado: usa multipart/form-data o application/json")


def _opt_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    return int(val)


def _opt_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _opt_bool(val: Any, default: bool = False) -> bool:
    if val is None or val == "":
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "si", "on")


def create_app(
    *,
    settings: FaceAppSettings | None = None,
    client: DBMSClient | None = None,
    extractor: FaceVectorExtractor | None = None,
) -> FastAPI:
    """Crea la aplicación FastAPI independiente para reconocimiento facial."""
    cfg = settings or FaceAppSettings.from_env()
    owns_client = client is None
    dbms_client = client or DBMSClient(base_url=cfg.dbms_api_url)
    face_extractor = extractor or _LazyExtractorProxy(cfg)
    service = FaceRecognitionService(cfg, dbms_client, face_extractor)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owns_client:
            dbms_client.close()

    app = FastAPI(
        title="Multimodal DBMS — Face Recognition App",
        description="Aplicación independiente de enrolamiento y reconocimiento facial sobre multimodal-dbms-lite.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = cfg
    app.state.dbms_client = dbms_client
    app.state.service = service

    @app.exception_handler(ConsentRequiredError)
    async def _consent_required(_request: Request, exc: ConsentRequiredError):
        return JSONResponse(status_code=403, content={"error": str(exc), "code": "consent_required"})

    @app.exception_handler(IdentityNotFoundError)
    async def _identity_not_found(_request: Request, exc: IdentityNotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc), "code": "identity_not_found"})

    @app.exception_handler(IdentityWithdrawnError)
    async def _identity_withdrawn(_request: Request, exc: IdentityWithdrawnError):
        return JSONResponse(status_code=409, content={"error": str(exc), "code": "identity_withdrawn"})

    @app.exception_handler(NoFaceDetectedError)
    async def _no_face(_request: Request, exc: NoFaceDetectedError):
        return JSONResponse(status_code=422, content={"error": str(exc), "code": "no_face_detected"})

    @app.exception_handler(MultipleFacesDetectedError)
    async def _multiple_faces(_request: Request, exc: MultipleFacesDetectedError):
        return JSONResponse(status_code=422, content={"error": str(exc), "code": "multiple_faces_detected"})

    @app.exception_handler(DBMSClientError)
    async def _dbms_error(_request: Request, exc: DBMSClientError):
        code = exc.status_code if exc.status_code >= 400 else 502
        return JSONResponse(status_code=code, content={"error": str(exc), "code": "dbms_error"})

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc), "code": "invalid_request"})

    @app.get("/health")
    def health():
        try:
            dbms_health = dbms_client.health()
            dbms_ok = True
        except DBMSClientError as exc:
            dbms_health = {"error": str(exc)}
            dbms_ok = False
        return {
            "status": "ok" if dbms_ok else "degraded",
            "service": "face-recognition",
            "model": face_extractor.model_name,
            "embedding_dim": face_extractor.embedding_dim,
            "defaults": {
                "index_type": cfg.default_index_type,
                "metric": cfg.default_metric,
                "k": cfg.default_k,
                "threshold": cfg.default_threshold,
            },
            "dbms": dbms_health,
        }

    @app.post("/identities", status_code=201)
    def register_identity(payload: IdentityCreateRequest):
        return service.register_identity(
            source=payload.source,
            display_label=payload.display_label,
            source_subject_id=payload.source_subject_id,
            consent_granted=payload.consent_granted,
            purpose=payload.purpose,
            notice_version=payload.notice_version,
            allow_pending=payload.allow_pending,
        )

    @app.get("/identities")
    def list_identities(
        source: str | None = Query(default=None),
        consent_status: str | None = Query(default=None),
    ):
        items = service.list_identities(source=source, consent_status=consent_status)
        return {"identities": items, "count": len(items)}

    @app.delete("/identities/{identity_id}")
    def withdraw_identity(identity_id: int):
        return service.withdraw_identity(identity_id)

    @app.delete("/identities")
    def purge_all_identities(samples_only: bool = Query(default=False)):
        return service.purge_all_faces(samples_only=samples_only)

    @app.post("/enroll", status_code=201)
    async def enroll_face(request: Request):
        raw_bytes, filename, fields = await _extract_upload_payload(request)
        if "identity_id" not in fields:
            raise ValueError("El campo 'identity_id' es obligatorio")
        identity_id = int(fields["identity_id"])
        split = str(fields.get("split") or "enroll")
        face_policy = str(fields.get("face_policy") or "strict_single")
        return await anyio.to_thread.run_sync(
            lambda: service.enroll_face(
                identity_id=identity_id,
                image_bytes=raw_bytes,
                relative_path=filename or f"upload_{identity_id}.jpg",
                split=split,
                face_policy=face_policy,
            )
        )

    @app.post("/recognize")
    async def recognize_face(request: Request):
        raw_bytes, _filename, fields = await _extract_upload_payload(request)
        return await anyio.to_thread.run_sync(
            lambda: service.recognize_face(
                image_bytes=raw_bytes,
                k=_opt_int(fields.get("k")),
                threshold=_opt_float(fields.get("threshold")),
                index_type=fields.get("index_type") or None,
                metric=fields.get("metric") or None,
                ef_search=_opt_int(fields.get("ef_search")),
                n_probe=_opt_int(fields.get("n_probe")),
                aggregation=str(fields.get("aggregation") or "max"),
                allow_dataset_identities=_opt_bool(
                    fields.get("allow_dataset_identities"), False
                ),
                face_policy=str(fields.get("face_policy") or "largest"),
            )
        )

    @app.post("/indexes/rebuild")
    def rebuild_index(payload: RebuildIndexRequest):
        return service.rebuild_vector_index(
            index_type=payload.index_type,
            dim=payload.dim,
            n_lists=payload.n_lists,
            m=payload.m,
            ef_construction=payload.ef_construction,
            ef_search=payload.ef_search,
        )

    return app


app = create_app()
