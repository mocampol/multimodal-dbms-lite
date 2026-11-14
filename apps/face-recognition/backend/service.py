from __future__ import annotations

import base64
import hashlib
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cache import IdentityCache
from .config import FaceAppSettings
from .dbms_client import DBMSClient
from .exceptions import (
    ConsentRequiredError,
    IdentityNotFoundError,
    IdentityWithdrawnError,
)
from .extractor import ExtractedFace, FaceVectorExtractor
from .schema_setup import (
    CONSENTS_TABLE,
    IDENTITIES_TABLE,
    SAMPLES_TABLE,
    ensure_face_schema,
)

ALLOWED_SOURCES = ("participant", "public_lfw", "public_celeba")


def _rows_as_dicts(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convierte la respuesta tabular de /query a una lista de diccionarios."""
    columns = query_result.get("columns", [])
    rows = query_result.get("rows", [])
    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            result.append(row)
        else:
            result.append(dict(zip(columns, row)))
    return result


def _sql_str(value: str) -> str:
    """Valida y formatea un literal string seguro para sentencias SQL del DBMS."""
    if any(ch in value for ch in ("'", "\x00", "\n", "\r")):
        raise ValueError("El valor contiene caracteres no permitidos para SQL literal")
    return f"'{value}'"


def _encode_vector_b64(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype="<f4").reshape(-1)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FaceRecognitionService:
    """Servicio de dominio para identidades, consentimiento, enrolamiento y reconocimiento facial."""

    def __init__(
        self,
        settings: FaceAppSettings,
        client: DBMSClient,
        extractor: FaceVectorExtractor,
        cache: IdentityCache | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.extractor = extractor
        self.cache = cache or IdentityCache(
            maxsize=settings.cache_maxsize,
            ttl_seconds=settings.cache_ttl_seconds,
            redis_url=settings.redis_url,
        )
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._schema_ready = False

    def ensure_ready(self) -> dict[str, Any]:
        """Garantiza que las tablas e índices vectoriales faciales existan en el DBMS."""
        with self._lock:
            result = ensure_face_schema(
                self.client,
                self.settings,
                dim=self.extractor.embedding_dim,
            )
            self._schema_ready = True
            return result

    def _ensure_ready_once(self) -> None:
        if not self._schema_ready:
            self.ensure_ready()

    def _next_id(self, table_name: str, pk_col: str) -> int:
        with self._lock:
            if table_name not in self._counters:
                res = self.client.query(f"SELECT {pk_col} FROM {table_name};")
                rows = _rows_as_dicts(res)
                max_id = max(
                    (int(row[pk_col]) for row in rows if row.get(pk_col) is not None),
                    default=0,
                )
                self._counters[table_name] = max_id
            self._counters[table_name] += 1
            return self._counters[table_name]

    def register_identity(
        self,
        *,
        source: str = "participant",
        display_label: str | None = None,
        source_subject_id: str | None = None,
        consent_granted: bool = False,
        purpose: str = "facial_recognition_demo",
        notice_version: str = "v1.0",
        allow_pending: bool = False,
    ) -> dict[str, Any]:
        """Registra una identidad cumpliendo las reglas de origen y consentimiento."""
        self._ensure_ready_once()

        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source inválido: {source!r}. Valores permitidos: {ALLOWED_SOURCES}"
            )

        clean_label = display_label.strip() if isinstance(display_label, str) and display_label.strip() else None

        if source == "participant":
            if not consent_granted and not allow_pending:
                raise ConsentRequiredError(
                    "Se requiere consentimiento explícito (consent_granted=true) para registrar a un participante"
                )
            consent_status = "granted" if consent_granted else "pending"
            subject_id = (
                source_subject_id.strip()
                if isinstance(source_subject_id, str) and source_subject_id.strip()
                else f"subj-{uuid.uuid4().hex[:12]}"
            )
        else:
            # Los datasets públicos nunca llevan consent_status='granted' ni exponen nombres
            consent_status = "dataset_only"
            clean_label = None
            if not source_subject_id or not source_subject_id.strip():
                raise ValueError("source_subject_id es obligatorio para datasets públicos")
            subject_id = source_subject_id.strip()

        identity_id = self._next_id(IDENTITIES_TABLE, "identity_id")
        identity_row = {
            "identity_id": identity_id,
            "source": source,
            "source_subject_id": subject_id,
            "display_label": clean_label,
            "consent_status": consent_status,
        }
        self.client.insert_rows(IDENTITIES_TABLE, [identity_row])

        consent_record: dict[str, Any] | None = None
        if source == "participant" and consent_granted:
            consent_id = self._next_id(CONSENTS_TABLE, "consent_id")
            consent_record = {
                "consent_id": consent_id,
                "identity_id": identity_id,
                "purpose": purpose.strip() or "facial_recognition_demo",
                "granted_at": datetime.now(timezone.utc).isoformat(),
                "notice_version": notice_version.strip() or "v1.0",
                "status": "granted",
                "withdrawn_at": None,
            }
            self.client.insert_rows(CONSENTS_TABLE, [consent_record])

        self.cache.set(identity_id, identity_row)
        return {
            **identity_row,
            "consent": consent_record,
        }

    def get_identity(self, identity_id: int, *, use_cache: bool = True) -> dict[str, Any]:
        """Obtiene una identidad por ID usando caché opcional."""
        self._ensure_ready_once()
        if use_cache:
            cached = self.cache.get(identity_id)
            if cached is not None:
                return dict(cached)

        res = self.client.query(
            "SELECT identity_id, source, source_subject_id, display_label, consent_status "
            f"FROM {IDENTITIES_TABLE} WHERE identity_id = {int(identity_id)};"
        )
        rows = _rows_as_dicts(res)
        if not rows:
            raise IdentityNotFoundError(f"Identidad {identity_id} no encontrada")
        identity = rows[0]
        self.cache.set(identity_id, identity)
        return identity

    def list_identities(
        self,
        *,
        source: str | None = None,
        consent_status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lista identidades registradas junto con el conteo actual de muestras faciales."""
        self._ensure_ready_once()
        res = self.client.query(
            "SELECT identity_id, source, source_subject_id, display_label, consent_status "
            f"FROM {IDENTITIES_TABLE};"
        )
        identities = _rows_as_dicts(res)

        sample_res = self.client.query(f"SELECT sample_id, identity_id FROM {SAMPLES_TABLE};")
        counts: dict[int, int] = {}
        for row in _rows_as_dicts(sample_res):
            iid = row.get("identity_id")
            if iid is not None:
                counts[int(iid)] = counts.get(int(iid), 0) + 1

        filtered: list[dict[str, Any]] = []
        for item in identities:
            if source is not None and item.get("source") != source:
                continue
            if consent_status is not None and item.get("consent_status") != consent_status:
                continue
            iid = int(item["identity_id"])
            filtered.append({**item, "sample_count": counts.get(iid, 0)})
        return filtered

    def withdraw_identity(self, identity_id: int) -> dict[str, Any]:
        """Revoca el consentimiento y elimina todas las muestras faciales asociadas."""
        identity = self.get_identity(identity_id, use_cache=False)
        now = datetime.now(timezone.utc).isoformat()
        iid = int(identity["identity_id"])

        self.client.query(
            f"UPDATE {IDENTITIES_TABLE} SET consent_status = 'withdrawn' "
            f"WHERE identity_id = {iid};"
        )
        self.client.query(
            f"UPDATE {CONSENTS_TABLE} SET status = 'withdrawn' WHERE identity_id = {iid};"
        )
        self.client.query(
            f"UPDATE {CONSENTS_TABLE} SET withdrawn_at = {_sql_str(now)} "
            f"WHERE identity_id = {iid};"
        )
        del_res = self.client.query(f"DELETE FROM {SAMPLES_TABLE} WHERE identity_id = {iid};")
        self.cache.invalidate(iid)

        return {
            "identity_id": iid,
            "consent_status": "withdrawn",
            "deleted_samples": int(del_res.get("rows_affected", 0)),
            "withdrawn_at": now,
        }

    def _count_rows_if_exists(self, table_name: str, pk_col: str, existing: set[str]) -> int:
        if table_name not in existing:
            return 0
        res = self.client.query(f"SELECT {pk_col} FROM {table_name};")
        return len(_rows_as_dicts(res))

    def purge_all_faces(self, *, samples_only: bool = False) -> dict[str, Any]:
        """Elimina todas las muestras faciales (y opcionalmente identidades y consentimientos).

        Usa DROP TABLE + ensure_ready() en lugar de DELETE masivo fila a fila para
        limpiar de forma instantánea y recuperarse incluso si algún índice B-Tree
        quedó desincronizado tras lotes abortados.
        """
        self._ensure_ready_once()
        existing = {item["name"] for item in self.client.list_tables()}

        deleted_samples = self._count_rows_if_exists(SAMPLES_TABLE, "sample_id", existing)
        deleted_consents = 0
        deleted_identities = 0
        if not samples_only:
            deleted_consents = self._count_rows_if_exists(CONSENTS_TABLE, "consent_id", existing)
            deleted_identities = self._count_rows_if_exists(IDENTITIES_TABLE, "identity_id", existing)

        if SAMPLES_TABLE in existing:
            self.client.query(f"DROP TABLE {SAMPLES_TABLE};")
        if not samples_only:
            if CONSENTS_TABLE in existing:
                self.client.query(f"DROP TABLE {CONSENTS_TABLE};")
            if IDENTITIES_TABLE in existing:
                self.client.query(f"DROP TABLE {IDENTITIES_TABLE};")

        self.cache.clear()
        with self._lock:
            if samples_only:
                self._counters.pop(SAMPLES_TABLE, None)
            else:
                self._counters.clear()
            self._schema_ready = False

        self.ensure_ready()

        return {
            "samples_only": bool(samples_only),
            "deleted_samples": deleted_samples,
            "deleted_identities": deleted_identities,
            "deleted_consents": deleted_consents,
        }

    def enroll_face(
        self,
        *,
        identity_id: int,
        image_bytes: bytes | None = None,
        image_path: str | Path | None = None,
        relative_path: str | None = None,
        split: str = "enroll",
        face_policy: str = "strict_single",
    ) -> dict[str, Any]:
        """Valida consentimiento, detecta rostro, extrae embedding y lo persiste en el DBMS."""
        identity = self.get_identity(identity_id, use_cache=False)
        self._assert_enroll_allowed(identity)

        source_input, digest, rel_path = self._resolve_image_input(
            identity_id=identity_id,
            image_bytes=image_bytes,
            image_path=image_path,
            relative_path=relative_path,
        )

        t0 = time.perf_counter()
        extracted, faces_detected = self.extractor.extract_primary_face(
            source_input, policy=face_policy
        )
        extract_ms = (time.perf_counter() - t0) * 1000.0

        sample_id = self._next_id(SAMPLES_TABLE, "sample_id")
        sample_row = {
            "sample_id": sample_id,
            "identity_id": int(identity_id),
            "relative_path": rel_path,
            "source": identity["source"],
            "split": split,
            "sha256": digest,
            "embedding": _encode_vector_b64(extracted.embedding),
            "embedding_dim": int(self.extractor.embedding_dim),
            "model_name": self.extractor.model_name,
        }
        t_ins = time.perf_counter()
        self.client.insert_rows(SAMPLES_TABLE, [sample_row])
        insert_ms = (time.perf_counter() - t_ins) * 1000.0

        return {
            "sample_id": sample_id,
            "identity_id": int(identity_id),
            "display_label": identity.get("display_label"),
            "source": identity["source"],
            "split": split,
            "relative_path": rel_path,
            "sha256": digest,
            "faces_detected": faces_detected,
            "detection_confidence": extracted.confidence,
            "bbox": list(extracted.bbox),
            "model_name": self.extractor.model_name,
            "embedding_dim": int(self.extractor.embedding_dim),
            "timings_ms": {
                "extraction_ms": round(extract_ms, 3),
                "insert_ms": round(insert_ms, 3),
            },
        }

    def recognize_face(
        self,
        *,
        image_bytes: bytes | None = None,
        image_path: str | Path | None = None,
        k: int | None = None,
        threshold: float | None = None,
        index_type: str | None = None,
        metric: str | None = None,
        ef_search: int | None = None,
        n_probe: int | None = None,
        aggregation: str = "max",
        allow_dataset_identities: bool = False,
        face_policy: str = "largest",
    ) -> dict[str, Any]:
        """Ejecuta detección -> embedding -> vector search y decide registrado vs desconocido."""
        self._ensure_ready_once()

        effective_k = k if k is not None else self.settings.default_k
        effective_threshold = (
            threshold if threshold is not None else self.settings.default_threshold
        )
        effective_index = (index_type or self.settings.default_index_type).lower()
        effective_metric = (metric or self.settings.default_metric).lower()
        effective_ef = ef_search if ef_search is not None else self.settings.hnsw_ef_search
        effective_probe = n_probe if n_probe is not None else self.settings.ivf_n_probe

        if effective_k <= 0:
            raise ValueError("k debe ser un entero positivo")
        if aggregation not in ("max", "mean_top"):
            raise ValueError("aggregation debe ser 'max' o 'mean_top'")

        source_input, _, _ = self._resolve_image_input(
            identity_id=None,
            image_bytes=image_bytes,
            image_path=image_path,
            relative_path=None,
        )

        t_det0 = time.perf_counter()
        extracted, faces_detected = self.extractor.extract_primary_face(
            source_input, policy=face_policy
        )
        extract_ms = (time.perf_counter() - t_det0) * 1000.0

        t_search0 = time.perf_counter()
        search_res = self.client.vector_search(
            SAMPLES_TABLE,
            extracted.embedding,
            column="embedding",
            k=effective_k,
            index_type=effective_index,
            metric=effective_metric,
            ef_search=effective_ef,
            n_probe=effective_probe,
        )
        roundtrip_search_ms = (time.perf_counter() - t_search0) * 1000.0

        candidates = self._aggregate_candidates(
            search_res.get("results", []),
            aggregation=aggregation,
            allow_dataset_identities=allow_dataset_identities,
        )

        best = candidates[0] if candidates else None
        is_registered = best is not None and best["score"] >= effective_threshold

        return {
            "registered": is_registered,
            "matched": is_registered,
            "status": "registered" if is_registered else "unknown",
            "label": (
                best["display_label"] or best["source_subject_id"]
                if is_registered and best is not None
                else "desconocido"
            ),
            "identity_id": best["identity_id"] if is_registered and best is not None else None,
            "score": round(best["score"], 6) if best is not None else 0.0,
            "threshold": float(effective_threshold),
            "faces_detected": faces_detected,
            "detection_confidence": extracted.confidence,
            "bbox": list(extracted.bbox),
            "model_name": self.extractor.model_name,
            "embedding_dim": int(self.extractor.embedding_dim),
            "index_type": effective_index,
            "used_index": bool(search_res.get("used_index", False)),
            "metric": effective_metric,
            "k": effective_k,
            "aggregation": aggregation,
            "candidates": candidates,
            "timings_ms": {
                "extraction_ms": round(extract_ms, 3),
                "dbms_search_ms": round(float(search_res.get("search_ms", 0.0)), 3),
                "search_roundtrip_ms": round(roundtrip_search_ms, 3),
                "total_ms": round(extract_ms + roundtrip_search_ms, 3),
            },
        }

    def rebuild_vector_index(
        self,
        *,
        index_type: str | None = None,
        dim: int | None = None,
        n_lists: int | None = None,
        m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
    ) -> dict[str, Any]:
        """Reconstruye un índice vectorial de face_samples con hiperparámetros ajustables."""
        self._ensure_ready_once()
        target_type = (index_type or self.settings.default_index_type).lower()
        target_dim = dim if dim is not None else self.extractor.embedding_dim

        if target_type == "hnsw":
            idx_name = "idx_face_samples_hnsw"
            params = {
                "m": m if m is not None else self.settings.hnsw_m,
                "ef_construction": (
                    ef_construction
                    if ef_construction is not None
                    else self.settings.hnsw_ef_construction
                ),
                "ef_search": ef_search if ef_search is not None else self.settings.hnsw_ef_search,
            }
        elif target_type == "ivf":
            idx_name = "idx_face_samples_ivf"
            params = {
                "n_lists": n_lists if n_lists is not None else self.settings.ivf_n_lists,
                "n_probe": self.settings.ivf_n_probe,
            }
        else:
            raise ValueError("index_type debe ser 'hnsw' o 'ivf'")

        self.client.query(f"DROP INDEX IF EXISTS {idx_name};")
        created = self.client.create_index(
            SAMPLES_TABLE,
            "embedding",
            target_type,
            index_name=idx_name,
            dim=target_dim,
            index_params=params,
        )
        return {
            "index_name": idx_name,
            "index_type": target_type,
            "dim": created.get("dim", target_dim),
            "index_params": params,
            "entry": created,
        }

    def _assert_enroll_allowed(self, identity: dict[str, Any]) -> None:
        status = identity.get("consent_status")
        source = identity.get("source")
        if status == "withdrawn":
            raise IdentityWithdrawnError(
                f"El consentimiento de la identidad {identity['identity_id']} fue revocado"
            )
        if source == "participant" and status != "granted":
            raise ConsentRequiredError(
                f"La identidad {identity['identity_id']} tiene estado '{status}'; "
                "se requiere consent_status='granted' para enrolar muestras"
            )
        if source in ("public_lfw", "public_celeba") and status != "dataset_only":
            raise ConsentRequiredError(
                f"Estado de consentimiento inválido '{status}' para origen público '{source}'"
            )

    def _resolve_image_input(
        self,
        *,
        identity_id: int | None,
        image_bytes: bytes | None,
        image_path: str | Path | None,
        relative_path: str | None,
    ) -> tuple[bytes | Path, str, str]:
        if image_bytes is not None:
            digest = _sha256_bytes(image_bytes)
            rel = relative_path or f"uploads/{identity_id or 'probe'}/{digest[:16]}.jpg"
            return image_bytes, digest, rel
        if image_path is not None:
            path = Path(image_path)
            raw = path.read_bytes()
            digest = _sha256_bytes(raw)
            rel = relative_path or str(path)
            return path, digest, rel
        raise ValueError("Debes proporcionar image_bytes o image_path")

    def _aggregate_candidates(
        self,
        raw_results: list[dict[str, Any]],
        *,
        aggregation: str,
        allow_dataset_identities: bool,
    ) -> list[dict[str, Any]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in raw_results:
            row = item.get("row", {})
            iid_val = row.get("identity_id")
            if iid_val is None:
                continue
            iid = int(iid_val)
            grouped.setdefault(iid, []).append(item)

        candidates: list[dict[str, Any]] = []
        for iid, items in grouped.items():
            try:
                identity = self.get_identity(iid, use_cache=True)
            except IdentityNotFoundError:
                continue

            status = identity.get("consent_status")
            source = identity.get("source")
            if status == "withdrawn":
                continue
            if source == "participant" and status != "granted":
                continue
            if source in ("public_lfw", "public_celeba") and not allow_dataset_identities:
                continue

            scores = sorted((float(x["score"]) for x in items), reverse=True)
            agg_score = (
                scores[0]
                if aggregation == "max"
                else float(sum(scores[:3]) / min(len(scores), 3))
            )
            best_sample = items[0]
            candidates.append(
                {
                    "identity_id": iid,
                    "source": source,
                    "source_subject_id": identity.get("source_subject_id"),
                    "display_label": identity.get("display_label"),
                    "consent_status": status,
                    "score": float(agg_score),
                    "best_sample_score": float(scores[0]),
                    "best_distance": float(best_sample["distance"]),
                    "matched_samples": len(items),
                    "sample_id": best_sample.get("row", {}).get("sample_id"),
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates
