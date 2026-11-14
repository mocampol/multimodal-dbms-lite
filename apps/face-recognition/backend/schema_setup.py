from __future__ import annotations

from .config import FaceAppSettings
from .dbms_client import DBMSClient


IDENTITIES_TABLE = "face_identities"
SAMPLES_TABLE = "face_samples"
CONSENTS_TABLE = "face_consents"


def ensure_face_schema(
    client: DBMSClient,
    settings: FaceAppSettings,
    dim: int | None = None,
    create_ivf: bool = True,
    create_hnsw: bool = True,
) -> dict:
    """
    Crea de manera idempotente las tablas e índices necesarios en el DBMS
    a través de su API REST.
    """
    target_dim = dim or settings.resolved_dim()
    existing_tables = {item["name"]: item for item in client.list_tables()}

    created_tables: list[str] = []
    if IDENTITIES_TABLE not in existing_tables:
        client.query(
            f"CREATE TABLE {IDENTITIES_TABLE} ("
            "identity_id INTEGER PRIMARY KEY, "
            "source VARCHAR(32) NOT NULL, "
            "source_subject_id VARCHAR(128) NOT NULL, "
            "display_label VARCHAR(128), "
            "consent_status VARCHAR(16) NOT NULL"
            ");"
        )
        client.create_index(IDENTITIES_TABLE, "identity_id", "btree", index_name="idx_face_identities_id")
        created_tables.append(IDENTITIES_TABLE)

    if CONSENTS_TABLE not in existing_tables:
        client.query(
            f"CREATE TABLE {CONSENTS_TABLE} ("
            "consent_id INTEGER PRIMARY KEY, "
            "identity_id INTEGER NOT NULL, "
            "purpose VARCHAR(128) NOT NULL, "
            "granted_at VARCHAR(32) NOT NULL, "
            "notice_version VARCHAR(32) NOT NULL, "
            "status VARCHAR(16) NOT NULL, "
            "withdrawn_at VARCHAR(32)"
            ");"
        )
        client.create_index(CONSENTS_TABLE, "identity_id", "btree", index_name="idx_face_consents_identity")
        created_tables.append(CONSENTS_TABLE)

    if SAMPLES_TABLE not in existing_tables:
        client.query(
            f"CREATE TABLE {SAMPLES_TABLE} ("
            "sample_id INTEGER PRIMARY KEY, "
            "identity_id INTEGER NOT NULL, "
            "relative_path VARCHAR(512) NOT NULL, "
            "source VARCHAR(32) NOT NULL, "
            "split VARCHAR(16) NOT NULL, "
            "sha256 VARCHAR(64) NOT NULL, "
            "embedding BYTEA NOT NULL, "
            "embedding_dim INTEGER NOT NULL, "
            "model_name VARCHAR(64) NOT NULL"
            ");"
        )
        client.create_index(SAMPLES_TABLE, "identity_id", "btree", index_name="idx_face_samples_identity")
        created_tables.append(SAMPLES_TABLE)

    samples_meta = client.get_table(SAMPLES_TABLE)
    existing_indexes = {
        (idx["column_name"], idx["index_type"]) for idx in samples_meta.get("indexes", [])
    }

    created_indexes: list[str] = []
    if create_hnsw and ("embedding", "hnsw") not in existing_indexes:
        client.create_index(
            SAMPLES_TABLE,
            "embedding",
            "hnsw",
            index_name="idx_face_samples_embedding_hnsw",
            dim=target_dim,
            index_params={
                "metric": settings.default_metric,
                "ef_construction": settings.hnsw_ef_construction,
                "M": settings.hnsw_m,
                "ef_search": settings.hnsw_ef_search,
            },
        )
        created_indexes.append("hnsw")

    if create_ivf and ("embedding", "ivf") not in existing_indexes:
        client.create_index(
            SAMPLES_TABLE,
            "embedding",
            "ivf",
            index_name="idx_face_samples_embedding_ivf",
            dim=target_dim,
            index_params={
                "metric": settings.default_metric,
                "n_lists": settings.ivf_n_lists,
                "n_probe": settings.ivf_n_probe,
            },
        )
        created_indexes.append("ivf")

    return {
        "created_tables": created_tables,
        "created_indexes": created_indexes,
        "dim": target_dim,
    }
