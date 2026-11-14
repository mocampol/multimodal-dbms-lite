import base64
import importlib.util
import json
import sys
from pathlib import Path

import anyio
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT / "api"), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from transaction.transaction_manager import TransactionManager


def _load_src_main():
    spec = importlib.util.spec_from_file_location("dbms_src_main", REPO_ROOT / "src" / "main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    def json(self):
        return json.loads(self._body.decode("utf-8"))


class _ASGIClient:
    """Cliente de prueba síncrono mínimo sobre ASGI (sin depender de httpx/httpx2)."""

    def __init__(self, app):
        self.app = app

    def post(self, path: str, *, json: dict | None = None) -> _Response:
        import json as _json

        raw = _json.dumps(json).encode("utf-8") if json is not None else b""
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode("ascii")),
        ]

        async def _run():
            status_code = 500
            chunks: list[bytes] = []
            sent = False
            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": headers,
                "client": ("127.0.0.1", 0),
                "server": ("testserver", 80),
            }

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": raw, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(msg):
                nonlocal status_code
                if msg["type"] == "http.response.start":
                    status_code = int(msg["status"])
                elif msg["type"] == "http.response.body":
                    chunks.append(msg.get("body", b""))

            await self.app(scope, receive, send)
            return _Response(status_code, b"".join(chunks))

        return anyio.run(_run)


def _load_api_main():
    spec = importlib.util.spec_from_file_location("dbms_api_main", REPO_ROOT / "api" / "main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_dbms_client(tmp_path, monkeypatch):
    import engine as api_engine
    from routers import query as query_router
    from routers import tables as tables_router
    from routers import vectors as vectors_router

    src_main = _load_src_main()
    api_main = _load_api_main()
    data_dir = tmp_path / "db_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    heap_factory = src_main.make_heap_factory(str(data_dir))
    cat = Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: src_main.make_sequential_factory(str(data_dir)),
        },
        index_buffer_factory=src_main.make_index_buffer_factory(str(data_dir)),
    )
    cat._transaction_manager = TransactionManager(str(data_dir / "transactions.wal"))

    monkeypatch.setattr(api_engine, "catalog", cat)
    monkeypatch.setattr(query_router, "catalog", cat)
    monkeypatch.setattr(tables_router, "catalog", cat)
    monkeypatch.setattr(vectors_router, "catalog", cat)

    yield _ASGIClient(api_main.app), cat


def _unit_vec(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_vectors_api_insert_index_search_and_delete(isolated_dbms_client):
    client, _cat = isolated_dbms_client

    r_create = client.post(
        "/query",
        json={
            "sql": (
                "CREATE TABLE embeddings ("
                "id INTEGER PRIMARY KEY, "
                "subject VARCHAR(64) NOT NULL, "
                "vec BYTEA NOT NULL);"
            )
        },
    )
    assert r_create.status_code == 200

    dim = 192
    v1 = _unit_vec(1, dim)
    v2 = _unit_vec(2, dim)
    v3 = _unit_vec(3, dim)

    r_ins = client.post(
        "/tables/embeddings/rows",
        json={
            "rows": [
                {"id": 1, "subject": "alice", "vec": base64.b64encode(v1.tobytes()).decode("ascii")},
                {"id": 2, "subject": "bob", "vec": v2.tolist()},
                {"id": 3, "subject": "carol", "vec": base64.b64encode(v3.tobytes()).decode("ascii")},
            ]
        },
    )
    assert r_ins.status_code == 200
    assert r_ins.json()["inserted"] == 3

    # Búsqueda antes de crear índice usa fallback vectorizado exacto sobre BYTEA
    r_search_fallback = client.post(
        "/tables/embeddings/vector-search",
        json={"column": "vec", "vector": v2.tolist(), "k": 2, "index_type": "hnsw", "metric": "cosine"},
    )
    assert r_search_fallback.status_code == 200
    body_fb = r_search_fallback.json()
    assert body_fb["used_index"] is False
    assert body_fb["dim"] == dim
    assert body_fb["results"][0]["row"]["subject"] == "bob"

    # Crear índice HNSW y luego IVF con parámetros dinámicos
    r_idx_hnsw = client.post(
        "/tables/embeddings/indexes",
        json={
            "column_name": "vec",
            "index_type": "hnsw",
            "index_name": "idx_emb_hnsw",
            "dim": dim,
            "index_params": {"m": 12, "ef_construction": 100, "ef_search": 80},
        },
    )
    assert r_idx_hnsw.status_code == 200
    assert r_idx_hnsw.json()["dim"] == dim

    r_idx_ivf = client.post(
        "/tables/embeddings/indexes",
        json={
            "column_name": "vec",
            "index_type": "ivf",
            "index_name": "idx_emb_ivf",
            "dim": dim,
            "index_params": {"n_lists": 2, "n_probe": 2},
        },
    )
    assert r_idx_ivf.status_code == 200
    assert r_idx_ivf.json()["dim"] == dim

    # Consultar usando ambos índices
    for idx_type in ("hnsw", "ivf"):
        r_search = client.post(
            "/tables/embeddings/vector-search",
            json={
                "column": "vec",
                "vector_b64": base64.b64encode(v2.tobytes()).decode("ascii"),
                "k": 2,
                "index_type": idx_type,
                "metric": "cosine",
                "ef_search": 120,
                "n_probe": 2,
            },
        )
        assert r_search.status_code == 200
        body = r_search.json()
        assert body["used_index"] is True
        assert body["dim"] == dim
        assert body["results"][0]["row"]["subject"] == "bob"
        assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-4)

    # Eliminar a bob vía SQL y comprobar que desaparece de resultados
    r_del = client.post("/query", json={"sql": "DELETE FROM embeddings WHERE id = 2;"})
    assert r_del.status_code == 200

    r_after = client.post(
        "/tables/embeddings/vector-search",
        json={"column": "vec", "vector": v2.tolist(), "k": 3, "index_type": "hnsw", "metric": "cosine"},
    )
    assert r_after.status_code == 200
    subjects = [item["row"]["subject"] for item in r_after.json()["results"]]
    assert "bob" not in subjects
