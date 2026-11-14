from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import urlencode

import anyio
import numpy as np
import requests

from .exceptions import DBMSClientError


async def _call_asgi(
    app: Any,
    method: str,
    path: str,
    body: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, bytes]:
    raw_path, _, query_str = path.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode("ascii"),
        "query_string": query_str.encode("ascii"),
        "headers": headers or [],
        "client": ("127.0.0.1", 0),
        "server": ("dbms.local", 80),
    }
    sent_body = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return status_code, b"".join(chunks)


class DBMSClient:
    """Cliente HTTP REST para consumir el motor multimodal-dbms-lite."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
        asgi_app: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._asgi_app = asgi_app
        self._session = None if asgi_app is not None else requests.Session()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if self._asgi_app is not None:
            full_path = path if not params else f"{path}?{urlencode(params)}"
            raw_body = b""
            headers: list[tuple[bytes, bytes]] = []
            if json_body is not None:
                raw_body = json.dumps(json_body).encode("utf-8")
                headers.append((b"content-type", b"application/json"))
                headers.append((b"content-length", str(len(raw_body)).encode("ascii")))
            try:
                status_code, raw_resp = anyio.run(
                    _call_asgi, self._asgi_app, method, full_path, raw_body, headers
                )
            except Exception as exc:
                raise DBMSClientError(f"Error invocando DBMS ASGI ({path}): {exc}") from exc
            text = raw_resp.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text) if text else None
            except ValueError:
                parsed = None
        else:
            assert self._session is not None
            url = f"{self.base_url}{path}"
            try:
                response = self._session.request(
                    method, url, json=json_body, params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                raise DBMSClientError(f"Error de conexión con DBMS ({path}): {exc}") from exc
            status_code = response.status_code
            text = response.text
            try:
                parsed = response.json()
            except ValueError:
                parsed = None

        if status_code >= 400:
            detail = (
                parsed.get("error") or parsed.get("detail") or text
                if isinstance(parsed, dict)
                else text
            )
            raise DBMSClientError(
                f"DBMS devolvió HTTP {status_code} en {path}: {detail}",
                status_code=status_code,
            )
        return parsed

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_tables(self) -> list[dict[str, Any]]:
        return self._request("GET", "/tables")

    def get_table(self, table_name: str) -> dict[str, Any]:
        return self._request("GET", f"/tables/{table_name}")

    def query(self, sql: str) -> dict[str, Any]:
        return self._request("POST", "/query", json_body={"sql": sql})

    def insert_rows(self, table_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", f"/tables/{table_name}/rows", json_body={"rows": rows})

    def create_index(
        self,
        table_name: str,
        column_name: str,
        index_type: str,
        index_name: str | None = None,
        dim: int | None = None,
        index_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "column_name": column_name,
            "index_type": index_type.lower(),
        }
        if index_name is not None:
            payload["index_name"] = index_name
        if dim is not None:
            payload["dim"] = dim
        if index_params is not None:
            payload["index_params"] = index_params
        return self._request("POST", f"/tables/{table_name}/indexes", json_body=payload)

    def vector_search(
        self,
        table_name: str,
        vector: np.ndarray | list[float],
        column: str = "embedding",
        k: int = 5,
        index_type: str = "hnsw",
        metric: str = "cosine",
        ef_search: int | None = None,
        n_probe: int | None = None,
    ) -> dict[str, Any]:
        vec = np.asarray(vector, dtype="<f4").reshape(-1)
        payload: dict[str, Any] = {
            "column": column,
            "vector_b64": base64.b64encode(vec.tobytes()).decode("ascii"),
            "k": int(k),
            "index_type": index_type.lower(),
            "metric": metric.lower(),
        }
        if ef_search is not None:
            payload["ef_search"] = int(ef_search)
        if n_probe is not None:
            payload["n_probe"] = int(n_probe)
        return self._request("POST", f"/tables/{table_name}/vector-search", json_body=payload)
