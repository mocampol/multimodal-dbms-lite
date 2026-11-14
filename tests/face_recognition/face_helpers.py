from __future__ import annotations

import io
import json
from typing import Any
from urllib.parse import urlencode

import anyio


class _ASGIResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body
        self.text = body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class ASGITestClient:
    """Cliente de prueba síncrono sobre ASGI con soporte JSON y multipart/form-data."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def get(self, path: str, params: dict[str, Any] | None = None) -> _ASGIResponse:
        return self._request("GET", path, params=params)

    def delete(self, path: str, params: dict[str, Any] | None = None) -> _ASGIResponse:
        return self._request("DELETE", path, params=params)

    def post(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, io.BytesIO | bytes, str]] | None = None,
    ) -> _ASGIResponse:
        payload = json if json is not None else json_body
        return self._request("POST", path, json_body=payload, data=data, files=files)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, io.BytesIO | bytes, str]] | None = None,
    ) -> _ASGIResponse:
        import json as _json

        query_string = urlencode({k: v for k, v in (params or {}).items() if v is not None}).encode("ascii")
        headers: list[tuple[bytes, bytes]] = []
        raw_body = b""

        if json_body is not None:
            raw_body = _json.dumps(json_body).encode("utf-8")
            headers.append((b"content-type", b"application/json"))
        elif files is not None or data is not None:
            boundary = "----faceapp-test-boundary-7e91c"
            parts: list[bytes] = []
            for key, value in (data or {}).items():
                parts.append(
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                    f"{value}\r\n".encode("utf-8")
                )
            for key, (filename, file_obj, mime_type) in (files or {}).items():
                content = file_obj.getvalue() if hasattr(file_obj, "getvalue") else bytes(file_obj)
                parts.append(
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                    f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8")
                    + content
                    + b"\r\n"
                )
            parts.append(f"--{boundary}--\r\n".encode("utf-8"))
            raw_body = b"".join(parts)
            headers.append((b"content-type", f"multipart/form-data; boundary={boundary}".encode("ascii")))

        headers.append((b"content-length", str(len(raw_body)).encode("ascii")))

        async def _call():
            status_code = 500
            chunks: list[bytes] = []
            sent = False
            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method.upper(),
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": query_string,
                "headers": headers,
                "client": ("127.0.0.1", 0),
                "server": ("testserver", 80),
            }

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": raw_body, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(msg):
                nonlocal status_code
                if msg["type"] == "http.response.start":
                    status_code = int(msg["status"])
                elif msg["type"] == "http.response.body":
                    chunks.append(msg.get("body", b""))

            await self.app(scope, receive, send)
            return _ASGIResponse(status_code, b"".join(chunks))

        return anyio.run(_call)


def make_face_payload(subject: str, *, noise: float = 0.0, seed_offset: int = 0) -> io.BytesIO:
    """Genera un payload sintético determinista consumible por FakeFaceExtractor."""
    body = json.dumps({"subject": subject, "noise": noise, "seed_offset": seed_offset}).encode("utf-8")
    return io.BytesIO(body)
