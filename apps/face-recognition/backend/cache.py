from __future__ import annotations

from collections import OrderedDict
import json
import threading
import time
from typing import Any


class IdentityCache:
    """
    Caché LRU con TTL en memoria para identidades frecuentes, con soporte opcional
    de Redis si `redis_url` está configurado y el paquete `redis` está disponible.
    """

    def __init__(
        self,
        maxsize: int = 1024,
        ttl_seconds: int = 300,
        redis_url: str | None = None,
    ) -> None:
        self.maxsize = max(1, maxsize)
        self.ttl_seconds = max(1, ttl_seconds)
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, tuple[float, dict[str, Any]]] = OrderedDict()
        self._redis = self._connect_redis(redis_url) if redis_url else None

    @staticmethod
    def _connect_redis(redis_url: str):
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    @staticmethod
    def _redis_key(identity_id: int) -> str:
        return f"face_app:identity:{identity_id}"

    def get(self, identity_id: int) -> dict[str, Any] | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(self._redis_key(identity_id))
                if raw is not None:
                    return json.loads(raw)
            except Exception:
                pass

        now = time.monotonic()
        with self._lock:
            item = self._entries.get(identity_id)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                del self._entries[identity_id]
                return None
            self._entries.move_to_end(identity_id)
            return dict(value)

    def set(self, identity_id: int, value: dict[str, Any]) -> None:
        payload = dict(value)
        if self._redis is not None:
            try:
                self._redis.setex(
                    self._redis_key(identity_id),
                    self.ttl_seconds,
                    json.dumps(payload, ensure_ascii=False),
                )
            except Exception:
                pass

        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._entries[identity_id] = (expires_at, payload)
            self._entries.move_to_end(identity_id)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)

    def invalidate(self, identity_id: int) -> None:
        if self._redis is not None:
            try:
                self._redis.delete(self._redis_key(identity_id))
            except Exception:
                pass
        with self._lock:
            self._entries.pop(identity_id, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
