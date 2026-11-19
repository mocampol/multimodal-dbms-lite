"""
API de detección (apps/audio-copyright/backend/detection_api.py). Llama a la app ASGI
directamente, como las pruebas de la app facial, y usa un buscador falso: no necesita
el DBMS ni el corpus.
"""

import io
import json
import sys
from pathlib import Path

import anyio
import numpy as np
import pytest
import soundfile as sf

APP_DIR = Path(__file__).resolve().parents[2] / "apps" / "audio-copyright"
for folder in ("features", "backend"):
    sys.path.insert(0, str(APP_DIR / folder))

import detection_api  # noqa: E402
from detection import DetectionSettings  # noqa: E402
from detection_api import create_app  # noqa: E402
from segment_extractor import SAMPLE_RATE  # noqa: E402

HOP, WINDOW = 2.5, 5.0


def call(app, method, path, body=b""):
    """Una petición a la app ASGI: (status, JSON)."""
    async def run():
        raw_path, _, query = path.partition("?")
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
                 "scheme": "http", "path": raw_path, "raw_path": raw_path.encode(), "query_string": query.encode(),
                 "headers": [(b"content-type", b"audio/mpeg")], "client": ("127.0.0.1", 0), "server": ("test", 80)}
        sent, status, chunks = False, 500, []

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await app(scope, receive, send)
        return status, json.loads(b"".join(chunks) or b"null")

    return anyio.run(run)


def wav_bytes(seconds, frequency=440.0):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    buffer = io.BytesIO()
    sf.write(buffer, (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


class FakeSearcher:
    """Devuelve candidatos fijos: la canción 2 copiada en los segmentos indicados."""

    def __init__(self, copied=(1, 2, 3), score=0.95, error=None):
        self.copied, self.score, self.error = copied, score, error
        self.calls = []

    def search_audio(self, signal, k=None, index_type=None, metric=None):
        if self.error:
            raise self.error
        self.calls.append({"samples": len(signal), "k": k, "index_type": index_type})
        duration = len(signal) / SAMPLE_RATE
        segments = []
        for index in range(int((duration - WINDOW) // HOP) + 1):
            start = index * HOP
            score = self.score if index in self.copied else 0.4
            segments.append({"segment_index": index, "start_s": start, "end_s": start + WINDOW, "candidates": [
                {"song_id": 2 if index in self.copied else 9, "title": "Food", "artist": "AWOL",
                 "license_family": "CC BY-NC-SA", "track_url": "http://fma/2",
                 "start_s": start + 10.0, "end_s": start + 15.0, "score": score}]})
        return {"duration_s": duration, "segments": segments, "index_type": index_type or "hnsw",
                "metric": "cosine", "k": k or 5, "used_index": True, "timings_ms": {"total_ms": 3.0}}


SETTINGS = DetectionSettings(threshold=0.9, min_segments=2, max_gap=1, offset_tolerance_s=HOP, margin=0.0)


@pytest.fixture
def searcher():
    return FakeSearcher()


@pytest.fixture
def app(searcher):
    return create_app(searcher=searcher, settings=SETTINGS)


# Detección

def test_a_copied_song_is_reported_with_its_interval_and_similarity(app, searcher):
    status, body = call(app, "POST", "/detections?filename=remix_2024.wav&k=3&index=ivf", wav_bytes(15))

    assert status == 200
    assert body["filename"] == "remix_2024.wav"
    assert body["verdict"] == "posible_coincidencia"
    [match] = body["matches"]
    assert (match["title"], match["artist"], match["similarity_pct"]) == ("Food", "AWOL", 95.0)
    assert (match["query_start_s"], match["query_end_s"]) == (2.5, 12.5)
    assert (match["reference_start_s"], match["reference_end_s"]) == (12.5, 22.5)
    assert searcher.calls == [{"samples": 15 * SAMPLE_RATE, "k": 3, "index_type": "ivf"}]


def test_no_similar_segment_is_an_explicit_no_match(app, searcher):
    searcher.copied = ()

    status, body = call(app, "POST", "/detections?filename=original.wav", wav_bytes(15))

    assert status == 200
    assert body["verdict"] == "sin_coincidencias"
    assert body["matches"] == [] and "Sin coincidencias" in body["message"]
    assert len(body["timeline"]) == 5


def test_the_threshold_can_be_raised_per_request(app):
    status, body = call(app, "POST", "/detections?filename=remix.wav&threshold=0.99", wav_bytes(15))

    assert status == 200 and body["verdict"] == "sin_coincidencias"
    assert body["settings"]["threshold"] == 0.99


def test_health_reports_whether_the_searcher_is_ready(app):
    assert call(app, "GET", "/health") == (200, {"status": "ok", "ready": True, "error": None,
                                                  "settings": SETTINGS.__dict__})


# Errores

@pytest.mark.parametrize("path, body, status, message", [
    ("/detections?filename=remix.wav", b"", 400, "vacío"),
    ("/detections?filename=remix.txt", b"hola", 415, "Formato no admitido"),
    ("/detections?filename=remix.mp3", b"esto no es audio", 422, "No se pudo leer el audio"),
    ("/detections?filename=remix.wav&index=lsh", b"x", 422, "Índice desconocido"),
])
def test_invalid_requests_explain_the_problem(app, path, body, status, message):
    got_status, got = call(app, "POST", path, body)

    assert got_status == status
    assert message in got["detail"]


def test_an_audio_shorter_than_one_segment_is_rejected(app):
    status, body = call(app, "POST", "/detections?filename=corto.wav", wav_bytes(3))

    assert status == 422 and "al menos 5 s" in body["detail"]


def test_k_outside_its_range_is_rejected(app):
    status, _ = call(app, "POST", "/detections?filename=remix.wav&k=0", wav_bytes(15))

    assert status == 422


def test_a_file_over_the_size_limit_is_rejected(app, monkeypatch):
    monkeypatch.setattr(detection_api, "MAX_UPLOAD_MB", 0.01)

    status, body = call(app, "POST", "/detections?filename=remix.wav", wav_bytes(15))

    assert status == 413 and "supera" in body["detail"]


def test_a_dbms_error_is_reported_as_bad_gateway(searcher, app):
    searcher.error = RuntimeError("POST /tables/audio_segments/vector-search respondió 500")

    status, body = call(app, "POST", "/detections?filename=remix.wav", wav_bytes(15))

    assert status == 502 and "vector-search" in body["detail"]


def test_without_codebook_detection_answers_service_unavailable(monkeypatch):
    def missing():
        raise FileNotFoundError("No existe el codebook data/audio/fma/segment_codebook.pkl")

    monkeypatch.setattr(detection_api.SegmentSearcher, "from_settings", staticmethod(missing))
    app = create_app(settings=SETTINGS)

    status, body = call(app, "POST", "/detections?filename=remix.wav", wav_bytes(15))

    assert status == 503 and "codebook" in body["detail"]
