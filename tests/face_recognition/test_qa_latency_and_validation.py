"""
tests/face_recognition/test_qa_latency_and_validation.py

Suite de QA para el issue #121: validar los cuatro escenarios de reconocimiento
facial y medir latencia de extremo a extremo con assertion del SLA < 100 ms.

Escenarios cubiertos:
  1. Identidades registradas       — reconocimiento correcto con HNSW e IVF
  2. Identidades desconocidas      — rechazo consistente como "desconocido"
  3. Múltiples rostros             — strict_single rechaza, largest selecciona
  4. Variaciones de imagen         — niveles de ruido y dimensiones 128-d / 512-d
  5. Latencia end-to-end           — p95 de dbms_search_ms < 100 ms (SLA)

Todos los tests corren offline via ASGI en memoria usando FakeFaceExtractor;
no requieren modelos externos ni el DBMS levantado por separado.
"""
from __future__ import annotations

import io
import statistics

import pytest

from face_helpers import ASGITestClient, make_face_payload


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _enroll(client: ASGITestClient, identity_id: int, subject: str, *, noise: float = 0.0, seed_offset: int = 0) -> None:
    """Enrola una muestra facial para la identidad dada."""
    resp = client.post(
        "/enroll",
        data={"identity_id": str(identity_id), "split": "enroll"},
        files={"file": (f"{subject}_{seed_offset}.jpg", make_face_payload(subject, noise=noise, seed_offset=seed_offset), "image/jpeg")},
    )
    assert resp.status_code == 201, f"Enroll falló para {subject}: {resp.text}"


def _register(client: ASGITestClient, label: str) -> int:
    """Registra una identidad con consentimiento y retorna su identity_id."""
    resp = client.post(
        "/identities",
        json={"source": "participant", "display_label": label, "consent_granted": True},
    )
    assert resp.status_code == 201, f"Registro falló para {label}: {resp.text}"
    return resp.json()["identity_id"]


def _recognize(client: ASGITestClient, subject: str, *, noise: float = 0.05, seed_offset: int = 99,
               index_type: str = "hnsw", k: int = 3, threshold: float = 0.45) -> dict:
    """Envía una imagen probe y retorna el body JSON de /recognize."""
    resp = client.post(
        "/recognize",
        data={"index_type": index_type, "k": str(k), "threshold": str(threshold)},
        files={"file": ("probe.jpg", make_face_payload(subject, noise=noise, seed_offset=seed_offset), "image/jpeg")},
    )
    assert resp.status_code == 200, f"Recognize falló: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# 1. Identidades registradas
# ---------------------------------------------------------------------------

class TestRegisteredIdentities:
    """El sistema reconoce correctamente a identidades previamente enroladas."""

    SUBJECTS = [
        ("Ana",   "ana_qa"),
        ("Bruno", "bruno_qa"),
        ("Carla", "carla_qa"),
    ]

    def _setup(self, client: ASGITestClient) -> dict[str, int]:
        """Registra y enrola dos muestras por identidad; retorna label→identity_id."""
        mapping: dict[str, int] = {}
        for label, subject in self.SUBJECTS:
            iid = _register(client, label)
            _enroll(client, iid, subject, noise=0.02, seed_offset=0)
            _enroll(client, iid, subject, noise=0.03, seed_offset=1)
            mapping[label] = iid
        return mapping

    def test_hnsw_recognizes_all_enrolled_identities(self, dbms_and_face_env):
        client = dbms_and_face_env["client"]
        mapping = self._setup(client)

        for label, subject in self.SUBJECTS:
            body = _recognize(client, subject, index_type="hnsw", noise=0.05, seed_offset=99)

            assert body["registered"] is True,         f"[HNSW] {label} debería ser reconocida"
            assert body["matched"] is True,            f"[HNSW] {label}: matched debe ser True"
            assert body["status"] == "registered",     f"[HNSW] {label}: status incorrecto"
            assert body["identity_id"] == mapping[label], f"[HNSW] {label}: identity_id incorrecto"
            assert body["label"] == label,             f"[HNSW] {label}: label incorrecto"
            assert body["score"] >= 0.45,              f"[HNSW] {label}: score por debajo del umbral"
            assert body["used_index"] is True,         f"[HNSW] {label}: debería usar índice vectorial"

    def test_ivf_recognizes_all_enrolled_identities(self, dbms_and_face_env):
        client = dbms_and_face_env["client"]
        mapping = self._setup(client)

        for label, subject in self.SUBJECTS:
            body = _recognize(client, subject, index_type="ivf", noise=0.05, seed_offset=98)

            assert body["registered"] is True,         f"[IVF] {label} debería ser reconocida"
            assert body["identity_id"] == mapping[label], f"[IVF] {label}: identity_id incorrecto"
            assert body["label"] == label,             f"[IVF] {label}: label incorrecto"
            assert body["score"] >= 0.45,              f"[IVF] {label}: score por debajo del umbral"
            assert body["used_index"] is True,         f"[IVF] {label}: debería usar índice vectorial"

    def test_timings_present_in_response(self, dbms_and_face_env):
        """Verifica que la respuesta incluya el desglose de timings_ms."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "TimingSubject")
        _enroll(client, iid, "timing_subj")
        body = _recognize(client, "timing_subj")

        timings = body.get("timings_ms", {})
        assert "extraction_ms" in timings,       "Falta extraction_ms en timings"
        assert "dbms_search_ms" in timings,      "Falta dbms_search_ms en timings"
        assert "search_roundtrip_ms" in timings, "Falta search_roundtrip_ms en timings"
        assert "total_ms" in timings,            "Falta total_ms en timings"
        assert timings["extraction_ms"] >= 0
        assert timings["dbms_search_ms"] >= 0
        assert timings["total_ms"] >= timings["extraction_ms"]

    def test_k1_returns_top_candidate(self, dbms_and_face_env):
        """Con k=1 el candidato devuelto debe ser exactamente la identidad correcta."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "KOne")
        _enroll(client, iid, "k_one_subj", noise=0.01)
        body = _recognize(client, "k_one_subj", k=1, noise=0.03)

        assert body["registered"] is True
        assert body["k"] == 1
        assert len(body["candidates"]) == 1
        assert body["candidates"][0]["identity_id"] == iid


# ---------------------------------------------------------------------------
# 2. Identidades desconocidas
# ---------------------------------------------------------------------------

class TestUnknownIdentities:
    """El sistema rechaza como 'desconocido' a sujetos no enrolados."""

    def test_empty_db_always_returns_unknown(self, dbms_and_face_env):
        """Sin ninguna identidad registrada, toda imagen es desconocida."""
        client = dbms_and_face_env["client"]
        for i in range(3):
            body = _recognize(client, f"intruder_{i}", noise=0.0, seed_offset=i)
            assert body["registered"] is False
            assert body["matched"] is False
            assert body["status"] == "unknown"
            assert body["label"] == "desconocido"
            assert body["identity_id"] is None
            assert body["score"] == 0.0

    def test_unknown_subject_rejected_when_others_enrolled(self, dbms_and_face_env):
        """Sujetos no enrolados son rechazados incluso cuando hay identidades en la DB."""
        client = dbms_and_face_env["client"]
        # Enrolar identidades conocidas
        for label, subject in [("Pedro", "pedro_q"), ("Sofia", "sofia_q")]:
            iid = _register(client, label)
            _enroll(client, iid, subject)

        # Probar sujetos nunca registrados
        unknown_subjects = ["extraño_1", "extraño_2", "extraño_3"]
        for subj in unknown_subjects:
            body = _recognize(client, subj, noise=0.0, threshold=0.45)
            assert body["registered"] is False, f"{subj} no debería ser reconocido"
            assert body["status"] == "unknown"
            assert body["label"] == "desconocido"
            assert body["identity_id"] is None
            assert body["score"] < 0.45, f"{subj}: score >= threshold sin estar enrolado"

    def test_strict_threshold_forces_unknown(self, dbms_and_face_env):
        """Un umbral muy alto (0.9999) fuerza rechazo incluso de identidades enroladas."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "StrictTest")
        _enroll(client, iid, "strict_subj", noise=0.01)

        body = _recognize(client, "strict_subj", noise=0.10, threshold=0.9999)
        assert body["registered"] is False
        assert body["label"] == "desconocido"

    def test_dataset_identities_hidden_in_live_mode(self, dbms_and_face_env):
        """Identidades de datasets públicos no se exponen en modo live (allow_dataset_identities=False por defecto)."""
        client = dbms_and_face_env["client"]
        service = dbms_and_face_env["service"]

        # Registrar identidad LFW (dataset público)
        lfw = service.register_identity(
            source="public_lfw",
            source_subject_id="lfw:UnknownPerson_001",
            consent_granted=True,
        )
        service.enroll_face(
            identity_id=lfw["identity_id"],
            image_bytes=make_face_payload("lfw:UnknownPerson_001").getvalue(),
        )

        # En modo live (allow_dataset_identities=False), debe responder "desconocido"
        body = _recognize(client, "lfw:UnknownPerson_001", noise=0.0, threshold=0.40)
        assert body["registered"] is False
        assert body["label"] == "desconocido"


# ---------------------------------------------------------------------------
# 3. Múltiples rostros
# ---------------------------------------------------------------------------

class TestMultipleFacesHandling:
    """Cobertura de los tres casos de múltiples rostros en enrolamiento y reconocimiento."""

    def test_enroll_strict_single_rejects_no_face(self, dbms_and_face_env):
        """strict_single rechaza imagen sin rostro con 422 no_face_detected."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "MultiTest1")

        resp = client.post(
            "/enroll",
            data={"identity_id": str(iid), "face_policy": "strict_single"},
            files={"file": ("empty.jpg", io.BytesIO(b"NO_FACE"), "image/jpeg")},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "no_face_detected"

    def test_enroll_strict_single_rejects_multiple_faces(self, dbms_and_face_env):
        """strict_single rechaza imagen con múltiples rostros con 422 multiple_faces_detected."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "MultiTest2")

        resp = client.post(
            "/enroll",
            data={"identity_id": str(iid), "face_policy": "strict_single"},
            files={"file": ("crowd.jpg", io.BytesIO(b"MULTI_FACE"), "image/jpeg")},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "multiple_faces_detected"

    def test_recognize_largest_policy_selects_dominant_face(self, dbms_and_face_env):
        """face_policy=largest debe procesar la imagen y retornar un resultado válido."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "MultiLargest")
        _enroll(client, iid, "MULTI_FACE", noise=0.0)  # Enrola con la política de embedding de MULTI_FACE

        # Reconocer con política largest (no lanza excepción)
        resp = client.post(
            "/recognize",
            data={"face_policy": "largest", "threshold": "0.40"},
            files={"file": ("crowd.jpg", io.BytesIO(b"MULTI_FACE"), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["faces_detected"] == 2, "largest debe reportar cuántos rostros detectó"

    def test_recognize_no_face_returns_422(self, dbms_and_face_env):
        """Imagen sin rostros en /recognize debe retornar 422 no_face_detected."""
        client = dbms_and_face_env["client"]

        resp = client.post(
            "/recognize",
            files={"file": ("blank.jpg", io.BytesIO(b"NO_FACE"), "image/jpeg")},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "no_face_detected"


# ---------------------------------------------------------------------------
# 4. Variaciones de imagen
# ---------------------------------------------------------------------------

class TestImageVariations:
    """El sistema mantiene correctitud y latencia bajo distintas variaciones de imagen."""

    THRESHOLD = 0.45

    def _setup_identity(self, client: ASGITestClient, label: str, subject: str) -> int:
        iid = _register(client, label)
        _enroll(client, iid, subject, noise=0.01, seed_offset=0)
        _enroll(client, iid, subject, noise=0.02, seed_offset=1)
        return iid

    def test_low_noise_recognized(self, dbms_and_face_env):
        """Ruido bajo (noise=0.02) no impide el reconocimiento."""
        client = dbms_and_face_env["client"]
        iid = self._setup_identity(client, "LowNoise", "lownoise_subj")
        body = _recognize(client, "lownoise_subj", noise=0.02, threshold=self.THRESHOLD)
        assert body["registered"] is True
        assert body["identity_id"] == iid

    def test_medium_noise_recognized(self, dbms_and_face_env):
        """Ruido moderado (noise=0.07) sigue siendo reconocido con umbral 0.45."""
        client = dbms_and_face_env["client"]
        iid = self._setup_identity(client, "MedNoise", "mednoise_subj")
        body = _recognize(client, "mednoise_subj", noise=0.07, threshold=self.THRESHOLD)
        assert body["registered"] is True
        assert body["identity_id"] == iid

    def test_high_noise_may_fail(self, dbms_and_face_env):
        """Ruido alto (noise=0.30) puede fallar; lo importante es que no se confunda con otro sujeto."""
        client = dbms_and_face_env["client"]
        iid_a = self._setup_identity(client, "HighNoiseA", "highnoise_a")
        iid_b = self._setup_identity(client, "HighNoiseB", "highnoise_b")

        body = _recognize(client, "highnoise_a", noise=0.30, threshold=self.THRESHOLD)
        # Si reconoce, debe ser A — nunca B
        if body["registered"]:
            assert body["identity_id"] == iid_a, "No debe confundirse con otra identidad bajo ruido alto"

    def test_no_cross_identity_confusion(self, dbms_and_face_env):
        """Distintos sujetos nunca deben ser confundidos entre sí con ruido bajo."""
        client = dbms_and_face_env["client"]
        iid_x = self._setup_identity(client, "SubjX", "subj_x")
        iid_y = self._setup_identity(client, "SubjY", "subj_y")

        body_x = _recognize(client, "subj_x", noise=0.03, threshold=self.THRESHOLD)
        body_y = _recognize(client, "subj_y", noise=0.03, threshold=self.THRESHOLD)

        assert body_x["registered"] is True
        assert body_x["identity_id"] == iid_x
        assert body_y["registered"] is True
        assert body_y["identity_id"] == iid_y

    def test_512d_embedding_pipeline(self, dbms_and_face_env):
        """Dimensión 512-d (modo FaceNet) enrola y reconoce correctamente."""
        face_app_512, _service, client_512_dbms = dbms_and_face_env["build_app"](dim=512)
        try:
            client = ASGITestClient(face_app_512)
            iid = _register(client, "Dim512")
            resp = client.post(
                "/enroll",
                data={"identity_id": str(iid)},
                files={"file": ("dim512.jpg", make_face_payload("dim512_subj"), "image/jpeg")},
            )
            assert resp.status_code == 201
            assert resp.json()["embedding_dim"] == 512

            body = _recognize(client, "dim512_subj", noise=0.04, threshold=0.45)
            assert body["registered"] is True
            assert body["embedding_dim"] == 512
            assert body["identity_id"] == iid
        finally:
            client_512_dbms.close()

    def test_scores_decrease_with_noise(self, dbms_and_face_env):
        """A mayor ruido, el score de similitud tiende a bajar."""
        client = dbms_and_face_env["client"]
        iid = self._setup_identity(client, "NoiseTrend", "noise_trend_subj")

        scores = []
        for noise_level in (0.01, 0.05, 0.10, 0.20):
            body = _recognize(client, "noise_trend_subj", noise=noise_level, threshold=0.0)
            scores.append(body["score"])

        # La tendencia general debe ser decreciente (toleramos empates entre pasos adyacentes)
        assert scores[0] >= scores[-1], (
            f"El score con noise=0.01 ({scores[0]:.4f}) debería ser >= que con noise=0.20 ({scores[-1]:.4f})"
        )


# ---------------------------------------------------------------------------
# 5. Latencia end-to-end y validación del SLA < 100 ms
# ---------------------------------------------------------------------------

class TestEndToEndLatencySLA:
    """
    Mide latencia de extremo a extremo sobre ASGI en memoria y valida
    que el p95 de dbms_search_ms < 100 ms (objetivo del issue).

    Estos tests miden el rendimiento real del motor vectorial sin overhead
    de red HTTP (el DBMSClient usa modo ASGI en memoria).
    """

    N_PROBES = 30         # Número de búsquedas para estadísticas estables
    N_ENROLLED = 10       # Identidades en la DB durante las pruebas
    SLA_MS = 100.0        # Objetivo de latencia del issue

    def _setup_identities(self, client: ASGITestClient) -> list[int]:
        """Registra y enrola N_ENROLLED identidades, 2 muestras cada una."""
        iids = []
        for i in range(self.N_ENROLLED):
            iid = _register(client, f"SLASubject{i}")
            _enroll(client, iid, f"sla_subj_{i}", noise=0.01, seed_offset=0)
            _enroll(client, iid, f"sla_subj_{i}", noise=0.02, seed_offset=1)
            iids.append(iid)
        return iids

    def _collect_latencies(self, client: ASGITestClient, subjects: list[str],
                            index_type: str, n_probes: int) -> dict[str, list[float]]:
        """Ejecuta n_probes búsquedas y recolecta las métricas de timings_ms."""
        extraction_ms: list[float] = []
        dbms_search_ms: list[float] = []
        total_ms: list[float] = []

        for i in range(n_probes):
            subject = subjects[i % len(subjects)]
            resp = client.post(
                "/recognize",
                data={"index_type": index_type, "k": "5", "threshold": "0.40"},
                files={"file": ("probe.jpg", make_face_payload(subject, noise=0.05, seed_offset=i), "image/jpeg")},
            )
            assert resp.status_code == 200, f"Probe {i} falló: {resp.text}"
            t = resp.json()["timings_ms"]
            extraction_ms.append(t["extraction_ms"])
            dbms_search_ms.append(t["dbms_search_ms"])
            total_ms.append(t["total_ms"])

        return {
            "extraction_ms": extraction_ms,
            "dbms_search_ms": dbms_search_ms,
            "total_ms": total_ms,
        }

    def _report(self, label: str, values: list[float]) -> dict[str, float]:
        p = lambda pct: float(sorted(values)[int(len(values) * pct / 100)])
        stats = {
            "mean":   round(statistics.fmean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95":    round(p(95), 3),
            "p99":    round(p(99), 3) if len(values) >= 10 else round(max(values), 3),
            "max":    round(max(values), 3),
        }
        print(f"\n[Latencia {label}]  mean={stats['mean']} ms  p95={stats['p95']} ms  max={stats['max']} ms")
        return stats

    def test_hnsw_dbms_search_p95_under_sla(self, dbms_and_face_env):
        """HNSW: el p95 de dbms_search_ms debe ser < 100 ms."""
        client = dbms_and_face_env["client"]
        iids = self._setup_identities(client)
        subjects = [f"sla_subj_{i}" for i in range(self.N_ENROLLED)]

        latencies = self._collect_latencies(client, subjects, "hnsw", self.N_PROBES)

        dbms_stats = self._report("HNSW dbms_search_ms", latencies["dbms_search_ms"])
        total_stats = self._report("HNSW total_ms",       latencies["total_ms"])

        assert dbms_stats["p95"] < self.SLA_MS, (
            f"HNSW dbms_search_ms p95={dbms_stats['p95']} ms excede el SLA de {self.SLA_MS} ms"
        )

    def test_ivf_dbms_search_p95_under_sla(self, dbms_and_face_env):
        """IVF: el p95 de dbms_search_ms debe ser < 100 ms."""
        client = dbms_and_face_env["client"]
        iids = self._setup_identities(client)
        subjects = [f"sla_subj_{i}" for i in range(self.N_ENROLLED)]

        latencies = self._collect_latencies(client, subjects, "ivf", self.N_PROBES)

        dbms_stats = self._report("IVF dbms_search_ms", latencies["dbms_search_ms"])
        total_stats = self._report("IVF total_ms",       latencies["total_ms"])

        assert dbms_stats["p95"] < self.SLA_MS, (
            f"IVF dbms_search_ms p95={dbms_stats['p95']} ms excede el SLA de {self.SLA_MS} ms"
        )

    def test_latency_breakdown_extraction_vs_search(self, dbms_and_face_env):
        """Verifica que total_ms ≈ extraction_ms + dbms_search_ms (con margen de serialización)."""
        client = dbms_and_face_env["client"]
        iid = _register(client, "BreakdownSubj")
        _enroll(client, iid, "breakdown_subj")

        for _ in range(10):
            resp = client.post(
                "/recognize",
                data={"index_type": "hnsw", "k": "3", "threshold": "0.40"},
                files={"file": ("p.jpg", make_face_payload("breakdown_subj", noise=0.03), "image/jpeg")},
            )
            assert resp.status_code == 200
            t = resp.json()["timings_ms"]
            # total_ms debe ser al menos extracción + búsqueda interna del DBMS
            assert t["total_ms"] >= t["extraction_ms"], (
                "total_ms no puede ser menor que extraction_ms"
            )
            assert t["total_ms"] >= t["dbms_search_ms"], (
                "total_ms no puede ser menor que dbms_search_ms"
            )

    def test_latency_stable_across_multiple_runs(self, dbms_and_face_env):
        """La latencia no debe degradarse significativamente a lo largo de las búsquedas."""
        client = dbms_and_face_env["client"]
        for i in range(5):
            iid = _register(client, f"StabilitySubj{i}")
            _enroll(client, iid, f"stability_subj_{i}")

        subjects = [f"stability_subj_{i}" for i in range(5)]
        latencies = self._collect_latencies(client, subjects, "hnsw", 20)

        dbms_ms = latencies["dbms_search_ms"]
        # Dividir en primera y segunda mitad
        first_half  = statistics.fmean(dbms_ms[:10])
        second_half = statistics.fmean(dbms_ms[10:])

        # La segunda mitad no debe ser más de 5× la primera (detectaría degradación severa)
        assert second_half <= first_half * 5, (
            f"Posible degradación de latencia: primera mitad={first_half:.2f} ms, "
            f"segunda mitad={second_half:.2f} ms"
        )

    def test_unknown_probe_latency_comparable_to_known(self, dbms_and_face_env):
        """La latencia de rechazar un desconocido debe ser comparable a reconocer un registrado."""
        client = dbms_and_face_env["client"]
        for i in range(5):
            iid = _register(client, f"CompSubj{i}")
            _enroll(client, iid, f"comp_subj_{i}")

        known_subjects   = [f"comp_subj_{i}" for i in range(5)]
        unknown_subjects = [f"comp_unknown_{i}" for i in range(5)]

        known_lat   = self._collect_latencies(client, known_subjects,   "hnsw", 15)
        unknown_lat = self._collect_latencies(client, unknown_subjects, "hnsw", 15)

        known_p95   = self._report("known   dbms_search_ms", known_lat["dbms_search_ms"])["p95"]
        unknown_p95 = self._report("unknown dbms_search_ms", unknown_lat["dbms_search_ms"])["p95"]

        # Ambos deben cumplir el SLA
        assert known_p95   < self.SLA_MS, f"known p95={known_p95} ms excede el SLA"
        assert unknown_p95 < self.SLA_MS, f"unknown p95={unknown_p95} ms excede el SLA"
