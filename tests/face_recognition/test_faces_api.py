from __future__ import annotations

import io

from face_helpers import ASGITestClient, make_face_payload


def test_empty_db_recognition_returns_desconocido(dbms_and_face_env):
    client = dbms_and_face_env["client"]

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    resp = client.post(
        "/recognize",
        files={"file": ("probe.jpg", make_face_payload("desconocido_1"), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["registered"] is False
    assert body["matched"] is False
    assert body["status"] == "unknown"
    assert body["label"] == "desconocido"
    assert body["identity_id"] is None
    assert body["score"] == 0.0


def test_full_enroll_recognize_unknown_threshold_and_withdraw_flow(dbms_and_face_env):
    client = dbms_and_face_env["client"]

    # 1. Bloquear registro sin consentimiento explícito
    r_no_consent = client.post(
        "/identities",
        json={"source": "participant", "display_label": "Ana", "consent_granted": False},
    )
    assert r_no_consent.status_code == 403
    assert r_no_consent.json()["code"] == "consent_required"

    # 2. Registrar dos identidades con consentimiento explícito
    r_ana = client.post(
        "/identities",
        json={
            "source": "participant",
            "display_label": "Ana",
            "consent_granted": True,
            "purpose": "control_acceso_lab",
        },
    )
    assert r_ana.status_code == 201
    ana_id = r_ana.json()["identity_id"]

    r_luis = client.post(
        "/identities",
        json={
            "source": "participant",
            "display_label": "Luis",
            "consent_granted": True,
        },
    )
    assert r_luis.status_code == 201
    luis_id = r_luis.json()["identity_id"]

    # 3. Rechazar imágenes sin rostro o con múltiples rostros en enrolamiento
    r_zero = client.post(
        "/enroll",
        data={"identity_id": str(ana_id)},
        files={"file": ("wall.jpg", io.BytesIO(b"NO_FACE"), "image/jpeg")},
    )
    assert r_zero.status_code == 422
    assert r_zero.json()["code"] == "no_face_detected"

    r_multi = client.post(
        "/enroll",
        data={"identity_id": str(ana_id), "face_policy": "strict_single"},
        files={"file": ("group.jpg", io.BytesIO(b"MULTI_FACE"), "image/jpeg")},
    )
    assert r_multi.status_code == 422
    assert r_multi.json()["code"] == "multiple_faces_detected"

    # 4. Enrolar muestras válidas para Ana y Luis
    for offset in range(2):
        r_enr = client.post(
            "/enroll",
            data={"identity_id": str(ana_id), "split": "enroll"},
            files={
                "file": (
                    f"ana_{offset}.jpg",
                    make_face_payload("ana_real", noise=0.04, seed_offset=offset),
                    "image/jpeg",
                )
            },
        )
        assert r_enr.status_code == 201
        assert r_enr.json()["embedding_dim"] == 128

    r_enr_luis = client.post(
        "/enroll",
        data={"identity_id": str(luis_id), "split": "enroll"},
        files={"file": ("luis_0.jpg", make_face_payload("luis_real", noise=0.03), "image/jpeg")},
    )
    assert r_enr_luis.status_code == 201

    # 5. Reconocer a Ana con HNSW y con IVF usando k y parámetros dinámicos
    for idx_type in ("hnsw", "ivf"):
        r_match = client.post(
            "/recognize",
            data={
                "k": "3",
                "threshold": "0.45",
                "index_type": idx_type,
                "ef_search": "64",
                "n_probe": "2",
            },
            files={"file": ("ana_probe.jpg", make_face_payload("ana_real", noise=0.05, seed_offset=99), "image/jpeg")},
        )
        assert r_match.status_code == 200
        m = r_match.json()
        assert m["registered"] is True
        assert m["matched"] is True
        assert m["status"] == "registered"
        assert m["label"] == "Ana"
        assert m["identity_id"] == ana_id
        assert m["used_index"] is True
        assert m["score"] >= 0.45
        assert m["k"] == 3

    # 6. Persona no registrada debe responder explícitamente "desconocido" sin match
    r_unk = client.post(
        "/recognize",
        data={"k": "3", "threshold": "0.40"},
        files={"file": ("intruder.jpg", make_face_payload("intruso_x"), "image/jpeg")},
    )
    assert r_unk.status_code == 200
    u = r_unk.json()
    assert u["registered"] is False
    assert u["matched"] is False
    assert u["status"] == "unknown"
    assert u["label"] == "desconocido"
    assert u["identity_id"] is None
    assert u["score"] < 0.40

    # 7. Umbral estricto (0.9999) fuerza rechazo incluso sobre muestra con ruido ligero
    r_strict = client.post(
        "/recognize",
        data={"k": "3", "threshold": "0.9999"},
        files={"file": ("ana_noisy.jpg", make_face_payload("ana_real", noise=0.15, seed_offset=5), "image/jpeg")},
    )
    assert r_strict.status_code == 200
    assert r_strict.json()["status"] == "unknown"
    assert r_strict.json()["label"] == "desconocido"
    assert r_strict.json()["identity_id"] is None

    # 8. Revocar consentimiento de Ana elimina sus muestras del índice y pasa a "desconocido"
    r_del = client.delete(f"/identities/{ana_id}")
    assert r_del.status_code == 200
    assert r_del.json()["consent_status"] == "withdrawn"
    assert r_del.json()["deleted_samples"] == 2

    r_after_del = client.post(
        "/recognize",
        data={"k": "3", "threshold": "0.40"},
        files={"file": ("ana_probe_after.jpg", make_face_payload("ana_real", noise=0.02), "image/jpeg")},
    )
    assert r_after_del.status_code == 200
    assert r_after_del.json()["registered"] is False
    assert r_after_del.json()["label"] == "desconocido"
    assert r_after_del.json()["identity_id"] is None

    # 9. Reconstruir índice IVF dinámicamente tras operaciones y comprobar que Luis sigue encontrado
    r_rebuild = client.post("/indexes/rebuild", json={"index_type": "ivf", "n_lists": 1})
    assert r_rebuild.status_code == 200
    assert r_rebuild.json()["index_type"] == "ivf"

    r_luis_probe = client.post(
        "/recognize",
        data={"index_type": "ivf", "k": "2", "threshold": "0.40"},
        files={"file": ("luis_probe.jpg", make_face_payload("luis_real", noise=0.02), "image/jpeg")},
    )
    assert r_luis_probe.status_code == 200
    assert r_luis_probe.json()["registered"] is True
    assert r_luis_probe.json()["label"] == "Luis"


def test_dynamic_512d_pipeline_end_to_end(dbms_and_face_env):
    face_app_512, _service_512, client_512_dbms = dbms_and_face_env["build_app"](dim=512)
    try:
        client = ASGITestClient(face_app_512)
        r_id = client.post(
            "/identities",
            json={"source": "participant", "display_label": "Elena512", "consent_granted": True},
        )
        assert r_id.status_code == 201
        elena_id = r_id.json()["identity_id"]

        r_enr = client.post(
            "/enroll",
            data={"identity_id": str(elena_id)},
            files={"file": ("elena.jpg", make_face_payload("elena_512"), "image/jpeg")},
        )
        assert r_enr.status_code == 201
        assert r_enr.json()["embedding_dim"] == 512

        r_rec = client.post(
            "/recognize",
            data={"index_type": "hnsw", "k": "2", "threshold": "0.45"},
            files={"file": ("elena_q.jpg", make_face_payload("elena_512", noise=0.05), "image/jpeg")},
        )
        assert r_rec.status_code == 200
        body = r_rec.json()
        assert body["embedding_dim"] == 512
        assert body["registered"] is True
        assert body["label"] == "Elena512"
    finally:
        client_512_dbms.close()


def test_purge_all_faces_samples_only_and_full_reset(dbms_and_face_env):
    client = dbms_and_face_env["client"]

    r_ana = client.post(
        "/identities",
        json={"source": "participant", "display_label": "Ana", "consent_granted": True},
    )
    assert r_ana.status_code == 201
    ana_id = r_ana.json()["identity_id"]

    r_enr = client.post(
        "/enroll",
        data={"identity_id": str(ana_id)},
        files={"file": ("ana.jpg", make_face_payload("ana_purge"), "image/jpeg")},
    )
    assert r_enr.status_code == 201

    # 1. Borrar solo muestras: conserva identidad con sample_count=0
    r_samples_only = client.delete("/identities", params={"samples_only": "true"})
    assert r_samples_only.status_code == 200
    assert r_samples_only.json()["samples_only"] is True
    assert r_samples_only.json()["deleted_samples"] >= 1
    assert r_samples_only.json()["deleted_identities"] == 0

    listed_after_samples = client.get("/identities").json()
    assert listed_after_samples["count"] == 1
    assert listed_after_samples["identities"][0]["sample_count"] == 0

    # Re-enrolar y hacer purga completa
    r_enr2 = client.post(
        "/enroll",
        data={"identity_id": str(ana_id)},
        files={"file": ("ana2.jpg", make_face_payload("ana_purge"), "image/jpeg")},
    )
    assert r_enr2.status_code == 201

    r_full = client.delete("/identities")
    assert r_full.status_code == 200
    assert r_full.json()["samples_only"] is False
    assert r_full.json()["deleted_samples"] >= 1
    assert r_full.json()["deleted_identities"] >= 1

    assert client.get("/identities").json()["count"] == 0
    probe = client.post(
        "/recognize",
        files={"file": ("probe.jpg", make_face_payload("ana_purge"), "image/jpeg")},
    )
    assert probe.status_code == 200
    assert probe.json()["registered"] is False
    assert probe.json()["label"] == "desconocido"
