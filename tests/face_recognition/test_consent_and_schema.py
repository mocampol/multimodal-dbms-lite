from __future__ import annotations

import pytest

from backend.exceptions import ConsentRequiredError, IdentityWithdrawnError
from backend.schema_setup import CONSENTS_TABLE, IDENTITIES_TABLE, SAMPLES_TABLE
from face_helpers import make_face_payload
from face_common import (
    build_benchmark_manifest,
    collect_celeba_subjects,
    collect_lfw_subjects,
)


def test_schema_creates_face_tables_and_dual_indexes(dbms_and_face_env):
    service = dbms_and_face_env["service"]
    cat = dbms_and_face_env["catalog"]

    res = service.ensure_ready()
    assert set(res["created_tables"]) == {IDENTITIES_TABLE, CONSENTS_TABLE, SAMPLES_TABLE}
    assert cat.table_exists(IDENTITIES_TABLE)
    assert cat.table_exists(CONSENTS_TABLE)
    assert cat.table_exists(SAMPLES_TABLE)
    assert cat.get_physical_index(SAMPLES_TABLE, "embedding", index_type="hnsw") is not None
    assert cat.get_physical_index(SAMPLES_TABLE, "embedding", index_type="ivf") is not None


def test_participant_without_consent_is_blocked_everywhere(dbms_and_face_env):
    service = dbms_and_face_env["service"]

    with pytest.raises(ConsentRequiredError):
        service.register_identity(
            source="participant",
            display_label="Sin Consentimiento",
            consent_granted=False,
        )

    pending = service.register_identity(
        source="participant",
        display_label="Pendiente",
        consent_granted=False,
        allow_pending=True,
    )
    assert pending["consent_status"] == "pending"

    with pytest.raises(ConsentRequiredError):
        service.enroll_face(
            identity_id=pending["identity_id"],
            image_bytes=make_face_payload("pendiente").getvalue(),
        )


def test_public_datasets_force_dataset_only_status_and_redacted_labels(dbms_and_face_env):
    service = dbms_and_face_env["service"]

    lfw = service.register_identity(
        source="public_lfw",
        source_subject_id="lfw:Person_001",
        display_label="No Debe Guardarse",
        consent_granted=True,
    )
    assert lfw["consent_status"] == "dataset_only"
    assert lfw["display_label"] is None

    celeba = service.register_identity(
        source="public_celeba",
        source_subject_id="celeba:42",
        display_label="Tampoco",
        consent_granted=True,
    )
    assert celeba["consent_status"] == "dataset_only"
    assert celeba["display_label"] is None

    service.enroll_face(
        identity_id=lfw["identity_id"],
        image_bytes=make_face_payload("lfw:Person_001").getvalue(),
    )

    # En modo live (allow_dataset_identities=False), un sujeto dataset_only no se retorna como registrado
    live_res = service.recognize_face(
        image_bytes=make_face_payload("lfw:Person_001").getvalue(),
        allow_dataset_identities=False,
    )
    assert live_res["registered"] is False
    assert live_res["label"] == "desconocido"

    # En modo benchmark (allow_dataset_identities=True), sí se evalúa
    bench_res = service.recognize_face(
        image_bytes=make_face_payload("lfw:Person_001").getvalue(),
        allow_dataset_identities=True,
    )
    assert bench_res["registered"] is True
    assert bench_res["identity_id"] == lfw["identity_id"]


def test_withdraw_consent_blocks_future_enrollment(dbms_and_face_env):
    service = dbms_and_face_env["service"]
    created = service.register_identity(
        source="participant",
        display_label="Carlos",
        consent_granted=True,
    )
    iid = created["identity_id"]
    service.enroll_face(identity_id=iid, image_bytes=make_face_payload("carlos").getvalue())

    revoked = service.withdraw_identity(iid)
    assert revoked["consent_status"] == "withdrawn"
    assert revoked["deleted_samples"] == 1

    with pytest.raises(IdentityWithdrawnError):
        service.enroll_face(identity_id=iid, image_bytes=make_face_payload("carlos").getvalue())


def test_prepare_datasets_builds_lfw_and_celeba_manifests(tmp_path):
    # Sintético LFW: 3 sujetos x 3 imágenes
    lfw_root = tmp_path / "lfw"
    for subj in ("subj_a", "subj_b", "subj_c"):
        d = lfw_root / subj
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{subj}_{i}.jpg").write_bytes(f"{subj}-{i}".encode())

    lfw_subjects = collect_lfw_subjects(lfw_root)
    assert len(lfw_subjects) == 3
    lfw_manifest = build_benchmark_manifest(
        lfw_subjects,
        dataset_name="lfw",
        source="public_lfw",
        min_images=3,
        max_known_identities=2,
        max_unknown_identities=1,
        enroll_per_identity=2,
        max_probes_per_identity=2,
        max_unknown_probes=5,
        repo_root=tmp_path,
    )
    assert lfw_manifest["counts"] == {"enroll": 4, "probe_known": 2, "probe_unknown": 2}

    # Sintético CelebA: identity_CelebA.txt + img_align_celeba/
    celeba_root = tmp_path / "celeba"
    img_dir = celeba_root / "img_align_celeba"
    img_dir.mkdir(parents=True)
    lines = []
    idx = 1
    for cid in ("101", "202"):
        for _ in range(3):
            fname = f"{idx:06d}.jpg"
            (img_dir / fname).write_bytes(f"img-{idx}".encode())
            lines.append(f"{fname} {cid}")
            idx += 1
    (celeba_root / "identity_CelebA.txt").write_text("\n".join(lines), encoding="utf-8")

    celeba_subjects = collect_celeba_subjects(celeba_root)
    assert set(celeba_subjects) == {"celeba:101", "celeba:202"}
    celeba_manifest = build_benchmark_manifest(
        celeba_subjects,
        dataset_name="celeba",
        source="public_celeba",
        min_images=3,
        max_known_identities=1,
        max_unknown_identities=1,
        enroll_per_identity=1,
        max_probes_per_identity=2,
        max_unknown_probes=3,
        repo_root=tmp_path,
    )
    assert celeba_manifest["counts"] == {"enroll": 1, "probe_known": 2, "probe_unknown": 2}
