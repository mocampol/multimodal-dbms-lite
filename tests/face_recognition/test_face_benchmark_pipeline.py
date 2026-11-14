from __future__ import annotations

from face_helpers import make_face_payload
from benchmark_face import (
    DEFAULT_THRESHOLDS,
    format_markdown_report,
    recommend_threshold,
    run_benchmark,
    write_csv_results,
)
from face_common import (
    build_benchmark_manifest,
    collect_celeba_subjects,
    collect_lfw_subjects,
    ensure_face_dataset,
    load_manifest_into_dbms,
)


def test_load_faces_and_benchmark_pipeline_end_to_end(dbms_and_face_env, tmp_path):
    service = dbms_and_face_env["service"]

    celeba_root = tmp_path / "celeba"
    img_dir = celeba_root / "img_align_celeba"
    img_dir.mkdir(parents=True)
    lines = []
    idx = 1
    for cid in ("s1", "s2", "s3_unknown"):
        for rep in range(3):
            fname = f"{idx:06d}.jpg"
            (img_dir / fname).write_bytes(
                make_face_payload(cid, noise=0.03, seed_offset=rep).getvalue()
            )
            lines.append(f"{fname} {cid}")
            idx += 1
    (celeba_root / "identity_CelebA.txt").write_text("\n".join(lines), encoding="utf-8")

    subjects = collect_celeba_subjects(celeba_root)
    manifest = build_benchmark_manifest(
        subjects,
        dataset_name="celeba",
        source="public_celeba",
        min_images=3,
        max_known_identities=2,
        max_unknown_identities=1,
        enroll_per_identity=2,
        max_probes_per_identity=1,
        max_unknown_probes=2,
        repo_root=tmp_path,
    )

    summary = load_manifest_into_dbms(manifest, service)
    assert summary["identities_registered"] == 2
    assert summary["samples_enrolled"] == 4
    assert summary["skipped"] == 0

    rows = run_benchmark(
        manifest,
        service,
        index_types=("hnsw", "ivf"),
        k_values=(1, 3),
        thresholds=(0.45,),
        allow_dataset_identities=True,
    )
    assert len(rows) == 4
    for r in rows:
        assert r["known_probes"] == manifest["counts"]["probe_known"] == 2
        assert r["unknown_probes"] == manifest["counts"]["probe_unknown"] == 1
        assert r["top1_accuracy"] == 1.0
        assert r["true_accept_rate"] == 1.0
        assert r["false_id_swap_rate"] == 0.0
        assert r["unknown_reject_rate"] == 1.0
        assert r["f1_score"] == 1.0
        assert r["dbms_search_ms_p95"] < 100.0

    rec = recommend_threshold(rows, max_far=0.02)
    assert rec["met_far_target"] is True
    assert rec["recommended"]["threshold"] == 0.45
    md = format_markdown_report(rows, rec)
    assert "Umbral recomendado" in md


def test_lfw_nested_seed_and_default_threshold_sweep(dbms_and_face_env, tmp_path):
    service = dbms_and_face_env["service"]

    lfw_nested = tmp_path / "lfw_root" / "lfw-deepfunneled"
    for person in ("Alice", "Bob", "Carol", "Dave"):
        pdir = lfw_nested / person
        pdir.mkdir(parents=True)
        for rep in range(3):
            (pdir / f"{person}_{rep:04d}.jpg").write_bytes(
                make_face_payload(person, noise=0.04, seed_offset=rep).getvalue()
            )

    subjects = collect_lfw_subjects(tmp_path / "lfw_root")
    assert set(subjects.keys()) == {"lfw:Alice", "lfw:Bob", "lfw:Carol", "lfw:Dave"}

    m1 = build_benchmark_manifest(
        subjects,
        dataset_name="lfw",
        source="public_lfw",
        min_images=3,
        max_known_identities=2,
        max_unknown_identities=2,
        enroll_per_identity=2,
        max_probes_per_identity=1,
        max_unknown_probes=2,
        seed=42,
        repo_root=tmp_path,
    )
    m2 = build_benchmark_manifest(
        subjects,
        dataset_name="lfw",
        source="public_lfw",
        min_images=3,
        max_known_identities=2,
        max_unknown_identities=2,
        enroll_per_identity=2,
        max_probes_per_identity=1,
        max_unknown_probes=2,
        seed=42,
        repo_root=tmp_path,
    )
    assert [e["source_subject_id"] for e in m1["entries"]] == [
        e["source_subject_id"] for e in m2["entries"]
    ]

    summary = load_manifest_into_dbms(m1, service)
    assert summary["identities_registered"] == 2
    assert summary["samples_enrolled"] == 4

    assert DEFAULT_THRESHOLDS == (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    rows = run_benchmark(
        m1,
        service,
        index_types=("hnsw", "ivf"),
        k_values=(3,),
        aggregations=("max", "mean_top"),
        allow_dataset_identities=True,
    )
    # 2 índices * 1 k * 2 aggregations * 9 thresholds = 36 filas
    assert len(rows) == 36
    rec = recommend_threshold(rows, max_far=0.02)
    assert rec["recommended"] is not None
    assert rec["met_far_target"] is True
    csv_out = tmp_path / "face_results.csv"
    write_csv_results(rows, csv_out)
    assert csv_out.is_file()
    assert len(csv_out.read_text(encoding="utf-8").splitlines()) == 37


def test_ensure_face_dataset_caches_and_auto_extracts(tmp_path):
    import io
    import tarfile
    import zipfile

    # 1. LFW missing -> downloader creates .tgz -> auto-extracted
    lfw_dir = tmp_path / "lfw"

    def _fake_lfw_dl(_urls, dest, _sha):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:gz") as tf:
            for idx in range(3):
                payload = make_face_payload("Alice", noise=0.01, seed_offset=idx).getvalue()
                info = tarfile.TarInfo(name=f"lfw_funneled/Alice/Alice_{idx:04d}.jpg")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        return dest

    resolved_lfw = ensure_face_dataset("lfw", lfw_dir, downloader=_fake_lfw_dl)
    assert resolved_lfw == lfw_dir
    assert "lfw:Alice" in collect_lfw_subjects(lfw_dir)

    # 2. Already cached -> downloader must NOT be invoked
    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("No debería descargar si el dataset ya existe")

    assert ensure_face_dataset("lfw", lfw_dir, downloader=_fail_if_called) == lfw_dir

    # 3. CelebA missing -> downloads identity_CelebA.txt + img_align_celeba.zip
    celeba_dir = tmp_path / "celeba"

    def _fake_celeba_dl(_urls, dest, _sha):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.name == "identity_CelebA.txt":
            dest.write_text("000001.jpg 7\n000002.jpg 7\n", encoding="utf-8")
        else:
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("000001.jpg", make_face_payload("s7", seed_offset=1).getvalue())
                zf.writestr("000002.jpg", make_face_payload("s7", seed_offset=2).getvalue())
        return dest

    resolved_celeba = ensure_face_dataset("celeba", celeba_dir, downloader=_fake_celeba_dl)
    assert resolved_celeba == celeba_dir
    assert len(collect_celeba_subjects(celeba_dir)["celeba:7"]) == 2
    assert ensure_face_dataset("celeba", celeba_dir, downloader=_fail_if_called) == celeba_dir
