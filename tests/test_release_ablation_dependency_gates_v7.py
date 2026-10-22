import hashlib
import json

import pytest

from reproducibility.release_ablation_dependency_gates_v7 import release, validate


def _manifest():
    return {
        "status": "verified",
        "artifact_integrity_verified": True,
        "errors": [],
        "partition_runs_disjoint": True,
        "independent_matrix_audit": {
            "passed": True,
            "jobs_audited": 10,
            "sha256": "a" * 64,
        },
        "merged_results": {
            "runs": [
                f"{method}_seed{seed}"
                for method in ("h2oplus-no-ratio", "h2oplus-no-warmstart")
                for seed in range(5101, 5106)
            ]
        },
        "scheduler_evidence": {
            "tasks": {
                "t82053": {"status": "done", "exit_code": 0},
                "t82054": {"status": "done", "exit_code": 0},
            }
        },
        "mechanism_analysis": {
            "sha256": "b" * 64,
            "methods": [
                "h2oplus",
                "h2oplus-no-ratio",
                "h2oplus-no-warmstart",
            ],
            "seeds": list(range(5101, 5106)),
            "supports_held_out_model_effect_claim": False,
        },
    }


def _verification(manifest_path):
    return {
        "schema_version": "transit-formal-ablation-manifest-verification-v7-v1",
        "passed": True,
        "errors": [],
        "manifest": {
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }


def test_verified_ablation_manifest_passes_gate_validation():
    validate(_manifest())


def test_nonzero_dispatch_exit_blocks_gate():
    manifest = _manifest()
    manifest["scheduler_evidence"]["tasks"]["t82053"]["exit_code"] = 1
    with pytest.raises(RuntimeError, match="exit-0"):
        validate(manifest)


def test_direct_backend_completion_allows_stale_queue_status():
    manifest = _manifest()
    task = manifest["scheduler_evidence"]["tasks"]["t82053"]
    task.update(
        {
            "status": "running",
            "exit_code": None,
            "completion_evidence": {
                "verified": True,
                "backend_exit": {"exit_code": 0},
                "host_execution": {"completed": True},
            },
        }
    )

    validate(manifest)


def test_release_preflights_both_gate_paths(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(_verification(manifest_path)))
    gate_dir = tmp_path / "gates"
    gate_dir.mkdir()
    occupied = gate_dir / "ablation_verified_jtl110gpu2.ready"
    occupied.write_text("existing\n")

    with pytest.raises(FileExistsError, match="occupied"):
        release(manifest_path, verification_path, gate_dir)

    assert not (gate_dir / "ablation_verified_jtl110gpu.ready").exists()
    assert not (gate_dir / "ablation_verified_jtl110gpu.ready.partial").exists()
    assert occupied.read_text() == "existing\n"


def test_release_creates_both_nonempty_disclosure_gates(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(_verification(manifest_path)))
    gate_dir = tmp_path / "gates"

    paths = release(manifest_path, verification_path, gate_dir)

    assert len(paths) == 2
    assert all(path.stat().st_size > 0 for path in paths)
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["artifact_integrity"] == "verified"
        assert payload["manifest_verification"]["sha256"]
        analysis = payload["mechanism_analysis"]
        assert analysis["supports_held_out_model_effect_claim"] is False
        assert analysis["does_not_select_or_replace_formal_results"] is True
        assert analysis["methods"] == [
            "h2oplus",
            "h2oplus-no-ratio",
            "h2oplus-no-warmstart",
        ]
    assert not list(gate_dir.glob("*.partial"))


def test_release_rejects_stale_manifest_verification(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    verification = _verification(manifest_path)
    verification["manifest"]["sha256"] = "0" * 64
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(verification))

    with pytest.raises(RuntimeError, match="independent passing verification"):
        release(manifest_path, verification_path, tmp_path / "gates")
