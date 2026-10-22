import json

import pytest

from reproducibility.release_baseline_dependency_gates_v7 import release, validate


def _manifest():
    return {
        "status": "verified",
        "artifact_integrity_verified": True,
        "errors": [],
        "partition_runs_disjoint": True,
        "independent_matrix_audit": {
            "passed": True,
            "jobs_audited": 35,
            "sha256": "a" * 64,
        },
        "merged_results": {"runs": [f"run-{index}" for index in range(35)]},
        "scheduler_evidence": {
            "tasks": {
                "t82016": {"status": "done", "exit_code": 0},
                "t82017": {"status": "done", "exit_code": 0},
            }
        },
        "scientific_stability": {
            "passed": False,
            "failed_seeds": [5101, 5103, 5104, 5105],
            "passed_seeds": [5102],
            "sha256": "b" * 64,
        },
    }


def test_scientific_stability_failure_does_not_block_integrity_gate():
    validate(_manifest())


def test_nonzero_dispatch_exit_blocks_gate():
    manifest = _manifest()
    manifest["scheduler_evidence"]["tasks"]["t82016"]["exit_code"] = 1
    try:
        validate(manifest)
    except RuntimeError as error:
        assert "exit-0" in str(error)
    else:
        raise AssertionError("gate validation accepted a nonzero baseline dispatch")


def test_release_preflights_both_gate_paths(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    gate_dir = tmp_path / "gates"
    gate_dir.mkdir()
    occupied = gate_dir / "baselines_verified_jtl110gpu2.ready"
    occupied.write_text("existing\n")

    with pytest.raises(FileExistsError, match="occupied"):
        release(manifest_path, gate_dir)

    assert not (gate_dir / "baselines_verified_jtl110gpu.ready").exists()
    assert not (gate_dir / "baselines_verified_jtl110gpu.ready.partial").exists()
    assert occupied.read_text() == "existing\n"


def test_release_creates_both_nonempty_disclosure_gates(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    gate_dir = tmp_path / "gates"

    paths = release(manifest_path, gate_dir)

    assert len(paths) == 2
    assert all(path.stat().st_size > 0 for path in paths)
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["artifact_integrity"] == "verified"
        assert payload["rlpd_scientific_stability"]["passed"] is False
        assert payload["rlpd_scientific_stability"]["failed_seeds"] == [
            5101,
            5103,
            5104,
            5105,
        ]
    assert not list(gate_dir.glob("*.partial"))
