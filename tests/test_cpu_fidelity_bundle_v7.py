from pathlib import Path
import subprocess
import sys

import pytest

from reproducibility.build_cpu_fidelity_bundle_v7 import (
    CPU_NODES,
    OUTPUT_PATHS,
    _data_policy,
    _is_mutable_output,
    _run_command,
)


def _file(path: str) -> dict:
    return {"path": path, "type": "file", "bytes": 1, "sha256": "0" * 64}


def test_cpu_fidelity_bundle_policy_allows_only_registered_raw_npz():
    entries = [
        _file(
            "bus_h2o/datasets_v7/formal_v2_zero/zero/cell/episode.full_service.npz"
        ),
        _file(
            "bus_h2o/datasets_v7/formal_v2_behavior/policy/cell/episode.full_service.npz"
        ),
        _file("bus_h2o/calibrated_env/observation_speed_profiles_v7_v1.npz"),
        _file("reproducibility/audit_simulator_fidelity_v7.py"),
    ]
    policy = _data_policy(entries)
    assert policy["training_hdf5_files"] == 0
    assert policy["model_checkpoint_files"] == 0
    assert policy["immutable_sumo_evidence_npz_files"] == 2


@pytest.mark.parametrize(
    "path",
    [
        "bus_h2o/datasets_v2/legacy.npz",
        "bus_h2o/datasets_v7/formal_v2_full/training.h5",
        "results/model.ckpt",
        "unregistered/evidence.npz",
    ],
)
def test_cpu_fidelity_bundle_policy_rejects_legacy_and_training_payloads(path):
    entries = [
        _file("bus_h2o/calibrated_env/observation_speed_profiles_v7_v1.npz"),
        _file(path),
    ]
    with pytest.raises(ValueError, match="data policy failed"):
        _data_policy(entries)


def test_cpu_fidelity_bundle_mutability_is_an_exact_allowlist():
    for path in OUTPUT_PATHS.values():
        assert _is_mutable_output(path.as_posix())
        assert _is_mutable_output(path.as_posix() + ".partial")
    assert not _is_mutable_output("results/unregistered.json")


def test_cpu_fidelity_commands_are_registered_to_six_cpu_nodes():
    assert CPU_NODES == (
        "node001",
        "node002",
        "node003",
        "node004",
        "node005",
        "node006",
    )
    command = _run_command("intervention", workers=64, server_label="node002")
    assert command[-2:] == ["--server-label", "node002"]
    assert command[command.index("--workers") + 1] == "64"
    with pytest.raises(ValueError, match="unsupported CPU fidelity node"):
        _run_command("zero", workers=30, server_label="local")


def test_cpu_fidelity_entrypoint_imports_from_an_arbitrary_working_directory(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "build_cpu_fidelity_bundle_v7.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{build,verify,run}" in completed.stdout
