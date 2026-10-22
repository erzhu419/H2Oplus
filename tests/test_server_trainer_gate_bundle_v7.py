from pathlib import Path
import subprocess
import sys

import pytest

from reproducibility.audit_device_cache_precision_v6 import (
    precision_audit_source_hashes,
)
from reproducibility.audit_reward_conditioning_v7 import (
    reward_audit_source_hashes,
)
from reproducibility.audit_transit_v7_dataset import (
    dataset_audit_source_hashes,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    entropy_pilot_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    reward_loss_source_hashes,
)
from reproducibility.server_trainer_gate_bundle_v7 import (
    INPUT_PATHS,
    OUTPUT_PATHS,
    _gate_command,
    _is_mutable_output,
    _require_build_interpreter,
)


def test_v7_trainer_gate_source_maps_are_workspace_relative():
    source_maps = (
        reward_audit_source_hashes(),
        dataset_audit_source_hashes(),
        reward_loss_source_hashes(),
        entropy_pilot_source_hashes(),
        precision_audit_source_hashes("v7"),
    )
    for source_map in source_maps:
        assert source_map
        assert all(not Path(path).is_absolute() for path in source_map)
        assert all(".." not in Path(path).parts for path in source_map)
        assert all(len(digest) == 64 for digest in source_map.values())


def test_v7_gate_bundle_has_exactly_two_v7_hdf5_inputs():
    datasets = [path for name, path in INPUT_PATHS.items() if name.endswith("dataset")]
    assert len(datasets) == 2
    assert all("datasets_v7" in path.parts for path in datasets)
    assert all(path.suffix == ".h5" for path in datasets)
    assert not any("snapshot" in part for path in datasets for part in path.parts)


def test_v7_gate_bundle_mutability_is_an_exact_allowlist():
    for path in OUTPUT_PATHS.values():
        assert _is_mutable_output(path.as_posix())
        assert _is_mutable_output(path.as_posix() + ".partial")
    assert not _is_mutable_output("results/unregistered.json")
    assert not _is_mutable_output(
        OUTPUT_PATHS["gate_chain"].with_name("other.json").as_posix()
    )


def test_v7_gate_bundle_runtime_lock_requires_registered_interpreter(tmp_path):
    _require_build_interpreter(Path(sys.executable))
    different = tmp_path / "different-python"
    different.write_bytes(b"not the active interpreter")
    with pytest.raises(RuntimeError, match="must be built by --python"):
        _require_build_interpreter(different)


def test_v7_gate_command_is_server_only_and_explicitly_routes_outputs():
    command = _gate_command(Path("/server/python"), "jtl110gpu2", 8)
    assert command[:3] == [
        "/server/python",
        "-u",
        "reproducibility/run_server_trainer_gates_v7.py",
    ]
    assert command[command.index("--server-label") + 1] == "jtl110gpu2"
    assert command[command.index("--device") + 1] == "cuda:0"
    for path in OUTPUT_PATHS.values():
        assert path.as_posix() in command


def test_server_gate_entrypoint_imports_from_an_arbitrary_working_directory(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "run_server_trainer_gates_v7.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--server-label" in completed.stdout
