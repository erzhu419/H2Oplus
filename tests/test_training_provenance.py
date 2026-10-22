import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from SimpleSAC.train_transit_v3 import (
    GPULease,
    NVIDIA_SMI_FALLBACKS,
    RESUME_IMMUTABLE_ARGUMENTS,
    _compiler_fingerprint,
    _git_provenance,
    reconcile_metrics,
    resolve_nvidia_smi,
    sha256,
    simulator_input_hashes,
    validate_resume,
)


def test_git_provenance_treats_missing_cli_as_optional(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)

    assert _git_provenance() == {
        "cli_available": False,
        "commit": None,
        "status": None,
        "errors": ["git executable is unavailable"],
    }


def test_git_provenance_records_nonzero_probe_without_crashing(
    tmp_path, monkeypatch
):
    executable = tmp_path / "git"
    executable.write_text("#!/bin/sh\necho not-a-repository >&2\nexit 128\n")
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _: str(executable))

    observed = _git_provenance()

    assert observed["cli_available"] is True
    assert observed["commit"] is None
    assert observed["status"] is None
    assert observed["errors"] == [
        "commit: not-a-repository",
        "status: not-a-repository",
    ]


def test_nvidia_smi_fallback_covers_wsl_driver_location(tmp_path, monkeypatch):
    executable = tmp_path / "nvidia-smi"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    assert resolve_nvidia_smi((executable,)) == str(executable)
    assert Path("/usr/lib/wsl/lib/nvidia-smi") in NVIDIA_SMI_FALLBACKS


def test_nvidia_smi_fallback_rejects_non_executable_file(tmp_path, monkeypatch):
    candidate = tmp_path / "nvidia-smi"
    candidate.write_text("not executable\n")
    candidate.chmod(0o644)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    # DrvFS can report every file as executable regardless of chmod. This unit
    # test controls the operating-system permission answer explicitly.
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)

    assert resolve_nvidia_smi((candidate,)) is None


def test_gpu_lease_queries_with_resolved_nvidia_smi(tmp_path, monkeypatch):
    executable = tmp_path / "nvidia-smi"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "SimpleSAC.train_transit_v3.resolve_nvidia_smi",
        lambda: str(executable),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unit-test-gpu")

    lease = GPULease("cuda:0", allow_shared=False)
    try:
        assert lease.physical_gpu == "unit-test-gpu"
        assert lease.nvidia_smi == str(executable)
    finally:
        lease.close()


def test_gpu_lease_enforces_exact_shared_process_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "SimpleSAC.train_transit_v3.resolve_nvidia_smi", lambda: "/bin/true"
    )
    monkeypatch.setattr("SimpleSAC.train_transit_v3.GPU_LEASE_ROOT", tmp_path)
    monkeypatch.setattr("SimpleSAC.train_transit_v3._run", lambda _: "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "quota-gpu")

    first = GPULease("cuda:0", max_canonical_processes_per_gpu=2)
    second = GPULease("cuda:0", max_canonical_processes_per_gpu=2)
    try:
        assert {first.slot_index, second.slot_index} == {0, 1}
        with pytest.raises(RuntimeError, match="hard quota of 2"):
            GPULease("cuda:0", max_canonical_processes_per_gpu=2)
    finally:
        second.close()
        first.close()

    replacement = GPULease("cuda:0", max_canonical_processes_per_gpu=2)
    replacement.close()


def test_gpu_lease_rejects_mixed_quota_contracts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "SimpleSAC.train_transit_v3.resolve_nvidia_smi", lambda: "/bin/true"
    )
    monkeypatch.setattr("SimpleSAC.train_transit_v3.GPU_LEASE_ROOT", tmp_path)
    monkeypatch.setattr("SimpleSAC.train_transit_v3._run", lambda _: "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "mixed-quota-gpu")

    shared = GPULease("cuda:0", max_canonical_processes_per_gpu=2)
    try:
        with pytest.raises(RuntimeError, match="different per-GPU quota"):
            GPULease("cuda:0", max_canonical_processes_per_gpu=1)
    finally:
        shared.close()


def test_compiler_required_identity_is_independent_of_include_location(
    tmp_path, monkeypatch
):
    first = tmp_path / "first" / "include"
    second = tmp_path / "second" / "include"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "crypt.h").write_text("/* pinned header */\n")
    (second / "crypt.h").write_text("/* pinned header */\n")
    monkeypatch.setenv("CC", "/bin/true")

    monkeypatch.setenv("CPATH", str(first))
    first_required, first_host = _compiler_fingerprint()
    monkeypatch.setenv("CPATH", str(second))
    second_required, second_host = _compiler_fingerprint()

    assert first_required == second_required
    assert first_host["cpath"]["value"] != second_host["cpath"]["value"]
    assert first_required["cpath"]["entries"][0]["crypt_h_sha256"] == sha256(
        first / "crypt.h"
    )


def _make_simulator_tree(tmp_path):
    sim_env = tmp_path / "sim"
    line_dir = sim_env / "data" / "7X"
    line_dir.mkdir(parents=True)
    (sim_env / "config.json").write_text('{"time_step": 1}')
    for name in (
        "stop_news.xlsx",
        "route_news.xlsx",
        "time_table.xlsx",
        "passenger_OD.xlsx",
    ):
        (line_dir / name).write_bytes(f"base:{name}".encode())
    calibrated = [
        line_dir / "time_table_sumo.xlsx",
        line_dir / "passenger_OD_sumo.xlsx",
    ]
    for path in calibrated:
        path.write_bytes(f"calibrated:{path.name}".encode())
    report = {
        "schema_version": "transit-simulator-demand-calibration-v1",
        "source_manifest_sha256": "manifest",
        "source_bus_objects_sha256": "buses",
        "output_sha256": {str(path.resolve()): sha256(path) for path in calibrated},
    }
    (sim_env / "simulator_demand_calibration.json").write_text(json.dumps(report))
    (sim_env / "segment_runtime_calibration_v4.json").write_text(
        json.dumps(
            {"schema_version": "transit-simulator-runtime-calibration-v4"}
        )
    )
    network = tmp_path / "network.xml"
    network.write_text("<network />")
    return sim_env, network, line_dir


def test_simulator_hashes_only_active_inputs_and_checks_calibration(tmp_path):
    sim_env, network, line_dir = _make_simulator_tree(tmp_path)
    initial = simulator_input_hashes(sim_env, network_path=network)
    assert initial["line_ids"] == ["7X"]
    assert len(initial["runtime_files_sha256"]) == 7

    (line_dir / "route_news_old.xlsx").write_bytes(b"ignored backup")
    ignored_backup = simulator_input_hashes(sim_env, network_path=network)
    assert ignored_backup["aggregate_sha256"] == initial["aggregate_sha256"]

    (line_dir / "route_news.xlsx").write_bytes(b"changed active route")
    changed_route = simulator_input_hashes(sim_env, network_path=network)
    assert changed_route["runtime_sha256"] != initial["runtime_sha256"]

    (line_dir / "passenger_OD_sumo.xlsx").write_bytes(b"tampered calibration")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        simulator_input_hashes(sim_env, network_path=network)


def test_v7_v2_runtime_hashing_rejects_failed_or_legacy_calibration(tmp_path):
    sim_env, network, _ = _make_simulator_tree(tmp_path)
    runtime = sim_env / "segment_runtime_calibration_v7_v2.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-runtime-calibration-v7-v2",
                "coverage_gate": {"passed": True},
            }
        )
    )
    valid = simulator_input_hashes(
        sim_env, network_path=network, runtime_calibration_version="v7_v2"
    )
    assert runtime.name in valid["runtime_files_sha256"]

    payload = json.loads(runtime.read_text())
    payload["coverage_gate"]["passed"] = False
    runtime.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="failed its coverage gate"):
        simulator_input_hashes(
            sim_env,
            network_path=network,
            runtime_calibration_version="v7_v2",
        )

    payload["coverage_gate"]["passed"] = True
    payload["schema_version"] = "transit-simulator-runtime-calibration-v7"
    runtime.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported schema"):
        simulator_input_hashes(
            sim_env,
            network_path=network,
            runtime_calibration_version="v7_v2",
        )


def test_v7_v4_runtime_hashing_binds_dynamic_speed_profile(tmp_path):
    sim_env, network, _ = _make_simulator_tree(tmp_path)
    profile = sim_env / "observation_speed_profiles_v7_v1.npz"
    profile.write_bytes(b"profile-v1")
    runtime = sim_env / "segment_runtime_calibration_v7_v4.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-runtime-calibration-v7-v4",
                "coverage_gate": {"passed": True},
                "backward_headway_observation": {
                    "dynamic_speed_profiles": {
                        "schema_version": (
                            "transit-simulator-observation-speed-profiles-v7-v1"
                        ),
                        "artifact": profile.name,
                        "sha256": sha256(profile),
                        "coverage_gate": {"passed": True},
                    }
                },
            }
        )
    )

    valid = simulator_input_hashes(
        sim_env, network_path=network, runtime_calibration_version="v7_v4"
    )
    assert valid["runtime_files_sha256"][profile.name] == sha256(profile)

    profile.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="profile hash changed"):
        simulator_input_hashes(
            sim_env, network_path=network, runtime_calibration_version="v7_v4"
        )


def test_v2_simulator_calibration_report_is_relocatable(tmp_path):
    sim_env, network, line_dir = _make_simulator_tree(tmp_path)
    report_path = sim_env / "simulator_demand_calibration.json"
    report = json.loads(report_path.read_text())
    report["schema_version"] = "transit-simulator-demand-calibration-v2"
    report["output_path_base"] = "report_directory"
    report["output_sha256"] = {
        str(Path(raw).resolve().relative_to(sim_env.resolve())): expected
        for raw, expected in report["output_sha256"].items()
    }
    report_path.write_text(json.dumps(report))
    original = simulator_input_hashes(sim_env, network_path=network)

    relocated = tmp_path / "relocated" / "sim"
    shutil.copytree(sim_env, relocated)
    moved = simulator_input_hashes(relocated, network_path=network)
    assert moved["runtime_sha256"] == original["runtime_sha256"]
    assert moved["aggregate_sha256"] == original["aggregate_sha256"]


def _upgrade_simulator_tree_to_v3(sim_env):
    journey_path = sim_env / "simulator_journeys_v1.json"
    journey_path.write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-journeys-v1",
                "journeys": [],
            }
        )
    )
    report_path = sim_env / "simulator_demand_calibration.json"
    report = json.loads(report_path.read_text())
    report["schema_version"] = "transit-simulator-demand-calibration-v3"
    report["output_path_base"] = "report_directory"
    report["journey_execution"] = {
        "artifact": journey_path.name,
        "schema_version": "transit-simulator-journeys-v1",
        "causal_transfer_chain": True,
    }
    report["output_sha256"] = {
        str(Path(raw).resolve().relative_to(sim_env.resolve())): expected
        for raw, expected in report["output_sha256"].items()
    }
    report["output_sha256"][journey_path.name] = sha256(journey_path)
    report_path.write_text(json.dumps(report))
    return journey_path


def test_v3_simulator_requires_untampered_causal_journey_artifact(tmp_path):
    sim_env, network, _ = _make_simulator_tree(tmp_path)
    journey_path = _upgrade_simulator_tree_to_v3(sim_env)

    valid = simulator_input_hashes(sim_env, network_path=network)
    assert journey_path.name in valid["runtime_files_sha256"]

    journey_path.write_text('{"schema_version":"tampered"}')
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        simulator_input_hashes(sim_env, network_path=network)

    journey_path.unlink()
    with pytest.raises(FileNotFoundError, match="journey artifact is missing"):
        simulator_input_hashes(sim_env, network_path=network)


def _resume_fixture():
    arguments = {name: 0 for name in RESUME_IMMUTABLE_ARGUMENTS}
    arguments.update(
        {
            "method": "h2oplus",
            "seed": 42,
            "run_name": "",
            "run_signature": "",
        }
    )
    trainer_config = {"hidden_dim": 32}
    contract_config = {"episode_seconds": 24_000.0}
    sources = {"trainer.py": "source-hash"}
    simulator = {"aggregate_sha256": "sim-hash"}
    runtime_fingerprint = {
        "required": {
            "python_version": "3.8.20",
            "packages": {"torch": "2.4.1"},
            "compiler": {"sha256": "compiler-hash"},
        },
        "host": {"platform": "host-a"},
    }
    metadata = {
        "schema_version": "transit-training-run-v4",
        "arguments": copy.deepcopy(arguments),
        "trainer_config": copy.deepcopy(trainer_config),
        "contract_config": copy.deepcopy(contract_config),
        "dataset": {"sha256": "dataset-hash"},
        "source_sha256": copy.deepcopy(sources),
        "simulator": copy.deepcopy(simulator),
        "hardware": {"runtime_fingerprint": copy.deepcopy(runtime_fingerprint)},
    }
    checkpoint = {
        "epoch": 7,
        "method": "h2oplus",
        "training_seed": 42,
        "dataset_sha256": "dataset-hash",
        "simulator_aggregate_sha256": "sim-hash",
        "run_signature": "",
        "runtime_state": {
            "path": "checkpoints/runtime_epoch000007.pkl",
            "sha256": "runtime-hash",
        },
    }
    return (
        metadata,
        arguments,
        trainer_config,
        contract_config,
        sources,
        simulator,
        runtime_fingerprint,
        checkpoint,
    )


def test_resume_validation_rejects_protocol_drift():
    (
        metadata,
        arguments,
        trainer_config,
        contract_config,
        sources,
        simulator,
        runtime_fingerprint,
        checkpoint,
    ) = _resume_fixture()
    assert validate_resume(
        metadata,
        arguments=arguments,
        trainer_config=trainer_config,
        contract_config=contract_config,
        dataset_sha256="dataset-hash",
        simulator=simulator,
        sources=sources,
        runtime_fingerprint=runtime_fingerprint,
        checkpoint_extra=checkpoint,
        target_epochs=10,
    ) == 7

    changed = copy.deepcopy(arguments)
    changed["updates_per_epoch"] = 1
    with pytest.raises(ValueError, match="arguments changed"):
        validate_resume(
            metadata,
            arguments=changed,
            trainer_config=trainer_config,
            contract_config=contract_config,
            dataset_sha256="dataset-hash",
            simulator=simulator,
            sources=sources,
            runtime_fingerprint=runtime_fingerprint,
            checkpoint_extra=checkpoint,
            target_epochs=10,
        )

    changed_simulator = {"aggregate_sha256": "different"}
    with pytest.raises(ValueError, match="simulator inputs changed"):
        validate_resume(
            metadata,
            arguments=arguments,
            trainer_config=trainer_config,
            contract_config=contract_config,
            dataset_sha256="dataset-hash",
            simulator=changed_simulator,
            sources=sources,
            runtime_fingerprint=runtime_fingerprint,
            checkpoint_extra=checkpoint,
            target_epochs=10,
        )


def test_resume_runtime_validation_rejects_required_drift_but_allows_host_drift():
    (
        metadata,
        arguments,
        trainer_config,
        contract_config,
        sources,
        simulator,
        runtime_fingerprint,
        checkpoint,
    ) = _resume_fixture()

    moved_host = copy.deepcopy(runtime_fingerprint)
    moved_host["host"]["platform"] = "host-b"
    assert validate_resume(
        metadata,
        arguments=arguments,
        trainer_config=trainer_config,
        contract_config=contract_config,
        dataset_sha256="dataset-hash",
        simulator=simulator,
        sources=sources,
        runtime_fingerprint=moved_host,
        checkpoint_extra=checkpoint,
        target_epochs=10,
    ) == 7

    changed_runtime = copy.deepcopy(runtime_fingerprint)
    changed_runtime["required"]["packages"]["torch"] = "2.5.0"
    with pytest.raises(ValueError, match="required runtime changed"):
        validate_resume(
            metadata,
            arguments=arguments,
            trainer_config=trainer_config,
            contract_config=contract_config,
            dataset_sha256="dataset-hash",
            simulator=simulator,
            sources=sources,
            runtime_fingerprint=changed_runtime,
            checkpoint_extra=checkpoint,
            target_epochs=10,
        )


def test_resume_reconciles_metrics_to_checkpoint(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(json.dumps({"epoch": epoch}) for epoch in (1, 2, 3)) + "\n"
    )
    archive = reconcile_metrics(
        metrics,
        completed_epochs=2,
        archive_suffix="test",
    )
    assert archive is not None
    assert [json.loads(line)["epoch"] for line in metrics.read_text().splitlines()] == [
        1,
        2,
    ]
    assert json.loads(archive.read_text())["epoch"] == 3
