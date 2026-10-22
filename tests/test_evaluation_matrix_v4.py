import json
from pathlib import Path

import pytest

from reproducibility import run_evaluation_matrix_v4 as matrix


def test_methods_manifest_resolves_checkpoints_and_validates_names(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "methods.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": matrix.METHODS_SCHEMA,
                "methods": [
                    {"name": "h2oplus-seed1", "type": "checkpoint", "checkpoint": "policy.pt"},
                    {"name": "zero", "type": "zero"},
                    {"name": "gain-0.05", "type": "headway-balance", "gain": 0.05},
                    {"name": "myopic", "type": "myopic-welfare"},
                ],
            }
        )
    )
    methods = matrix.load_methods(manifest)
    assert methods[0]["checkpoint"] == checkpoint.resolve()
    assert methods[0]["checkpoint_sha256"] == matrix.sha256(checkpoint)
    assert methods[2]["gain"] == 0.05

    manifest.write_text(
        json.dumps(
            {
                "schema_version": matrix.METHODS_SCHEMA,
                "methods": [{"name": "../escape", "type": "zero"}],
            }
        )
    )
    try:
        matrix.load_methods(manifest)
    except ValueError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("unsafe method name was accepted")


def test_complete_evaluation_requires_all_provenance_and_event_log_hash(tmp_path):
    output = tmp_path / "evaluation.json"
    event_log = tmp_path / "evaluation.npz"
    event_log.write_bytes(b"event log")
    source_hashes = {"evaluator.py": "source"}
    runtime_hashes = {"scenario.xml": "runtime"}
    job = {
        "output": output,
        "event_log": event_log,
        "signature": "exact",
        "dataset_sha256": "dataset",
        "source_sha256": source_hashes,
        "sumo_version": "SUMO 1.27.1",
        "scenario": {"runtime_input_sha256": runtime_hashes},
    }
    payload = {
        "schema_version": "transit-evaluation-v4",
        "evaluation_job": {"signature": "exact"},
        "dataset": {"sha256": "dataset"},
        "source_sha256": source_hashes,
        "runtime": {"sumo_version": "SUMO 1.27.1"},
        "scenario": {"runtime_input_sha256": runtime_hashes},
        "event_log": {"sha256": matrix.sha256(event_log)},
    }
    output.write_text(json.dumps(payload))
    assert matrix._is_complete(job)

    payload["dataset"]["sha256"] = "changed"
    output.write_text(json.dumps(payload))
    assert not matrix._is_complete(job)


def test_complete_v6_evaluation_requires_fixed_horizon_welfare_invariants(tmp_path):
    output = tmp_path / "evaluation.json"
    job = {
        "output": output,
        "event_log": None,
        "signature": "exact",
        "dataset_sha256": "dataset",
        "source_sha256": {"evaluator.py": "source"},
        "sumo_version": "SUMO 1.27.1",
        "scenario": {"runtime_input_sha256": {"scenario.xml": "runtime"}},
        "mdp_version": "v6",
        "result_schema": "transit-evaluation-v6",
        "max_events": 0,
        "max_sim_time_seconds": 0,
    }
    payload = {
        "schema_version": "transit-evaluation-v6",
        "evaluation_job": {"signature": "exact"},
        "dataset": {"sha256": "dataset"},
        "source_sha256": {"evaluator.py": "source"},
        "runtime": {"sumo_version": "SUMO 1.27.1"},
        "scenario": {"runtime_input_sha256": {"scenario.xml": "runtime"}},
        "episode": {
            "observed_end_time_seconds": 24_000,
            "terminal_transitions": 1,
            "control_decisions": 10,
            "transitions": 10,
        },
        "passenger_welfare": {
            "chronological_coverage_errors": {
                "wait_passenger_seconds": 0.0,
                "in_vehicle_passenger_seconds": 0.0,
            }
        },
        "ledger_audit": {
            "count_source_audit": {
                "max_waiting_count_error": 0,
                "max_in_vehicle_count_error": 0,
                "subscription_fallback_queries": 0,
            }
        },
        "service": {
            "completed_trips": 389,
            "departed_trips": 389,
            "active_trips_at_horizon": 0,
        },
        "event_log": None,
    }
    output.write_text(json.dumps(payload))
    assert matrix._is_complete(job)
    payload["passenger_welfare"]["chronological_coverage_errors"][
        "wait_passenger_seconds"
    ] = 1.0
    output.write_text(json.dumps(payload))
    assert not matrix._is_complete(job)


def test_v7_checkpoint_manifest_requires_explicit_algorithm_and_training_seed(
    tmp_path,
):
    checkpoint = tmp_path / "final_actor.pt"
    checkpoint.write_bytes(b"policy")
    manifest = tmp_path / "methods.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": matrix.METHODS_SCHEMAS["v7"],
                "study_stage": "formal",
                "selection_rule": matrix.V7_POLICY_SELECTION_RULE,
                "methods": [
                    {
                        "name": "rlpd-seed3",
                        "type": "checkpoint",
                        "checkpoint": str(checkpoint),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="explicit algorithm"):
        matrix.load_methods(manifest, matrix.METHODS_SCHEMAS["v7"])

    payload = json.loads(manifest.read_text())
    payload["methods"][0].update(
        {
            "algorithm": "rlpd",
            "training_seed": 3,
            "training_run_signature": "1" * 64,
            "training_admission_sha256": "2" * 64,
            "training_checkpoint_sha256": "3" * 64,
            "training_plan_sha256": "4" * 64,
        }
    )
    manifest.write_text(json.dumps(payload))
    methods = matrix.load_methods(manifest, matrix.METHODS_SCHEMAS["v7"])
    assert methods[0]["algorithm"] == "rlpd"
    assert methods[0]["training_seed"] == 3


def test_v7_job_command_passes_one_parent_dataset_hash_and_policy_identity():
    job = {
        "evaluator": Path("evaluate_transit_v7.py"),
        "method": {
            "name": "rlpd-seed3",
            "algorithm": "rlpd",
            "training_seed": 3,
            "type": "checkpoint",
            "checkpoint": Path("final_actor.pt"),
            "training_run_signature": "1" * 64,
            "training_admission_sha256": "2" * 64,
            "training_checkpoint_sha256": "3" * 64,
            "training_plan_sha256": "4" * 64,
        },
        "mdp_version": "v7",
        "sumo_seed": 2101,
        "scenario": {"path": Path("scenario.json")},
        "dataset": Path("dataset.h5"),
        "dataset_sha256": "d" * 64,
        "dataset_size_bytes": 123,
        "dataset_mtime_ns": 456,
        "output": Path("evaluation.json"),
        "event_log": None,
        "signature": "signature",
        "device": "cpu",
        "max_events": 0,
        "max_sim_time_seconds": 0,
        "sumo_dir": Path("sumo"),
        "edge_xml": Path("edge.xml"),
        "sumo_backend": "libsumo",
        "sumo_runtime_version": "1.25.0",
        "sumo_runtime_signature": "5" * 64,
        "sumo_runtime_lock": Path("runtime_lock.json"),
        "sumo_runtime_lock_sha256": "6" * 64,
    }
    command = matrix._job_command(job)
    assert command[command.index("--verified-dataset-sha256") + 1] == "d" * 64
    assert command[command.index("--expected-algorithm") + 1] == "rlpd"
    assert command[command.index("--expected-training-seed") + 1] == "3"
    assert command[command.index("--expected-training-run-signature") + 1] == (
        "1" * 64
    )
    assert command[
        command.index("--expected-training-admission-sha256") + 1
    ] == ("2" * 64)
    assert command[
        command.index("--expected-training-checkpoint-sha256") + 1
    ] == ("3" * 64)
    assert command[command.index("--sumo-backend") + 1] == "libsumo"
    assert command[
        command.index("--expected-sumo-runtime-signature") + 1
    ] == ("5" * 64)
    assert command[
        command.index("--expected-sumo-runtime-lock-sha256") + 1
    ] == ("6" * 64)


def test_complete_v7_evaluation_requires_integrity_and_service_audits(tmp_path):
    output = tmp_path / "evaluation.json"
    source = {"evaluator.py": "source"}
    scenario_runtime = {"scenario.xml": "runtime"}
    job = {
        "output": output,
        "event_log": None,
        "signature": "exact",
        "dataset_sha256": "dataset",
        "dataset_size_bytes": 123,
        "dataset_mtime_ns": 456,
        "source_sha256": source,
        "sumo_version": "1.25.0",
        "sumo_backend": "libsumo",
        "sumo_runtime_signature": "5" * 64,
        "sumo_runtime_lock_sha256": "6" * 64,
        "sumo_runtime_fingerprint": {
            "portable_identity": {"backend": "libsumo", "runtime": "locked"}
        },
        "scenario": {"runtime_input_sha256": scenario_runtime},
        "mdp_version": "v7",
        "result_schema": "transit-evaluation-v7",
        "max_events": 0,
        "max_sim_time_seconds": 0,
        "method": {"type": "zero"},
    }
    payload = {
        "schema_version": "transit-evaluation-v7",
        "evaluation_job": {"signature": "exact"},
        "dataset": {
            "sha256": "dataset",
            "size_bytes": 123,
            "mtime_ns": 456,
            "verification": "parent-matrix-sha256-plus-child-stat-lock",
        },
        "source_sha256": source,
        "runtime": {
            "sumo_version": "1.25.0",
            "sumo_backend": "libsumo",
            "sumo_runtime_signature": "5" * 64,
            "sumo_runtime_lock": {"sha256": "6" * 64},
            "sumo_runtime_fingerprint": {
                "portable_identity": {"backend": "libsumo", "runtime": "locked"}
            },
        },
        "scenario": {"runtime_input_sha256": scenario_runtime},
        "episode": {
            "observed_end_time_seconds": 24_000,
            "terminal_transitions": 1,
            "control_decisions": 10,
            "transitions": 10,
        },
        "passenger_welfare": {
            "chronological_coverage_errors": {
                "wait_passenger_seconds": 0.0,
                "in_vehicle_passenger_seconds": 0.0,
                "reward": 0.0,
            }
        },
        "ledger_audit": {
            "count_source_audit": {
                "max_waiting_count_error": 0,
                "max_in_vehicle_count_error": 0,
                "subscription_fallback_queries": 0,
            }
        },
        "bus_service_semantics_audit": {"passed": True},
        "service": {
            "completed_trips": 389,
            "departed_trips": 389,
            "active_trips_at_horizon": 0,
        },
        "event_log": None,
    }
    output.write_text(json.dumps(payload))
    assert matrix._is_complete(job)
    payload["runtime"]["sumo_runtime_signature"] = "6" * 64
    output.write_text(json.dumps(payload))
    assert not matrix._is_complete(job)
    payload["runtime"]["sumo_runtime_signature"] = "5" * 64
    payload["bus_service_semantics_audit"]["passed"] = False
    output.write_text(json.dumps(payload))
    assert not matrix._is_complete(job)


def test_existing_matrix_manifest_is_archived_before_retry(tmp_path):
    manifest = tmp_path / "matrix_manifest.json"
    manifest.write_text('{"attempt": 1}\n')
    digest = matrix.sha256(manifest)

    archive = matrix._archive_existing_manifest(manifest)

    assert archive is not None
    assert archive.parent == tmp_path / "matrix_manifest_history"
    assert digest[:16] in archive.name
    assert json.loads(archive.read_text()) == {"attempt": 1}
    assert not manifest.exists()
    assert matrix._archive_existing_manifest(manifest) is None


def test_v7_auto_workers_respect_physical_cores_and_available_memory():
    workers, evidence = matrix.auto_worker_count(
        100,
        physical_cores=24,
        available_memory_bytes=20 * 1024**3,
    )
    assert workers == 13
    assert evidence["cpu_worker_limit"] == 16
    assert evidence["memory_worker_limit"] == 13
    assert evidence["reserved_physical_cores"] == 8
    assert evidence["selected_workers"] == 13

    workers, evidence = matrix.auto_worker_count(
        4,
        physical_cores=1,
        available_memory_bytes=8 * 1024**3,
    )
    assert workers == 1
    assert evidence["reserved_physical_cores"] == 0

    workers, evidence = matrix.auto_worker_count(
        200,
        physical_cores=96,
        available_memory_bytes=200 * 1024**3,
    )
    assert workers == 64
    assert evidence["cpu_worker_fraction"] == "2/3"
    assert evidence["reserved_physical_cores"] == 32


def test_v7_partition_keys_ignore_host_specific_runtime_input_paths():
    def job(runtime_path, method_name, seed):
        return {
            "method": {
                "name": method_name,
                "algorithm": "rlpd",
                "training_seed": 5101,
                "type": "checkpoint",
                "checkpoint_sha256": "a" * 64,
            },
            "environment_block": f"environment-{seed}",
            "sumo_seed": seed,
            "scenario": {
                "manifest_sha256": "b" * 64,
                "runtime_input_sha256": {runtime_path: "c" * 64},
            },
        }

    local = job("/local/scenario.xml", "rlpd-seed5101", 2101)
    remote = job("/remote/scenario.xml", "rlpd-seed5101", 2101)
    assert matrix.portable_job_key(local) == matrix.portable_job_key(remote)


def test_v7_partition_is_complete_balanced_and_deterministic():
    jobs = [
        {
            "method": {
                "name": f"rlpd-seed{seed}",
                "algorithm": "rlpd",
                "training_seed": seed,
                "type": "checkpoint",
                "checkpoint_sha256": f"{seed:064x}",
            },
            "environment_block": f"environment-{cell}",
            "sumo_seed": 2100 + cell,
            "scenario": {
                "manifest_sha256": f"{cell:064x}",
                "runtime_input_sha256": {f"/input/{cell}": f"{cell + 20:064x}"},
            },
        }
        for seed in (5101, 5102, 5103)
        for cell in (1, 2, 3)
    ]
    metadata = {
        "mdp_version": "v7",
        "dataset_sha256": "d" * 64,
        "methods_manifest_sha256": "e" * 64,
        "environment_design": {"sha256": "f" * 64},
        "evaluator_sha256": "1" * 64,
        "source_sha256": {"/source.py": "2" * 64},
        "sumo_version": "1.25.0",
        "sumo_backend": "libsumo",
        "sumo_runtime_signature": "5" * 64,
        "sumo_runtime_lock": {"sha256": "6" * 64},
    }
    partitions = [
        matrix.partition_jobs(jobs, metadata, shard_count=2, shard_index=index)
        for index in range(2)
    ]
    selected_keys = [
        matrix.portable_job_key(job)
        for selected, _ in partitions
        for job in selected
    ]
    assert sorted(selected_keys) == sorted(matrix.portable_job_key(job) for job in jobs)
    assert [len(selected) for selected, _ in partitions] == [5, 4]
    assert len({partition["full_matrix_signature"] for _, partition in partitions}) == 1


def test_explicit_environment_design_pairs_unique_demand_and_sumo_seeds(
    tmp_path, monkeypatch
):
    design = tmp_path / "environment_design.json"
    design.write_text(
        json.dumps(
            {
                "schema_version": matrix.ENVIRONMENT_DESIGN_SCHEMA,
                "blocks": [
                    {
                        "name": "block-1",
                        "sumo_seed": 2101,
                        "scenario_manifests": ["r3101-low.json", "r3101-high.json"],
                    },
                    {
                        "name": "block-2",
                        "sumo_seed": 2102,
                        "scenario_manifests": ["r3102-low.json", "r3102-high.json"],
                    },
                ],
            }
        )
    )

    def fake_load(paths):
        scenarios = []
        for path in paths:
            realization = int(path.name.split("-")[0][1:])
            scale = 0.75 if "low" in path.name else 1.25
            scenarios.append(
                {
                    "path": path,
                    "tag": path.stem,
                    "manifest_sha256": path.stem,
                    "runtime_input_sha256": {path.stem: "hash"},
                    "demand_scale": scale,
                    "selection_seed": realization,
                    "depart_jitter_seconds": 120.0,
                }
            )
        return scenarios

    monkeypatch.setattr(matrix, "load_scenarios", fake_load)
    cells, evidence = matrix.load_environment_design(design)

    assert len(cells) == 4
    assert {cell["environment_block"] for cell in cells} == {"block-1", "block-2"}
    assert {cell["sumo_seed"] for cell in cells} == {2101, 2102}
    assert evidence["mode"] == "explicit-blocks"
    assert evidence["sha256"] == matrix.sha256(design)

    payload = json.loads(design.read_text())
    payload["blocks"][1]["sumo_seed"] = 2101
    design.write_text(json.dumps(payload))
    try:
        matrix.load_environment_design(design)
    except ValueError as error:
        assert "reused" in str(error)
    else:
        raise AssertionError("a reused SUMO seed was accepted")
