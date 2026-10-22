import json
import os
import copy
from dataclasses import asdict
from pathlib import Path

import pytest

import reproducibility.run_training_matrix_v4 as training_matrix
from reproducibility.run_training_matrix_v4 import (
    PLAN_SCHEMA,
    SCHEMA_CONTRACTS,
    V7_ADMISSION_BUILDER,
    _archive_existing_manifest,
    _is_complete,
    _training_admission_artifact_descriptor,
    _v7_initializer_records,
    _prepare_existing_run,
    _resolve_required_environment,
    _select_jobs,
    job_command,
    job_environment,
    load_plan,
    run_job,
    validate_training_runtime,
)
from reproducibility.build_training_admission_v7 import (
    SCHEMA_VERSION as V7_ADMISSION_SCHEMA,
)
from reproducibility.run_training_dispatch_v6 import (
    _select_host,
    _validate_v7_capacity_audit,
    job_set_sha256,
    load_dispatch,
)
from SimpleSAC.train_transit_v3 import (
    collect_runtime_fingerprint,
    sha256,
    simulator_input_hashes,
    source_hashes,
)


def _runtime_lock() -> dict:
    return {
        "python_version": "3.8.20",
        "python_implementation": "CPython",
        "python_cache_tag": "cpython-38",
        "python_executable_sha256": "1" * 64,
        "packages": {"torch": "2.4.1"},
        "torch_cuda_version": "12.4",
        "torch_binary_sha256": {
            "python_extension": "2" * 64,
            "cuda_library": "3" * 64,
        },
        "compiler": {"sha256": "4" * 64, "version": "gcc test"},
    }
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


def _simulator_tree(tmp_path: Path) -> Path:
    sim_env = tmp_path / "sim"
    line = sim_env / "data" / "7X"
    line.mkdir(parents=True)
    (sim_env / "config.json").write_text('{"time_step": 1}')
    for name in (
        "stop_news.xlsx",
        "route_news.xlsx",
        "time_table.xlsx",
        "passenger_OD.xlsx",
    ):
        (line / name).write_bytes(name.encode())
    calibrated = (line / "time_table_sumo.xlsx", line / "passenger_OD_sumo.xlsx")
    for path in calibrated:
        path.write_bytes(("calibrated:" + path.name).encode())
    (sim_env / "simulator_demand_calibration.json").write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-demand-calibration-v1",
                "source_manifest_sha256": "manifest",
                "source_bus_objects_sha256": "buses",
                "output_sha256": {
                    str(path.resolve()): sha256(path) for path in calibrated
                },
            }
        )
    )
    (sim_env / "segment_runtime_calibration_v4.json").write_text(
        json.dumps({"schema_version": "transit-simulator-runtime-calibration-v4"})
    )
    profile = sim_env / "observation_speed_profiles_v7_v1.npz"
    profile.write_bytes(b"unit-speed-profile")
    (sim_env / "segment_runtime_calibration_v7_v4.json").write_text(
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
    return sim_env


def _plan(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset.h5"
    dataset.write_bytes(b"canonical-v4")
    sim_env = _simulator_tree(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA,
                "dataset": str(dataset),
                "sim_env": str(sim_env),
                "output_root": str(tmp_path / "runs"),
                "device": "cpu",
                "shared_arguments": {"epochs": 3, "events_per_epoch": 5},
                "jobs": [
                    {
                        "name": "h2oplus-seed41",
                        "method": "h2oplus",
                        "seed": 41,
                        "arguments": {"target_entropy": 0.0},
                    }
                ],
            }
        )
    )
    return plan


def _v7_admission(
    tmp_path: Path,
    *,
    dataset: Path,
    sim_env: Path,
    selected_arguments: dict,
) -> Path:
    admission = tmp_path / "training-admission-v7.json"
    evidence_dir = tmp_path / "admission-evidence"
    evidence_dir.mkdir()
    evidence = {}
    for name in (
        "dataset_audit",
        "serialized_credit_audit",
        "reward_audit",
        "reward_loss_pilot",
        "entropy_pilot",
        "device_precision_audit",
        "zero_fidelity_audit",
        "intervention_fidelity_audit",
        "method_protocol_audit",
    ):
        path = evidence_dir / f"{name}.json"
        schema = f"test-{name}-v1"
        path.write_text(json.dumps({"schema_version": schema}))
        evidence[name] = {
            "path": os.path.relpath(path, admission.parent),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "schema_version": schema,
        }
    simulator = simulator_input_hashes(
        sim_env, runtime_calibration_version="v7_v4"
    )
    admission.write_text(
        json.dumps(
            {
                "schema_version": V7_ADMISSION_SCHEMA,
                "study_stage": "formal",
                "passed": True,
                "dataset": {
                    "path": os.path.relpath(dataset, admission.parent),
                    "sha256": sha256(dataset),
                    "bytes": dataset.stat().st_size,
                },
                "simulator": {
                    "path": os.path.relpath(sim_env, admission.parent),
                    "aggregate_sha256": simulator["aggregate_sha256"],
                    "runtime_sha256": simulator["runtime_sha256"],
                },
                "selected_arguments": selected_arguments,
                "evidence": evidence,
                "training_source_sha256": source_hashes("v7"),
                "training_runtime_lock": _runtime_lock(),
                "builder_sha256": sha256(V7_ADMISSION_BUILDER),
            }
        )
    )
    return admission


def test_v7_execution_rejects_runtime_different_from_precision_gate():
    import torch

    observed = collect_runtime_fingerprint(torch, "cpu")["required"]
    evidence = {
        "mdp_version": "v7",
        "device": "cpu",
        "resolved_required_environment": {},
        "training_admission": {"runtime_lock": observed},
    }
    assert validate_training_runtime(evidence)["required"] == observed

    drifted = copy.deepcopy(evidence)
    drifted["training_admission"]["runtime_lock"]["packages"]["torch"] = "0.0"
    with pytest.raises(RuntimeError, match="precision-gate runtime"):
        validate_training_runtime(drifted)


def test_training_plan_builds_stable_signed_command(tmp_path):
    plan = _plan(tmp_path)
    jobs, evidence = load_plan(plan)
    assert len(jobs) == 1
    job = jobs[0]
    assert len(job["signature"]) == 64
    assert evidence["dataset_sha256"] == sha256(job["dataset"])
    command = job_command(job, resume=job["run_dir"] / "checkpoints" / "latest.pt")
    assert command[1].endswith("train_transit_v4.py")
    assert command[command.index("--run-signature") + 1] == job["signature"]
    assert command[command.index("--target-entropy") + 1] == "0.0"
    assert "--resume" in command
    assert "--allow-shared-gpu" not in command
    assert command[command.index("--max-canonical-processes-per-gpu") + 1] == "1"

    payload = json.loads(plan.read_text())
    payload["allow_shared_gpu"] = True
    payload["allow_external_gpu_processes"] = True
    payload["shared_arguments"]["compile_networks"] = True
    payload["shared_arguments"]["compile_mode"] = "default"
    shared_plan = tmp_path / "shared-plan.json"
    shared_plan.write_text(json.dumps(payload))
    shared_job = load_plan(shared_plan)[0][0]
    assert "--allow-shared-gpu" in job_command(shared_job)
    assert "--allow-external-gpu-processes" in job_command(shared_job)
    assert "--compile-networks" in job_command(shared_job)
    assert job_command(shared_job)[
        job_command(shared_job).index("--compile-mode") + 1
    ] == "default"
    assert shared_job["signature"] != job["signature"]

    payload = json.loads(plan.read_text())
    payload["jobs"][0]["max_canonical_processes_per_gpu"] = 2
    quota_plan = tmp_path / "quota-plan.json"
    quota_plan.write_text(json.dumps(payload))
    quota_job = load_plan(quota_plan)[0][0]
    quota_command = job_command(quota_job)
    assert quota_command[
        quota_command.index("--max-canonical-processes-per-gpu") + 1
    ] == "2"
    assert quota_job["signature"] != job["signature"]


def test_v6_training_plan_locks_v6_entrypoint_sources_and_schemas(tmp_path):
    original = _plan(tmp_path)
    payload = json.loads(original.read_text())
    payload["schema_version"] = SCHEMA_CONTRACTS["v6"]["plan"]
    payload["mdp_version"] = "v6"
    payload["output_root"] = str(tmp_path / "runs-v6")
    plan = tmp_path / "plan-v6.json"
    plan.write_text(json.dumps(payload))

    jobs, evidence = load_plan(plan)
    job = jobs[0]
    command = job_command(job)
    assert command[1].endswith("train_transit_v6.py")
    assert job["mdp_version"] == "v6"
    assert evidence["mdp_version"] == "v6"
    assert evidence["matrix_schema"] == "transit-training-matrix-v6"
    assert evidence["summary_schema"] == "transit-training-summary-v7"
    assert "bus_h2o/mdp_contract_v6.py" in evidence["source_sha256"]

    payload["mdp_version"] = "v5"
    mismatch = tmp_path / "plan-v6-mismatch.json"
    mismatch.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="implies v6"):
        load_plan(mismatch)


def test_v7_training_plan_emits_episode_demand_schedule(tmp_path):
    original = _plan(tmp_path)
    payload = json.loads(original.read_text())
    payload["schema_version"] = SCHEMA_CONTRACTS["v7"]["plan"]
    payload["mdp_version"] = "v7"
    payload["output_root"] = str(tmp_path / "runs-v7")
    payload["shared_arguments"].update(
        {
            "sim_demand_scales": [0.75, 1.0, 1.25],
            "reward_scale": 1e-5,
            "critic_loss": "huber",
            "huber_delta": 1.0,
            "device_cache_observation_dtype": "float32",
        }
    )
    selected_arguments = {
        "sim_demand_scales": [0.75, 1.0, 1.25],
        "reward_scale": 1e-5,
        "critic_loss": "huber",
        "huber_delta": 1.0,
        "target_entropy": 0.0,
        "device_cache_observation_dtype": "float32",
    }
    payload["training_admission"] = str(
        _v7_admission(
            tmp_path,
            dataset=Path(payload["dataset"]),
            sim_env=Path(payload["sim_env"]),
            selected_arguments=selected_arguments,
        )
    )
    plan = tmp_path / "plan-v7.json"
    plan.write_text(json.dumps(payload))

    jobs, evidence = load_plan(plan)
    command = job_command(jobs[0])

    assert command[1].endswith("train_transit_v7.py")
    start = command.index("--sim-demand-scales")
    assert command[start + 1 : start + 4] == ["0.75", "1.0", "1.25"]
    assert evidence["matrix_schema"] == "transit-training-matrix-v7"
    assert evidence["summary_schema"] == "transit-training-summary-v8"
    assert "bus_h2o/mdp_contract_v7.py" in evidence["source_sha256"]
    assert "--training-admission" in command


def test_v7_training_plan_rejects_missing_admission(tmp_path):
    original = _plan(tmp_path)
    payload = json.loads(original.read_text())
    payload["schema_version"] = SCHEMA_CONTRACTS["v7"]["plan"]
    payload["mdp_version"] = "v7"
    plan = tmp_path / "plan-v7-missing-admission.json"
    plan.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="requires training_admission"):
        load_plan(plan)


def test_training_plan_rejects_reserved_or_duplicate_jobs(tmp_path):
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["jobs"][0]["arguments"]["device"] = "cuda:0"
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported arguments"):
        load_plan(plan)

    payload["jobs"][0]["arguments"].pop("device")
    payload["jobs"].append(dict(payload["jobs"][0]))
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate training job"):
        load_plan(plan)


def test_training_plan_rejects_invalid_or_ambiguous_gpu_quotas(tmp_path):
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["jobs"][0]["max_canonical_processes_per_gpu"] = True
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="max_canonical_processes_per_gpu"):
        load_plan(plan)

    payload["jobs"][0]["max_canonical_processes_per_gpu"] = 2
    payload["allow_shared_gpu"] = True
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="legacy allow_shared_gpu"):
        load_plan(plan)


def test_training_plan_resolves_and_hashes_required_environment(tmp_path):
    plan = _plan(tmp_path)
    include_dir = tmp_path / "compiler-include"
    include_dir.mkdir()
    header = include_dir / "crypt.h"
    header.write_text("/* fixed compile header */\n")
    payload = json.loads(plan.read_text())
    payload["required_environment"] = {
        "CC": "/bin/true",
        "CPATH": include_dir.name,
        "TMPDIR": str(tmp_path),
    }
    plan.write_text(json.dumps(payload))

    job, evidence = load_plan(plan)
    job = job[0]
    environment = job_environment(job)
    assert environment["CC"] == str(Path("/bin/true").resolve())
    assert environment["CPATH"] == str(include_dir.resolve())
    assert environment["TMPDIR"] == str(tmp_path.resolve())
    assert evidence["required_environment"]["CC"]["sha256"] == sha256(
        Path("/bin/true").resolve()
    )
    assert evidence["required_environment"]["CPATH"][0][
        "crypt_h_sha256"
    ] == sha256(header)
    assert evidence["required_environment"]["TMPDIR"] == {
        "contract": "writable-executable-directory"
    }
    assert evidence["resolved_required_environment"] == {
        "CC": str(Path("/bin/true").resolve()),
        "CPATH": str(include_dir.resolve()),
        "TMPDIR": str(tmp_path.resolve()),
    }

    original_signature = job["signature"]
    header.write_text("/* changed compile header */\n")
    changed_job = load_plan(plan)[0][0]
    assert changed_job["signature"] != original_signature

    payload["required_environment"]["PYTHONPATH"] = "/tmp/injected"
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported variables"):
        load_plan(plan)


def test_required_environment_restricts_and_hashes_preload_library(tmp_path):
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["required_environment"] = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4"
    }
    plan.write_text(json.dumps(payload))

    jobs, evidence = load_plan(plan)
    preload = Path("/usr/lib/x86_64-linux-gnu/libcurl.so.4").resolve(strict=True)
    assert job_environment(jobs[0])["LD_PRELOAD"] == (
        "/usr/lib/x86_64-linux-gnu/libcurl.so.4"
    )
    assert evidence["required_environment"]["LD_PRELOAD"] == {
        "contract": "system-libcurl-soname-v4",
        "sha256": sha256(preload),
    }
    assert evidence["required_environment_identity"]["LD_PRELOAD"] == {
        "contract": "system-libcurl-soname-v4"
    }

    payload["required_environment"]["LD_PRELOAD"] = "/tmp/injected.so"
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="restricted to"):
        load_plan(plan)


def test_required_environment_identity_ignores_host_paths(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "crypt.h").write_text("/* identical compatibility header */\n")
    (second / "crypt.h").write_text("/* identical compatibility header */\n")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")

    first_values, first_identity = _resolve_required_environment(
        {"CC": "/bin/true", "CPATH": str(first), "TMPDIR": str(first)},
        plan_path=plan_path,
    )
    second_values, second_identity = _resolve_required_environment(
        {"CC": "/bin/true", "CPATH": str(second), "TMPDIR": str(second)},
        plan_path=plan_path,
    )

    assert first_values != second_values
    assert first_identity == second_identity


def test_training_plan_creates_user_scoped_temporary_directory(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    payload = json.loads(plan.read_text())
    payload["required_environment"] = {
        "CC": "/bin/true",
        "TMPDIR": "~/.cache/h2oplus-v6/tmp",
    }
    plan.write_text(json.dumps(payload))

    jobs, _ = load_plan(plan)

    expected = home / ".cache" / "h2oplus-v6" / "tmp"
    assert expected.is_dir()
    assert job_environment(jobs[0])["TMPDIR"] == str(expected.resolve())


def test_existing_signed_run_resumes_or_archives_pre_checkpoint_state(tmp_path):
    jobs, _ = load_plan(_plan(tmp_path))
    job = jobs[0]
    run_dir = job["run_dir"]
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_signature": job["signature"]})
    )

    checkpoint, archive = _prepare_existing_run(job)
    assert checkpoint is None
    assert archive is not None and archive.is_dir()
    assert not run_dir.exists()
    assert json.loads((archive / "metadata.json").read_text())["run_signature"] == job[
        "signature"
    ]

    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_signature": job["signature"]})
    )
    expected_checkpoint = run_dir / "checkpoints" / "latest.pt"
    expected_checkpoint.write_bytes(b"checkpoint")
    checkpoint, archive = _prepare_existing_run(job)
    assert checkpoint == expected_checkpoint
    assert archive is None


def test_existing_unsigned_run_is_never_overwritten(tmp_path):
    jobs, _ = load_plan(_plan(tmp_path))
    job = jobs[0]
    job["run_dir"].mkdir(parents=True)
    with pytest.raises(RuntimeError, match="different or missing signature"):
        _prepare_existing_run(job)


def test_existing_matrix_manifest_is_archived_before_a_new_attempt(tmp_path):
    manifest = tmp_path / "matrix_manifest.json"
    manifest.write_text('{"attempt": 1}\n')
    expected_hash = sha256(manifest)
    archive = _archive_existing_manifest(manifest)
    assert archive is not None and archive.is_file()
    assert not manifest.exists()
    assert sha256(archive) == expected_hash
    assert _archive_existing_manifest(manifest) is None


def test_verified_existing_run_remains_completed_in_new_manifest(tmp_path):
    jobs, evidence = load_plan(_plan(tmp_path))
    job = jobs[0]
    checkpoint = job["run_dir"] / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"complete checkpoint")
    (job["run_dir"] / "metadata.json").write_text(
        json.dumps(
            {
                "mdp_version": "v4",
                "run_signature": job["signature"],
                "dataset": {"sha256": evidence["dataset_sha256"]},
                "simulator": {
                    "aggregate_sha256": evidence["simulator"]["aggregate_sha256"]
                },
                "source_sha256": evidence["source_sha256"],
            }
        )
    )
    (job["run_dir"] / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": evidence["summary_schema"],
                "run_signature": job["signature"],
                "method": job["method"],
                "seed": job["seed"],
                "checkpoint": {"sha256": sha256(checkpoint)},
            }
        )
    )

    result = run_job(job, evidence, tmp_path / "logs")

    assert result["status"] == "completed"
    assert result["reused_existing_run"] is True
    assert not (tmp_path / "logs").exists()


def test_v7_completion_requires_semantically_locked_final_policy(tmp_path):
    original = _plan(tmp_path)
    payload = json.loads(original.read_text())
    payload["schema_version"] = SCHEMA_CONTRACTS["v7"]["plan"]
    payload["mdp_version"] = "v7"
    payload["output_root"] = str(tmp_path / "runs-v7-complete")
    selected_arguments = {
        "sim_demand_scales": [0.75, 1.0, 1.25],
        "reward_scale": 1e-5,
        "critic_loss": "huber",
        "huber_delta": 1.0,
        "target_entropy": 0.0,
        "device_cache_observation_dtype": "float32",
    }
    payload["shared_arguments"].update(selected_arguments)
    payload["training_admission"] = str(
        _v7_admission(
            tmp_path,
            dataset=Path(payload["dataset"]),
            sim_env=Path(payload["sim_env"]),
            selected_arguments=selected_arguments,
        )
    )
    plan = tmp_path / "plan-v7-complete.json"
    plan.write_text(json.dumps(payload))
    jobs, evidence = load_plan(plan)
    job = jobs[0]
    artifact_admission = _training_admission_artifact_descriptor(evidence)
    assert artifact_admission is not None
    assert "runtime_lock" in evidence["training_admission"]
    assert "runtime_lock" not in artifact_admission
    recorded_admission = {
        **artifact_admission,
        "path": f"/remote/certificate/{Path(artifact_admission['path']).name}",
    }
    run_dir = job["run_dir"]
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"complete v7 training checkpoint")
    checkpoint_sha256 = sha256(checkpoint)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mdp_version": "v7",
                "run_signature": job["signature"],
                "dataset": {"sha256": evidence["dataset_sha256"]},
                "simulator": {
                    "aggregate_sha256": evidence["simulator"][
                        "aggregate_sha256"
                    ]
                },
                "source_sha256": evidence["source_sha256"],
                "training_admission": recorded_admission,
            }
        )
    )

    class Replay:
        contract_config = {
            "horizon_seconds": 24_000.0,
        }

    trainer = TransitH2OTrainerV7(
        Replay(),
        config=TransitTrainerConfig(
            hidden_dim=8,
            ensemble_size=2,
            batch_size=4,
        ),
        device="cpu",
        seed=job["seed"],
    )
    final_policy = checkpoint.parent / "final_actor.pt"
    policy_extra = {
        "phase": "complete",
        "run_signature": job["signature"],
        "method": job["method"],
        "method_profile": asdict(METHOD_PROFILES[job["method"]]),
        "training_seed": job["seed"],
        "dataset_sha256": evidence["dataset_sha256"],
        "simulator_runtime_sha256": evidence["simulator"]["runtime_sha256"],
        "simulator_aggregate_sha256": evidence["simulator"][
            "aggregate_sha256"
        ],
        "source_sha256": evidence["source_sha256"],
        "mdp_version": "v7",
        "training_checkpoint_sha256": checkpoint_sha256,
        "training_admission": recorded_admission,
        "total_updates": 123,
        "simulator_decision_events": 456,
    }
    trainer.save_policy_artifact(final_policy, extra=policy_extra)
    recorded_run_dir = Path("/remote/formal-output") / job["name"]
    summary = {
        "schema_version": evidence["summary_schema"],
        "run_dir": str(recorded_run_dir),
        "run_signature": job["signature"],
        "method": job["method"],
        "seed": job["seed"],
        "total_updates": 123,
        "simulator_decision_events": 456,
        "checkpoint": {"sha256": checkpoint_sha256},
        "final_policy": {
            "path": str(recorded_run_dir / "checkpoints" / "final_actor.pt"),
            "sha256": sha256(final_policy),
            "source_training_checkpoint_sha256": checkpoint_sha256,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    assert _is_complete(job, evidence)

    import torch

    artifact = torch.load(final_policy, map_location="cpu", weights_only=False)
    artifact["extra"]["training_admission"] = {
        **recorded_admission,
        "sha256": "0" * 64,
    }
    torch.save(artifact, final_policy)
    summary["final_policy"]["sha256"] = sha256(final_policy)
    (run_dir / "summary.json").write_text(json.dumps(summary))
    assert not _is_complete(job, evidence)


def test_successful_matrix_writes_terminal_manifest_and_done_marker(
    tmp_path, monkeypatch, capsys
):
    plan = _plan(tmp_path)
    jobs, evidence = load_plan(plan)
    monkeypatch.setattr(
        training_matrix,
        "parse_args",
        lambda: type("Args", (), {"plan": plan, "job_name": None})(),
    )
    monkeypatch.setattr(
        training_matrix,
        "validate_training_runtime",
        lambda _evidence: None,
    )
    monkeypatch.setattr(
        training_matrix,
        "run_job",
        lambda job, _evidence, _log_dir: {
            "status": "completed",
            "run_dir": str(job["run_dir"]),
            "reused_existing_run": True,
        },
    )

    training_matrix.main()

    manifest_path = Path(evidence["output_root"]) / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_utc"]
    assert len(manifest["results"]) == len(jobs)
    assert "DONE training matrix" in capsys.readouterr().out


def test_v7_initializer_completion_rechecks_real_artifact_provenance(tmp_path):
    import torch

    class Replay:
        contract_config = {"horizon_seconds": 24_000.0}
        online_size = 0

    seed = 5101
    source_signature = "a" * 64
    source_config = TransitTrainerConfig(
        hidden_dim=8,
        ensemble_size=2,
        batch_size=4,
        offline_objective="iql",
        use_ratio_correction=True,
        mixed_backup_lambda=0.1,
    )
    target_config = TransitTrainerConfig(
        hidden_dim=8,
        ensemble_size=2,
        batch_size=4,
        offline_objective="iql",
        use_ratio_correction=False,
        mixed_backup_lambda=0.1,
    )
    source = TransitH2OTrainerV7(
        Replay(), config=source_config, device="cpu", seed=seed
    )
    source.total_updates = 50_000
    initializer = (
        tmp_path
        / f"h2oplus_seed{seed}"
        / "checkpoints"
        / "offline_initializer.pt"
    )
    admission_descriptor = {
        "path": str(tmp_path / "training_admission.json"),
        "sha256": "admission-sha",
        "bytes": 123,
        "schema_version": "transit-training-admission-v7-v2",
        "selected_arguments": {},
    }
    recorded_admission = {
        **admission_descriptor,
        "path": "/remote/certificate/training_admission.json",
    }
    evidence = {
        "dataset_sha256": "dataset-sha",
        "simulator": {
            "runtime_sha256": "runtime-sha",
            "aggregate_sha256": "simulator-sha",
        },
        "source_sha256": {"trainer.py": "source-sha"},
        "training_admission": admission_descriptor,
    }
    contract = {"horizon_seconds": 24_000.0}
    offline_metrics = {"train/critic_loss": 1.25}
    extra = {
        "phase": "offline_pretraining",
        "source_method": "h2oplus",
        "source_run_name": f"h2oplus_seed{seed}",
        "source_run_signature": source_signature,
        "source_training_seed": seed,
        "offline_updates_completed": 50_000,
        "trainer_total_updates": 50_000,
        "dataset_sha256": evidence["dataset_sha256"],
        "simulator_runtime_sha256": evidence["simulator"]["runtime_sha256"],
        "simulator_aggregate_sha256": evidence["simulator"]["aggregate_sha256"],
        "source_sha256": evidence["source_sha256"],
        "contract_config": contract,
        "mdp_version": "v7",
        "training_admission": recorded_admission,
        "offline_metrics": offline_metrics,
    }
    source.save_offline_initializer(initializer, extra=extra)
    descriptor = {
        "path": str(initializer),
        "sha256": sha256(initializer),
        "bytes": initializer.stat().st_size,
        "schema_version": "transit-offline-initializer-v7-v1",
        "source_method": "h2oplus",
        "source_run_name": f"h2oplus_seed{seed}",
        "source_run_signature": source_signature,
        "source_training_seed": seed,
        "offline_updates": 50_000,
    }
    relocated_descriptor = {
        **descriptor,
        "path": (
            f"/remote/formal-output/h2oplus_seed{seed}/checkpoints/"
            "offline_initializer.pt"
        ),
    }
    job = {
        "method": "h2oplus-no-ratio",
        "seed": seed,
        "arguments": {"offline_updates": 50_000},
        "run_dir": tmp_path / "target",
        "offline_initializer": initializer,
        "offline_initializer_source_signature": source_signature,
    }
    summary = {
        "offline_metrics": offline_metrics,
        "offline_initializer_consumed": relocated_descriptor,
        "offline_initializer_produced": None,
    }
    metadata = {
        "trainer_config": asdict(target_config),
        "contract_config": contract,
        "training_admission": recorded_admission,
        "offline_initializer_consumed": relocated_descriptor,
        "offline_initializer_produced": None,
    }

    consumed, produced = _v7_initializer_records(
        job, evidence, summary, metadata
    )
    assert consumed == relocated_descriptor
    assert produced is None

    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    payload["extra"]["source_run_signature"] = "0" * 64
    torch.save(payload, initializer)
    with pytest.raises(ValueError, match="provenance"):
        _v7_initializer_records(job, evidence, summary, metadata)


def test_job_shard_selection_is_explicit_ordered_and_unique(tmp_path):
    jobs, _ = load_plan(_plan(tmp_path))
    second = dict(jobs[0])
    second["name"] = "h2oplus-seed42"
    selected = _select_jobs([jobs[0], second], [second["name"], jobs[0]["name"]])
    assert [job["name"] for job in selected] == [second["name"], jobs[0]["name"]]
    with pytest.raises(ValueError, match="duplicates"):
        _select_jobs(jobs, [jobs[0]["name"], jobs[0]["name"]])
    with pytest.raises(ValueError, match="no requested jobs"):
        _select_jobs(jobs, ["missing-seed1"])


def _dispatch_plan(tmp_path):
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["plan_id"] = "unit-dispatch"
    payload["jobs"] = [
        {
            "name": "rlpd-seed41",
            "method": "rlpd",
            "seed": 41,
            "max_canonical_processes_per_gpu": 2,
        },
        {
            "name": "rlpd-seed42",
            "method": "rlpd",
            "seed": 42,
            "max_canonical_processes_per_gpu": 2,
        },
    ]
    plan.write_text(json.dumps(payload))
    return plan


def _dispatch_payload(plan, jobs):
    return {
        "schema_version": "transit-training-dispatch-v1",
        "plan_id": "unit-dispatch",
        "plan": {"path": str(plan), "sha256": sha256(plan)},
        "job_set_sha256": job_set_sha256(jobs),
        "hosts": [
            {
                "hostname": "gpu-host",
                "gpus": [
                    {
                        "physical_gpu": "0",
                        "stages": [
                            {
                                "name": "rlpd-pair",
                                "workers": [
                                    {"name": "slot0", "jobs": ["rlpd-seed41"]},
                                    {"name": "slot1", "jobs": ["rlpd-seed42"]},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_dispatch_is_exact_partition_and_respects_parallel_quota(tmp_path):
    plan = _dispatch_plan(tmp_path)
    jobs = load_plan(plan)[0]
    dispatch = tmp_path / "dispatch.json"
    dispatch.write_text(json.dumps(_dispatch_payload(plan, jobs)))

    loaded, loaded_jobs, evidence = load_dispatch(dispatch)

    assert loaded["plan_id"] == "unit-dispatch"
    assert len(loaded_jobs) == 2
    assert evidence["plan_sha256"] == sha256(plan)


def test_dispatch_host_alias_is_verified_and_cannot_be_overridden(monkeypatch):
    dispatch = {
        "hosts": [
            {
                "hostname": "jtl110gpu",
                "hostname_aliases": ["huiwei-Super-Server"],
            }
        ]
    }
    monkeypatch.setattr(
        "reproducibility.run_training_dispatch_v6.socket.gethostname",
        lambda: "huiwei-Super-Server",
    )
    assert _select_host(dispatch, "jtl110gpu")["hostname"] == "jtl110gpu"
    with pytest.raises(RuntimeError, match="does not uniquely match"):
        _select_host(dispatch, "jtl110gpu2")


def test_v7_capacity_audit_is_bound_to_host_data_sources_and_freeze(tmp_path):
    capacity = tmp_path / "capacity.json"
    payload = {
        "schema_version": "transit-v7-gpu-capacity-audit-v1",
        "host_label": "jtl110gpu",
        "observed_hostname": "huiwei-Super-Server",
        "physical_gpu": "0",
        "cuda_visible_devices_for_workers": "0",
        "method": "rlpd",
        "phase": "online",
        "processes": 2,
        "dataset_sha256": "dataset-sha",
        "max_offline_rows": 0,
        "online_rows": 120_000,
        "device_cache": "float32",
        "warmup_updates": 100,
        "measured_updates": 5_000,
        "minimum_free_bytes": 2**30,
        "minimum_aggregate_speedup": 1.2,
        "minimum_worker_rate_fraction": 0.55,
        "reward_scale": 1e-4,
        "critic_loss": "huber",
        "huber_delta": 1.0,
        "target_entropy": 0.0,
        "decision": {
            "passed": True,
            "checks": {
                "all_workers_completed": True,
                "registered_processes_overlapped": True,
                "no_unregistered_compute_process": True,
                "stable_device_capacity": True,
                "minimum_free_memory_headroom": True,
                "identical_worker_source_hashes": True,
                "all_worker_rates_available": True,
                "minimum_aggregate_speedup": True,
                "minimum_worker_rate_fraction": True,
                "single_and_multi_process_source_hashes_match": True,
            },
        },
        "workers": [
            {"returncode": 0, "report_validation_failures": []},
            {"returncode": 0, "report_validation_failures": []},
        ],
        "source_sha256": {
            "reproducibility/audit_gpu_capacity_v7.py": "capacity-source",
            "reproducibility/benchmark_training_kernel_v4.py": "benchmark-source",
        },
        "benchmark_source_sha256": {
            "reproducibility/benchmark_training_kernel_v4.py": "benchmark-source",
            "SimpleSAC/train_transit_v3.py": "trainer-source",
        },
        "single_process_reference": {
            "sha256": "1" * 64,
            "updates_per_second": 100.0,
            "benchmark_source_sha256": {
                "reproducibility/benchmark_training_kernel_v4.py": "benchmark-source",
                "SimpleSAC/train_transit_v3.py": "trainer-source",
            },
        },
    }
    capacity.write_text(json.dumps(payload))
    plan = {
        "shared_arguments": {
            "online_capacity": 120_000,
            "device_cache_observation_dtype": "float32",
            "reward_scale": 1e-4,
            "critic_loss": "huber",
            "huber_delta": 1.0,
            "target_entropy": 0.0,
        },
        "registration": {
            "compute_policy": {
                "same_gpu_capacity_gate": {
                    "minimum_warmup_updates_per_process": 100,
                    "minimum_measured_updates_per_process": 5_000,
                    "minimum_free_memory_bytes": 2**30,
                    "single_process_reference_required": True,
                    "minimum_aggregate_speedup": 1.2,
                    "minimum_worker_rate_fraction": 0.55,
                    "protocol": {"sha256": "capacity-source"},
                    "benchmark_protocol": {"sha256": "benchmark-source"},
                }
            }
        },
    }
    evidence = {
        "dataset_sha256": "dataset-sha",
        "source_sha256": {
            "SimpleSAC/train_transit_v3.py": "trainer-source"
        },
    }
    descriptor = {"path": str(capacity), "sha256": sha256(capacity)}
    result = _validate_v7_capacity_audit(
        dispatch_path=tmp_path / "dispatch.json",
        descriptor=descriptor,
        plan_payload=plan,
        evidence=evidence,
        host={
            "hostname": "jtl110gpu",
            "hostname_aliases": ["huiwei-Super-Server"],
        },
        physical_gpu="0",
        jobs=[
            {
                "method": "rlpd",
                "max_canonical_processes_per_gpu": 2,
            }
        ],
    )
    assert result["sha256"] == sha256(capacity)

    payload["dataset_sha256"] = "another-dataset"
    capacity.write_text(json.dumps(payload))
    descriptor["sha256"] = sha256(capacity)
    with pytest.raises(ValueError, match="stale or incompatible"):
        _validate_v7_capacity_audit(
            dispatch_path=tmp_path / "dispatch.json",
            descriptor=descriptor,
            plan_payload=plan,
            evidence=evidence,
            host={
                "hostname": "jtl110gpu",
                "hostname_aliases": ["huiwei-Super-Server"],
            },
            physical_gpu="0",
            jobs=[
                {
                    "method": "rlpd",
                    "max_canonical_processes_per_gpu": 2,
                }
            ],
        )


def test_dispatch_rejects_over_quota_or_duplicate_assignments(tmp_path):
    plan = _dispatch_plan(tmp_path)
    jobs = load_plan(plan)[0]
    payload = _dispatch_payload(plan, jobs)
    payload["hosts"][0]["gpus"][0]["stages"][0]["workers"].append(
        {"name": "slot2", "jobs": ["rlpd-seed41"]}
    )
    dispatch = tmp_path / "dispatch-over-quota.json"
    dispatch.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="workers but quota 2"):
        load_dispatch(dispatch)

    payload = _dispatch_payload(plan, jobs)
    payload["hosts"][0]["gpus"][0]["stages"][0]["workers"][1]["jobs"] = [
        "rlpd-seed41"
    ]
    dispatch.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicates"):
        load_dispatch(dispatch)
