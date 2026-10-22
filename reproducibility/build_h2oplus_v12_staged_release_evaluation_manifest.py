#!/usr/bin/env python3
"""Audit and seal the H2O+ v12 staged-release development evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v12_staged_release_results"
PARTITIONS = RESULTS / "evaluation_partitions_v1"
CANONICAL = RESULTS / "evaluation_canonical_v1"
EVIDENCE = RESULTS / "evidence/evaluation_v1"
QUEUE = Path("/home/erzhu419/.claude/scheduler/queue.json")

AUDIT_PATH = RESULTS / "evaluation_audit_v1.json"
SCHEDULER_EVIDENCE_PATH = EVIDENCE / "scheduler_evidence_v1.json"
MANIFEST_PATH = RESULTS / "evaluation_file_manifest_v1.json"

EXPECTED_HASHES = {
    "registration": "59cb1e14ad0bbc8944472f5bbf3e849a62b65c4517766e3fd3845b76a5ec3ba3",
    "authorization": "bb1eeda1c451916aba7576f0592418471f42afaefe499386482be540def7fa23",
    "correction": "40dabd0dddab6ce7fd8f08ac025f8cc96a5545d1c36ff24730ce885b5eb67b25",
    "environment": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "plan": "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa",
    "protocol": "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
    "freeze": "0d8ab1e7852c679ae6e444d1c767799635c8790e4bc787587210a63b0e92dfe1",
    "methods": "d61acae2314d0588b0dc6aa84c0a0bf2f168d4bdf1f18c5d19b2d0dbdec5c374",
    "source_freeze": "2ab7b6c4d8eb94977df097ab37ff8671becdd11fbf4c2ffe562eecb6bce6cd28",
    "local_verification": "8dd611ab2d434fd03e9ab445e705fc9023e6ebd47dd2d30f3dc42fd5b14d8c06",
    "deployment_sync": "f882c926cb6badd8699d49a95d79ba4b5d7a6ffdad31d69d82ff240684cd0b56",
    "remote_verification": "ea3e403f44482ff92c3fd4758198dea62a40c914472a71df6875a722d887f038",
    "submission": "331a62b1659f9f309718443e75c9e952ff64116c48272da14f1d9296fa603dd7",
    "training_closeout": "998f47488b50b2cd40f9a6ddd5a399712bb9208cb1ef883cad1181f800385b66",
    "training_verification": "6835e3c545be22c25bd9e3c7eae7ef987bb1864e027c4fef9849788f744868c8",
    "training_audit": "8e34c4cd69f86d7333dfd5d2a57696ce427388349c07c377bb279b957fd1bfc7",
    "training_manifest": "9ed7ccbddf0832e0c8b4532421e000745f9c60f03b9dfc01a40d970503c2b08b",
    "deployment": "68155a903e305ce04cd1654487502ae5356219ba43b1fc953f474efc5eb8e973",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "scheduler_sync": "e9ea76b428831f6669eb625506ca76fb401aec07f098899a4a6528f2991bc379",
    "source_sync": "4d4076e6c5c9d6a7c882302eca8252220cf8c8554e75c871ac6db88f077dd193",
}

CONTROL_PATHS = {
    "registration": REPO / "reproducibility/h2oplus_v12_staged_release_registration_v2.json",
    "authorization": REPO / "reproducibility/h2oplus_v12_staged_release_authorization_v2.json",
    "correction": REPO
    / "reproducibility/h2oplus_v12_staged_release_deployment_correction_v2.json",
    "environment": REPO / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
    "plan": REPO / "reproducibility/h2oplus_v12_staged_release_training_plan_v2.json",
    "protocol": REPO / "reproducibility/h2oplus_v12_staged_release_protocol_v2.json",
    "freeze": REPO / "reproducibility/h2oplus_v12_staged_release_freeze_v2.json",
    "methods": REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_methods_v1.json",
    "source_freeze": REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_source_freeze_v1.json",
    "local_verification": REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_deployment_local_verification_v1.json",
    "deployment_sync": REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_deployment_sync_v1.json",
    "remote_verification": REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_deployment_v1_jtl110gpu_verification.json",
    "submission": REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_submission_v1.json",
    "training_closeout": REPO
    / "reproducibility/h2oplus_v12_staged_release_branch_closeout_v2.json",
    "training_verification": REPO
    / "reproducibility/h2oplus_v12_staged_release_branch_closeout_verification_v2.json",
    "training_audit": RESULTS / "branch_audit_v2.json",
    "training_manifest": RESULTS / "branch_file_manifest_v2.json",
    "deployment": REPO / "reproducibility/deployments/h2oplus_v12_staged_release_evaluation_v1/workspace/deployment_manifest.json",
    "runtime_lock": REPO / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    "scheduler_sync": EVIDENCE / "scheduler_evidence_v1.json",
    "source_sync": EVIDENCE / "source_sync_v1.json",
}

EXPECTED_DEPLOYMENT_AGGREGATE = (
    "3193c5a556c81b7f3103dedd4bec4065b93db4290d67c94fce20d819eb29b355"
)
EXPECTED_DATASET_SHA256 = (
    "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
)
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_ENV = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
        "deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/"
        "toolchain/libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
}
TASKS = {
    "t83115": {
        "shard": "shard0_jtl110gpu",
        "status_file": EVIDENCE / "backend_exit_t83115.status",
        "log_file": EVIDENCE / "scheduler_t83115.log",
        "expected_jobs": 315,
    },
    "t83116": {
        "shard": "shard1_jtl110gpu",
        "status_file": EVIDENCE / "backend_exit_t83116.status",
        "log_file": EVIDENCE / "scheduler_t83116.log",
        "expected_jobs": 315,
    },
}
BLOCKS = (
    "v12-staged-release-development-01",
    "v12-staged-release-development-02",
    "v12-staged-release-development-03",
)
SCALES = {0.85, 1.0, 1.15}
RESULT_SOURCE_SUFFIXES = {
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
}
METHOD_BINDING_FIELDS = {
    "actor_tensor_sha256",
    "actual_online_events",
    "algorithm",
    "artifact_lineage",
    "checkpoint_sha256",
    "learner_update_steps",
    "name",
    "online_events",
    "ratio_optimizer_steps",
    "training_admission_sha256",
    "training_checkpoint_sha256",
    "training_run_signature",
    "training_seed",
    "type",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return False


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def file_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def records_aggregate(records: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _suffix_hash(source: dict, suffix: str) -> str | None:
    values = [value for path, value in source.items() if path.endswith(suffix)]
    return values[0] if len(values) == 1 else None


def validate_controls() -> tuple[dict, set[Path], dict[str, dict], dict[str, str]]:
    records = {}
    for name, path in CONTROL_PATHS.items():
        actual = sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"control identity drifted for {name}: {actual}")
        records[name] = file_record(path)

    deployment = load_json(CONTROL_PATHS["deployment"])
    submission = load_json(CONTROL_PATHS["submission"])
    methods_payload = load_json(CONTROL_PATHS["methods"])
    source_freeze = load_json(CONTROL_PATHS["source_freeze"])
    training_closeout = load_json(CONTROL_PATHS["training_closeout"])
    training_verification = load_json(CONTROL_PATHS["training_verification"])
    if (
        deployment.get("aggregate_sha256") != EXPECTED_DEPLOYMENT_AGGREGATE
        or len(deployment.get("entries", [])) != 612
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or submission.get("physical_actor_count") != 70
        or submission.get("logical_method_budget_alias_count") != 180
        or submission.get("total_remote_sumo_episodes") != 630
        or methods_payload.get("schema_version")
        != "transit-evaluation-methods-v12-staged-release-v1"
        or methods_payload.get("physical_method_count") != 70
        or methods_payload.get("logical_method_budget_alias_count") != 180
        or len(methods_payload.get("methods", [])) != 70
        or len(methods_payload.get("aliases", [])) != 180
        or source_freeze.get("source_file_count") != 6
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("evaluation_outcomes_read") is not False
        or training_closeout.get("artifact_integrity_passed") is not True
        or training_closeout.get(
            "scientific_stability_passed_for_all_30_training_runs"
        )
        is not True
        or training_closeout.get("exact_seed_matched_10k_handoffs_passed") is not True
        or training_closeout.get("training_run_count") != 30
        or training_closeout.get("development_evaluation_gate_open") is not True
        or training_verification.get("status") != "passed"
        or training_verification.get("artifact_integrity_passed") is not True
        or training_verification.get(
            "scientific_stability_passed_for_all_30_training_runs"
        )
        is not True
        or training_verification.get("exact_seed_matched_10k_handoffs_passed")
        is not True
    ):
        raise ValueError("v12 evaluation control contract failed")
    methods = {item["name"]: item for item in methods_payload["methods"]}
    if len(methods) != 70:
        raise ValueError("duplicate physical methods in v12 manifest")
    aliases = methods_payload["aliases"]
    if len({item["name"] for item in aliases}) != 180:
        raise ValueError("duplicate logical aliases in v12 manifest")
    for alias in aliases:
        physical = methods.get(alias.get("physical_method"))
        if (
            physical is None
            or alias.get("training_seed") != physical.get("training_seed")
            or alias.get("actor_tensor_sha256") != physical.get("actor_tensor_sha256")
        ):
            raise ValueError("v12 logical alias identity binding failed")
    identities = methods_payload.get("identity_evidence", [])
    if (
        len(identities) != 5
        or {item.get("training_seed") for item in identities} != set(range(6201, 6206))
        or any(item.get("v12_logical_aliases") != 20 for item in identities)
        or any(
            item.get("prefix_zero_5k_10k_equal_fixed_h2oplus_zero") is not True
            for item in identities
        )
        or any(
            item.get("prefix_zero_5k_10k_temperature_equal") is not True
            for item in identities
        )
        or any(len(item.get("prefix_source_policies", [])) != 3 for item in identities)
        or any(len(item.get("branch_source_policies", [])) != 5 for item in identities)
    ):
        raise ValueError("v12 actor/temperature identity evidence failed")
    frozen_sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    if len(frozen_sources) != 6:
        raise ValueError("v12 source-freeze cardinality drifted")
    source_files = {REPO / path for path in frozen_sources}
    if any(sha256(path) != frozen_sources[relative(path)] for path in source_files):
        raise ValueError("v12 frozen source identity drifted")
    return records, set(CONTROL_PATHS.values()) | source_files, methods, frozen_sources


def _queue_tasks() -> dict[str, dict]:
    payload = load_json(QUEUE)
    rows = payload.get("tasks", payload if isinstance(payload, list) else [])
    return {row["id"]: row for row in rows if row.get("id") in TASKS}


def validate_scheduler() -> tuple[dict, set[Path]]:
    queue_tasks = _queue_tasks()
    submission = load_json(CONTROL_PATHS["submission"])
    submitted = {item["task_id"]: item for item in submission["tasks"]}
    if set(queue_tasks) != set(TASKS) or set(submitted) != set(TASKS):
        raise ValueError("v12 scheduler task coverage drifted")
    evidence = {}
    files: set[Path] = set()
    for task_id, expected in TASKS.items():
        task = queue_tasks[task_id]
        task_submission = submitted[task_id]
        status_path = expected["status_file"]
        log_path = expected["log_file"]
        fields = status_path.read_text().strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"malformed token-bound backend status for {task_id}")
        exit_code, backend_finished_at, token = fields
        shard_index = int(task_submission["shard_index"])
        host_path = PARTITIONS / expected["shard"] / "host_execution.json"
        host = load_json(host_path)
        env = task.get("extra_env", {})
        if (
            exit_code != "0"
            or token != task.get("exit_status_token")
            or int(backend_finished_at) != int(task.get("backend_finished_at"))
            or task.get("status") != "done"
            or task.get("exit_code") != 0
            or task.get("node") != "jtl110gpu"
            or task.get("cpu_declared_cores") != 12
            or task.get("est_vram_mb") != 0
            or task.get("signature") != task_submission.get("signature")
            or task_submission.get("cpu_cores") != 12
            or task_submission.get("ram_mb") != 32768
            or task_submission.get("est_vram_mb") != 0
            or any(env.get(name) != value for name, value in EXPECTED_ENV.items())
            or "CUDA_VISIBLE_DEVICES" in env
            or host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("workers") != 12
            or host.get("training") is not False
            or host.get("formal_held_out_outcomes_read") is not False
            or host.get("evidence", {}).get("training_scientific_stability_passed") is not True
            or host.get("evidence", {}).get("stability_failure_runs") != []
            or sum(item.get("selected_job_count", 0) for item in host.get("completed_blocks", []))
            != expected["expected_jobs"]
            or log_path.read_text().strip()
            != f"DONE H2O+ v12 staged-release shard {shard_index}: {expected['expected_jobs']}/{expected['expected_jobs']} jobs"
        ):
            raise ValueError(f"scheduler/backend/host contract failed for {task_id}")
        files.update({status_path, log_path, host_path})
        evidence[task_id] = {
            "backend_finished_at": int(backend_finished_at),
            "cpu_declared_cores": 12,
            "exit_code": 0,
            "exit_status_token": token,
            "host_execution": file_record(host_path),
            "node": "jtl110gpu",
            "queue_finished_at": task.get("finished_at"),
            "queue_status": "done",
            "scheduler_log": file_record(log_path),
            "status_file": file_record(status_path),
        }
    payload = load_json(SCHEDULER_EVIDENCE_PATH)
    if (
        payload.get("schema_version")
        != "transit-h2oplus-v12-staged-release-evaluation-scheduler-evidence-v1"
        or payload.get("passed") is not True
        or payload.get("token_bound_backend_exit0") is not True
        or set(payload.get("tasks", {})) != set(TASKS)
    ):
        raise ValueError("sealed v12 scheduler evidence failed")
    for task_id, checked in evidence.items():
        sealed = payload["tasks"][task_id]
        if (
            sealed.get("backend_returncode") != 0
            or sealed.get("backend_token") != checked["exit_status_token"]
            or sealed.get("status") != "done"
            or sealed.get("token_matches") is not True
            or sealed.get("residual_evaluation_process_count") != 0
        ):
            raise ValueError(f"sealed v12 scheduler task evidence failed: {task_id}")
    files.add(SCHEDULER_EVIDENCE_PATH)
    return payload, files


def _validate_training(payload: dict, method: dict) -> None:
    training = payload.get("policy", {}).get("training_metadata", {})
    admission = training.get("training_admission", {})
    common = {
        "training_seed": method["training_seed"],
        "run_signature": method["training_run_signature"],
        "training_checkpoint_sha256": method["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if any(training.get(key) != value for key, value in common.items()):
        raise ValueError("v12 result common training provenance drifted")
    if method["artifact_lineage"] == "v8-fewshot":
        fields = {
            "method": method["algorithm"],
            "fewshot_online_events_nominal": method["online_events"],
            "fewshot_online_events_actual": method["actual_online_events"],
            "fewshot_learner_update_steps": method["learner_update_steps"],
            "fewshot_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        protocol_hash = training.get("fewshot_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v8-fewshot-development-v1"
    elif method["artifact_lineage"] == "v10-delayed":
        variant = method["algorithm"][len("h2oplus-v10-") :]
        fields = {
            "method": "h2oplus",
            "variant": variant,
            "v10_online_events_nominal": method["online_events"],
            "v10_online_events_actual": method["actual_online_events"],
            "v10_learner_update_steps": method["learner_update_steps"],
            "v10_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        intervention = training.get("v10_intervention", {})
        if intervention.get("variant") != variant or intervention.get("unit_ratio_weights") is not True:
            raise ValueError("v10 intervention provenance drifted")
        protocol_hash = training.get("v10_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v10-delayed-development-v4"
    elif method["artifact_lineage"] == "v11-actor-hold":
        fields = {
            "method": "h2oplus",
            "variant": "actor-hold-online-only",
            "stage": "candidate",
            "v11_online_events_nominal": method["online_events"],
            "v11_online_events_actual": method["actual_online_events"],
            "v11_learner_update_steps": method["learner_update_steps"],
            "v11_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v11_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v11_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        intervention = training.get("v11_intervention", {})
        if (
            intervention.get("variant") != "actor-hold-online-only"
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError("v11 intervention provenance drifted")
        protocol_hash = training.get("v11_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v11-actor-hold-development-v2"
    elif method["artifact_lineage"] == "v12-staged-release":
        variant = method["algorithm"][len("h2oplus-v12-") :]
        fields = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v12_online_events_nominal": method["online_events"],
            "v12_online_events_actual": method["actual_online_events"],
            "v12_learner_update_steps": method["learner_update_steps"],
            "v12_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v12_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v12_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        flags = {
            "abrupt-control": (False, False, False),
            "actor-ramp": (True, False, False),
            "temperature-ramp": (False, True, False),
            "actor-anchor": (False, False, True),
            "staged-combined": (True, True, True),
        }[variant]
        intervention = training.get("v12_intervention", {})
        if (
            intervention.get("variant") != variant
            or intervention.get("actor_learning_rate_ramp") is not flags[0]
            or intervention.get("temperature_learning_rate_ramp") is not flags[1]
            or intervention.get("actor_state_anchor") is not flags[2]
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError("v12 intervention provenance drifted")
        protocol_hash = training.get("v12_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v12-staged-release-development-v2"
    else:
        raise ValueError("unsupported v12 physical actor lineage")
    if (
        any(training.get(key) != value for key, value in fields.items())
        or admission.get("schema_version") != admission_schema
        or admission.get("sha256") != method["training_admission_sha256"]
        or protocol_hash != method["source_protocol_sha256"]
    ):
        raise ValueError("v12 result lineage-specific training provenance drifted")


def validate_result(
    job: dict,
    output: Path,
    method: dict,
    source_reference: dict | None,
    frozen_sources: dict[str, str],
) -> dict:
    payload = load_json(output)
    job_method = job.get("method", {})
    source = payload.get("source_sha256", {})
    coverage = payload.get("passenger_welfare", {}).get("chronological_coverage_errors", {})
    semantics = payload.get("bus_service_semantics_audit", {})
    ledger = payload.get("ledger_audit", {})
    passengers = payload.get("passengers", {})
    welfare = payload.get("passenger_welfare", {})
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or int(payload.get("sumo_seed")) != int(job.get("sumo_seed"))
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 43
        or any(_suffix_hash(source, suffix) != frozen_sources[suffix] for suffix in RESULT_SOURCE_SUFFIXES)
        or any(job_method.get(key) != method.get(key) for key in METHOD_BINDING_FIELDS)
        or job_method.get("training_plan_sha256") != method.get("source_protocol_sha256")
        or job.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("runtime", {}).get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("policy", {}).get("sha256") != job_method.get("checkpoint_sha256")
        or payload.get("policy", {}).get("actor_tensor_sha256") != method.get("actor_tensor_sha256")
        or method.get("artifact_lineage") == "v12-staged-release"
        and payload.get("policy", {}).get("temperature_tensor_sha256")
        != method.get("temperature_tensor_sha256")
        or payload.get("scenario", {}).get("manifest_sha256") != job.get("scenario", {}).get("manifest_sha256")
        or not coverage
        or any(abs(float(value)) > 1e-9 for value in coverage.values())
        or semantics.get("passed") is not True
        or semantics.get("require_all_departed_trips_completed") is not True
        or ledger.get("legacy_sampled_trip_histories_used_as_ground_truth") is not False
        or passengers.get("legacy_sampled_trip_histories_used_for_welfare") is not False
        or passengers.get("active_state_count_balance_error") != 0
        or welfare.get("primary") is not True
        or abs(float(welfare.get("maximum_reward_reconstruction_error", math.inf))) > 1e-9
    ):
        raise ValueError(f"v12 result provenance or service semantics failed: {output}")
    _validate_training(payload, method)
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across sealed v12 results: {output}")
    return source


def validate_partitions(
    methods: dict[str, dict], frozen_sources: dict[str, str]
) -> tuple[dict, set[Path], set[str], set[str]]:
    stats = {}
    files: set[Path] = set()
    signatures: set[str] = set()
    source_manifest_hashes: set[str] = set()
    source_reference = None
    for shard_index, shard in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = PARTITIONS / shard
        shard_files = {path for path in root.rglob("*") if path.is_file()}
        results = sorted(root.rglob("sumo_seed*.json"))
        manifests = sorted(root.rglob("matrix_manifest.shard-*.json"))
        host = load_json(root / "host_execution.json")
        if (
            len(results) != TASKS[("t83115", "t83116")[shard_index]]["expected_jobs"]
            or len(manifests) != 3
            or host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("training") is not False
            or host.get("formal_held_out_outcomes_read") is not False
            or len(host.get("completed_blocks", [])) != 3
            or sum(item.get("selected_job_count", 0) for item in host["completed_blocks"])
            != TASKS[("t83115", "t83116")[shard_index]]["expected_jobs"]
        ):
            raise ValueError(f"partition host/result contract failed: {shard}")
        completed = {Path(item["path"]).parent.name: item for item in host["completed_blocks"]}
        shard_signatures = set()
        for manifest_path in manifests:
            block = manifest_path.parent.name
            manifest = load_json(manifest_path)
            jobs = manifest.get("jobs", [])
            partition = manifest.get("matrix_partition", {})
            item = completed.get(block)
            if (
                block not in BLOCKS
                or item is None
                or item.get("sha256") != sha256(manifest_path)
                or item.get("selected_job_count") != 105
                or item.get("source_file_count") != 43
                or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
                or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
                or manifest.get("methods_manifest_sha256") != EXPECTED_HASHES["methods"]
                or manifest.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
                or len(manifest.get("source_sha256", {})) != 43
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count") != 105
                or partition.get("full_job_count") != 210
                or len(jobs) != 105
            ):
                raise ValueError(f"partition matrix contract failed: {manifest_path}")
            source_manifest_hashes.add(sha256(manifest_path))
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                output = Path(job["output"]).expanduser().resolve()
                if method is None or not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"missing, out-of-root, or unregistered partition output: {output}")
                source_reference = validate_result(job, output, method, source_reference, frozen_sources)
                signature = str(job["signature"])
                if signature in signatures:
                    raise ValueError("duplicate v12 source-partition signature")
                signatures.add(signature)
                shard_signatures.add(signature)
        expected_jobs = TASKS[("t83115", "t83116")[shard_index]]["expected_jobs"]
        if len(shard_signatures) != expected_jobs:
            raise ValueError(f"partition signature count drifted: {shard}")
        text_logs = [path for path in shard_files if path.suffix in {".log", ".out", ".err"}]
        bad_markers = ("Traceback (most recent call last)", "Segmentation fault", "FATAL:")
        if any(marker in path.read_text(errors="replace") for path in text_logs for marker in bad_markers):
            raise ValueError(f"text-log failure marker in partition: {shard}")
        files.update(shard_files)
        stats[shard] = {
            "file_count": len(shard_files),
            "matrix_manifest_count": 3,
            "result_count": expected_jobs,
            "unique_job_signature_count": expected_jobs,
        }
    if len(signatures) != 630 or len(source_manifest_hashes) != 6 or source_reference is None:
        raise ValueError("v12 source partition union failed")
    return stats, files, signatures, source_manifest_hashes


def validate_canonical(
    methods: dict[str, dict], partition_signatures: set[str], source_manifest_hashes: set[str]
) -> tuple[dict, set[Path]]:
    files = set()
    signatures = set()
    outputs = set()
    block_stats = {}
    expected_cells = {(name, scale) for name in methods for scale in SCALES}
    for block_index, block in enumerate(BLOCKS, start=1):
        path = CANONICAL / block / "matrix_manifest.json"
        matrix = load_json(path)
        jobs = matrix.get("jobs", [])
        merge = matrix.get("matrix_merge", {})
        cells = set()
        if (
            len(jobs) != 210
            or matrix.get("schema_version") != "transit-evaluation-matrix-v7"
            or matrix.get("methods_manifest_sha256") != EXPECTED_HASHES["methods"]
            or matrix.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
            or merge.get("all_outputs_verified") is not True
            or merge.get("method_count") != 70
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
            or not {item.get("sha256") for item in merge["source_manifests"]}.issubset(source_manifest_hashes)
        ):
            raise ValueError(f"canonical matrix contract failed: {block}")
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            signature = str(job["signature"])
            output = Path(job["output"]).expanduser().resolve()
            if (
                name not in methods
                or scale not in SCALES
                or job.get("environment_block")
                not in {block, f"sumo-seed-{5100 + block_index}"}
                or signature in signatures
                or output in outputs
                or signature not in partition_signatures
                or not output.is_file()
            ):
                raise ValueError("canonical v12 signature/output binding failed")
            payload = load_json(output)
            if payload.get("evaluation_job", {}).get("signature") != signature or not finite(payload):
                raise ValueError(f"canonical output validation failed: {output}")
            cells.add((name, scale))
            signatures.add(signature)
            outputs.add(output)
        if cells != expected_cells:
            raise ValueError(f"incomplete canonical v12 Cartesian matrix: {block}")
        files.add(path)
        block_stats[block] = {
            "job_count": 210,
            "matrix_manifest_sha256": sha256(path),
            "physical_method_count": 70,
            "source_manifest_count": 2,
        }
    if signatures != partition_signatures or len(outputs) != 630:
        raise ValueError("canonical v12 union differs from source partitions")
    return {
        "blocks": block_stats,
        "demand_scales": sorted(SCALES),
        "matrix_count": 3,
        "physical_method_count": 70,
        "result_count": 630,
        "unique_job_signature_count": 630,
    }, files


def build() -> tuple[dict, dict]:
    controls, control_files, methods, frozen_sources = validate_controls()
    scheduler, scheduler_files = validate_scheduler()
    partition_stats, partition_files, signatures, source_manifest_hashes = validate_partitions(methods, frozen_sources)
    canonical_stats, canonical_files = validate_canonical(methods, signatures, source_manifest_hashes)
    audit = {
        "schema_version": "transit-h2oplus-v12-staged-release-evaluation-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 3,
            "corrected_source_file_count": 43,
            "logical_method_budget_alias_count": 180,
            "passed": True,
            "physical_method_count": 70,
            "result_count": 630,
            "source_manifest_count": 6,
            "source_partition_count": 2,
            "unique_job_signature_count": 630,
        },
        "canonical": canonical_stats,
        "controls": controls,
        "deployment_aggregate_sha256": EXPECTED_DEPLOYMENT_AGGREGATE,
        "partitions": partition_stats,
        "scheduler": scheduler,
        "path_resolution": {
            "original_partition_manifests_rewritten": False,
            "local_remote_results_symlink": "reproducibility/deployments/h2oplus_v12_staged_release_evaluation_v1/remote_results_v1",
            "symlink_target": "../../h2oplus_v12_staged_release_results/evaluation_partitions_v1",
        },
        "training_scientific_stability": {
            "all_30_training_runs_stability_gate_passed": True,
            "artifact_integrity_passed": True,
            "failure_runs": [],
            "separate_from_artifact_integrity": True,
        },
        "scientific_results": {
            "analysis_completed": False,
            "artifact_integrity_is_not_scientific_performance": True,
            "formal_confirmation_authorized_or_submitted": False,
            "preregistered_gates_applied": False,
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "passed": True,
    }
    atomic_json(AUDIT_PATH, audit)

    tool_files = {
        Path(__file__).resolve(),
        REPO / "reproducibility/verify_h2oplus_v12_staged_release_evaluation_manifest.py",
        REPO / "reproducibility/sync_h2oplus_v12_staged_release_evaluation_v1.py",
        REPO / "reproducibility/merge_evaluation_matrix_v7.py",
        REPO / "reproducibility/analyze_h2oplus_v12_staged_release.py",
    }
    files = control_files | scheduler_files | partition_files | canonical_files | tool_files | {AUDIT_PATH}
    records = [file_record(path) for path in sorted(files)]
    manifest = {
        "schema_version": "transit-h2oplus-v12-staged-release-evaluation-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": audit["artifact_integrity"],
        "audit": file_record(AUDIT_PATH),
        "file_count": len(records),
        "file_records_aggregate_sha256": records_aggregate(records),
        "files": records,
        "formal_held_out_outcomes_read": False,
        "scientific_analysis": {
            "completed": False,
            "formal_confirmation_authorized_or_submitted": False,
        },
        "study_stage": "development",
        "supports_current_formal_held_out_model_effect_claim": False,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
    }
    atomic_json(MANIFEST_PATH, manifest)
    return audit, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--scheduler-evidence", type=Path, default=SCHEDULER_EVIDENCE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global AUDIT_PATH, MANIFEST_PATH, SCHEDULER_EVIDENCE_PATH
    AUDIT_PATH = args.audit.expanduser().resolve()
    MANIFEST_PATH = args.manifest.expanduser().resolve()
    SCHEDULER_EVIDENCE_PATH = args.scheduler_evidence.expanduser().resolve()
    _, manifest = build()
    print(MANIFEST_PATH)
    print(manifest["file_records_aggregate_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
