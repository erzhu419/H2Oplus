#!/usr/bin/env python3
"""Seal the v13 v1 pre-training admission incident without running training or SUMO."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEPLOYMENT = REPRO / "deployments/h2oplus_v13_proximal_release_pilot_v1"
WORKSPACE = DEPLOYMENT / "workspace"
RESULTS = DEPLOYMENT / "remote_results"
INCIDENT = (
    REPRO
    / "h2oplus_v13_proximal_release_incidents"
    / "v1_pretraining_admission_misbind"
)
SUBMISSION = REPRO / "h2oplus_v13_proximal_release_submission_v1.json"
SUBMISSION_VERIFICATION = (
    REPRO / "h2oplus_v13_proximal_release_submission_verification_v1.json"
)
SUBMISSION_SHA256 = (
    "2782fbbd02e618c536464fe87620591b2f8d180fbd3bc4e41e91bdaba481f277"
)
SUBMISSION_VERIFICATION_SHA256 = (
    "71ecbb8ff93e06c2dfbfbedcb63136a0b53f257598ee7f6a06e95d9562e78d28"
)
FAILED_QUEUE_IDS = {"t83252", "t83253", "t83254", "t83255"}
CANCELLED_QUEUE_IDS = {f"t{task}" for task in range(83256, 83267)}
BACKEND_FAILED_IDS = {
    "t83252",
    "t83253",
    "t83254",
    "t83255",
    "t83256",
    "t83257",
}
STARTED_CANCELLED_IDS = {"t83258", "t83260"}
QUEUED_CANCELLED_IDS = CANCELLED_QUEUE_IDS - {
    "t83256",
    "t83257",
    *STARTED_CANCELLED_IDS,
}
EXPECTED_MATERIALIZED_IDS = BACKEND_FAILED_IDS | STARTED_CANCELLED_IDS
EXPECTED_ERROR = "ValueError: v13 development admission is stale or misbound"
EXPECTED_V12_SOURCE_OMISSIONS = {
    "SimpleSAC/transit_v12/__init__.py",
    "SimpleSAC/transit_v12/provenance.py",
    "SimpleSAC/transit_v12/trainer.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def argument(job: dict[str, Any], name: str) -> str:
    values = job["command_arguments"]
    index = values.index(name)
    return values[index + 1]


def runtime_source_paths(wrapper: Path) -> set[str]:
    tree = ast.parse(wrapper.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_v13_source_sha256"
    ]
    if len(functions) != 1:
        raise ValueError("v13 wrapper lacks a unique runtime source-map function")
    assignments = [
        node
        for node in functions[0].body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "paths" for target in node.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Tuple):
        raise ValueError("v13 runtime source-map tuple is not statically auditable")
    paths = {"SimpleSAC/train_transit_v13_proximal_release.py"}
    for value in assignments[0].value.elts[1:]:
        if (
            not isinstance(value, ast.BinOp)
            or not isinstance(value.op, ast.Div)
            or not isinstance(value.right, ast.Constant)
            or not isinstance(value.right.value, str)
        ):
            raise ValueError("v13 runtime source-map path is not statically auditable")
        paths.add(value.right.value)
    return paths


def admission_diagnosis(protocol: dict[str, Any]) -> dict[str, Any]:
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from SimpleSAC import train_transit_v3 as base

    job = protocol["training_jobs"][0]
    admission_path = Path(job["training_admission"]).resolve(strict=True)
    admission = json.loads(admission_path.read_text())
    dataset_path = Path(argument(job, "--dataset"))
    simulator = base.simulator_input_hashes(
        Path(argument(job, "--sim-env")), runtime_calibration_version="v7_v4"
    )
    simulator["demand_scales"] = [0.75, 1.0, 1.25]
    simulator["demand_scale_schedule"] = "cyclic-by-episode"
    training_sources = base.source_hashes("v7")
    wrapper = WORKSPACE / "SimpleSAC/train_transit_v13_proximal_release.py"
    runtime_paths = runtime_source_paths(wrapper)
    runtime_sources = {
        relative: sha256(WORKSPACE / relative) for relative in sorted(runtime_paths)
    }
    protocol_path = WORKSPACE / "reproducibility/h2oplus_v13_control_v1/protocol.json"
    checks = {
        "schema_version": admission.get("schema_version")
        == "transit-training-admission-v13-proximal-release-development-v1",
        "study_stage": admission.get("study_stage") == "development",
        "passed": admission.get("passed") is True,
        "formal_held_out_outcomes_read": admission.get(
            "formal_held_out_outcomes_read"
        )
        is False,
        "dataset": admission.get("dataset", {}).get("sha256")
        == sha256(dataset_path),
        "simulator_aggregate": admission.get("simulator", {}).get(
            "aggregate_sha256"
        )
        == simulator.get("aggregate_sha256"),
        "simulator_runtime": admission.get("simulator", {}).get("runtime_sha256")
        == simulator.get("runtime_sha256"),
        "training_source_sha256": admission.get("training_source_sha256")
        == training_sources,
        "v13_source_sha256": admission.get("v13_source_sha256")
        == runtime_sources,
        "v13_protocol_sha256": admission.get("v13_protocol", {}).get("sha256")
        == sha256(protocol_path),
        "v13_wrapper_sha256": admission.get("v13_wrapper", {}).get("sha256")
        == sha256(wrapper),
        "variant": admission.get("variant") == job["variant"],
        "stage": admission.get("stage") == job["stage"],
        "method": admission.get("method") == argument(job, "--method"),
        "seed": admission.get("seed") == int(argument(job, "--seed")),
        "run_name": admission.get("run_name") == argument(job, "--run-name"),
        "run_signature": admission.get("run_signature") == job["run_signature"],
        "admission_path": admission_path
        == Path(job["training_admission"]).resolve(),
    }
    admission_sources = admission["v13_source_sha256"]
    diagnosis = {
        "schema_version": "transit-h2oplus-v13-pretraining-admission-diagnosis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "scientific_training_executed": False,
        "sumo_executed_by_diagnosis": False,
        "formal_held_out_outcomes_read": False,
        "checked_task": job["name"],
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "admission_only_source_paths": sorted(
            set(admission_sources) - set(runtime_sources)
        ),
        "runtime_only_source_paths": sorted(
            set(runtime_sources) - set(admission_sources)
        ),
        "shared_source_hash_mismatches": sorted(
            key
            for key in set(runtime_sources) & set(admission_sources)
            if runtime_sources[key] != admission_sources[key]
        ),
        "root_cause": (
            "The frozen admission correctly bound the inherited transit_v12 source "
            "layer, while the runtime _v13_source_sha256 function omitted those "
            "three files."
        ),
    }
    if (
        diagnosis["failed_checks"] != ["v13_source_sha256"]
        or set(diagnosis["admission_only_source_paths"])
        != EXPECTED_V12_SOURCE_OMISSIONS
        or diagnosis["runtime_only_source_paths"]
        or diagnosis["shared_source_hash_mismatches"]
    ):
        raise ValueError("v13 v1 admission failure is not the registered sole defect")
    return diagnosis


def scheduler_rows(task_ids: set[str]) -> list[dict[str, Any]]:
    queue_path = Path.home() / ".claude/scheduler/queue.json"
    payload = json.loads(queue_path.read_text())
    rows = payload.get("tasks", payload.get("queue", []))
    selected = [row for row in rows if row.get("id") in task_ids]
    if {row["id"] for row in selected} != task_ids:
        raise ValueError("v13 v1 task set is incomplete in scheduler queue")
    return sorted(selected, key=lambda row: int(row["id"][1:]))


def backend_evidence() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    evidence_root = INCIDENT / "remote_evidence"
    for path in sorted(evidence_root.glob("*/tmp/scheduleurm_exit_t832*_*.status")):
        task_id = path.name.split("_")[2]
        fields = path.read_text().strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"invalid backend evidence: {path}")
        if task_id in rows:
            raise ValueError(f"duplicate backend evidence: {task_id}")
        rows[task_id] = {
            "task_id": task_id,
            "node": path.parents[1].name,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "exit_code": int(fields[0]),
            "exit_epoch": int(fields[1]),
            "launch_token": fields[2],
        }
    if set(rows) != BACKEND_FAILED_IDS or any(
        row["exit_code"] != 1 for row in rows.values()
    ):
        raise ValueError("v13 v1 backend-failure set drifted")
    return rows


def log_evidence() -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted((INCIDENT / "remote_evidence").glob("*/tmp/sched_t832*.log")):
        task_id = path.stem[len("sched_") :]
        text = path.read_text()
        expected_nonempty = task_id in BACKEND_FAILED_IDS
        if expected_nonempty != bool(text):
            raise ValueError(f"v13 v1 log emptiness drifted: {task_id}")
        if expected_nonempty and EXPECTED_ERROR not in text:
            raise ValueError(f"v13 v1 log lacks the registered error: {task_id}")
        rows[task_id] = {
            "task_id": task_id,
            "node": path.parents[1].name,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "contains_registered_error": EXPECTED_ERROR in text,
        }
    expected = BACKEND_FAILED_IDS | STARTED_CANCELLED_IDS
    if set(rows) != expected:
        raise ValueError("v13 v1 synchronized log set drifted")
    return rows


def materialization_evidence(
    tasks: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from SimpleSAC.transit_v13.provenance import checkpoint_state_sha256

    jobs = {job["name"]: job for job in protocol["training_jobs"]}
    expected_names = {tasks[task_id]["name"] for task_id in EXPECTED_MATERIALIZED_IDS}
    observed_names = {path.name for path in RESULTS.iterdir() if path.is_dir()}
    if observed_names != expected_names:
        raise ValueError("v13 v1 materialized result-directory set drifted")
    rows = []
    for name in sorted(observed_names):
        run_dir = RESULTS / name
        job = jobs[name]
        task_id = next(
            task_id for task_id, task in tasks.items() if task["name"] == name
        )
        record_path = run_dir / "v13_branch_materialization.json"
        record = json.loads(record_path.read_text())
        checkpoint = run_dir / "checkpoints/latest.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        forbidden = [
            run_dir / "summary.json",
            run_dir / "v13_export_summary.json",
            run_dir / "host_execution.json",
            run_dir / "v13_policies",
        ]
        source_manifest = json.loads(Path(job["branch_handoff_manifest"]).read_text())
        if (
            record.get("passed") is not True
            or record.get("run_signature") != job["run_signature"]
            or record.get("variant") != job["variant"]
            or record.get("seed") != job["seed"]
            or checkpoint_state_sha256(checkpoint)
            != source_manifest["checkpoint_state_sha256"]
            or checkpoint_state_sha256(checkpoint)
            != record["checkpoint_state_sha256"]
            or int(payload.get("total_updates", -1)) != 55_000
            or int(payload.get("online_critic_updates", -1)) != 5_000
            or any(path.exists() for path in forbidden)
        ):
            raise ValueError(f"v13 v1 materialization is not a pristine 10k handoff: {name}")
        rows.append(
            {
                "task_id": task_id,
                "run_name": name,
                "node": tasks[task_id]["require_node"],
                "run_signature": job["run_signature"],
                "variant": job["variant"],
                "seed": job["seed"],
                "materialization": descriptor(record_path),
                "checkpoint": descriptor(checkpoint),
                "checkpoint_state_sha256": record["checkpoint_state_sha256"],
                "total_updates": int(payload["total_updates"]),
                "online_critic_updates": int(payload["online_critic_updates"]),
                "terminal_summary_present": False,
                "v13_policy_export_present": False,
                "host_execution_present": False,
                "optimizer_updates_after_v12_handoff": 0,
            }
        )
    return rows


def file_manifest() -> dict[str, Any]:
    roots = {
        "remote_results": RESULTS,
        "remote_evidence": INCIDENT / "remote_evidence",
    }
    entries = []
    for label, root in roots.items():
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            entries.append(
                {
                    "scope": label,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "schema_version": "transit-h2oplus-v13-pretraining-incident-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "formal_held_out_outcomes_read": False,
    }


def main() -> int:
    if sha256(SUBMISSION) != SUBMISSION_SHA256:
        raise ValueError("v13 v1 submission record drifted")
    if sha256(SUBMISSION_VERIFICATION) != SUBMISSION_VERIFICATION_SHA256:
        raise ValueError("v13 v1 submission verification drifted")
    submission = json.loads(SUBMISSION.read_text())
    task_ids = {row["task_id"] for row in submission["tasks"]}
    expected_ids = {f"t{task}" for task in range(83252, 83267)}
    if task_ids != expected_ids:
        raise ValueError("v13 v1 submission task set drifted")
    tasks = {row["task_id"]: row for row in submission["tasks"]}
    queue_rows = scheduler_rows(task_ids)
    statuses = {row["id"]: row["status"] for row in queue_rows}
    if (
        {task_id for task_id, status in statuses.items() if status == "failed"}
        != FAILED_QUEUE_IDS
        or {task_id for task_id, status in statuses.items() if status == "cancelled"}
        != CANCELLED_QUEUE_IDS
    ):
        raise ValueError("v13 v1 scheduler terminal states drifted")

    control = WORKSPACE / "reproducibility/h2oplus_v13_control_v1"
    protocol = json.loads((control / "protocol.json").read_text())
    diagnosis = admission_diagnosis(protocol)
    diagnosis_path = INCIDENT / "admission_predicate_diagnosis_v1.json"
    atomic_json(diagnosis_path, diagnosis)

    scheduler_payload = {
        "schema_version": "transit-h2oplus-v13-pretraining-incident-scheduler-snapshot-v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": queue_rows,
        "task_ids": sorted(task_ids, key=lambda value: int(value[1:])),
        "queue_failed_task_ids": sorted(FAILED_QUEUE_IDS),
        "queue_cancelled_task_ids": sorted(CANCELLED_QUEUE_IDS),
        "backend_nonzero_overrides_queue_cancelled_for": sorted(
            BACKEND_FAILED_IDS & CANCELLED_QUEUE_IDS
        ),
    }
    scheduler_path = INCIDENT / "scheduler_terminal_snapshot_v1.json"
    atomic_json(scheduler_path, scheduler_payload)

    backends = backend_evidence()
    logs = log_evidence()
    materializations = materialization_evidence(tasks, protocol)
    manifest_payload = file_manifest()
    manifest_path = INCIDENT / "incident_file_manifest_v1.json"
    atomic_json(manifest_path, manifest_payload)
    closeout = {
        "schema_version": "transit-h2oplus-v13-pretraining-admission-incident-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed-invalid-pretraining-lineage",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "remote_sumo_executed": False,
        "v13_scientific_training_executed": False,
        "scientific_result_registered": False,
        "invalid_lineage_reusable": False,
        "invalid_task_ids": sorted(task_ids, key=lambda value: int(value[1:])),
        "backend_failed_before_training_task_ids": sorted(BACKEND_FAILED_IDS),
        "started_then_cancelled_without_backend_task_ids": sorted(
            STARTED_CANCELLED_IDS
        ),
        "cancelled_before_start_task_ids": sorted(QUEUED_CANCELLED_IDS),
        "queue_status_note": (
            "t83256 and t83257 are queue-cancelled due to the exact batch cancellation, "
            "but their token-bound backend rc=1 evidence takes precedence."
        ),
        "root_cause": diagnosis["root_cause"],
        "sole_failed_admission_predicate": "v13_source_sha256",
        "missing_runtime_bindings": sorted(EXPECTED_V12_SOURCE_OMISSIONS),
        "materialized_v12_handoffs": materializations,
        "materialized_run_count": len(materializations),
        "materialization_interpretation": (
            "These directories contain exact v12 10k handoff copies and startup "
            "resource traces only; they are not v13 trained checkpoints or outcomes."
        ),
        "backend_evidence": [backends[key] for key in sorted(backends)],
        "log_evidence": [logs[key] for key in sorted(logs)],
        "submission": descriptor(SUBMISSION),
        "submission_verification": descriptor(SUBMISSION_VERIFICATION),
        "scheduler_snapshot": descriptor(scheduler_path),
        "admission_diagnosis": descriptor(diagnosis_path),
        "file_manifest": descriptor(manifest_path),
        "replacement_policy": (
            "Any correction must use a fresh deployment, controls, admissions, output "
            "root, run names, scheduler signatures and task IDs, and must pass an "
            "executable no-training admission preflight before submission."
        ),
    }
    output = INCIDENT / "incident_closeout_v1.json"
    atomic_json(output, closeout)
    print(
        json.dumps(
            {
                "closeout": descriptor(output),
                "file_manifest": descriptor(manifest_path),
                "admission_diagnosis": descriptor(diagnosis_path),
                "scheduler_snapshot": descriptor(scheduler_path),
                "backend_failed_before_training": len(BACKEND_FAILED_IDS),
                "started_then_cancelled": len(STARTED_CANCELLED_IDS),
                "cancelled_before_start": len(QUEUED_CANCELLED_IDS),
                "materialized_v12_handoffs": len(materializations),
                "v13_scientific_training_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
