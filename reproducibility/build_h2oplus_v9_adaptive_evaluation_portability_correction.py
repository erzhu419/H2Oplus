#!/usr/bin/env python3
"""Seal the v1 portability incident and emit a relocatable v9 methods manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-portability-correction-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v9-adaptive-v1"
EXPECTED_METHODS_V1 = "f0245dd15a0b89bc1d654b52b27f9c4560a4704aad75d9ce1e1fdb7ae5d8bf76"
EXPECTED_COUNT_CORRECTION = "4fc9324b140f6af387a4b8011b40c1d6994c57636d909ca065b17d4d0e27c6ba"
EXPECTED_DEPLOYMENT_V1 = "70f096bec8cb211bd97641c38e2ad6836f04e72bd7574ab9663a6ee4a0614c8c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, owner: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    shown = os.path.relpath(resolved, owner.parent) if owner is not None else str(resolved)
    return {"path": shown, "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods-v1",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--count-correction",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_evaluation_count_correction_v1.json",
    )
    parser.add_argument(
        "--submission-v1",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_evaluation_submission_v2.json",
    )
    parser.add_argument(
        "--deployment-v1",
        type=Path,
        default=REPRO
        / "deployments/h2oplus_v9_adaptive_evaluation_v1/workspace/deployment_manifest.json",
    )
    parser.add_argument(
        "--remote-verification-v1",
        type=Path,
        default=REPRO
        / "h2oplus_v9_adaptive_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--incident-root",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_incidents/evaluation_v1_portability",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path.home() / ".claude/scheduler/queue.json",
    )
    parser.add_argument(
        "--methods-output",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_evaluation_methods_v2.json",
    )
    parser.add_argument(
        "--correction-output",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_evaluation_portability_correction_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods_v1 = args.methods_v1.expanduser().resolve(strict=True)
    count_correction = args.count_correction.expanduser().resolve(strict=True)
    submission_v1 = args.submission_v1.expanduser().resolve(strict=True)
    deployment_v1 = args.deployment_v1.expanduser().resolve(strict=True)
    verification_v1 = args.remote_verification_v1.expanduser().resolve(strict=True)
    incident_root = args.incident_root.expanduser().resolve(strict=True)
    queue_path = args.queue.expanduser().resolve(strict=True)
    methods_output = args.methods_output.expanduser().resolve()
    correction_output = args.correction_output.expanduser().resolve()
    if methods_output.exists() or correction_output.exists():
        raise FileExistsError("refusing to replace an existing portability correction")
    if sha256(methods_v1) != EXPECTED_METHODS_V1:
        raise ValueError("v1 methods identity drifted")
    if sha256(count_correction) != EXPECTED_COUNT_CORRECTION:
        raise ValueError("count correction identity drifted")
    if sha256(deployment_v1) != EXPECTED_DEPLOYMENT_V1:
        raise ValueError("v1 deployment identity drifted")

    host_execution = incident_root / "jtl110gpu/shard0_jtl110gpu/host_execution.json"
    exit_status = incident_root / "jtl110gpu/scheduleurm_exit_t82565_80c641a7fe95af89a97eb647.status"
    scheduler_log = incident_root / "jtl110gpu/sched_t82565.log"
    execution = json.loads(host_execution.read_text())
    if (
        execution.get("status") != "failed"
        or execution.get("returncode") != 1
        or execution.get("completed_blocks") != []
        or "h2oplus_v9_adaptive_evaluation_count_correction_v1.json" not in execution.get("error", "")
        or list(incident_root.rglob("sumo_seed*.json"))
        or list(incident_root.rglob("matrix_manifest*.json"))
    ):
        raise ValueError("v1 operational failure evidence drifted")
    exit_parts = exit_status.read_text().split()
    if exit_parts[:1] != ["1"] or len(exit_parts) != 3:
        raise ValueError("v1 backend exit evidence drifted")

    queue = json.loads(queue_path.read_text())
    tasks = queue if isinstance(queue, list) else queue.get("tasks", queue.get("queue", []))
    task_map = {task.get("id"): task for task in tasks if task.get("id") in {"t82565", "t82566"}}
    failed = task_map.get("t82565", {})
    cancelled = task_map.get("t82566", {})
    if (
        failed.get("status") != "failed"
        or failed.get("exit_code") != 1
        or failed.get("exit_status_token") != exit_parts[2]
        or cancelled.get("status") != "cancelled"
        or cancelled.get("started_at") is not None
    ):
        raise ValueError("v1 scheduler terminal evidence drifted")

    methods = json.loads(methods_v1.read_text())
    if (
        methods.get("schema_version") != METHODS_SCHEMA
        or methods.get("physical_method_count") != 42
        or methods.get("method_budget_alias_count") != 60
        or methods.get("formal_held_out_outcomes_read") is not False
        or methods.get("count_correction", {}).get("sha256") != EXPECTED_COUNT_CORRECTION
        or not Path(methods["count_correction"]["path"]).is_absolute()
    ):
        raise ValueError("v1 methods portability premise drifted")

    queue_evidence = {
        task_id: {
            key: task_map[task_id].get(key)
            for key in (
                "id", "status", "signature", "require_node", "started_at", "finished_at",
                "exit_code", "exit_status_path", "exit_status_token", "last_block_reason",
            )
        }
        for task_id in ("t82565", "t82566")
    }
    correction = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "v1-invalid-before-sumo-v2-portability-correction",
        "scope": "deployment-path portability only",
        "reason": (
            "the v1 methods manifest retained an absolute build-host count-correction path; "
            "the corrected manifest uses a relative path resolved from its own directory"
        ),
        "formal_held_out_outcomes_read": False,
        "scientific_evaluation_outcomes_read": False,
        "training_or_policy_artifacts_changed": False,
        "evaluation_source_files_changed": False,
        "v1_remote_sumo_episode_count": 0,
        "v1_tasks": queue_evidence,
        "invalid_v1_policy": "never merge, analyze, select, reuse, or requeue",
        "failed_host_execution": descriptor(host_execution),
        "backend_exit": descriptor(exit_status),
        "scheduler_log": descriptor(scheduler_log),
        "v1_submission": descriptor(submission_v1),
        "v1_deployment_manifest": descriptor(deployment_v1),
        "v1_remote_verification": descriptor(verification_v1),
        "count_correction": descriptor(count_correction),
        "original_methods_manifest": descriptor(methods_v1),
        "corrected_methods_manifest": {
            "path": methods_output.name,
            "schema_version": METHODS_SCHEMA,
            "count_correction_sha256": EXPECTED_COUNT_CORRECTION,
        },
    }
    atomic_json(correction_output, correction)
    methods["count_correction"] = descriptor(count_correction, methods_output)
    methods["portability_correction"] = descriptor(correction_output, methods_output)
    atomic_json(methods_output, methods)
    payload = {
        "correction": descriptor(correction_output),
        "methods": descriptor(methods_output),
        "v1_remote_sumo_episode_count": 0,
        "physical_methods": methods.get("physical_method_count"),
        "aliases": methods.get("method_budget_alias_count"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
