#!/usr/bin/env python3
"""Seal scheduler-created retries of the invalid v9 v1 launch lineage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
QUEUE = Path.home() / ".claude/scheduler/queue.json"
INCIDENT = REPRO / "h2oplus_v9_adaptive_incidents/v1_initializer_bridge"
ARCHIVE = INCIDENT / "automatic_retries"
OUTPUT = REPRO / "h2oplus_v9_adaptive_v1_retry_amendment.json"
TASK_IDS = (
    "t82489",
    "t82490",
    "t82491",
    "t82493",
    "t82494",
    "t82495",
    "t82496",
    "t82497",
    "t82498",
)
FAILED_TASKS = set(TASK_IDS[:7])
CANCELLED_AFTER_BACKEND_EXIT_TASKS = set(TASK_IDS[7:])
EXPECTED_ERROR = (
    "offline initializer is restricted to registered H2O+ ablation and "
    "sensitivity branches"
)


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


def queue_tasks() -> dict[str, dict[str, Any]]:
    payload = json.loads(QUEUE.read_text())
    rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    selected = {row["id"]: row for row in rows if row.get("id") in TASK_IDS}
    if set(selected) != set(TASK_IDS):
        raise ValueError("v1 automatic-retry scheduler evidence is incomplete")
    return selected


def task_evidence(task: dict[str, Any]) -> dict[str, Any]:
    task_id = task["id"]
    expected_status = "failed" if task_id in FAILED_TASKS else "cancelled"
    if task.get("status") != expected_status:
        raise ValueError(
            f"{task_id} status is {task.get('status')}, expected {expected_status}"
        )
    if task.get("project") != "h2oplus-v9-development":
        raise ValueError(f"{task_id} is not in the invalid v1 project")
    result_dir = str(task.get("result_dir", ""))
    if "/h2oplus_v9_adaptive_pilot_v1/" not in result_dir:
        raise ValueError(f"{task_id} does not target the invalid v1 deployment")
    node = str(task["node"])
    log_path = ARCHIVE / node / Path(str(task["log_path"])).name
    exit_path = ARCHIVE / node / Path(str(task["exit_status_path"])).name
    log = log_path.read_text()
    if EXPECTED_ERROR not in log:
        raise ValueError(f"{task_id} does not contain the registered v1 error")
    fields = exit_path.read_text().strip().split("\t")
    if len(fields) != 3 or fields[0] != "1":
        raise ValueError(f"{task_id} backend exit evidence is malformed")
    if fields[2] != task.get("exit_status_token"):
        raise ValueError(f"{task_id} backend token does not match queue state")
    return {
        "task_id": task_id,
        "status": task["status"],
        "scheduler_retry_count": task.get("retry_count"),
        "signature": task.get("signature"),
        "node": node,
        "gpu_idx": task.get("gpu_idx"),
        "submitted_at": task.get("submitted_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "remote_pids": task.get("remote_pids", []),
        "cancel_reason": task.get("cancel_reason"),
        "last_block_reason": task.get("last_block_reason"),
        "result_dir": result_dir,
        "scheduler_log": descriptor(log_path),
        "backend_exit": descriptor(exit_path),
        "backend_returncode": 1,
        "backend_exit_epoch": int(fields[1]),
        "backend_token": fields[2],
        "failure_before_training_setup": True,
        "eligible_as_scientific_evidence": False,
        "requeue_allowed": False,
    }


def remote_v1_process_evidence(host: str) -> dict[str, Any]:
    command = "pgrep -af '[t]rain_transit_v9_adaptive.py' || true"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"remote v1 process audit failed on {host}: {result.stderr}")
    v1_lines = [
        line
        for line in result.stdout.splitlines()
        if "h2oplus_v9_adaptive_pilot_v1" in line
    ]
    if v1_lines:
        raise RuntimeError(f"residual v1 process remains on {host}: {result.stdout}")
    return {
        "host": host,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": result.returncode,
        "active_v9_process_lines": [
            line for line in result.stdout.splitlines() if line.strip()
        ],
        "no_v1_training_process": True,
    }


def main() -> int:
    tasks = queue_tasks()
    evidence = [task_evidence(tasks[task_id]) for task_id in TASK_IDS]
    retry_counts = [row["scheduler_retry_count"] for row in evidence]
    if retry_counts != [1, 1, 1, 2, 2, 2, 3, 3, 3]:
        raise ValueError(f"unexpected v1 retry-count sequence: {retry_counts}")

    v1_root = REPRO / "deployments/h2oplus_v9_adaptive_pilot_v1"
    result_root = v1_root / "remote_results"
    scientific_files: list[Path] = []
    if result_root.exists():
        for pattern in ("metrics.jsonl", "summary.json", "v9_export_summary.json"):
            scientific_files.extend(result_root.rglob(pattern))
    if scientific_files:
        raise RuntimeError("v1 automatic retries produced scientific training artifacts")

    queue_snapshot = ARCHIVE / "queue_snapshot.json"
    queue_snapshot.write_text(
        json.dumps(
            {
                "schema_version": (
                    "transit-h2oplus-v9-adaptive-v1-auto-retry-queue-snapshot-v1"
                ),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "task_ids": list(TASK_IDS),
                "excluded_unrelated_task_ids": ["t82492"],
                "tasks": evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    payload = {
        "schema_version": "transit-h2oplus-v9-adaptive-v1-retry-amendment-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "v1-automatic-retry-lineage-invalid-sealed",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "amends_without_mutation": descriptor(
            REPRO / "h2oplus_v9_adaptive_correction_v2.json"
        ),
        "corrected_v2_deployment_manifest_unchanged": descriptor(
            REPRO
            / "deployments/h2oplus_v9_adaptive_pilot_v2/workspace/deployment_manifest_v9.json"
        ),
        "invalid_automatic_retry_lineage": {
            "task_ids": list(TASK_IDS),
            "failed_task_ids": sorted(FAILED_TASKS),
            "cancelled_after_backend_exit_task_ids": sorted(
                CANCELLED_AFTER_BACKEND_EXIT_TASKS
            ),
            "excluded_unrelated_task_ids": ["t82492"],
            "queue_snapshot": descriptor(queue_snapshot),
            "tasks": evidence,
            "scientific_training_artifacts_produced": False,
            "eligible_as_scientific_evidence": False,
            "requeue_allowed": False,
        },
        "defect_message": EXPECTED_ERROR,
        "remote_process_audit": [
            remote_v1_process_evidence("jtl110gpu"),
            remote_v1_process_evidence("jtl110gpu2"),
        ],
        "generator": descriptor(Path(__file__).resolve()),
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
