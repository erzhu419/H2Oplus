#!/usr/bin/env python3
"""Seal the invalid v9 v1 launch lineage before building correction v2."""

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
OUTPUT = REPRO / "h2oplus_v9_adaptive_correction_v2.json"
TASK_IDS = tuple(f"t{value}" for value in range(82479, 82488))
FAILED_TASKS = {"t82479", "t82480", "t82481"}
LAUNCHED_CANCELLED_TASKS = {"t82482", "t82483", "t82484", "t82486"}
NEVER_LAUNCHED_TASKS = {"t82485", "t82487"}
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


def _queue_tasks() -> dict[str, dict[str, Any]]:
    payload = json.loads(QUEUE.read_text())
    rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    selected = {row["id"]: row for row in rows if row.get("id") in TASK_IDS}
    if set(selected) != set(TASK_IDS):
        raise ValueError("v1 scheduler evidence is incomplete")
    return selected


def _source_map(root: Path) -> dict[str, str]:
    paths = (
        root / "SimpleSAC/train_transit_v9_adaptive.py",
        root / "SimpleSAC/transit_v9/__init__.py",
        root / "SimpleSAC/transit_v9/trainer.py",
    )
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths}


def _remote_process_evidence(host: str) -> dict[str, Any]:
    command = (
        "pgrep -af '[t]rain_transit_v9_adaptive.py' || true; "
        "nvidia-smi --query-compute-apps=pid,used_memory "
        "--format=csv,noheader || true"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"remote v9 process audit failed on {host}: {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if any("train_transit_v9_adaptive.py" in line for line in lines):
        raise RuntimeError(f"residual v9 process remains on {host}")
    return {
        "host": host,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": result.returncode,
        "stdout_lines": lines,
        "no_v9_training_process": True,
    }


def _task_evidence(task: dict[str, Any]) -> dict[str, Any]:
    task_id = task["id"]
    expected_status = "failed" if task_id in FAILED_TASKS else "cancelled"
    if task.get("status") != expected_status:
        raise ValueError(
            f"{task_id} status is {task.get('status')}, expected {expected_status}"
        )
    launched = task_id not in NEVER_LAUNCHED_TASKS
    evidence = {
        "task_id": task_id,
        "status": task["status"],
        "signature": task.get("signature"),
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "submitted_at": task.get("submitted_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "remote_pids": task.get("remote_pids", []),
        "last_block_reason": task.get("last_block_reason"),
        "launched": launched,
    }
    if not launched:
        if task.get("started_at") is not None or task.get("remote_pids"):
            raise ValueError(f"{task_id} was expected never to launch")
        return evidence

    node = str(task["node"])
    log_name = Path(str(task["log_path"])).name
    exit_name = Path(str(task["exit_status_path"])).name
    log_path = INCIDENT / node / log_name
    exit_path = INCIDENT / node / exit_name
    log = log_path.read_text()
    if EXPECTED_ERROR not in log:
        raise ValueError(f"{task_id} does not contain the registered v1 error")
    fields = exit_path.read_text().strip().split("\t")
    if len(fields) != 3 or fields[0] != "1":
        raise ValueError(f"{task_id} backend exit evidence is malformed")
    if fields[2] != task.get("exit_status_token"):
        raise ValueError(f"{task_id} backend token does not match queue state")
    evidence.update(
        {
            "scheduler_log": descriptor(log_path),
            "backend_exit": descriptor(exit_path),
            "backend_returncode": 1,
            "backend_exit_epoch": int(fields[1]),
            "backend_token": fields[2],
            "failure_before_training_setup": True,
        }
    )
    return evidence


def main() -> int:
    tasks = _queue_tasks()
    evidence = [_task_evidence(tasks[task_id]) for task_id in TASK_IDS]
    v1_root = REPRO / "deployments/h2oplus_v9_adaptive_pilot_v1"
    result_root = v1_root / "remote_results"
    scientific_files = []
    if result_root.exists():
        for pattern in ("metrics.jsonl", "summary.json", "v9_export_summary.json"):
            scientific_files.extend(result_root.rglob(pattern))
    if scientific_files:
        raise RuntimeError("v1 unexpectedly produced scientific training artifacts")

    queue_snapshot = INCIDENT / "queue_snapshot.json"
    queue_snapshot.write_text(
        json.dumps(
            {
                "schema_version": "transit-h2oplus-v9-adaptive-v1-queue-snapshot-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "tasks": evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    payload = {
        "schema_version": "transit-h2oplus-v9-adaptive-correction-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "v1-invalid-sealed-v2-correction-authorized",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "invalid_v1_lineage": {
            "task_ids": list(TASK_IDS),
            "failed_task_ids": sorted(FAILED_TASKS),
            "cancelled_after_failed-preflight_task_ids": sorted(
                LAUNCHED_CANCELLED_TASKS
            ),
            "cancelled_before-launch_task_ids": sorted(NEVER_LAUNCHED_TASKS),
            "queue_snapshot": descriptor(queue_snapshot),
            "submission_record": descriptor(
                REPRO / "h2oplus_v9_adaptive_submission_record_v1.json"
            ),
            "deployment_manifest": descriptor(
                v1_root / "workspace/deployment_manifest_v9.json"
            ),
            "scientific_training_artifacts_produced": False,
            "eligible_as_scientific_evidence": False,
            "requeue_allowed": False,
        },
        "defect": {
            "class": "pre-training offline-initializer compatibility guard",
            "message": EXPECTED_ERROR,
            "cause": (
                "the v9 wrapper installed a stricter registered initializer validator, "
                "but the frozen v7 main preflight applied its separate legacy branch "
                "allowlist before calling that validator"
            ),
        },
        "correction": {
            "scope": "v9 wrapper process only; frozen v7 files remain unchanged",
            "behavior": (
                "authorize method h2oplus only after the v9 validator is installed and "
                "only when the legacy v7 branch registry exactly matches its expected set"
            ),
            "corrected_source_sha256": _source_map(ROOT),
            "focused_test_command": (
                "TMPDIR=/tmp/h2oplus-v9-pytest PYTHONDONTWRITEBYTECODE=1 "
                "python3 -m pytest -q -s tests/test_h2oplus_v9_adaptation.py "
                "tests/test_v7_networks.py tests/test_policy_export_v7.py"
            ),
            "focused_test_result": "22 passed",
            "new_deployment_and_task_ids_required": True,
        },
        "remote_process_audit": [
            _remote_process_evidence("jtl110gpu"),
            _remote_process_evidence("jtl110gpu2"),
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
