#!/usr/bin/env python3
"""Seal the invalid v10 prefix-v3 admission-schema lineage before deployment v4."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEDULER_ROOT = ROOT.parent.parent / "scheduleurm"
QUEUE = Path.home() / ".claude/scheduler/queue.json"
INCIDENT = REPRO / "h2oplus_v10_delayed_incidents/prefix_v3_admission_schema"
OUTPUT = REPRO / "h2oplus_v10_delayed_correction_v4.json"
FAILED_TASKS = {"t82647", "t82648", "t82649"}
CANCELLED_TASKS = {"t82650", "t82651", "t82653", "t82654", "t82655"}
TASK_IDS = FAILED_TASKS | CANCELLED_TASKS
EXPECTED_ERROR = "ValueError: v10 development admission is stale or misbound"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def descriptor(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            resolved.relative_to(base.resolve()).as_posix()
            if base is not None
            else str(resolved)
        ),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def source_map(root: Path) -> dict[str, str]:
    paths = (
        root / "SimpleSAC/train_transit_v10_delayed.py",
        root / "SimpleSAC/transit_v10/__init__.py",
        root / "SimpleSAC/transit_v10/provenance.py",
        root / "SimpleSAC/transit_v10/trainer.py",
    )
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths}


def queue_tasks() -> dict[str, dict[str, Any]]:
    payload = json.loads(QUEUE.read_text())
    rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    selected = {row["id"]: row for row in rows if row.get("id") in TASK_IDS}
    if set(selected) != TASK_IDS:
        raise ValueError("v10 invalid-prefix-v3 scheduler evidence is incomplete")
    return selected


def scheduler_module():
    sys.path.insert(0, str(SCHEDULER_ROOT / "skill"))
    import scheduler  # type: ignore  # noqa: PLC0415

    return scheduler


def archive_failed_task(
    task: dict[str, Any], scheduler: Any
) -> dict[str, Any]:
    task_id = task["id"]
    if task.get("status") != "failed" or task.get("exit_code") != 1:
        raise ValueError(f"{task_id} is not a terminal exit-1 failure")
    node = str(task["node"])
    log_path = str(task["log_path"])
    backend_path = str(task["exit_status_path"])
    rc_log, log, log_err = scheduler.run_on(
        node, f"cat {log_path}", timeout=20, check=False
    )
    rc_backend, backend, backend_err = scheduler.run_on(
        node, f"cat {backend_path}", timeout=20, check=False
    )
    if rc_log or log_err or EXPECTED_ERROR not in log:
        raise ValueError(f"{task_id} lacks the registered protocol failure")
    fields = backend.strip().split("\t")
    if (
        rc_backend
        or backend_err
        or len(fields) != 3
        or fields[0] != "1"
        or fields[2] != task.get("exit_status_token")
    ):
        raise ValueError(f"{task_id} backend exit evidence is malformed")
    result_dir = Path(task["result_dir"])
    local_result = INCIDENT / "results" / result_dir.name
    if local_result.exists():
        raise FileExistsError(f"refusing to overwrite incident result: {local_result}")
    ok, sync_message = scheduler._sync_one_result(
        {
            "node": node,
            "result_dir": str(result_dir),
            "local_result_dir": str(local_result),
        }
    )
    if not ok:
        raise RuntimeError(f"{task_id} incident result sync failed: {sync_message}")
    files = [path for path in sorted(local_result.rglob("*")) if path.is_file()]
    relative = {path.relative_to(local_result).as_posix() for path in files}
    if relative != {"resources.csv", "run.log"}:
        raise ValueError(f"{task_id} has unexpected startup artifacts: {relative}")
    partial_entries = [descriptor(path, base=local_result) for path in files]

    scheduler_dir = INCIDENT / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    local_log = scheduler_dir / f"{task_id}.log"
    local_backend = scheduler_dir / f"{task_id}.backend"
    local_log.write_text(log)
    local_backend.write_text(backend)
    seed = int(str(task["description"]).split("seed ", 1)[1].split()[0])
    return {
        "task_id": task_id,
        "status": task["status"],
        "seed": seed,
        "signature": task.get("signature"),
        "node": node,
        "gpu_idx": task.get("gpu_idx"),
        "submitted_at": task.get("submitted_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "backend_returncode": 1,
        "backend_exit_epoch": int(fields[1]),
        "backend_token": fields[2],
        "scheduler_log": descriptor(local_log),
        "backend_exit": descriptor(local_backend),
        "result_directory_created": True,
        "startup_partial_result": {
            "root": str(local_result.resolve()),
            "entries": partial_entries,
            "entry_count": len(partial_entries),
            "aggregate_sha256": stable_object_sha256({"entries": partial_entries}),
            "result_sync": sync_message,
        },
        "learner_update_or_sumo_episode_started": False,
        "eligible_as_scientific_evidence": False,
        "requeue_allowed": False,
    }


def cancelled_task_evidence(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("status") != "cancelled":
        raise ValueError(f"{task['id']} is not cancelled")
    if task.get("started_at") is not None or task.get("remote_pids"):
        raise ValueError(f"{task['id']} unexpectedly launched")
    return {
        "task_id": task["id"],
        "status": task["status"],
        "parent_id": task.get("parent_id"),
        "retry_count": task.get("retry_count"),
        "signature": task.get("signature"),
        "submitted_at": task.get("submitted_at"),
        "finished_at": task.get("finished_at"),
        "launched": False,
        "remote_training_or_sumo_executed": False,
        "requeue_allowed": False,
    }


def remote_process_audit(scheduler: Any) -> list[dict[str, Any]]:
    evidence = []
    command = "pgrep -af '[t]rain_transit_v10_delayed.py' || true"
    for node in ("jtl110gpu", "jtl110gpu2"):
        rc, stdout, stderr = scheduler.run_on(
            node, command, timeout=20, check=False
        )
        if rc or stdout.strip():
            raise RuntimeError(f"residual v10 process on {node}: {stdout} {stderr}")
        evidence.append(
            {
                "node": node,
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "returncode": rc,
                "stdout": stdout,
                "stderr": stderr,
                "no_v10_training_process": True,
            }
        )
    return evidence


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace correction: {OUTPUT}")
    INCIDENT.mkdir(parents=True, exist_ok=True)
    tasks = queue_tasks()
    scheduler = scheduler_module()
    failed = [
        archive_failed_task(tasks[task_id], scheduler)
        for task_id in sorted(FAILED_TASKS)
    ]
    cancelled = [
        cancelled_task_evidence(tasks[task_id])
        for task_id in sorted(CANCELLED_TASKS)
    ]
    queue_snapshot = INCIDENT / "queue_snapshot.json"
    snapshot = {
        "schema_version": "transit-h2oplus-v10-prefix-v3-queue-snapshot-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failed_tasks": failed,
        "cancelled_tasks": cancelled,
    }
    queue_snapshot.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    v3_workspace = REPRO / "deployments/h2oplus_v10_delayed_pilot_v3/workspace"
    payload = {
        "schema_version": "transit-h2oplus-v10-delayed-correction-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prefix-v3-invalid-sealed-v4-correction-authorized",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "invalid_v3_lineage": {
            "failed_task_ids": sorted(FAILED_TASKS),
            "cancelled_before_launch_task_ids": sorted(CANCELLED_TASKS),
            "queue_snapshot": descriptor(queue_snapshot),
            "prefix_submission": descriptor(
                REPRO / "h2oplus_v10_delayed_prefix_submission_v3.json"
            ),
            "deployment_manifest": descriptor(
                v3_workspace / "deployment_manifest_v10.json"
            ),
            "preflight_closeout": descriptor(
                REPRO / "h2oplus_v10_delayed_preflight_closeout_v3.json"
            ),
            "scientific_training_results_produced": False,
            "eligible_as_scientific_evidence": False,
            "requeue_allowed": False,
        },
        "prior_correction": descriptor(
            REPRO / "h2oplus_v10_delayed_correction_v3.json"
        ),
        "defect": {
            "class": "v10 launcher/admission schema mismatch",
            "message": EXPECTED_ERROR,
            "cause": (
                "the corrected v3 deployment froze admission schema v3 while the "
                "launcher admission constant remained at schema v1"
            ),
            "trigger": "fail-closed runtime admission check before learner or SUMO startup",
        },
        "correction": {
            "scope": "v10 overlay only; frozen v7/v8/v9 source and evidence unchanged",
            "old_launcher_admission_schema": "transit-training-admission-v10-delayed-development-v1",
            "new_launcher_admission_schema": "transit-training-admission-v10-delayed-development-v4",
            "new_launcher_protocol_schema": "transit-h2oplus-v10-delayed-protocol-v4",
            "invalid_source_sha256": source_map(v3_workspace),
            "corrected_source_sha256": source_map(ROOT),
            "focused_test_command": (
                "TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 pytest -q "
                "reproducibility/test_h2oplus_v10_delayed.py"
            ),
            "focused_test_result": "13 passed",
            "new_deployment_plan_signatures_result_roots_and_task_ids_required": True,
        },
        "remote_process_audit": remote_process_audit(scheduler),
        "generator": descriptor(Path(__file__).resolve()),
    }
    payload["evidence_aggregate_sha256"] = stable_object_sha256(
        {
            "failed": failed,
            "cancelled": cancelled,
            "remote_process_audit": payload["remote_process_audit"],
        }
    )
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
