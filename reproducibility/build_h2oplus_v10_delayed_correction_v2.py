#!/usr/bin/env python3
"""Seal the invalid v10 prefix-v1 lineage before corrected deployment v2."""

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
INCIDENT = REPRO / "h2oplus_v10_delayed_incidents/prefix_v1_actor_states"
OUTPUT = REPRO / "h2oplus_v10_delayed_correction_v2.json"
FAILED_TASKS = {"t82617", "t82618", "t82619", "t82620"}
CANCELLED_TASKS = {"t82621", "t82623", "t82624", "t82625", "t82626"}
TASK_IDS = FAILED_TASKS | CANCELLED_TASKS
EXPECTED_ERROR = "TypeError: _actor_states() takes 2 positional arguments but 3 were given"


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
        raise ValueError("v10 invalid-prefix scheduler evidence is incomplete")
    return selected


def archived_result_manifest(run_root: Path) -> dict[str, Any]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"missing archived invalid result: {run_root}")
    files = [path for path in sorted(run_root.rglob("*")) if path.is_file()]
    relative = {path.relative_to(run_root).as_posix() for path in files}
    forbidden = {"summary.json", "metrics.jsonl", "v10_handoff/manifest.json"}
    if relative & forbidden:
        raise ValueError(f"invalid prefix unexpectedly completed: {run_root}")
    expected = {
        "checkpoints/latest.pt",
        "checkpoints/runtime_epoch000000.pkl",
        "metadata.json",
        "phase_metrics.json",
        "resources.csv",
        "run.log",
        "v10_checkpoints/online_events_000000000.pt",
        "v10_checkpoints/online_events_000005000.pt",
        "v10_policies/online_events_000000000.pt",
        "v10_policies/online_events_000005000.pt",
        "v10_policies/policy_manifest.json",
    }
    if relative != expected:
        raise ValueError(f"unexpected invalid-prefix artifact set: {run_root}")
    entries = [descriptor(path, base=run_root) for path in files]
    return {
        "root": str(run_root.resolve()),
        "entries": entries,
        "entry_count": len(entries),
        "aggregate_sha256": stable_object_sha256({"entries": entries}),
        "summary_present": False,
        "metrics_history_present": False,
        "handoff_present": False,
        "eligible_as_scientific_evidence": False,
    }


def failed_task_evidence(task: dict[str, Any]) -> dict[str, Any]:
    task_id = task["id"]
    if task.get("status") != "failed":
        raise ValueError(f"{task_id} is not failed")
    log_path = INCIDENT / "scheduler" / f"{task_id}.log"
    backend_path = INCIDENT / "scheduler" / f"{task_id}.backend"
    log = log_path.read_text()
    if EXPECTED_ERROR not in log:
        raise ValueError(f"{task_id} lacks the registered actor-state failure")
    fields = backend_path.read_text().strip().split("\t")
    if (
        len(fields) != 3
        or fields[0] != "1"
        or fields[2] != task.get("exit_status_token")
    ):
        raise ValueError(f"{task_id} backend exit evidence is malformed")
    seed = int(str(task["description"]).split("seed ", 1)[1].split()[0])
    run_root = INCIDENT / f"h2oplus_v10_shared_prefix_seed{seed}"
    return {
        "task_id": task_id,
        "status": task["status"],
        "seed": seed,
        "signature": task.get("signature"),
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "submitted_at": task.get("submitted_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "backend_returncode": 1,
        "backend_exit_epoch": int(fields[1]),
        "backend_token": fields[2],
        "scheduler_log": descriptor(log_path),
        "backend_exit": descriptor(backend_path),
        "archived_partial_result": archived_result_manifest(run_root),
        "failure_before_first_online_actor_update": True,
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


def remote_process_audit() -> list[dict[str, Any]]:
    skill = SCHEDULER_ROOT / "skill"
    sys.path.insert(0, str(skill))
    import scheduler  # type: ignore  # noqa: PLC0415

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
    tasks = queue_tasks()
    failed = [failed_task_evidence(tasks[task_id]) for task_id in sorted(FAILED_TASKS)]
    cancelled = [
        cancelled_task_evidence(tasks[task_id])
        for task_id in sorted(CANCELLED_TASKS)
    ]
    queue_snapshot = INCIDENT / "queue_snapshot.json"
    snapshot = {
        "schema_version": "transit-h2oplus-v10-prefix-v1-queue-snapshot-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failed_tasks": failed,
        "cancelled_tasks": cancelled,
    }
    queue_snapshot.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    v1_workspace = REPRO / "deployments/h2oplus_v10_delayed_pilot_v1/workspace"
    payload = {
        "schema_version": "transit-h2oplus-v10-delayed-correction-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prefix-v1-invalid-sealed-v2-correction-authorized",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "invalid_v1_lineage": {
            "failed_task_ids": sorted(FAILED_TASKS),
            "cancelled_before_launch_task_ids": sorted(CANCELLED_TASKS),
            "queue_snapshot": descriptor(queue_snapshot),
            "prefix_submission": descriptor(
                REPRO / "h2oplus_v10_delayed_prefix_submission_v1.json"
            ),
            "deployment_manifest": descriptor(
                v1_workspace / "deployment_manifest_v10.json"
            ),
            "partial_results_preserved": True,
            "scientific_training_results_produced": False,
            "eligible_as_scientific_evidence": False,
            "requeue_allowed": False,
        },
        "defect": {
            "class": "v10 mixed-replay actor-state dispatch signature",
            "message": EXPECTED_ERROR,
            "cause": (
                "TransitH2OTrainer._actor_states is a static method, but the v10 "
                "fallback passed self as a third positional argument"
            ),
            "trigger": "first online actor update after the restored 5k warmup",
        },
        "correction": {
            "scope": "v10 overlay only; frozen v7/v8/v9 source and evidence unchanged",
            "old_call": "TransitH2OTrainer._actor_states(self, offline_batch, online_batch)",
            "new_call": "TransitH2OTrainer._actor_states(offline_batch, online_batch)",
            "invalid_source_sha256": source_map(v1_workspace),
            "corrected_source_sha256": source_map(ROOT),
            "focused_test_command": (
                "TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 pytest -q "
                "reproducibility/test_h2oplus_v10_delayed.py"
            ),
            "focused_test_result": "12 passed",
            "new_deployment_plan_signatures_result_roots_and_task_ids_required": True,
        },
        "remote_process_audit": remote_process_audit(),
        "generator": descriptor(Path(__file__).resolve()),
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
