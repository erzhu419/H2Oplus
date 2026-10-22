#!/usr/bin/env python3
"""Synchronize and seal exact route-aware evidence for v10-v4 branch runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sync_h2oplus_v10_delayed_prefix_v4 import (
    REMOTE_ROOT,
    REPRO,
    ROOT,
    atomic_json,
    descriptor,
    load_scheduler,
    local_inventory,
    queue_tasks,
    remote_inventory,
)


SUBMISSION_SCHEMA = "transit-h2oplus-v10-delayed-branch-submission-v4"
SYNC_SCHEMA = "transit-h2oplus-v10-delayed-branch-sync-v4"
SCHEDULER_SCHEMA = "transit-h2oplus-v10-delayed-branch-scheduler-evidence-v4"


def sync_run(scheduler, node: str, remote: Path, local: Path) -> None:
    """Stream into a fresh staging root, then atomically install the mirror."""
    local.parent.mkdir(parents=True, exist_ok=True)
    target = scheduler._ssh_target_for_node(node)
    shell = scheduler._ssh_rsync_shell_for_node(node)
    remote_command = (
        f"tar -C {shlex.quote(str(remote.parent))} -czf - {shlex.quote(remote.name)}"
    )
    failures = []
    for attempt in range(1, 4):
        staging_parent = Path(
            tempfile.mkdtemp(prefix=f".{local.name}.sync-", dir=local.parent)
        )
        try:
            command = (
                f"{shell} {shlex.quote(target)} {shlex.quote(remote_command)} | "
                f"tar -xzf - -C {shlex.quote(str(staging_parent))}"
            )
            completed = subprocess.run(
                ["bash", "-o", "pipefail", "-c", command],
                text=True,
                capture_output=True,
                check=False,
            )
            staged_run = staging_parent / remote.name
            if completed.returncode == 0 and staged_run.is_dir():
                if local.exists():
                    shutil.rmtree(local)
                os.replace(staged_run, local)
                return
            failures.append(
                {
                    "attempt": attempt,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip()[-1000:],
                }
            )
            if attempt < 3:
                time.sleep(5 * attempt)
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)
    raise RuntimeError(
        f"route-aware compressed sync failed for {node}:{remote}: "
        f"{json.dumps(failures, sort_keys=True)}"
    )


def remote_inventory_with_retry(scheduler, node: str, root: Path) -> list[dict[str, object]]:
    failures = []
    for attempt in range(1, 4):
        try:
            return remote_inventory(scheduler, node, root)
        except (RuntimeError, OSError, json.JSONDecodeError) as error:
            failures.append({"attempt": attempt, "error": str(error)[-1000:]})
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(
        f"remote inventory retries exhausted for {node}:{root}: "
        f"{json.dumps(failures, sort_keys=True)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root",
        type=Path,
        default=ROOT.parent.parent / "scheduleurm",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path.home() / ".claude/scheduler/queue.json",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_branch_submission_v4.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/branches_v4",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/evidence_branches_v4",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    submission_path = args.submission.expanduser().resolve(strict=True)
    submission = json.loads(submission_path.read_text())
    expected = submission.get("tasks", [])
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("status") != "submitted"
        or submission.get("submitted_stage") != "branch-training"
        or submission.get("prefix_training_rerun") is not False
        or submission.get("development_evaluation_submitted") is not False
        or submission.get("formal_held_out_outcomes_read") is not False
        or len(expected) != 25
    ):
        raise ValueError("v10-v4 branch submission is stale or incomplete")

    by_id = {task["id"]: task for task in queue_tasks(args.queue)}
    task_ids = {item["task_id"] for item in expected}
    if not task_ids <= set(by_id):
        raise ValueError("hot queue does not contain every v10-v4 branch task")
    rows = [by_id[task_id] for task_id in sorted(task_ids)]
    if any(task.get("status") != "done" or task.get("exit_code") != 0 for task in rows):
        raise RuntimeError("v10-v4 branch scheduler tasks are not all done with exit0")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    def sync_item(item: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        task = by_id[item["task_id"]]
        node = item["require_node"]
        if (
            task.get("node") != node
            or task.get("signature") != item["scheduler_signature"]
            or task.get("result_dir") != str(REMOTE_ROOT / item["name"])
        ):
            raise ValueError(f"scheduler binding drifted for {item['task_id']}")

        exit_path = task["exit_status_path"]
        log_path = task["log_path"]
        command = (
            f"cat {shlex.quote(exit_path)}; printf '\\n__LOG__\\n'; "
            f"cat {shlex.quote(log_path)}; printf '\\n__PROC__\\n'; "
            "ps -eo pid=,args= | grep -F 'SimpleSAC/train_transit_v10_delayed.py' "
            "| grep -v grep || true"
        )
        probe_failures = []
        for attempt in range(1, 4):
            rc, out, err = scheduler.run_on(node, command, timeout=60, check=False)
            if rc == 0 and "\n__LOG__\n" in out and "\n__PROC__\n" in out:
                break
            probe_failures.append(
                {"attempt": attempt, "returncode": rc, "error": (err or out)[-1000:]}
            )
            if attempt < 3:
                time.sleep(5 * attempt)
        if rc != 0 or "\n__LOG__\n" not in out or "\n__PROC__\n" not in out:
            raise RuntimeError(
                f"remote completion evidence failed for {item['task_id']}: "
                f"{json.dumps(probe_failures, sort_keys=True)}"
            )
        exit_text, remainder = out.split("\n__LOG__\n", 1)
        log_text, process_text = remainder.split("\n__PROC__\n", 1)
        fields = exit_text.strip().split("\t")
        if (
            len(fields) != 3
            or int(fields[0]) != 0
            or fields[2] != task.get("exit_status_token")
            or process_text.strip()
        ):
            raise ValueError(f"token-bound backend closure failed for {item['task_id']}")
        backend_local = evidence_root / f"{item['task_id']}.backend"
        log_local = evidence_root / f"sched_{item['task_id']}.log"
        backend_local.write_text(exit_text.strip() + "\n")
        log_local.write_text(log_text.rstrip() + "\n")

        remote_run = REMOTE_ROOT / item["name"]
        local_run = output_root / item["name"]
        remote_records = remote_inventory_with_retry(scheduler, node, remote_run)
        local_records = local_inventory(local_run) if local_run.is_dir() else []
        if remote_records != local_records:
            sync_run(scheduler, node, remote_run, local_run)
            local_records = local_inventory(local_run)
        if remote_records != local_records:
            raise ValueError(f"content inventory mismatch for {item['name']}")
        sync_row = {
            "task_id": item["task_id"],
            "run_name": item["name"],
            "variant": item["variant"],
            "seed": item["seed"],
            "source_node": node,
            "source_root": str(remote_run),
            "local_root": str(local_run),
            "file_count": len(local_records),
            "total_bytes": sum(record["bytes"] for record in local_records),
            "files": local_records,
        }
        scheduler_row = {
            "task_id": item["task_id"],
            "run_name": item["name"],
            "variant": item["variant"],
            "seed": item["seed"],
            "node": node,
            "status": task["status"],
            "exit_code": task["exit_code"],
            "finished_at": task.get("finished_at"),
            "signature": task["signature"],
            "backend_returncode": int(fields[0]),
            "backend_finished_epoch": int(fields[1]),
            "backend_token": fields[2],
            "token_matches": True,
            "residual_canonical_process_count": 0,
            "backend_exit": descriptor(backend_local),
            "scheduler_log": descriptor(log_local),
        }
        return sync_row, scheduler_row

    with ThreadPoolExecutor(max_workers=4) as executor:
        completed = list(executor.map(sync_item, expected))
    sync_rows = [row[0] for row in completed]
    scheduler_rows = [row[1] for row in completed]

    captured = datetime.now(timezone.utc).isoformat()
    queue_path = evidence_root / "queue_snapshot_branches_v4.json"
    atomic_json(
        queue_path,
        {
            "schema_version": "transit-h2oplus-v10-delayed-branch-queue-snapshot-v4",
            "captured_utc": captured,
            "tasks": sorted(rows, key=lambda row: row["id"]),
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_branches_v4.json"
    atomic_json(
        scheduler_path,
        {
            "schema_version": SCHEDULER_SCHEMA,
            "captured_utc": captured,
            "submission": descriptor(submission_path),
            "queue_snapshot": descriptor(queue_path),
            "tasks": sorted(scheduler_rows, key=lambda row: row["task_id"]),
        },
    )
    sync_path = evidence_root / "source_sync_branches_v4.json"
    atomic_json(
        sync_path,
        {
            "schema_version": SYNC_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "transport": "scheduleurm-route-aware-compressed-tar-stream",
            "checksum_mode": "sha256-per-file-before-and-after-transfer",
            "formal_held_out_outcomes_read": False,
            "runs": sorted(sync_rows, key=lambda row: row["run_name"]),
        },
    )
    print(
        json.dumps(
            {
                "runs": len(sync_rows),
                "files": sum(row["file_count"] for row in sync_rows),
                "bytes": sum(row["total_bytes"] for row in sync_rows),
                "scheduler_evidence": descriptor(scheduler_path),
                "sync_evidence": descriptor(sync_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
