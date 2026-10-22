#!/usr/bin/env python3
"""Synchronize and seal exact route-aware evidence for v10-v4 prefix runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
REMOTE_ROOT = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v10_delayed_pilot_v4/remote_results"
)
SYNC_SCHEMA = "transit-h2oplus-v10-delayed-prefix-sync-v4"
SCHEDULER_SCHEMA = "transit-h2oplus-v10-delayed-prefix-scheduler-evidence-v4"


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
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_scheduler(scheduler_root: Path):
    sys.path.insert(0, str(scheduler_root / "skill"))
    return importlib.import_module("scheduler")


def queue_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve(strict=True).read_text())
    tasks = payload.get("tasks", [])
    return list(tasks.values()) if isinstance(tasks, dict) else list(tasks)


def local_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def remote_inventory(scheduler, node: str, root: Path) -> list[dict[str, Any]]:
    code = (
        "import hashlib,json,sys; from pathlib import Path; root=Path(sys.argv[1]); "
        "files=sorted(p for p in root.rglob('*') if p.is_file()); out=[]; "
        "\nfor p in files:\n"
        " h=hashlib.sha256()\n"
        " with p.open('rb') as f:\n"
        "  for c in iter(lambda:f.read(8388608),b''): h.update(c)\n"
        " out.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})\n"
        "print(json.dumps(out,sort_keys=True))"
    )
    command = f"python3 -c {shlex.quote(code)} {shlex.quote(str(root))}"
    rc, out, err = scheduler.run_on(node, command, timeout=300, check=False)
    if rc != 0:
        raise RuntimeError(f"remote inventory failed on {node}: {err or out}")
    return json.loads(out)


def sync_run(scheduler, node: str, remote: Path, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    target = scheduler._ssh_target_for_node(node)
    shell = scheduler._ssh_rsync_shell_for_node(node)
    command = [
        "rsync",
        "-a",
        "--omit-dir-times",
        "--checksum",
        "-e",
        shell,
        f"{target}:{remote}/",
        f"{local}/",
    ]
    failures = []
    for attempt in range(1, 4):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
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
    raise RuntimeError(
        f"route-aware rsync failed for {node}:{remote}: "
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
        default=REPRO / "h2oplus_v10_delayed_prefix_submission_v4.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/prefix_v4",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/evidence_prefix_v4",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = load_scheduler(scheduler_root)
    submission_path = args.submission.expanduser().resolve(strict=True)
    submission = json.loads(submission_path.read_text())
    expected = submission.get("tasks", [])
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v10-delayed-prefix-submission-v4"
        or submission.get("status") != "submitted"
        or submission.get("branch_training_submitted") is not False
        or submission.get("formal_held_out_outcomes_read") is not False
        or len(expected) != 5
    ):
        raise ValueError("v10-v4 prefix submission is stale or incomplete")

    by_id = {task["id"]: task for task in queue_tasks(args.queue)}
    task_ids = {item["task_id"] for item in expected}
    if not task_ids <= set(by_id):
        raise ValueError("hot queue does not contain every v10-v4 prefix task")
    rows = [by_id[task_id] for task_id in sorted(task_ids)]
    if any(task.get("status") != "done" or task.get("exit_code") != 0 for task in rows):
        raise RuntimeError("v10-v4 prefix scheduler tasks are not all done with exit0")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    scheduler_rows = []
    sync_rows = []
    for item in expected:
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
        rc, out, err = scheduler.run_on(node, command, timeout=60, check=False)
        if rc != 0 or "\n__LOG__\n" not in out or "\n__PROC__\n" not in out:
            raise RuntimeError(f"remote completion evidence failed for {item['task_id']}: {err}")
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
        sync_run(scheduler, node, remote_run, local_run)
        remote_records = remote_inventory(scheduler, node, remote_run)
        local_records = local_inventory(local_run)
        if remote_records != local_records:
            raise ValueError(f"content inventory mismatch for {item['name']}")
        sync_rows.append(
            {
                "task_id": item["task_id"],
                "run_name": item["name"],
                "source_node": node,
                "source_root": str(remote_run),
                "local_root": str(local_run),
                "file_count": len(local_records),
                "total_bytes": sum(record["bytes"] for record in local_records),
                "files": local_records,
            }
        )
        scheduler_rows.append(
            {
                "task_id": item["task_id"],
                "run_name": item["name"],
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
        )

    captured = datetime.now(timezone.utc).isoformat()
    queue_path = evidence_root / "queue_snapshot_prefix_v4.json"
    atomic_json(
        queue_path,
        {
            "schema_version": "transit-h2oplus-v10-delayed-prefix-queue-snapshot-v4",
            "captured_utc": captured,
            "tasks": sorted(rows, key=lambda row: row["id"]),
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_prefix_v4.json"
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
    sync_path = evidence_root / "source_sync_prefix_v4.json"
    atomic_json(
        sync_path,
        {
            "schema_version": SYNC_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "transport": "scheduleurm-route-aware-rsync",
            "checksum_mode": "sha256-per-file-after-rsync",
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
