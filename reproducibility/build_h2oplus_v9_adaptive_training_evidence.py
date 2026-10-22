#!/usr/bin/env python3
"""Capture exact scheduler, backend-exit, process, and source-sync evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v9_adaptive_pilot_v2/remote_results"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed


def remote_inventory(host: str, names: list[str]) -> tuple[int, int]:
    code = (
        "import json,sys; from pathlib import Path; "
        f"root=Path({str(REMOTE_ROOT)!r}); names=json.loads(sys.argv[1]); "
        "files=[p for n in names for p in (root/n).rglob('*') if p.is_file()]; "
        "print(json.dumps({'files':len(files),'bytes':sum(p.stat().st_size for p in files)}))"
    )
    command = f"python3 -c {shlex.quote(code)} {shlex.quote(json.dumps(names))}"
    payload = json.loads(run(["ssh", host, command]).stdout)
    return int(payload["files"]), int(payload["bytes"])


def residual_processes(host: str) -> list[str]:
    command = (
        "ps -eo pid=,args= | "
        "grep -F 'SimpleSAC/train_transit_v9_adaptive.py' | "
        "grep -v grep || true"
    )
    completed = subprocess.run(
        ["ssh", host, command], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(f"remote process probe failed on {host}: {completed.stderr}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def rsync_differences(host: str, names: list[str], training_root: Path) -> list[str]:
    command = [
        "rsync", "-a", "--omit-dir-times", "--checksum", "--dry-run",
        "--itemize-changes", "--prune-empty-dirs",
    ]
    for name in names:
        command.append(f"--include=/{name}/***")
    command.extend([
        "--exclude=*", f"{host}:{REMOTE_ROOT}/", f"{training_root}/",
    ])
    output = run(command).stdout
    return [line for line in output.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission = json.loads(args.submission.expanduser().resolve(strict=True).read_text())
    queue = json.loads(args.queue.expanduser().resolve(strict=True).read_text())
    training_root = args.training_root.expanduser().resolve(strict=True)
    evidence_root = args.evidence_root.expanduser().resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    queue_tasks = queue.get("tasks", [])
    if isinstance(queue_tasks, dict):
        queue_tasks = list(queue_tasks.values())
    by_id = {item["id"]: item for item in queue_tasks}

    host_processes = {
        host: residual_processes(host) for host in {item["require_node"] for item in submission["tasks"]}
    }
    scheduler_rows = []
    queue_snapshot = []
    for expected in submission["tasks"]:
        task = by_id[expected["task_id"]]
        exit_source = Path(task["exit_status_path"])
        exit_path = evidence_root / exit_source.name
        fields = exit_path.read_text().strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"malformed backend exit: {exit_path}")
        log_path = evidence_root / f"sched_{expected['task_id']}.log"
        queue_snapshot.append(task)
        scheduler_rows.append({
            "task_id": expected["task_id"],
            "run_name": expected["name"],
            "node": task.get("node"),
            "status": task.get("status"),
            "exit_code": task.get("exit_code"),
            "finished_at": task.get("finished_at"),
            "signature": task.get("signature"),
            "exit_status_token": task.get("exit_status_token"),
            "backend_returncode": int(fields[0]),
            "backend_finished_epoch": int(fields[1]),
            "backend_token": fields[2],
            "token_matches": fields[2] == task.get("exit_status_token"),
            "backend_exit": descriptor(exit_path),
            "scheduler_log": descriptor(log_path),
            "residual_process_count": len(host_processes[expected["require_node"]]),
        })

    snapshot_path = evidence_root / "queue_snapshot_v9_training_v2.json"
    atomic_json(snapshot_path, {
        "schema_version": "transit-h2oplus-v9-adaptive-queue-snapshot-v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": sorted(queue_snapshot, key=lambda item: item["id"]),
    })
    scheduler_path = evidence_root / "scheduler_evidence_v9_training_v2.json"
    atomic_json(scheduler_path, {
        "schema_version": "transit-h2oplus-v9-adaptive-scheduler-evidence-v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "queue_snapshot": descriptor(snapshot_path),
        "remote_process_probe": host_processes,
        "tasks": sorted(scheduler_rows, key=lambda item: item["task_id"]),
    })

    grouped: dict[str, list[str]] = {}
    for item in submission["tasks"]:
        grouped.setdefault(item["require_node"], []).append(item["name"])
    sync_paths = []
    for host, names in sorted(grouped.items()):
        names = sorted(names)
        remote_files, remote_bytes = remote_inventory(host, names)
        local_paths = [
            path
            for name in names
            for path in (training_root / name).rglob("*")
            if path.is_file()
        ]
        differences = rsync_differences(host, names, training_root)
        path = evidence_root / f"source_sync_{host}_v9_training_v2.json"
        atomic_json(path, {
            "schema_version": "transit-h2oplus-v9-adaptive-source-sync-v1",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "passed": not differences,
            "source_host": host,
            "source_root": str(REMOTE_ROOT),
            "local_root": str(training_root),
            "run_names": names,
            "checksum_mode": "rsync-content-checksum-dry-run",
            "differences": differences,
            "source_file_count": remote_files,
            "local_file_count": len(local_paths),
            "source_total_bytes": remote_bytes,
            "local_total_bytes": sum(path.stat().st_size for path in local_paths),
        })
        sync_paths.append(path)

    print(json.dumps({
        "scheduler_evidence": descriptor(scheduler_path),
        "queue_snapshot": descriptor(snapshot_path),
        "sync_evidence": [descriptor(path) for path in sync_paths],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
