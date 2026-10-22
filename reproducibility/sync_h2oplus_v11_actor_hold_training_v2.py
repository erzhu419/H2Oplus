#!/usr/bin/env python3
"""Synchronize exact v11-v2 training runs after all five jobs close."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
REMOTE_ROOT = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v11_actor_hold_pilot_v2/remote_results"
)
SUBMISSION_SCHEMA = "transit-h2oplus-v11-actor-hold-submission-v2"
SYNC_SCHEMA = "transit-h2oplus-v11-actor-hold-training-sync-v2"
SCHEDULER_SCHEMA = "transit-h2oplus-v11-actor-hold-scheduler-evidence-v2"


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


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


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


def remote_inventory_with_retry(
    scheduler, node: str, root: Path
) -> list[dict[str, Any]]:
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


def sync_run(scheduler, node: str, remote: Path, local: Path) -> None:
    """Install a complete compressed mirror atomically."""
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
            staged = staging_parent / remote.name
            if completed.returncode == 0 and staged.is_dir():
                if local.exists():
                    shutil.rmtree(local)
                os.replace(staged, local)
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
        default=REPRO / "h2oplus_v11_actor_hold_submission_v2.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_results/training_v2",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_results/evidence_training_v2",
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
        or submission.get("study_stage") != "development"
        or submission.get("development_evaluation_submitted") is not False
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or len(expected) != 5
        or {item.get("seed") for item in expected} != set(range(6201, 6206))
    ):
        raise ValueError("v11-v2 training submission is stale or incomplete")

    by_id = {task["id"]: task for task in queue_tasks(args.queue)}
    task_ids = {item["task_id"] for item in expected}
    if not task_ids <= set(by_id):
        raise ValueError("hot queue does not contain every v11-v2 training task")
    rows = [by_id[task_id] for task_id in sorted(task_ids)]
    if any(task.get("status") != "done" or task.get("exit_code") != 0 for task in rows):
        raise RuntimeError("v11-v2 training tasks are not all done with exit0")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    expected_names = {item["name"] for item in expected}
    unexpected = {
        path.name for path in output_root.iterdir() if path.is_dir()
    } - expected_names
    if unexpected:
        raise ValueError(f"unexpected canonical training directories: {sorted(unexpected)}")

    def sync_item(
        item: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        task = by_id[item["task_id"]]
        node = item["require_node"]
        remote_run = REMOTE_ROOT / item["name"]
        if (
            task.get("node") != node
            or task.get("signature") != item["scheduler_signature"]
            or task.get("result_dir") != str(remote_run)
            or task.get("cpu_declared_cores") != 6
            or "cpu=6 ram=32768MB vram=8192MB" not in item["scheduler_output"]
        ):
            raise ValueError(f"scheduler binding drifted for {item['task_id']}")

        exit_path = task["exit_status_path"]
        log_path = task["log_path"]
        process_filter = shlex.quote(
            f"--run-name {item['name']}"
        )
        command = (
            f"cat {shlex.quote(exit_path)}; printf '\\n__LOG__\\n'; "
            f"cat {shlex.quote(log_path)}; printf '\\n__PROC__\\n'; "
            "ps -eo pid=,args= | grep -F 'SimpleSAC/train_transit_v11_actor_hold.py' "
            f"| grep -F -- {process_filter} | grep -v grep || true"
        )
        failures = []
        for attempt in range(1, 4):
            rc, out, err = scheduler.run_on(node, command, timeout=60, check=False)
            if rc == 0 and "\n__LOG__\n" in out and "\n__PROC__\n" in out:
                break
            failures.append(
                {"attempt": attempt, "returncode": rc, "error": (err or out)[-1000:]}
            )
            if attempt < 3:
                time.sleep(5 * attempt)
        if rc != 0 or "\n__LOG__\n" not in out or "\n__PROC__\n" not in out:
            raise RuntimeError(
                f"remote completion evidence failed for {item['task_id']}: "
                f"{json.dumps(failures, sort_keys=True)}"
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

        local_run = output_root / item["name"]
        remote_records = remote_inventory_with_retry(scheduler, node, remote_run)
        local_records = inventory(local_run) if local_run.is_dir() else []
        if remote_records != local_records:
            sync_run(scheduler, node, remote_run, local_run)
            local_records = inventory(local_run)
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
            "gpu_idx": task.get("gpu_idx"),
            "declared_cpu_cores": task.get("cpu_declared_cores"),
            "declared_ram_mb": 32_768,
            "declared_vram_mb": 8_192,
            "live_ram_estimate_mb": task.get("ram_mb"),
            "live_vram_estimate_mb": task.get("est_vram_mb"),
            "status": task["status"],
            "exit_code": task["exit_code"],
            "finished_at": task.get("finished_at"),
            "signature": task["signature"],
            "required_environment": task.get("extra_env"),
            "backend_returncode": int(fields[0]),
            "backend_finished_epoch": int(fields[1]),
            "backend_token": fields[2],
            "token_matches": True,
            "residual_canonical_process_count": 0,
            "backend_exit": descriptor(backend_local),
            "scheduler_log": descriptor(log_local),
        }
        return sync_row, scheduler_row

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = list(executor.map(sync_item, expected))
    sync_rows = [row[0] for row in completed]
    scheduler_rows = [row[1] for row in completed]
    actual_names = {path.name for path in output_root.iterdir() if path.is_dir()}
    if actual_names != expected_names:
        raise ValueError("canonical training directory set is incomplete")

    captured = datetime.now(timezone.utc).isoformat()
    queue_path = evidence_root / "queue_snapshot_training_v2.json"
    atomic_json(
        queue_path,
        {
            "schema_version": "transit-h2oplus-v11-actor-hold-queue-snapshot-v2",
            "captured_utc": captured,
            "tasks": sorted(rows, key=lambda row: row["id"]),
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_training_v2.json"
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
    sync_path = evidence_root / "source_sync_training_v2.json"
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
