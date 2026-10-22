#!/usr/bin/env python3
"""Synchronize exact scheduler and file evidence for the v12-v2 prefix runs."""

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
REMOTE_ROOT = REPRO / (
    "deployments/h2oplus_v12_staged_release_pilot_v2/remote_results"
)
SUBMISSION_SHA256 = (
    "3a5dfe377818cfc33fded151a173c74d7feeb2b1a012fe9dec1e98f71842813d"
)
SYNC_SCHEMA = "transit-h2oplus-v12-staged-release-prefix-sync-v2"
SCHEDULER_SCHEMA = (
    "transit-h2oplus-v12-staged-release-prefix-scheduler-evidence-v2"
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
    raw = payload.get("tasks", []) if isinstance(payload, dict) else payload
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def archive_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        normalized = line.strip("\x00 \t\r\n")
        if normalized:
            rows.append(json.loads(normalized))
    return rows


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


def sync_run(scheduler, node: str, remote: Path, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=False)
    command = [
        "rsync",
        "-a",
        "--omit-dir-times",
        "--checksum",
        "-e",
        scheduler._ssh_rsync_shell_for_node(node),
        f"{scheduler._ssh_target_for_node(node)}:{remote}/",
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
    raise RuntimeError(json.dumps(failures, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / ".claude/scheduler/queue_archive.jsonl",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_prefix_submission_v2.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_results/prefix_v2",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_results/evidence_prefix_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v12 prefix submission record drifted")
    submission = json.loads(submission_path.read_text())
    expected = submission.get("tasks", [])
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v12-staged-release-prefix-submission-v2"
        or submission.get("status") != "submitted"
        or submission.get("submitted_prefix_tasks") != 5
        or submission.get("submitted_branch_tasks") != 0
        or submission.get("development_evaluation_submitted") is not False
        or submission.get("formal_held_out_outcomes_read") is not False
        or len(expected) != 5
    ):
        raise ValueError("v12 prefix submission record is stale or incomplete")

    queue_path = args.queue.expanduser().resolve(strict=True)
    rows = queue_tasks(queue_path)
    by_id = {row["id"]: row for row in rows}
    expected_ids = {item["task_id"] for item in expected}
    if expected_ids != {"t83029", "t83030", "t83031", "t83032", "t83033"}:
        raise ValueError("unexpected v12 prefix task lineage")
    if not expected_ids <= set(by_id):
        raise ValueError("hot queue does not contain every v12 prefix task")

    all_rows = rows + archive_tasks(args.archive.expanduser())
    expected_signatures = {item["scheduler_signature"] for item in expected}
    for signature in expected_signatures:
        matches = [row for row in all_rows if row.get("signature") == signature]
        if len(matches) != 1 or matches[0].get("id") not in expected_ids:
            raise ValueError(f"duplicate or retried prefix signature: {signature}")

    selected = [by_id[task_id] for task_id in sorted(expected_ids)]
    if any(row.get("status") != "done" or row.get("exit_code") != 0 for row in selected):
        raise RuntimeError("v12 prefix tasks have not all closed with scheduler exit0")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    if output_root.exists() or evidence_root.exists():
        raise FileExistsError("refusing to overwrite existing v12 prefix evidence")
    output_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)

    scheduler_rows = []
    sync_rows = []
    for item in expected:
        task = by_id[item["task_id"]]
        remote_run = Path(item["result_root"])
        if (
            task.get("node") != item["require_node"]
            or task.get("require_node") != item["require_node"]
            or task.get("signature") != item["scheduler_signature"]
            or task.get("result_dir") != str(remote_run)
            or task.get("cpu_declared_cores") != 6
            or task.get("est_vram_mb") != 8192
        ):
            raise ValueError(f"scheduler binding drifted for {item['task_id']}")
        if remote_run.parent != REMOTE_ROOT:
            raise ValueError(f"remote result root drifted for {item['task_id']}")

        exit_path = task["exit_status_path"]
        log_path = task["log_path"]
        run_pattern = f"--run-name {item['name']}"
        command = (
            f"cat {shlex.quote(exit_path)}; printf '\n__LOG__\n'; "
            f"cat {shlex.quote(log_path)}; printf '\n__PROC__\n'; "
            "ps -eo pid=,args= | grep -F "
            f"{shlex.quote(run_pattern)} | grep -F "
            f"{shlex.quote('SimpleSAC/train_transit_v12_staged_release.py')} "
            "| grep -v grep || true"
        )
        rc, out, err = scheduler.run_on(
            item["require_node"], command, timeout=60, check=False
        )
        if rc != 0 or "\n__LOG__\n" not in out or "\n__PROC__\n" not in out:
            raise RuntimeError(f"remote completion evidence failed: {err or out}")
        exit_text, remainder = out.split("\n__LOG__\n", 1)
        log_text, process_text = remainder.split("\n__PROC__\n", 1)
        fields = exit_text.strip().split("\t")
        if (
            len(fields) != 3
            or int(fields[0]) != 0
            or fields[2] != task.get("exit_status_token")
            or process_text.strip()
        ):
            raise ValueError(f"token-bound backend closure failed: {item['task_id']}")

        backend_local = evidence_root / f"{item['task_id']}.backend"
        log_local = evidence_root / f"sched_{item['task_id']}.log"
        backend_local.write_text(exit_text.strip() + "\n")
        log_local.write_text(log_text.rstrip() + "\n")

        local_run = output_root / item["name"]
        sync_run(scheduler, item["require_node"], remote_run, local_run)
        remote_records = remote_inventory(
            scheduler, item["require_node"], remote_run
        )
        local_records = inventory(local_run)
        if remote_records != local_records:
            raise ValueError(f"remote/local inventory mismatch: {item['name']}")
        sync_rows.append(
            {
                "task_id": item["task_id"],
                "run_name": item["name"],
                "source_node": item["require_node"],
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
                "node": item["require_node"],
                "status": task["status"],
                "exit_code": task["exit_code"],
                "signature": task["signature"],
                "submitted_at": task.get("submitted_at"),
                "started_at": task.get("started_at"),
                "finished_at": task.get("finished_at"),
                "declared_cpu_cores": task.get("cpu_declared_cores"),
                "declared_vram_mb": task.get("est_vram_mb"),
                "peak_ram_mb": task.get("peak_ram_mb"),
                "peak_vram_mb": task.get("peak_vram_mb"),
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
    queue_snapshot = evidence_root / "queue_snapshot_prefix_v2.json"
    atomic_json(
        queue_snapshot,
        {
            "schema_version": (
                "transit-h2oplus-v12-staged-release-prefix-queue-snapshot-v2"
            ),
            "captured_utc": captured,
            "tasks": sorted(selected, key=lambda row: row["id"]),
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_prefix_v2.json"
    atomic_json(
        scheduler_path,
        {
            "schema_version": SCHEDULER_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "submission": descriptor(submission_path),
            "queue_snapshot": descriptor(queue_snapshot),
            "expected_task_ids": sorted(expected_ids),
            "tasks": sorted(scheduler_rows, key=lambda row: row["task_id"]),
            "duplicate_or_retry_count": 0,
            "formal_held_out_outcomes_read": False,
        },
    )
    sync_path = evidence_root / "prefix_sync_v2.json"
    atomic_json(
        sync_path,
        {
            "schema_version": SYNC_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "scheduler_evidence": descriptor(scheduler_path),
            "runs": sorted(sync_rows, key=lambda row: row["run_name"]),
            "total_runs": len(sync_rows),
            "total_files": sum(row["file_count"] for row in sync_rows),
            "formal_held_out_outcomes_read": False,
        },
    )
    print(json.dumps({"sync": descriptor(sync_path), "runs": len(sync_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
