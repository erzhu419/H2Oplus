#!/usr/bin/env python3
"""Synchronize exact scheduler and file evidence for all v13-v2 branches."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sync_h2oplus_v12_staged_release_prefix_v2 import (
    REPRO,
    ROOT,
    archive_tasks,
    atomic_json,
    descriptor,
    inventory,
    load_scheduler,
    queue_tasks,
    remote_inventory,
    sync_run,
)


REMOTE_ROOT = (
    REPRO
    / "deployments/h2oplus_v13_proximal_release_pilot_v2/remote_results_v2"
)
SUBMISSION_SHA256 = (
    "cfd3f6454c66eabba7e2c726080456dcf4e736aa718582b86469b26bfdeb6533"
)
SUBMISSION_SCHEMA = "transit-h2oplus-v13-proximal-release-submission-v2"
SYNC_SCHEMA = "transit-h2oplus-v13-proximal-release-branch-sync-v2"
SCHEDULER_SCHEMA = (
    "transit-h2oplus-v13-proximal-release-branch-scheduler-evidence-v2"
)
EXPECTED_TASK_IDS = {
    *(f"t{task_id}" for task_id in range(83304, 83310)),
    *(f"t{task_id}" for task_id in range(83314, 83323)),
}
EXPECTED_ENVIRONMENT = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
        "deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/"
        "toolchain/libxcrypt-4.4.36-hd590300_1/include"
    ),
    "H2OPLUS_V13_PROTOCOL_SHA256": (
        "8f330bc7a55b35c9c10ae03434ddbecd30528e4529b01e734dbbd6c44bdd5f12"
    ),
    "H2OPLUS_V13_BASE_DEPLOYMENT_MANIFEST_SHA256": (
        "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
    ),
}


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
        default=REPRO / "h2oplus_v13_proximal_release_submission_v2.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_results/branches_v2",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_results/evidence_branches_v2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete local transfer after revalidating remote closure.",
    )
    return parser.parse_args()


def existing_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def write_immutable_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"existing transfer evidence drifted: {path}")
        return
    path.write_text(content)


def sync_run_resumable(scheduler: Any, node: str, remote: Path, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-a",
        "--omit-dir-times",
        "--checksum",
        "--partial",
        "--partial-dir=.rsync-partial",
        "-e",
        scheduler._ssh_rsync_shell_for_node(node),
        f"{scheduler._ssh_target_for_node(node)}:{remote}/",
        f"{local}/",
    ]
    failures = []
    for attempt in range(1, 9):
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
        if attempt < 8:
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(json.dumps(failures, sort_keys=True))


def remote_inventory_with_retry(
    scheduler: Any, node: str, root: Path
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


def main() -> int:
    args = parse_args()
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    submission_path = args.submission.expanduser().resolve(strict=True)
    if descriptor(submission_path)["sha256"] != SUBMISSION_SHA256:
        raise ValueError("v13 branch submission record drifted")
    submission = json.loads(submission_path.read_text())
    expected = submission.get("tasks", [])
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("status") != "submitted"
        or submission.get("submitted_stage") != "branch-training"
        or submission.get("submitted_branch_tasks") != 15
        or submission.get("submitted_prefix_tasks") != 0
        or submission.get("prefix_training_rerun") is not False
        or submission.get("development_evaluation_submitted") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("formal_held_out_outcomes_read") is not False
        or len(expected) != 15
    ):
        raise ValueError("v13 branch submission record is stale or incomplete")

    queue_path = args.queue.expanduser().resolve(strict=True)
    rows = queue_tasks(queue_path)
    by_id = {row["id"]: row for row in rows}
    task_ids = {item["task_id"] for item in expected}
    if task_ids != EXPECTED_TASK_IDS or not task_ids <= set(by_id):
        raise ValueError("unexpected or missing v13 branch task lineage")

    all_rows = rows + archive_tasks(args.archive.expanduser())
    for item in expected:
        matches = [
            row
            for row in all_rows
            if row.get("signature") == item["scheduler_signature"]
        ]
        if len(matches) != 1 or matches[0].get("id") != item["task_id"]:
            raise ValueError(
                f"duplicate or retried branch signature: {item['scheduler_signature']}"
            )

    selected = [by_id[task_id] for task_id in sorted(task_ids)]
    if any(row.get("status") != "done" or row.get("exit_code") != 0 for row in selected):
        raise RuntimeError("v13 branch tasks have not all closed with scheduler exit0")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    resumed = output_root.exists() or evidence_root.exists()
    if resumed and not args.resume:
        raise FileExistsError(
            "incomplete v13 branch evidence exists; use --resume to revalidate and continue"
        )
    if resumed:
        if not output_root.is_dir() or not evidence_root.is_dir():
            raise ValueError("resume requires both existing transfer directories")
        final_names = {
            "queue_snapshot_branches_v2.json",
            "scheduler_evidence_branches_v2.json",
            "branch_sync_v2.json",
        }
        if any((evidence_root / name).exists() for name in final_names):
            raise FileExistsError("refusing to resume an already finalized transfer")
    else:
        output_root.mkdir(parents=True)
        evidence_root.mkdir(parents=True)

    node_locks = {
        node: threading.RLock() for node in {item["require_node"] for item in expected}
    }

    def sync_item_unlocked(
        item: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        task = by_id[item["task_id"]]
        remote_run = Path(item["result_root"])
        extra_env = task.get("extra_env", {})
        if (
            task.get("node") != item["require_node"]
            or task.get("require_node") != item["require_node"]
            or task.get("signature") != item["scheduler_signature"]
            or task.get("result_dir") != str(remote_run)
            or task.get("cpu_declared_cores") != 6
            or task.get("est_vram_mb") != 8192
            or any(extra_env.get(key) != value for key, value in EXPECTED_ENVIRONMENT.items())
            or item.get("declared_ram_mb") != 32768
        ):
            raise ValueError(f"scheduler/environment binding drifted: {item['task_id']}")
        if remote_run.parent != REMOTE_ROOT:
            raise ValueError(f"remote result root drifted: {item['task_id']}")

        run_pattern = f"--run-name {item['name']}"
        command = (
            f"cat {shlex.quote(task['exit_status_path'])}; printf '\n__LOG__\n'; "
            f"cat {shlex.quote(task['log_path'])}; printf '\n__PROC__\n'; "
            "ps -eo pid=,args= | grep -F "
            f"{shlex.quote(run_pattern)} | grep -F "
            f"{shlex.quote('SimpleSAC/train_transit_v13_proximal_release.py')} "
            "| grep -v grep || true"
        )
        failures = []
        for attempt in range(1, 9):
            rc, out, err = scheduler.run_on(
                item["require_node"], command, timeout=60, check=False
            )
            if rc == 0 and "\n__LOG__\n" in out and "\n__PROC__\n" in out:
                break
            failures.append(
                {"attempt": attempt, "returncode": rc, "error": (err or out)[-1000:]}
            )
            if attempt < 8:
                time.sleep(min(30, 5 * attempt))
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
            or fields[0] != "0"
            or fields[2] != task.get("exit_status_token")
            or process_text.strip()
        ):
            raise ValueError(f"token-bound backend closure failed: {item['task_id']}")

        backend_local = evidence_root / f"{item['task_id']}.backend"
        log_local = evidence_root / f"sched_{item['task_id']}.log"
        write_immutable_text(backend_local, exit_text.strip() + "\n")
        write_immutable_text(log_local, log_text.rstrip() + "\n")

        local_run = output_root / item["name"]
        prior_files = existing_files(local_run)
        with node_locks[item["require_node"]]:
            sync_run_resumable(
                scheduler, item["require_node"], remote_run, local_run
            )
        remote_records = remote_inventory_with_retry(
            scheduler, item["require_node"], remote_run
        )
        local_records = inventory(local_run)
        if remote_records != local_records:
            raise ValueError(f"remote/local inventory mismatch: {item['name']}")
        sync_row = {
            "task_id": item["task_id"],
            "run_name": item["name"],
            "variant": item["variant"],
            "seed": item["seed"],
            "source_node": item["require_node"],
            "source_root": str(remote_run),
            "local_root": str(local_run),
            "file_count": len(local_records),
            "total_bytes": sum(record["bytes"] for record in local_records),
            "files": local_records,
            "resumed_existing_local_root": bool(prior_files),
            "resume_prior_files": prior_files,
        }
        scheduler_row = {
            "task_id": item["task_id"],
            "run_name": item["name"],
            "variant": item["variant"],
            "seed": item["seed"],
            "node": item["require_node"],
            "status": task["status"],
            "exit_code": task["exit_code"],
            "signature": task["signature"],
            "submitted_at": task.get("submitted_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "declared_cpu_cores": task.get("cpu_declared_cores"),
            "submitted_declared_ram_mb": item.get("declared_ram_mb"),
            "scheduler_live_family_ram_mb": task.get("ram_mb"),
            "declared_vram_mb": task.get("est_vram_mb"),
            "peak_ram_mb": task.get("peak_ram_mb"),
            "peak_vram_mb": task.get("peak_vram_mb"),
            "required_environment": {
                **EXPECTED_ENVIRONMENT,
            },
            "scheduler_assigned_cuda_visible_devices": True,
            "backend_returncode": int(fields[0]),
            "backend_finished_epoch": int(fields[1]),
            "backend_token": fields[2],
            "token_matches": True,
            "residual_canonical_process_count": 0,
            "backend_exit": descriptor(backend_local),
            "scheduler_log": descriptor(log_local),
        }
        progress_path = evidence_root / f"{item['task_id']}.sync_progress.json"
        atomic_json(
            progress_path,
            {
                "schema_version": (
                    "transit-h2oplus-v13-proximal-release-branch-sync-progress-v2"
                ),
                "submission_sha256": SUBMISSION_SHA256,
                "task_id": item["task_id"],
                "scheduler_signature": item["scheduler_signature"],
                "sync_row": sync_row,
                "scheduler_row": scheduler_row,
            },
        )
        return sync_row, scheduler_row

    def sync_item(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        progress_path = evidence_root / f"{item['task_id']}.sync_progress.json"
        task = by_id[item["task_id"]]
        with node_locks[item["require_node"]]:
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                sync_row = progress.get("sync_row", {})
                scheduler_row = progress.get("scheduler_row", {})
                local_run = output_root / item["name"]
                if (
                    progress.get("schema_version")
                    != "transit-h2oplus-v13-proximal-release-branch-sync-progress-v2"
                    or progress.get("submission_sha256") != SUBMISSION_SHA256
                    or progress.get("task_id") != item["task_id"]
                    or progress.get("scheduler_signature")
                    != item["scheduler_signature"]
                    or task.get("status") != "done"
                    or task.get("exit_code") != 0
                    or scheduler_row.get("signature") != task.get("signature")
                    or scheduler_row.get("backend_token")
                    != task.get("exit_status_token")
                    or descriptor(
                        evidence_root / f"{item['task_id']}.backend"
                    )
                    != scheduler_row.get("backend_exit")
                    or descriptor(
                        evidence_root / f"sched_{item['task_id']}.log"
                    )
                    != scheduler_row.get("scheduler_log")
                    or inventory(local_run) != sync_row.get("files")
                ):
                    raise ValueError(
                        f"saved synchronization progress drifted: {item['task_id']}"
                    )
                return sync_row, scheduler_row
            return sync_item_unlocked(item)

    work_items = sorted(
        expected,
        key=lambda item: (
            (output_root / item["name"]).exists(),
            item["task_id"],
        ),
    )
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as executor:
        completed = list(executor.map(sync_item, work_items))
    sync_rows = [row[0] for row in completed]
    scheduler_rows = [row[1] for row in completed]

    captured = datetime.now(timezone.utc).isoformat()
    queue_snapshot = evidence_root / "queue_snapshot_branches_v2.json"
    atomic_json(
        queue_snapshot,
        {
            "schema_version": (
                "transit-h2oplus-v13-proximal-release-branch-queue-snapshot-v2"
            ),
            "captured_utc": captured,
            "tasks": sorted(selected, key=lambda row: row["id"]),
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_branches_v2.json"
    atomic_json(
        scheduler_path,
        {
            "schema_version": SCHEDULER_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "submission": descriptor(submission_path),
            "queue_snapshot": descriptor(queue_snapshot),
            "expected_task_ids": sorted(task_ids),
            "tasks": sorted(scheduler_rows, key=lambda row: row["task_id"]),
            "duplicate_or_retry_count": 0,
            "formal_held_out_outcomes_read": False,
        },
    )
    sync_path = evidence_root / "branch_sync_v2.json"
    atomic_json(
        sync_path,
        {
            "schema_version": SYNC_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "transport": "scheduleurm-route-aware-rsync",
            "checksum_mode": "sha256-per-file-before-and-after-transfer",
            "resumed_after_transport_interruption": resumed,
            "resume_policy": (
                "revalidate-token-and-log-then-rsync-partial-with-per-node-lock"
                if resumed
                else None
            ),
            "scheduler_evidence": descriptor(scheduler_path),
            "runs": sorted(sync_rows, key=lambda row: row["run_name"]),
            "total_runs": len(sync_rows),
            "total_files": sum(row["file_count"] for row in sync_rows),
            "total_bytes": sum(row["total_bytes"] for row in sync_rows),
            "formal_held_out_outcomes_read": False,
        },
    )
    print(
        json.dumps(
            {
                "sync": descriptor(sync_path),
                "scheduler_evidence": descriptor(scheduler_path),
                "runs": len(sync_rows),
                "files": sum(row["file_count"] for row in sync_rows),
                "bytes": sum(row["total_bytes"] for row in sync_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
