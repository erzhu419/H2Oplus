#!/usr/bin/env python3
"""Synchronize exact v12 development-evaluation partitions after both shards close."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sync_h2oplus_v11_actor_hold_training_v2 import (
    atomic_json,
    descriptor,
    inventory,
    load_scheduler,
    queue_tasks,
    remote_inventory_with_retry,
    sha256,
    sync_run,
)


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
REMOTE_ROOT = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v12_staged_release_evaluation_v1/remote_results_v1"
)
SUBMISSION_SHA256 = "331a62b1659f9f309718443e75c9e952ff64116c48272da14f1d9296fa603dd7"
SUBMISSION_SCHEMA = "transit-h2oplus-v12-staged-release-evaluation-submission-v1"
SYNC_SCHEMA = "transit-h2oplus-v12-staged-release-evaluation-sync-v1"
SCHEDULER_SCHEMA = "transit-h2oplus-v12-staged-release-evaluation-scheduler-evidence-v1"
TASKS = {
    "t83115": {"shard_index": 0, "shard": "shard0_jtl110gpu"},
    "t83116": {"shard_index": 1, "shard": "shard1_jtl110gpu"},
}
EXPECTED_ENV = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
        "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
        "libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
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
        "--submission",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_evaluation_submission_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_results/evaluation_partitions_v1",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO
        / "h2oplus_v12_staged_release_results/evidence/evaluation_v1",
    )
    return parser.parse_args()


def validate_local_partition(root: Path, shard_index: int) -> None:
    host = json.loads((root / "host_execution.json").read_text())
    summary = json.loads((root / "host_summary.json").read_text())
    results = list(root.rglob("sumo_seed*.json"))
    manifests = sorted(root.rglob("matrix_manifest.shard-*.json"))
    if (
        host.get("status") != "completed"
        or host.get("returncode") != 0
        or host.get("formal_held_out_outcomes_read") is not False
        or host.get("training") is not False
        or host.get("workers") != 12
        or host.get("shard_index") != shard_index
        or host.get("shard_count") != 2
        or summary.get("completed_jobs") != 315
        or summary.get("formal_held_out_outcomes_read") is not False
        or len(results) != 315
        or len(manifests) != 3
    ):
        raise ValueError(f"v12 synchronized partition contract failed: {root}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        partition = manifest.get("matrix_partition", {})
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(manifest.get("jobs", [])) != 105
            or len(manifest.get("source_sha256", {})) != 43
            or partition.get("full_job_count") != 210
            or partition.get("selected_job_count") != 105
            or partition.get("shard_count") != 2
            or partition.get("shard_index") != shard_index
        ):
            raise ValueError(f"v12 synchronized matrix contract failed: {manifest_path}")


def install_remote_results_link(output_root: Path) -> Path:
    link = (
        REPRO
        / "deployments/h2oplus_v12_staged_release_evaluation_v1/remote_results_v1"
    )
    expected = Path("../../h2oplus_v12_staged_release_results/evaluation_partitions_v1")
    if link.is_symlink():
        if Path(os.readlink(link)) != expected:
            raise ValueError(f"v12 local remote-results symlink drifted: {link}")
    elif link.exists():
        raise ValueError(f"v12 local remote-results path is not a symlink: {link}")
    else:
        link.symlink_to(expected)
    if link.resolve(strict=True) != output_root.resolve(strict=True):
        raise ValueError("v12 remote-results symlink target drifted")
    return link


def main() -> int:
    args = parse_args()
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v12 evaluation submission identity drifted")
    submission = json.loads(submission_path.read_text())
    submitted = {item["task_id"]: item for item in submission.get("tasks", [])}
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "development"
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("total_remote_sumo_episodes") != 630
        or set(submitted) != set(TASKS)
    ):
        raise ValueError("v12 evaluation submission is stale or incomplete")

    by_id = {task["id"]: task for task in queue_tasks(args.queue)}
    if not set(TASKS) <= set(by_id):
        raise ValueError("scheduler queue lacks a registered v12 evaluation task")
    rows = [by_id[task_id] for task_id in sorted(TASKS)]
    for task in rows:
        expected = TASKS[task["id"]]
        item = submitted[task["id"]]
        env = task.get("extra_env", {})
        if (
            task.get("status") != "done"
            or task.get("exit_code") != 0
            or task.get("node") != "jtl110gpu"
            or task.get("require_node") != "jtl110gpu"
            or task.get("cpu_declared_cores") != 12
            or task.get("gpu_idx") is not None
            or task.get("est_vram_mb") != 0
            or item.get("shard_index") != expected["shard_index"]
            or item.get("expected_jobs") != 315
            or item.get("cpu_cores") != 12
            or item.get("ram_mb") != 32768
            or item.get("est_vram_mb") != 0
            or item.get("signature") != task.get("signature")
            or item.get("remote_result_dir") != task.get("result_dir")
            or any(env.get(key) != value for key, value in EXPECTED_ENV.items())
            or "CUDA_VISIBLE_DEVICES" in env
        ):
            raise ValueError(f"v12 scheduler binding drifted for {task['id']}")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    sync_rows: list[dict[str, Any]] = []
    scheduler_rows: list[dict[str, Any]] = []
    for task_id in sorted(TASKS):
        task = by_id[task_id]
        expected = TASKS[task_id]
        remote_run = REMOTE_ROOT / expected["shard"]
        local_run = output_root / expected["shard"]
        status_path = task.get("exit_status_path")
        log_path = task.get("log_path")
        if not status_path or not log_path or task.get("result_dir") != str(remote_run):
            raise ValueError(f"v12 remote evidence path drifted for {task_id}")
        process_pattern = (
            "run_h2oplus_v12_staged_release_evaluation_dispatch.py|"
            "run_evaluation_matrix_v12_staged_release.py|"
            "evaluate_transit_v12_staged_release.py"
        )
        command = (
            f"cat {shlex.quote(status_path)}; printf '\\n__LOG__\\n'; "
            f"cat {shlex.quote(log_path)}; printf '\\n__PROC__\\n'; "
            f"ps -eo pid=,args= | grep -E {shlex.quote(process_pattern)} "
            "| grep -v grep || true"
        )
        failures = []
        for attempt in range(1, 4):
            rc, out, err = scheduler.run_on(
                "jtl110gpu", command, timeout=60, check=False
            )
            if rc == 0 and "\n__LOG__\n" in out and "\n__PROC__\n" in out:
                break
            failures.append(
                {"attempt": attempt, "returncode": rc, "error": (err or out)[-1000:]}
            )
            if attempt < 3:
                time.sleep(5 * attempt)
        if rc != 0 or "\n__LOG__\n" not in out or "\n__PROC__\n" not in out:
            raise RuntimeError(f"v12 remote completion probe failed: {task_id}: {failures}")
        exit_text, remainder = out.split("\n__LOG__\n", 1)
        log_text, process_text = remainder.split("\n__PROC__\n", 1)
        fields = exit_text.strip().split("\t")
        shard_index = expected["shard_index"]
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != task.get("exit_status_token")
            or int(fields[1]) != int(task.get("backend_finished_at"))
            or process_text.strip()
            or f"DONE H2O+ v12 staged-release shard {shard_index}: 315/315 jobs"
            not in log_text
        ):
            raise ValueError(f"v12 token-bound backend closure failed: {task_id}")

        backend_local = evidence_root / f"backend_exit_{task_id}.status"
        log_local = evidence_root / f"scheduler_{task_id}.log"
        backend_local.write_text(exit_text.strip() + "\n")
        log_local.write_text(log_text.rstrip() + "\n")
        remote_records = remote_inventory_with_retry(scheduler, "jtl110gpu", remote_run)
        local_records = inventory(local_run) if local_run.is_dir() else []
        if remote_records != local_records:
            sync_run(scheduler, "jtl110gpu", remote_run, local_run)
            local_records = inventory(local_run)
        if remote_records != local_records:
            raise ValueError(f"v12 partition inventory mismatch: {task_id}")
        validate_local_partition(local_run, shard_index)
        sync_rows.append(
            {
                "task_id": task_id,
                "shard_index": shard_index,
                "source_node": "jtl110gpu",
                "source_root": str(remote_run),
                "local_root": str(local_run),
                "expected_results": 315,
                "file_count": len(local_records),
                "total_bytes": sum(record["bytes"] for record in local_records),
                "files": local_records,
            }
        )
        scheduler_rows.append(
            {
                "task_id": task_id,
                "shard_index": shard_index,
                "node": task["node"],
                "status": task["status"],
                "exit_code": task["exit_code"],
                "finished_at": task.get("finished_at"),
                "signature": task["signature"],
                "declared_cpu_cores": task.get("cpu_declared_cores"),
                "declared_ram_mb": submitted[task_id].get("ram_mb"),
                "declared_vram_mb": submitted[task_id].get("est_vram_mb"),
                "live_ram_estimate_mb": task.get("ram_mb"),
                "peak_ram_mb": task.get("peak_ram_mb"),
                "required_environment": task.get("extra_env"),
                "backend_returncode": int(fields[0]),
                "backend_finished_epoch": int(fields[1]),
                "backend_token": fields[2],
                "token_matches": True,
                "residual_evaluation_process_count": 0,
                "backend_exit": descriptor(backend_local),
                "scheduler_log": descriptor(log_local),
            }
        )
    if sum(row["expected_results"] for row in sync_rows) != 630:
        raise ValueError("v12 synchronized result cardinality drifted")

    link = install_remote_results_link(output_root)
    captured = datetime.now(timezone.utc).isoformat()
    queue_path = evidence_root / "queue_snapshot_evaluation_v1.json"
    atomic_json(
        queue_path,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-evaluation-queue-snapshot-v1",
            "captured_utc": captured,
            "tasks": sorted(rows, key=lambda row: row["id"]),
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_v1.json"
    atomic_json(
        scheduler_path,
        {
            "schema_version": SCHEDULER_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "token_bound_backend_exit0": True,
            "submission": descriptor(submission_path),
            "queue_snapshot": descriptor(queue_path),
            "tasks": {
                row["task_id"]: row
                for row in sorted(scheduler_rows, key=lambda row: row["task_id"])
            },
        },
    )
    sync_path = evidence_root / "source_sync_v1.json"
    atomic_json(
        sync_path,
        {
            "schema_version": SYNC_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "transport": "scheduleurm-route-aware-compressed-tar-stream",
            "checksum_mode": "sha256-per-file-before-and-after-transfer",
            "formal_held_out_outcomes_read": False,
            "local_sumo_executed": False,
            "local_remote_results_symlink": str(link),
            "partitions": sorted(sync_rows, key=lambda row: row["shard_index"]),
        },
    )
    print(
        json.dumps(
            {
                "partitions": len(sync_rows),
                "results": 630,
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
