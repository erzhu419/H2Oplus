#!/usr/bin/env python3
"""Synchronize the exact v17 development partition after token-bound exit 0."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

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
REMOTE_RUN = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v17_tail_service_reserve_evaluation_v1/remote_results_v1/single_jtl110gpu"
)
SUBMISSION_SHA256 = "37795ac948a19bb49667d4a54b7304be6c10952a39a79430da3b18644aeda804"
SUBMISSION_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-submission-v1"
SYNC_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-sync-v1"
SCHEDULER_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-scheduler-evidence-v1"
TASK_ID = "t83728"
EXPECTED_TOKEN = "1787585967966166158:1528677:126605756510208"
EXPECTED_ENV = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
        "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
        "libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
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
        default=REPRO / "h2oplus_v17_tail_service_reserve_evaluation_submission_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO
        / "h2oplus_v17_tail_service_reserve_results/evaluation_partition_v1",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO
        / "h2oplus_v17_tail_service_reserve_results/evidence/evaluation_v1",
    )
    return parser.parse_args()


def validate_partition(root: Path) -> None:
    host = json.loads((root / "host_execution.json").read_text())
    summary = json.loads((root / "host_summary.json").read_text())
    results = sorted(root.rglob("sumo_seed*.json"))
    manifests = sorted(root.glob("*/matrix_manifest.json"))
    completed = host.get("completed_blocks", [])
    if (
        host.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-evaluation-host-execution-v1"
        or host.get("status") != "completed"
        or host.get("completed_results") != 135
        or host.get("expected_results") != 135
        or host.get("formal_held_out_outcomes_read") is not False
        or host.get("training") is not False
        or host.get("workers") != 12
        or len(completed) != 3
        or summary.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-evaluation-host-summary-v1"
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 135
        or summary.get("formal_held_out_outcomes_read") is not False
        or summary.get("training") is not False
        or len(summary.get("matrix_manifests", [])) != 3
        or len(results) != 135
        or len(manifests) != 3
    ):
        raise ValueError(f"v17 synchronized partition contract failed: {root}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        partition = manifest.get("matrix_partition", {})
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(manifest.get("jobs", [])) != 45
            or len(manifest.get("source_sha256", {})) != 53
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v17 synchronized matrix contract failed: {manifest_path}")


def install_results_link(output_root: Path) -> Path:
    link = REPRO / (
        "deployments/h2oplus_v17_tail_service_reserve_evaluation_v1/remote_results_v1"
    )
    expected = Path("../../h2oplus_v17_tail_service_reserve_results/evaluation_partition_v1")
    if link.is_symlink():
        if Path(os.readlink(link)) != expected:
            raise ValueError(f"v17 local results symlink drifted: {link}")
    elif link.exists():
        raise ValueError(f"v17 local results path is not a symlink: {link}")
    else:
        link.symlink_to(expected)
    if link.resolve(strict=True) != output_root.resolve(strict=True):
        raise ValueError("v17 local results symlink target drifted")
    return link


def main() -> int:
    args = parse_args()
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v17 evaluation submission identity drifted")
    submission = json.loads(submission_path.read_text())
    submitted = submission.get("task", {})
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "development"
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or submission.get("total_remote_sumo_episodes") != 135
        or submission.get("task_count") != 1
        or submitted.get("task_id") != TASK_ID
        or submitted.get("expected_jobs") != 135
        or submitted.get("remote_result_dir") != str(REMOTE_RUN)
    ):
        raise ValueError("v17 evaluation submission is stale or incomplete")

    by_id = {task["id"]: task for task in queue_tasks(args.queue)}
    if TASK_ID not in by_id:
        raise ValueError("scheduler queue lacks registered v17 task")
    task = by_id[TASK_ID]
    env = task.get("extra_env", {})
    if (
        task.get("status") != "done"
        or task.get("node") != "jtl110gpu"
        or task.get("require_node") != "jtl110gpu"
        or task.get("cpu_declared_cores") != 12
        or task.get("est_vram_mb") != 0
        or task.get("exit_status_token") != EXPECTED_TOKEN
        or task.get("signature") != submitted.get("signature")
        or task.get("result_dir") != str(REMOTE_RUN)
        or submitted.get("cpu_cores") != 12
        or submitted.get("ram_mb") != 32768
        or submitted.get("est_vram_mb") != 0
        or any(env.get(key) != value for key, value in EXPECTED_ENV.items())
        or "CUDA_VISIBLE_DEVICES" in env
    ):
        raise ValueError("v17 scheduler registration binding drifted")

    status_path = task.get("exit_status_path")
    log_path = task.get("log_path")
    if not status_path or not log_path:
        raise ValueError("v17 scheduler evidence paths are missing")
    process_pattern = (
        "run_h2oplus_v17_tail_service_reserve_evaluation_dispatch.py|"
        "run_evaluation_matrix_v17_tail_service_reserve.py|"
        "evaluate_transit_v17_tail_service_reserve.py"
    )
    command = (
        f"cat {shlex.quote(status_path)}; printf '\n__LOG__\n'; "
        f"cat {shlex.quote(log_path)}; printf '\n__PROC__\n'; "
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
        raise RuntimeError(f"v17 remote completion probe failed: {failures}")
    exit_text, remainder = out.split("\n__LOG__\n", 1)
    log_text, process_text = remainder.split("\n__PROC__\n", 1)
    fields = exit_text.strip().split("\t")
    if (
        len(fields) != 3
        or fields[0] != "0"
        or fields[2] != EXPECTED_TOKEN
        or fields[2] != task.get("exit_status_token")
        or int(fields[1]) < int(task.get("started_at", 0))
        or int(fields[1]) > int(task.get("finished_at", 0))
        or "v17_dispatch_started" not in log_text
        or "v17_inputs_verified" not in log_text
        or log_text.count("v17_block_started") != 3
        or log_text.count("v17_block_completed") != 3
        or "v17_dispatch_completed" not in log_text
        or any(
            marker in log_text.lower()
            for marker in ("traceback", "exception", "v17_dispatch_failed")
        )
        or process_text.strip()
    ):
        raise ValueError("v17 token-bound backend closure failed")

    output_root = args.output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    local_run = output_root / "single_jtl110gpu"
    backend_local = evidence_root / f"backend_exit_{TASK_ID}.status"
    log_local = evidence_root / f"scheduler_{TASK_ID}.log"
    backend_local.write_text(exit_text.strip() + "\n")
    log_local.write_text(log_text)

    remote_records = remote_inventory_with_retry(scheduler, "jtl110gpu", REMOTE_RUN)
    local_records = inventory(local_run) if local_run.is_dir() else []
    if remote_records != local_records:
        sync_run(scheduler, "jtl110gpu", REMOTE_RUN, local_run)
        local_records = inventory(local_run)
    if remote_records != local_records:
        raise ValueError("v17 partition inventory mismatch")
    validate_partition(local_run)
    link = install_results_link(output_root)

    captured = datetime.now(timezone.utc).isoformat()
    queue_snapshot = evidence_root / "queue_snapshot_evaluation_v1.json"
    atomic_json(
        queue_snapshot,
        {
            "schema_version": (
                "transit-h2oplus-v17-tail-service-reserve-evaluation-queue-snapshot-v1"
            ),
            "captured_utc": captured,
            "tasks": [task],
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
            "scheduler_frontend_completion": {
                "observed_status": task.get("status"),
                "lifecycle_log_nonempty": True,
                "scientific_result_status_inferred_from_scheduler_frontend_only": False,
            },
            "submission": descriptor(submission_path),
            "queue_snapshot": descriptor(queue_snapshot),
            "task": {
                "task_id": TASK_ID,
                "node": task.get("node"),
                "signature": task.get("signature"),
                "declared_cpu_cores": task.get("cpu_declared_cores"),
                "submitted_ram_mb": submitted.get("ram_mb"),
                "declared_vram_mb": task.get("est_vram_mb"),
                "peak_ram_mb": task.get("peak_ram_mb"),
                "required_environment": env,
                "backend_returncode": int(fields[0]),
                "backend_finished_epoch": int(fields[1]),
                "backend_token": fields[2],
                "token_matches": True,
                "residual_evaluation_process_count": 0,
                "backend_exit": descriptor(backend_local),
                "scheduler_log": descriptor(log_local),
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
            "partition": {
                "task_id": TASK_ID,
                "source_node": "jtl110gpu",
                "source_root": str(REMOTE_RUN),
                "local_root": str(local_run),
                "expected_results": 135,
                "file_count": len(local_records),
                "total_bytes": sum(record["bytes"] for record in local_records),
                "files": local_records,
            },
        },
    )
    print(
        json.dumps(
            {
                "results": 135,
                "files": len(local_records),
                "bytes": sum(record["bytes"] for record in local_records),
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
