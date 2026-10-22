#!/usr/bin/env python3
"""Synchronize the exact v19-v2 development partition after backend exit 0."""

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
    "h2oplus_v19_completion_safety_evaluation_v2/remote_results_v2/single_jtl110gpu"
)
SUBMISSION_SHA256 = "5d80abee37280730cc974936a967010d2c219d779d12d845b927d9ba38a869f2"
SUBMISSION_SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-submission-v2"
SYNC_SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-sync-v2"
SCHEDULER_SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-scheduler-evidence-v2"
TASK_ID = "t83842"
EXPECTED_TOKEN = "1787660665540388240:1528677:126605756510208"
EXPECTED_CLASSIFIER_REASON = (
    "never entered training: lifetime 10947s but peak_vram=0 and no training "
    "markers in log tail (likely OOM/silent-kill mid-init)"
)
EXPECTED_RETRIES = ("t83858", "t83859", "t83860")


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
        default=REPRO / "h2oplus_v19_completion_safety_evaluation_submission_v2.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRO / "h2oplus_v19_completion_safety_results/evaluation_partition_v2",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPRO / "h2oplus_v19_completion_safety_results/evidence/evaluation_v2",
    )
    return parser.parse_args()


def validate_partition(root: Path) -> None:
    host = json.loads((root / "host_execution.json").read_text())
    summary = json.loads((root / "host_summary.json").read_text())
    immutable = host.get("immutable_evidence", {})
    results = sorted(root.rglob("sumo_seed*.json"))
    manifests = sorted(root.glob("*/matrix_manifest.json"))
    completed = host.get("completed_blocks", [])
    if (
        host.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-evaluation-host-execution-v1"
        or host.get("status") != "completed"
        or host.get("completed_results") != 135
        or host.get("expected_results") != 135
        or host.get("formal_held_out_outcomes_read") is not False
        or host.get("training") is not False
        or host.get("workers") != 12
        or immutable.get("parent_v18_artifact_integrity_passed") is not True
        or immutable.get("parent_training_stability_passed") is not True
        or len(completed) != 3
        or summary.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-evaluation-host-summary-v1"
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 135
        or summary.get("formal_held_out_outcomes_read") is not False
        or summary.get("training") is not False
        or len(summary.get("matrix_manifests", [])) != 3
        or len(results) != 135
        or len(manifests) != 3
    ):
        raise ValueError(f"v19 synchronized partition contract failed: {root}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        partition = manifest.get("matrix_partition", {})
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(manifest.get("jobs", [])) != 45
            or len(manifest.get("source_sha256", {})) != 55
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v19 synchronized matrix contract failed: {manifest_path}")


def install_results_link(output_root: Path) -> Path:
    link = REPRO / (
        "deployments/h2oplus_v19_completion_safety_evaluation_v2/remote_results_v2"
    )
    expected = Path("../../h2oplus_v19_completion_safety_results/evaluation_partition_v2")
    if link.is_symlink():
        if Path(os.readlink(link)) != expected:
            raise ValueError(f"v19 local results symlink drifted: {link}")
    elif link.exists():
        raise ValueError(f"v19 local results path is not a symlink: {link}")
    else:
        link.symlink_to(expected)
    if link.resolve(strict=True) != output_root.resolve(strict=True):
        raise ValueError("v19 local results symlink target drifted")
    return link


def main() -> int:
    args = parse_args()
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v19 evaluation submission identity drifted")
    submission = json.loads(submission_path.read_text())
    submitted = submission.get("task", {})
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "development"
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or submission.get("screen_type") != "paired-mechanism-screen"
        or submission.get("v19_v1_results_reused") is not False
        or submission.get("v19_v1_completed_scientific_result_count") != 0
        or submission.get("total_remote_sumo_episodes") != 135
        or submission.get("task_count") != 1
        or submitted.get("task_id") != TASK_ID
        or submitted.get("expected_jobs") != 135
        or submitted.get("remote_result_dir") != str(REMOTE_RUN)
    ):
        raise ValueError("v19 evaluation submission is stale or incomplete")

    tasks = queue_tasks(args.queue)
    by_id = {task["id"]: task for task in tasks}
    if TASK_ID not in by_id:
        raise ValueError("scheduler queue lacks registered v19 task")
    task = by_id[TASK_ID]
    automatic_retries = sorted(
        (
            other
            for other in tasks
            if other.get("id") != TASK_ID
            and other.get("signature") == submitted.get("signature")
        ),
        key=lambda item: int(str(item["id"])[1:]),
    )
    if (
        tuple(item.get("id") for item in automatic_retries) != EXPECTED_RETRIES
        or any(
            other.get("status") != "failed"
            or other.get("exit_code") != 0
            or other.get("result_dir") != str(REMOTE_RUN)
            or other.get("result_sync_attempts") != 0
            for other in automatic_retries
        )
    ):
        raise ValueError("v19 scheduler retry-chain evidence is not closed")
    env = task.get("extra_env", {})
    required_env = submission.get("required_environment", {})
    if (
        task.get("status") != "failed"
        or task.get("last_block_reason") != EXPECTED_CLASSIFIER_REASON
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
        or env != required_env
        or "CUDA_VISIBLE_DEVICES" in env
    ):
        raise ValueError("v19 scheduler registration binding drifted")

    status_path = task.get("exit_status_path")
    log_path = task.get("log_path")
    if not status_path or not log_path:
        raise ValueError("v19 scheduler evidence paths are missing")
    process_pattern = (
        "run_h2oplus_v19_completion_safety_evaluation_dispatch.py|"
        "run_evaluation_matrix_v19_completion_safety.py|"
        "evaluate_transit_v19_completion_safety_reserve.py"
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
        raise RuntimeError(f"v19 remote completion probe failed: {failures}")
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
        or "v19_dispatch_started" not in log_text
        or "v19_inputs_verified" not in log_text
        or log_text.count("v19_block_started") != 3
        or log_text.count("v19_block_completed") != 3
        or "v19_dispatch_completed" not in log_text
        or any(
            marker in log_text.lower()
            for marker in ("traceback", "exception", "v19_dispatch_failed")
        )
        or process_text.strip()
    ):
        raise ValueError("v19 token-bound backend closure failed")

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
        raise ValueError("v19 partition inventory mismatch")
    validate_partition(local_run)
    link = install_results_link(output_root)

    captured = datetime.now(timezone.utc).isoformat()
    queue_snapshot = evidence_root / "queue_snapshot_evaluation_v2.json"
    atomic_json(
        queue_snapshot,
        {
            "schema_version": (
                "transit-h2oplus-v19-completion-safety-evaluation-queue-snapshot-v2"
            ),
            "captured_utc": captured,
            "tasks": [task, *automatic_retries],
        },
    )
    scheduler_path = evidence_root / "scheduler_evidence_v2.json"
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
                "completion_verified_from_frontend_status": False,
                "administrative_classifier_false_negative": True,
                "classifier_reason": task.get("last_block_reason"),
                "backend_and_host_evidence_override_frontend_classifier": True,
                "scientific_result_status_inferred_from_scheduler_frontend_only": False,
            },
            "scheduler_generated_retry_chain": {
                "present": True,
                "task_ids": [item["id"] for item in automatic_retries],
                "all_backend_exit0": True,
                "shared_existing_result_directory": True,
                "authorized_as_scientific_tasks": False,
                "additional_results_counted": False,
                "result_sync_attempts": 0,
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
    sync_path = evidence_root / "source_sync_v2.json"
    atomic_json(
        sync_path,
        {
            "schema_version": SYNC_SCHEMA,
            "captured_utc": captured,
            "passed": True,
            "transport": "scheduleurm-route-aware-compressed-tar-stream",
            "checksum_mode": "sha256-per-file-before-and-after-transfer",
            "formal_held_out_outcomes_read": False,
            "v19_v1_results_reused": False,
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
