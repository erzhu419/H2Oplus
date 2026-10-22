#!/usr/bin/env python3
"""Independently verify the v13 scheduler submission without launching work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SUBMISSION_SHA256 = "2782fbbd02e618c536464fe87620591b2f8d180fbd3bc4e41e91bdaba481f277"
EXPECTED_IDS = {f"t{value}" for value in range(83252, 83267)}
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_descriptor(value: dict[str, Any]) -> Path:
    path = Path(value["path"]).expanduser().resolve(strict=True)
    if path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]:
        raise ValueError(f"submission descriptor drifted: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_submission_v1.json",
    )
    parser.add_argument(
        "--scheduler-state",
        type=Path,
        default=Path.home() / ".claude/scheduler/queue.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_submission_verification_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v13 submission record drifted")
    submission = json.loads(submission_path.read_text())
    tasks = submission.get("tasks", [])
    task_ids = {row.get("task_id") for row in tasks}
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-submission-v1"
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "development"
        or submission.get("prefix_training_rerun") is not False
        or submission.get("submitted_prefix_tasks") != 0
        or submission.get("submitted_branch_tasks") != 15
        or len(tasks) != 15
        or task_ids != EXPECTED_IDS
        or len({row.get("scheduler_signature") for row in tasks}) != 15
        or len({row.get("run_signature") for row in tasks}) != 15
        or any(row.get("returncode") != 0 for row in tasks)
        or submission.get("development_evaluation_submitted") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v13 submission scope or task identities drifted")

    descriptor_labels = (
        "training_plan",
        "protocol",
        "freeze",
        "deployment_manifest",
        "base_deployment_manifest",
        "local_verification",
        "deployment_sync",
        "remote_verification",
        "submission_program",
    )
    paths = {label: verify_descriptor(submission[label]) for label in descriptor_labels}
    plan = json.loads(paths["training_plan"].read_text())
    jobs = {job["name"]: job for job in plan["jobs"]}
    if set(jobs) != {row["name"] for row in tasks}:
        raise ValueError("v13 submitted names differ from the frozen plan")

    queue_path = args.scheduler_state.expanduser().resolve(strict=True)
    queue = json.loads(queue_path.read_text())
    raw = queue.get("tasks", queue)
    rows = {row["id"]: row for row in raw} if isinstance(raw, list) else raw
    environment = submission["required_environment"]
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    workspace = paths["deployment_manifest"].parent
    wrapper = workspace / "SimpleSAC/train_transit_v13_proximal_release.py"
    snapshots = []
    for submitted in tasks:
        task_id = submitted["task_id"]
        scheduler_row = rows.get(task_id)
        if scheduler_row is None:
            raise ValueError(f"v13 scheduler task is missing: {task_id}")
        job = jobs[submitted["name"]]
        seed = int(job["seed"])
        run_dir = Path(submitted["result_root"])
        expected_command = shlex.join(
            [python, str(wrapper), *job["command_arguments"]]
        )
        status = scheduler_row.get("status")
        if (
            submitted.get("seed") != seed
            or submitted.get("variant") != job["variant"]
            or submitted.get("proximal_rate") != job["proximal_rate"]
            or submitted.get("predesignated_candidate")
            != job["predesignated_candidate"]
            or submitted.get("branch_handoff_manifest")
            != job["branch_handoff_manifest"]
            or submitted.get("source_handoff") != job["source_handoff"]
            or submitted.get("require_node") != NODE_BY_SEED[seed]
            or submitted.get("declared_cpu_cores") != 6
            or submitted.get("declared_ram_mb") != 32768
            or submitted.get("declared_vram_mb") != 8192
            or scheduler_row.get("signature") != submitted["scheduler_signature"]
            or scheduler_row.get("project") != "h2oplus-v13-development-v1"
            or scheduler_row.get("cmd") != expected_command
            or scheduler_row.get("cwd") != str(workspace)
            or scheduler_row.get("require_node") != NODE_BY_SEED[seed]
            or scheduler_row.get("cpu_cores") != 6
            or scheduler_row.get("ram_mb") != 32768
            or scheduler_row.get("est_vram_mb") != 8192
            or scheduler_row.get("ckpt_dir") != str(run_dir / "checkpoints")
            or scheduler_row.get("result_dir") != str(run_dir)
            or scheduler_row.get("local_result_dir") != str(run_dir)
            or scheduler_row.get("extra_env") != environment
            or "CUDA_VISIBLE_DEVICES" in scheduler_row.get("extra_env", {})
            or scheduler_row.get("resume_flag") != "--resume"
            or scheduler_row.get("allow_initial_resume_scan_error") is not True
            or scheduler_row.get("skip_launch_staging") is not True
            or scheduler_row.get("allow_remote_large_data") is not True
            or scheduler_row.get("env_spec") != "none"
            or "--allow-no-resume" in scheduler_row.get("cmd", "")
            or status not in {"queued", "running", "syncing", "completed"}
        ):
            raise ValueError(f"v13 scheduler contract drifted: {task_id}")
        if status == "running" and (
            scheduler_row.get("node") != NODE_BY_SEED[seed]
            or not isinstance(scheduler_row.get("gpu_idx"), int)
            or not scheduler_row.get("remote_pids")
        ):
            raise ValueError(f"v13 running placement is incomplete: {task_id}")
        snapshots.append(
            {
                "task_id": task_id,
                "name": submitted["name"],
                "status": status,
                "node": scheduler_row.get("node"),
                "gpu_idx": scheduler_row.get("gpu_idx"),
                "remote_pids": scheduler_row.get("remote_pids", []),
                "dispatch_attempt_count": scheduler_row.get("dispatch_attempt_count"),
                "last_block_reason": scheduler_row.get("last_block_reason"),
            }
        )

    payload = {
        "schema_version": "transit-h2oplus-v13-proximal-release-submission-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "submission": {
            "path": str(submission_path),
            "bytes": submission_path.stat().st_size,
            "sha256": sha256(submission_path),
        },
        "verified_task_count": len(snapshots),
        "verified_task_ids": sorted(task_ids, key=lambda value: int(value[1:])),
        "scheduler_snapshot": snapshots,
        "queue_snapshot_path": str(queue_path),
        "queue_snapshot_mtime_ns": queue_path.stat().st_mtime_ns,
        "prefix_training_rerun": False,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "verification_executed_training_or_sumo": False,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
