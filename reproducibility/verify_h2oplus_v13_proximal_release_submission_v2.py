#!/usr/bin/env python3
"""Independently verify the v13 v2 scheduler submission without launching work."""

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
SUBMISSION_SHA256 = "cfd3f6454c66eabba7e2c726080456dcf4e736aa718582b86469b26bfdeb6533"
EXPECTED_IDS = {
    "t83304",
    "t83305",
    "t83306",
    "t83307",
    "t83308",
    "t83309",
    "t83314",
    "t83315",
    "t83316",
    "t83317",
    "t83318",
    "t83319",
    "t83320",
    "t83321",
    "t83322",
}
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}
EXPECTED_INPUT_HASHES = {
    "training_plan": "e67989fd379ea74fc2c3d3a6ce749b2f1b3bd3134724e0b2628c938dd777ab63",
    "protocol": "8f330bc7a55b35c9c10ae03434ddbecd30528e4529b01e734dbbd6c44bdd5f12",
    "freeze": "2a42e1b8c5c9dd16355a95cef5de032a6db706437ee38e8792b61309c1af0606",
    "pretraining_correction": "d5a0682f4511c511caeeb345494949b5325e8d73f1c69c0eed98e8f4e11c5b48",
    "deployment_manifest": "ce401b074dd83a5bd5f892d314217967a24c9e4c55b4d714784234371f195cb7",
    "base_deployment_manifest": "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd",
    "local_verification": "dedf39fa8796b6a7d9b3e239ad1fa204324bb7800bc30562c5cc20699900db07",
    "local_runtime_preflight": "29700e4c976692bf556fbf8cbe2aa4ae5d1d9881e0d7a8ceeba48035496b3986",
    "deployment_sync": "49f8a76fb04b90ee347e9e0bbbea3cdec324835b7856c36c08c78cd1f5365854",
    "remote_verification": "8fdffd8f0e29a0fbf220467378d76ae9a00f478ae019b4073052d287b80888d4",
    "remote_runtime_preflight": "60e61bd8ecfc5b6ea140e004306262a3444c0f069bbbe382b98f33302847c913",
    "submission_program": "d977e04453cf895c8f33fe0047d2fbb4936b0d5ddfc1d9f71934f520bb209e8c",
}
RESOURCE_FAMILY = "h2oplus-v13-proximal-release-branch-v2"


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


def verify_descriptor(value: dict[str, Any], expected_sha256: str) -> Path:
    path = Path(value["path"]).expanduser().resolve(strict=True)
    if (
        path.stat().st_size != value["bytes"]
        or sha256(path) != value["sha256"]
        or value["sha256"] != expected_sha256
    ):
        raise ValueError(f"submission descriptor drifted: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_submission_v2.json",
    )
    parser.add_argument(
        "--scheduler-state",
        type=Path,
        default=Path.home() / ".claude/scheduler/queue.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / "h2oplus_v13_proximal_release_submission_verification_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v13 v2 submission record drifted")
    submission = json.loads(submission_path.read_text())
    tasks = submission.get("tasks", [])
    task_ids = {row.get("task_id") for row in tasks}
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-submission-v2"
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
        or any(
            row.get("returncode") != 0
            for node in submission.get("remote_result_root_absence", [])
            for row in node.get("attempts", [])[-1:]
        )
        or submission.get("development_evaluation_submitted") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v13 v2 submission scope or task identities drifted")

    descriptor_labels = tuple(EXPECTED_INPUT_HASHES)
    paths = {
        label: verify_descriptor(submission[label], EXPECTED_INPUT_HASHES[label])
        for label in descriptor_labels
    }
    plan = json.loads(paths["training_plan"].read_text())
    jobs = {job["name"]: job for job in plan["jobs"]}
    if set(jobs) != {row["name"] for row in tasks}:
        raise ValueError("v13 v2 submitted names differ from the frozen plan")

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
            raise ValueError(f"v13 v2 scheduler task is missing: {task_id}")
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
            or scheduler_row.get("project") != "h2oplus-v13-development-v2"
            or scheduler_row.get("cmd") != expected_command
            or scheduler_row.get("cwd") != str(workspace)
            or scheduler_row.get("require_node") != NODE_BY_SEED[seed]
            or scheduler_row.get("cpu_cores") != 6
            or scheduler_row.get("ram_mb") != 32768
            or scheduler_row.get("est_vram_mb") != 8192
            or scheduler_row.get("resource_family") != RESOURCE_FAMILY
            or scheduler_row.get("vram_resource_family") != RESOURCE_FAMILY
            or scheduler_row.get("ram_resource_family") != RESOURCE_FAMILY
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
            or "--admission-preflight-only" in scheduler_row.get("cmd", "")
            or status not in {"queued", "running", "syncing", "completed"}
        ):
            raise ValueError(f"v13 v2 scheduler contract drifted: {task_id}")
        if status == "running" and (
            scheduler_row.get("node") != NODE_BY_SEED[seed]
            or not isinstance(scheduler_row.get("gpu_idx"), int)
            or not scheduler_row.get("remote_pids")
        ):
            raise ValueError(f"v13 v2 running placement is incomplete: {task_id}")
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
        "schema_version": "transit-h2oplus-v13-proximal-release-submission-verification-v2",
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
