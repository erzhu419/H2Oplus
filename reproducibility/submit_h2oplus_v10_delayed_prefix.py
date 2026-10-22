#!/usr/bin/env python3
"""Submit only the five frozen v10 shared-prefix jobs through scheduleurm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v10-delayed-prefix-submission-v1"
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}
HOSTNAME_BY_NODE = {
    "jtl110gpu": "huiwei-Super-Server",
    "jtl110gpu2": "erzhu419-Super-Server",
}


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
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root",
        type=Path,
        default=ROOT.parent.parent / "scheduleurm",
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v10_delayed_pilot_v1",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_deployment_verification_v1_local.json",
    )
    parser.add_argument(
        "--preflight-submission",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_preflight_submission_v2.json",
    )
    parser.add_argument(
        "--preflight-replacement-submission",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_preflight_submission_v3.json",
    )
    parser.add_argument(
        "--preflight-closeout",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_preflight_closeout_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_prefix_submission_v1.json",
    )
    return parser.parse_args()


def _verify_deployment_result(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_host: str | None,
) -> None:
    payload = json.loads(path.expanduser().resolve(strict=True).read_text())
    if (
        payload.get("passed") is not True
        or payload.get("deployment_manifest_sha256") != expected_manifest_sha256
        or payload.get("overlay_entries") != 46
        or payload.get("prefix_jobs") != 5
        or payload.get("branch_jobs") != 25
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("local_training_or_sumo_executed") is not False
        or (
            expected_host is not None
            and payload.get("verification_host") != expected_host
        )
    ):
        raise ValueError(f"stale or failed v10 deployment verification: {path}")


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing to duplicate v10 prefix submission: {record_path}")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v10_control"
    plan_path = control / "training_plan.json"
    protocol_path = control / "protocol.json"
    deployment_path = workspace / "deployment_manifest_v10.json"
    base_manifest = workspace / "deployment_manifest.json"
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    manifest_sha256 = sha256(deployment_path)

    local_verification = args.local_verification.expanduser().resolve(strict=True)
    remote_verifications = {
        "jtl110gpu": deployment_root
        / "preflight_results_v2/jtl110gpu/verification.json",
        "jtl110gpu2": deployment_root
        / "preflight_results_v3/jtl110gpu2/verification.json",
    }
    _verify_deployment_result(
        local_verification,
        expected_manifest_sha256=manifest_sha256,
        expected_host=None,
    )
    for node, path in remote_verifications.items():
        _verify_deployment_result(
            path,
            expected_manifest_sha256=manifest_sha256,
            expected_host=HOSTNAME_BY_NODE[node],
        )

    preflight_path = args.preflight_submission.expanduser().resolve(strict=True)
    preflight = json.loads(preflight_path.read_text())
    preflight_tasks = preflight.get("tasks", [])
    if (
        preflight.get("status") != "submitted"
        or preflight.get("deployment_manifest", {}).get("sha256") != manifest_sha256
        or preflight.get("remote_training_or_sumo_executed") is not False
        or preflight.get("formal_held_out_outcomes_read") is not False
        or len(preflight_tasks) != 2
        or {task.get("node") for task in preflight_tasks} != set(NODE_BY_SEED.values())
        or any(not task.get("task_id") for task in preflight_tasks)
    ):
        raise ValueError("v10 deployment preflight submission is stale or incomplete")
    replacement_path = (
        args.preflight_replacement_submission.expanduser().resolve(strict=True)
    )
    replacement = json.loads(replacement_path.read_text())
    closeout_path = args.preflight_closeout.expanduser().resolve(strict=True)
    closeout = json.loads(closeout_path.read_text())
    if (
        replacement.get("status") != "submitted"
        or replacement.get("supersedes_failed_task") != "t82612"
        or replacement.get("failed_task_requeued") is not False
        or replacement.get("task", {}).get("task_id") != "t82616"
        or replacement.get("deployment_manifest", {}).get("sha256")
        != manifest_sha256
        or closeout.get("status") != "passed"
        or closeout.get("deployment_manifest_sha256") != manifest_sha256
        or {task.get("task_id") for task in closeout.get("valid_tasks", [])}
        != {"t82611", "t82616"}
        or closeout.get("invalid_incident_task", {}).get("task_id") != "t82612"
        or closeout.get("invalid_incident_task", {}).get("must_never_be_requeued")
        is not True
        or closeout.get("exact_remote_to_local_sha256_match") is not True
        or closeout.get("prefix_submission_gate_open") is not True
        or closeout.get("branch_submission_gate_open") is not False
        or closeout.get("remote_training_or_sumo_executed") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v10 preflight replacement or closeout is stale")

    prefix_jobs = [job for job in plan.get("jobs", []) if job.get("stage") == "prefix"]
    branch_jobs = [job for job in plan.get("jobs", []) if job.get("stage") == "branch"]
    if (
        plan.get("jobs") != protocol.get("training_jobs")
        or len(prefix_jobs) != 5
        or len(branch_jobs) != 25
        or {int(job["seed"]) for job in prefix_jobs} != set(NODE_BY_SEED)
        or {job.get("variant") for job in prefix_jobs} != {"shared-prefix"}
        or any(job.get("online_event_cap") != 10000 for job in prefix_jobs)
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("staged_release", {}).get("prefix_jobs") != 5
        or protocol.get("staged_release", {}).get("branch_jobs") != 25
    ):
        raise ValueError("frozen v10 protocol and staged training plan disagree")

    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": str(
            workspace
            / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(workspace),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "H2OPLUS_V10_WORKSPACE_ROOT": str(workspace),
        "H2OPLUS_V10_PROTOCOL": str(protocol_path),
        "H2OPLUS_V10_PROTOCOL_SHA256": sha256(protocol_path),
        "H2OPLUS_V10_BASE_DEPLOYMENT_MANIFEST": str(base_manifest),
        "H2OPLUS_V10_BASE_DEPLOYMENT_MANIFEST_SHA256": sha256(base_manifest),
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v10_delayed.py"
    submitted = []
    for job in prefix_jobs:
        seed = int(job["seed"])
        node = NODE_BY_SEED[seed]
        run_dir = Path(plan["remote_output_root"]) / job["name"]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        scheduler_signature = (
            f"H2Oplus/v10/development/delayed-v1/prefix/seed{seed}/"
            f"{job['run_signature'][:16]}"
        )
        submit = [
            "python3",
            str(scheduler),
            "submit",
            "--description",
            f"H2O+ v10 shared prefix seed {seed} to 10k development events",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            scheduler_signature,
            "--resource-family",
            "h2oplus-v10-delayed-worker-v1",
            "--vram-resource-family",
            "h2oplus-v10-delayed-worker-v1",
            "--ram-resource-family",
            "h2oplus-v10-delayed-worker-v1",
            "--vram",
            "8192",
            "--ram-mb",
            "32768",
            "--cpu",
            "6",
            "--priority",
            "high",
            "--project",
            "h2oplus-v10-development-v1",
            "--require-node",
            node,
            "--ckpt-dir",
            str(run_dir / "checkpoints"),
            "--result-dir",
            str(run_dir),
            "--local-result-dir",
            str(run_dir),
            "--ckpt-glob",
            "latest.pt",
            "--resume-flag=--resume",
            "--allow-initial-resume-scan-error",
            "--env-spec",
            "none",
            "--allow-remote-large-data",
            "--skip-launch-staging",
            "--env",
            *[f"{key}={value}" for key, value in environment.items()],
        ]
        result = subprocess.run(
            submit,
            cwd=scheduler_root,
            text=True,
            capture_output=True,
            check=False,
        )
        combined = "\n".join(
            value for value in (result.stdout, result.stderr) if value
        )
        match = re.search(r"\bt\d+\b", combined)
        item = {
            "name": job["name"],
            "stage": job["stage"],
            "variant": job["variant"],
            "seed": seed,
            "run_signature": job["run_signature"],
            "scheduler_signature": scheduler_signature,
            "require_node": node,
            "returncode": result.returncode,
            "task_id": match.group(0) if match else None,
            "scheduler_output": combined.strip(),
        }
        submitted.append(item)
        if result.returncode or item["task_id"] is None:
            break

    passed = len(submitted) == 5 and all(
        item["returncode"] == 0 and item["task_id"] for item in submitted
    )
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "partial-failure",
        "study_stage": "development",
        "submitted_stage": "prefix-only",
        "branch_training_submitted": False,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "training_plan": descriptor(plan_path),
        "protocol": descriptor(protocol_path),
        "deployment_manifest": descriptor(deployment_path),
        "preflight_submission": descriptor(preflight_path),
        "preflight_replacement_submission": descriptor(replacement_path),
        "preflight_closeout": descriptor(closeout_path),
        "preflight_task_ids": ["t82611", "t82616"],
        "deployment_verifications": [
            descriptor(local_verification),
            *[descriptor(path) for path in remote_verifications.values()],
        ],
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "exactly one physical GPU per task"
        },
        "tasks": submitted,
    }
    _write(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
