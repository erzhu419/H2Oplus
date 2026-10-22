#!/usr/bin/env python3
"""Submit the corrected frozen v9 adaptation pilot v2 through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v9-adaptive-submission-v2"


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
        default=REPRO / "deployments/h2oplus_v9_adaptive_pilot_v2",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_deployment_verification_v2_local.json",
    )
    parser.add_argument(
        "--remote-verification-jtl110gpu",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_deployment_verification_v2_jtl110gpu.json",
    )
    parser.add_argument(
        "--remote-verification-jtl110gpu2",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_deployment_verification_v2_jtl110gpu2.json",
    )
    return parser.parse_args()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v9_control"
    plan_path = control / "training_plan.json"
    protocol_path = control / "protocol.json"
    correction_path = control / "correction.json"
    deployment_path = workspace / "deployment_manifest_v9.json"
    base_manifest = workspace / "deployment_manifest.json"
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    correction = json.loads(correction_path.read_text())
    verification_paths = (
        args.local_verification,
        args.remote_verification_jtl110gpu,
        args.remote_verification_jtl110gpu2,
    )
    for path in verification_paths:
        payload = json.loads(path.expanduser().resolve(strict=True).read_text())
        if (
            payload.get("passed") is not True
            or payload.get("deployment_manifest_sha256") != sha256(deployment_path)
            or payload.get("formal_held_out_outcomes_read") is not False
            or payload.get("local_training_or_sumo_executed") is not False
        ):
            raise ValueError(f"deployment verification is stale or failed: {path}")
    if (
        plan.get("jobs") != protocol.get("training_jobs")
        or len(plan.get("jobs", [])) != 9
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or plan.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or correction.get("status")
        != "v1-invalid-sealed-v2-correction-authorized"
    ):
        raise ValueError("frozen v9 protocol and training plan disagree")

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
        "H2OPLUS_V9_WORKSPACE_ROOT": str(workspace),
        "H2OPLUS_V9_PROTOCOL": str(protocol_path),
        "H2OPLUS_V9_PROTOCOL_SHA256": sha256(protocol_path),
        "H2OPLUS_V9_BASE_DEPLOYMENT_MANIFEST": str(base_manifest),
        "H2OPLUS_V9_BASE_DEPLOYMENT_MANIFEST_SHA256": sha256(base_manifest),
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v9_adaptive.py"
    submitted = []
    for job in plan["jobs"]:
        node = "jtl110gpu2" if int(job["seed"]) == 6202 else "jtl110gpu"
        run_dir = Path(plan["remote_output_root"]) / job["name"]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        scheduler_signature = (
            f"H2Oplus/v9/development/adaptive-v2/{job['variant']}/"
            f"seed{job['seed']}/{job['run_signature'][:16]}"
        )
        submit = [
            "python3",
            str(scheduler),
            "submit",
            "--description",
            f"H2O+ v9 corrected-v2 development {job['variant']} seed {job['seed']} to 25k events",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            scheduler_signature,
            "--resource-family",
            "h2oplus-v9-adaptive-worker-v2",
            "--vram-resource-family",
            "h2oplus-v9-adaptive-worker-v2",
            "--ram-resource-family",
            "h2oplus-v9-adaptive-worker-v2",
            "--vram",
            "8192",
            "--ram-mb",
            "32768",
            "--cpu",
            "6",
            "--priority",
            "high",
            "--project",
            "h2oplus-v9-development-v2",
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
        combined = "\n".join(value for value in (result.stdout, result.stderr) if value)
        match = re.search(r"\bt\d+\b", combined)
        record = {
            "name": job["name"],
            "variant": job["variant"],
            "seed": job["seed"],
            "run_signature": job["run_signature"],
            "scheduler_signature": scheduler_signature,
            "require_node": node,
            "returncode": result.returncode,
            "task_id": match.group(0) if match else None,
            "scheduler_output": combined.strip(),
        }
        submitted.append(record)
        if result.returncode or record["task_id"] is None:
            break

    record_path = REPRO / "h2oplus_v9_adaptive_submission_record_v2.json"
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if len(submitted) == 9 and all(not item["returncode"] for item in submitted) else "partial-failure",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_plan": descriptor(plan_path),
        "protocol": descriptor(protocol_path),
        "correction": descriptor(correction_path),
        "deployment_manifest": descriptor(deployment_path),
        "deployment_verifications": [
            descriptor(path.expanduser().resolve()) for path in verification_paths
        ],
        "required_environment": environment,
        "tasks": submitted,
    }
    _write(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
