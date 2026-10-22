#!/usr/bin/env python3
"""Submit the corrected jtl110gpu2-only v10 deployment verification."""

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
SCHEMA = "transit-h2oplus-v10-delayed-preflight-submission-v3"
NODE = "jtl110gpu2"


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
        "--correction",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_preflight_correction_v2.json",
    )
    parser.add_argument(
        "--staging-probe",
        type=Path,
        default=REPRO
        / "h2oplus_v10_delayed_preflight_staging_probe_jtl110gpu2_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_preflight_submission_v3.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(
            f"refusing to duplicate corrected v10 preflight: {record_path}"
        )
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    manifest = workspace / "deployment_manifest_v10.json"
    manifest_hash = sha256(manifest)
    local_verification_path = args.local_verification.expanduser().resolve(strict=True)
    local_verification = json.loads(local_verification_path.read_text())
    correction_path = args.correction.expanduser().resolve(strict=True)
    correction = json.loads(correction_path.read_text())
    probe_path = args.staging_probe.expanduser().resolve(strict=True)
    probe = json.loads(probe_path.read_text())
    prior_submission_path = REPRO / "h2oplus_v10_delayed_preflight_submission_v2.json"
    if (
        local_verification.get("passed") is not True
        or local_verification.get("deployment_manifest_sha256") != manifest_hash
        or correction.get("status")
        != "failed-staging-sealed-fresh-verification-authorized"
        or correction.get("failed_task", {}).get("task_id") != "t82612"
        or correction.get("failed_task", {}).get("started_at") is not None
        or correction.get("failed_task_must_never_be_requeued") is not True
        or correction.get("fresh_task_must_use_skip_launch_staging") is not True
        or correction.get("prior_submission_sha256")
        != sha256(prior_submission_path)
        or correction.get("staging_probe_sha256") != sha256(probe_path)
        or correction.get("deployment_manifest_sha256") != manifest_hash
        or probe.get("passed") is not True
        or probe.get("purpose") != "staging-completeness-only"
        or probe.get("scheduler_node") != NODE
        or probe.get("deployment_manifest_sha256") != manifest_hash
        or probe.get("overlay_entries") != 46
        or probe.get("formal_held_out_outcomes_read") is not False
        or probe.get("local_training_or_sumo_executed") is not False
    ):
        raise ValueError("v10 jtl110gpu2 staging correction is stale or invalid")

    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    verifier = workspace / "reproducibility/verify_h2oplus_v10_delayed_deployment.py"
    result_root = deployment_root / "preflight_results_v3" / NODE
    output = result_root / "verification.json"
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
    }
    command = shlex.join(
        [python, str(verifier), "--workspace", str(workspace), "--output", str(output)]
    )
    signature = (
        "H2Oplus/v10/development/delayed-v1/preflight-v3/"
        f"{NODE}/{manifest_hash[:16]}"
    )
    submit = [
        "python3",
        str(scheduler),
        "submit",
        "--description",
        "H2O+ v10 delayed corrected no-SUMO preflight on jtl110gpu2",
        "--cmd",
        command,
        "--cwd",
        str(workspace),
        "--signature",
        signature,
        "--resource-family",
        "h2oplus-v10-delayed-preflight-v1",
        "--vram",
        "0",
        "--ram-mb",
        "8192",
        "--cpu",
        "1",
        "--priority",
        "high",
        "--project",
        "h2oplus-v10-development-v1",
        "--require-node",
        NODE,
        "--result-dir",
        str(result_root),
        "--local-result-dir",
        str(result_root),
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
    passed = result.returncode == 0 and match is not None
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "failed",
        "study_stage": "development",
        "purpose": "replacement-deployment-verification-only",
        "supersedes_failed_task": "t82612",
        "failed_task_requeued": False,
        "remote_training_or_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "deployment_manifest": descriptor(manifest),
        "local_verification": descriptor(local_verification_path),
        "correction": descriptor(correction_path),
        "staging_probe": descriptor(probe_path),
        "prior_submission": descriptor(prior_submission_path),
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "empty for the CPU-only task; injected by scheduleurm"
        },
        "task": {
            "node": NODE,
            "returncode": result.returncode,
            "task_id": match.group(0) if match else None,
            "signature": signature,
            "result_root": str(result_root),
            "verification_output": str(output),
            "scheduler_output": combined.strip(),
        },
    }
    _write(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
