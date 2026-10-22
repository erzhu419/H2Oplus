#!/usr/bin/env python3
"""Submit the single path-corrected v20 formal confirmation task."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import submit_h2oplus_v20_formal_confirmation as base


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v2"
WORKSPACE = DEPLOYMENT / "workspace"
REPO = WORKSPACE / "H2Oplus"
OVERLAY = DEPLOYMENT / "overlay"
RESULT_ROOT = DEPLOYMENT / "remote_results_v2/single_jtl110gpu"
SIGNATURE = (
    "H2Oplus/v20/formal-confirmation-v2/ad327b1a054c/"
    "single/jtl110gpu/glibc227"
)
HASHES = dict(base.HASHES)
HASHES.update(
    {
        "deployment_manifest": "ad327b1a054c90094598d36e5ba905465599868cb1088a3c786d9846963586ca",
        "deployment_aggregate": "35cae5822fba4f5381429f1267ccfa7976c9ead4387a1fb4e008276b5c21f91a",
        "local_deployment_audit": "db8e35baf7398d80da77627bdeb5099309cbbda82e1f920fcfad67cb5f753374",
        "deployment_sync": "15471a99d8637edfe026367319ce88f2b967b29a48eb5fda3161ce61a01cde80",
        "remote_verification": "3dd11854d568e9c196c36f66fe2a4cbc83508cab25d0d1f076b1a8eb5b345f8e",
        "deployment_correction": "144ce914c4cf915e181ca2f1b67711e46b7a8f6ae9d8ab97a2364bef77bdb66e",
        "incident_closeout": "4800eb9cdb792df562a1a4e55e636e84e44737ed94bcc6ca10e16f2bc549b783",
        "incident_verification": "dfe9737ee9440130d6e038db04c8ea6cb272fd11f0cb3c5325b8b32328bdb922",
    }
)
V1_PATHS = base.paths


def paths() -> dict:
    bound = V1_PATHS()
    bound.update(
        {
            "deployment_manifest": OVERLAY / "deployment_manifest.json",
            "local_deployment_audit": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_audit_v2.json",
            "deployment_sync": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_sync_v2.json",
            "remote_verification": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_jtl110gpu_verification_v2.json",
            "deployment_correction": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_path_correction_v2.json",
            "incident_closeout": REPRO
            / "h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
            "v1_deployment_path_mapping/incident_closeout_v1.json",
            "incident_verification": REPRO
            / "h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
            "v1_deployment_path_mapping/incident_verification_v1.json",
        }
    )
    return bound


def configure_base() -> None:
    base.DEPLOYMENT = DEPLOYMENT
    base.WORKSPACE = WORKSPACE
    base.REPO = REPO
    base.OVERLAY = OVERLAY
    base.RESULT_ROOT = RESULT_ROOT
    base.SIGNATURE = SIGNATURE
    base.HASHES = HASHES
    base.paths = paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument("--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_submission_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_base()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    checked = base.verify_inputs(args.queue, scheduler_root)
    correction = json.loads(checked["paths"]["deployment_correction"].read_text())
    incident = json.loads(checked["paths"]["incident_closeout"].read_text())
    incident_verification = json.loads(
        checked["paths"]["incident_verification"].read_text()
    )
    if (
        correction.get("authorization_basis", {}).get(
            "one_fresh_successor_submission_authorized"
        )
        is not True
        or correction.get("scientific_behavior_changed") is not False
        or correction.get("v1_incident_results_reused") is not False
        or incident.get("task_id") != "t84205"
        or incident.get("completed_formal_result_count") != 0
        or incident.get("remote_sumo_episode_count") != 0
        or incident.get("formal_held_out_outcomes_read") is not False
        or incident_verification.get("passed") is not True
    ):
        raise ValueError("formal v1 incident or successor authorization drifted")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite corrected submission evidence: {output}")
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(REPO),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256": (
            "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"
        ),
        "H2OPLUS_V20_FORMAL_CONFIRMATION_PROTOCOL_SHA256": HASHES["protocol"],
        "H2OPLUS_V20_FORMAL_DEPLOYMENT_CORRECTION_SHA256": HASHES[
            "deployment_correction"
        ],
    }
    submit = [
        sys.executable,
        "skill/scheduler.py",
        "submit",
        "--description",
        "H2O+ v20 path-corrected formal confirmation (1350 CPU evaluation episodes)",
        "--cmd",
        checked["command"],
        "--cwd",
        str(REPO),
        "--signature",
        SIGNATURE,
        "--resource-family",
        "h2oplus-v20-formal-confirmation-eval-worker-v2",
        "--vram",
        "0",
        "--ram-mb",
        "32768",
        "--cpu",
        "12",
        "--priority",
        "high",
        "--project",
        "h2oplus-v20-formal-confirmation",
        "--require-node",
        "jtl110gpu",
        "--allow-remote-large-data",
        "--skip-launch-staging",
        "--env",
        *(f"{key}={value}" for key, value in environment.items()),
    ]
    completed = subprocess.run(
        submit, cwd=scheduler_root, text=True, capture_output=True, check=False
    )
    match = re.search(r"submitted\s+(t\d+)", completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(
            f"corrected formal scheduler submission failed: rc={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    task_id = match.group(1)
    record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-submission-v2",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "task_count": 1,
        "prior_zero_result_incident_task_count": 1,
        "total_remote_sumo_episodes": 1350,
        "physical_behavior_count": 45,
        "formal_block_count": 10,
        "jobs_per_block": 135,
        "predesignated_candidate": "h2oplus-v19-completion-safety-quarter-load-sixty",
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_outcomes_read": False,
        "training": False,
        "scientific_sources_changed_from_v1": False,
        "v1_incident_results_reused": False,
        "v1_task_restarted": False,
        "scheduler_result_sync_opted_in": False,
        "result_sync_mode": "dedicated-frozen-allowlist-only",
        "checkpoint_optimizer_replay_or_workspace_sync_authorized": False,
        "task": {
            "task_id": task_id,
            "signature": SIGNATURE,
            "host": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "est_vram_mb": 0,
            "expected_jobs": 1350,
            "remote_result_dir": str(RESULT_ROOT),
            "command": checked["command"],
            "success_marker": "DONE",
        },
        "prior_incident_task": {
            "task_id": "t84205",
            "completed_formal_result_count": 0,
            "remote_sumo_episode_count": 0,
            "closeout_sha256": HASHES["incident_closeout"],
            "verification_sha256": HASHES["incident_verification"],
        },
        "bindings": {
            name: {"path": str(checked["paths"][name]), "sha256": digest}
            for name, digest in HASHES.items()
            if name in checked["paths"]
        },
        "deployment_aggregate_sha256": HASHES["deployment_aggregate"],
        "scheduler_stdout": completed.stdout.strip(),
        "scheduler_stderr": completed.stderr.strip(),
    }
    base.atomic_json(output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
