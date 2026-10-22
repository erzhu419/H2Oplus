#!/usr/bin/env python3
"""Submit the single JSON-safe preflight-verified v20 formal task."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import submit_h2oplus_v20_formal_confirmation as base


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v4"
WORKSPACE = DEPLOYMENT / "workspace"
REPO = WORKSPACE / "H2Oplus"
OVERLAY = DEPLOYMENT / "overlay"
RESULT_ROOT = DEPLOYMENT / "remote_results_v4/single_jtl110gpu"
SIGNATURE = (
    "H2Oplus/v20/formal-confirmation-v4/6e7c16e278fc/"
    "single/jtl110gpu/glibc227"
)
HASHES = dict(base.HASHES)
HASHES.update(
    {
        "deployment_manifest": "6e7c16e278fc5cae6097f669d5030d6e17a1f045fcbb6ce0e9be66de4058abe4",
        "deployment_aggregate": "ab8955cdafff8eadeb0d1744b2389c0649f234112f6403d8c65af9e17f5600fc",
        "deployment_manager": "19712915a5bc5a972d53754e74032f5bbca2a03c313d7dbee80aaf8776801005",
        "dispatch_wrapper": "156b5900178689956b3856319e4baf586381c6d39c8a8f0920f8732fbdb02713",
        "local_deployment_audit": "c474c66ad2a25204e00482df696d075cfc13e96d5fddc28fd0ac95b0d132aa06",
        "deployment_sync": "c144ac544bde04fcc15cbefdef907d12b86250b9a7eadb4efc4b206502c0d319",
        "remote_verification": "fc0e32872c999f617c924fde10fd0e24974a35ffc05e1437469bdd166dbd0e47",
        "deployment_correction": "6c8b33a88ef4c1ba63cd70dcef74f52be50984cc4a1f298e1cc4f2c0a1ad6504",
        "v1_incident_closeout": "4800eb9cdb792df562a1a4e55e636e84e44737ed94bcc6ca10e16f2bc549b783",
        "v1_incident_verification": "dfe9737ee9440130d6e038db04c8ea6cb272fd11f0cb3c5325b8b32328bdb922",
        "v2_incident_closeout": "90b028bdebbfbd61cd0ac1f8bb65fbd77454a1e1cf46448ec0a4a6de307c4e6e",
        "v2_incident_verification": "c45b07b8b1187165a74bd4ad5f08eb204bc567339966eefc3e0c101727f1d405",
        "v3_incident_closeout": "f8d179f425e21d66622875738578f295754fe87e2dc71435ced2aa2f2011d2e0",
        "v3_incident_verification": "324fe1396806fa1d0eba64f5d1107cb35be9721f73bf93393cb5f5a84ccb79c5",
    }
)
V1_PATHS = base.paths
V1_COMMAND = base.command


def paths() -> dict:
    bound = V1_PATHS()
    incident_root = REPRO / "h2oplus_v20_formal_confirmation_results/evaluation_incidents"
    bound.update(
        {
            "deployment_manifest": OVERLAY / "deployment_manifest.json",
            "deployment_manager": REPRO
            / "manage_h2oplus_v20_formal_confirmation_deployment_v4.py",
            "dispatch_wrapper": REPRO
            / "run_h2oplus_v20_formal_confirmation_dispatch_v4.py",
            "local_deployment_audit": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_audit_v4.json",
            "deployment_sync": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_sync_v4.json",
            "remote_verification": REPRO
            / "h2oplus_v20_formal_confirmation_deployment_jtl110gpu_verification_v4.json",
            "deployment_correction": REPRO
            / "h2oplus_v20_formal_confirmation_serialization_correction_v4.json",
            "v1_incident_closeout": incident_root
            / "v1_deployment_path_mapping/incident_closeout_v1.json",
            "v1_incident_verification": incident_root
            / "v1_deployment_path_mapping/incident_verification_v1.json",
            "v2_incident_closeout": incident_root
            / "v2_manifest_contract/incident_closeout_v2.json",
            "v2_incident_verification": incident_root
            / "v2_manifest_contract/incident_verification_v2.json",
            "v3_incident_closeout": incident_root
            / "v3_posixpath_serialization/incident_closeout_v3.json",
            "v3_incident_verification": incident_root
            / "v3_posixpath_serialization/incident_verification_v3.json",
        }
    )
    return bound


def command() -> str:
    args = shlex.split(V1_COMMAND())
    expected = str(
        REPO / "reproducibility/run_h2oplus_v20_formal_confirmation_dispatch.py"
    )
    if len(args) < 2 or args[1] != expected:
        raise ValueError(f"frozen dispatch command drifted: {args[:2]}")
    args[1] = str(
        REPO / "reproducibility/run_h2oplus_v20_formal_confirmation_dispatch_v4.py"
    )
    return shlex.join(args)


def configure_base() -> None:
    base.DEPLOYMENT = DEPLOYMENT
    base.WORKSPACE = WORKSPACE
    base.REPO = REPO
    base.OVERLAY = OVERLAY
    base.RESULT_ROOT = RESULT_ROOT
    base.SIGNATURE = SIGNATURE
    base.HASHES = HASHES
    base.paths = paths
    base.command = command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_submission_v4.json",
    )
    return parser.parse_args()


def zero_result_incident(payload: dict) -> bool:
    return (
        payload.get("completed_formal_result_count") == 0
        and payload.get("remote_sumo_episode_count") == 0
        and payload.get("formal_held_out_outcomes_read") is False
    )


def main() -> int:
    args = parse_args()
    configure_base()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    checked = base.verify_inputs(args.queue, scheduler_root)
    payload = {
        name: json.loads(checked["paths"][name].read_text())
        for name in (
            "deployment_correction",
            "remote_verification",
            "v1_incident_closeout",
            "v1_incident_verification",
            "v2_incident_closeout",
            "v2_incident_verification",
            "v3_incident_closeout",
            "v3_incident_verification",
        )
    }
    correction = payload["deployment_correction"]
    remote = payload["remote_verification"]
    v1 = payload["v1_incident_closeout"]
    v2 = payload["v2_incident_closeout"]
    v3 = payload["v3_incident_closeout"]
    v3_task_ids = [row.get("task_id") for row in v3.get("tasks", [])]
    if (
        correction.get("authorization_basis", {}).get(
            "one_fresh_successor_submission_authorized"
        )
        is not True
        or correction.get("scientific_behavior_changed") is not False
        or correction.get("v1_v2_or_v3_incident_results_reused") is not False
        or correction.get("representation_change", {}).get(
            "python38_full_immutable_evidence_json_preflight_required_before_submission"
        )
        is not True
        or v1.get("task_id") != "t84205"
        or v2.get("task_id") != "t84206"
        or v3_task_ids != ["t84210", "t84211"]
        or not all(zero_result_incident(item) for item in (v1, v2, v3))
        or any(
            payload[name].get("passed") is not True
            for name in (
                "v1_incident_verification",
                "v2_incident_verification",
                "v3_incident_verification",
            )
        )
        or remote.get("passed") is not True
        or remote.get("remote_sumo_executed") is not False
        or remote.get("preflight_result_files_created") is not False
        or remote.get("json_safe_dispatch_full_evidence_preflight", {}).get(
            "passed"
        )
        is not True
        or remote.get("json_safe_dispatch_full_evidence_preflight", {}).get(
            "immutable_evidence_json_serializable"
        )
        is not True
        or remote.get("json_safe_dispatch_full_evidence_preflight", {}).get(
            "block_count"
        )
        != 10
    ):
        raise ValueError("formal incident, JSON preflight, or successor boundary drifted")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal submission evidence: {output}")

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
        "H2O+ v20 JSON-safe formal confirmation (1350 CPU evaluation episodes)",
        "--cmd",
        checked["command"],
        "--cwd",
        str(REPO),
        "--signature",
        SIGNATURE,
        "--resource-family",
        "h2oplus-v20-formal-confirmation-eval-worker-v4",
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
            f"formal scheduler submission failed: rc={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    task_id = match.group(1)
    record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-submission-v4",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "task_count": 1,
        "prior_incident_group_count": 3,
        "prior_incident_task_count": 4,
        "total_remote_sumo_episodes": 1350,
        "physical_behavior_count": 45,
        "formal_block_count": 10,
        "jobs_per_block": 135,
        "predesignated_candidate": "h2oplus-v19-completion-safety-quarter-load-sixty",
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_outcomes_read": False,
        "training": False,
        "scientific_sources_changed_from_v1": False,
        "prior_incident_results_reused": False,
        "prior_incident_task_restarted": False,
        "json_safe_full_evidence_preflight_passed": True,
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
        "prior_incident_tasks": [
            {
                "task_ids": ["t84205"],
                "completed_formal_result_count": 0,
                "remote_sumo_episode_count": 0,
                "closeout_sha256": HASHES["v1_incident_closeout"],
                "verification_sha256": HASHES["v1_incident_verification"],
            },
            {
                "task_ids": ["t84206"],
                "completed_formal_result_count": 0,
                "remote_sumo_episode_count": 0,
                "closeout_sha256": HASHES["v2_incident_closeout"],
                "verification_sha256": HASHES["v2_incident_verification"],
            },
            {
                "task_ids": ["t84210", "t84211"],
                "completed_formal_result_count": 0,
                "remote_sumo_episode_count": 0,
                "closeout_sha256": HASHES["v3_incident_closeout"],
                "verification_sha256": HASHES["v3_incident_verification"],
            },
        ],
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
