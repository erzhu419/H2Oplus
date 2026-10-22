#!/usr/bin/env python3
"""Submit exactly five frozen v12 shared-prefix development jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v12-staged-release-prefix-submission-v2"
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}
EXPECTED = {
    "plan": "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa",
    "protocol": "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
    "manifest": "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9",
    "deployment_correction": (
        "40dabd0dddab6ce7fd8f08ac025f8cc96a5545d1c36ff24730ce885b5eb67b25"
    ),
    "preflight_closeout": (
        "24aa2a30ef34c6e79cc567995b427ec8db94c5eaa303bb72a595caf6891b8e80"
    ),
    "preflight_verification": (
        "75fc5357ba271d8cb25affc6b1b2a0d35f8600a81940ba5562d409e5c4c2ed03"
    ),
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def existing_signatures(queue_path: Path, archive_path: Path) -> set[str]:
    payload = json.loads(queue_path.read_text())
    raw = payload if isinstance(payload, list) else payload.get("tasks", [])
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    signatures = {row.get("signature") for row in rows if row.get("signature")}
    if archive_path.exists():
        for line in archive_path.read_text().splitlines():
            normalized = line.strip("\x00 \t\r\n")
            if not normalized:
                continue
            row = json.loads(normalized)
            if row.get("signature"):
                signatures.add(row["signature"])
    return signatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v12_staged_release_pilot_v2",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_prefix_submission_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate v12 prefix submission: {record_path}")
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler_cli = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v12_control_v2"
    plan_path = control / "training_plan.json"
    protocol_path = control / "protocol.json"
    correction_path = control / "deployment_correction.json"
    manifest = workspace / "deployment_manifest_v12_v2.json"
    base_manifest = workspace / "deployment_manifest.json"
    closeout_path = REPRO / "h2oplus_v12_staged_release_preflight_closeout_v4.json"
    verification_path = (
        REPRO / "h2oplus_v12_staged_release_preflight_closeout_verification_v4.json"
    )
    bound = {
        "plan": plan_path,
        "protocol": protocol_path,
        "manifest": manifest,
        "deployment_correction": correction_path,
        "preflight_closeout": closeout_path,
        "preflight_verification": verification_path,
    }
    for name, path in bound.items():
        if sha256(path) != EXPECTED[name]:
            raise ValueError(f"frozen v12 prefix gate drifted: {name}")

    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    closeout = json.loads(closeout_path.read_text())
    verification = json.loads(verification_path.read_text())
    if (
        plan.get("schema_version")
        != "transit-h2oplus-v12-staged-release-training-plan-v2"
        or plan.get("status") != "frozen-before-v12-training"
        or plan.get("formal_held_out_outcomes_read") is not False
        or protocol.get("schema_version")
        != "transit-h2oplus-v12-staged-release-protocol-v2"
        or protocol.get("status")
        != "frozen-before-v12-training-and-development-outcomes"
        or protocol.get("local_training_or_sumo_allowed") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or plan.get("jobs") != protocol.get("training_jobs")
        or len(plan.get("jobs", [])) != 30
        or closeout.get("status")
        != "passed-with-preserved-scheduler-terminal-correction"
        or closeout.get("prefix_training_submission_gate_open") is not True
        or closeout.get("branch_training_submission_gate_open") is not False
        or closeout.get("development_evaluation_gate_open") is not False
        or closeout.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or verification.get("status") != "passed"
        or verification.get("exact_remote_verifications") != 2
        or verification.get("prefix_training_submission_gate_open") is not True
        or verification.get("branch_training_submission_gate_open") is not False
        or verification.get("development_evaluation_gate_open") is not False
        or verification.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or verification.get("closeout", {}).get("sha256") != sha256(closeout_path)
    ):
        raise ValueError("v12 corrected prefix release gate is closed")

    jobs = [
        job
        for job in plan["jobs"]
        if job.get("stage") == "prefix" and job.get("variant") == "shared-prefix"
    ]
    if (
        len(jobs) != 5
        or {int(job["seed"]) for job in jobs} != set(NODE_BY_SEED)
        or len({job["name"] for job in jobs}) != 5
        or len({job["run_signature"] for job in jobs}) != 5
        or any("--allow-no-resume" in job.get("command_arguments", []) for job in jobs)
    ):
        raise ValueError("v12 shared-prefix matrix is not exact")

    signatures = {
        job["name"]: (
            f"H2Oplus/v12/development/staged-release-v2/shared-prefix/"
            f"seed{job['seed']}/{job['run_signature'][:16]}"
        )
        for job in jobs
    }
    state_dir = Path.home() / ".claude/scheduler"
    existing = existing_signatures(
        state_dir / "queue.json", state_dir / "queue_archive.jsonl"
    )
    duplicates = sorted(set(signatures.values()) & existing)
    if duplicates:
        raise FileExistsError(f"v12 prefix signatures already exist: {duplicates}")

    sys.path.insert(0, str(scheduler_root / "skill"))
    import scheduler as scheduler_runtime  # type: ignore  # noqa: PLC0415

    jobs_by_node = {
        node: [job for job in jobs if NODE_BY_SEED[int(job["seed"])] == node]
        for node in sorted(set(NODE_BY_SEED.values()))
    }
    remote_absence = []
    for node, node_jobs in jobs_by_node.items():
        statements = ["found=0"]
        for job in node_jobs:
            run_dir = Path(plan["remote_output_root"]) / job["name"]
            statements.append(
                f"if test -e {shlex.quote(str(run_dir))}; then "
                f"printf 'EXISTS\\t%s\\n' {shlex.quote(job['name'])}; found=1; "
                f"else printf 'ABSENT\\t%s\\n' {shlex.quote(job['name'])}; fi"
            )
        statements.append('exit "$found"')
        command = "; ".join(statements)
        attempts = []
        for attempt in range(1, 4):
            rc, stdout, stderr = scheduler_runtime.run_on(
                node, command, timeout=30, check=False
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": rc,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            if rc in {0, 1}:
                break
            if attempt < 3:
                time.sleep(5 * attempt)
        remote_absence.append(
            {
                "node": node,
                "run_names": [job["name"] for job in node_jobs],
                "attempts": attempts,
                "returncode": rc,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        if rc == 1:
            raise FileExistsError(f"v12 prefix result root already exists on {node}")
        if rc != 0:
            raise ConnectionError(
                f"v12 prefix root absence check failed on {node}: {stderr or stdout}"
            )

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
        "H2OPLUS_V12_WORKSPACE_ROOT": str(workspace),
        "H2OPLUS_V12_PROTOCOL": str(protocol_path),
        "H2OPLUS_V12_PROTOCOL_SHA256": sha256(protocol_path),
        "H2OPLUS_V12_BASE_DEPLOYMENT_MANIFEST": str(base_manifest),
        "H2OPLUS_V12_BASE_DEPLOYMENT_MANIFEST_SHA256": sha256(base_manifest),
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v12_staged_release.py"
    submitted = []
    for job in jobs:
        node = NODE_BY_SEED[int(job["seed"])]
        run_dir = Path(plan["remote_output_root"]) / job["name"]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        scheduler_signature = signatures[job["name"]]
        submit = [
            "python3",
            str(scheduler_cli),
            "submit",
            "--description",
            f"H2O+ v12 shared-prefix development seed {job['seed']} to 10k events",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            scheduler_signature,
            "--resource-family",
            "h2oplus-v12-staged-release-prefix-v2",
            "--vram-resource-family",
            "h2oplus-v12-staged-release-prefix-v2",
            "--ram-resource-family",
            "h2oplus-v12-staged-release-prefix-v2",
            "--vram",
            "8192",
            "--ram-mb",
            "32768",
            "--cpu",
            "6",
            "--priority",
            "high",
            "--project",
            "h2oplus-v12-development-v2",
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
        submitted.append(
            {
                "name": job["name"],
                "variant": job["variant"],
                "stage": job["stage"],
                "seed": job["seed"],
                "run_signature": job["run_signature"],
                "scheduler_signature": scheduler_signature,
                "require_node": node,
                "result_root": str(run_dir),
                "returncode": result.returncode,
                "task_id": match.group(0) if match else None,
                "scheduler_output": combined.strip(),
            }
        )
        if result.returncode or match is None:
            break

    passed = len(submitted) == 5 and all(
        row["returncode"] == 0 and row["task_id"] for row in submitted
    )
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "partial-failure",
        "study_stage": "development",
        "training_plan": descriptor(plan_path),
        "protocol": descriptor(protocol_path),
        "deployment_correction": descriptor(correction_path),
        "deployment_manifest": descriptor(manifest),
        "preflight_closeout": descriptor(closeout_path),
        "preflight_independent_verification": descriptor(verification_path),
        "remote_result_root_absence": remote_absence,
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "exactly one physical GPU assigned by scheduleurm"
        },
        "tasks": submitted,
        "submitted_prefix_tasks": len(submitted),
        "submitted_branch_tasks": 0,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
