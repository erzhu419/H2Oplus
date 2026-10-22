#!/usr/bin/env python3
"""Submit exactly five frozen v11 actor-hold jobs through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v11-actor-hold-submission-v2"
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


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def existing_signatures(queue_path: Path, archive_path: Path) -> set[str]:
    payload = json.loads(queue_path.read_text())
    tasks = payload.get("tasks", [])
    rows = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
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


def require_remote_run_root_absent(
    *,
    node: str,
    run_name: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    if returncode == 0:
        return
    if returncode == 1:
        raise FileExistsError(
            f"remote v11 run root already exists for {run_name} on {node}"
        )
    detail = (stderr or stdout or "no transport diagnostic").strip()
    raise ConnectionError(
        f"remote v11 run-root absence could not be verified on {node}: "
        f"rc={returncode}: {detail}"
    )


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
        default=REPRO / "deployments/h2oplus_v11_actor_hold_pilot_v2",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_deployment_verification_local_v2.json",
    )
    parser.add_argument(
        "--preflight-closeout",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_preflight_closeout_v2.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_submission_v2.json",
    )
    return parser.parse_args()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate v11 submission: {record_path}")
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v11_actor_hold_control"
    plan_path = control / "training_plan.json"
    protocol_path = control / "protocol.json"
    rationale_path = control / "design_rationale.json"
    correction_path = control / "boundary_correction.json"
    deployment_path = workspace / "deployment_manifest_v11.json"
    base_manifest = workspace / "deployment_manifest.json"
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    rationale = json.loads(rationale_path.read_text())
    correction = json.loads(correction_path.read_text())
    local_verification_path = args.local_verification.expanduser().resolve(
        strict=True
    )
    preflight_path = args.preflight_closeout.expanduser().resolve(strict=True)
    local_verification = json.loads(local_verification_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if (
        local_verification.get("passed") is not True
        or local_verification.get("deployment_manifest_sha256")
        != sha256(deployment_path)
        or local_verification.get("formal_held_out_outcomes_read") is not False
        or local_verification.get("local_training_or_sumo_executed") is not False
        or local_verification.get("v2_boundary_correction_sha256")
        != sha256(correction_path)
        or preflight.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-preflight-closeout-v2"
        or preflight.get("status") != "passed"
        or preflight.get("training_submission_gate_open") is not True
        or preflight.get("exact_remote_verifications") != 2
        or preflight.get("deployment_manifest", {}).get("sha256")
        != sha256(deployment_path)
        or preflight.get("formal_held_out_outcomes_read") is not False
        or preflight.get("local_training_or_sumo_executed") is not False
        or preflight.get("v2_boundary_correction", {}).get("sha256")
        != sha256(correction_path)
        or correction.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-correction-v2"
        or correction.get("status")
        != "corrected-before-any-v11-training-or-sumo-execution"
        or correction.get("v1_training_or_sumo_executed") is not False
    ):
        raise ValueError("v11 deployment verification gate is stale or failed")
    if (
        plan.get("jobs") != protocol.get("training_jobs")
        or len(plan.get("jobs", [])) != 5
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or protocol.get("design_rationale", {}).get("sha256")
        != sha256(rationale_path)
        or plan.get("design_rationale", {}).get("sha256")
        != sha256(rationale_path)
        or rationale.get("status")
        != "frozen-before-v11-training-or-new-development-outcomes"
        or {job.get("seed") for job in plan.get("jobs", [])}
        != set(NODE_BY_SEED)
        or {job.get("variant") for job in plan.get("jobs", [])}
        != {"actor-hold-online-only"}
        or any(job.get("stage") != "candidate" for job in plan.get("jobs", []))
    ):
        raise ValueError("frozen v11 protocol and training plan disagree")

    signatures = {
        job["name"]: (
            f"H2Oplus/v11/development/actor-hold-v2/{job['variant']}/"
            f"seed{job['seed']}/{job['run_signature'][:16]}"
        )
        for job in plan["jobs"]
    }
    state_dir = Path.home() / ".claude/scheduler"
    already = existing_signatures(
        state_dir / "queue.json", state_dir / "queue_archive.jsonl"
    )
    duplicates = sorted(set(signatures.values()) & already)
    if duplicates:
        raise FileExistsError(f"v11 signatures already exist: {duplicates}")

    sys.path.insert(0, str(scheduler_root / "skill"))
    import scheduler as scheduler_runtime  # type: ignore  # noqa: PLC0415

    jobs_by_node = {
        node: [job for job in plan["jobs"] if NODE_BY_SEED[int(job["seed"])] == node]
        for node in sorted(set(NODE_BY_SEED.values()))
    }
    remote_absence = []
    for node, node_jobs in jobs_by_node.items():
        checks = ["found=0"]
        for job in node_jobs:
            run_dir = Path(plan["remote_output_root"]) / job["name"]
            quoted_path = shlex.quote(str(run_dir))
            quoted_name = shlex.quote(job["name"])
            checks.append(
                f"if test -e {quoted_path}; then "
                f"printf 'EXISTS\\t%s\\n' {quoted_name}; found=1; "
                f"else printf 'ABSENT\\t%s\\n' {quoted_name}; fi"
            )
        checks.append('exit "$found"')
        command = "; ".join(checks)
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
        require_remote_run_root_absent(
            node=node,
            run_name=", ".join(job["name"] for job in node_jobs),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
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
        "H2OPLUS_V11_WORKSPACE_ROOT": str(workspace),
        "H2OPLUS_V11_PROTOCOL": str(protocol_path),
        "H2OPLUS_V11_PROTOCOL_SHA256": sha256(protocol_path),
        "H2OPLUS_V11_BASE_DEPLOYMENT_MANIFEST": str(base_manifest),
        "H2OPLUS_V11_BASE_DEPLOYMENT_MANIFEST_SHA256": sha256(base_manifest),
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v11_actor_hold.py"
    submitted = []
    for job in plan["jobs"]:
        node = NODE_BY_SEED[int(job["seed"])]
        run_dir = Path(plan["remote_output_root"]) / job["name"]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        scheduler_signature = signatures[job["name"]]
        submit = [
            "python3",
            str(scheduler),
            "submit",
            "--description",
            f"H2O+ v11 actor-hold development seed {job['seed']} to 25k events",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            scheduler_signature,
            "--resource-family",
            "h2oplus-v11-actor-hold-worker-v2",
            "--vram-resource-family",
            "h2oplus-v11-actor-hold-worker-v2",
            "--ram-resource-family",
            "h2oplus-v11-actor-hold-worker-v2",
            "--vram",
            "8192",
            "--ram-mb",
            "32768",
            "--cpu",
            "6",
            "--priority",
            "high",
            "--project",
            "h2oplus-v11-development-v2",
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

    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "submitted"
            if len(submitted) == 5
            and all(not item["returncode"] for item in submitted)
            else "partial-failure"
        ),
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_plan": descriptor(plan_path),
        "protocol": descriptor(protocol_path),
        "design_rationale": descriptor(rationale_path),
        "v2_boundary_correction": descriptor(correction_path),
        "deployment_manifest": descriptor(deployment_path),
        "local_deployment_verification": descriptor(local_verification_path),
        "preflight_closeout": descriptor(preflight_path),
        "remote_result_root_absence": remote_absence,
        "required_environment": environment,
        "tasks": submitted,
    }
    _write(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
