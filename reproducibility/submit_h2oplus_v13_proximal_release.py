#!/usr/bin/env python3
"""Submit exactly 15 preregistered v13 development branches."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}
RATES = {
    "proximal-weak": 1.0e-5,
    "proximal-medium": 2.5e-5,
    "proximal-strong": 5.0e-5,
}
PLAN_SHA256 = "ca26225ab13a70e3da9d409e70fdb2bdb44b8b568c62d70cf5cffba723d7f431"
PROTOCOL_SHA256 = "f5c58487dd37ac9766d80af7d391e1d69bfce7c18f2c0269da414b39ef34702d"
FREEZE_SHA256 = "50719bb445a54fd50a7ecf5584f9ec2e2e53a068c2cf93718cf74782d755f4f8"
DEPLOYMENT_SHA256 = "6689b3db11669eb0e8c5fd2ec71765a268e69db8ed72f0ae734376217f910cec"
BASE_DEPLOYMENT_SHA256 = "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
LOCAL_VERIFICATION_SHA256 = "a0fc71a609fecbc3b63da032e15981360b179840fc06dc112263957a3aa8234d"
SYNC_SHA256 = "c3f1ae1ea25113e5cacc5717d2aa908aa51dec73e9222c525c4023b485829499"
REMOTE_VERIFICATION_SHA256 = "eadf4f6dffd21aa5ba0df1cf9e77459685802f0a95194f2cbe0b76dcc60b5254"


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
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def existing_signatures(queue_path: Path, archive_path: Path) -> set[str]:
    payload = json.loads(queue_path.read_text())
    raw = payload.get("tasks", []) if isinstance(payload, dict) else payload
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    signatures = {row.get("signature") for row in rows if row.get("signature")}
    if archive_path.exists():
        for line in archive_path.read_text(errors="replace").splitlines():
            normalized = line.strip("\x00 \t\r\n")
            if not normalized:
                continue
            row = json.loads(normalized)
            if row.get("signature"):
                signatures.add(row["signature"])
    return signatures


def run_on_bounded(scheduler, node: str, command: str):
    attempts = []
    for attempt in range(1, 4):
        rc, stdout, stderr = scheduler.run_on(
            node, command, timeout=120, check=False
        )
        attempts.append(
            {
                "attempt": attempt,
                "returncode": rc,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }
        )
        if rc in {0, 1}:
            break
        if attempt < 3:
            time.sleep(5 * attempt)
    return rc, stdout, stderr, attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v13_proximal_release_pilot_v1",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_submission_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate v13 submission: {record_path}")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root / "skill"))
    scheduler = importlib.import_module("scheduler")
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v13_control_v1"
    paths = {
        "training_plan": control / "training_plan.json",
        "protocol": control / "protocol.json",
        "freeze": control / "freeze.json",
        "deployment_manifest": workspace / "deployment_manifest_v13_v1.json",
        "base_deployment_manifest": workspace / "deployment_manifest.json",
        "local_verification": REPRO
        / "h2oplus_v13_proximal_release_deployment_verification_v1_local.json",
        "deployment_sync": REPRO
        / "h2oplus_v13_proximal_release_deployment_sync_v1.json",
        "remote_verification": REPRO
        / "h2oplus_v13_proximal_release_deployment_verification_v1_remote.json",
    }
    expected_hashes = {
        "training_plan": PLAN_SHA256,
        "protocol": PROTOCOL_SHA256,
        "freeze": FREEZE_SHA256,
        "deployment_manifest": DEPLOYMENT_SHA256,
        "base_deployment_manifest": BASE_DEPLOYMENT_SHA256,
        "local_verification": LOCAL_VERIFICATION_SHA256,
        "deployment_sync": SYNC_SHA256,
        "remote_verification": REMOTE_VERIFICATION_SHA256,
    }
    for label, path in paths.items():
        if sha256(path.expanduser().resolve(strict=True)) != expected_hashes[label]:
            raise ValueError(f"frozen v13 submission input drifted: {label}")

    plan = json.loads(paths["training_plan"].read_text())
    protocol = json.loads(paths["protocol"].read_text())
    freeze = json.loads(paths["freeze"].read_text())
    local = json.loads(paths["local_verification"].read_text())
    sync = json.loads(paths["deployment_sync"].read_text())
    remote = json.loads(paths["remote_verification"].read_text())
    jobs = plan.get("jobs", [])
    if (
        jobs != protocol.get("training_jobs")
        or len(jobs) != 15
        or {job.get("variant") for job in jobs} != set(RATES)
        or {int(job.get("seed", -1)) for job in jobs} != set(NODE_BY_SEED)
        or any(job.get("stage") != "branch" for job in jobs)
        or any(job.get("proximal_rate") != RATES[job["variant"]] for job in jobs)
        or any(job.get("epochs") != 40 for job in jobs)
        or any(job.get("online_event_cap") != 25000 for job in jobs)
        or any(job.get("expected_total_updates") != 70000 for job in jobs)
        or any(job.get("export_online_events") != [25000] for job in jobs)
        or any("--allow-no-resume" in job["command_arguments"] for job in jobs)
        or sum(bool(job.get("predesignated_candidate")) for job in jobs) != 5
        or any(
            bool(job.get("predesignated_candidate"))
            != (job.get("variant") == "proximal-medium")
            for job in jobs
        )
        or freeze.get("prefix_runs_rerun") != 0
        or freeze.get("new_training_runs") != 15
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or local.get("passed") is not True
        or sync.get("passed") is not True
        or remote.get("passed") is not True
        or {row.get("node") for row in remote.get("nodes", [])}
        != set(NODE_BY_SEED.values())
        or any(row.get("passed") is not True for row in remote.get("nodes", []))
    ):
        raise ValueError("v13 training release gate is not independently open")

    signatures = {
        job["name"]: (
            f"H2Oplus/v13/development/proximal-release-v1/{job['variant']}/"
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
        raise FileExistsError(f"v13 scheduler signatures already exist: {duplicates}")

    result_root = Path(plan["remote_output_root"])
    if result_root.exists():
        raise FileExistsError(f"local v13 result root already exists: {result_root}")
    absence = []
    for node in sorted(set(NODE_BY_SEED.values())):
        statements = ["found=0"]
        for job in jobs:
            run_dir = result_root / job["name"]
            statements.append(
                f"if test -e {shlex.quote(str(run_dir))}; then "
                f"printf 'EXISTS\\t%s\\n' {shlex.quote(job['name'])}; found=1; "
                f"else printf 'ABSENT\\t%s\\n' {shlex.quote(job['name'])}; fi"
            )
        statements.append('exit "$found"')
        rc, stdout, stderr, attempts = run_on_bounded(
            scheduler, node, "; ".join(statements)
        )
        absence.append(
            {
                "node": node,
                "checked_run_names": [job["name"] for job in jobs],
                "returncode": rc,
                "attempts": attempts,
            }
        )
        if rc == 1:
            raise FileExistsError(f"v13 result root already exists on {node}")
        if rc != 0:
            raise ConnectionError(f"v13 result-root check failed on {node}: {stderr or stdout}")

    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(workspace),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "H2OPLUS_V13_WORKSPACE_ROOT": str(workspace),
        "H2OPLUS_V13_PROTOCOL": str(paths["protocol"]),
        "H2OPLUS_V13_PROTOCOL_SHA256": PROTOCOL_SHA256,
        "H2OPLUS_V13_BASE_DEPLOYMENT_MANIFEST": str(
            paths["base_deployment_manifest"]
        ),
        "H2OPLUS_V13_BASE_DEPLOYMENT_MANIFEST_SHA256": BASE_DEPLOYMENT_SHA256,
    }
    payload = {
        "schema_version": "transit-h2oplus-v13-proximal-release-submission-v1",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "study_stage": "development",
        "submitted_stage": "branch-training",
        "prefix_training_rerun": False,
        **{label: descriptor(path) for label, path in paths.items()},
        "submission_program": descriptor(Path(__file__)),
        "pre_submission_signature_matches": [],
        "remote_result_root_absence": absence,
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "exactly one physical GPU assigned by scheduleurm"
        },
        "tasks": [],
        "submitted_prefix_tasks": 0,
        "submitted_branch_tasks": 0,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(record_path, payload)

    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v13_proximal_release.py"
    scheduler_cli = scheduler_root / "skill/scheduler.py"
    for job in jobs:
        seed = int(job["seed"])
        node = NODE_BY_SEED[seed]
        run_dir = result_root / job["name"]
        signature = signatures[job["name"]]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        submit = [
            "python3",
            str(scheduler_cli),
            "submit",
            "--description",
            f"H2O+ v13 {job['variant']} seed {seed} 10k-to-25k development branch",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            signature,
            "--resource-family",
            "h2oplus-v13-proximal-release-branch-v1",
            "--vram-resource-family",
            "h2oplus-v13-proximal-release-branch-v1",
            "--ram-resource-family",
            "h2oplus-v13-proximal-release-branch-v1",
            "--vram",
            "8192",
            "--ram-mb",
            "32768",
            "--cpu",
            "6",
            "--priority",
            "high",
            "--project",
            "h2oplus-v13-development-v1",
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
        payload["tasks"].append(
            {
                "name": job["name"],
                "stage": "branch",
                "variant": job["variant"],
                "proximal_rate": job["proximal_rate"],
                "predesignated_candidate": job["predesignated_candidate"],
                "seed": seed,
                "run_signature": job["run_signature"],
                "scheduler_signature": signature,
                "require_node": node,
                "branch_handoff_manifest": job["branch_handoff_manifest"],
                "source_handoff": job["source_handoff"],
                "result_root": str(run_dir),
                "declared_cpu_cores": 6,
                "declared_ram_mb": 32768,
                "declared_vram_mb": 8192,
                "returncode": result.returncode,
                "task_id": match.group(0) if match else None,
                "scheduler_output": combined.strip(),
            }
        )
        payload["submitted_branch_tasks"] = len(payload["tasks"])
        if result.returncode or match is None:
            payload["status"] = "partial-failure"
            atomic_json(record_path, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 1
        atomic_json(record_path, payload)

    payload["status"] = "submitted"
    atomic_json(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
