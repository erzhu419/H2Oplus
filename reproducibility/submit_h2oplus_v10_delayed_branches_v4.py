#!/usr/bin/env python3
"""Submit exactly 25 audited v10-v4 development branch jobs through scheduleurm."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v10_delayed_results"
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}
PLAN_SHA256 = "20d87242072b4aab8332d25e2d2d27af986818339b6c63672d781607e7a0b829"
PROTOCOL_SHA256 = "83a45e7d06ec30bdaba3631d7d8c6854e7609b8dcff9c46aa735d9a7944daaf3"
DEPLOYMENT_SHA256 = "4979e6863eedda06a0627fde7c67b684339a11702e03862a34ad1005ef1ad696"
BASE_DEPLOYMENT_SHA256 = "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_scheduler(root: Path):
    sys.path.insert(0, str(root / "skill"))
    return importlib.import_module("scheduler")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v10_delayed_pilot_v4",
    )
    parser.add_argument(
        "--prefix-closeout",
        type=Path,
        default=RESULTS / "prefix_closeout_v4.json",
    )
    parser.add_argument(
        "--prefix-verification",
        type=Path,
        default=RESULTS / "prefix_closeout_verification_v4.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_branch_submission_v4.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate v10-v4 branch submission: {record_path}")
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = load_scheduler(scheduler_root)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v10_control"
    plan_path = control / "training_plan.json"
    protocol_path = control / "protocol.json"
    deployment_path = workspace / "deployment_manifest_v10.json"
    base_deployment = workspace / "deployment_manifest.json"
    if (
        sha256(plan_path) != PLAN_SHA256
        or sha256(protocol_path) != PROTOCOL_SHA256
        or sha256(deployment_path) != DEPLOYMENT_SHA256
        or sha256(base_deployment) != BASE_DEPLOYMENT_SHA256
    ):
        raise ValueError("frozen v10-v4 deployment controls drifted")
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    closeout_path = args.prefix_closeout.expanduser().resolve(strict=True)
    verification_path = args.prefix_verification.expanduser().resolve(strict=True)
    closeout = json.loads(closeout_path.read_text())
    verification = json.loads(verification_path.read_text())
    if (
        closeout.get("schema_version") != "transit-h2oplus-v10-delayed-prefix-closeout-v4"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed") is not True
        or closeout.get("branch_submission_gate_open") is not True
        or closeout.get("prefix_runs") != 5
        or closeout.get("exact_handoffs") != 5
        or closeout.get("branch_jobs_registered") != 25
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or verification.get("schema_version")
        != "transit-h2oplus-v10-delayed-prefix-closeout-verification-v4"
        or verification.get("passed") is not True
        or verification.get("producer_independent") is not True
        or verification.get("formal_held_out_outcomes_read") is not False
        or verification.get("prefix_closeout", {}).get("sha256") != sha256(closeout_path)
    ):
        raise ValueError("v10-v4 prefix gate is not independently open")

    prefix_jobs = [job for job in plan["jobs"] if job["stage"] == "prefix"]
    branch_jobs = [job for job in plan["jobs"] if job["stage"] == "branch"]
    variants = {job["variant"] for job in branch_jobs}
    if (
        plan["jobs"] != protocol.get("training_jobs")
        or len(prefix_jobs) != 5
        or len(branch_jobs) != 25
        or variants != {"unit-control", "delayed-replay", "delayed-actor", "delayed-order", "delayed-combined"}
        or any(job["online_event_cap"] != 25000 or job["epochs"] != 40 for job in branch_jobs)
        or any(job["branch_handoff_manifest"] is None for job in branch_jobs)
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("frozen v10-v4 branch matrix drifted")

    signatures = {
        job["name"]: (
            f"H2Oplus/v10/development/delayed-v4/branch/{job['variant']}/"
            f"seed{job['seed']}/{job['run_signature'][:16]}"
        )
        for job in branch_jobs
    }
    state_dir = Path.home() / ".claude/scheduler"
    already = existing_signatures(state_dir / "queue.json", state_dir / "queue_archive.jsonl")
    duplicates = sorted(set(signatures.values()) & already)
    if duplicates:
        raise FileExistsError(f"v10-v4 branch signatures already exist: {duplicates}")

    remote_handoffs = []
    for job in prefix_jobs:
        node = NODE_BY_SEED[int(job["seed"])]
        local_run = RESULTS / "prefix_v4" / job["name"]
        local_manifest = local_run / "v10_handoff/manifest.json"
        manifest = json.loads(local_manifest.read_text())
        remote_manifest = Path(plan["remote_output_root"]) / job["name"] / "v10_handoff/manifest.json"
        code = (
            "import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
            "m=json.loads(p.read_text()); rows=[]; "
            "\nfor r in m['files']:\n"
            " q=p.parent/r['path']; h=hashlib.sha256(q.read_bytes()).hexdigest(); "
            " rows.append({'path':r['path'],'bytes':q.stat().st_size,'sha256':h})\n"
            "print(json.dumps({'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':rows},sort_keys=True))"
        )
        command = f"python3 -c {shlex.quote(code)} {shlex.quote(str(remote_manifest))}"
        rc, out, err = scheduler.run_on(node, command, timeout=120, check=False)
        if rc != 0:
            raise RuntimeError(f"remote handoff verification failed on {node}: {err or out}")
        observed = json.loads(out)
        if (
            observed.get("manifest_sha256") != sha256(local_manifest)
            or observed.get("files") != manifest.get("files")
        ):
            raise ValueError(f"remote handoff drifted for {job['name']}")
        remote_handoffs.append(
            {
                "prefix_run": job["name"],
                "seed": job["seed"],
                "node": node,
                "manifest": descriptor(local_manifest),
                "files": observed["files"],
            }
        )

    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": str(workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(workspace),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "H2OPLUS_V10_WORKSPACE_ROOT": str(workspace),
        "H2OPLUS_V10_PROTOCOL": str(protocol_path),
        "H2OPLUS_V10_PROTOCOL_SHA256": PROTOCOL_SHA256,
        "H2OPLUS_V10_BASE_DEPLOYMENT_MANIFEST": str(base_deployment),
        "H2OPLUS_V10_BASE_DEPLOYMENT_MANIFEST_SHA256": BASE_DEPLOYMENT_SHA256,
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v10_delayed.py"
    scheduler_cli = scheduler_root / "skill/scheduler.py"
    submitted = []
    for job in branch_jobs:
        seed = int(job["seed"])
        node = NODE_BY_SEED[seed]
        run_dir = Path(plan["remote_output_root"]) / job["name"]
        scheduler_signature = signatures[job["name"]]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        submit = [
            "python3", str(scheduler_cli), "submit",
            "--description", f"H2O+ v10 {job['variant']} seed {seed} 10k-to-25k development branch",
            "--cmd", command,
            "--cwd", str(workspace),
            "--signature", scheduler_signature,
            "--resource-family", "h2oplus-v10-delayed-worker-v4",
            "--vram-resource-family", "h2oplus-v10-delayed-worker-v4",
            "--ram-resource-family", "h2oplus-v10-delayed-worker-v4",
            "--vram", "8192",
            "--ram-mb", "32768",
            "--cpu", "6",
            "--priority", "high",
            "--project", "h2oplus-v10-development-v4",
            "--require-node", node,
            "--ckpt-dir", str(run_dir / "checkpoints"),
            "--result-dir", str(run_dir),
            "--local-result-dir", str(run_dir),
            "--ckpt-glob", "latest.pt",
            "--resume-flag=--resume",
            "--allow-initial-resume-scan-error",
            "--env-spec", "none",
            "--allow-remote-large-data",
            "--skip-launch-staging",
            "--env", *[f"{key}={value}" for key, value in environment.items()],
        ]
        result = subprocess.run(submit, cwd=scheduler_root, text=True, capture_output=True, check=False)
        combined = "\n".join(value for value in (result.stdout, result.stderr) if value)
        match = re.search(r"\bt\d+\b", combined)
        row = {
            "name": job["name"],
            "stage": "branch",
            "variant": job["variant"],
            "seed": seed,
            "run_signature": job["run_signature"],
            "scheduler_signature": scheduler_signature,
            "require_node": node,
            "prefix_run": job["prefix"]["name"],
            "branch_handoff_manifest": job["branch_handoff_manifest"],
            "returncode": result.returncode,
            "task_id": match.group(0) if match else None,
            "scheduler_output": combined.strip(),
        }
        submitted.append(row)
        if result.returncode or row["task_id"] is None:
            break

    passed = len(submitted) == 25 and all(row["returncode"] == 0 and row["task_id"] for row in submitted)
    payload = {
        "schema_version": "transit-h2oplus-v10-delayed-branch-submission-v4",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "partial-failure",
        "study_stage": "development",
        "submitted_stage": "branch-training",
        "prefix_training_rerun": False,
        "development_evaluation_submitted": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_plan": descriptor(plan_path),
        "protocol": descriptor(protocol_path),
        "deployment_manifest": descriptor(deployment_path),
        "prefix_closeout": descriptor(closeout_path),
        "prefix_verification": descriptor(verification_path),
        "remote_handoff_verification": remote_handoffs,
        "required_environment": environment,
        "scheduler_managed_environment": {"CUDA_VISIBLE_DEVICES": "exactly one physical GPU per task"},
        "tasks": submitted,
    }
    atomic_json(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
