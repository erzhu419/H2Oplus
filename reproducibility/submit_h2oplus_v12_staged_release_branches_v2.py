#!/usr/bin/env python3
"""Submit exactly 25 independently released v12-v2 development branches."""

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
RESULTS = REPRO / "h2oplus_v12_staged_release_results"
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}
PLAN_SHA256 = "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa"
PROTOCOL_SHA256 = "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
DEPLOYMENT_SHA256 = "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9"
DEPLOYMENT_CORRECTION_SHA256 = (
    "40dabd0dddab6ce7fd8f08ac025f8cc96a5545d1c36ff24730ce885b5eb67b25"
)
BASE_DEPLOYMENT_SHA256 = (
    "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
)
PREFIX_CLOSEOUT_SHA256 = (
    "8bb303dbe038089422ba88f28bf06cd47a672b71f929d1c14306679b943d0ef1"
)
PREFIX_VERIFICATION_SHA256 = (
    "b13b7f0827326f52c6c9f512a7385e71e5abf40a8f15df0e70d5ec26126242e7"
)
VARIANTS = {
    "abrupt-control",
    "actor-ramp",
    "temperature-ramp",
    "actor-anchor",
    "staged-combined",
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
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


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
    raw = payload.get("tasks", []) if isinstance(payload, dict) else payload
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    signatures = {row.get("signature") for row in rows if row.get("signature")}
    if archive_path.exists():
        for line in archive_path.read_text(errors="replace").splitlines():
            normalized = line.strip("\x00 \t\r\n")
            if normalized:
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
        "--prefix-closeout",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_prefix_closeout_v2.json",
    )
    parser.add_argument(
        "--prefix-verification",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_prefix_closeout_verification_v2.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_branch_submission_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate v12 branch submission: {record_path}")
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = load_scheduler(scheduler_root)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v12_control_v2"
    plan_path = control / "training_plan.json"
    protocol_path = control / "protocol.json"
    correction_path = control / "deployment_correction.json"
    deployment_path = workspace / "deployment_manifest_v12_v2.json"
    base_deployment = workspace / "deployment_manifest.json"
    closeout_path = args.prefix_closeout.expanduser().resolve(strict=True)
    verification_path = args.prefix_verification.expanduser().resolve(strict=True)
    expected = {
        plan_path: PLAN_SHA256,
        protocol_path: PROTOCOL_SHA256,
        correction_path: DEPLOYMENT_CORRECTION_SHA256,
        deployment_path: DEPLOYMENT_SHA256,
        base_deployment: BASE_DEPLOYMENT_SHA256,
        closeout_path: PREFIX_CLOSEOUT_SHA256,
        verification_path: PREFIX_VERIFICATION_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"frozen v12 branch control drifted: {path}")

    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    closeout = json.loads(closeout_path.read_text())
    verification = json.loads(verification_path.read_text())
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v12-staged-release-prefix-closeout-v2"
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_five_prefix_runs") is not True
        or closeout.get("exact_seed_matched_10k_handoffs_passed") is not True
        or closeout.get("branch_training_submission_gate_open") is not True
        or closeout.get("development_evaluation_gate_open") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("schema_version")
        != "transit-h2oplus-v12-staged-release-prefix-closeout-independent-verification-v2"
        or verification.get("status") != "passed"
        or verification.get("verified_run_count") != 5
        or verification.get("verified_file_count") != 120
        or verification.get("branch_training_submission_gate_open") is not True
        or verification.get("development_evaluation_gate_open") is not False
        or verification.get("formal_held_out_outcomes_read") is not False
        or verification.get("closeout", {}).get("sha256") != sha256(closeout_path)
    ):
        raise ValueError("v12 branch release gate is not independently open")

    prefix_jobs = [job for job in plan["jobs"] if job.get("stage") == "prefix"]
    branch_jobs = [job for job in plan["jobs"] if job.get("stage") == "branch"]
    if (
        plan.get("jobs") != protocol.get("training_jobs")
        or len(prefix_jobs) != 5
        or len(branch_jobs) != 25
        or {job["variant"] for job in branch_jobs} != VARIANTS
        or any(job["online_event_cap"] != 25000 for job in branch_jobs)
        or any(job["epochs"] != 40 for job in branch_jobs)
        or any(job["expected_total_updates"] != 70000 for job in branch_jobs)
        or any(job["export_online_events"] != [25000] for job in branch_jobs)
        or any(not job.get("branch_handoff_manifest") for job in branch_jobs)
        or any("--allow-no-resume" in job["command_arguments"] for job in branch_jobs)
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("frozen v12 branch matrix drifted")

    signatures = {
        job["name"]: (
            f"H2Oplus/v12/development/staged-release-v2/branch/{job['variant']}/"
            f"seed{job['seed']}/{job['run_signature'][:16]}"
        )
        for job in branch_jobs
    }
    state_dir = Path.home() / ".claude/scheduler"
    existing = existing_signatures(
        state_dir / "queue.json", state_dir / "queue_archive.jsonl"
    )
    duplicates = sorted(set(signatures.values()) & existing)
    if duplicates:
        raise FileExistsError(f"v12 branch signatures already exist: {duplicates}")

    expected_remote_root = Path(plan["remote_output_root"])
    by_node = {
        node: [job for job in branch_jobs if NODE_BY_SEED[int(job["seed"])] == node]
        for node in sorted(set(NODE_BY_SEED.values()))
    }
    remote_absence = []
    for node, jobs in by_node.items():
        statements = ["found=0"]
        for job in jobs:
            run_dir = expected_remote_root / job["name"]
            statements.append(
                f"if test -e {shlex.quote(str(run_dir))}; then "
                f"printf 'EXISTS\\t%s\\n' {shlex.quote(job['name'])}; found=1; "
                f"else printf 'ABSENT\\t%s\\n' {shlex.quote(job['name'])}; fi"
            )
        statements.append('exit "$found"')
        attempts = []
        for attempt in range(1, 4):
            rc, out, err = scheduler.run_on(
                node, "; ".join(statements), timeout=60, check=False
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": rc,
                    "stdout": out,
                    "stderr": err,
                }
            )
            if rc in {0, 1}:
                break
            if attempt < 3:
                time.sleep(5 * attempt)
        remote_absence.append(
            {
                "node": node,
                "run_names": [job["name"] for job in jobs],
                "attempts": attempts,
                "returncode": rc,
            }
        )
        if rc == 1:
            raise FileExistsError(f"v12 branch result root already exists on {node}")
        if rc != 0:
            raise ConnectionError(f"v12 branch root check failed on {node}: {err or out}")

    remote_handoffs = []
    for job in prefix_jobs:
        seed = int(job["seed"])
        node = NODE_BY_SEED[seed]
        local_manifest = (
            RESULTS / "prefix_v2" / job["name"] / "v12_handoff/manifest.json"
        )
        local_payload = json.loads(local_manifest.read_text())
        remote_manifest = (
            expected_remote_root / job["name"] / "v12_handoff/manifest.json"
        )
        code = (
            "import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
            "m=json.loads(p.read_text()); rows=[]; "
            "\nfor r in m['files']:\n"
            " q=p.parent/r['path']; h=hashlib.sha256(); "
            " f=q.open('rb'); "
            "\n while True:\n"
            "  c=f.read(8388608)\n"
            "  if not c: break\n"
            "  h.update(c)\n"
            " f.close(); rows.append({'path':r['path'],'bytes':q.stat().st_size,'sha256':h.hexdigest()})\n"
            "print(json.dumps({'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':rows},sort_keys=True))"
        )
        command = f"python3 -c {shlex.quote(code)} {shlex.quote(str(remote_manifest))}"
        rc, out, err = scheduler.run_on(node, command, timeout=300, check=False)
        if rc != 0:
            raise RuntimeError(f"remote handoff verification failed on {node}: {err or out}")
        observed = json.loads(out)
        if (
            observed.get("manifest_sha256") != sha256(local_manifest)
            or observed.get("files") != local_payload.get("files")
            or local_payload.get("checkpoint_state_sha256")
            != next(
                row["checkpoint_state_sha256"]
                for row in verification["runs"]
                if row["seed"] == seed
            )
        ):
            raise ValueError(f"remote handoff drifted for seed {seed}")
        remote_handoffs.append(
            {
                "prefix_run": job["name"],
                "seed": seed,
                "node": node,
                "manifest": descriptor(local_manifest),
                "checkpoint_state_sha256": local_payload["checkpoint_state_sha256"],
                "files": observed["files"],
            }
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
        "H2OPLUS_V12_PROTOCOL_SHA256": PROTOCOL_SHA256,
        "H2OPLUS_V12_BASE_DEPLOYMENT_MANIFEST": str(base_deployment),
        "H2OPLUS_V12_BASE_DEPLOYMENT_MANIFEST_SHA256": BASE_DEPLOYMENT_SHA256,
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    wrapper = workspace / "SimpleSAC/train_transit_v12_staged_release.py"
    scheduler_cli = scheduler_root / "skill/scheduler.py"
    submitted = []
    for job in branch_jobs:
        seed = int(job["seed"])
        node = NODE_BY_SEED[seed]
        run_dir = expected_remote_root / job["name"]
        scheduler_signature = signatures[job["name"]]
        command = shlex.join([python, str(wrapper), *job["command_arguments"]])
        submit = [
            "python3",
            str(scheduler_cli),
            "submit",
            "--description",
            f"H2O+ v12 {job['variant']} seed {seed} 10k-to-25k development branch",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            scheduler_signature,
            "--resource-family",
            "h2oplus-v12-staged-release-branch-v2",
            "--vram-resource-family",
            "h2oplus-v12-staged-release-branch-v2",
            "--ram-resource-family",
            "h2oplus-v12-staged-release-branch-v2",
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
                "stage": "branch",
                "variant": job["variant"],
                "seed": seed,
                "run_signature": job["run_signature"],
                "scheduler_signature": scheduler_signature,
                "require_node": node,
                "branch_handoff_manifest": job["branch_handoff_manifest"],
                "result_root": str(run_dir),
                "returncode": result.returncode,
                "task_id": match.group(0) if match else None,
                "scheduler_output": combined.strip(),
            }
        )
        if result.returncode or match is None:
            break

    passed = len(submitted) == 25 and all(
        row["returncode"] == 0 and row["task_id"] for row in submitted
    )
    payload = {
        "schema_version": "transit-h2oplus-v12-staged-release-branch-submission-v2",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "partial-failure",
        "study_stage": "development",
        "submitted_stage": "branch-training",
        "prefix_training_rerun": False,
        "training_plan": descriptor(plan_path),
        "protocol": descriptor(protocol_path),
        "deployment_correction": descriptor(correction_path),
        "deployment_manifest": descriptor(deployment_path),
        "prefix_closeout": descriptor(closeout_path),
        "prefix_independent_verification": descriptor(verification_path),
        "remote_result_root_absence": remote_absence,
        "remote_handoff_verification": remote_handoffs,
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "exactly one physical GPU assigned by scheduleurm"
        },
        "tasks": submitted,
        "submitted_prefix_tasks": 0,
        "submitted_branch_tasks": len(submitted),
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
