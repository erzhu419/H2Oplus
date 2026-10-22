#!/usr/bin/env python3
"""Stage and verify the v11 actor-hold deployment on both training hosts."""

from __future__ import annotations

import argparse
import hashlib
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
SCHEMA = "transit-h2oplus-v11-actor-hold-preflight-submission-v2"
NODES = ("jtl110gpu", "jtl110gpu2")


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


def _existing_signatures(queue_path: Path, archive_path: Path) -> set[str]:
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
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_preflight_submission_v2.json",
    )
    parser.add_argument(
        "--rationale",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_rationale_v1.json",
    )
    parser.add_argument(
        "--correction",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_correction_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing to duplicate v11 preflight: {record_path}")
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    manifest = workspace / "deployment_manifest_v11.json"
    local_verification_path = args.local_verification.expanduser().resolve(strict=True)
    rationale_path = args.rationale.expanduser().resolve(strict=True)
    correction_path = args.correction.expanduser().resolve(strict=True)
    local_verification = json.loads(local_verification_path.read_text())
    rationale = json.loads(rationale_path.read_text())
    correction = json.loads(correction_path.read_text())
    if (
        local_verification.get("passed") is not True
        or local_verification.get("deployment_manifest_sha256") != sha256(manifest)
        or local_verification.get("formal_held_out_outcomes_read") is not False
        or local_verification.get("local_training_or_sumo_executed") is not False
        or local_verification.get("v2_boundary_correction_sha256")
        != sha256(correction_path)
    ):
        raise ValueError("local v11 deployment verification is stale or failed")
    if (
        rationale.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-rationale-v1"
        or rationale.get("status")
        != "frozen-before-v11-training-or-new-development-outcomes"
        or rationale.get("formal_held_out_outcomes_read") is not False
        or rationale.get("local_training_or_sumo_executed") is not False
        or local_verification.get("design_rationale_sha256")
        != sha256(rationale_path)
        or correction.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-correction-v2"
        or correction.get("status")
        != "corrected-before-any-v11-training-or-sumo-execution"
        or correction.get("v1_training_or_sumo_executed") is not False
    ):
        raise ValueError("v11 preflight rationale is stale or invalid")

    signatures = {
        node: (
            f"H2Oplus/v11/development/actor-hold-v2/preflight-v2/{node}/"
            f"{sha256(manifest)[:16]}"
        )
        for node in NODES
    }
    state_dir = Path.home() / ".claude/scheduler"
    existing = _existing_signatures(
        state_dir / "queue.json", state_dir / "queue_archive.jsonl"
    )
    duplicates = sorted(set(signatures.values()) & existing)
    if duplicates:
        raise FileExistsError(f"v11 preflight signatures already exist: {duplicates}")

    sys.path.insert(0, str(scheduler_root / "skill"))
    import scheduler as scheduler_runtime  # type: ignore  # noqa: PLC0415

    bootstrap_workspace = (
        REPRO / "deployments/h2oplus_v11_actor_hold_pilot_v1/workspace"
    )
    bootstrap = []
    for node in NODES:
        command = (
            f"test -d {shlex.quote(str(bootstrap_workspace))} && "
            f"test ! -e {shlex.quote(str(workspace))} && "
            f"mkdir -p {shlex.quote(str(deployment_root))} && "
            f"cp -al {shlex.quote(str(bootstrap_workspace))} "
            f"{shlex.quote(str(workspace))}"
        )
        rc, stdout, stderr = scheduler_runtime.run_on(
            node, command, timeout=120, check=False
        )
        bootstrap.append(
            {
                "node": node,
                "command": command,
                "returncode": rc,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        if rc:
            raise RuntimeError(f"v11 remote bootstrap failed on {node}: {stderr}")

    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    verifier = workspace / "reproducibility/verify_h2oplus_v11_actor_hold_deployment.py"
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
    submitted = []
    for node in NODES:
        result_root = deployment_root / "preflight_results_v2" / node
        output = result_root / "verification.json"
        command = shlex.join(
            [python, str(verifier), "--workspace", str(workspace), "--output", str(output)]
        )
        signature = signatures[node]
        submit = [
            "python3",
            str(scheduler),
            "submit",
            "--description",
            f"H2O+ v11 actor-hold deployment no-SUMO preflight on {node}",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            signature,
            "--resource-family",
            "h2oplus-v11-actor-hold-preflight-v2",
            "--vram",
            "0",
            "--ram-mb",
            "8192",
            "--cpu",
            "1",
            "--priority",
            "high",
            "--project",
            "h2oplus-v11-development-v2",
            "--require-node",
            node,
            "--result-dir",
            str(result_root),
            "--local-result-dir",
            str(result_root),
            "--env-spec",
            "none",
            "--allow-remote-large-data",
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
        submitted.append(
            {
                "node": node,
                "returncode": result.returncode,
                "task_id": match.group(0) if match else None,
                "signature": signature,
                "result_root": str(result_root),
                "verification_output": str(output),
                "scheduler_output": combined.strip(),
            }
        )
        if result.returncode or match is None:
            break

    passed = len(submitted) == len(NODES) and all(
        item["returncode"] == 0 and item["task_id"] for item in submitted
    )
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "partial-failure",
        "study_stage": "development",
        "purpose": "deployment-hash-and-loader-preflight-only",
        "remote_training_or_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "deployment_manifest": descriptor(manifest),
        "local_verification": descriptor(local_verification_path),
        "design_rationale": descriptor(rationale_path),
        "v2_boundary_correction": descriptor(correction_path),
        "remote_workspace_bootstrap": bootstrap,
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "empty for the CPU-only task; injected by scheduleurm"
        },
        "tasks": submitted,
    }
    _write(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
