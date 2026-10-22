#!/usr/bin/env python3
"""Submit one corrected no-training/no-SUMO v12-v3 CUDA preflight per host."""

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
SCHEMA = "transit-h2oplus-v12-staged-release-preflight-submission-v3"
NODE_SEED = {"jtl110gpu": 6201, "jtl110gpu2": 6202}
EXPECTED_MANIFEST_SHA256 = (
    "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9"
)
EXPECTED_REMOTE_VERIFICATION_SHA256 = (
    "ddccb0b092515edb6774544c74a8156234f2f11e287c2d4809fb76aa931db6c9"
)
EXPECTED_PREFLIGHT_CORRECTION_SHA256 = (
    "3d8350feca804a4c54366cb2b9c71334400688b9567023d1c454472003e5a98e"
)


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
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v12_staged_release_pilot_v2",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v12_staged_release_deployment_verification_v2_remote.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO
        / "h2oplus_v12_staged_release_preflight_submission_v3.json",
    )
    parser.add_argument(
        "--preflight-correction",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_preflight_correction_v3.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate v12-v3 preflight: {record_path}")
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler_cli = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v12_control_v2"
    manifest = workspace / "deployment_manifest_v12_v2.json"
    base_manifest = workspace / "deployment_manifest.json"
    protocol_path = control / "protocol.json"
    plan_path = control / "training_plan.json"
    correction_path = control / "deployment_correction.json"
    preflight_correction_path = args.preflight_correction.expanduser().resolve(
        strict=True
    )
    preflight_correction = json.loads(preflight_correction_path.read_text())
    remote_verification_path = args.remote_verification.expanduser().resolve(strict=True)
    remote_verification = json.loads(remote_verification_path.read_text())
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if (
        sha256(manifest) != EXPECTED_MANIFEST_SHA256
        or sha256(remote_verification_path) != EXPECTED_REMOTE_VERIFICATION_SHA256
        or sha256(preflight_correction_path)
        != EXPECTED_PREFLIGHT_CORRECTION_SHA256
        or preflight_correction.get("status")
        != "frozen-after-scheduler-refusal-before-task-creation"
        or preflight_correction.get("preflight_v2_task_ids") != []
        or preflight_correction.get("preflight_v2_training_or_sumo_executed")
        is not False
        or preflight_correction.get("preflight_v2_scheduler_signatures_registered")
        is not False
        or remote_verification.get("passed") is not True
        or len(remote_verification.get("nodes", [])) != 2
        or not all(row.get("passed") is True for row in remote_verification["nodes"])
        or remote_verification.get("formal_held_out_outcomes_read") is not False
        or remote_verification.get("local_or_remote_sumo_simulation_executed")
        is not False
        or remote_verification.get("training_executed") is not False
        or plan.get("jobs") != protocol.get("training_jobs")
        or len(plan.get("jobs", [])) != 30
        or len([job for job in plan["jobs"] if job.get("stage") == "prefix"]) != 5
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
    ):
        raise ValueError("v12-v2 remote deployment gate is stale or failed")

    prefix_jobs = {
        int(job["seed"]): job
        for job in plan["jobs"]
        if job.get("stage") == "prefix" and job.get("variant") == "shared-prefix"
    }
    if set(prefix_jobs) != {6201, 6202, 6203, 6204, 6205}:
        raise ValueError("v12-v2 prefix matrix is incomplete")

    signatures = {
        node: (
            f"H2Oplus/v12/development/staged-release-v2/preflight-v3/{node}/"
            f"{EXPECTED_MANIFEST_SHA256[:16]}"
        )
        for node in NODE_SEED
    }
    state_dir = Path.home() / ".claude/scheduler"
    existing = existing_signatures(
        state_dir / "queue.json", state_dir / "queue_archive.jsonl"
    )
    duplicates = sorted(set(signatures.values()) & existing)
    if duplicates:
        raise FileExistsError(f"v12-v3 preflight signatures already exist: {duplicates}")

    sys.path.insert(0, str(scheduler_root / "skill"))
    import scheduler as scheduler_runtime  # type: ignore  # noqa: PLC0415

    result_roots = {
        node: deployment_root / "preflight_results_v3" / node for node in NODE_SEED
    }
    remote_absence = []
    for node, result_root in result_roots.items():
        rc, stdout, stderr = scheduler_runtime.run_on(
            node,
            f"test ! -e {shlex.quote(str(result_root))}",
            timeout=30,
            check=False,
        )
        remote_absence.append(
            {
                "node": node,
                "path": str(result_root),
                "returncode": rc,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        if rc == 1:
            raise FileExistsError(f"v12-v3 preflight root exists on {node}")
        if rc != 0:
            raise ConnectionError(
                f"cannot verify v12-v3 preflight root absence on {node}: {stderr}"
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
    submitted = []
    for node, seed in NODE_SEED.items():
        job = prefix_jobs[seed]
        result_root = result_roots[node]
        output = result_root / "cuda_preflight.json"
        remote_code = f'''
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

workspace = Path({str(workspace)!r}).resolve(strict=True)
protocol_path = Path({str(protocol_path)!r}).resolve(strict=True)
output = Path({str(output)!r})
protocol = json.loads(protocol_path.read_text())
job = next(row for row in protocol["training_jobs"] if row["run_signature"] == {job["run_signature"]!r})
sys.argv = [str(workspace / "SimpleSAC/train_transit_v12_staged_release.py"), *job["command_arguments"]]
import SimpleSAC.train_transit_v12_staged_release as launcher

initializer_path = Path(job["offline_initializer"]["path"]).resolve(strict=True)
initializer = torch.load(initializer_path, map_location="cpu", weights_only=False)
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not cuda_visible or "," in cuda_visible:
    raise RuntimeError("scheduler did not assign exactly one CUDA device")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("assigned CUDA device is unavailable")
probe = torch.arange(32, dtype=torch.float32, device="cuda")
probe_checksum = float((probe * probe).sum().cpu().item())
torch.cuda.synchronize()
source_map = launcher._v12_source_sha256()
if source_map != protocol["v12_source_sha256"]:
    raise RuntimeError("v12 launcher source map drifted")
if initializer.get("artifact_schema") != "transit-offline-initializer-v7-v1":
    raise RuntimeError("v12 initializer schema drifted")
payload = {{
    "schema_version": "transit-h2oplus-v12-staged-release-cuda-preflight-v3",
    "passed": True,
    "hostname": os.uname().nodename,
    "cuda_visible_devices": cuda_visible,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "probe_checksum": probe_checksum,
    "run_name": job["name"],
    "run_signature": job["run_signature"],
    "seed": job["seed"],
    "variant": launcher.VARIANT,
    "stage": launcher.STAGE,
    "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    "initializer_sha256": hashlib.sha256(initializer_path.read_bytes()).hexdigest(),
    "source_sha256": source_map,
    "training_executed": False,
    "sumo_simulation_executed": False,
    "formal_held_out_outcomes_read": False,
}}
if output.exists():
    raise FileExistsError("v12-v3 CUDA preflight output already exists")
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
'''
        command = shlex.join([python, "-c", remote_code])
        signature = signatures[node]
        submit = [
            "python3",
            str(scheduler_cli),
            "submit",
            "--description",
            f"H2O+ v12-v3 corrected no-training CUDA preflight on {node}",
            "--cmd",
            command,
            "--cwd",
            str(workspace),
            "--signature",
            signature,
            "--resource-family",
            "h2oplus-v12-staged-release-preflight-v3",
            "--vram-resource-family",
            "h2oplus-v12-staged-release-preflight-v3",
            "--ram-resource-family",
            "h2oplus-v12-staged-release-preflight-v3",
            "--vram",
            "1024",
            "--ram-mb",
            "8192",
            "--cpu",
            "1",
            "--priority",
            "high",
            "--project",
            "h2oplus-v12-development-v2",
            "--require-node",
            node,
            "--result-dir",
            str(result_root),
            "--local-result-dir",
            str(result_root),
            "--env-spec",
            "none",
            "--allow-remote-large-data",
            "--skip-launch-staging",
            "--allow-no-ckpt",
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
                "seed": seed,
                "run_name": job["name"],
                "run_signature": job["run_signature"],
                "task_id": match.group(0) if match else None,
                "signature": signature,
                "result_root": str(result_root),
                "verification_output": str(output),
                "returncode": result.returncode,
                "scheduler_output": combined.strip(),
            }
        )
        if result.returncode or match is None:
            break

    passed = len(submitted) == len(NODE_SEED) and all(
        item["returncode"] == 0 and item["task_id"] for item in submitted
    )
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted" if passed else "partial-failure",
        "study_stage": "development",
        "purpose": "scheduler-assigned-cuda-and-full-launcher-import-preflight-only",
        "deployment_manifest": descriptor(manifest),
        "protocol": descriptor(protocol_path),
        "training_plan": descriptor(plan_path),
        "deployment_correction": descriptor(correction_path),
        "preflight_correction": descriptor(preflight_correction_path),
        "remote_deployment_verification": descriptor(remote_verification_path),
        "remote_result_root_absence": remote_absence,
        "required_environment": environment,
        "scheduler_managed_environment": {
            "CUDA_VISIBLE_DEVICES": "exactly one physical GPU assigned by scheduleurm"
        },
        "tasks": submitted,
        "training_executed": False,
        "sumo_simulation_executed": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
