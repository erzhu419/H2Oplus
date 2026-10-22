#!/usr/bin/env python3
"""Submit the frozen v14 development evaluation through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v14-terminal-taper-evaluation-submission-v1"


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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite submission record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v14_terminal_taper_evaluation_v1/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_evaluation_deployment_local_verification_v1.json",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--deployment-sync",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_evaluation_deployment_sync_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_evaluation_submission_v1.json",
    )
    names = (
        "registration",
        "authorization",
        "freeze",
        "design",
        "protocol",
        "training_closeout",
        "training_verification",
        "training_audit",
        "training_file_manifest",
        "source_freeze",
        "methods",
        "dataset",
        "runtime_lock",
        "deployment_manifest",
        "deployment_aggregate",
        "local_verification",
        "remote_verification",
        "deployment_sync",
    )
    for name in names:
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hashes = {
        name: getattr(args, f"expected_{name}_sha256")
        for name in (
            "registration",
            "authorization",
            "freeze",
            "design",
            "protocol",
            "training_closeout",
            "training_verification",
            "training_audit",
            "training_file_manifest",
            "source_freeze",
            "methods",
            "dataset",
            "runtime_lock",
            "deployment_manifest",
            "deployment_aggregate",
            "local_verification",
            "remote_verification",
            "deployment_sync",
        )
    }
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    repo = deployment_root / "H2Oplus"
    local_verification_path = args.local_verification.expanduser().resolve(strict=True)
    remote_verification_path = args.remote_verification.expanduser().resolve(strict=True)
    sync_path = args.deployment_sync.expanduser().resolve(strict=True)
    evidence_paths = {
        "local_verification": local_verification_path,
        "remote_verification": remote_verification_path,
        "deployment_sync": sync_path,
    }
    for name, path in evidence_paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"{name} identity drifted")
    local_verification = json.loads(local_verification_path.read_text())
    remote_verification = json.loads(remote_verification_path.read_text())
    sync = json.loads(sync_path.read_text())
    if (
        local_verification.get("passed") is not True
        or local_verification.get("manifest_sha256") != hashes["deployment_manifest"]
        or local_verification.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or remote_verification.get("passed") is not True
        or remote_verification.get("node") != "jtl110gpu"
        or remote_verification.get("formal_held_out_outcomes_read") is not False
        or remote_verification.get("remote_sumo_simulation_executed") is not False
        or remote_verification.get("expected", {}).get("manifest_sha256")
        != hashes["deployment_manifest"]
        or remote_verification.get("expected", {}).get("aggregate_sha256")
        != hashes["deployment_aggregate"]
        or sync.get("passed") is not True
        or sync.get("manifest_sha256") != hashes["deployment_manifest"]
        or sync.get("aggregate_sha256") != hashes["deployment_aggregate"]
    ):
        raise ValueError("v14 deployment evidence is stale or failed")

    paths = {
        "registration": repo / "reproducibility/h2oplus_v14_terminal_taper_registration_v1.json",
        "authorization": repo / "reproducibility/h2oplus_v14_terminal_taper_authorization_v1.json",
        "freeze": repo / "reproducibility/h2oplus_v14_terminal_taper_freeze_v1.json",
        "design": repo / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "protocol": repo / "reproducibility/h2oplus_v14_terminal_taper_protocol_v1.json",
        "training_closeout": repo / "reproducibility/h2oplus_v14_terminal_taper_branch_closeout_v1.json",
        "training_verification": repo / "reproducibility/h2oplus_v14_terminal_taper_branch_closeout_verification_v1.json",
        "training_audit": repo / "reproducibility/h2oplus_v14_terminal_taper_results/branch_audit_v1.json",
        "training_file_manifest": repo / "reproducibility/h2oplus_v14_terminal_taper_results/branch_file_manifest_v1.json",
        "source_freeze": repo / "reproducibility/h2oplus_v14_terminal_taper_evaluation_source_freeze_v1.json",
        "methods": repo / "reproducibility/h2oplus_v14_terminal_taper_evaluation_methods_v1.json",
        "dataset": repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": deployment_root / "deployment_manifest.json",
    }
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"local frozen input drifted: {name}")
    manifest = json.loads(paths["deployment_manifest"].read_text())
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v14-terminal-taper-evaluation-deployment-manifest-v1"
        or manifest.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or manifest.get("entry_count") != args.expected_entry_count
        or manifest.get("physical_actor_count") != 70
        or manifest.get("logical_alias_count") != 180
        or manifest.get("expected_remote_sumo_episodes") != 630
        or manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("v14 deployment manifest drifted")

    signatures = [
        f"H2Oplus/v14/development/terminal-taper-eval/{hashes['deployment_manifest'][:12]}/shard{index}/jtl110gpu/glibc227"
        for index in range(2)
    ]
    queue = json.loads((Path.home() / ".claude/scheduler/queue.json").read_text())
    tasks = queue if isinstance(queue, list) else queue.get("tasks", queue.get("queue", []))
    duplicates = [
        {"id": task.get("id"), "status": task.get("status"), "signature": task.get("signature")}
        for task in tasks
        if task.get("signature") in signatures
    ]
    if duplicates:
        raise ValueError(f"refusing duplicate v14 evaluation submission: {duplicates}")

    ready = {
        "status": "validated",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "manifest_sha256": hashes["deployment_manifest"],
        "deployment_aggregate_sha256": hashes["deployment_aggregate"],
        "physical_actor_count": 70,
        "logical_method_budget_alias_count": 180,
        "development_block_count": 3,
        "demand_scale_count": 3,
        "jobs_per_shard": [315, 315],
        "total_remote_sumo_episodes": 630,
        "duplicate_signatures": duplicates,
    }
    if args.validate_only:
        print(json.dumps(ready, indent=2, sort_keys=True))
        return 0

    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(repo),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    dispatcher = repo / "reproducibility/run_h2oplus_v14_terminal_taper_evaluation_dispatch.py"
    remote_results_root = deployment_root.parent / "remote_results_v1"
    local_results_root = REPRO / "h2oplus_v14_terminal_taper_results/evaluation_partitions_v1"
    sumo_dir = deployment_root / "SUMO_ruiguang/online_control"
    edge_xml = repo / "bus_h2o/network_data/a_sorted_busline_edge.xml"
    submitted = []
    jobs_per_shard = (315, 315)
    for shard_index, signature in enumerate(signatures):
        remote_result = remote_results_root / f"shard{shard_index}_jtl110gpu"
        local_result = local_results_root / f"shard{shard_index}_jtl110gpu"
        command_parts = [
            python,
            str(dispatcher),
            "--registration", str(paths["registration"]),
            "--authorization", str(paths["authorization"]),
            "--freeze", str(paths["freeze"]),
            "--environment-design", str(paths["design"]),
            "--protocol", str(paths["protocol"]),
            "--training-closeout", str(paths["training_closeout"]),
            "--training-verification", str(paths["training_verification"]),
            "--training-audit", str(paths["training_audit"]),
            "--training-file-manifest", str(paths["training_file_manifest"]),
            "--source-freeze", str(paths["source_freeze"]),
            "--methods-manifest", str(paths["methods"]),
            "--dataset", str(paths["dataset"]),
            "--sumo-runtime-lock", str(paths["runtime_lock"]),
            "--deployment-manifest", str(paths["deployment_manifest"]),
            "--sumo-dir", str(sumo_dir),
            "--edge-xml", str(edge_xml),
            "--expected-registration-sha256", hashes["registration"],
            "--expected-authorization-sha256", hashes["authorization"],
            "--expected-freeze-sha256", hashes["freeze"],
            "--expected-environment-design-sha256", hashes["design"],
            "--expected-protocol-sha256", hashes["protocol"],
            "--expected-training-closeout-sha256", hashes["training_closeout"],
            "--expected-training-verification-sha256", hashes["training_verification"],
            "--expected-training-audit-sha256", hashes["training_audit"],
            "--expected-training-file-manifest-sha256", hashes["training_file_manifest"],
            "--expected-source-freeze-sha256", hashes["source_freeze"],
            "--expected-methods-sha256", hashes["methods"],
            "--expected-dataset-sha256", hashes["dataset"],
            "--expected-runtime-lock-sha256", hashes["runtime_lock"],
            "--expected-deployment-manifest-sha256", hashes["deployment_manifest"],
            "--output-root", str(remote_result),
            "--execution-json", str(remote_result / "host_execution.json"),
            "--workers", "12",
            "--shard-count", "2",
            "--shard-index", str(shard_index),
        ]
        command = shlex.join(command_parts)
        submit = [
            "python3", str(scheduler), "submit",
            "--description",
            f"H2O+ v14 terminal-taper development evaluation shard {shard_index} "
            f"({jobs_per_shard[shard_index]} CPU episodes)",
            "--cmd", command,
            "--cwd", str(repo),
            "--signature", signature,
            "--resource-family", "h2oplus-v14-terminal-taper-eval-worker-v1",
            "--vram", "0",
            "--ram-mb", "32768",
            "--cpu", "12",
            "--priority", "high",
            "--project", "h2oplus-v14-development",
            "--require-node", "jtl110gpu",
            "--result-dir", str(remote_result),
            "--local-result-dir", str(local_result),
            "--env-spec", "none",
            "--allow-remote-large-data",
            "--skip-launch-staging",
            "--env", *[f"{key}={value}" for key, value in environment.items()],
        ]
        result = subprocess.run(
            submit, cwd=scheduler_root, text=True, capture_output=True, check=False
        )
        combined = "\n".join(value for value in (result.stdout, result.stderr) if value)
        match = re.search(r"\bt\d+\b", combined)
        submitted.append(
            {
                "shard_index": shard_index,
                "task_id": match.group(0) if match else None,
                "returncode": result.returncode,
                "scheduler_output": combined.strip(),
                "signature": signature,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "require_node": "jtl110gpu",
                "cpu_cores": 12,
                "ram_mb": 32768,
                "est_vram_mb": 0,
                "expected_jobs": jobs_per_shard[shard_index],
                "remote_result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
        if result.returncode or submitted[-1]["task_id"] is None:
            break
    status = (
        "submitted"
        if len(submitted) == 2 and all(not item["returncode"] for item in submitted)
        else "partial-failure"
    )
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "physical_actor_count": 70,
        "logical_method_budget_alias_count": 180,
        "development_block_count": 3,
        "demand_scale_count": 3,
        "shard_count": 2,
        "jobs_per_shard": list(jobs_per_shard),
        "total_remote_sumo_episodes": 630,
        "natural_serial_execution_on_one_12_core_host": True,
        "required_environment": environment,
        "cuda_visible_devices_managed_by_scheduler": True,
        "immutable_hashes": hashes,
        "local_verification": descriptor(local_verification_path),
        "remote_verification": descriptor(remote_verification_path),
        "deployment_sync": descriptor(sync_path),
        "training_stability_separate_from_artifact_integrity": {
            "all_15_passed": True,
            "failure_runs": [],
            "development_evaluation_release_gate_open": True,
        },
        "tasks": submitted,
        "next_stage": (
            "after both tasks have token-bound backend exit0 and exactly 315/315 valid "
            "results, synchronize, merge 630 physical evaluations, independently audit "
            "every file, and apply all preregistered development gates"
        ),
    }
    record = args.record.expanduser().resolve()
    atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
