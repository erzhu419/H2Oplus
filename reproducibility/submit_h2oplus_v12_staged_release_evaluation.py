#!/usr/bin/env python3
"""Submit the frozen v12 staged-release development evaluation through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v12-staged-release-evaluation-submission-v1"
HASHES = {
    "registration": "59cb1e14ad0bbc8944472f5bbf3e849a62b65c4517766e3fd3845b76a5ec3ba3",
    "authorization": "bb1eeda1c451916aba7576f0592418471f42afaefe499386482be540def7fa23",
    "freeze": "0d8ab1e7852c679ae6e444d1c767799635c8790e4bc787587210a63b0e92dfe1",
    "design": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "protocol": "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
    "correction": "40dabd0dddab6ce7fd8f08ac025f8cc96a5545d1c36ff24730ce885b5eb67b25",
    "training_closeout": "998f47488b50b2cd40f9a6ddd5a399712bb9208cb1ef883cad1181f800385b66",
    "training_verification": "6835e3c545be22c25bd9e3c7eae7ef987bb1864e027c4fef9849788f744868c8",
    "training_audit": "8e34c4cd69f86d7333dfd5d2a57696ce427388349c07c377bb279b957fd1bfc7",
    "training_file_manifest": "9ed7ccbddf0832e0c8b4532421e000745f9c60f03b9dfc01a40d970503c2b08b",
    "source_freeze": "2ab7b6c4d8eb94977df097ab37ff8671becdd11fbf4c2ffe562eecb6bce6cd28",
    "methods": "d61acae2314d0588b0dc6aa84c0a0bf2f168d4bdf1f18c5d19b2d0dbdec5c374",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": "68155a903e305ce04cd1654487502ae5356219ba43b1fc953f474efc5eb8e973",
    "deployment_aggregate": "3193c5a556c81b7f3103dedd4bec4065b93db4290d67c94fce20d819eb29b355",
    "local_verification": "8dd611ab2d434fd03e9ab445e705fc9023e6ebd47dd2d30f3dc42fd5b14d8c06",
    "remote_verification": "ea3e403f44482ff92c3fd4758198dea62a40c914472a71df6875a722d887f038",
    "deployment_sync": "f882c926cb6badd8699d49a95d79ba4b5d7a6ffdad31d69d82ff240684cd0b56",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v12_staged_release_evaluation_v1/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_evaluation_deployment_local_verification_v1.json",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--deployment-sync",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_evaluation_deployment_sync_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_evaluation_submission_v1.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        if sha256(path) != HASHES[name]:
            raise ValueError(f"{name} identity drifted")
    local_verification = json.loads(local_verification_path.read_text())
    remote_verification = json.loads(remote_verification_path.read_text())
    sync = json.loads(sync_path.read_text())
    if (
        local_verification.get("passed") is not True
        or local_verification.get("manifest_sha256") != HASHES["deployment_manifest"]
        or remote_verification.get("passed") is not True
        or remote_verification.get("node") != "jtl110gpu"
        or remote_verification.get("formal_held_out_outcomes_read") is not False
        or remote_verification.get("remote_sumo_simulation_executed") is not False
        or remote_verification.get("expected", {}).get("manifest_sha256")
        != HASHES["deployment_manifest"]
        or sync.get("passed") is not True
        or sync.get("manifest_sha256") != HASHES["deployment_manifest"]
    ):
        raise ValueError("v12 deployment evidence is stale or failed")

    paths = {
        "registration": repo / "reproducibility/h2oplus_v12_staged_release_registration_v2.json",
        "authorization": repo / "reproducibility/h2oplus_v12_staged_release_authorization_v2.json",
        "freeze": repo / "reproducibility/h2oplus_v12_staged_release_freeze_v2.json",
        "design": repo / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "protocol": repo / "reproducibility/h2oplus_v12_staged_release_protocol_v2.json",
        "correction": repo / "reproducibility/h2oplus_v12_staged_release_deployment_correction_v2.json",
        "training_closeout": repo / "reproducibility/h2oplus_v12_staged_release_branch_closeout_v2.json",
        "training_verification": repo / "reproducibility/h2oplus_v12_staged_release_branch_closeout_verification_v2.json",
        "training_audit": repo / "reproducibility/h2oplus_v12_staged_release_results/branch_audit_v2.json",
        "training_file_manifest": repo / "reproducibility/h2oplus_v12_staged_release_results/branch_file_manifest_v2.json",
        "source_freeze": repo / "reproducibility/h2oplus_v12_staged_release_evaluation_source_freeze_v1.json",
        "methods": repo / "reproducibility/h2oplus_v12_staged_release_evaluation_methods_v1.json",
        "dataset": repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": deployment_root / "deployment_manifest.json",
    }
    for name, path in paths.items():
        if sha256(path) != HASHES[name]:
            raise ValueError(f"local frozen input drifted: {name}")
    manifest = json.loads(paths["deployment_manifest"].read_text())
    if (
        manifest.get("aggregate_sha256") != HASHES["deployment_aggregate"]
        or manifest.get("entry_count") != 612
    ):
        raise ValueError("v12 deployment manifest drifted")

    signatures = [
        f"H2Oplus/v12/development/staged-release-eval/{HASHES['deployment_manifest'][:12]}/shard{index}/jtl110gpu/glibc227"
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
        raise ValueError(f"refusing duplicate v12 evaluation submission: {duplicates}")

    ready = {
        "status": "validated",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "manifest_sha256": HASHES["deployment_manifest"],
        "deployment_aggregate_sha256": HASHES["deployment_aggregate"],
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
    dispatcher = repo / "reproducibility/run_h2oplus_v12_staged_release_evaluation_dispatch.py"
    remote_results_root = deployment_root.parent / "remote_results_v1"
    local_results_root = REPRO / "h2oplus_v12_staged_release_results/evaluation_partitions_v1"
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
            "--correction", str(paths["correction"]),
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
            "--expected-registration-sha256", HASHES["registration"],
            "--expected-authorization-sha256", HASHES["authorization"],
            "--expected-freeze-sha256", HASHES["freeze"],
            "--expected-environment-design-sha256", HASHES["design"],
            "--expected-protocol-sha256", HASHES["protocol"],
            "--expected-correction-sha256", HASHES["correction"],
            "--expected-training-closeout-sha256", HASHES["training_closeout"],
            "--expected-training-verification-sha256", HASHES["training_verification"],
            "--expected-training-audit-sha256", HASHES["training_audit"],
            "--expected-training-file-manifest-sha256", HASHES["training_file_manifest"],
            "--expected-source-freeze-sha256", HASHES["source_freeze"],
            "--expected-methods-sha256", HASHES["methods"],
            "--expected-dataset-sha256", HASHES["dataset"],
            "--expected-runtime-lock-sha256", HASHES["runtime_lock"],
            "--expected-deployment-manifest-sha256", HASHES["deployment_manifest"],
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
            f"H2O+ v12 staged-release development evaluation shard {shard_index} "
            f"({jobs_per_shard[shard_index]} CPU episodes)",
            "--cmd", command,
            "--cwd", str(repo),
            "--signature", signature,
            "--resource-family", "h2oplus-v12-staged-release-eval-worker-v1",
            "--vram", "0",
            "--ram-mb", "32768",
            "--cpu", "12",
            "--priority", "high",
            "--project", "h2oplus-v12-development",
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
        "immutable_hashes": HASHES,
        "local_verification": descriptor(local_verification_path),
        "remote_verification": descriptor(remote_verification_path),
        "deployment_sync": descriptor(sync_path),
        "training_stability_separate_from_artifact_integrity": {
            "all_30_passed": True,
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
