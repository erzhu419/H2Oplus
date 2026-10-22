#!/usr/bin/env python3
"""Submit the frozen v11 actor-hold development evaluation through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v11-actor-hold-evaluation-submission-v1"
HASHES = {
    "registration": "caab9c2c3aa7ee2ce977c13403b415be9b0debc2d7c88fc6b7a12c51b3e9efb5",
    "authorization": "8dc0ca71f8c45abc3e5e86af789281e4e43a882533ba4202ca567f00d87281aa",
    "freeze": "095754d08f14b6e2eb84baadfa6ce1a8d4100ebac5b92f34891b2d1e01afed04",
    "design": "0f005b94d5ecd361d7e10bac363be8dac5e6c9cdad7cbaf3f063744d280871de",
    "protocol": "c5b162541c6ce2476bb0f57f6dc936de13b85b5e1e6ede77b533a5cf9a4dfced",
    "correction": "8f1defa7d756f6336d26af0b1fe8ebf87db39f81554c89891fc2fbaaf17d1258",
    "training_closeout": "489c7efe5cf91f73bb31a9b6df03441c2fb3bcd4d280c65493c5ae8ec3af01a6",
    "training_verification": "ff0ef7c69bac850bf9f743275ffd9bbfc87c0b1c9713a83769a052bd936d256e",
    "training_audit": "623ae58932c5ec3ffb80fbe82b2d159c6f0fe96ee0a1f12eeb4ca1c49385d516",
    "training_file_manifest": "0a70f10026e1a80d14176648eea866e3e52c00a56ba30acfc48b414e323025fa",
    "source_freeze": "fe61024af63450c6502767c2fcee7915c2b2e8216cf23812f63523e241b237c7",
    "methods": "6716c56c73e9d491431b1ab019727f9bfd18a2242cb7500826cf7ccc0628b260",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": "b579fade4a0c4ea6dac164620dd50818818fc49666d365ddff63a26ab989246d",
    "deployment_aggregate": "c8db97ab6d63000df936b987c7bc1eff88dd1dba8ca4d4a9baff3e698cb16741",
    "local_verification": "70b8b4473b7c4c24a146a1c25a276f7bb0b7bde642d774461bbca62dadede40a",
    "remote_verification": "e393ebcb2f025cd485e5dd455baa83467c0bfed0c86b2e0ef66d84501afd1446",
    "deployment_sync": "75dcacf761206bcd64f7b74fc2b7668c843f3b12f3dfa5302599a5db68d885a4",
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
        default=REPRO / "deployments/h2oplus_v11_actor_hold_evaluation_v1/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_evaluation_deployment_local_verification_v1.json",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--deployment-sync",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_evaluation_deployment_sync_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_evaluation_submission_v1.json",
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
        raise ValueError("v11 deployment evidence is stale or failed")

    paths = {
        "registration": repo / "reproducibility/h2oplus_v11_actor_hold_registration_v2.json",
        "authorization": repo / "reproducibility/h2oplus_v11_actor_hold_authorization_v2.json",
        "freeze": repo / "reproducibility/h2oplus_v11_actor_hold_freeze_v2.json",
        "design": repo / "reproducibility/h2oplus_v11_actor_hold_environment_v2.json",
        "protocol": repo / "reproducibility/h2oplus_v11_actor_hold_protocol_v2.json",
        "correction": repo / "reproducibility/h2oplus_v11_actor_hold_correction_v2.json",
        "training_closeout": repo / "reproducibility/h2oplus_v11_actor_hold_results/training_closeout_v2.json",
        "training_verification": repo / "reproducibility/h2oplus_v11_actor_hold_results/training_closeout_verification_v2.json",
        "training_audit": repo / "reproducibility/h2oplus_v11_actor_hold_results/training_audit_v2.json",
        "training_file_manifest": repo / "reproducibility/h2oplus_v11_actor_hold_results/training_file_manifest_v2.json",
        "source_freeze": repo / "reproducibility/h2oplus_v11_actor_hold_evaluation_source_freeze_v1.json",
        "methods": repo / "reproducibility/h2oplus_v11_actor_hold_evaluation_methods_v1.json",
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
        or manifest.get("entry_count") != 545
    ):
        raise ValueError("v11 deployment manifest drifted")

    signatures = [
        f"H2Oplus/v11/development/actor-hold-eval/{HASHES['deployment_manifest'][:12]}/shard{index}/jtl110gpu/glibc227"
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
        raise ValueError(f"refusing duplicate v11 evaluation submission: {duplicates}")

    ready = {
        "status": "validated",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "manifest_sha256": HASHES["deployment_manifest"],
        "deployment_aggregate_sha256": HASHES["deployment_aggregate"],
        "physical_actor_count": 45,
        "logical_method_budget_alias_count": 80,
        "development_block_count": 3,
        "demand_scale_count": 3,
        "jobs_per_shard": [204, 201],
        "total_remote_sumo_episodes": 405,
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
    dispatcher = repo / "reproducibility/run_h2oplus_v11_actor_hold_evaluation_dispatch.py"
    remote_results_root = deployment_root.parent / "remote_results_v1"
    local_results_root = REPRO / "h2oplus_v11_actor_hold_results/evaluation_partitions_v1"
    sumo_dir = deployment_root / "SUMO_ruiguang/online_control"
    edge_xml = repo / "bus_h2o/network_data/a_sorted_busline_edge.xml"
    submitted = []
    jobs_per_shard = (204, 201)
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
            f"H2O+ v11 actor-hold development evaluation shard {shard_index} ({jobs_per_shard[shard_index]} CPU episodes)",
            "--cmd", command,
            "--cwd", str(repo),
            "--signature", signature,
            "--resource-family", "h2oplus-v11-actor-hold-eval-worker-v1",
            "--vram", "0",
            "--ram-mb", "32768",
            "--cpu", "12",
            "--priority", "high",
            "--project", "h2oplus-v11-development",
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
        "physical_actor_count": 45,
        "logical_method_budget_alias_count": 80,
        "development_block_count": 3,
        "demand_scale_count": 3,
        "shard_count": 2,
        "jobs_per_shard": list(jobs_per_shard),
        "total_remote_sumo_episodes": 405,
        "natural_serial_execution_on_one_12_core_host": True,
        "required_environment": environment,
        "cuda_visible_devices_managed_by_scheduler": True,
        "immutable_hashes": HASHES,
        "local_verification": descriptor(local_verification_path),
        "remote_verification": descriptor(remote_verification_path),
        "deployment_sync": descriptor(sync_path),
        "training_stability_separate_from_artifact_integrity": {
            "all_5_passed": True,
            "failure_runs": [],
            "development_evaluation_release_gate_open": True,
        },
        "tasks": submitted,
        "next_stage": (
            "after both tasks have token-bound backend exit0 and exactly 204/201 valid "
            "results, synchronize, merge 405 physical evaluations, independently audit "
            "every file, and apply all preregistered development gates"
        ),
    }
    record = args.record.expanduser().resolve()
    atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
