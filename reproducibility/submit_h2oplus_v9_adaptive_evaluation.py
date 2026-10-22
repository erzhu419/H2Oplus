#!/usr/bin/env python3
"""Submit the frozen v9 adaptive development evaluation through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-submission-v3"

HASHES = {
    "registration": "f4f6922324ef518acde9b5489328bf13c7154012dfd6f4cc2b61797adb90785e",
    "authorization": "c99b2cb637018a67c04d3be8e088195c8ec9c4d05f3fcf02ca7e672b967f0fa6",
    "freeze": "b3265d5e55e4df53afacd50c9f54cc3530a5d8f28a7d3eceec217655dc9bd822",
    "design": "cf6faac3b684b4bddfd1391925d67d7c3d5202978bd2c6d02e6326a6bb5dd8f5",
    "protocol": "e0652db511c5ca46e0e14801895bdfdc57761223842a4e30673075b9163d3c7d",
    "count_correction": "4fc9324b140f6af387a4b8011b40c1d6994c57636d909ca065b17d4d0e27c6ba",
    "training_closeout": "fc263b3d51ac5a532bd13dc3fdaecf43153e1336d9d57a4ce6b87e6eac18bafd",
    "training_verification": "9926dea64d692619808c2a2e03382e7620fd71358223119ac066d217ceff080e",
    "training_audit": "f126301d52ccdc665c8c1fdc20611c5f87d7879fcce14296d4984b47de997959",
    "training_file_manifest": "010ddf60b394230cdeb28edd32e4df52d189d754c1a2a2301e9c40f0ed46f4ce",
    "source_freeze": "f717bebbe0ed838c104ebb716161fcc7d9be79b64ed9ac3b207ba638139a37cc",
    "source_correction": "2e28bd6dfe71e3d1adf82866137da0a8a678048c7a6a91aaa00df8b39e782f74",
    "methods": "e1c83b52c78dc60d4a8da8888f3b2a0d82f2c6ea77ca4b9d372d56a33eb58f0c",
    "portability_correction": "b5063671012b48c55f537ab5f017356ff2a72229f59aa74b5eabf67214842d23",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": "64180676bf222d8cf8877c9a3c3d80b8f7aa86d60ee2a60e042bc28b242ff0e3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
        default=REPRO / "deployments/h2oplus_v9_adaptive_evaluation_v3/workspace",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v9_adaptive_evaluation_deployment_v3_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_evaluation_submission_v3.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    repo = deployment_root / "H2Oplus"
    verification_path = args.remote_verification.expanduser().resolve(strict=True)
    verification = json.loads(verification_path.read_text())
    if (
        verification.get("passed") is not True
        or verification.get("node") != "jtl110gpu"
        or verification.get("formal_held_out_outcomes_read") is not False
        or verification.get("remote_sumo_simulation_executed") is not False
        or verification.get("expected", {}).get("deployment_manifest_sha256")
        not in (None, HASHES["deployment_manifest"])
        or verification.get("expected", {}).get("manifest_sha256")
        != HASHES["deployment_manifest"]
    ):
        raise ValueError("remote deployment verification is stale or failed")

    paths = {
        "registration": repo / "reproducibility/h2oplus_v9_adaptive_registration_v2.json",
        "authorization": repo / "reproducibility/h2oplus_v9_adaptive_authorization_v2.json",
        "freeze": repo / "reproducibility/h2oplus_v9_adaptive_freeze_v2.json",
        "design": repo / "reproducibility/h2oplus_v9_adaptive_environment_v2.json",
        "protocol": repo / "reproducibility/h2oplus_v9_adaptive_protocol_v2.json",
        "count_correction": repo / "reproducibility/h2oplus_v9_adaptive_evaluation_count_correction_v1.json",
        "training_closeout": repo / "reproducibility/h2oplus_v9_adaptive_results/training_closeout.json",
        "training_verification": repo / "reproducibility/h2oplus_v9_adaptive_results/training_file_manifest_verification.json",
        "training_audit": repo / "reproducibility/h2oplus_v9_adaptive_results/training_audit.json",
        "training_file_manifest": repo / "reproducibility/h2oplus_v9_adaptive_results/training_file_manifest.json",
        "source_freeze": repo / "reproducibility/h2oplus_v9_adaptive_evaluation_source_freeze_v2.json",
        "source_correction": repo / "reproducibility/h2oplus_v9_adaptive_evaluation_source_correction_v2.json",
        "methods": repo / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v2.json",
        "portability_correction": repo / "reproducibility/h2oplus_v9_adaptive_evaluation_portability_correction_v1.json",
        "dataset": repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": deployment_root / "deployment_manifest.json",
    }
    for name, path in paths.items():
        if sha256(path) != HASHES[name]:
            raise ValueError(f"local frozen input drifted: {name}")

    signatures = [
        f"H2Oplus/v9/development/adaptive-eval/{HASHES['deployment_manifest'][:12]}/shard{index}/jtl110gpu/glibc227"
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
        raise ValueError(f"refusing duplicate v9 evaluation submission: {duplicates}")

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
    dispatcher = repo / "reproducibility/run_h2oplus_v9_adaptive_evaluation_dispatch.py"
    remote_results_root = deployment_root.parent / "remote_results_v3"
    local_results_root = REPRO / "h2oplus_v9_adaptive_results/evaluation_partitions_v3"
    sumo_dir = deployment_root / "SUMO_ruiguang/online_control"
    edge_xml = repo / "bus_h2o/network_data/a_sorted_busline_edge.xml"
    submitted = []
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
            "--count-correction", str(paths["count_correction"]),
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
            "--expected-design-sha256", HASHES["design"],
            "--expected-protocol-sha256", HASHES["protocol"],
            "--expected-count-correction-sha256", HASHES["count_correction"],
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
            "--description", f"H2O+ v9 adaptive development evaluation shard {shard_index} (126 CPU episodes)",
            "--cmd", command,
            "--cwd", str(repo),
            "--signature", signature,
            "--resource-family", "h2oplus-v9-adaptive-eval-worker-v1",
            "--vram", "0",
            "--ram-mb", "32768",
            "--cpu", "12",
            "--priority", "high",
            "--project", "h2oplus-v9-development",
            "--require-node", "jtl110gpu",
            "--result-dir", str(remote_result),
            "--local-result-dir", str(local_result),
            "--env-spec", "none",
            "--allow-remote-large-data",
            "--skip-launch-staging",
            "--env", *[f"{key}={value}" for key, value in environment.items()],
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
                "remote_result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
        if result.returncode or submitted[-1]["task_id"] is None:
            break

    status = "submitted" if len(submitted) == 2 and all(not item["returncode"] for item in submitted) else "partial-failure"
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "physical_actor_count": 42,
        "logical_method_budget_alias_count": 60,
        "development_block_count": 2,
        "demand_scale_count": 3,
        "shard_count": 2,
        "jobs_per_shard": 126,
        "total_remote_sumo_episodes": 252,
        "natural_serial_execution_on_one_12_core_host": True,
        "required_environment": environment,
        "cuda_visible_devices_managed_by_scheduler": True,
        "immutable_hashes": HASHES,
        "remote_verification": descriptor(verification_path),
        "superseded_invalid_v1_deployment_submission": {
            **descriptor(REPRO / "h2oplus_v9_adaptive_evaluation_submission_v2.json"),
            "reason": (
                "t82565 failed before SUMO because the v1 methods manifest retained a "
                "build-host absolute path; t82566 was cancelled before launch"
            ),
        },
        "portability_correction": descriptor(
            REPRO / "h2oplus_v9_adaptive_evaluation_portability_correction_v1.json"
        ),
        "source_correction": descriptor(
            REPRO / "h2oplus_v9_adaptive_evaluation_source_correction_v2.json"
        ),
        "tasks": submitted,
        "next_stage": (
            "after both tasks have token-bound backend exit 0 and exactly 126 valid results, "
            "synchronize, independently audit 252 files, seal a file manifest, and apply all "
            "preregistered development gates"
        ),
    }
    record = args.record.expanduser().resolve()
    atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
