#!/usr/bin/env python3
"""Submit the frozen v10 delayed development evaluation through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v10-delayed-evaluation-submission-v1"
HASHES = {
    "registration": "59fbc0c7f1148fe74c4c8a0c8af0879c7900753520b5b112e0526eb3c05acf56",
    "authorization": "7aa6e68db01503dc9e1c3bbfbce1639dd08890e5fb46fec2cda347ade6942741",
    "freeze": "3776bf088ba7c7cf3c2e65fb8407afb2adb9c8fbcc2266b7781bfd8d1b04cb64",
    "design": "ee3de3b3990e35c579f5fcfdbafe16168a83f5011f9e5b5ce44b59cea5365bb4",
    "protocol": "83a45e7d06ec30bdaba3631d7d8c6854e7609b8dcff9c46aa735d9a7944daaf3",
    "correction": "2bf2a6d707283ae694e18ce796a209810415a2e079ca6b023c88a7b2e6857163",
    "branch_closeout": "9204b4196f40a7fa4ce0bab1277d8c7bbbe930aafa644760b7251d463fbc6498",
    "branch_verification": "101cadbfc03bde2719f0a786a961006ed2d1d86a27080376afd59842ff4bcb17",
    "branch_audit": "e851b3413d19062b2e981df89bfc92bdd9908f256bc80031cf3ae4c13328fabe",
    "branch_file_manifest": "d794161e0faaec6d5ebbcd2039aab25c0051166802d577743188c8e55e29c926",
    "prefix_closeout": "6a05ddc9ef1a3a07605851360d3733d6442f6aa27e4fb2b65d69f4b2a25951ad",
    "prefix_verification": "1d1d8b592485d24038cf363b233a4228c96d4a156c00804c6eadfbe3b0906c65",
    "source_freeze": "449d5432720b1b59ff9865b4bb83bb15ea1a6f5c0096ea01cad33766a9c73d9f",
    "methods": "dca00ae59b34e0810ec407c0df3a08583d35eac2b404a0b879866922b2d0d9f7",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": "ebb36040134679458f7ee1e818e8a3631c6929916b95e4b627542defc9fa2d4c",
    "deployment_aggregate": "2da927fe95618ddfc28e0e2d795c69c379760eee770f77d323e38fd829ad23e8",
    "remote_verification": "9dbffa503a52d8cbe60918d2b9b35251b681e5499239d4137039ca9847ab778e",
    "deployment_sync": "ae879be0635b8b0816ef1c8d2c534237aba4d1d30454d0070d4a0018284e1ab5",
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
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v10_delayed_evaluation_v1/workspace",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--deployment-sync",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_evaluation_deployment_sync_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_evaluation_submission_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    scheduler = scheduler_root / "skill/scheduler.py"
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    repo = deployment_root / "H2Oplus"
    verification_path = args.remote_verification.expanduser().resolve(strict=True)
    sync_path = args.deployment_sync.expanduser().resolve(strict=True)
    if sha256(verification_path) != HASHES["remote_verification"]:
        raise ValueError("remote verification identity drifted")
    if sha256(sync_path) != HASHES["deployment_sync"]:
        raise ValueError("deployment sync identity drifted")
    verification = json.loads(verification_path.read_text())
    sync = json.loads(sync_path.read_text())
    if (
        verification.get("passed") is not True
        or verification.get("node") != "jtl110gpu"
        or verification.get("formal_held_out_outcomes_read") is not False
        or verification.get("remote_sumo_simulation_executed") is not False
        or verification.get("expected", {}).get("manifest_sha256") != HASHES["deployment_manifest"]
        or sync.get("passed") is not True
        or sync.get("manifest_sha256") != HASHES["deployment_manifest"]
    ):
        raise ValueError("remote v10 deployment evidence is stale or failed")

    paths = {
        "registration": repo / "reproducibility/h2oplus_v10_delayed_registration_v4.json",
        "authorization": repo / "reproducibility/h2oplus_v10_delayed_authorization_v4.json",
        "freeze": repo / "reproducibility/h2oplus_v10_delayed_freeze_v4.json",
        "design": repo / "reproducibility/h2oplus_v10_delayed_environment_v4.json",
        "protocol": repo / "reproducibility/h2oplus_v10_delayed_protocol_v4.json",
        "correction": repo / "reproducibility/h2oplus_v10_delayed_correction_v4.json",
        "branch_closeout": repo / "reproducibility/h2oplus_v10_delayed_results/branch_closeout_v4.json",
        "branch_verification": repo / "reproducibility/h2oplus_v10_delayed_results/branch_closeout_verification_v4.json",
        "branch_audit": repo / "reproducibility/h2oplus_v10_delayed_results/branch_audit_v4.json",
        "branch_file_manifest": repo / "reproducibility/h2oplus_v10_delayed_results/branch_file_manifest_v4.json",
        "prefix_closeout": repo / "reproducibility/h2oplus_v10_delayed_results/prefix_closeout_v4.json",
        "prefix_verification": repo / "reproducibility/h2oplus_v10_delayed_results/prefix_closeout_verification_v4.json",
        "source_freeze": repo / "reproducibility/h2oplus_v10_delayed_evaluation_source_freeze_v1.json",
        "methods": repo / "reproducibility/h2oplus_v10_delayed_evaluation_methods_v1.json",
        "dataset": repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": deployment_root / "deployment_manifest.json",
    }
    for name, path in paths.items():
        if name in HASHES and sha256(path) != HASHES[name]:
            raise ValueError(f"local frozen input drifted: {name}")
    manifest = json.loads(paths["deployment_manifest"].read_text())
    if manifest.get("aggregate_sha256") != HASHES["deployment_aggregate"]:
        raise ValueError("deployment aggregate drifted")

    signatures = [
        f"H2Oplus/v10/development/delayed-eval/{HASHES['deployment_manifest'][:12]}/shard{index}/jtl110gpu/glibc227"
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
        raise ValueError(f"refusing duplicate v10 evaluation submission: {duplicates}")

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
    dispatcher = repo / "reproducibility/run_h2oplus_v10_delayed_evaluation_dispatch.py"
    remote_results_root = deployment_root.parent / "remote_results_v1"
    local_results_root = REPRO / "h2oplus_v10_delayed_results/evaluation_partitions_v1"
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
            "--correction", str(paths["correction"]),
            "--branch-closeout", str(paths["branch_closeout"]),
            "--branch-verification", str(paths["branch_verification"]),
            "--branch-audit", str(paths["branch_audit"]),
            "--branch-file-manifest", str(paths["branch_file_manifest"]),
            "--prefix-closeout", str(paths["prefix_closeout"]),
            "--prefix-verification", str(paths["prefix_verification"]),
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
            "--expected-correction-sha256", HASHES["correction"],
            "--expected-branch-closeout-sha256", HASHES["branch_closeout"],
            "--expected-branch-verification-sha256", HASHES["branch_verification"],
            "--expected-branch-audit-sha256", HASHES["branch_audit"],
            "--expected-branch-file-manifest-sha256", HASHES["branch_file_manifest"],
            "--expected-prefix-closeout-sha256", HASHES["prefix_closeout"],
            "--expected-prefix-verification-sha256", HASHES["prefix_verification"],
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
            "--description", f"H2O+ v10 delayed development evaluation shard {shard_index} (270 CPU episodes)",
            "--cmd", command,
            "--cwd", str(repo),
            "--signature", signature,
            "--resource-family", "h2oplus-v10-delayed-eval-worker-v1",
            "--vram", "0",
            "--ram-mb", "32768",
            "--cpu", "12",
            "--priority", "high",
            "--project", "h2oplus-v10-development",
            "--require-node", "jtl110gpu",
            "--result-dir", str(remote_result),
            "--local-result-dir", str(local_result),
            "--env-spec", "none",
            "--allow-remote-large-data",
            "--skip-launch-staging",
            "--env", *[f"{key}={value}" for key, value in environment.items()],
        ]
        result = subprocess.run(submit, cwd=scheduler_root, text=True, capture_output=True, check=False)
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
        "physical_actor_count": 60,
        "logical_method_budget_alias_count": 140,
        "development_block_count": 3,
        "demand_scale_count": 3,
        "shard_count": 2,
        "jobs_per_shard": 270,
        "total_remote_sumo_episodes": 540,
        "natural_serial_execution_on_one_12_core_host": True,
        "required_environment": environment,
        "cuda_visible_devices_managed_by_scheduler": True,
        "immutable_hashes": HASHES,
        "remote_verification": descriptor(verification_path),
        "deployment_sync": descriptor(sync_path),
        "training_stability_separate_from_artifact_integrity": {
            "all_30_passed": False,
            "failure_runs": ["h2oplus_v10c4_delayed_order_seed6205"],
            "formal_eligibility_already_blocked": True,
        },
        "tasks": submitted,
        "next_stage": (
            "after both tasks have token-bound backend exit0 and exactly 270 valid results, "
            "synchronize, merge 540 physical evaluations, independently audit every file, "
            "and apply all preregistered development gates"
        ),
    }
    record = args.record.expanduser().resolve()
    atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
