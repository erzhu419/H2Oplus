#!/usr/bin/env python3
"""Submit the frozen v15 development evaluation through scheduleurm."""

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
SCHEMA = "transit-h2oplus-v15-hold-calibration-evaluation-submission-v1"


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
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v15_hold_calibration_evaluation_v1/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v15_hold_calibration_evaluation_deployment_local_verification_v1.json",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v15_hold_calibration_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    parser.add_argument(
        "--deployment-sync",
        type=Path,
        default=REPRO
        / "h2oplus_v15_hold_calibration_evaluation_deployment_sync_v1.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_evaluation_submission_v1.json",
    )
    names = (
        "registration",
        "authorization",
        "environment_design",
        "protocol",
        "v14_analysis",
        "v14_closeout",
        "v14_verification",
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
        parser.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256", required=True
        )
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hash_names = (
        "registration",
        "authorization",
        "environment_design",
        "protocol",
        "v14_analysis",
        "v14_closeout",
        "v14_verification",
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
    hashes = {
        name: getattr(args, f"expected_{name}_sha256") for name in hash_names
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
        or local_verification.get("entry_count") != args.expected_entry_count
        or local_verification.get("local_or_remote_sumo_simulation_executed") is not False
        or remote_verification.get("passed") is not True
        or remote_verification.get("node") != "jtl110gpu"
        or remote_verification.get("formal_held_out_outcomes_read") is not False
        or remote_verification.get("remote_sumo_simulation_executed") is not False
        or remote_verification.get("expected", {}).get("manifest_sha256")
        != hashes["deployment_manifest"]
        or remote_verification.get("expected", {}).get("aggregate_sha256")
        != hashes["deployment_aggregate"]
        or remote_verification.get("expected", {}).get("entry_count")
        != args.expected_entry_count
        or sync.get("passed") is not True
        or sync.get("node") != "jtl110gpu"
        or sync.get("manifest_sha256") != hashes["deployment_manifest"]
        or sync.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or sync.get("entry_count") != args.expected_entry_count
        or sync.get("expected_new_remote_sumo_episodes") != 135
        or sync.get("local_or_remote_sumo_simulation_executed") is not False
    ):
        raise ValueError("v15 deployment evidence is stale or failed")

    paths = {
        "registration": repo
        / "reproducibility/h2oplus_v15_hold_calibration_registration_v1.json",
        "authorization": repo
        / "reproducibility/h2oplus_v15_hold_calibration_authorization_v1.json",
        "environment_design": repo
        / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "protocol": repo
        / "reproducibility/h2oplus_v15_hold_calibration_protocol_v1.json",
        "v14_analysis": repo
        / "reproducibility/h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
        "v14_closeout": repo
        / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_v1.json",
        "v14_verification": repo
        / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_verification_v1.json",
        "source_freeze": repo
        / "reproducibility/h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json",
        "methods": repo
        / "reproducibility/h2oplus_v15_hold_calibration_evaluation_methods_v1.json",
        "dataset": repo
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": repo
        / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": deployment_root / "deployment_manifest.json",
    }
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"frozen deployment input drifted: {name}")
    manifest = json.loads(paths["deployment_manifest"].read_text())
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v15-hold-calibration-evaluation-deployment-manifest-v1"
        or manifest.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or manifest.get("entry_count") != args.expected_entry_count
        or manifest.get("physical_behavior_count") != 15
        or manifest.get("logical_method_count") != 15
        or manifest.get("new_actor_file_count") != 0
        or manifest.get("expected_new_remote_sumo_episodes") != 135
        or manifest.get("critical_matrix_source_file_count") != 49
        or manifest.get("formal_held_out_outcomes_included") is not False
        or manifest.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("v15 deployment manifest drifted")

    signature = (
        "H2Oplus/v15/development/hold-calibration-eval/"
        f"{hashes['deployment_manifest'][:12]}/single/jtl110gpu/glibc227"
    )
    queue = json.loads((Path.home() / ".claude/scheduler/queue.json").read_text())
    tasks = queue if isinstance(queue, list) else queue.get("tasks", queue.get("queue", []))
    duplicates = [
        {
            "id": task.get("id"),
            "status": task.get("status"),
            "signature": task.get("signature"),
        }
        for task in tasks
        if task.get("signature") == signature
    ]
    if duplicates:
        raise ValueError(f"refusing duplicate v15 evaluation submission: {duplicates}")

    ready = {
        "status": "validated",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "manifest_sha256": hashes["deployment_manifest"],
        "deployment_aggregate_sha256": hashes["deployment_aggregate"],
        "physical_behavior_count": 15,
        "unique_parent_actor_count": 5,
        "development_block_count": 3,
        "jobs_per_block": 45,
        "total_remote_sumo_episodes": 135,
        "signature": signature,
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
    dispatcher = repo
    dispatcher /= "reproducibility/run_h2oplus_v15_hold_calibration_evaluation_dispatch.py"
    remote_result = deployment_root.parent / "remote_results_v1/single_jtl110gpu"
    local_result = REPRO
    local_result /= "h2oplus_v15_hold_calibration_results/evaluation_partition_v1/single_jtl110gpu"
    sumo_dir = deployment_root / "SUMO_ruiguang/online_control"
    edge_xml = repo / "bus_h2o/network_data/a_sorted_busline_edge.xml"
    command_parts = [
        python,
        str(dispatcher),
        "--registration", str(paths["registration"]),
        "--authorization", str(paths["authorization"]),
        "--environment-design", str(paths["environment_design"]),
        "--protocol", str(paths["protocol"]),
        "--v14-analysis", str(paths["v14_analysis"]),
        "--v14-closeout", str(paths["v14_closeout"]),
        "--v14-verification", str(paths["v14_verification"]),
        "--source-freeze", str(paths["source_freeze"]),
        "--methods-manifest", str(paths["methods"]),
        "--dataset", str(paths["dataset"]),
        "--sumo-runtime-lock", str(paths["runtime_lock"]),
        "--deployment-manifest", str(paths["deployment_manifest"]),
        "--sumo-dir", str(sumo_dir),
        "--edge-xml", str(edge_xml),
        "--output-root", str(remote_result),
        "--execution-json", str(remote_result / "host_execution.json"),
        "--expected-registration-sha256", hashes["registration"],
        "--expected-authorization-sha256", hashes["authorization"],
        "--expected-environment-design-sha256", hashes["environment_design"],
        "--expected-protocol-sha256", hashes["protocol"],
        "--expected-v14-analysis-sha256", hashes["v14_analysis"],
        "--expected-v14-closeout-sha256", hashes["v14_closeout"],
        "--expected-v14-verification-sha256", hashes["v14_verification"],
        "--expected-source-freeze-sha256", hashes["source_freeze"],
        "--expected-methods-sha256", hashes["methods"],
        "--expected-dataset-sha256", hashes["dataset"],
        "--expected-runtime-lock-sha256", hashes["runtime_lock"],
        "--expected-deployment-manifest-sha256", hashes["deployment_manifest"],
        "--expected-matrix-source-file-count", "49",
        "--workers", "12",
    ]
    command = shlex.join(command_parts)
    submit = [
        "python3", str(scheduler), "submit",
        "--description", "H2O+ v15 hold-calibration development evaluation (135 CPU episodes)",
        "--cmd", command,
        "--cwd", str(repo),
        "--signature", signature,
        "--resource-family", "h2oplus-v15-hold-calibration-eval-worker-v1",
        "--vram", "0",
        "--ram-mb", "32768",
        "--cpu", "12",
        "--priority", "high",
        "--project", "h2oplus-v15-development",
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
    task = {
        "task_id": match.group(0) if match else None,
        "returncode": result.returncode,
        "scheduler_output": combined.strip(),
        "signature": signature,
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "require_node": "jtl110gpu",
        "cpu_cores": 12,
        "ram_mb": 32768,
        "est_vram_mb": 0,
        "expected_jobs": 135,
        "remote_result_dir": str(remote_result),
        "local_result_dir": str(local_result),
    }
    status = (
        "submitted"
        if result.returncode == 0 and task["task_id"] is not None
        else "submission-failure"
    )
    payload = {
        "schema_version": SCHEMA,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "prior_v14_development_outcomes_used_for_design": True,
        "physical_behavior_count": 15,
        "unique_parent_actor_count": 5,
        "development_block_count": 3,
        "jobs_per_block": 45,
        "task_count": 1,
        "total_remote_sumo_episodes": 135,
        "required_environment": environment,
        "cuda_visible_devices_managed_by_scheduler": True,
        "immutable_hashes": hashes,
        "local_verification": descriptor(local_verification_path),
        "remote_verification": descriptor(remote_verification_path),
        "deployment_sync": descriptor(sync_path),
        "parent_v14_evidence": {
            "artifact_integrity_passed": True,
            "training_stability_passed": True,
            "development_scientific_gate_passed": False,
        },
        "task": task,
        "next_stage": (
            "after exact token-bound backend exit0, completed host_execution, exactly "
            "135 valid results and three 45-job manifests, synchronize and independently "
            "audit every file before running the frozen v15 development analyzer"
        ),
    }
    record = args.record.expanduser().resolve()
    atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
