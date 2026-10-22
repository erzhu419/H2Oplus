#!/usr/bin/env python3
"""Uniquely submit the final-boundary-corrected v18-v3 development screen."""

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


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-submission-v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v18-v3 submission: {path}")
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
        default=REPRO
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v3/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_audit_v3.json",
    )
    parser.add_argument(
        "--remote-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_v3_jtl110gpu_verification_v5.json",
    )
    parser.add_argument(
        "--deployment-sync",
        type=Path,
        default=REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_sync_v3.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v3.json",
    )
    names = (
        "registration",
        "authorization",
        "correction",
        "v1_incident_closeout",
        "v1_incident_verification",
        "v2_incident_closeout",
        "v2_incident_verification",
        "rationale",
        "environment_design",
        "protocol",
        "parent_v17_analysis",
        "parent_v17_audit",
        "parent_v17_closeout",
        "parent_v17_verification",
        "data_environment_audit",
        "data_environment_verification",
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
        "correction",
        "v1_incident_closeout",
        "v1_incident_verification",
        "v2_incident_closeout",
        "v2_incident_verification",
        "rationale",
        "environment_design",
        "protocol",
        "parent_v17_analysis",
        "parent_v17_audit",
        "parent_v17_closeout",
        "parent_v17_verification",
        "data_environment_audit",
        "data_environment_verification",
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
    record = args.record.expanduser().resolve()
    if record.exists():
        raise FileExistsError(f"v18-v3 submission record exists: {record}")

    local_path = args.local_verification.expanduser().resolve(strict=True)
    remote_path = args.remote_verification.expanduser().resolve(strict=True)
    sync_path = args.deployment_sync.expanduser().resolve(strict=True)
    for name, path in (
        ("local_verification", local_path),
        ("remote_verification", remote_path),
        ("deployment_sync", sync_path),
    ):
        if sha256(path) != hashes[name]:
            raise ValueError(f"v18-v3 {name} identity drifted")
    local = json.loads(local_path.read_text())
    remote = json.loads(remote_path.read_text())
    sync = json.loads(sync_path.read_text())
    if (
        local.get("passed") is not True
        or local.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-audit-v3"
        or local.get("manifest_sha256") != hashes["deployment_manifest"]
        or local.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or local.get("entry_count") != args.expected_entry_count
        or local.get("local_or_remote_sumo_simulation_executed") is not False
        or remote.get("passed") is not True
        or remote.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-remote-verification-v5"
        or remote.get("node") != "jtl110gpu"
        or remote.get("formal_held_out_outcomes_read") is not False
        or remote.get("remote_sumo_simulation_executed") is not False
        or remote.get("expected", {}).get("manifest_sha256")
        != hashes["deployment_manifest"]
        or remote.get("expected", {}).get("aggregate_sha256")
        != hashes["deployment_aggregate"]
        or remote.get("expected", {}).get("entry_count")
        != args.expected_entry_count
        or remote.get("expected", {}).get("source_file_count") != 59
        or remote.get("observation", {}).get("binding_smoke_count") != 15
        or remote.get("observation", {}).get("all_binding_smokes_passed") is not True
        or sync.get("passed") is not True
        or sync.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-sync-v3"
        or sync.get("node") != "jtl110gpu"
        or sync.get("manifest_sha256") != hashes["deployment_manifest"]
        or sync.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or sync.get("entry_count") != args.expected_entry_count
        or sync.get("expected_new_remote_sumo_episodes") != 135
        or sync.get("v18_v1_or_v2_results_reused") is not False
    ):
        raise ValueError("v18-v3 deployment evidence is stale or failed")

    v1_incident_root = (
        repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_incidents/v1_parser_contract"
    )
    v2_incident_root = (
        repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_incidents/v2_namespace_binding"
    )
    paths = {
        "registration": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_successor_registration_v3.json",
        "authorization": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_successor_authorization_v3.json",
        "correction": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_namespace_binding_correction_v3.json",
        "v1_incident_closeout": v1_incident_root / "incident_closeout_v1.json",
        "v1_incident_verification": v1_incident_root / "incident_verification_v1.json",
        "v2_incident_closeout": v2_incident_root / "incident_closeout_v2.json",
        "v2_incident_verification": v2_incident_root / "incident_verification_v2.json",
        "rationale": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_rationale_v1.json",
        "environment_design": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "protocol": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "parent_v17_analysis": repo
        / "reproducibility/h2oplus_v17_tail_service_reserve_results/tail_service_reserve_analysis_v2.json",
        "parent_v17_audit": repo
        / "reproducibility/h2oplus_v17_tail_service_reserve_results/evaluation_audit_v2.json",
        "parent_v17_closeout": repo
        / "reproducibility/h2oplus_v17_tail_service_reserve_results/evaluation_closeout_v2.json",
        "parent_v17_verification": repo
        / "reproducibility/h2oplus_v17_tail_service_reserve_results/evaluation_closeout_verification_v2.json",
        "data_environment_audit": repo
        / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": repo
        / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
        "source_freeze": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v3.json",
        "methods": repo
        / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "dataset": repo
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": repo
        / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": deployment_root / "deployment_manifest.json",
    }
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"v18-v3 frozen input drifted: {name}")
    manifest = json.loads(paths["deployment_manifest"].read_text())
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v3"
        or manifest.get("aggregate_sha256") != hashes["deployment_aggregate"]
        or manifest.get("entry_count") != args.expected_entry_count
        or manifest.get("total_file_bytes") != 954442582
        or manifest.get("critical_matrix_source_file_count") != 59
        or manifest.get("scientific_source_file_count") != 24
        or manifest.get("namespace_binding_correction_only") is not True
        or manifest.get("scientific_behavior_unchanged_from_v18_v1_and_v2") is not True
        or manifest.get("v18_v1_result_count") != 0
        or manifest.get("v18_v2_result_count") != 0
        or manifest.get("v18_v1_or_v2_results_reused") is not False
        or manifest.get("formal_held_out_outcomes_included") is not False
        or manifest.get("training") is not False
    ):
        raise ValueError("v18-v3 deployment manifest drifted")

    signature = (
        "H2Oplus/v18-v3/development/empty-tail-reserve-eval/"
        f"{hashes['deployment_manifest'][:12]}/single/jtl110gpu/glibc227"
    )
    remote_result = deployment_root.parent / "remote_results_v3/single_jtl110gpu"
    local_result = (
        REPRO
        / "h2oplus_v18_empty_tail_reserve_results/evaluation_partition_v3/"
        "single_jtl110gpu"
    )
    queue_payload = json.loads((Path.home() / ".claude/scheduler/queue.json").read_text())
    tasks = (
        queue_payload
        if isinstance(queue_payload, list)
        else queue_payload.get("tasks", queue_payload.get("queue", []))
    )
    # A no-SUMO verifier can be auto-adopted with the deployment as its cwd.
    # Scientific duplicates are identified by their exact signature or result roots.
    markers = (str(remote_result), str(local_result), signature)
    duplicates = [
        {"id": task.get("id"), "status": task.get("status"), "signature": task.get("signature")}
        for task in tasks
        if any(marker in json.dumps(task, sort_keys=True) for marker in markers)
    ]
    if duplicates:
        raise ValueError(f"refusing duplicate v18-v3 submission: {duplicates}")
    ready = {
        "status": "validated",
        "study_stage": "development",
        "signature": signature,
        "physical_behavior_count": 15,
        "development_block_count": 3,
        "jobs_per_block": 45,
        "total_remote_sumo_episodes": 135,
        "duplicate_matches": duplicates,
        "v18_v1_result_count": 0,
        "v18_v2_result_count": 0,
        "v18_v1_or_v2_results_reused": False,
        "formal_held_out_outcomes_read": False,
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
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
    }
    dispatcher = (
        repo
        / "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v3.py"
    )
    command_parts = [
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python",
        str(dispatcher),
        "--registration", str(paths["registration"]),
        "--authorization", str(paths["authorization"]),
        "--correction", str(paths["correction"]),
        "--v1-incident-closeout", str(paths["v1_incident_closeout"]),
        "--v1-incident-verification", str(paths["v1_incident_verification"]),
        "--v2-incident-closeout", str(paths["v2_incident_closeout"]),
        "--v2-incident-verification", str(paths["v2_incident_verification"]),
        "--rationale", str(paths["rationale"]),
        "--environment-design", str(paths["environment_design"]),
        "--protocol", str(paths["protocol"]),
        "--parent-v17-audit", str(paths["parent_v17_audit"]),
        "--parent-v17-closeout", str(paths["parent_v17_closeout"]),
        "--parent-v17-verification", str(paths["parent_v17_verification"]),
        "--data-environment-audit", str(paths["data_environment_audit"]),
        "--data-environment-verification", str(paths["data_environment_verification"]),
        "--source-freeze", str(paths["source_freeze"]),
        "--methods-manifest", str(paths["methods"]),
        "--dataset", str(paths["dataset"]),
        "--sumo-runtime-lock", str(paths["runtime_lock"]),
        "--deployment-manifest", str(paths["deployment_manifest"]),
        "--sumo-dir", str(deployment_root / "SUMO_ruiguang/online_control"),
        "--edge-xml", str(repo / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
        "--output-root", str(remote_result),
        "--execution-json", str(remote_result / "host_execution.json"),
        "--expected-registration-sha256", hashes["registration"],
        "--expected-authorization-sha256", hashes["authorization"],
        "--expected-correction-sha256", hashes["correction"],
        "--expected-v1-incident-closeout-sha256", hashes["v1_incident_closeout"],
        "--expected-v1-incident-verification-sha256", hashes["v1_incident_verification"],
        "--expected-v2-incident-closeout-sha256", hashes["v2_incident_closeout"],
        "--expected-v2-incident-verification-sha256", hashes["v2_incident_verification"],
        "--expected-rationale-sha256", hashes["rationale"],
        "--expected-environment-design-sha256", hashes["environment_design"],
        "--expected-protocol-sha256", hashes["protocol"],
        "--expected-parent-v17-audit-sha256", hashes["parent_v17_audit"],
        "--expected-parent-v17-closeout-sha256", hashes["parent_v17_closeout"],
        "--expected-parent-v17-verification-sha256", hashes["parent_v17_verification"],
        "--expected-data-environment-audit-sha256", hashes["data_environment_audit"],
        "--expected-data-environment-verification-sha256", hashes["data_environment_verification"],
        "--expected-source-freeze-sha256", hashes["source_freeze"],
        "--expected-methods-sha256", hashes["methods"],
        "--expected-dataset-sha256", hashes["dataset"],
        "--expected-runtime-lock-sha256", hashes["runtime_lock"],
        "--expected-deployment-manifest-sha256", hashes["deployment_manifest"],
        "--expected-matrix-source-file-count", "59",
        "--workers", "12",
    ]
    command = shlex.join(command_parts)
    submit = [
        "python3", str(scheduler), "submit",
        "--description", "H2O+ v18-v3 final-boundary-corrected empty-tail development screen (135 CPU episodes)",
        "--cmd", command,
        "--cwd", str(repo),
        "--signature", signature,
        "--resource-family", "h2oplus-v18-v3-empty-tail-reserve-eval-worker-v1",
        "--vram", "0",
        "--ram-mb", "32768",
        "--cpu", "12",
        "--priority", "high",
        "--project", "h2oplus-v18-development",
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
        "screen_type": "paired-mechanism-screen",
        "namespace_binding_correction_only": True,
        "scientific_behavior_unchanged_from_v18_v1_and_v2": True,
        "v18_v1_result_count": 0,
        "v18_v2_result_count": 0,
        "v18_v1_or_v2_results_reused": False,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_run": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "task_count": 1,
        "total_remote_sumo_episodes": 135,
        "predesignated_candidate": "h2oplus-v18-empty-tail-reserve-empty-forty",
        "diagnostic_cannot_replace_candidate": True,
        "passing_screen_only_permits_full_development_control_matrix": True,
        "formal_confirmation_eligibility_created_by_this_screen": False,
        "required_environment": environment,
        "cuda_visible_devices_managed_by_scheduler": True,
        "duplicate_submission_override_used": False,
        "allow_no_resume_used": False,
        "force_dispatch_used": False,
        "immutable_hashes": hashes,
        "local_verification": descriptor(local_path),
        "remote_verification": descriptor(remote_path),
        "deployment_sync": descriptor(sync_path),
        "task": task,
        "next_stage": (
            "Require token-bound backend exit0, completed host_execution, exactly "
            "135 valid results and three 45-job manifests before synchronization."
        ),
    }
    atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
