#!/usr/bin/env python3
"""Submit the single authorized v19-v2 development evaluation task."""

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


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEPLOYMENT = REPRO / "deployments/h2oplus_v19_completion_safety_evaluation_v2"
WORKSPACE = DEPLOYMENT / "workspace"
REPO = WORKSPACE / "H2Oplus"
MANIFEST_SHA256 = "3f58a40bde629420cfdbb20edef5c7a0bd2dbcb259bdce6baab4455282809864"
AGGREGATE_SHA256 = "ef668968b10b1bd77348ed20699f6004824e0f88d797e106c2110305891c086e"
SOURCE_FREEZE_SHA256 = "6c77e7117b1c7499635e66e05d500bbc77e01e769530c1fe0d04b514c5dc11d7"
PROTOCOL_SHA256 = "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"
SIGNATURE = (
    "H2Oplus/v19-v2/development/completion-safety-eval/"
    "3f58a40bde62/single/jtl110gpu/glibc227"
)
HASHES = {
    "authorization": "d81add0e13aed729412950f79a947aa8dee8e18f1df4eb3fcd5e622992e419c8",
    "rationale": "ee16744487f43645babf9322ea42ffa016ddd3b9fe12a866d5adf11198af9c0b",
    "registration": "536741a185e2d9b10c99ed821292a51c64bf64e8cb35771075ab787565246fd4",
    "protocol": PROTOCOL_SHA256,
    "correction": "2aec76694036ad038d2d15b4197c1444db8587171898ed0abdb7bdc410b4c282",
    "methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "environment": "5cfb1318418b64e619d4be727c41a0ce0eccab8c096020482586d8286cee3a05",
    "source_freeze": SOURCE_FREEZE_SHA256,
    "parent_v18_audit": "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    "parent_v18_closeout": "e2a0285e9d95bc999cbe371ac261afb1f4526ff7a6722ba2d9d114f67c4c401c",
    "parent_v18_verification": "6a08fa047a4616589fba04cd39eec41e9450ef1ce1fd100933039cb7354fa35a",
    "design_audit": "2238ceb015a456b80c2be5d4d06ebd96d2293fdbc88cadcfb08f087d2dafe1e8",
    "design_verification": "3f8c0e03514b2fb6e2ed46249cfe8260cfdc3c9a2872316f6575de1504543a78",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": MANIFEST_SHA256,
    "deployment_aggregate": AGGREGATE_SHA256,
    "local_verification": "cdbb13044e86cb733d74818344ec59cd4dfd3dc80463479b112b4d5149439503",
    "deployment_sync": "6c3ce430a341bfb19afd4f3bcf27c3b47661705a363612764f62336000278a62",
    "remote_verification": "e1c6d6dfdb2e1db59b210d0c17174c657897e452d4a132fd2eb249a77d032cf6",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v19 submission record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command() -> str:
    p = REPO / "reproducibility"
    result_root = DEPLOYMENT / "remote_results_v2/single_jtl110gpu"
    args = [
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python",
        str(p / "run_h2oplus_v19_completion_safety_evaluation_dispatch.py"),
        "--registration", str(p / "h2oplus_v19_completion_safety_registration_v1.json"),
        "--authorization", str(p / "h2oplus_v19_completion_safety_authorization_v1.json"),
        "--rationale", str(p / "h2oplus_v19_completion_safety_rationale_v1.json"),
        "--environment-design", str(p / "h2oplus_v19_completion_safety_environment_v1.json"),
        "--protocol", str(p / "h2oplus_v19_completion_safety_protocol_v1.json"),
        "--correction", str(p / "h2oplus_v19_completion_safety_environment_binding_correction_v2.json"),
        "--parent-v18-audit", str(p / "h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json"),
        "--parent-v18-closeout", str(p / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json"),
        "--parent-v18-verification", str(p / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_verification_v3.json"),
        "--design-audit", str(p / "h2oplus_v19_completion_safety_design_audit_v2.json"),
        "--design-verification", str(p / "h2oplus_v19_completion_safety_design_verification_v2.json"),
        "--data-environment-audit", str(p / "h2oplus_v17_data_environment_convergence_audit_v1.json"),
        "--data-environment-verification", str(p / "h2oplus_v17_data_environment_convergence_verification_v1.json"),
        "--source-freeze", str(p / "h2oplus_v19_completion_safety_source_freeze_v2.json"),
        "--methods-manifest", str(p / "h2oplus_v19_completion_safety_evaluation_methods_v1.json"),
        "--dataset", str(REPO / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"),
        "--sumo-runtime-lock", str(p / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"),
        "--deployment-manifest", str(WORKSPACE / "deployment_manifest.json"),
        "--output-root", str(result_root),
        "--execution-json", str(result_root / "host_execution.json"),
        "--sumo-dir", str(WORKSPACE / "SUMO_ruiguang/online_control"),
        "--edge-xml", str(REPO / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
        "--expected-registration-sha256", HASHES["registration"],
        "--expected-authorization-sha256", HASHES["authorization"],
        "--expected-rationale-sha256", HASHES["rationale"],
        "--expected-environment-design-sha256", HASHES["environment"],
        "--expected-protocol-sha256", HASHES["protocol"],
        "--expected-correction-sha256", HASHES["correction"],
        "--expected-parent-v18-audit-sha256", HASHES["parent_v18_audit"],
        "--expected-parent-v18-closeout-sha256", HASHES["parent_v18_closeout"],
        "--expected-parent-v18-verification-sha256", HASHES["parent_v18_verification"],
        "--expected-design-audit-sha256", HASHES["design_audit"],
        "--expected-design-verification-sha256", HASHES["design_verification"],
        "--expected-data-environment-audit-sha256", HASHES["data_environment_audit"],
        "--expected-data-environment-verification-sha256", HASHES["data_environment_verification"],
        "--expected-source-freeze-sha256", HASHES["source_freeze"],
        "--expected-methods-sha256", HASHES["methods"],
        "--expected-dataset-sha256", HASHES["dataset"],
        "--expected-runtime-lock-sha256", HASHES["runtime_lock"],
        "--expected-deployment-manifest-sha256", HASHES["deployment_manifest"],
        "--expected-matrix-source-file-count", "55",
        "--workers", "12",
    ]
    return shlex.join(args)


def verify_inputs(queue_path: Path) -> dict:
    paths = {
        "authorization": REPRO / "h2oplus_v19_completion_safety_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v19_completion_safety_rationale_v1.json",
        "registration": REPRO / "h2oplus_v19_completion_safety_registration_v1.json",
        "protocol": REPRO / "h2oplus_v19_completion_safety_protocol_v1.json",
        "correction": REPRO / "h2oplus_v19_completion_safety_environment_binding_correction_v2.json",
        "methods": REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v19_completion_safety_environment_v1.json",
        "source_freeze": REPRO / "h2oplus_v19_completion_safety_source_freeze_v2.json",
        "parent_v18_audit": REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
        "parent_v18_closeout": REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
        "parent_v18_verification": REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_verification_v3.json",
        "design_audit": REPRO / "h2oplus_v19_completion_safety_design_audit_v2.json",
        "design_verification": REPRO / "h2oplus_v19_completion_safety_design_verification_v2.json",
        "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": WORKSPACE / "deployment_manifest.json",
        "local_verification": REPRO / "h2oplus_v19_completion_safety_evaluation_deployment_audit_v2.json",
        "deployment_sync": REPRO / "h2oplus_v19_completion_safety_evaluation_deployment_sync_v2.json",
        "remote_verification": REPRO / "h2oplus_v19_completion_safety_evaluation_deployment_v2_jtl110gpu_verification_v1.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != HASHES[name]:
            raise ValueError(f"v19 submission input drifted: {name}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json" and name != "dataset"
    }
    if (
        payloads["local_verification"].get("passed") is not True
        or payloads["deployment_sync"].get("passed") is not True
        or payloads["remote_verification"].get("passed") is not True
        or payloads["remote_verification"].get("remote_sumo_simulation_executed") is not False
        or payloads["correction"].get("execution_boundary", {}).get("completed_scientific_result_count") != 0
        or payloads["registration"].get("v19_development_evaluation_outcomes_read") is not False
    ):
        raise ValueError("v19 submission verification boundary drifted")
    task_command = command()
    queue = json.loads(queue_path.expanduser().resolve(strict=True).read_text())
    duplicates = [
        {"id": row.get("id"), "status": row.get("status"), "signature": row.get("signature")}
        for row in queue.get("tasks", [])
        if row.get("signature") == SIGNATURE or row.get("cmd") == task_command
    ]
    if duplicates:
        raise ValueError(f"refusing duplicate v19 submission: {duplicates}")
    return {"paths": paths, "command": task_command, "duplicates": duplicates}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v19_completion_safety_evaluation_submission_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    checked = verify_inputs(args.queue)
    local_result = REPRO / "h2oplus_v19_completion_safety_results/evaluation_partition_v2/single_jtl110gpu"
    remote_result = DEPLOYMENT / "remote_results_v2/single_jtl110gpu"
    if local_result.exists() or remote_result.exists():
        raise FileExistsError("v19 result destination already exists")
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include",
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(REPO),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256": PROTOCOL_SHA256,
    }
    submit = [
        sys.executable,
        "skill/scheduler.py",
        "submit",
        "--description",
        "H2O+ v19-v2 completion-safety paired development screen (135 CPU episodes)",
        "--cmd", checked["command"],
        "--cwd", str(REPO),
        "--signature", SIGNATURE,
        "--resource-family", "h2oplus-v19-v2-completion-safety-eval-worker-v1",
        "--vram", "0",
        "--ram-mb", "32768",
        "--cpu", "12",
        "--priority", "high",
        "--project", "h2oplus-v19-development",
        "--require-node", "jtl110gpu",
        "--result-dir", str(remote_result),
        "--local-result-dir", str(local_result),
        "--allow-remote-large-data",
        "--skip-launch-staging",
        "--env",
        *(f"{key}={value}" for key, value in environment.items()),
    ]
    completed = subprocess.run(
        submit, cwd=scheduler_root, text=True, capture_output=True, check=False
    )
    match = re.search(r"submitted\s+(t\d+)", completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(
            f"v19 scheduler submission failed: rc={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    task_id = match.group(1)
    record = {
        "schema_version": "transit-h2oplus-v19-completion-safety-evaluation-submission-v2",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "adaptive_development_screen": True,
        "task_count": 1,
        "total_remote_sumo_episodes": 135,
        "physical_behavior_count": 15,
        "development_block_count": 3,
        "jobs_per_block": 45,
        "predesignated_candidate": "h2oplus-v19-completion-safety-quarter-load-sixty",
        "diagnostic_cannot_replace_candidate": True,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "predeployment_environment_binding_correction_only": True,
        "v19_v1_completed_scientific_result_count": 0,
        "v19_v1_results_reused": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_confirmation_eligibility_created_by_this_screen": False,
        "passing_screen_only_permits_fresh_full_development_control_matrix": True,
        "local_training_or_sumo_run": False,
        "duplicate_submission_override_used": False,
        "force_dispatch_used": False,
        "allow_no_resume_used": False,
        "cuda_visible_devices_managed_by_scheduler": True,
        "immutable_hashes": HASHES,
        "required_environment": environment,
        "task": {
            "task_id": task_id,
            "signature": SIGNATURE,
            "command_sha256": hashlib.sha256(checked["command"].encode()).hexdigest(),
            "require_node": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "est_vram_mb": 0,
            "expected_jobs": 135,
            "remote_result_dir": str(remote_result),
            "local_result_dir": str(local_result),
            "returncode": completed.returncode,
            "scheduler_output": completed.stdout.strip(),
        },
        "next_stage": (
            "Allow natural scheduler dispatch; require exact token-bound backend "
            "exit0, completed host execution, 135 valid results and three complete "
            "45-job manifests before synchronization and frozen analysis."
        ),
    }
    atomic_json(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
