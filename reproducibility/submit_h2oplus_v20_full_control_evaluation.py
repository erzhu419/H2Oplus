#!/usr/bin/env python3
"""Submit the single authorized v20 full development control matrix."""

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
DEPLOYMENT = REPRO / "deployments/h2oplus_v20_full_control_evaluation_v1"
WORKSPACE = DEPLOYMENT / "workspace"
REPO = WORKSPACE / "H2Oplus"
MANIFEST_SHA256 = "009ba0c8e5725936fa984e6361e77d175801cc514e81af531c6db0a621137206"
AGGREGATE_SHA256 = "f504b4b23139931724ca874dae3a2db018a41f9ab83090f7e63046ee4f26960c"
SOURCE_FREEZE_SHA256 = "b30de4416829a0bfa683184712a842dcff64732d27c3c51a5f8a3b107da22159"
V19_PROTOCOL_SHA256 = "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"
V20_PROTOCOL_SHA256 = "7211d32aeb1416840de442ca43fa997a918dfa282afcec5009091f78834ffa5d"
SIGNATURE = (
    "H2Oplus/v20/development/full-control-eval/"
    "009ba0c8e572/single/jtl110gpu/glibc227"
)
HASHES = {
    "authorization": "ab535766ffad546c07e4291bfd2f0519d7cfa0e83f4cb3f4022c369d4cb19a06",
    "rationale": "e93a4114760e8755d27dcd7042ca666f4aefdf17e03d3bb55990068cf552998c",
    "registration": "f930a3612720097d4f5e5b635501405245f04ffd54445f389a23800d06f7f7d8",
    "protocol": V20_PROTOCOL_SHA256,
    "methods": "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9",
    "environment": "7210826548c94aaa8fbbbb73b62eec5450fec131f2ae3dd1d1dbc93922dff2ba",
    "design_audit": "0c583278b57eee5b5d479db78d294643f11b5d04a51840e35eeaf97e14a0015c",
    "design_verification": "66cc5133fa2a25bc323f880d148dba282db9708161af3df4c5f764541c3d70b9",
    "source_freeze": SOURCE_FREEZE_SHA256,
    "parent_v19_audit": "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    "parent_v19_closeout": "3e95e4cf964983257fd3fabc5cab2987e1a249b5b9a25d3ae455f081932b3f74",
    "parent_v19_verification": "ec44263eebc8ba61049973f34aadf5154d0b07f5ccc096d3659662002b95f9ab",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": MANIFEST_SHA256,
    "deployment_aggregate": AGGREGATE_SHA256,
    "local_verification": "815e4b68ad2849e5a397ddcf6fbf13ae2738678b9234b5f14fd4d3f417fd0244",
    "deployment_sync": "0d7d0b27423cb3ac9a7a6822caa3109fa559e6839442a430eb5fc8d9aae0e0db",
    "remote_verification": "c5e74ac8440ddecbc9fee303ea193003f8443a12cba7e97f68b8d312fffa2e75",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v20 submission record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def command() -> str:
    p = REPO / "reproducibility"
    result_root = DEPLOYMENT / "remote_results_v1/single_jtl110gpu"
    args = [
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python",
        str(p / "run_h2oplus_v20_full_control_evaluation_dispatch.py"),
        "--registration",
        str(p / "h2oplus_v20_full_control_registration_v1.json"),
        "--authorization",
        str(p / "h2oplus_v20_full_control_authorization_v1.json"),
        "--rationale",
        str(p / "h2oplus_v20_full_control_rationale_v1.json"),
        "--environment-design",
        str(p / "h2oplus_v20_full_control_environment_v1.json"),
        "--protocol",
        str(p / "h2oplus_v20_full_control_protocol_v1.json"),
        "--design-audit",
        str(p / "h2oplus_v20_full_control_design_audit_v1.json"),
        "--design-verification",
        str(p / "h2oplus_v20_full_control_design_verification_v1.json"),
        "--parent-v19-audit",
        str(p / "h2oplus_v19_completion_safety_results/evaluation_audit_v2.json"),
        "--parent-v19-closeout",
        str(p / "h2oplus_v19_completion_safety_results/evaluation_closeout_v2.json"),
        "--parent-v19-verification",
        str(
            p
            / "h2oplus_v19_completion_safety_results/evaluation_closeout_verification_v2.json"
        ),
        "--data-environment-audit",
        str(p / "h2oplus_v17_data_environment_convergence_audit_v1.json"),
        "--data-environment-verification",
        str(p / "h2oplus_v17_data_environment_convergence_verification_v1.json"),
        "--source-freeze",
        str(p / "h2oplus_v20_full_control_source_freeze_v1.json"),
        "--methods-manifest",
        str(p / "h2oplus_v20_full_control_evaluation_methods_v1.json"),
        "--dataset",
        str(REPO / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"),
        "--sumo-runtime-lock",
        str(p / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"),
        "--deployment-manifest",
        str(WORKSPACE / "deployment_manifest.json"),
        "--output-root",
        str(result_root),
        "--execution-json",
        str(result_root / "host_execution.json"),
        "--sumo-dir",
        str(WORKSPACE / "SUMO_ruiguang/online_control"),
        "--edge-xml",
        str(REPO / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
        "--expected-registration-sha256",
        HASHES["registration"],
        "--expected-authorization-sha256",
        HASHES["authorization"],
        "--expected-rationale-sha256",
        HASHES["rationale"],
        "--expected-environment-design-sha256",
        HASHES["environment"],
        "--expected-protocol-sha256",
        HASHES["protocol"],
        "--expected-design-audit-sha256",
        HASHES["design_audit"],
        "--expected-design-verification-sha256",
        HASHES["design_verification"],
        "--expected-parent-v19-audit-sha256",
        HASHES["parent_v19_audit"],
        "--expected-parent-v19-closeout-sha256",
        HASHES["parent_v19_closeout"],
        "--expected-parent-v19-verification-sha256",
        HASHES["parent_v19_verification"],
        "--expected-data-environment-audit-sha256",
        HASHES["data_environment_audit"],
        "--expected-data-environment-verification-sha256",
        HASHES["data_environment_verification"],
        "--expected-source-freeze-sha256",
        HASHES["source_freeze"],
        "--expected-methods-sha256",
        HASHES["methods"],
        "--expected-dataset-sha256",
        HASHES["dataset"],
        "--expected-runtime-lock-sha256",
        HASHES["runtime_lock"],
        "--expected-deployment-manifest-sha256",
        HASHES["deployment_manifest"],
        "--expected-matrix-source-file-count",
        "57",
        "--workers",
        "12",
    ]
    return shlex.join(args)


def verify_inputs(queue_path: Path) -> dict:
    paths = {
        "authorization": REPRO / "h2oplus_v20_full_control_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v20_full_control_rationale_v1.json",
        "registration": REPRO / "h2oplus_v20_full_control_registration_v1.json",
        "protocol": REPRO / "h2oplus_v20_full_control_protocol_v1.json",
        "methods": REPRO / "h2oplus_v20_full_control_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v20_full_control_environment_v1.json",
        "design_audit": REPRO / "h2oplus_v20_full_control_design_audit_v1.json",
        "design_verification": REPRO / "h2oplus_v20_full_control_design_verification_v1.json",
        "source_freeze": REPRO / "h2oplus_v20_full_control_source_freeze_v1.json",
        "parent_v19_audit": REPRO / "h2oplus_v19_completion_safety_results/evaluation_audit_v2.json",
        "parent_v19_closeout": REPRO / "h2oplus_v19_completion_safety_results/evaluation_closeout_v2.json",
        "parent_v19_verification": REPRO / "h2oplus_v19_completion_safety_results/evaluation_closeout_verification_v2.json",
        "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": WORKSPACE / "deployment_manifest.json",
        "local_verification": REPRO / "h2oplus_v20_full_control_evaluation_deployment_audit_v1.json",
        "deployment_sync": REPRO / "h2oplus_v20_full_control_evaluation_deployment_sync_v1.json",
        "remote_verification": REPRO / "h2oplus_v20_full_control_evaluation_deployment_jtl110gpu_verification_v1.json",
    }
    for name, path in paths.items():
        observed = sha256(path.resolve(strict=True))
        if observed != HASHES[name]:
            raise ValueError(f"v20 submission input drifted: {name}: {observed}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json" and name != "dataset"
    }
    if (
        payloads["local_verification"].get("passed") is not True
        or payloads["deployment_sync"].get("passed") is not True
        or payloads["remote_verification"].get("passed") is not True
        or payloads["remote_verification"].get("remote_sumo_simulation_executed")
        is not False
        or payloads["design_audit"].get("scientific_outcomes_available") is not False
        or payloads["registration"].get("v20_development_evaluation_outcomes_read")
        is not False
        or payloads["registration"].get(
            "formal_confirmation_stage_authorized_or_submitted"
        )
        is not False
    ):
        raise ValueError("v20 submission verification boundary drifted")
    task_command = command()
    queue = json.loads(queue_path.expanduser().resolve(strict=True).read_text())
    duplicates = [
        {
            "id": row.get("id"),
            "status": row.get("status"),
            "signature": row.get("signature"),
        }
        for row in queue.get("tasks", [])
        if row.get("signature") == SIGNATURE or row.get("cmd") == task_command
    ]
    if duplicates:
        raise ValueError(f"refusing duplicate v20 submission: {duplicates}")
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
        default=REPRO / "h2oplus_v20_full_control_evaluation_submission_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    checked = verify_inputs(args.queue)
    local_result = REPRO / (
        "h2oplus_v20_full_control_results/"
        "evaluation_partition_v1/single_jtl110gpu"
    )
    remote_result = DEPLOYMENT / "remote_results_v1/single_jtl110gpu"
    if local_result.exists() or remote_result.exists():
        raise FileExistsError("v20 result destination already exists")
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
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
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256": V19_PROTOCOL_SHA256,
        "H2OPLUS_V20_FULL_CONTROL_PROTOCOL_SHA256": V20_PROTOCOL_SHA256,
    }
    submit = [
        sys.executable,
        "skill/scheduler.py",
        "submit",
        "--description",
        "H2O+ v20 full development control matrix (405 CPU evaluation episodes)",
        "--cmd",
        checked["command"],
        "--cwd",
        str(REPO),
        "--signature",
        SIGNATURE,
        "--resource-family",
        "h2oplus-v20-full-control-eval-worker-v1",
        "--vram",
        "0",
        "--ram-mb",
        "32768",
        "--cpu",
        "12",
        "--priority",
        "high",
        "--project",
        "h2oplus-v20-development",
        "--require-node",
        "jtl110gpu",
        "--result-dir",
        str(remote_result),
        "--local-result-dir",
        str(local_result),
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
            f"v20 scheduler submission failed: rc={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    task_id = match.group(1)
    record = {
        "schema_version": "transit-h2oplus-v20-full-control-evaluation-submission-v1",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "study_stage": "development",
        "matrix_type": "full-development-control-matrix",
        "adaptive_development_matrix": True,
        "task_count": 1,
        "total_remote_sumo_episodes": 405,
        "physical_behavior_count": 45,
        "development_block_count": 3,
        "jobs_per_block": 135,
        "predesignated_candidate": (
            "h2oplus-v19-completion-safety-quarter-load-sixty"
        ),
        "prior_v14_through_v19_development_outcomes_used_for_design": True,
        "material_data_environment_validity_factors_remain": True,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passing_only_creates_separate_formal_confirmation_eligibility": True,
        "task": {
            "task_id": task_id,
            "signature": SIGNATURE,
            "host": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "est_vram_mb": 0,
            "expected_jobs": 405,
            "remote_result_dir": str(remote_result),
            "local_result_dir": str(local_result),
            "command": checked["command"],
            "success_marker": "DONE",
        },
        "bindings": {
            key: {"path": str(checked["paths"][key]), "sha256": value}
            for key, value in HASHES.items()
            if key in checked["paths"]
        },
        "scheduler_stdout": completed.stdout.strip(),
        "scheduler_stderr": completed.stderr.strip(),
    }
    atomic_json(args.output.expanduser().resolve(), record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
