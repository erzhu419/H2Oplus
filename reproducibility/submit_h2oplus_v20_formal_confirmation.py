#!/usr/bin/env python3
"""Submit the single authorized v20 formal confirmation task."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v1"
WORKSPACE = DEPLOYMENT / "workspace"
REPO = WORKSPACE / "H2Oplus"
OVERLAY = DEPLOYMENT / "overlay"
RESULT_ROOT = DEPLOYMENT / "remote_results_v1/single_jtl110gpu"
SIGNATURE = (
    "H2Oplus/v20/formal-confirmation/91eef065317f/"
    "single/jtl110gpu/glibc227"
)
HASHES = {
    "execution_authorization": "c9e2cdfe10074e5ddded7f756c682b9c5a63d3e5fc6fc52c87498289407226d5",
    "preparation_authorization": "7a7b79f029a5437a80a31637c1bc057571849623b9b04a92a67bcf66586882c9",
    "rationale": "72e6d2961846e458721f121ae75eded862bed0556c929dc38015a3b2d9103a4d",
    "registration": "c33111cd72b741a801804cd065f42629094d45c0fe03a5b88f9d95dba62149ce",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "environment": "50b831923b76a81c773cf7dc19dcc2e10c729ea8fc7b3be6982fc01b7c3cc141",
    "design_audit": "f0f7f68d8f1273f49e55b1444e5c74282b6aeaf68efaa2f8a0aa0f4261408757",
    "design_verification": "16d33c2b12f280ed8d1595afca91b52043a8a87a844aa7a4cda0227e574f6e29",
    "source_freeze": "70f387eeb232d5021f33b12ce5e653d0dea500b905827d7c8d525fb738e23c25",
    "parent_v20_analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "parent_v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "parent_v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "parent_v20_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "deployment_manifest": "91eef065317f1ba8a17fca839f8ce84371555fbb581a515eb474cc8d4bc7cd17",
    "deployment_aggregate": "0bbce7865eaf72f49eb8846054aeb9c6a6e8cc356d763b7e47859dc18a6e203e",
    "local_deployment_audit": "e83d1435d19bd13e884a4b35fb724807a4c93a41310b0c2d36a92b879791e917",
    "deployment_sync": "7675e6c6c9a5d6dee10b84c944d3f99237ba4ba4363c12e9be4211d0a121c287",
    "remote_verification": "bddaa8679e7aa107cd95ddea59defc827619f1dfbc398505302d4c80d74fa3ae",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal submission evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def paths() -> dict:
    return {
        "execution_authorization": REPRO / "h2oplus_v20_formal_confirmation_execution_authorization_v1.json",
        "preparation_authorization": REPRO / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v20_formal_confirmation_rationale_v1.json",
        "registration": REPRO / "h2oplus_v20_formal_confirmation_registration_v1.json",
        "protocol": REPRO / "h2oplus_v20_formal_confirmation_protocol_v1.json",
        "methods": REPRO / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v20_formal_confirmation_environment_v1.json",
        "design_audit": REPRO / "h2oplus_v20_formal_confirmation_design_audit_v1.json",
        "design_verification": REPRO / "h2oplus_v20_formal_confirmation_design_verification_v1.json",
        "source_freeze": REPRO / "h2oplus_v20_formal_confirmation_source_freeze_v1.json",
        "parent_v20_analysis": REPRO / "h2oplus_v20_full_control_results/full_control_analysis_v1.json",
        "parent_v20_audit": REPRO / "h2oplus_v20_full_control_results/evaluation_audit_v1.json",
        "parent_v20_closeout": REPRO / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
        "parent_v20_verification": REPRO / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
        "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": OVERLAY / "deployment_manifest.json",
        "local_deployment_audit": REPRO / "h2oplus_v20_formal_confirmation_deployment_audit_v1.json",
        "deployment_sync": REPRO / "h2oplus_v20_formal_confirmation_deployment_sync_v1.json",
        "remote_verification": REPRO / "h2oplus_v20_formal_confirmation_deployment_jtl110gpu_verification_v1.json",
    }


def command() -> str:
    p = REPO / "reproducibility"
    args = [
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python",
        str(p / "run_h2oplus_v20_formal_confirmation_dispatch.py"),
        "--execution-authorization",
        str(p / "h2oplus_v20_formal_confirmation_execution_authorization_v1.json"),
        "--preparation-authorization",
        str(p / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json"),
        "--rationale",
        str(p / "h2oplus_v20_formal_confirmation_rationale_v1.json"),
        "--registration",
        str(p / "h2oplus_v20_formal_confirmation_registration_v1.json"),
        "--environment",
        str(p / "h2oplus_v20_formal_confirmation_environment_v1.json"),
        "--protocol",
        str(p / "h2oplus_v20_formal_confirmation_protocol_v1.json"),
        "--design-audit",
        str(p / "h2oplus_v20_formal_confirmation_design_audit_v1.json"),
        "--design-verification",
        str(p / "h2oplus_v20_formal_confirmation_design_verification_v1.json"),
        "--parent-v20-analysis",
        str(p / "h2oplus_v20_full_control_results/full_control_analysis_v1.json"),
        "--parent-v20-audit",
        str(p / "h2oplus_v20_full_control_results/evaluation_audit_v1.json"),
        "--parent-v20-closeout",
        str(p / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json"),
        "--parent-v20-verification",
        str(p / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json"),
        "--data-environment-audit",
        str(p / "h2oplus_v17_data_environment_convergence_audit_v1.json"),
        "--data-environment-verification",
        str(p / "h2oplus_v17_data_environment_convergence_verification_v1.json"),
        "--source-freeze",
        str(p / "h2oplus_v20_formal_confirmation_source_freeze_v1.json"),
        "--methods-manifest",
        str(p / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json"),
        "--dataset",
        str(REPO / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"),
        "--sumo-runtime-lock",
        str(p / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"),
        "--deployment-manifest",
        str(WORKSPACE / "deployment_manifest.json"),
        "--output-root",
        str(RESULT_ROOT),
        "--execution-json",
        str(RESULT_ROOT / "host_execution.json"),
        "--sumo-dir",
        str(WORKSPACE / "SUMO_ruiguang/online_control"),
        "--edge-xml",
        str(
            WORKSPACE
            / "SUMO_ruiguang/online_control/intersection_delay/a_sorted_busline_edge.xml"
        ),
    ]
    expected_names = (
        "execution_authorization",
        "preparation_authorization",
        "rationale",
        "registration",
        "environment",
        "protocol",
        "design_audit",
        "design_verification",
        "parent_v20_analysis",
        "parent_v20_audit",
        "parent_v20_closeout",
        "parent_v20_verification",
        "data_environment_audit",
        "data_environment_verification",
        "source_freeze",
        "methods",
        "dataset",
        "runtime_lock",
        "deployment_manifest",
    )
    for name in expected_names:
        args.extend(["--expected-" + name.replace("_", "-") + "-sha256", HASHES[name]])
    args.extend(["--workers", "12"])
    return shlex.join(args)


def verify_inputs(queue: Path, scheduler_root: Path) -> dict:
    bound = paths()
    for name, path in bound.items():
        observed = sha256(path.resolve(strict=True))
        if observed != HASHES[name]:
            raise ValueError(f"formal submission input drifted: {name}: {observed}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in bound.items()
        if path.suffix == ".json" and name not in {"dataset"}
    }
    scope = payloads["execution_authorization"].get("authorization_scope", {})
    if (
        scope.get("formal_confirmation_submission_authorized") is not True
        or scope.get("formal_held_out_outcome_access_authorized") is not True
        or payloads["local_deployment_audit"].get("passed") is not True
        or payloads["deployment_sync"].get("passed") is not True
        or payloads["remote_verification"].get("passed") is not True
        or payloads["remote_verification"].get("remote_sumo_executed") is not False
        or payloads["source_freeze"].get("formal_confirmation_submission_authorized")
        is not False
        or payloads["source_freeze"].get("formal_held_out_outcomes_read") is not False
        or payloads["parent_v20_closeout"].get("formal_confirmation_eligible")
        is not True
        or payloads["parent_v20_verification"].get("formal_confirmation_eligible")
        is not True
    ):
        raise ValueError("formal submission authorization or evidence boundary drifted")
    task_command = command()
    queue_payload = json.loads(queue.expanduser().resolve(strict=True).read_text())
    duplicates = [
        {"id": row.get("id"), "status": row.get("status"), "signature": row.get("signature")}
        for row in queue_payload.get("tasks", [])
        if row.get("signature") == SIGNATURE or row.get("cmd") == task_command
    ]
    if duplicates:
        raise ValueError(f"refusing duplicate formal submission: {duplicates}")
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    rc, stdout, stderr = scheduler.run_on(
        "jtl110gpu",
        f"test ! -e {shlex.quote(str(RESULT_ROOT))}",
        timeout=30,
        check=False,
    )
    if rc != 0:
        raise FileExistsError(
            f"remote formal result destination already exists: {stdout or stderr}"
        )
    return {"paths": bound, "command": task_command}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument("--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_submission_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    checked = verify_inputs(args.queue, scheduler_root)
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
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256": (
            "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"
        ),
        "H2OPLUS_V20_FORMAL_CONFIRMATION_PROTOCOL_SHA256": HASHES["protocol"],
    }
    submit = [
        sys.executable,
        "skill/scheduler.py",
        "submit",
        "--description",
        "H2O+ v20 endpoint-only formal confirmation (1350 CPU evaluation episodes)",
        "--cmd",
        checked["command"],
        "--cwd",
        str(REPO),
        "--signature",
        SIGNATURE,
        "--resource-family",
        "h2oplus-v20-formal-confirmation-eval-worker-v1",
        "--vram",
        "0",
        "--ram-mb",
        "32768",
        "--cpu",
        "12",
        "--priority",
        "high",
        "--project",
        "h2oplus-v20-formal-confirmation",
        "--require-node",
        "jtl110gpu",
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
            f"formal scheduler submission failed: rc={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    task_id = match.group(1)
    record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-submission-v1",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "task_count": 1,
        "total_remote_sumo_episodes": 1350,
        "physical_behavior_count": 45,
        "formal_block_count": 10,
        "jobs_per_block": 135,
        "predesignated_candidate": "h2oplus-v19-completion-safety-quarter-load-sixty",
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_outcomes_read": False,
        "training": False,
        "scheduler_result_sync_opted_in": False,
        "result_sync_mode": "dedicated-frozen-allowlist-only",
        "checkpoint_optimizer_replay_or_workspace_sync_authorized": False,
        "task": {
            "task_id": task_id,
            "signature": SIGNATURE,
            "host": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "est_vram_mb": 0,
            "expected_jobs": 1350,
            "remote_result_dir": str(RESULT_ROOT),
            "command": checked["command"],
            "success_marker": "DONE",
        },
        "bindings": {
            name: {"path": str(checked["paths"][name]), "sha256": digest}
            for name, digest in HASHES.items()
            if name in checked["paths"]
        },
        "deployment_aggregate_sha256": HASHES["deployment_aggregate"],
        "scheduler_stdout": completed.stdout.strip(),
        "scheduler_stderr": completed.stderr.strip(),
    }
    atomic_json(args.output.expanduser().resolve(), record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
