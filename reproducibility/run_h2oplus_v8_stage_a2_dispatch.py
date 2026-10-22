#!/usr/bin/env python3
"""Run one provenance-locked CPU shard of the v8 Stage A2 SAC screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    sha256,
    sumo_runtime_fingerprint,
)
from reproducibility.run_evaluation_matrix_v4 import METHODS_SCHEMAS, load_methods
from reproducibility.run_h2oplus_v8_stage_a_dispatch import (
    block_command,
    verify_block_manifest,
    verify_design,
    write_json_atomic,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-registration-v1"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-authorization-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-freeze-v1"
EXPORT_SCHEMA = "transit-h2oplus-v8-stage-a2-policy-export-v1"
DEPLOYMENT_SCHEMA = "transit-h2oplus-v8-stage-a2-deployment-v1"
EXECUTION_SCHEMA = "transit-h2oplus-v8-stage-a2-host-execution-v1"
SUMMARY_SCHEMA = "transit-h2oplus-v8-stage-a2-host-summary-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "registration",
        "authorization",
        "freeze",
        "environment_design",
        "methods_manifest",
        "policy_export_manifest",
        "dataset",
        "sumo_runtime_lock",
        "deployment_manifest",
        "stage_a_file_manifest",
        "stage_a_verification",
        "baseline_manifest",
        "baseline_audit",
        "output_root",
        "execution_json",
        "sumo_dir",
        "edge_xml",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in (
        "registration",
        "authorization",
        "freeze",
        "design",
        "methods",
        "policy_export",
        "dataset",
        "runtime_lock",
        "deployment_manifest",
        "stage_a_file_manifest",
        "stage_a_verification",
        "baseline_manifest",
        "baseline_audit",
    ):
        parser.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256", required=True
        )
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_bound_json(
    path: Path, expected_sha256: str, schema: str | None = None
) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    actual = sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"hash mismatch for {resolved}: expected {expected_sha256}, got {actual}"
        )
    payload = json.loads(resolved.read_text())
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"schema mismatch for {resolved}")
    return resolved, payload


def verify_policy_export(path: Path, payload: dict) -> None:
    root = path.parent
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 30:
        raise ValueError("Stage A2 export must bind exactly 30 compact files")
    for entry in files:
        source = (root / entry["path"]).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError(f"export path escapes or is absent: {source}")
        if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
            raise ValueError(f"export source drifted: {source}")
    if payload.get("payload_policy") != {
        "compact_final_actor_files": 10,
        "metadata_files": 10,
        "optimizer_or_replay_files": 0,
        "summary_files": 10,
        "training_checkpoint_files": 0,
    }:
        raise ValueError("Stage A2 export payload policy drifted")


def _freeze_bound(freeze: dict, key: str, expected: str) -> None:
    if freeze.get(key, {}).get("sha256") != expected:
        raise ValueError(f"freeze does not bind {key}")


def verify_inputs(args: argparse.Namespace) -> dict:
    registration_path, registration = read_bound_json(
        args.registration, args.expected_registration_sha256, REGISTRATION_SCHEMA
    )
    if (
        registration.get("status") != "registered-not-submitted"
        or registration.get("execution_authorized") is not False
        or registration.get("local_sumo_or_training_allowed") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("training") is not False
    ):
        raise ValueError("Stage A2 registration scope drifted")
    authorization_path, authorization = read_bound_json(
        args.authorization, args.expected_authorization_sha256, AUTHORIZATION_SCHEMA
    )
    if (
        authorization.get("authorization_decision")
        != "authorize-stage-a2-registration-and-execution"
        or authorization.get("formal_held_out_access_authorized") is not False
        or authorization.get("local_sumo_or_training_authorized") is not False
        or authorization.get("stage_b_authorized") is not False
        or authorization.get("registration", {}).get("sha256")
        != args.expected_registration_sha256
    ):
        raise ValueError("Stage A2 authorization drifted")
    freeze_path, freeze = read_bound_json(
        args.freeze, args.expected_freeze_sha256, FREEZE_SCHEMA
    )
    if (
        freeze.get("status") != "frozen-before-stage-A2-SAC-outcomes"
        or freeze.get("training") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("stage_b_authorized_or_submitted") is not False
    ):
        raise ValueError("Stage A2 freeze scope drifted")

    design_path, design = read_bound_json(
        args.environment_design, args.expected_design_sha256,
        "transit-h2oplus-v8-development-environment-v1"
    )
    blocks = verify_design(design_path, design)
    methods_path = args.methods_manifest.expanduser().resolve()
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("Stage A2 methods hash drifted")
    methods = load_methods(methods_path, METHODS_SCHEMAS["v7"])
    expected_pairs = {
        (algorithm, seed)
        for algorithm in ("sac-mixed", "sac-online")
        for seed in range(5101, 5106)
    }
    if {(item["algorithm"], item["training_seed"]) for item in methods} != expected_pairs:
        raise ValueError("Stage A2 method/seed matrix drifted")
    export_path, export = read_bound_json(
        args.policy_export_manifest,
        args.expected_policy_export_sha256,
        EXPORT_SCHEMA,
    )
    verify_policy_export(export_path, export)

    dataset = args.dataset.expanduser().resolve()
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("canonical Stage A2 dataset drifted")
    stage_a_manifest_path, stage_a_manifest = read_bound_json(
        args.stage_a_file_manifest,
        args.expected_stage_a_file_manifest_sha256,
        "transit-h2oplus-v8-stage-a-file-manifest-v1",
    )
    stage_a_verification_path, stage_a_verification = read_bound_json(
        args.stage_a_verification,
        args.expected_stage_a_verification_sha256,
        "transit-h2oplus-v8-stage-a-file-manifest-verification-v1",
    )
    if (
        stage_a_manifest.get("artifact_integrity", {}).get("passed") is not True
        or stage_a_manifest.get("artifact_integrity", {}).get("result_count") != 150
        or stage_a_verification.get("passed") is not True
        or stage_a_verification.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("sealed Stage A H2O+ reference is not verified")
    baseline_manifest_path, baseline_manifest = read_bound_json(
        args.baseline_manifest,
        args.expected_baseline_manifest_sha256,
        "transit-formal-baseline-merge-manifest-v7-v1",
    )
    baseline_audit_path, baseline_audit = read_bound_json(
        args.baseline_audit, args.expected_baseline_audit_sha256
    )
    if (
        baseline_manifest.get("artifact_integrity_verified") is not True
        or baseline_manifest.get("status") != "verified"
        or baseline_audit.get("passed") is not True
        or baseline_audit.get("jobs_audited") != 35
    ):
        raise ValueError("SAC source baselines are not independently verified")

    deployment_path, deployment = read_bound_json(
        args.deployment_manifest,
        args.expected_deployment_manifest_sha256,
        DEPLOYMENT_SCHEMA,
    )
    if (
        deployment.get("training") is not False
        or deployment.get("study_stage") != "development"
        or deployment.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("Stage A2 deployment scope drifted")
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND, expected_version=FORMAL_V7_SUMO_VERSION
    )
    if (
        active_runtime["portable_signature"] != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise RuntimeError("active SUMO runtime differs from the Stage A2 lock")

    for key, expected in (
        ("registration", args.expected_registration_sha256),
        ("execution_authorization", args.expected_authorization_sha256),
        ("environment_design", args.expected_design_sha256),
        ("methods_manifest", args.expected_methods_sha256),
        ("policy_export_manifest", args.expected_policy_export_sha256),
        ("dataset", args.expected_dataset_sha256),
        ("sumo_runtime_lock", args.expected_runtime_lock_sha256),
        ("reference_stage_a_file_manifest", args.expected_stage_a_file_manifest_sha256),
        ("reference_stage_a_verification", args.expected_stage_a_verification_sha256),
        ("baseline_manifest", args.expected_baseline_manifest_sha256),
        ("baseline_audit", args.expected_baseline_audit_sha256),
    ):
        _freeze_bound(freeze, key, expected)

    return {
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "design": {"path": str(design_path), "sha256": sha256(design_path)},
        "methods": {"path": str(methods_path), "sha256": sha256(methods_path)},
        "policy_export": {"path": str(export_path), "sha256": sha256(export_path)},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "stage_a_file_manifest": {
            "path": str(stage_a_manifest_path),
            "sha256": sha256(stage_a_manifest_path),
        },
        "stage_a_verification": {
            "path": str(stage_a_verification_path),
            "sha256": sha256(stage_a_verification_path),
        },
        "baseline_manifest": {
            "path": str(baseline_manifest_path),
            "sha256": sha256(baseline_manifest_path),
        },
        "baseline_audit": {
            "path": str(baseline_audit_path),
            "sha256": sha256(baseline_audit_path),
        },
        "runtime_lock": runtime_identity,
        "runtime_fingerprint": active_runtime,
        "deployment_manifest": {
            "path": str(deployment_path),
            "sha256": sha256(deployment_path),
            "aggregate_sha256": deployment.get("aggregate_sha256"),
        },
        "blocks": blocks,
    }


def main() -> int:
    args = parse_args()
    if args.workers != 12 or args.shard_count != 2 or args.shard_index not in (0, 1):
        raise ValueError("Stage A2 is frozen to 12 workers and two shards")
    output_root = args.output_root.expanduser().resolve()
    execution_path = args.execution_json.expanduser().resolve()
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "starting",
        "started_utc": utc_now(),
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "workers": args.workers,
        "training": False,
        "formal_held_out_outcomes_read": False,
        "completed_blocks": [],
    }
    write_json_atomic(execution_path, execution)
    try:
        evidence = verify_inputs(args)
        execution.update({"status": "running", "evidence": evidence})
        write_json_atomic(execution_path, execution)
        output_root.mkdir(parents=True, exist_ok=True)
        completed = []
        suffix = f".shard-{args.shard_index:04d}-of-{args.shard_count:04d}"
        for block in evidence["blocks"]:
            block_output = output_root / block["name"]
            block_output.mkdir(parents=True, exist_ok=True)
            command = block_command(args, block, block_output)
            log_path = output_root / f"{block['name']}.shard-{args.shard_index}.log"
            execution["active_block"] = block["name"]
            execution["active_command"] = command
            execution["active_log"] = str(log_path)
            write_json_atomic(execution_path, execution)
            with log_path.open("a") as log:
                log.write(f"[{utc_now()}] START {' '.join(command)}\n")
                log.flush()
                import subprocess

                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log.write(f"[{utc_now()}] RETURN_CODE {result.returncode}\n")
            if result.returncode != 0:
                raise RuntimeError(
                    f"block {block['name']} returned {result.returncode}; see {log_path}"
                )
            manifest_path = block_output / f"matrix_manifest{suffix}.json"
            completed.append(verify_block_manifest(manifest_path, args))
            execution["completed_blocks"] = completed
            write_json_atomic(execution_path, execution)
        if sum(item["selected_job_count"] for item in completed) != 75:
            raise ValueError("Stage A2 shard did not complete exactly 75 jobs")
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "completed_utc": utc_now(),
            "hostname": os.uname().nodename,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "completed_blocks": completed,
            "completed_jobs": 75,
            "training": False,
            "formal_held_out_outcomes_read": False,
            "evidence": evidence,
        }
        summary_path = output_root / "host_summary.json"
        write_json_atomic(summary_path, summary)
        execution.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "returncode": 0,
                "active_block": None,
                "active_command": None,
                "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            }
        )
        write_json_atomic(execution_path, execution)
        print(f"DONE H2O+ v8 Stage A2 shard {args.shard_index}: 75/75 jobs", flush=True)
        return 0
    except Exception as error:
        execution.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "returncode": 1,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json_atomic(execution_path, execution)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
