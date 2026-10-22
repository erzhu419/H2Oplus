#!/usr/bin/env python3
"""Run one fail-closed remote CPU shard of the v8 few-shot evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
)
from reproducibility.run_evaluation_matrix_v8_fewshot import (
    METHODS_SCHEMA,
    load_development_methods,
)
from reproducibility.h2oplus_v8_fewshot_correction import (
    CORRECTION_SCHEMAS,
    validate_evaluation_correction,
)


EXECUTION_SCHEMA = "transit-h2oplus-v8-fewshot-host-execution-v1"
SUMMARY_SCHEMA = "transit-h2oplus-v8-fewshot-host-summary-v1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_bound(path: Path, expected: str, schema: str | None = None) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected:
        raise ValueError(f"identity drift for {resolved}: {actual}")
    payload = json.loads(resolved.read_text())
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"schema drift for {resolved}")
    return resolved, payload


def resolve_relative(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def verify_design(path: Path, payload: dict) -> list[dict]:
    if (
        payload.get("study_stage") != "development"
        or payload.get("role") != "few-shot sample-efficiency comparison; not formal held-out evaluation"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("disjointness", {}).get("verified_disjoint") is not True
        or any(
            value
            for key, value in payload.get("disjointness", {}).items()
            if key.endswith("_overlap")
        )
    ):
        raise ValueError("few-shot environment scope drifted")
    blocks = payload.get("blocks", [])
    if len(blocks) != 3 or payload.get("environment_cells") != 9:
        raise ValueError("few-shot environment must contain three blocks and nine cells")
    result = []
    names, sumo_seeds, demand_seeds = set(), set(), set()
    for block in blocks:
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        raw_manifests = block.get("scenario_manifests", [])
        if (
            not isinstance(name, str)
            or not SAFE_NAME.fullmatch(name)
            or name in names
            or not isinstance(sumo_seed, int)
            or isinstance(sumo_seed, bool)
            or sumo_seed in sumo_seeds
            or not isinstance(demand_seed, int)
            or isinstance(demand_seed, bool)
            or demand_seed in demand_seeds
            or len(raw_manifests) != 3
        ):
            raise ValueError(f"malformed few-shot block: {block!r}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
        manifests, scales = [], []
        for raw in raw_manifests:
            manifest = resolve_relative(path, raw)
            _, evidence = load_validated_scenario(manifest)
            if evidence.get("selection_seed") != demand_seed:
                raise ValueError(f"development demand seed drifted in {manifest}")
            manifests.append(str(manifest))
            scales.append(float(evidence["demand_scale"]))
        if sorted(scales) != [0.85, 1.0, 1.15]:
            raise ValueError(f"development demand scales drifted in {name}")
        result.append({"name": name, "sumo_seed": sumo_seed, "manifests": manifests})
    return result


def verify_inputs(args: argparse.Namespace) -> dict:
    registration_path, registration = read_bound(
        args.registration,
        args.expected_registration_sha256,
        "transit-h2oplus-v8-fewshot-registration-v1",
    )
    authorization_path, authorization = read_bound(
        args.authorization,
        args.expected_authorization_sha256,
        "transit-h2oplus-v8-fewshot-authorization-v1",
    )
    freeze_path, freeze = read_bound(
        args.freeze,
        args.expected_freeze_sha256,
        "transit-h2oplus-v8-fewshot-freeze-v1",
    )
    design_path, design = read_bound(
        args.environment_design,
        args.expected_design_sha256,
        "transit-h2oplus-v8-fewshot-development-environment-v1",
    )
    protocol_path, protocol = read_bound(
        args.protocol,
        args.expected_protocol_sha256,
        "transit-h2oplus-v8-fewshot-protocol-v1",
    )
    correction_path, correction = read_bound(
        args.evaluation_correction,
        args.expected_evaluation_correction_sha256,
    )
    if correction.get("schema_version") not in CORRECTION_SCHEMAS:
        raise ValueError("few-shot evaluation correction schema drifted")
    corrections = validate_evaluation_correction(
        correction, args.expected_protocol_sha256
    )
    closeout_path, closeout = read_bound(
        args.training_closeout,
        args.expected_training_closeout_sha256,
        "transit-h2oplus-v8-fewshot-training-closeout-v1",
    )
    verification_path, verification = read_bound(
        args.training_verification,
        args.expected_training_verification_sha256,
        "transit-h2oplus-v8-fewshot-training-verification-v1",
    )
    methods_path = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("few-shot methods identity drifted")
    methods = load_development_methods(methods_path, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("few-shot dataset identity drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("few-shot deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or authorization.get("formal_held_out_access_authorized") is not False
        or authorization.get("local_training_or_sumo_authorized") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or correction.get("scope") != "evaluation-provenance-propagation-only"
        or correction.get("training_or_policy_artifacts_changed") is not False
        or correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("frozen_protocol_sha256") != args.expected_protocol_sha256
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or verification.get("passed") is not True
        or len(methods) != 70
        or deployment.get("formal_held_out_outcomes_included") is not False
        or deployment.get("training") is not False
        or deployment.get("methods_manifest", {}).get("sha256") != args.expected_methods_sha256
        or deployment.get("training_closeout", {}).get("sha256")
        != args.expected_training_closeout_sha256
        or deployment.get("training_verification", {}).get("sha256")
        != args.expected_training_verification_sha256
        or deployment.get("evaluation_correction", {}).get("sha256")
        != args.expected_evaluation_correction_sha256
        or active_runtime["portable_signature"] != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise ValueError("few-shot dispatch scope or immutable evidence drifted")
    blocks = verify_design(design_path, design)
    return {
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "environment_design": {"path": str(design_path), "sha256": sha256(design_path)},
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "evaluation_correction": {
            "path": str(correction_path),
            "sha256": sha256(correction_path),
            "schema_version": correction["schema_version"],
            "source_replacements": len(corrections),
        },
        "training_closeout": {"path": str(closeout_path), "sha256": sha256(closeout_path)},
        "training_verification": {"path": str(verification_path), "sha256": sha256(verification_path)},
        "methods": {"path": str(methods_path), "sha256": sha256(methods_path), "count": len(methods)},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "deployment_manifest": {
            "path": str(deployment_path),
            "sha256": sha256(deployment_path),
            "aggregate_sha256": deployment.get("aggregate_sha256"),
        },
        "runtime_lock": runtime_identity,
        "runtime_fingerprint": active_runtime,
        "blocks": blocks,
    }


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility/run_evaluation_matrix_v8_fewshot.py"),
        "--mdp-version", "v7",
        "--methods-manifest", str(args.methods_manifest.expanduser().resolve()),
        "--scenario-manifests", *(str(path) for path in block["manifests"]),
        "--sumo-seeds", str(block["sumo_seed"]),
        "--dataset", str(args.dataset.expanduser().resolve()),
        "--output-root", str(output),
        "--workers", str(args.workers),
        "--shard-count", str(args.shard_count),
        "--shard-index", str(args.shard_index),
        "--device", "cpu",
        "--sumo-backend", FORMAL_V7_SUMO_BACKEND,
        "--expected-sumo-runtime-version", FORMAL_V7_SUMO_VERSION,
        "--sumo-runtime-lock", str(args.sumo_runtime_lock.expanduser().resolve()),
        "--sumo-dir", str(args.sumo_dir.expanduser().resolve()),
        "--edge-xml", str(args.edge_xml.expanduser().resolve()),
    ]


def verify_block(path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(path.read_text())
    partition = payload.get("matrix_partition", {})
    jobs = payload.get("jobs", [])
    if (
        payload.get("schema_version") != "transit-evaluation-matrix-v7"
        or payload.get("methods_manifest_sha256") != args.expected_methods_sha256
        or payload.get("dataset_sha256") != args.expected_dataset_sha256
        or payload.get("sumo_runtime_lock", {}).get("sha256") != args.expected_runtime_lock_sha256
        or partition.get("shard_count") != 2
        or partition.get("shard_index") != args.shard_index
        or partition.get("full_job_count") != 210
        or partition.get("selected_job_count") != 105
        or len(jobs) != 105
    ):
        raise ValueError(f"completed few-shot block manifest drifted: {path}")
    signatures = set()
    for job in jobs:
        output = Path(job["output"])
        if not output.is_file():
            raise FileNotFoundError(output)
        result = json.loads(output.read_text())
        signature = job["signature"]
        if (
            result.get("schema_version") != "transit-evaluation-v7"
            or result.get("evaluation_job", {}).get("signature") != signature
            or signature in signatures
        ):
            raise ValueError(f"few-shot result binding drifted: {output}")
        signatures.add(signature)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "full_matrix_signature": partition["full_matrix_signature"],
        "selected_job_keys_sha256": partition["selected_job_keys_sha256"],
        "selected_job_count": 105,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration", "authorization", "freeze", "environment_design", "protocol",
        "evaluation_correction",
        "training_closeout", "training_verification", "methods_manifest", "dataset",
        "sumo_runtime_lock", "deployment_manifest", "output_root", "execution_json",
        "sumo_dir", "edge_xml",
    )
    for name in paths:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    expected = (
        "registration", "authorization", "freeze", "design", "protocol",
        "evaluation_correction",
        "training_closeout", "training_verification", "methods", "dataset",
        "runtime_lock", "deployment_manifest",
    )
    for name in expected:
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != 12 or args.shard_count != 2 or args.shard_index not in (0, 1):
        raise ValueError("few-shot evaluation is frozen to 12 workers and two shards")
    output_root = args.output_root.expanduser().resolve()
    execution_path = args.execution_json.expanduser().resolve()
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "starting",
        "started_utc": utc_now(),
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "workers": 12,
        "shard_count": 2,
        "shard_index": args.shard_index,
        "training": False,
        "formal_held_out_outcomes_read": False,
        "completed_blocks": [],
    }
    atomic_json(execution_path, execution)
    try:
        evidence = verify_inputs(args)
        execution.update({"status": "running", "evidence": evidence})
        atomic_json(execution_path, execution)
        output_root.mkdir(parents=True, exist_ok=True)
        completed = []
        suffix = f".shard-{args.shard_index:04d}-of-0002"
        for block in evidence["blocks"]:
            block_output = output_root / block["name"]
            block_output.mkdir(parents=True, exist_ok=True)
            command = block_command(args, block, block_output)
            log_path = output_root / f"{block['name']}.shard-{args.shard_index}.log"
            execution.update({
                "active_block": block["name"],
                "active_command": command,
                "active_log": str(log_path),
            })
            atomic_json(execution_path, execution)
            with log_path.open("a") as log:
                log.write(f"[{utc_now()}] START {' '.join(command)}\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                log.write(f"[{utc_now()}] RETURN_CODE {result.returncode}\n")
            if result.returncode != 0:
                raise RuntimeError(f"block {block['name']} returned {result.returncode}")
            manifest_path = block_output / f"matrix_manifest{suffix}.json"
            completed.append(verify_block(manifest_path, args))
            execution["completed_blocks"] = completed
            atomic_json(execution_path, execution)
        if sum(item["selected_job_count"] for item in completed) != 315:
            raise ValueError("few-shot shard did not complete exactly 315 jobs")
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "completed_utc": utc_now(),
            "hostname": os.uname().nodename,
            "workers": 12,
            "shard_count": 2,
            "shard_index": args.shard_index,
            "completed_blocks": completed,
            "completed_jobs": 315,
            "training": False,
            "formal_held_out_outcomes_read": False,
            "evidence": evidence,
        }
        summary_path = output_root / "host_summary.json"
        atomic_json(summary_path, summary)
        execution.update({
            "status": "completed",
            "completed_utc": utc_now(),
            "returncode": 0,
            "active_block": None,
            "active_command": None,
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        })
        atomic_json(execution_path, execution)
        print(f"DONE H2O+ v8 few-shot shard {args.shard_index}: 315/315 jobs", flush=True)
        return 0
    except Exception as error:
        execution.update({
            "status": "failed",
            "failed_utc": utc_now(),
            "returncode": 1,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        atomic_json(execution_path, execution)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
