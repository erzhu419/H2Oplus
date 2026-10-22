#!/usr/bin/env python3
"""Run one fail-closed remote CPU shard of the v9 adaptive evaluation."""

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
from reproducibility.run_evaluation_matrix_v9_adaptive import (
    METHODS_SCHEMA,
    load_development_methods,
)


EXECUTION_SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-host-execution-v1"
SUMMARY_SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-host-summary-v1"
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
    disjointness = payload.get("disjointness", {})
    if (
        payload.get("study_stage") != "development"
        or payload.get("formal_held_out_outcomes_read") is not False
        or disjointness.get("verified_disjoint") is not True
        or any(value for key, value in disjointness.items() if key.endswith("_overlap"))
        or payload.get("environment_cells") != 6
    ):
        raise ValueError("v9 development environment scope drifted")
    blocks = payload.get("blocks", [])
    if len(blocks) != 2:
        raise ValueError("v9 evaluation requires exactly two development blocks")
    result, names, sumo_seeds, demand_seeds = [], set(), set(), set()
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
            raise ValueError(f"malformed v9 development block: {block!r}")
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


def _source_freeze_map(payload: dict) -> dict[str, str]:
    rows = payload.get("sources", [])
    result = {}
    for row in rows:
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or path in result:
            raise ValueError("malformed source freeze")
        result[path] = digest
    return result


def verify_inputs(args: argparse.Namespace) -> dict:
    registration_path, registration = read_bound(
        args.registration,
        args.expected_registration_sha256,
        "transit-h2oplus-v9-adaptive-registration-v2",
    )
    authorization_path, authorization = read_bound(
        args.authorization,
        args.expected_authorization_sha256,
        "transit-h2oplus-v9-adaptive-authorization-v2",
    )
    freeze_path, freeze = read_bound(
        args.freeze,
        args.expected_freeze_sha256,
        "transit-h2oplus-v9-adaptive-freeze-v2",
    )
    design_path, design = read_bound(
        args.environment_design,
        args.expected_design_sha256,
        "transit-h2oplus-v9-adaptive-environment-v2",
    )
    protocol_path, protocol = read_bound(
        args.protocol,
        args.expected_protocol_sha256,
        "transit-h2oplus-v9-adaptive-protocol-v2",
    )
    count_path, count = read_bound(
        args.count_correction,
        args.expected_count_correction_sha256,
        "transit-h2oplus-v9-adaptive-evaluation-count-correction-v1",
    )
    closeout_path, closeout = read_bound(
        args.training_closeout,
        args.expected_training_closeout_sha256,
        "transit-h2oplus-v9-adaptive-training-closeout-v1",
    )
    verification_path, verification = read_bound(
        args.training_verification,
        args.expected_training_verification_sha256,
        "transit-h2oplus-v9-adaptive-training-verification-v1",
    )
    audit_path, audit = read_bound(
        args.training_audit,
        args.expected_training_audit_sha256,
        "transit-h2oplus-v9-adaptive-training-audit-v1",
    )
    manifest_path, training_manifest = read_bound(
        args.training_file_manifest,
        args.expected_training_file_manifest_sha256,
        "transit-h2oplus-v9-adaptive-training-file-manifest-v1",
    )
    source_path, source_freeze = read_bound(
        args.source_freeze,
        args.expected_source_freeze_sha256,
        "transit-h2oplus-v9-adaptive-evaluation-source-freeze-v1",
    )
    methods_path = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("v9 methods identity drifted")
    methods = load_development_methods(methods_path, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("v9 dataset identity drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("v9 deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    sources = _source_freeze_map(source_freeze)
    for relative in (
        "SimpleSAC/evaluate_transit_v9_adaptive.py",
        "reproducibility/run_evaluation_matrix_v9_adaptive.py",
        "reproducibility/run_h2oplus_v9_adaptive_evaluation_dispatch.py",
        "reproducibility/analyze_h2oplus_v9_adaptive.py",
    ):
        path = ROOT / relative
        if sources.get(relative) != sha256(path):
            raise ValueError(f"v9 evaluation source drifted: {relative}")
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or authorization.get("formal_held_out_access_authorized") is not False
        or authorization.get("local_training_or_sumo_authorized") is not False
        or authorization.get("remote_development_evaluation_after_training_authorized") is not True
        or freeze.get("formal_held_out_outcomes_read") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or count.get("performance_or_evaluation_outcomes_read") is not False
        or count.get("corrected_unique_actor_tensors") != 42
        or count.get("corrected_remote_sumo_episodes") != 252
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("development_evaluation_may_proceed") is not True
        or verification.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or training_manifest.get("file_count") != verification.get("checked_files")
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("evaluation_outcomes_read") is not False
        or source_freeze.get("physical_actor_count") != 42
        or len(methods) != 42
        or deployment.get("methods_manifest", {}).get("sha256") != args.expected_methods_sha256
        or deployment.get("source_freeze", {}).get("sha256") != args.expected_source_freeze_sha256
        or deployment.get("count_correction", {}).get("sha256") != args.expected_count_correction_sha256
        or deployment.get("training_closeout", {}).get("sha256") != args.expected_training_closeout_sha256
        or deployment.get("training_verification", {}).get("sha256") != args.expected_training_verification_sha256
        or deployment.get("sumo_runtime_signature") != runtime_identity["runtime_signature"]
        or active_runtime["portable_signature"] != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise ValueError("v9 evaluation scope or immutable evidence drifted")
    return {
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "environment_design": {"path": str(design_path), "sha256": sha256(design_path)},
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "count_correction": {"path": str(count_path), "sha256": sha256(count_path)},
        "training_closeout": {"path": str(closeout_path), "sha256": sha256(closeout_path)},
        "training_verification": {"path": str(verification_path), "sha256": sha256(verification_path)},
        "training_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "training_file_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "source_freeze": {"path": str(source_path), "sha256": sha256(source_path)},
        "methods": {"path": str(methods_path), "sha256": sha256(methods_path), "count": 42},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "deployment_manifest": {
            "path": str(deployment_path),
            "sha256": sha256(deployment_path),
            "aggregate_sha256": deployment.get("aggregate_sha256"),
        },
        "runtime_lock": runtime_identity,
        "runtime_fingerprint": active_runtime,
        "blocks": verify_design(design_path, design),
    }


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility/run_evaluation_matrix_v9_adaptive.py"),
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
        or partition.get("full_job_count") != 126
        or partition.get("selected_job_count") != 63
        or len(jobs) != 63
    ):
        raise ValueError(f"completed v9 block manifest drifted: {path}")
    signatures = set()
    source_hashes = payload.get("source_sha256")
    for job in jobs:
        output = Path(job["output"])
        if not output.is_file():
            raise FileNotFoundError(output)
        result = json.loads(output.read_text())
        signature = job["signature"]
        if (
            result.get("schema_version") != "transit-evaluation-v7"
            or result.get("evaluation_job", {}).get("signature") != signature
            or result.get("source_sha256") != source_hashes
            or signature in signatures
        ):
            raise ValueError(f"v9 result binding drifted: {output}")
        signatures.add(signature)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "full_matrix_signature": partition["full_matrix_signature"],
        "selected_job_keys_sha256": partition["selected_job_keys_sha256"],
        "selected_job_count": 63,
        "source_file_count": len(source_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration", "authorization", "freeze", "environment_design", "protocol",
        "count_correction", "training_closeout", "training_verification", "training_audit",
        "training_file_manifest", "source_freeze", "methods_manifest", "dataset",
        "sumo_runtime_lock", "deployment_manifest", "output_root", "execution_json",
        "sumo_dir", "edge_xml",
    )
    for name in paths:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    expected = (
        "registration", "authorization", "freeze", "design", "protocol",
        "count_correction", "training_closeout", "training_verification", "training_audit",
        "training_file_manifest", "source_freeze", "methods", "dataset", "runtime_lock",
        "deployment_manifest",
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
        raise ValueError("v9 evaluation is frozen to 12 workers and two shards")
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
                result = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
                )
                log.write(f"[{utc_now()}] RETURN_CODE {result.returncode}\n")
            if result.returncode != 0:
                raise RuntimeError(f"block {block['name']} returned {result.returncode}")
            manifest = block_output / f"matrix_manifest{suffix}.json"
            completed.append(verify_block(manifest, args))
            execution["completed_blocks"] = completed
            atomic_json(execution_path, execution)
        if sum(item["selected_job_count"] for item in completed) != 126:
            raise ValueError("v9 shard did not complete exactly 126 jobs")
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "completed_utc": utc_now(),
            "hostname": os.uname().nodename,
            "workers": 12,
            "shard_count": 2,
            "shard_index": args.shard_index,
            "completed_blocks": completed,
            "completed_jobs": 126,
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
        print(f"DONE H2O+ v9 adaptive shard {args.shard_index}: 126/126 jobs", flush=True)
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
