#!/usr/bin/env python3
"""Run one deterministic remote shard of the v8 Stage A development screen."""

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
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    load_methods,
)


DESIGN_SCHEMA = "transit-h2oplus-v8-development-environment-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-stage-a-freeze-v1"
AUTH_SCHEMA = "transit-h2oplus-v8-development-execution-authorization-v1"
REGISTRATION_SCHEMA = "transit-h2oplus-v8-development-pilot-registration-v1"
EXPORT_SCHEMA = "transit-h2oplus-v8-stage-a-policy-export-v1"
EXECUTION_SCHEMA = "transit-h2oplus-v8-stage-a-host-execution-v1"
SUMMARY_SCHEMA = "transit-h2oplus-v8-stage-a-host-summary-v1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--environment-design", type=Path, required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--policy-export-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sumo-runtime-lock", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--sumo-dir", type=Path, required=True)
    parser.add_argument("--edge-xml", type=Path, required=True)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--expected-methods-sha256", required=True)
    parser.add_argument("--expected-policy-export-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-runtime-lock-sha256", required=True)
    parser.add_argument("--expected-deployment-manifest-sha256", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_bound_json(path: Path, expected_sha256: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    actual = sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"hash mismatch for {resolved}: expected {expected_sha256}, got {actual}"
        )
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema mismatch for {resolved}")
    return resolved, payload


def resolve_relative(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_design(design_path: Path, design: dict) -> list[dict]:
    if (
        design.get("status") != "frozen-before-stage-A-outcomes"
        or design.get("study_stage") != "development"
        or design.get("role") != "candidate screen; not formal held-out evaluation"
        or design.get("formal_held_out_outcomes_read") is not None
        and design.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("development environment design scope drifted")
    disjointness = design.get("disjointness", {})
    overlap_keys = (
        "held_out_demand_seed_overlap",
        "held_out_manifest_hash_overlap",
        "held_out_runtime_hash_overlap",
        "held_out_sumo_seed_overlap",
        "training_demand_seed_overlap",
        "training_manifest_hash_overlap",
        "training_runtime_hash_overlap",
        "training_sumo_seed_overlap",
    )
    if (
        disjointness.get("verified_disjoint") is not True
        or disjointness.get("formal_held_out_outcomes_read") is not False
        or any(disjointness.get(key) != [] for key in overlap_keys)
    ):
        raise ValueError("development environment is not verified disjoint")
    blocks = design.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise ValueError("Stage A requires exactly five development blocks")
    names, sumo_seeds, demand_seeds, descriptors = set(), set(), set(), []
    for block in blocks:
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        raw_manifests = block.get("scenario_manifests")
        if (
            not isinstance(name, str)
            or not SAFE_NAME.fullmatch(name)
            or name in names
            or isinstance(sumo_seed, bool)
            or not isinstance(sumo_seed, int)
            or sumo_seed in sumo_seeds
            or isinstance(demand_seed, bool)
            or not isinstance(demand_seed, int)
            or demand_seed in demand_seeds
            or not isinstance(raw_manifests, list)
            or len(raw_manifests) != 3
        ):
            raise ValueError(f"malformed development block: {block!r}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
        manifests = []
        scales = []
        for value in raw_manifests:
            manifest = resolve_relative(design_path, value)
            _, evidence = load_validated_scenario(manifest)
            # Execution evidence is persisted as JSON before any SUMO process
            # starts, so descriptors must remain JSON-native.
            manifests.append(str(manifest))
            scales.append(float(evidence["demand_scale"]))
            if evidence.get("selection_seed") != demand_seed:
                raise ValueError(f"demand seed drift in {manifest}")
        if sorted(scales) != [0.85, 1.0, 1.15]:
            raise ValueError(f"demand scale drift in block {name}")
        descriptors.append(
            {"name": name, "sumo_seed": sumo_seed, "manifests": manifests}
        )
    if design.get("environment_cells") != 15:
        raise ValueError("Stage A environment-cell count drifted")
    return descriptors


def verify_policy_export(export_path: Path, payload: dict) -> None:
    root = export_path.parent
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 30:
        raise ValueError("policy export must bind exactly 30 compact files")
    for entry in files:
        path = (root / entry["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"policy export path escapes or is missing: {path}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"policy export file drifted: {path}")
    policy = payload.get("payload_policy", {})
    if policy != {
        "compact_final_actor_files": 10,
        "metadata_files": 10,
        "optimizer_or_replay_files": 0,
        "summary_files": 10,
        "training_checkpoint_files": 0,
    }:
        raise ValueError("policy export payload policy drifted")


def verify_inputs(args: argparse.Namespace) -> dict:
    registration_path, registration = read_bound_json(
        args.registration,
        args.expected_registration_sha256,
        REGISTRATION_SCHEMA,
    )
    if (
        registration.get("execution_authorized") is not False
        or registration.get("status") != "registered-not-submitted"
        or registration.get("local_sumo_or_training_allowed") is not False
    ):
        raise ValueError("immutable v8 registration drifted")
    authorization_path, authorization = read_bound_json(
        args.authorization,
        args.expected_authorization_sha256,
        AUTH_SCHEMA,
    )
    if (
        authorization.get("authorization_decision")
        != "authorize-stage-a-execution"
        or authorization.get("formal_held_out_access_authorized") is not False
        or authorization.get("stage_b_training_authorized_before_stage_a_gate")
        is not False
        or authorization.get("registration", {}).get("sha256")
        != args.expected_registration_sha256
    ):
        raise ValueError("Stage A execution authorization drifted")
    freeze_path, freeze = read_bound_json(
        args.freeze, args.expected_freeze_sha256, FREEZE_SCHEMA
    )
    if (
        freeze.get("status") != "frozen-before-stage-A-outcomes"
        or freeze.get("training") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("stage_b_policy")
        != "blocked unless every registered Stage A gate passes"
    ):
        raise ValueError("Stage A freeze drifted")
    design_path, design = read_bound_json(
        args.environment_design, args.expected_design_sha256, DESIGN_SCHEMA
    )
    methods_path = args.methods_manifest.expanduser().resolve()
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("Stage A methods manifest hash drifted")
    methods = load_methods(methods_path, METHODS_SCHEMAS["v7"])
    expected_pairs = {
        (algorithm, seed)
        for algorithm in ("h2oplus", "h2oplus-lambda0p5")
        for seed in range(5101, 5106)
    }
    if {(item["algorithm"], item["training_seed"]) for item in methods} != expected_pairs:
        raise ValueError("Stage A method/seed matrix drifted")
    export_path, export = read_bound_json(
        args.policy_export_manifest,
        args.expected_policy_export_sha256,
        EXPORT_SCHEMA,
    )
    verify_policy_export(export_path, export)
    dataset = args.dataset.expanduser().resolve()
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("canonical Stage A dataset hash drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve()
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("deployment manifest hash drifted")
    deployment = json.loads(deployment_path.read_text())
    if deployment.get("formal_held_out_outcomes_included") is not False:
        raise ValueError("deployment does not exclude formal held-out outcomes")
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    if (
        active_runtime["portable_signature"]
        != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise RuntimeError("active SUMO runtime differs from Stage A lock")
    if freeze.get("environment_design", {}).get("sha256") != args.expected_design_sha256:
        raise ValueError("freeze does not bind the environment design")
    if freeze.get("methods_manifest", {}).get("sha256") != args.expected_methods_sha256:
        raise ValueError("freeze does not bind the methods manifest")
    if freeze.get("dataset", {}).get("sha256") != args.expected_dataset_sha256:
        raise ValueError("freeze does not bind the dataset")
    if freeze.get("execution_authorization", {}).get("sha256") != args.expected_authorization_sha256:
        raise ValueError("freeze does not bind the execution authorization")
    blocks = verify_design(design_path, design)
    return {
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "design": {"path": str(design_path), "sha256": sha256(design_path)},
        "methods": {"path": str(methods_path), "sha256": sha256(methods_path)},
        "policy_export": {"path": str(export_path), "sha256": sha256(export_path)},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "runtime_lock": runtime_identity,
        "runtime_fingerprint": active_runtime,
        "deployment_manifest": {
            "path": str(deployment_path),
            "sha256": sha256(deployment_path),
            "aggregate_sha256": deployment.get("aggregate_sha256"),
        },
        "blocks": blocks,
    }


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility" / "run_evaluation_matrix_v4.py"),
        "--mdp-version",
        "v7",
        "--methods-manifest",
        str(args.methods_manifest.expanduser().resolve()),
        "--scenario-manifests",
        *(str(path) for path in block["manifests"]),
        "--sumo-seeds",
        str(block["sumo_seed"]),
        "--dataset",
        str(args.dataset.expanduser().resolve()),
        "--output-root",
        str(output),
        "--workers",
        str(args.workers),
        "--shard-count",
        str(args.shard_count),
        "--shard-index",
        str(args.shard_index),
        "--device",
        "cpu",
        "--sumo-backend",
        FORMAL_V7_SUMO_BACKEND,
        "--expected-sumo-runtime-version",
        FORMAL_V7_SUMO_VERSION,
        "--sumo-runtime-lock",
        str(args.sumo_runtime_lock.expanduser().resolve()),
        "--sumo-dir",
        str(args.sumo_dir.expanduser().resolve()),
        "--edge-xml",
        str(args.edge_xml.expanduser().resolve()),
    ]


def verify_block_manifest(path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(path.read_text())
    partition = payload.get("matrix_partition", {})
    if (
        payload.get("schema_version") != "transit-evaluation-matrix-v7"
        or payload.get("methods_manifest_sha256") != args.expected_methods_sha256
        or payload.get("dataset_sha256") != args.expected_dataset_sha256
        or payload.get("sumo_runtime_lock", {}).get("sha256")
        != args.expected_runtime_lock_sha256
        or partition.get("shard_count") != args.shard_count
        or partition.get("shard_index") != args.shard_index
        or partition.get("full_job_count") != 30
        or partition.get("selected_job_count") != 15
        or len(payload.get("jobs", [])) != 15
    ):
        raise ValueError(f"invalid completed block manifest {path}")
    for job in payload["jobs"]:
        result = Path(job["output"])
        if not result.is_file():
            raise FileNotFoundError(result)
        result_payload = json.loads(result.read_text())
        if result_payload.get("schema_version") != "transit-evaluation-v7":
            raise ValueError(f"invalid Stage A result schema: {result}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "full_matrix_signature": partition["full_matrix_signature"],
        "selected_job_keys_sha256": partition["selected_job_keys_sha256"],
        "selected_job_count": partition["selected_job_count"],
    }


def main() -> int:
    args = parse_args()
    if args.workers != 12 or args.shard_count != 2 or args.shard_index not in (0, 1):
        raise ValueError("Stage A is frozen to 12 workers and two shards")
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
            raise ValueError("Stage A shard did not complete exactly 75 jobs")
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
        print(f"DONE H2O+ v8 Stage A shard {args.shard_index}: 75/75 jobs", flush=True)
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
