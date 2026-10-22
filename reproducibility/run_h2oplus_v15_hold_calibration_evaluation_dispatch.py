#!/usr/bin/env python3
"""Run the fail-closed remote CPU-only v15 development evaluation."""

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
from reproducibility.run_evaluation_matrix_v15_hold_calibration import (
    METHODS_SCHEMA,
    load_development_methods,
)


EXECUTION_SCHEMA = "transit-h2oplus-v15-hold-calibration-evaluation-host-execution-v1"
SUMMARY_SCHEMA = "transit-h2oplus-v15-hold-calibration-evaluation-host-summary-v1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
MATRIX_SOURCES = (
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "SimpleSAC/evaluate_transit_v13_proximal_release.py",
    "SimpleSAC/evaluate_transit_v14_terminal_taper.py",
    "SimpleSAC/evaluate_transit_v15_hold_calibration.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v15_hold_calibration_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v15_hold_calibration.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_bound(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"identity drift for {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
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
        or payload.get("environment_cells") != 9
    ):
        raise ValueError("frozen development environment scope drifted")
    blocks = payload.get("blocks", [])
    result, names, sumo_seeds, demand_seeds = [], set(), set(), set()
    for block in blocks:
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        manifests = [resolve_relative(path, raw) for raw in block.get("scenario_manifests", [])]
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
            or len(manifests) != 3
        ):
            raise ValueError(f"malformed v15 development block: {block!r}")
        scales = []
        for manifest in manifests:
            _, evidence = load_validated_scenario(manifest)
            if evidence.get("selection_seed") != demand_seed:
                raise ValueError(f"development demand seed drifted in {manifest}")
            scales.append(float(evidence["demand_scale"]))
        if sorted(scales) != [0.85, 1.0, 1.15]:
            raise ValueError(f"development demand scales drifted in {name}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
        result.append(
            {"name": name, "sumo_seed": sumo_seed, "manifests": [str(path) for path in manifests]}
        )
    expected_names = {f"v12-staged-release-development-0{index}" for index in range(1, 4)}
    if names != expected_names or sumo_seeds != {5101, 5102, 5103}:
        raise ValueError("v15 development block identity drifted")
    return sorted(result, key=lambda row: row["name"])


def _freeze_map(rows: list[dict], label: str) -> dict[str, str]:
    result = {}
    for row in rows:
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or path in result:
            raise ValueError(f"malformed v15 {label} source freeze")
        result[path] = digest
    return result


def verify_inputs(args: argparse.Namespace) -> dict:
    controls = (
        ("registration", "transit-h2oplus-v15-hold-calibration-registration-v1"),
        ("authorization", "transit-h2oplus-v15-hold-calibration-authorization-v1"),
        ("environment_design", "transit-h2oplus-v12-staged-release-environment-v2"),
        ("protocol", "transit-h2oplus-v15-hold-calibration-protocol-v1"),
        ("v14_analysis", "transit-h2oplus-v14-terminal-taper-analysis-v1"),
        ("v14_closeout", "transit-h2oplus-v14-terminal-taper-evaluation-closeout-v1"),
        (
            "v14_verification",
            "transit-h2oplus-v14-terminal-taper-evaluation-closeout-verification-v1",
        ),
        (
            "source_freeze",
            "transit-h2oplus-v15-hold-calibration-evaluation-source-freeze-v1",
        ),
    )
    loaded = {
        name: read_bound(getattr(args, name), getattr(args, f"expected_{name}_sha256"), schema)
        for name, schema in controls
    }
    methods_path = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("v15 methods identity drifted")
    methods = load_development_methods(methods_path, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("dataset identity drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("v15 deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    registration = loaded["registration"][1]
    authorization = loaded["authorization"][1]
    environment = loaded["environment_design"][1]
    protocol = loaded["protocol"][1]
    v14_analysis = loaded["v14_analysis"][1]
    v14_closeout = loaded["v14_closeout"][1]
    v14_verification = loaded["v14_verification"][1]
    source_freeze = loaded["source_freeze"][1]
    matrix_sources = _freeze_map(source_freeze.get("matrix_sources", []), "matrix")
    operational_sources = _freeze_map(source_freeze.get("operational_sources", []), "operational")
    if set(matrix_sources) != set(MATRIX_SOURCES) or set(operational_sources) != set(OPERATIONAL_SOURCES):
        raise ValueError("v15 scientific source set drifted")
    for relative, digest in {**matrix_sources, **operational_sources}.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"v15 scientific source drifted: {relative}")
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("v15_development_evaluation_outcomes_read") is not False
        or registration.get("new_training_runs") != 0
        or registration.get("candidate", {}).get("algorithm") != "h2oplus-v15-hold-quarter"
        or authorization.get("authorization_scope", {}).get("remote_development_evaluation_authorized") is not True
        or authorization.get("authorization_scope", {}).get("remote_sumo_episode_cap") != 135
        or authorization.get("authorization_scope", {}).get("formal_held_out_access_authorized") is not False
        or authorization.get("authorization_scope", {}).get("formal_confirmation_stage_authorized") is not False
        or authorization.get("authorization_scope", {}).get("local_sumo_authorized") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("v15_development_evaluation_outcomes_read") is not False
        or protocol.get("evaluation", {}).get("new_remote_sumo_episodes") != 135
        or v14_analysis.get("formal_confirmation_eligible") is not False
        or v14_closeout.get("passed") is not True
        or v14_verification.get("passed") is not True
        or environment.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("v15_evaluation_outcomes_read") is not False
        or source_freeze.get("physical_behavior_count") != 15
        or source_freeze.get("expected_new_remote_sumo_episodes") != 135
        or len(methods) != 15
        or deployment.get("schema_version")
        != "transit-h2oplus-v15-hold-calibration-evaluation-deployment-manifest-v1"
        or deployment.get("physical_behavior_count") != 15
        or deployment.get("expected_new_remote_sumo_episodes") != 135
        or deployment.get("methods_manifest", {}).get("sha256") != args.expected_methods_sha256
        or deployment.get("source_freeze", {}).get("sha256") != args.expected_source_freeze_sha256
        or deployment.get("sumo_runtime_signature") != runtime_identity["runtime_signature"]
        or active_runtime["portable_signature"] != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise ValueError("v15 evaluation scope or immutable evidence drifted")
    return {
        "registration": {"path": str(loaded["registration"][0]), "sha256": args.expected_registration_sha256},
        "authorization": {"path": str(loaded["authorization"][0]), "sha256": args.expected_authorization_sha256},
        "environment_design": {
            "path": str(loaded["environment_design"][0]),
            "sha256": args.expected_environment_design_sha256,
        },
        "protocol": {"path": str(loaded["protocol"][0]), "sha256": args.expected_protocol_sha256},
        "v14_analysis": {"path": str(loaded["v14_analysis"][0]), "sha256": args.expected_v14_analysis_sha256},
        "v14_closeout": {"path": str(loaded["v14_closeout"][0]), "sha256": args.expected_v14_closeout_sha256},
        "v14_verification": {
            "path": str(loaded["v14_verification"][0]),
            "sha256": args.expected_v14_verification_sha256,
        },
        "source_freeze": {"path": str(loaded["source_freeze"][0]), "sha256": args.expected_source_freeze_sha256},
        "methods": {"path": str(methods_path), "sha256": args.expected_methods_sha256, "count": 15},
        "dataset": {"path": str(dataset), "sha256": args.expected_dataset_sha256},
        "deployment_manifest": {"path": str(deployment_path), "sha256": args.expected_deployment_manifest_sha256},
        "runtime_lock": runtime_identity,
        "runtime_fingerprint": active_runtime,
        "blocks": verify_design(loaded["environment_design"][0], environment),
        "parent_v14_artifact_integrity_passed": True,
        "parent_v14_training_stability_passed": True,
    }


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility/run_evaluation_matrix_v15_hold_calibration.py"),
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
        "1",
        "--shard-index",
        "0",
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


def verify_block(path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(path.read_text())
    partition = payload.get("matrix_partition", {})
    jobs = payload.get("jobs", [])
    source_hashes = payload.get("source_sha256")
    if (
        payload.get("schema_version") != "transit-evaluation-matrix-v7"
        or payload.get("methods_manifest_sha256") != args.expected_methods_sha256
        or payload.get("dataset_sha256") != args.expected_dataset_sha256
        or payload.get("sumo_runtime_lock", {}).get("sha256") != args.expected_runtime_lock_sha256
        or partition.get("shard_count") != 1
        or partition.get("shard_index") != 0
        or partition.get("full_job_count") != 45
        or partition.get("selected_job_count") != 45
        or len(jobs) != 45
        or not isinstance(source_hashes, dict)
        or len(source_hashes) != args.expected_matrix_source_file_count
    ):
        raise ValueError(f"completed v15 block manifest drifted: {path}")
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
            or result.get("source_sha256") != source_hashes
            or result.get("policy", {}).get("hold_calibration", {}).get("formal_held_out_outcomes_read") is not False
            or signature in signatures
        ):
            raise ValueError(f"v15 result binding drifted: {output}")
        signatures.add(signature)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "full_matrix_signature": partition["full_matrix_signature"],
        "selected_job_keys_sha256": partition["selected_job_keys_sha256"],
        "selected_job_count": 45,
        "source_file_count": len(source_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration",
        "authorization",
        "environment_design",
        "protocol",
        "v14_analysis",
        "v14_closeout",
        "v14_verification",
        "source_freeze",
        "methods_manifest",
        "dataset",
        "sumo_runtime_lock",
        "deployment_manifest",
        "output_root",
        "execution_json",
        "sumo_dir",
        "edge_xml",
    )
    for name in paths:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    expected = (
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
    )
    for name in expected:
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--expected-matrix-source-file-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execution_path = args.execution_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    started = utc_now()
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "running",
        "started_utc": started,
        "study_stage": "development",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "workers": args.workers,
        "expected_results": 135,
        "completed_results": 0,
        "completed_blocks": [],
    }
    atomic_json(execution_path, execution)
    try:
        if args.workers != 12:
            raise ValueError("v15 dispatch requires exactly 12 workers")
        evidence = verify_inputs(args)
        output_root.mkdir(parents=True, exist_ok=True)
        completed = []
        for block in evidence["blocks"]:
            block_root = output_root / block["name"]
            log_path = output_root / f"{block['name']}.log"
            execution["active_block"] = block["name"]
            execution["immutable_evidence"] = evidence
            atomic_json(execution_path, execution)
            command = block_command(args, block, block_root)
            with log_path.open("a") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"v15 matrix block failed: {block['name']}: {result.returncode}")
            manifest = verify_block(block_root / "matrix_manifest.json", args)
            completed.append({"block": block["name"], "manifest": manifest, "log": str(log_path)})
            execution["completed_blocks"] = completed
            execution["completed_results"] = len(completed) * 45
            execution.pop("active_block", None)
            atomic_json(execution_path, execution)
        execution.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "completed_results": 135,
                "formal_held_out_outcomes_read": False,
            }
        )
        atomic_json(execution_path, execution)
        atomic_json(
            output_root / "host_summary.json",
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "completed",
                "created_utc": utc_now(),
                "physical_evaluation_results": 135,
                "matrix_manifests": completed,
                "formal_held_out_outcomes_read": False,
                "training": False,
            },
        )
        return 0
    except Exception as exc:
        execution.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "formal_held_out_outcomes_read": False,
            }
        )
        atomic_json(execution_path, execution)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
