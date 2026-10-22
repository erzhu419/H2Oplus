#!/usr/bin/env python3
"""Run the fail-closed Python 3.8-compatible v17-v2 development evaluation."""

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
from reproducibility.run_evaluation_matrix_v17_tail_service_reserve import (
    METHODS_SCHEMA,
    load_development_methods,
)


EXECUTION_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-host-execution-v2"
SUMMARY_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-host-summary-v2"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
MATRIX_SOURCES = (
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "SimpleSAC/evaluate_transit_v13_proximal_release.py",
    "SimpleSAC/evaluate_transit_v14_terminal_taper.py",
    "SimpleSAC/evaluate_transit_v15_hold_calibration.py",
    "SimpleSAC/evaluate_transit_v16_headway_reserve.py",
    "SimpleSAC/evaluate_transit_v17_tail_service_reserve.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
    "reproducibility/run_evaluation_matrix_v16_headway_reserve.py",
    "reproducibility/run_evaluation_matrix_v17_tail_service_reserve.py",
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2.py",
    "reproducibility/analyze_h2oplus_v17_tail_service_reserve.py",
)
REQUIRED_ENVIRONMENT = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
        "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
        "libxcrypt-4.4.36-hd590300_1/include"
    ),
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: str, **fields: object) -> None:
    print(
        json.dumps({"event": event, "utc": utc_now(), **fields}, sort_keys=True),
        flush=True,
    )


def v17_algorithm_variant(algorithm: str) -> str:
    prefix = "h2oplus-v17-tail-service-reserve-"
    if not algorithm.startswith(prefix):
        raise ValueError(f"v17-v2 algorithm prefix drifted: {algorithm!r}")
    return algorithm[len(prefix):]


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
            raise ValueError(f"malformed v17 development block: {block!r}")
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
        raise ValueError("v17 development block identity drifted")
    return sorted(result, key=lambda row: row["name"])


def _freeze_map(rows: list[dict], label: str) -> dict[str, str]:
    result = {}
    for row in rows:
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or path in result:
            raise ValueError(f"malformed v17 {label} source freeze")
        result[path] = digest
    return result


def verify_inputs(args: argparse.Namespace) -> dict:
    environment_mismatches = {
        key: {"observed": os.environ.get(key), "expected": value}
        for key, value in REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != value
    }
    if environment_mismatches:
        raise ValueError(f"frozen v17 environment drifted: {environment_mismatches}")
    controls = (
        (
            "registration",
            "transit-h2oplus-v17-tail-service-reserve-successor-registration-v2",
        ),
        (
            "authorization",
            "transit-h2oplus-v17-tail-service-reserve-successor-authorization-v2",
        ),
        (
            "correction",
            "transit-h2oplus-v17-tail-service-reserve-python38-correction-v2",
        ),
        ("incident_closeout", "transit-h2oplus-v17-python38-incident-closeout-v1"),
        (
            "incident_verification",
            "transit-h2oplus-v17-python38-incident-verification-v1",
        ),
        (
            "data_environment_audit",
            "transit-h2oplus-v17-data-environment-convergence-audit-v1",
        ),
        (
            "data_environment_verification",
            "transit-h2oplus-v17-data-environment-convergence-audit-verification-v1",
        ),
        ("environment_design", "transit-h2oplus-v12-staged-release-environment-v2"),
        ("protocol", "transit-h2oplus-v17-tail-service-reserve-protocol-v1"),
        ("v16_analysis", "transit-h2oplus-v16-headway-reserve-analysis-v1"),
        ("v16_audit", "transit-h2oplus-v16-headway-reserve-evaluation-audit-v1"),
        ("v16_closeout", "transit-h2oplus-v16-headway-reserve-evaluation-closeout-v1"),
        (
            "v16_verification",
            "transit-h2oplus-v16-headway-reserve-evaluation-closeout-verification-v1",
        ),
        (
            "source_freeze",
            "transit-h2oplus-v17-tail-service-reserve-evaluation-source-freeze-v2",
        ),
    )
    loaded = {
        name: read_bound(getattr(args, name), getattr(args, f"expected_{name}_sha256"), schema)
        for name, schema in controls
    }
    methods_path = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("v17 methods identity drifted")
    methods = load_development_methods(methods_path, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("dataset identity drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("v17 deployment identity drifted")
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
    correction = loaded["correction"][1]
    incident_closeout = loaded["incident_closeout"][1]
    incident_verification = loaded["incident_verification"][1]
    data_environment_audit = loaded["data_environment_audit"][1]
    data_environment_verification = loaded["data_environment_verification"][1]
    environment = loaded["environment_design"][1]
    protocol = loaded["protocol"][1]
    v16_analysis = loaded["v16_analysis"][1]
    v16_audit = loaded["v16_audit"][1]
    v16_closeout = loaded["v16_closeout"][1]
    v16_verification = loaded["v16_verification"][1]
    source_freeze = loaded["source_freeze"][1]
    matrix_sources = _freeze_map(source_freeze.get("matrix_sources", []), "matrix")
    operational_sources = _freeze_map(source_freeze.get("operational_sources", []), "operational")
    if set(matrix_sources) != set(MATRIX_SOURCES) or set(operational_sources) != set(OPERATIONAL_SOURCES):
        raise ValueError("v17 scientific source set drifted")
    for relative, digest in {**matrix_sources, **operational_sources}.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"v17 scientific source drifted: {relative}")
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("v17_v2_development_evaluation_outcomes_read") is not False
        or registration.get("new_training_runs") != 0
        or registration.get(
            "prior_v14_v15_v16_development_outcomes_used_for_design"
        )
        is not True
        or registration.get("candidate", {}).get("algorithm")
        != "h2oplus-v17-tail-service-reserve-thirty"
        or registration.get("candidate", {}).get("tail_service_floor_seconds") != 30.0
        or authorization.get("authorization_scope", {}).get("remote_development_evaluation_authorized") is not True
        or authorization.get("authorization_scope", {}).get("remote_sumo_episode_cap") != 135
        or authorization.get("authorization_scope", {}).get("formal_held_out_access_authorized") is not False
        or authorization.get("authorization_scope", {}).get("formal_confirmation_stage_authorized") is not False
        or authorization.get("authorization_scope", {}).get("local_sumo_authorized") is not False
        or authorization.get("authorization_scope", {}).get("duplicate_submission_authorized") is not False
        or correction.get("scope") != "Python 3.8 compatibility and fresh evidence identity only"
        or correction.get("scientific_behavior_changed") is not False
        or correction.get("v1_incident_results_reused") is not False
        or incident_closeout.get("status") != "sealed-invalid-runtime-lineage"
        or incident_closeout.get("incident_results_reusable") is not False
        or incident_closeout.get("scientific_numeric_outcomes_inspected") is not False
        or incident_verification.get("passed") is not True
        or data_environment_audit.get("formal_held_out_outcomes_read") is not False
        or data_environment_audit.get("conclusions", {}).get("artifact_integrity") != "passed"
        or data_environment_audit.get("conclusions", {}).get("material_data_or_environment_factors_present") is not True
        or data_environment_verification.get("passed") is not True
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("v17_development_evaluation_outcomes_read") is not False
        or protocol.get("evaluation", {}).get("new_remote_sumo_episodes") != 135
        or v16_analysis.get("formal_confirmation_eligible") is not False
        or v16_analysis.get("artifact_integrity", {}).get("passed") is not True
        or v16_analysis.get("parent_training_scientific_stability", {}).get("passed") is not True
        or v16_audit.get("passed") is not True
        or v16_closeout.get("passed") is not True
        or v16_closeout.get("artifact_integrity", {}).get("passed") is not True
        or v16_verification.get("passed") is not True
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (v16_analysis, v16_audit, v16_closeout, v16_verification)
        )
        or environment.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("v17_v2_evaluation_outcomes_read") is not False
        or source_freeze.get(
            "prior_v14_v15_v16_development_outcomes_used_for_design"
        )
        is not True
        or source_freeze.get("physical_behavior_count") != 15
        or source_freeze.get("expected_new_remote_sumo_episodes") != 135
        or source_freeze.get("expected_matrix_source_file_count") != 53
        or source_freeze.get("matrix_source_count") != 14
        or source_freeze.get("operational_source_count") != 2
        or len(methods) != 15
        or deployment.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-evaluation-deployment-manifest-v2"
        or deployment.get("physical_behavior_count") != 15
        or deployment.get("expected_new_remote_sumo_episodes") != 135
        or deployment.get("methods_manifest", {}).get("sha256") != args.expected_methods_sha256
        or deployment.get("source_freeze", {}).get("sha256") != args.expected_source_freeze_sha256
        or deployment.get("sumo_runtime_signature") != runtime_identity["runtime_signature"]
        or active_runtime["portable_signature"] != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise ValueError("v17 evaluation scope or immutable evidence drifted")
    return {
        "registration": {"path": str(loaded["registration"][0]), "sha256": args.expected_registration_sha256},
        "authorization": {"path": str(loaded["authorization"][0]), "sha256": args.expected_authorization_sha256},
        "correction": {"path": str(loaded["correction"][0]), "sha256": args.expected_correction_sha256},
        "incident_closeout": {
            "path": str(loaded["incident_closeout"][0]),
            "sha256": args.expected_incident_closeout_sha256,
        },
        "incident_verification": {
            "path": str(loaded["incident_verification"][0]),
            "sha256": args.expected_incident_verification_sha256,
        },
        "data_environment_audit": {
            "path": str(loaded["data_environment_audit"][0]),
            "sha256": args.expected_data_environment_audit_sha256,
        },
        "data_environment_verification": {
            "path": str(loaded["data_environment_verification"][0]),
            "sha256": args.expected_data_environment_verification_sha256,
        },
        "environment_design": {
            "path": str(loaded["environment_design"][0]),
            "sha256": args.expected_environment_design_sha256,
        },
        "protocol": {"path": str(loaded["protocol"][0]), "sha256": args.expected_protocol_sha256},
        "v16_analysis": {"path": str(loaded["v16_analysis"][0]), "sha256": args.expected_v16_analysis_sha256},
        "v16_audit": {"path": str(loaded["v16_audit"][0]), "sha256": args.expected_v16_audit_sha256},
        "v16_closeout": {"path": str(loaded["v16_closeout"][0]), "sha256": args.expected_v16_closeout_sha256},
        "v16_verification": {
            "path": str(loaded["v16_verification"][0]),
            "sha256": args.expected_v16_verification_sha256,
        },
        "source_freeze": {"path": str(loaded["source_freeze"][0]), "sha256": args.expected_source_freeze_sha256},
        "methods": {"path": str(methods_path), "sha256": args.expected_methods_sha256, "count": 15},
        "dataset": {"path": str(dataset), "sha256": args.expected_dataset_sha256},
        "deployment_manifest": {"path": str(deployment_path), "sha256": args.expected_deployment_manifest_sha256},
        "runtime_lock": runtime_identity,
        "runtime_fingerprint": active_runtime,
        "required_environment": dict(REQUIRED_ENVIRONMENT),
        "blocks": verify_design(loaded["environment_design"][0], environment),
        "parent_v16_artifact_integrity_passed": True,
        "parent_v14_training_stability_passed": True,
    }


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility/run_evaluation_matrix_v17_tail_service_reserve.py"),
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
        raise ValueError(f"completed v17 block manifest drifted: {path}")
    source_freeze = json.loads(args.source_freeze.expanduser().resolve(strict=True).read_text())
    for source in source_freeze.get("matrix_sources", []):
        relative = str(source.get("path", ""))
        digest = source.get("sha256")
        if not any(str(source_path).endswith(relative) and value == digest for source_path, value in source_hashes.items()):
            raise ValueError(f"v17 matrix source binding missing: {relative}")
    signatures = set()
    for job in jobs:
        output = Path(job["output"])
        if not output.is_file():
            raise FileNotFoundError(output)
        result = json.loads(output.read_text())
        signature = job["signature"]
        method = job.get("method", {})
        reserve = result.get("policy", {}).get("tail_service_reserve", {})
        algorithm = str(method.get("algorithm", ""))
        variant = v17_algorithm_variant(algorithm)
        if (
            result.get("schema_version") != "transit-evaluation-v7"
            or result.get("evaluation_job", {}).get("signature") != signature
            or result.get("source_sha256") != source_hashes
            or result.get("method") != method.get("name")
            or reserve.get("schema_version") != "transit-h2oplus-v17-tail-service-reserve-v1"
            or reserve.get("variant") != variant
            or reserve.get("tail_service_floor_seconds")
            != method.get("tail_service_floor_seconds")
            or reserve.get("activation_start_fraction")
            != method.get("activation_start_fraction")
            or reserve.get("activation_end_fraction")
            != method.get("activation_end_fraction")
            or reserve.get("max_onboard_capacity_fraction")
            != method.get("max_onboard_capacity_fraction")
            or reserve.get("parent_base_hold_scale")
            != method.get("parent_base_hold_scale")
            or reserve.get("parent_headway_balance_gain")
            != method.get("parent_headway_balance_gain")
            or reserve.get("parent_reserve_horizon_fraction")
            != method.get("parent_reserve_horizon_fraction")
            or reserve.get("requires_forward_bus_present") is not True
            or reserve.get("requires_backward_bus_absent") is not True
            or reserve.get("requires_positive_eligible_arrival_rate") is not True
            or reserve.get("v17_protocol_sha256") != args.expected_protocol_sha256
            or reserve.get("parent_action_preserved_outside_guard") is not True
            or reserve.get("guarded_hold_may_exceed_parent") is not True
            or reserve.get("physical_max_hold_preserved") is not True
            or reserve.get("predesignated_candidate")
            is not (algorithm == "h2oplus-v17-tail-service-reserve-thirty")
            or reserve.get("checkpoint_selection_by_behavior") is not False
            or reserve.get("formal_held_out_outcomes_read") is not False
            or signature in signatures
        ):
            raise ValueError(f"v17 result binding drifted: {output}")
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
        "correction",
        "incident_closeout",
        "incident_verification",
        "data_environment_audit",
        "data_environment_verification",
        "environment_design",
        "protocol",
        "v16_analysis",
        "v16_audit",
        "v16_closeout",
        "v16_verification",
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
        "correction",
        "incident_closeout",
        "incident_verification",
        "data_environment_audit",
        "data_environment_verification",
        "environment_design",
        "protocol",
        "v16_analysis",
        "v16_audit",
        "v16_closeout",
        "v16_verification",
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
    emit("v17_dispatch_started", workers=args.workers, expected_results=135)
    try:
        if args.workers != 12:
            raise ValueError("v17 dispatch requires exactly 12 workers")
        evidence = verify_inputs(args)
        emit(
            "v17_inputs_verified",
            blocks=len(evidence["blocks"]),
            methods=evidence["methods"]["count"],
        )
        output_root.mkdir(parents=True, exist_ok=True)
        completed = []
        for block in evidence["blocks"]:
            block_root = output_root / block["name"]
            log_path = output_root / f"{block['name']}.log"
            execution["active_block"] = block["name"]
            execution["immutable_evidence"] = evidence
            atomic_json(execution_path, execution)
            command = block_command(args, block, block_root)
            emit("v17_block_started", block=block["name"], completed_results=len(completed) * 45)
            with log_path.open("a") as log:
                log.write(
                    json.dumps(
                        {"event": "v17_matrix_started", "utc": utc_now(), "block": block["name"]},
                        sort_keys=True,
                    )
                    + "\n"
                )
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"v17 matrix block failed: {block['name']}: {result.returncode}")
            manifest = verify_block(block_root / "matrix_manifest.json", args)
            completed.append({"block": block["name"], "manifest": manifest, "log": str(log_path)})
            execution["completed_blocks"] = completed
            execution["completed_results"] = len(completed) * 45
            execution.pop("active_block", None)
            atomic_json(execution_path, execution)
            emit(
                "v17_block_completed",
                block=block["name"],
                completed_results=execution["completed_results"],
            )
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
        emit("v17_dispatch_completed", completed_results=135)
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
        emit("v17_dispatch_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
