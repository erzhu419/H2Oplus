#!/usr/bin/env python3
"""Run the parser-corrected v18-v2 development evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility import (
    run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch as v1,
)


EXECUTION_SCHEMA = (
    "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-execution-v2"
)
SUMMARY_SCHEMA = (
    "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-summary-v2"
)
PARENT_REGISTRATION_SHA256 = (
    "45e160f7104a8a494d3231576c15c09ba9f28f184e3a6da4ac6637a34e3135dc"
)
PARENT_AUTHORIZATION_SHA256 = (
    "8354467e41971cc39dc99382a2327a75c358f8baa06df22e215854a5c8036f92"
)
PARENT_SOURCE_FREEZE_SHA256 = (
    "1769740ba02cb14a9ca483460c44f4651109fe9e297dec9d42409bc802df49a7"
)
PARENT_DEPLOYMENT_MANIFEST_SHA256 = (
    "b837b687951c8dc7f84b04defea90158b67fe9f5ec34ace96c204b698852d9b7"
)
PARENT_REGISTRATION = (
    ROOT / "reproducibility/h2oplus_v18_empty_tail_reserve_registration_v1.json"
)
PARENT_AUTHORIZATION = (
    ROOT / "reproducibility/h2oplus_v18_empty_tail_reserve_authorization_v1.json"
)
PARENT_SOURCE_FREEZE = (
    ROOT / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v1.json"
)
PARENT_DEPLOYMENT_MANIFEST = (
    ROOT
    / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_deployment_manifest_v1.json"
)
MATRIX_SOURCES = tuple(
    [*v1.MATRIX_SOURCES]
    + [
        "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py",
        "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v2.py",
    ]
)
OPERATIONAL_SOURCES = tuple(
    [*v1.OPERATIONAL_SOURCES]
    + [
        "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v2.py"
    ]
)


def _read_bound(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"v18-v2 identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"v18-v2 schema drifted: {resolved}")
    return resolved, payload


def verify_inputs(args: argparse.Namespace) -> dict:
    parent = copy.copy(args)
    parent.registration = PARENT_REGISTRATION
    parent.authorization = PARENT_AUTHORIZATION
    parent.source_freeze = PARENT_SOURCE_FREEZE
    parent.deployment_manifest = PARENT_DEPLOYMENT_MANIFEST
    parent.expected_registration_sha256 = PARENT_REGISTRATION_SHA256
    parent.expected_authorization_sha256 = PARENT_AUTHORIZATION_SHA256
    parent.expected_source_freeze_sha256 = PARENT_SOURCE_FREEZE_SHA256
    parent.expected_deployment_manifest_sha256 = PARENT_DEPLOYMENT_MANIFEST_SHA256
    evidence = v1._verify_inputs_v1(parent)

    controls = (
        (
            "registration",
            "transit-h2oplus-v18-empty-tail-reserve-successor-registration-v2",
        ),
        (
            "authorization",
            "transit-h2oplus-v18-empty-tail-reserve-successor-authorization-v2",
        ),
        (
            "correction",
            "transit-h2oplus-v18-empty-tail-reserve-parser-correction-v2",
        ),
        (
            "incident_closeout",
            "transit-h2oplus-v18-empty-tail-reserve-incident-closeout-v1",
        ),
        (
            "incident_verification",
            "transit-h2oplus-v18-empty-tail-reserve-incident-verification-v1",
        ),
    )
    loaded = {
        name: _read_bound(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), schema
        )
        for name, schema in controls
    }
    source_path, source_freeze = _read_bound(
        args.source_freeze,
        args.expected_source_freeze_sha256,
        "transit-h2oplus-v18-empty-tail-reserve-source-freeze-v1",
    )
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("v18-v2 deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    matrix_sources = v1._freeze_map(source_freeze.get("matrix_sources", []), "matrix")
    operational_sources = v1._freeze_map(
        source_freeze.get("operational_sources", []), "operational"
    )
    for relative, digest in {**matrix_sources, **operational_sources}.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"v18-v2 frozen source drifted: {relative}")
    registration = loaded["registration"][1]
    authorization = loaded["authorization"][1]
    correction = loaded["correction"][1]
    incident = loaded["incident_closeout"][1]
    incident_verification = loaded["incident_verification"][1]
    scope = authorization.get("authorization_scope", {})
    if (
        set(matrix_sources) != set(MATRIX_SOURCES)
        or set(operational_sources) != set(OPERATIONAL_SOURCES)
        or source_freeze.get("evidence_revision") != "v2-parser-contract-correction"
        or source_freeze.get("v18_v1_result_count") != 0
        or source_freeze.get("v18_v1_results_reused") is not False
        or source_freeze.get("v18_v2_development_evaluation_outcomes_read")
        is not False
        or source_freeze.get("expected_matrix_source_file_count") != 57
        or source_freeze.get("matrix_source_count") != 18
        or source_freeze.get("operational_source_count") != 3
        or registration.get("scientific_design_identity", {}).get(
            "unchanged_from_v18_v1"
        )
        is not True
        or registration.get("v18_v1_scientific_numeric_outcomes_read") is not False
        or registration.get("v18_v2_development_evaluation_outcomes_read") is not False
        or scope.get("remote_development_evaluation_authorized") is not True
        or scope.get("remote_host") != "jtl110gpu"
        or scope.get("remote_sumo_episode_cap") != 135
        or scope.get("unique_task_count_cap") != 1
        or any(
            scope.get(key) is not False
            for key in (
                "duplicate_submission_authorized",
                "force_dispatch_authorized",
                "reuse_v1_incident_results_authorized",
                "local_sumo_authorized",
                "local_training_authorized",
                "new_training_authorized",
                "formal_held_out_access_authorized",
                "formal_confirmation_stage_authorized",
                "move_host_authorized",
            )
        )
        or correction.get("scientific_behavior_changed") is not False
        or correction.get("methods_protocol_scenarios_or_gates_changed") is not False
        or correction.get("v18_v1_results_reused") is not False
        or correction.get("parser_stack_smoke_required") is not True
        or incident.get("completed_scientific_result_count") != 0
        or incident.get("scientific_numeric_outcomes_read") is not False
        or incident.get("v18_v1_results_reusable") is not False
        or incident_verification.get("passed") is not True
        or deployment.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v2"
        or deployment.get("physical_behavior_count") != 15
        or deployment.get("expected_new_remote_sumo_episodes") != 135
        or deployment.get("source_freeze", {}).get("sha256")
        != args.expected_source_freeze_sha256
        or deployment.get("methods_manifest", {}).get("sha256")
        != args.expected_methods_sha256
    ):
        raise ValueError("v18-v2 successor scope or binding drifted")

    evidence.update(
        {
            "registration": {
                "path": str(loaded["registration"][0]),
                "sha256": args.expected_registration_sha256,
            },
            "authorization": {
                "path": str(loaded["authorization"][0]),
                "sha256": args.expected_authorization_sha256,
            },
            "correction": {
                "path": str(loaded["correction"][0]),
                "sha256": args.expected_correction_sha256,
            },
            "incident_closeout": {
                "path": str(loaded["incident_closeout"][0]),
                "sha256": args.expected_incident_closeout_sha256,
            },
            "incident_verification": {
                "path": str(loaded["incident_verification"][0]),
                "sha256": args.expected_incident_verification_sha256,
            },
            "source_freeze": {
                "path": str(source_path),
                "sha256": args.expected_source_freeze_sha256,
            },
            "deployment_manifest": {
                "path": str(deployment_path),
                "sha256": args.expected_deployment_manifest_sha256,
            },
            "v18_v1_result_count": 0,
            "v18_v1_results_reused": False,
            "parser_contract_correction_only": True,
        }
    )
    return evidence


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    command = v1._block_command_v1(args, block, output)
    command[1] = str(
        ROOT / "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v2.py"
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration",
        "authorization",
        "correction",
        "incident_closeout",
        "incident_verification",
        "rationale",
        "environment_design",
        "protocol",
        "parent_v17_audit",
        "parent_v17_closeout",
        "parent_v17_verification",
        "data_environment_audit",
        "data_environment_verification",
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
        "rationale",
        "environment_design",
        "protocol",
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
    )
    for name in expected:
        parser.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256", required=True
        )
    parser.add_argument("--expected-matrix-source-file-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


# Retain direct handles so the wrapper cannot recurse after monkeypatching.
v1._verify_inputs_v1 = v1.verify_inputs
v1._block_command_v1 = v1.block_command
v1.verify_inputs = verify_inputs
v1.block_command = block_command
v1.parse_args = parse_args
v1.EXECUTION_SCHEMA = EXECUTION_SCHEMA
v1.SUMMARY_SCHEMA = SUMMARY_SCHEMA


if __name__ == "__main__":
    raise SystemExit(v1.main())
