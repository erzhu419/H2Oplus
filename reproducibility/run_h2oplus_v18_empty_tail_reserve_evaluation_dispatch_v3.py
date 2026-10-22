#!/usr/bin/env python3
"""Run the final-boundary-corrected v18-v3 development evaluation."""

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
    run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v2 as v2,
)


EXECUTION_SCHEMA = (
    "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-execution-v3"
)
SUMMARY_SCHEMA = (
    "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-summary-v3"
)
PARENT = {
    "registration": "0a36553e568664594a7c25139f10a62f481be16b98af767b27add4bd05a0f6d7",
    "authorization": "e1f9a4b3237468e984d463ed42315e53906f95ee004f91d07086fa428713c085",
    "correction": "6fee854c13083c00c05a9b60ab9340191effd827ac460d1bf7953cb4839e3822",
    "source_freeze": "7be31254cf1433642fd6078781dc483101ca3c2baf9c53be9dd0605ff69819ee",
    "deployment_manifest": "7f6595be6c1f239e1c5c16f85aad7667ed9e87841c02c8eae9b15d0439004600",
    "v1_incident_closeout": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "v1_incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
}
PARENT_PATHS = {
    "registration": ROOT
    / "reproducibility/h2oplus_v18_empty_tail_reserve_successor_registration_v2.json",
    "authorization": ROOT
    / "reproducibility/h2oplus_v18_empty_tail_reserve_successor_authorization_v2.json",
    "correction": ROOT
    / "reproducibility/h2oplus_v18_empty_tail_reserve_parser_correction_v2.json",
    "source_freeze": ROOT
    / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v2.json",
    "deployment_manifest": ROOT
    / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_deployment_manifest_v2.json",
    "v1_incident_closeout": ROOT
    / (
        "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_incidents/v1_parser_contract/incident_closeout_v1.json"
    ),
    "v1_incident_verification": ROOT
    / (
        "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_incidents/v1_parser_contract/incident_verification_v1.json"
    ),
}
MATRIX_SOURCES = tuple(
    [*v2.MATRIX_SOURCES]
    + [
        "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py",
        "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v3.py",
    ]
)
OPERATIONAL_SOURCES = tuple(
    [*v2.OPERATIONAL_SOURCES]
    + [
        "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v3.py"
    ]
)


def _read_bound(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"v18-v3 identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"v18-v3 schema drifted: {resolved}")
    return resolved, payload


def verify_inputs(args: argparse.Namespace) -> dict:
    parent = copy.copy(args)
    for name in ("registration", "authorization", "correction", "source_freeze"):
        setattr(parent, name, PARENT_PATHS[name])
        setattr(parent, f"expected_{name}_sha256", PARENT[name])
    parent.incident_closeout = PARENT_PATHS["v1_incident_closeout"]
    parent.incident_verification = PARENT_PATHS["v1_incident_verification"]
    parent.expected_incident_closeout_sha256 = PARENT["v1_incident_closeout"]
    parent.expected_incident_verification_sha256 = PARENT[
        "v1_incident_verification"
    ]
    parent.deployment_manifest = PARENT_PATHS["deployment_manifest"]
    parent.expected_deployment_manifest_sha256 = PARENT["deployment_manifest"]
    evidence = v2._verify_inputs_v2(parent)

    controls = (
        (
            "registration",
            "transit-h2oplus-v18-empty-tail-reserve-successor-registration-v3",
        ),
        (
            "authorization",
            "transit-h2oplus-v18-empty-tail-reserve-successor-authorization-v3",
        ),
        (
            "correction",
            "transit-h2oplus-v18-empty-tail-reserve-namespace-correction-v3",
        ),
        (
            "v1_incident_closeout",
            "transit-h2oplus-v18-empty-tail-reserve-incident-closeout-v1",
        ),
        (
            "v1_incident_verification",
            "transit-h2oplus-v18-empty-tail-reserve-incident-verification-v1",
        ),
        (
            "v2_incident_closeout",
            "transit-h2oplus-v18-empty-tail-reserve-incident-closeout-v2",
        ),
        (
            "v2_incident_verification",
            "transit-h2oplus-v18-empty-tail-reserve-incident-verification-v2",
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
        raise ValueError("v18-v3 deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    matrix_sources = v2.v1._freeze_map(
        source_freeze.get("matrix_sources", []), "matrix"
    )
    operational_sources = v2.v1._freeze_map(
        source_freeze.get("operational_sources", []), "operational"
    )
    for relative, digest in {**matrix_sources, **operational_sources}.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"v18-v3 frozen source drifted: {relative}")
    registration = loaded["registration"][1]
    authorization = loaded["authorization"][1]
    correction = loaded["correction"][1]
    v1_incident = loaded["v1_incident_closeout"][1]
    v1_verification = loaded["v1_incident_verification"][1]
    v2_incident = loaded["v2_incident_closeout"][1]
    v2_verification = loaded["v2_incident_verification"][1]
    scope = authorization.get("authorization_scope", {})
    if (
        set(matrix_sources) != set(MATRIX_SOURCES)
        or set(operational_sources) != set(OPERATIONAL_SOURCES)
        or source_freeze.get("evidence_revision")
        != "v3-final-namespace-binding-correction"
        or source_freeze.get("v18_v1_result_count") != 0
        or source_freeze.get("v18_v2_result_count") != 0
        or source_freeze.get("v18_v1_results_reused") is not False
        or source_freeze.get("v18_v2_results_reused") is not False
        or source_freeze.get("v18_v3_development_evaluation_outcomes_read")
        is not False
        or source_freeze.get("expected_matrix_source_file_count") != 59
        or source_freeze.get("matrix_source_count") != 20
        or source_freeze.get("operational_source_count") != 4
        or registration.get("scientific_design_identity", {}).get(
            "unchanged_from_v18_v1_and_v2"
        )
        is not True
        or registration.get("v18_v3_development_evaluation_outcomes_read") is not False
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
                "reuse_v2_incident_results_authorized",
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
        or correction.get("final_namespace_binding_smoke_required") is not True
        or correction.get("v18_v1_or_v2_results_reused") is not False
        or v1_incident.get("completed_scientific_result_count") != 0
        or v1_incident.get("v18_v1_results_reusable") is not False
        or v1_verification.get("passed") is not True
        or v2_incident.get("completed_scientific_result_count") != 0
        or v2_incident.get("v18_v2_results_reusable") is not False
        or v2_verification.get("passed") is not True
        or deployment.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v3"
        or deployment.get("physical_behavior_count") != 15
        or deployment.get("expected_new_remote_sumo_episodes") != 135
        or deployment.get("source_freeze", {}).get("sha256")
        != args.expected_source_freeze_sha256
        or deployment.get("methods_manifest", {}).get("sha256")
        != args.expected_methods_sha256
    ):
        raise ValueError("v18-v3 successor scope or binding drifted")

    for name, (path, _) in loaded.items():
        evidence[name] = {
            "path": str(path),
            "sha256": getattr(args, f"expected_{name}_sha256"),
        }
    evidence.update(
        {
            "source_freeze": {
                "path": str(source_path),
                "sha256": args.expected_source_freeze_sha256,
            },
            "deployment_manifest": {
                "path": str(deployment_path),
                "sha256": args.expected_deployment_manifest_sha256,
            },
            "v18_v1_result_count": 0,
            "v18_v2_result_count": 0,
            "v18_v1_or_v2_results_reused": False,
            "namespace_binding_correction_only": True,
        }
    )
    return evidence


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    command = v2._block_command_v2(args, block, output)
    command[1] = str(
        ROOT / "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v3.py"
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration",
        "authorization",
        "correction",
        "v1_incident_closeout",
        "v1_incident_verification",
        "v2_incident_closeout",
        "v2_incident_verification",
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
        "v1_incident_closeout",
        "v1_incident_verification",
        "v2_incident_closeout",
        "v2_incident_verification",
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
v2._verify_inputs_v2 = v2.verify_inputs
v2._block_command_v2 = v2.block_command
v2.v1.verify_inputs = verify_inputs
v2.v1.block_command = block_command
v2.v1.parse_args = parse_args
v2.v1.EXECUTION_SCHEMA = EXECUTION_SCHEMA
v2.v1.SUMMARY_SCHEMA = SUMMARY_SCHEMA


if __name__ == "__main__":
    raise SystemExit(v2.v1.main())
