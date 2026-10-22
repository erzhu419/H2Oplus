#!/usr/bin/env python3
"""Register the final-boundary-corrected v18-v3 successor."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT_V1 = REPRO / (
    "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
    "v1_parser_contract"
)
INCIDENT_V2 = REPRO / (
    "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
    "v2_namespace_binding"
)
AUTHORIZATION_OUTPUT = (
    REPRO / "h2oplus_v18_empty_tail_reserve_successor_authorization_v3.json"
)
CORRECTION_OUTPUT = (
    REPRO / "h2oplus_v18_empty_tail_reserve_namespace_binding_correction_v3.json"
)
REGISTRATION_OUTPUT = (
    REPRO / "h2oplus_v18_empty_tail_reserve_successor_registration_v3.json"
)
EXPECTED = {
    "v2_authorization": "e1f9a4b3237468e984d463ed42315e53906f95ee004f91d07086fa428713c085",
    "v2_correction": "6fee854c13083c00c05a9b60ab9340191effd827ac460d1bf7953cb4839e3822",
    "v2_registration": "0a36553e568664594a7c25139f10a62f481be16b98af767b27add4bd05a0f6d7",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "v2_source_freeze": "7be31254cf1433642fd6078781dc483101ca3c2baf9c53be9dd0605ff69819ee",
    "v2_submission": "21b8ddd509fb2b61a7d7127398758c15d628218e18f5ac0cd9708a6ae809ea3c",
    "v2_deployment_manifest": "7f6595be6c1f239e1c5c16f85aad7667ed9e87841c02c8eae9b15d0439004600",
    "v1_incident_closeout": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "v1_incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
    "v2_incident_closeout": "c2cf9e59248887be24df96e53dcd4cb09a7f5c9fb5e5a8e8fd7746c6cf6d3568",
    "v2_incident_verification": "ee32840d6de1e10a7b3c9a622c1ff50ab33d63234b3cb3616d37af18a256f9b8",
}
INPUTS = {
    "v2_authorization": REPRO
    / "h2oplus_v18_empty_tail_reserve_successor_authorization_v2.json",
    "v2_correction": REPRO
    / "h2oplus_v18_empty_tail_reserve_parser_correction_v2.json",
    "v2_registration": REPRO
    / "h2oplus_v18_empty_tail_reserve_successor_registration_v2.json",
    "protocol": REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json",
    "methods": REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
    "environment": REPRO / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
    "v2_source_freeze": REPRO
    / "h2oplus_v18_empty_tail_reserve_source_freeze_v2.json",
    "v2_submission": REPRO
    / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v2.json",
    "v2_deployment_manifest": REPRO
    / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v2/workspace/deployment_manifest.json",
    "v1_incident_closeout": INCIDENT_V1 / "incident_closeout_v1.json",
    "v1_incident_verification": INCIDENT_V1 / "incident_verification_v1.json",
    "v2_incident_closeout": INCIDENT_V2 / "incident_closeout_v2.json",
    "v2_incident_verification": INCIDENT_V2 / "incident_verification_v2.json",
}
OLD_EVALUATOR = ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py"
NEW_EVALUATOR = ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py"
OLD_MATRIX = REPRO / "run_evaluation_matrix_v18_empty_tail_reserve_v2.py"
NEW_MATRIX = REPRO / "run_evaluation_matrix_v18_empty_tail_reserve_v3.py"
OLD_DISPATCH = REPRO / "run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v2.py"
NEW_DISPATCH = REPRO / "run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v3.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def future_descriptor(path: Path, payload: dict) -> dict:
    content = encode(payload)
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def validate_inputs() -> tuple[dict, dict]:
    for name, path in INPUTS.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v18-v3 predecessor input drifted: {name}")
    incident_v1 = json.loads(INPUTS["v1_incident_closeout"].read_text())
    incident_v2 = json.loads(INPUTS["v2_incident_closeout"].read_text())
    verification_v1 = json.loads(INPUTS["v1_incident_verification"].read_text())
    verification_v2 = json.loads(INPUTS["v2_incident_verification"].read_text())
    if (
        incident_v1.get("completed_scientific_result_count") != 0
        or incident_v1.get("v18_v1_results_reusable") is not False
        or verification_v1.get("passed") is not True
        or incident_v2.get("completed_scientific_result_count") != 0
        or incident_v2.get("v18_v2_results_reusable") is not False
        or verification_v2.get("passed") is not True
        or any(
            item.get("scientific_numeric_outcomes_read") is not False
            for item in (incident_v1, incident_v2, verification_v1, verification_v2)
        )
    ):
        raise ValueError("v18 predecessor incident boundary drifted")
    evaluator = NEW_EVALUATOR.resolve(strict=True).read_text()
    matrix = NEW_MATRIX.resolve(strict=True).read_text()
    dispatch = NEW_DISPATCH.resolve(strict=True).read_text()
    if (
        "_bind_empty_tail_expectations" not in evaluator
        or "Parent parsers run first" not in evaluator
        or "v1.build_development_policy" not in evaluator
        or "evaluate_transit_v18_empty_tail_reserve_v3.py" not in matrix
        or "run_evaluation_matrix_v18_empty_tail_reserve_v3.py" not in dispatch
        or "def empty_tail_reserve_action" in evaluator
        or "def load_development_methods" in matrix
    ):
        raise ValueError("v18-v3 final-binding correction contract is incomplete")
    return incident_v1, incident_v2


def atomic_group(rows: list[tuple[Path, dict]]) -> None:
    for path, _ in rows:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite v18-v3 control: {path}")
    partials = []
    try:
        for path, payload in rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_bytes(encode(payload))
            partials.append((partial, path))
        for partial, path in partials:
            os.replace(partial, path)
    finally:
        for partial, _ in partials:
            partial.unlink(missing_ok=True)


def main() -> int:
    incident_v1, incident_v2 = validate_inputs()
    created = datetime.now(timezone.utc).isoformat()
    authorization = {
        "schema_version": (
            "transit-h2oplus-v18-empty-tail-reserve-successor-authorization-v3"
        ),
        "created_utc": created,
        "status": "authorized-before-any-v18-v3-development-outcome",
        "study_stage": "development",
        "request_summary": (
            "Continue improvement after sealing the zero-result v18-v2 namespace incident."
        ),
        "authorization_scope": {
            "remote_development_evaluation_authorized": True,
            "remote_host": "jtl110gpu",
            "remote_sumo_episode_cap": 135,
            "unique_task_count_cap": 1,
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
            "duplicate_submission_authorized": False,
            "force_dispatch_authorized": False,
            "reuse_v1_incident_results_authorized": False,
            "reuse_v2_incident_results_authorized": False,
            "local_sumo_authorized": False,
            "local_training_authorized": False,
            "new_training_authorized": False,
            "formal_held_out_access_authorized": False,
            "formal_confirmation_stage_authorized": False,
            "move_host_authorized": False,
        },
        "predecessor_episode_accounting": {
            "v18_v1_completed_results": 0,
            "v18_v2_completed_results": 0,
            "counted_as_v18_v3_results": 0,
            "reusable": False,
        },
        "formal_held_out_outcomes_read": False,
    }
    correction = {
        "schema_version": (
            "transit-h2oplus-v18-empty-tail-reserve-namespace-correction-v3"
        ),
        "created_utc": created,
        "status": "corrected-before-any-v18-v3-development-outcome",
        "study_stage": "development",
        "scope": "final namespace rebinding and fresh evidence identity only",
        "root_cause": incident_v2["root_cause"]["summary"],
        "changed_source_paths": [
            str(NEW_EVALUATOR.relative_to(ROOT)),
            str(NEW_MATRIX.relative_to(ROOT)),
            str(NEW_DISPATCH.relative_to(ROOT)),
        ],
        "binding_change": (
            "The already unique v18 empty-tail expectation values are copied into "
            "the legacy frozen-validator fields only after every parent parser has run."
        ),
        "parent_v17_option_names_preserved": True,
        "final_namespace_binding_smoke_required": True,
        "scientific_behavior_changed": False,
        "methods_protocol_scenarios_or_gates_changed": False,
        "training_or_policy_artifacts_changed": False,
        "runtime_changed": False,
        "v18_v1_or_v2_results_reused": False,
        "v18_v1_or_v2_scientific_numeric_outcomes_read": False,
        "formal_held_out_outcomes_read": False,
        "old_evaluator": descriptor(OLD_EVALUATOR),
        "new_evaluator": descriptor(NEW_EVALUATOR),
        "old_matrix": descriptor(OLD_MATRIX),
        "new_matrix": descriptor(NEW_MATRIX),
        "old_dispatch": descriptor(OLD_DISPATCH),
        "new_dispatch": descriptor(NEW_DISPATCH),
        "v1_incident_closeout": descriptor(INPUTS["v1_incident_closeout"]),
        "v1_incident_verification": descriptor(INPUTS["v1_incident_verification"]),
        "v2_incident_closeout": descriptor(INPUTS["v2_incident_closeout"]),
        "v2_incident_verification": descriptor(INPUTS["v2_incident_verification"]),
    }
    registration = {
        "schema_version": (
            "transit-h2oplus-v18-empty-tail-reserve-successor-registration-v3"
        ),
        "created_utc": created,
        "status": "registered-before-any-v18-v3-development-outcome",
        "study_stage": "development",
        "candidate": {
            "algorithm": "h2oplus-v18-empty-tail-reserve-empty-forty",
            "empty_reserve_floor_seconds": 40.0,
            "max_onboard_capacity_fraction": 0.0,
        },
        "diagnostics_not_eligible_for_selection": [
            "h2oplus-v18-empty-tail-reserve-baseline",
            "h2oplus-v18-empty-tail-reserve-one-seat-forty",
        ],
        "scientific_design_identity": {
            "unchanged_from_v18_v1_and_v2": True,
            "protocol": descriptor(INPUTS["protocol"]),
            "methods": descriptor(INPUTS["methods"]),
            "environment": descriptor(INPUTS["environment"]),
            "physical_behavior_count": 15,
            "development_blocks": 3,
            "jobs_per_block": 45,
            "new_remote_sumo_episodes": 135,
        },
        "operational_successor_identity": {
            "fresh_registration": True,
            "fresh_source_freeze_required": True,
            "fresh_deployment_required": True,
            "fresh_output_root_required": True,
            "unique_submission_required": True,
        },
        "selection_rules": {
            "candidate_selection_after_v18_v3_outcomes": False,
            "diagnostics_cannot_replace_candidate": True,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "v18_v1_or_v2_results_forbidden": True,
        },
        "authorization": future_descriptor(AUTHORIZATION_OUTPUT, authorization),
        "correction": future_descriptor(CORRECTION_OUTPUT, correction),
        "incidents": {
            "v1": {
                "task_id": incident_v1["task_id"],
                "closeout": descriptor(INPUTS["v1_incident_closeout"]),
                "verification": descriptor(INPUTS["v1_incident_verification"]),
                "completed_result_count": 0,
            },
            "v2": {
                "task_id": incident_v2["task_id"],
                "closeout": descriptor(INPUTS["v2_incident_closeout"]),
                "verification": descriptor(INPUTS["v2_incident_verification"]),
                "completed_result_count": 0,
            },
            "result_reuse_forbidden": True,
        },
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
        "v18_v1_or_v2_scientific_numeric_outcomes_read": False,
        "v18_v3_development_evaluation_outcomes_read": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
    }
    atomic_group(
        [
            (AUTHORIZATION_OUTPUT, authorization),
            (CORRECTION_OUTPUT, correction),
            (REGISTRATION_OUTPUT, registration),
        ]
    )
    print(
        json.dumps(
            {
                "authorization": future_descriptor(AUTHORIZATION_OUTPUT, authorization),
                "correction": future_descriptor(CORRECTION_OUTPUT, correction),
                "registration": future_descriptor(REGISTRATION_OUTPUT, registration),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
