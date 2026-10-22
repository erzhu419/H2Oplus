#!/usr/bin/env python3
"""Register the collision-free v18-v2 successor before any v2 outcome."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT = (
    REPRO
    / "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/v1_parser_contract"
)
AUTHORIZATION_OUTPUT = (
    REPRO / "h2oplus_v18_empty_tail_reserve_successor_authorization_v2.json"
)
CORRECTION_OUTPUT = (
    REPRO / "h2oplus_v18_empty_tail_reserve_parser_correction_v2.json"
)
REGISTRATION_OUTPUT = (
    REPRO / "h2oplus_v18_empty_tail_reserve_successor_registration_v2.json"
)
EXPECTED = {
    "v1_authorization": "8354467e41971cc39dc99382a2327a75c358f8baa06df22e215854a5c8036f92",
    "v1_registration": "45e160f7104a8a494d3231576c15c09ba9f28f184e3a6da4ac6637a34e3135dc",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "v1_source_freeze": "1769740ba02cb14a9ca483460c44f4651109fe9e297dec9d42409bc802df49a7",
    "v1_submission": "6ddda52a1619288451a0c5d0dbb8ab4052a526db6e0f17d0d37b814ca4d613b5",
    "v1_deployment_manifest": "b837b687951c8dc7f84b04defea90158b67fe9f5ec34ace96c204b698852d9b7",
    "incident_closeout": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
}
INPUTS = {
    "v1_authorization": REPRO
    / "h2oplus_v18_empty_tail_reserve_authorization_v1.json",
    "v1_registration": REPRO
    / "h2oplus_v18_empty_tail_reserve_registration_v1.json",
    "protocol": REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json",
    "methods": REPRO
    / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
    "environment": REPRO / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
    "v1_source_freeze": REPRO
    / "h2oplus_v18_empty_tail_reserve_source_freeze_v1.json",
    "v1_submission": REPRO
    / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v1.json",
    "v1_deployment_manifest": REPRO
    / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v1/workspace/deployment_manifest.json",
    "incident_closeout": INCIDENT / "incident_closeout_v1.json",
    "incident_verification": INCIDENT / "incident_verification_v1.json",
}
OLD_EVALUATOR = ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve.py"
NEW_EVALUATOR = ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py"
OLD_MATRIX = REPRO / "run_evaluation_matrix_v18_empty_tail_reserve.py"
NEW_MATRIX = REPRO / "run_evaluation_matrix_v18_empty_tail_reserve_v2.py"
OLD_DISPATCH = REPRO / "run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py"
NEW_DISPATCH = REPRO / "run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v2.py"


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


def validate_inputs() -> dict:
    for name, path in INPUTS.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v18-v2 predecessor input drifted: {name}")
    incident = json.loads(INPUTS["incident_closeout"].read_text())
    verification = json.loads(INPUTS["incident_verification"].read_text())
    if (
        incident.get("status") != "sealed-invalid-zero-result-incident"
        or incident.get("completed_scientific_result_count") != 0
        or incident.get("scientific_numeric_outcomes_read") is not False
        or incident.get("v18_v1_results_reusable") is not False
        or verification.get("passed") is not True
        or verification.get("observed_result_file_count") != 0
        or verification.get("scientific_numeric_outcomes_read") is not False
    ):
        raise ValueError("v18-v1 incident boundary drifted")
    evaluator = NEW_EVALUATOR.resolve(strict=True).read_text()
    matrix = NEW_MATRIX.resolve(strict=True).read_text()
    dispatch = NEW_DISPATCH.resolve(strict=True).read_text()
    required = (
        "--expected-empty-activation-start-fraction",
        "--expected-empty-activation-end-fraction",
        "--expected-empty-max-onboard-capacity-fraction",
    )
    if (
        any(flag not in evaluator or flag not in matrix for flag in required)
        or "v1.v17.main()" not in evaluator
        or "v1.v17.job_command(job)" not in matrix
        or "Skip only the frozen v18-v1 outer parser" not in evaluator
        or "run_evaluation_matrix_v18_empty_tail_reserve_v2.py" not in dispatch
        or "def empty_tail_reserve_action" in evaluator
        or "def load_development_methods" in matrix
    ):
        raise ValueError("v18-v2 parser-only correction contract is incomplete")
    return incident


def atomic_group(rows: list[tuple[Path, dict]]) -> None:
    for path, _ in rows:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite v18-v2 control: {path}")
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
    incident = validate_inputs()
    created = datetime.now(timezone.utc).isoformat()
    authorization = {
        "schema_version": (
            "transit-h2oplus-v18-empty-tail-reserve-successor-authorization-v2"
        ),
        "created_utc": created,
        "status": "authorized-before-any-v18-v2-development-outcome",
        "study_stage": "development",
        "request_summary": (
            "Continue improvement after sealing the zero-result v18-v1 parser incident."
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
            "local_sumo_authorized": False,
            "local_training_authorized": False,
            "new_training_authorized": False,
            "formal_held_out_access_authorized": False,
            "formal_confirmation_stage_authorized": False,
            "move_host_authorized": False,
        },
        "v18_v1_episode_accounting": {
            "completed_results": 0,
            "counted_as_v18_v2_results": 0,
            "reusable": False,
        },
        "formal_held_out_outcomes_read": False,
    }
    correction = {
        "schema_version": (
            "transit-h2oplus-v18-empty-tail-reserve-parser-correction-v2"
        ),
        "created_utc": created,
        "status": "corrected-before-any-v18-v2-development-outcome",
        "study_stage": "development",
        "scope": "nested argparse option names and fresh evidence identity only",
        "root_cause": (
            "The v18-v1 outer parser consumed three option names also required by "
            "the nested v17 parser, so all jobs failed before SUMO rollout."
        ),
        "changed_source_paths": [
            str(NEW_EVALUATOR.relative_to(ROOT)),
            str(NEW_MATRIX.relative_to(ROOT)),
            str(NEW_DISPATCH.relative_to(ROOT)),
        ],
        "renamed_v18_specific_options": {
            "--expected-activation-start-fraction": (
                "--expected-empty-activation-start-fraction"
            ),
            "--expected-activation-end-fraction": (
                "--expected-empty-activation-end-fraction"
            ),
            "--expected-max-onboard-capacity-fraction": (
                "--expected-empty-max-onboard-capacity-fraction"
            ),
        },
        "parent_v17_option_names_preserved": True,
        "parser_stack_smoke_required": True,
        "scientific_behavior_changed": False,
        "methods_protocol_scenarios_or_gates_changed": False,
        "training_or_policy_artifacts_changed": False,
        "runtime_changed": False,
        "v18_v1_results_reused": False,
        "v18_v1_scientific_numeric_outcomes_read": False,
        "formal_held_out_outcomes_read": False,
        "old_evaluator": descriptor(OLD_EVALUATOR),
        "new_evaluator": descriptor(NEW_EVALUATOR),
        "old_matrix": descriptor(OLD_MATRIX),
        "new_matrix": descriptor(NEW_MATRIX),
        "old_dispatch": descriptor(OLD_DISPATCH),
        "new_dispatch": descriptor(NEW_DISPATCH),
        "incident_closeout": descriptor(INPUTS["incident_closeout"]),
        "incident_verification": descriptor(INPUTS["incident_verification"]),
    }
    registration = {
        "schema_version": (
            "transit-h2oplus-v18-empty-tail-reserve-successor-registration-v2"
        ),
        "created_utc": created,
        "status": "registered-before-any-v18-v2-development-outcome",
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
            "unchanged_from_v18_v1": True,
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
            "candidate_selection_after_v18_v2_outcomes": False,
            "diagnostics_cannot_replace_candidate": True,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "v18_v1_results_forbidden": True,
        },
        "authorization": future_descriptor(AUTHORIZATION_OUTPUT, authorization),
        "correction": future_descriptor(CORRECTION_OUTPUT, correction),
        "v18_v1_incident": {
            "task_id": incident["task_id"],
            "closeout": descriptor(INPUTS["incident_closeout"]),
            "verification": descriptor(INPUTS["incident_verification"]),
            "completed_result_count": 0,
            "result_reuse_forbidden": True,
        },
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
        "v18_v1_scientific_numeric_outcomes_read": False,
        "v18_v2_development_evaluation_outcomes_read": False,
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
                "authorization": future_descriptor(
                    AUTHORIZATION_OUTPUT, authorization
                ),
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
