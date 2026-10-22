#!/usr/bin/env python3
"""Preregister the v17 guarded tail-service completion-safety screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
V16_RESULTS = REPRO / "h2oplus_v16_headway_reserve_results"
SEEDS = tuple(range(6201, 6206))
VARIANTS = ("twenty", "thirty", "forty")
TAIL_SERVICE_FLOORS = {"twenty": 20.0, "thirty": 30.0, "forty": 40.0}
PARENT_BASE_HOLD_SCALE = 0.125
PARENT_HEADWAY_BALANCE_GAIN = 1.0
PARENT_RESERVE_HORIZON_FRACTION = 0.75
ACTIVATION_START_FRACTION = 0.5
ACTIVATION_END_FRACTION = 0.75
MAX_ONBOARD_CAPACITY_FRACTION = 0.25
EXPECTED = {
    "authorization": "6871d458b23bbc796aa8f0a981eb21f865f6a5a0d61c3c427173573e44261e4a",
    "design_evidence": "5bc0e0366fa19ef9597db20de702d554bb1dc05ddc6ec7ddb1fdedc72818b24e",
    "v16_methods": "35047fd2ce616acbccd4592ae333a7ba84a1e4231c1a2505c7768d96b91de06a",
    "v16_protocol": "978faf1b8ddf51128b5fac570552e8388c3d882e39244c4b6f0ec7615e79a43b",
    "v16_source_freeze": "606a15bf7db8b9aa1c4dbefa7b9bcc12bc94ca60e605de5a3e185eefec679577",
    "v16_audit": "ed5e58e52f1ea50d7195a10b313bd9c36e88b618fa802de2693ffdfe133112c9",
    "v16_analysis": "4a88f48946c2ec478ca47274df39e0ccdf3535c1c21ad41141142d529617f75e",
    "v16_closeout": "ef7d456e6c71bb2fbcbd176642c01acf123f8e709cefffd2b8d4ae26a850ddf5",
    "v16_verification": "54b25d3c9307e9f926319b7f9ae543e6747d4639ce9b8af762e123bade9f4c85",
}
METHODS_SELECTION_RULE = (
    "sealed v16 headway-reserve-three-quarter 25k parents and exact seed-matched "
    "actors fixed before v17; the tail-only low-load positive-arrival guard, "
    "candidate 30-second floor and diagnostic 20/40-second floors fixed before any "
    "v17 outcome; physical behavior identity is seed plus actor tensor SHA256 plus "
    "the complete inherited v16 transform and guarded v17 floor"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, owner: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": os.path.relpath(resolved, owner.parent) if owner else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_bound(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"frozen identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"frozen schema drifted: {resolved}")
    return resolved, payload


def write_once(path: Path, payload: dict) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen preregistration: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_authorization_v1.json",
    )
    parser.add_argument(
        "--design-evidence",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_offline_design_evidence_v1.json",
    )
    parser.add_argument(
        "--v16-methods",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--v16-protocol",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_protocol_v1.json",
    )
    parser.add_argument(
        "--v16-source-freeze",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
    )
    parser.add_argument(
        "--v16-audit",
        type=Path,
        default=V16_RESULTS / "evaluation_audit_v1.json",
    )
    parser.add_argument(
        "--v16-analysis",
        type=Path,
        default=V16_RESULTS / "headway_reserve_analysis_v1.json",
    )
    parser.add_argument(
        "--v16-closeout",
        type=Path,
        default=V16_RESULTS / "evaluation_closeout_v1.json",
    )
    parser.add_argument(
        "--v16-verification",
        type=Path,
        default=V16_RESULTS / "evaluation_closeout_verification_v1.json",
    )
    parser.add_argument(
        "--rationale-output",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_rationale_v1.json",
    )
    parser.add_argument(
        "--registration-output",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_registration_v1.json",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_protocol_v1.json",
    )
    parser.add_argument(
        "--methods-output",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    authorization_path, authorization = load_bound(
        args.authorization,
        EXPECTED["authorization"],
        "transit-h2oplus-v17-tail-service-reserve-authorization-v1",
    )
    design_evidence_path, design_evidence = load_bound(
        args.design_evidence,
        EXPECTED["design_evidence"],
        "transit-h2oplus-v17-tail-service-reserve-offline-design-evidence-v1",
    )
    v16_methods_path, v16_methods = load_bound(
        args.v16_methods,
        EXPECTED["v16_methods"],
        "transit-evaluation-methods-v16-headway-reserve-v1",
    )
    v16_protocol_path, v16_protocol = load_bound(
        args.v16_protocol,
        EXPECTED["v16_protocol"],
        "transit-h2oplus-v16-headway-reserve-protocol-v1",
    )
    v16_source_freeze_path, v16_source_freeze = load_bound(
        args.v16_source_freeze,
        EXPECTED["v16_source_freeze"],
        "transit-h2oplus-v16-headway-reserve-evaluation-source-freeze-v1",
    )
    v16_audit_path, v16_audit = load_bound(
        args.v16_audit,
        EXPECTED["v16_audit"],
        "transit-h2oplus-v16-headway-reserve-evaluation-audit-v1",
    )
    v16_analysis_path, v16_analysis = load_bound(
        args.v16_analysis,
        EXPECTED["v16_analysis"],
        "transit-h2oplus-v16-headway-reserve-analysis-v1",
    )
    v16_closeout_path, v16_closeout = load_bound(
        args.v16_closeout,
        EXPECTED["v16_closeout"],
        "transit-h2oplus-v16-headway-reserve-evaluation-closeout-v1",
    )
    v16_verification_path, v16_verification = load_bound(
        args.v16_verification,
        EXPECTED["v16_verification"],
        "transit-h2oplus-v16-headway-reserve-evaluation-closeout-verification-v1",
    )
    checks = v16_analysis.get("development_scientific_result", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if value is False)
    expected_failed = ["completion_rate_noninferiority_against_all_controls"]
    if (
        failed != expected_failed
        or v16_analysis.get("formal_confirmation_eligible") is not False
        or v16_analysis.get("formal_held_out_outcomes_read") is not False
        or v16_audit.get("passed") is not True
        or v16_closeout.get("passed") is not True
        or v16_verification.get("passed") is not True
        or v16_source_freeze.get("v16_evaluation_outcomes_read") is not False
        or authorization.get("authorization_scope", {}).get(
            "remote_development_evaluation_authorized"
        )
        is not True
        or authorization.get("authorization_scope", {}).get("remote_sumo_episode_cap")
        != 135
        or authorization.get("authorization_scope", {}).get("new_training_authorized")
        is not False
        or design_evidence.get("candidate_selection", {}).get(
            "candidate_floor_seconds"
        )
        != 30.0
        or design_evidence.get("candidate_selection", {}).get(
            "diagnostic_floor_seconds"
        )
        != [20.0, 40.0]
        or design_evidence.get("actor_forward_pass", {}).get("sumo_executed")
        is not False
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (
                authorization,
                design_evidence,
                v16_methods,
                v16_protocol,
                v16_source_freeze,
                v16_audit,
                v16_analysis,
                v16_closeout,
                v16_verification,
            )
        )
    ):
        raise ValueError("v16 development closeout or failure diagnosis drifted")

    comparisons = v16_analysis["development_scientific_result"]["control_comparisons"]
    completion = v16_analysis["development_scientific_result"][
        "completion_aulc_against_controls"
    ]
    observed = {
        "candidate_aulc": v16_analysis["development_scientific_result"][
            "all_algorithm_mean_aulc"
        ]["h2oplus-v16-headway-reserve-three-quarter"],
        "candidate_endpoint": v16_analysis["development_scientific_result"][
            "mean_primary_at_25000"
        ]["h2oplus-v16-headway-reserve-three-quarter"],
        "candidate_completion_aulc": completion["h2oplus"]["candidate_aulc"],
        "relative_vs_v11": comparisons["h2oplus-v11-actor-hold"][
            "relative_improvement"
        ],
        "relative_vs_v12": comparisons["h2oplus-v12-actor-ramp"][
            "relative_improvement"
        ],
        "completion_delta_vs_h2oplus": completion["h2oplus"]["delta"],
        "completion_delta_vs_sac": completion["sac-online"]["delta"],
        "completion_delta_vs_v10": completion["h2oplus-v10-delayed-actor"]["delta"],
    }
    expected_observed = {
        "candidate_aulc": 2435.4733301441265,
        "candidate_endpoint": 2408.7944811656685,
        "candidate_completion_aulc": 0.9642396602171646,
        "relative_vs_v11": 0.010208638349098213,
        "relative_vs_v12": 0.01228893534282494,
        "completion_delta_vs_h2oplus": -0.00408723947011802,
        "completion_delta_vs_sac": -0.003995643385337866,
        "completion_delta_vs_v10": -0.0032797706096310275,
    }
    if any(
        not math.isclose(observed[key], value, rel_tol=0.0, abs_tol=1e-12)
        for key, value in expected_observed.items()
    ):
        raise ValueError("v16 development signal used for v17 design drifted")

    outputs = (
        args.rationale_output,
        args.registration_output,
        args.protocol_output,
        args.methods_output,
    )
    existing = [
        str(path.expanduser().resolve())
        for path in outputs
        if path.expanduser().resolve().exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite v17 preregistration outputs: {existing}")

    now = datetime.now(timezone.utc).isoformat()
    frozen_v16 = {
        "methods": descriptor(v16_methods_path),
        "protocol": descriptor(v16_protocol_path),
        "source_freeze": descriptor(v16_source_freeze_path),
        "evaluation_audit": descriptor(v16_audit_path),
        "development_analysis": descriptor(v16_analysis_path),
        "evaluation_closeout": descriptor(v16_closeout_path),
        "evaluation_closeout_verification": descriptor(v16_verification_path),
    }
    v17_design_inputs = {
        "authorization": descriptor(authorization_path),
        "offline_design_evidence": descriptor(design_evidence_path),
    }
    rationale = {
        "schema_version": "transit-h2oplus-v17-tail-service-reserve-rationale-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "frozen-before-any-v17-outcome",
        "formal_held_out_outcomes_read": False,
        "v17_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_development_outcomes_used_for_design": True,
        "v16_failed_preregistered_gates": failed,
        "v16_observed_design_signal": observed,
        "diagnosis": {
            "single_failed_boundary": (
                "the sealed v16 candidate passed endpoint, AULC, seed-win and "
                "demand-scale gates but missed only completion-rate noninferiority"
            ),
            "intervention": (
                "preserve the complete v16 three-quarter policy except on low-load tail "
                "buses with a forward bus, no backward bus and positive eligible arrival "
                "rate during the fixed 0.5-0.75 horizon interval"
            ),
            "mechanistic_boundary": (
                "inside the guard, raise the inherited physical hold to a fixed floor; "
                "outside the guard, preserve the sealed v16 action bit-for-bit"
            ),
        },
        "fixed_design": {
            "parent": (
                "sealed v16 headway-reserve-three-quarter 25k behaviors, exact seeds "
                "6201-6205"
            ),
            "predesignated_candidate": "tail-service-reserve-thirty",
            "candidate_floor_seconds": 30.0,
            "diagnostic_floor_seconds": [20.0, 40.0],
            "activation_start_fraction": ACTIVATION_START_FRACTION,
            "activation_end_fraction": ACTIVATION_END_FRACTION,
            "max_onboard_capacity_fraction": MAX_ONBOARD_CAPACITY_FRACTION,
            "requires_forward_bus_present": True,
            "requires_backward_bus_absent": True,
            "requires_positive_eligible_arrival_rate": True,
            "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
            "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
            "parent_reserve_horizon_fraction": PARENT_RESERVE_HORIZON_FRACTION,
            "tail_service_formula": (
                "if guard: min(60, max(v16_parent_hold, registered_floor_seconds)); "
                "otherwise: v16_parent_hold"
            ),
            "checkpoint_selection_by_behavior": False,
            "new_training_runs": 0,
            "new_remote_development_episodes": 135,
        },
        "v17_design_inputs": v17_design_inputs,
        "frozen_v16_evidence": frozen_v16,
    }
    write_once(args.rationale_output, rationale)
    rationale_path = args.rationale_output.expanduser().resolve(strict=True)

    registration = {
        "schema_version": "transit-h2oplus-v17-tail-service-reserve-registration-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "registered-before-any-v17-outcome",
        "formal_held_out_outcomes_read": False,
        "v17_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_development_outcomes_used_for_design": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "candidate": {
            "algorithm": "h2oplus-v17-tail-service-reserve-thirty",
            "tail_service_floor_seconds": 30.0,
        },
        "diagnostics_not_eligible_for_selection": [
            {
                "algorithm": "h2oplus-v17-tail-service-reserve-twenty",
                "tail_service_floor_seconds": 20.0,
            },
            {
                "algorithm": "h2oplus-v17-tail-service-reserve-forty",
                "tail_service_floor_seconds": 40.0,
            },
        ],
        "selection_rules": {
            "checkpoint_selection_by_behavior": False,
            "candidate_selection_after_v17_outcomes": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "expand_only_through_frozen_v14_prefix_aliases": True,
            "diagnostics_cannot_replace_candidate": True,
            "physical_behavior_identity": [
                "training_seed",
                "actor_tensor_sha256",
                "parent_base_hold_scale",
                "parent_headway_balance_gain",
                "parent_reserve_horizon_fraction",
                "tail_service_floor_seconds",
                "activation_start_fraction",
                "activation_end_fraction",
                "max_onboard_capacity_fraction",
            ],
        },
        "episode_accounting": {
            "physical_methods": 15,
            "development_cells_per_method": 9,
            "new_remote_sumo_episodes": 135,
            "reused_frozen_v14_control_episodes": 630,
            "duplicate_control_episodes_allowed": False,
        },
        "rationale": descriptor(rationale_path),
        "v17_design_inputs": v17_design_inputs,
        "frozen_v16_evidence": frozen_v16,
    }
    write_once(args.registration_output, registration)
    registration_path = args.registration_output.expanduser().resolve(strict=True)

    analysis_gates = dict(v16_protocol["analysis_gates"])
    analysis_gates["predesignated_candidate"] = "tail-service-reserve-thirty"
    protocol = {
        "schema_version": "transit-h2oplus-v17-tail-service-reserve-protocol-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "frozen-before-any-v17-outcome",
        "formal_held_out_outcomes_read": False,
        "v17_development_evaluation_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration_path),
        "rationale": descriptor(rationale_path),
        "parent_v16_protocol": descriptor(v16_protocol_path),
        "tail_service_reserve": {
            "parent_algorithm": "h2oplus-v16-headway-reserve-three-quarter",
            "parent_actor_algorithm": "h2oplus-v14-terminal-quarter",
            "parent_budget_online_events": 25_000,
            "activation_budget_online_events": 25_000,
            "predesignated_candidate": "tail-service-reserve-thirty",
            "candidate_floor_seconds": 30.0,
            "diagnostic_floor_seconds": [20.0, 40.0],
            "activation_start_fraction": ACTIVATION_START_FRACTION,
            "activation_end_fraction": ACTIVATION_END_FRACTION,
            "max_onboard_capacity_fraction": MAX_ONBOARD_CAPACITY_FRACTION,
            "requires_forward_bus_present": True,
            "requires_backward_bus_absent": True,
            "requires_positive_eligible_arrival_rate": True,
            "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
            "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
            "parent_reserve_horizon_fraction": PARENT_RESERVE_HORIZON_FRACTION,
            "unit": "physical_hold_seconds",
            "parent_action_preserved_outside_guard": True,
            "guarded_hold_may_exceed_parent": True,
            "physical_max_hold_preserved": True,
            "raw_action_multiplication_forbidden": True,
            "actor_or_temperature_tensor_modification_forbidden": True,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
        },
        "evaluation": {
            "training_seeds": list(SEEDS),
            "development_blocks": [
                "v12-staged-release-development-01",
                "v12-staged-release-development-02",
                "v12-staged-release-development-03",
            ],
            "demand_scales": [0.85, 1.0, 1.15],
            "new_physical_methods": 15,
            "new_remote_sumo_episodes": 135,
            "jobs_per_block": 45,
            "workers": 12,
            "host": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
            "formal_held_out_outcomes_read": False,
        },
        "analysis": {
            "budgets": [0, 5000, 10000, 25000],
            "prefix_source": "frozen v14 terminal-quarter logical results at 0/5k/10k",
            "control_source": "frozen v14 development closeout only",
            "candidate_25000_source": "new v17 tail-service-reserve-thirty results only",
            "diagnostic_25000_sources": [
                "tail-service-reserve-twenty",
                "tail-service-reserve-forty",
            ],
            "formula_source": "unchanged frozen v16 AULC, endpoint, seed-win, completion and demand-scale gates",
            "diagnostics_cannot_replace_candidate": True,
            "all_v16_numeric_thresholds_retained": True,
        },
        "analysis_gates": analysis_gates,
        "v17_design_inputs": v17_design_inputs,
        "frozen_v16_evidence": frozen_v16,
    }
    write_once(args.protocol_output, protocol)
    protocol_path = args.protocol_output.expanduser().resolve(strict=True)

    parent_rows = {
        int(row["training_seed"]): row
        for row in v16_methods["methods"]
        if row.get("algorithm") == "h2oplus-v16-headway-reserve-three-quarter"
    }
    if set(parent_rows) != set(SEEDS):
        raise ValueError("sealed v16 three-quarter parent grid is incomplete")
    methods_output = args.methods_output.expanduser().resolve()
    methods, aliases, identities = [], [], []
    behavior_keys: set[
        tuple[int, str, float, float, float, float, float, float, float]
    ] = set()
    for seed in SEEDS:
        parent = parent_rows[seed]
        checkpoint = Path(parent["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = v16_methods_path.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if sha256(checkpoint) != parent["checkpoint_sha256"]:
            raise ValueError(f"sealed v16 parent checkpoint drifted: seed {seed}")
        seed_rows = []
        for variant in VARIANTS:
            floor_seconds = TAIL_SERVICE_FLOORS[variant]
            algorithm = f"h2oplus-v17-tail-service-reserve-{variant}"
            name = f"{algorithm}-seed{seed}-events000025000"
            key = (
                seed,
                parent["actor_tensor_sha256"],
                PARENT_BASE_HOLD_SCALE,
                PARENT_HEADWAY_BALANCE_GAIN,
                PARENT_RESERVE_HORIZON_FRACTION,
                floor_seconds,
                ACTIVATION_START_FRACTION,
                ACTIVATION_END_FRACTION,
                MAX_ONBOARD_CAPACITY_FRACTION,
            )
            if key in behavior_keys:
                raise ValueError(f"duplicate v17 physical behavior: {key}")
            behavior_keys.add(key)
            row = dict(parent)
            row.update(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "variant": variant,
                    "checkpoint": os.path.relpath(checkpoint, methods_output.parent),
                    "artifact_lineage": "v17-tail-service-reserve",
                    "parent_artifact_lineage": "v16-headway-reserve",
                    "parent_method": parent["name"],
                    "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
                    "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
                    "parent_reserve_horizon_fraction": PARENT_RESERVE_HORIZON_FRACTION,
                    "tail_service_floor_seconds": floor_seconds,
                    "activation_start_fraction": ACTIVATION_START_FRACTION,
                    "activation_end_fraction": ACTIVATION_END_FRACTION,
                    "max_onboard_capacity_fraction": MAX_ONBOARD_CAPACITY_FRACTION,
                    "predesignated_candidate": variant == "thirty",
                    "source_protocol_sha256": EXPECTED["v16_protocol"],
                    "v17_protocol_sha256": sha256(protocol_path),
                }
            )
            methods.append(row)
            aliases.append(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "online_events": 25_000,
                    "physical_method": name,
                    "actor_tensor_sha256": parent["actor_tensor_sha256"],
                    "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
                    "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
                    "parent_reserve_horizon_fraction": PARENT_RESERVE_HORIZON_FRACTION,
                    "tail_service_floor_seconds": floor_seconds,
                    "activation_start_fraction": ACTIVATION_START_FRACTION,
                    "activation_end_fraction": ACTIVATION_END_FRACTION,
                    "max_onboard_capacity_fraction": MAX_ONBOARD_CAPACITY_FRACTION,
                    "parent_method": parent["name"],
                    "predesignated_candidate": variant == "thirty",
                }
            )
            seed_rows.append(
                {
                    "variant": variant,
                    "tail_service_floor_seconds": floor_seconds,
                    "physical_method": name,
                    "behavior_identity": list(key),
                }
            )
        identities.append(
            {
                "training_seed": seed,
                "parent_method": parent["name"],
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "actor_tensor_sha256": parent["actor_tensor_sha256"],
                "temperature_tensor_sha256": parent["temperature_tensor_sha256"],
                "tail_service_reserve_physical_behaviors": seed_rows,
            }
        )

    methods_payload = {
        "schema_version": "transit-evaluation-methods-v17-tail-service-reserve-v1",
        "created_utc": now,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v17_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_development_outcomes_used_for_design": True,
        "selection_rule": METHODS_SELECTION_RULE,
        "predesignated_candidate": "h2oplus-v17-tail-service-reserve-thirty",
        "diagnostic_variants": [
            "h2oplus-v17-tail-service-reserve-twenty",
            "h2oplus-v17-tail-service-reserve-forty",
        ],
        "physical_method_count": 15,
        "logical_method_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "v8_protocol_sha256": v16_methods["v8_protocol_sha256"],
        "v10_protocol_sha256": v16_methods["v10_protocol_sha256"],
        "v11_protocol_sha256": v16_methods["v11_protocol_sha256"],
        "v12_protocol_sha256": v16_methods["v12_protocol_sha256"],
        "v13_protocol_sha256": v16_methods["v13_protocol_sha256"],
        "v14_protocol_sha256": v16_methods["v14_protocol_sha256"],
        "v15_protocol_sha256": v16_methods["v15_protocol_sha256"],
        "v16_protocol_sha256": v16_methods["v16_protocol_sha256"],
        "v17_protocol_sha256": sha256(protocol_path),
        "bindings": {
            "rationale": descriptor(rationale_path, methods_output),
            "registration": descriptor(registration_path, methods_output),
            "protocol": descriptor(protocol_path, methods_output),
            "authorization": descriptor(authorization_path, methods_output),
            "offline_design_evidence": descriptor(design_evidence_path, methods_output),
            "v16_methods": descriptor(v16_methods_path, methods_output),
            "v16_source_freeze": descriptor(v16_source_freeze_path, methods_output),
            "v16_audit": descriptor(v16_audit_path, methods_output),
            "v16_analysis": descriptor(v16_analysis_path, methods_output),
            "v16_closeout": descriptor(v16_closeout_path, methods_output),
            "v16_verification": descriptor(v16_verification_path, methods_output),
        },
        "identity_evidence": identities,
        "aliases": sorted(aliases, key=lambda row: row["name"]),
        "methods": sorted(methods, key=lambda row: row["name"]),
    }
    write_once(methods_output, methods_payload)
    print(
        json.dumps(
            {
                "rationale_sha256": sha256(rationale_path),
                "registration_sha256": sha256(registration_path),
                "protocol_sha256": sha256(protocol_path),
                "methods_sha256": sha256(methods_output),
                "physical_methods": len(methods),
                "new_remote_sumo_episodes": 135,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
