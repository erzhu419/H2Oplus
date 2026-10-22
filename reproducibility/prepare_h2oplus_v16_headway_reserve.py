#!/usr/bin/env python3
"""Preregister the v16 early headway-reserve development screen."""

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
V15_RESULTS = REPRO / "h2oplus_v15_hold_calibration_results"
SEEDS = tuple(range(6201, 6206))
VARIANTS = ("half", "three-quarter", "full")
HORIZON_CUTOFFS = {"half": 0.5, "three-quarter": 0.75, "full": 1.0}
BASE_HOLD_SCALE = 0.125
HEADWAY_BALANCE_GAIN = 1.0
EXPECTED = {
    "v15_methods": "7712a35cbb1adbc55acf50bd8843bb3f4f7dc17090d0e65deeeb10ffe5177e5a",
    "v15_protocol": "6cd856504438ad0e9614b49156100cef12f24201b41562bf223e898f15ef0a8c",
    "v15_source_freeze": "af025dbd817a4d7c4661227fefae7653ecc5ca9bf666e94a1c0a8ca7e05f642d",
    "v15_audit": "a060b49e3155092542681c7d6afef66bcb2b31ab04ffe523273754ae2dcfa7e8",
    "v15_analysis": "d122aa79f6ee2cd5010e241607347921ed9bdc80bbed6ba41c3105e601ac9d9c",
    "v15_closeout": "2e4598c433eb6ffaf18fb9919b608b00d5cdd17f5a53e466ca587d6c8213feab",
    "v15_verification": "f5edd5c2ef651172068c9cd595e6cb7710a3ffdd66e545671f5b4a014885e0b6",
}
METHODS_SELECTION_RULE = (
    "sealed v15 hold-eighth 25k parents and exact seed-matched actors fixed before "
    "v16; base scale 0.125, schedule-adjusted headway gain 1.0, candidate reserve "
    "cutoff 0.75 and diagnostic cutoffs 0.5/1.0 fixed before any v16 outcome; "
    "physical behavior identity is seed plus actor tensor SHA256 plus the complete "
    "headway-reserve transform"
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
        "--v15-methods",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--v15-protocol",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_protocol_v1.json",
    )
    parser.add_argument(
        "--v15-source-freeze",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json",
    )
    parser.add_argument(
        "--v15-audit",
        type=Path,
        default=V15_RESULTS / "evaluation_audit_v1.json",
    )
    parser.add_argument(
        "--v15-analysis",
        type=Path,
        default=V15_RESULTS / "hold_calibration_analysis_v1.json",
    )
    parser.add_argument(
        "--v15-closeout",
        type=Path,
        default=V15_RESULTS / "evaluation_closeout_v1.json",
    )
    parser.add_argument(
        "--v15-verification",
        type=Path,
        default=V15_RESULTS / "evaluation_closeout_verification_v1.json",
    )
    parser.add_argument(
        "--rationale-output",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_rationale_v1.json",
    )
    parser.add_argument(
        "--registration-output",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_registration_v1.json",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_protocol_v1.json",
    )
    parser.add_argument(
        "--methods-output",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_evaluation_methods_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v15_methods_path, v15_methods = load_bound(
        args.v15_methods,
        EXPECTED["v15_methods"],
        "transit-evaluation-methods-v15-hold-calibration-v1",
    )
    v15_protocol_path, v15_protocol = load_bound(
        args.v15_protocol,
        EXPECTED["v15_protocol"],
        "transit-h2oplus-v15-hold-calibration-protocol-v1",
    )
    v15_source_freeze_path, v15_source_freeze = load_bound(
        args.v15_source_freeze,
        EXPECTED["v15_source_freeze"],
        "transit-h2oplus-v15-hold-calibration-evaluation-source-freeze-v1",
    )
    v15_audit_path, v15_audit = load_bound(
        args.v15_audit,
        EXPECTED["v15_audit"],
        "transit-h2oplus-v15-hold-calibration-evaluation-audit-v1",
    )
    v15_analysis_path, v15_analysis = load_bound(
        args.v15_analysis,
        EXPECTED["v15_analysis"],
        "transit-h2oplus-v15-hold-calibration-analysis-v1",
    )
    v15_closeout_path, v15_closeout = load_bound(
        args.v15_closeout,
        EXPECTED["v15_closeout"],
        "transit-h2oplus-v15-hold-calibration-evaluation-closeout-v1",
    )
    v15_verification_path, v15_verification = load_bound(
        args.v15_verification,
        EXPECTED["v15_verification"],
        "transit-h2oplus-v15-hold-calibration-evaluation-closeout-verification-v1",
    )
    checks = v15_analysis.get("development_scientific_result", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if value is False)
    expected_failed = sorted(
        (
            "completion_rate_noninferiority_against_all_controls",
            "relative_aulc_improvement_over_all_controls",
        )
    )
    if (
        failed != expected_failed
        or v15_analysis.get("formal_confirmation_eligible") is not False
        or v15_analysis.get("formal_held_out_outcomes_read") is not False
        or v15_audit.get("passed") is not True
        or v15_closeout.get("passed") is not True
        or v15_verification.get("passed") is not True
        or v15_source_freeze.get("v15_evaluation_outcomes_read") is not False
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (
                v15_methods,
                v15_protocol,
                v15_source_freeze,
                v15_audit,
                v15_analysis,
                v15_closeout,
                v15_verification,
            )
        )
    ):
        raise ValueError("v15 development closeout or failure diagnosis drifted")

    comparisons = v15_analysis["development_scientific_result"]["control_comparisons"]
    completion = v15_analysis["development_scientific_result"][
        "completion_aulc_against_controls"
    ]
    observed = {
        "candidate_aulc": v15_analysis["development_scientific_result"][
            "all_algorithm_mean_aulc"
        ]["h2oplus-v15-hold-quarter"],
        "eighth_aulc": v15_analysis["development_scientific_result"][
            "all_algorithm_mean_aulc"
        ]["h2oplus-v15-hold-eighth"],
        "candidate_endpoint": v15_analysis["development_scientific_result"][
            "mean_primary_at_25000"
        ]["h2oplus-v15-hold-quarter"],
        "eighth_endpoint": v15_analysis["development_scientific_result"][
            "mean_primary_at_25000"
        ]["h2oplus-v15-hold-eighth"],
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
        "candidate_aulc": 2441.8244047906414,
        "eighth_aulc": 2436.147166644969,
        "candidate_endpoint": 2429.964729987385,
        "eighth_endpoint": 2411.0406028351426,
        "relative_vs_v11": 0.0076275224959640245,
        "relative_vs_v12": 0.009713244357756664,
        "completion_delta_vs_h2oplus": -0.004033188697071144,
        "completion_delta_vs_sac": -0.0039415926122909895,
        "completion_delta_vs_v10": -0.003225719836584151,
    }
    if any(
        not math.isclose(observed[key], value, rel_tol=0.0, abs_tol=1e-12)
        for key, value in expected_observed.items()
    ):
        raise ValueError("v15 development signal used for v16 design drifted")

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
        raise FileExistsError(f"refusing to overwrite v16 preregistration outputs: {existing}")

    now = datetime.now(timezone.utc).isoformat()
    frozen_v15 = {
        "methods": descriptor(v15_methods_path),
        "protocol": descriptor(v15_protocol_path),
        "source_freeze": descriptor(v15_source_freeze_path),
        "evaluation_audit": descriptor(v15_audit_path),
        "development_analysis": descriptor(v15_analysis_path),
        "evaluation_closeout": descriptor(v15_closeout_path),
        "evaluation_closeout_verification": descriptor(v15_verification_path),
    }
    rationale = {
        "schema_version": "transit-h2oplus-v16-headway-reserve-rationale-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "frozen-before-any-v16-outcome",
        "formal_held_out_outcomes_read": False,
        "v16_development_evaluation_outcomes_read": False,
        "prior_v14_and_v15_development_outcomes_used_for_design": True,
        "v15_failed_preregistered_gates": failed,
        "v15_observed_design_signal": observed,
        "diagnosis": {
            "uniform_scale_tradeoff": (
                "v15 hold-eighth improved the 25k primary endpoint in 39 of 45 paired "
                "cells relative to hold-quarter, while hold-three-eighths improved "
                "completion in 34 of 45 cells and increased mean primary time"
            ),
            "intervention": (
                "retain the low-cost eighth-scale action everywhere and add a bounded "
                "schedule-adjusted headway reserve only while the episode remains inside "
                "a fixed early-horizon window"
            ),
            "mechanistic_boundary": (
                "the transformed hold is never negative, never exceeds the sealed parent "
                "actor hold, and remains exactly zero whenever the parent actor emits zero"
            ),
        },
        "fixed_design": {
            "parent": "sealed v15 hold-eighth 25k behaviors, exact seeds 6201-6205",
            "predesignated_candidate": "headway-reserve-three-quarter",
            "candidate_horizon_cutoff": 0.75,
            "diagnostic_horizon_cutoffs": [0.5, 1.0],
            "base_hold_scale": BASE_HOLD_SCALE,
            "headway_balance_gain": HEADWAY_BALANCE_GAIN,
            "headway_reserve_formula": (
                "before cutoff: min(parent_hold, max(0.125*parent_hold, "
                "0.5*((H_b-T_b)-(H_f-T_f)))); after cutoff: 0.125*parent_hold"
            ),
            "checkpoint_selection_by_behavior": False,
            "new_training_runs": 0,
            "new_remote_development_episodes": 135,
        },
        "frozen_v15_evidence": frozen_v15,
    }
    write_once(args.rationale_output, rationale)
    rationale_path = args.rationale_output.expanduser().resolve(strict=True)

    registration = {
        "schema_version": "transit-h2oplus-v16-headway-reserve-registration-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "registered-before-any-v16-outcome",
        "formal_held_out_outcomes_read": False,
        "v16_development_evaluation_outcomes_read": False,
        "prior_v14_and_v15_development_outcomes_used_for_design": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "candidate": {
            "algorithm": "h2oplus-v16-headway-reserve-three-quarter",
            "reserve_horizon_fraction": 0.75,
            "base_hold_scale": BASE_HOLD_SCALE,
            "headway_balance_gain": HEADWAY_BALANCE_GAIN,
        },
        "diagnostics_not_eligible_for_selection": [
            {
                "algorithm": "h2oplus-v16-headway-reserve-half",
                "reserve_horizon_fraction": 0.5,
            },
            {
                "algorithm": "h2oplus-v16-headway-reserve-full",
                "reserve_horizon_fraction": 1.0,
            },
        ],
        "selection_rules": {
            "checkpoint_selection_by_behavior": False,
            "candidate_selection_after_v16_outcomes": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "expand_only_through_frozen_v14_prefix_aliases": True,
            "diagnostics_cannot_replace_candidate": True,
            "physical_behavior_identity": [
                "training_seed",
                "actor_tensor_sha256",
                "base_hold_scale",
                "headway_balance_gain",
                "reserve_horizon_fraction",
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
        "frozen_v15_evidence": frozen_v15,
    }
    write_once(args.registration_output, registration)
    registration_path = args.registration_output.expanduser().resolve(strict=True)

    analysis_gates = dict(v15_protocol["analysis_gates"])
    analysis_gates["predesignated_candidate"] = "headway-reserve-three-quarter"
    protocol = {
        "schema_version": "transit-h2oplus-v16-headway-reserve-protocol-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "frozen-before-any-v16-outcome",
        "formal_held_out_outcomes_read": False,
        "v16_development_evaluation_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration_path),
        "rationale": descriptor(rationale_path),
        "parent_v15_protocol": descriptor(v15_protocol_path),
        "headway_reserve": {
            "parent_algorithm": "h2oplus-v15-hold-eighth",
            "parent_actor_algorithm": "h2oplus-v14-terminal-quarter",
            "parent_budget_online_events": 25_000,
            "activation_budget_online_events": 25_000,
            "predesignated_candidate": "headway-reserve-three-quarter",
            "candidate_horizon_cutoff": 0.75,
            "diagnostic_horizon_cutoffs": [0.5, 1.0],
            "base_hold_scale": BASE_HOLD_SCALE,
            "headway_balance_gain": HEADWAY_BALANCE_GAIN,
            "headway_formula_source": "SimpleSAC.transit_v4.policies.headway_balance_hold_seconds_v4",
            "unit": "physical_hold_seconds",
            "zero_hold_preserved_exactly": True,
            "transformed_hold_never_exceeds_parent": True,
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
            "candidate_25000_source": "new v16 three-quarter reserve results only",
            "diagnostic_25000_sources": ["headway-reserve-half", "headway-reserve-full"],
            "formula_source": "unchanged frozen v15 AULC, endpoint, seed-win, completion and demand-scale gates",
            "diagnostics_cannot_replace_candidate": True,
            "all_v15_numeric_thresholds_retained": True,
        },
        "analysis_gates": analysis_gates,
        "frozen_v15_evidence": frozen_v15,
    }
    write_once(args.protocol_output, protocol)
    protocol_path = args.protocol_output.expanduser().resolve(strict=True)

    parent_rows = {
        int(row["training_seed"]): row
        for row in v15_methods["methods"]
        if row.get("algorithm") == "h2oplus-v15-hold-eighth"
    }
    if set(parent_rows) != set(SEEDS):
        raise ValueError("sealed v15 hold-eighth parent grid is incomplete")
    methods_output = args.methods_output.expanduser().resolve()
    methods, aliases, identities = [], [], []
    behavior_keys: set[tuple[int, str, float, float, float]] = set()
    for seed in SEEDS:
        parent = parent_rows[seed]
        checkpoint = Path(parent["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = v15_methods_path.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if sha256(checkpoint) != parent["checkpoint_sha256"]:
            raise ValueError(f"sealed v15 parent checkpoint drifted: seed {seed}")
        seed_rows = []
        for variant in VARIANTS:
            cutoff = HORIZON_CUTOFFS[variant]
            algorithm = f"h2oplus-v16-headway-reserve-{variant}"
            name = f"{algorithm}-seed{seed}-events000025000"
            key = (
                seed,
                parent["actor_tensor_sha256"],
                BASE_HOLD_SCALE,
                HEADWAY_BALANCE_GAIN,
                cutoff,
            )
            if key in behavior_keys:
                raise ValueError(f"duplicate v16 physical behavior: {key}")
            behavior_keys.add(key)
            row = dict(parent)
            row.update(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "variant": variant,
                    "checkpoint": os.path.relpath(checkpoint, methods_output.parent),
                    "artifact_lineage": "v16-headway-reserve",
                    "parent_artifact_lineage": "v15-hold-calibration",
                    "parent_method": parent["name"],
                    "base_hold_scale": BASE_HOLD_SCALE,
                    "headway_balance_gain": HEADWAY_BALANCE_GAIN,
                    "reserve_horizon_fraction": cutoff,
                    "predesignated_candidate": variant == "three-quarter",
                    "source_protocol_sha256": EXPECTED["v15_protocol"],
                    "v16_protocol_sha256": sha256(protocol_path),
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
                    "base_hold_scale": BASE_HOLD_SCALE,
                    "headway_balance_gain": HEADWAY_BALANCE_GAIN,
                    "reserve_horizon_fraction": cutoff,
                    "parent_method": parent["name"],
                    "predesignated_candidate": variant == "three-quarter",
                }
            )
            seed_rows.append(
                {
                    "variant": variant,
                    "reserve_horizon_fraction": cutoff,
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
                "headway_reserve_physical_behaviors": seed_rows,
            }
        )

    methods_payload = {
        "schema_version": "transit-evaluation-methods-v16-headway-reserve-v1",
        "created_utc": now,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v16_development_evaluation_outcomes_read": False,
        "prior_v14_and_v15_development_outcomes_used_for_design": True,
        "selection_rule": METHODS_SELECTION_RULE,
        "predesignated_candidate": "h2oplus-v16-headway-reserve-three-quarter",
        "diagnostic_variants": [
            "h2oplus-v16-headway-reserve-half",
            "h2oplus-v16-headway-reserve-full",
        ],
        "physical_method_count": 15,
        "logical_method_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "v8_protocol_sha256": v15_methods["v8_protocol_sha256"],
        "v10_protocol_sha256": v15_methods["v10_protocol_sha256"],
        "v11_protocol_sha256": v15_methods["v11_protocol_sha256"],
        "v12_protocol_sha256": v15_methods["v12_protocol_sha256"],
        "v13_protocol_sha256": v15_methods["v13_protocol_sha256"],
        "v14_protocol_sha256": v15_methods["v14_protocol_sha256"],
        "v15_protocol_sha256": v15_methods["v15_protocol_sha256"],
        "v16_protocol_sha256": sha256(protocol_path),
        "bindings": {
            "rationale": descriptor(rationale_path, methods_output),
            "registration": descriptor(registration_path, methods_output),
            "protocol": descriptor(protocol_path, methods_output),
            "v15_methods": descriptor(v15_methods_path, methods_output),
            "v15_source_freeze": descriptor(v15_source_freeze_path, methods_output),
            "v15_audit": descriptor(v15_audit_path, methods_output),
            "v15_analysis": descriptor(v15_analysis_path, methods_output),
            "v15_closeout": descriptor(v15_closeout_path, methods_output),
            "v15_verification": descriptor(v15_verification_path, methods_output),
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
