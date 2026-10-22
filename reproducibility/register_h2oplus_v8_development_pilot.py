#!/usr/bin/env python3
"""Register, but never submit, the evidence-bound H2O+ v8 development pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "transit-h2oplus-v8-development-pilot-registration-v1"
V7_METHODS = (
    "h2oplus",
    "h2oplus-source",
    "h2oplus-no-ratio",
    "h2oplus-no-warmstart",
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
)
HELD_OUT_SUMO_SEEDS = tuple(range(2101, 2111))
HELD_OUT_DEMAND_SEEDS = tuple(range(3101, 3111))
DEVELOPMENT_SUMO_SEEDS = tuple(range(4101, 4106))
DEVELOPMENT_DEMAND_SEEDS = tuple(range(4201, 4206))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def register(args: argparse.Namespace) -> dict[str, Any]:
    analysis_path = args.mechanism_analysis.expanduser().resolve()
    sensitivity_manifest_path = args.sensitivity_manifest.expanduser().resolve()
    sensitivity_verification_path = (
        args.sensitivity_verification.expanduser().resolve()
    )
    sensitivity_stability_path = args.sensitivity_stability.expanduser().resolve()
    ablation_manifest_path = args.ablation_manifest.expanduser().resolve()
    ablation_verification_path = args.ablation_verification.expanduser().resolve()
    core_manifest_path = args.core_manifest.expanduser().resolve()
    held_out_design_path = args.held_out_design.expanduser().resolve()
    required = (
        analysis_path,
        sensitivity_manifest_path,
        sensitivity_verification_path,
        sensitivity_stability_path,
        ablation_manifest_path,
        ablation_verification_path,
        core_manifest_path,
        held_out_design_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing v8 registration evidence: " + ", ".join(missing))

    analysis = json.loads(analysis_path.read_text())
    sensitivity_manifest = json.loads(sensitivity_manifest_path.read_text())
    sensitivity_verification = json.loads(sensitivity_verification_path.read_text())
    sensitivity_stability = json.loads(sensitivity_stability_path.read_text())
    ablation_manifest = json.loads(ablation_manifest_path.read_text())
    ablation_verification = json.loads(ablation_verification_path.read_text())
    held_out = json.loads(held_out_design_path.read_text())

    if tuple(analysis.get("methods", [])) != V7_METHODS or len(analysis.get("runs", [])) != 40:
        raise ValueError("complete mechanism analysis is not the exact 8-method x 5-seed set")
    interpretation = analysis.get("scientific_interpretation", {})
    if (
        interpretation.get("supports_held_out_model_effect_claim") is not False
        or interpretation.get("supports_formal_method_selection") is not False
    ):
        raise ValueError("mechanism analysis lacks required scientific disclaimers")
    if not (
        sensitivity_manifest.get("artifact_integrity_verified") is True
        and sensitivity_verification.get("passed") is True
        and ablation_manifest.get("artifact_integrity_verified") is True
        and ablation_verification.get("passed") is True
    ):
        raise ValueError("v7 ablation or sensitivity integrity evidence is incomplete")
    if sensitivity_stability.get("artifact_integrity_implication") != "none":
        raise ValueError("sensitivity stability evidence is not separated from integrity")

    comparison = analysis.get("paired_training_simulator_comparisons", {}).get(
        "h2oplus-lambda0p5", {}
    )
    if comparison.get("seeds_with_higher_training_simulator_reward") != 5:
        raise ValueError("lambda0p5 does not have the observed 5/5 training-internal pattern")
    lambda_runs = [
        run
        for run in sensitivity_stability.get("runs", [])
        if run.get("method") == "h2oplus-lambda0p5"
    ]
    if len(lambda_runs) != 5 or any(not run.get("passed") for run in lambda_runs):
        raise ValueError("lambda0p5 lacks five stable sensitivity runs")

    disjointness = held_out.get("training_disjointness", {})
    held_out_sumo = tuple(disjointness.get("evaluation_sumo_seeds", []))
    held_out_demand = tuple(disjointness.get("evaluation_demand_realization_seeds", []))
    training_sumo = tuple(disjointness.get("training_sumo_seeds", []))
    training_demand = tuple(disjointness.get("training_demand_realization_seeds", []))
    if held_out_sumo != HELD_OUT_SUMO_SEEDS or held_out_demand != HELD_OUT_DEMAND_SEEDS:
        raise ValueError("held-out environment seed contract changed")
    if set(DEVELOPMENT_SUMO_SEEDS) & set(held_out_sumo) or set(
        DEVELOPMENT_DEMAND_SEEDS
    ) & set(held_out_demand):
        raise ValueError("development and held-out seed blocks overlap")
    if set(DEVELOPMENT_SUMO_SEEDS) & set(training_sumo) or set(
        DEVELOPMENT_DEMAND_SEEDS
    ) & set(training_demand):
        raise ValueError("development and training seed blocks overlap")

    return {
        "schema_version": SCHEMA_VERSION,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "status": "registered-not-submitted",
        "campaign": "H2O+ v8 disjoint-development improvement pilot",
        "execution_authorized": False,
        "scheduler_task_ids": [],
        "submission_requires_explicit_user_approval": True,
        "local_sumo_or_training_allowed": False,
        "frozen_v7_code_modification_allowed": False,
        "scientific_scope": {
            "supports_current_held_out_model_effect_claim": False,
            "supports_current_formal_method_selection": False,
            "purpose": (
                "screen one evidence-motivated candidate on development scenarios "
                "before any single-use formal held-out confirmation"
            ),
        },
        "evidence": {
            "complete_v7_mechanism_analysis": _descriptor(analysis_path),
            "sensitivity_manifest": _descriptor(sensitivity_manifest_path),
            "sensitivity_manifest_verification": _descriptor(
                sensitivity_verification_path
            ),
            "sensitivity_stability_audit": _descriptor(sensitivity_stability_path),
            "ablation_manifest": _descriptor(ablation_manifest_path),
            "ablation_manifest_verification": _descriptor(
                ablation_verification_path
            ),
            "core_manifest": _descriptor(core_manifest_path),
            "frozen_held_out_environment_design": _descriptor(held_out_design_path),
        },
        "observed_training_internal_pattern": {
            "candidate": "h2oplus-lambda0p5",
            "tail_reward_mean_delta_vs_h2oplus": comparison["mean_delta"],
            "tail_reward_median_delta_vs_h2oplus": comparison["median_delta"],
            "paired_seed_deltas_vs_h2oplus": comparison[
                "paired_seed_deltas_vs_h2oplus"
            ],
            "seeds_with_higher_training_simulator_reward": 5,
            "stability_failures": [],
            "ratio_diagnostic": analysis["ratio_diagnostic"],
            "interpretation": (
                "candidate-generation evidence only; training simulator rewards "
                "cannot establish model effectiveness"
            ),
        },
        "development_environment": {
            "status": "design-registered; manifests not yet materialized",
            "network": "same calibrated physical network; independent stochastic cells",
            "blocks": [
                {
                    "name": f"v8-development-{index:02d}",
                    "sumo_seed": sumo_seed,
                    "demand_realization_seed": demand_seed,
                }
                for index, (sumo_seed, demand_seed) in enumerate(
                    zip(DEVELOPMENT_SUMO_SEEDS, DEVELOPMENT_DEMAND_SEEDS), start=1
                )
            ],
            "demand_scales": [0.85, 1.0, 1.15],
            "depart_jitter_seconds": 60.0,
            "environment_cells": 15,
            "required_pre_execution_gate": (
                "materialize and hash all 15 manifests; prove zero manifest/runtime "
                "hash overlap and zero seed overlap with frozen training and held-out designs"
            ),
            "frozen_formal_held_out_sumo_seeds": list(HELD_OUT_SUMO_SEEDS),
            "frozen_formal_held_out_demand_seeds": list(HELD_OUT_DEMAND_SEEDS),
            "frozen_v7_training_sumo_seeds": list(training_sumo),
            "frozen_v7_training_demand_seeds": list(training_demand),
            "formal_held_out_may_be_read_for_selection": False,
        },
        "stages": [
            {
                "name": "A-existing-policy-development-screen",
                "training": False,
                "policies": ["h2oplus", "h2oplus-lambda0p5"],
                "training_seeds": [5101, 5102, 5103, 5104, 5105],
                "primary_endpoint": "generalized_seconds_per_passenger",
                "primary_direction": "lower-is-better",
                "paired_unit": "training seed x development environment block",
                "advance_gate": {
                    "mean_relative_improvement_at_least": 0.02,
                    "training_seeds_with_lower_paired_mean_at_least": 4,
                    "one_sided_paired_permutation_p_less_than": 0.05,
                    "completion_rate_noninferiority_margin": 0.005,
                    "no_demand_scale_mean_relative_degradation_greater_than": 0.01,
                },
            },
            {
                "name": "B-v8-ess-tempered-ratio-development",
                "blocked_by": "A-existing-policy-development-screen.advance_gate",
                "training": True,
                "training_seeds": [6101, 6102, 6103, 6104, 6105],
                "candidate": "h2oplus-v8-lambda0p5-ess-tempered-ratio",
                "implementation_contract": {
                    "new_namespace_and_checkpoint_schema_required": True,
                    "mixed_backup_lambda": 0.5,
                    "ratio_log_weight_clip": [
                        -1.3862943611198906,
                        1.3862943611198906,
                    ],
                    "ratio_temperature_beta_range": [0.0, 1.0],
                    "ratio_temperature_rule": (
                        "choose the largest detached beta by deterministic bisection "
                        "such that batch ESS fraction is at least 0.5"
                    ),
                    "ratio_weight_normalization": "divide by detached batch mean",
                    "target_batch_ess_fraction_minimum": 0.5,
                    "preserved_contracts": [
                        "gamma=1 terminal semantics",
                        "100000-decision-event hard cap",
                        "same canonical v7 dataset and state/action contract",
                        "one canonical process per physical GPU",
                        "all registered seeds retained",
                    ],
                    "required_unit_gates": [
                        "finite positive normalized weights",
                        "deterministic beta and ESS bound",
                        "beta=1 when the untempered batch already passes",
                        "exact checkpoint/resume equivalence",
                        "v7 artifacts remain loadable only through the frozen v7 path",
                    ],
                },
                "advance_gate": {
                    "artifact_integrity": "all five seeds complete and independently audited",
                    "scientific_stability_failures_allowed": 0,
                    "mean_relative_improvement_vs_strongest_stage_A_policy_at_least": 0.02,
                    "training_seeds_with_lower_paired_mean_at_least": 4,
                    "one_sided_paired_permutation_p_less_than": 0.05,
                    "completion_rate_noninferiority_margin": 0.005,
                },
            },
            {
                "name": "C-single-use-formal-confirmation",
                "blocked_by": "B-v8-ess-tempered-ratio-development.advance_gate",
                "status": "not authorized",
                "selection_lock": (
                    "freeze one candidate, code hash, checkpoints, endpoints and analysis "
                    "before accessing the existing formal held-out SUMO environment"
                ),
                "post_confirmation_tuning_allowed": False,
            },
        ],
        "stop_rules": [
            "stop if any development scenario overlaps frozen training or held-out inputs",
            "stop on nonfinite state, nonzero exit, or event/update contract failure",
            "do not replace a failed seed or select an earlier checkpoint",
            "do not proceed past a failed stage gate",
            "do not submit any task from this registration artifact automatically",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-analysis", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-verification", type=Path, required=True)
    parser.add_argument("--sensitivity-stability", type=Path, required=True)
    parser.add_argument("--ablation-manifest", type=Path, required=True)
    parser.add_argument("--ablation-verification", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--held-out-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = register(args)
    _write_json(args.output, payload)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
