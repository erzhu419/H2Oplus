#!/usr/bin/env python3
"""Preregister v15 hold calibration and build its 15-method manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
V14_RESULTS = REPRO / "h2oplus_v14_terminal_taper_results"
SEEDS = tuple(range(6201, 6206))
VARIANTS = ("eighth", "quarter", "three-eighths")
HOLD_SCALES = {"eighth": 0.125, "quarter": 0.25, "three-eighths": 0.375}
EXPECTED = {
    "v14_methods": "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b",
    "v14_protocol": "ba275fc4c134feadb564b49548699571ca5bf6bfd1aee529759d6ce791df12c5",
    "v14_analysis": "937a5300988311f80a84dc1220dfee3f0f96f2b6f83691d4e309c38d6bbe6856",
    "v14_evaluation_closeout": "95fef5b3a5206ce14a3da164d9f2948b0dc3b5d61ee1d1b34c52c431d1cd1d32",
    "v14_evaluation_verification": "0e20ccc1e7855e4feb5860e5cb334cd60237f28032a948b7325037ad0aa61dc0",
}
METHODS_SELECTION_RULE = (
    "sealed v14 terminal-quarter 25k parents fixed before v15; candidate scale "
    "0.25 and diagnostic scales 0.125/0.375 fixed before any v15 outcome; "
    "physical behavior identity is seed plus actor tensor SHA256 plus hold scale"
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
        "--v14-methods",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--v14-protocol",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_protocol_v1.json",
    )
    parser.add_argument(
        "--v14-analysis",
        type=Path,
        default=V14_RESULTS / "terminal_taper_analysis_v1.json",
    )
    parser.add_argument(
        "--v14-evaluation-closeout",
        type=Path,
        default=V14_RESULTS / "evaluation_closeout_v1.json",
    )
    parser.add_argument(
        "--v14-evaluation-verification",
        type=Path,
        default=V14_RESULTS / "evaluation_closeout_verification_v1.json",
    )
    parser.add_argument(
        "--rationale-output",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_rationale_v1.json",
    )
    parser.add_argument(
        "--registration-output",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_registration_v1.json",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_protocol_v1.json",
    )
    parser.add_argument(
        "--methods-output",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_evaluation_methods_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v14_methods_path, v14_methods = load_bound(
        args.v14_methods,
        EXPECTED["v14_methods"],
        "transit-evaluation-methods-v14-terminal-taper-v1",
    )
    v14_protocol_path, v14_protocol = load_bound(
        args.v14_protocol,
        EXPECTED["v14_protocol"],
        "transit-h2oplus-v14-terminal-taper-protocol-v1",
    )
    v14_analysis_path, v14_analysis = load_bound(
        args.v14_analysis,
        EXPECTED["v14_analysis"],
        "transit-h2oplus-v14-terminal-taper-analysis-v1",
    )
    v14_closeout_path, v14_closeout = load_bound(
        args.v14_evaluation_closeout,
        EXPECTED["v14_evaluation_closeout"],
        "transit-h2oplus-v14-terminal-taper-evaluation-closeout-v1",
    )
    v14_verification_path, v14_verification = load_bound(
        args.v14_evaluation_verification,
        EXPECTED["v14_evaluation_verification"],
        "transit-h2oplus-v14-terminal-taper-evaluation-closeout-verification-v1",
    )
    checks = v14_analysis.get("development_scientific_result", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if value is False)
    expected_failed = sorted(
        (
            "candidate_mean_lower_than_all_controls_at_25000",
            "completion_rate_noninferiority_against_all_controls",
            "relative_aulc_improvement_over_all_controls",
            "seed_wins_over_all_controls",
        )
    )
    if (
        failed != expected_failed
        or v14_analysis.get("formal_confirmation_eligible") is not False
        or v14_analysis.get("formal_held_out_outcomes_read") is not False
        or v14_closeout.get("passed") is not True
        or v14_verification.get("passed") is not True
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (v14_methods, v14_protocol, v14_analysis, v14_closeout, v14_verification)
        )
    ):
        raise ValueError("v14 development closeout or failure diagnosis drifted")

    outputs = (
        args.rationale_output,
        args.registration_output,
        args.protocol_output,
        args.methods_output,
    )
    existing = [str(path.expanduser().resolve()) for path in outputs if path.expanduser().resolve().exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite v15 preregistration outputs: {existing}")

    now = datetime.now(timezone.utc).isoformat()
    frozen_v14 = {
        "methods": descriptor(v14_methods_path),
        "protocol": descriptor(v14_protocol_path),
        "development_analysis": descriptor(v14_analysis_path),
        "evaluation_closeout": descriptor(v14_closeout_path),
        "evaluation_closeout_verification": descriptor(v14_verification_path),
    }
    rationale = {
        "schema_version": "transit-h2oplus-v15-hold-calibration-rationale-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "frozen-before-any-v15-outcome",
        "formal_held_out_outcomes_read": False,
        "v15_development_evaluation_outcomes_read": False,
        "prior_development_outcomes_used_for_design": True,
        "v14_failed_preregistered_gates": failed,
        "diagnosis": {
            "terminal_taper_direction": (
                "v14 terminal actor learning-rate taper degraded the 25k endpoint and "
                "AULC relative to stronger preregistered controls"
            ),
            "operational_signal": (
                "the v14 terminal-quarter parent retained a completion buffer but emitted "
                "excess physical holding; primary welfare is two times wait plus ride"
            ),
            "intervention_boundary": (
                "calibrate only physical hold seconds after deterministic actor inference; "
                "do not alter actor tensors, checkpoints, training, state, reward, or SUMO"
            ),
        },
        "fixed_design": {
            "parent": "sealed v14 terminal-quarter 25k actors, exact seed-matched 6201-6205",
            "predesignated_candidate": "hold-quarter",
            "candidate_hold_scale": 0.25,
            "diagnostic_hold_scales": [0.125, 0.375],
            "scale_unit": "physical_hold_seconds",
            "zero_hold_preserved_exactly": True,
            "activation_budget_online_events": 25_000,
            "checkpoint_selection_by_behavior": False,
            "new_training_runs": 0,
            "new_remote_development_episodes": 135,
        },
        "exploratory_surrogate_not_evidence": {
            "used_only_to_fix_the_candidate_before_v15": True,
            "model": "v14 development-cell fixed effects plus quadratic holding summaries",
            "predicted_candidate_primary_at_25000": 2426.34,
            "predicted_candidate_completion_at_25000": 0.96703,
            "warning": (
                "counterfactual predictions are not v15 outcomes and cannot establish a gate, "
                "formal effect, or independent confirmation"
            ),
        },
        "frozen_v14_evidence": frozen_v14,
    }
    write_once(args.rationale_output, rationale)
    rationale_path = args.rationale_output.expanduser().resolve(strict=True)

    registration = {
        "schema_version": "transit-h2oplus-v15-hold-calibration-registration-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "registered-before-any-v15-outcome",
        "formal_held_out_outcomes_read": False,
        "v15_development_evaluation_outcomes_read": False,
        "prior_development_outcomes_used_for_design": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "candidate": {
            "algorithm": "h2oplus-v15-hold-quarter",
            "hold_scale": 0.25,
            "unit": "physical_hold_seconds",
        },
        "diagnostics_not_eligible_for_selection": [
            {"algorithm": "h2oplus-v15-hold-eighth", "hold_scale": 0.125},
            {"algorithm": "h2oplus-v15-hold-three-eighths", "hold_scale": 0.375},
        ],
        "selection_rules": {
            "checkpoint_selection_by_behavior": False,
            "candidate_selection_after_v15_outcomes": False,
            "expand_only_through_frozen_v14_prefix_aliases": True,
            "physical_behavior_identity": ["training_seed", "actor_tensor_sha256", "hold_scale"],
        },
        "episode_accounting": {
            "physical_methods": 15,
            "development_cells_per_method": 9,
            "new_remote_sumo_episodes": 135,
            "reused_frozen_v14_control_episodes": 630,
            "duplicate_control_episodes_allowed": False,
        },
        "rationale": descriptor(rationale_path),
        "frozen_v14_evidence": frozen_v14,
    }
    write_once(args.registration_output, registration)
    registration_path = args.registration_output.expanduser().resolve(strict=True)

    analysis_gates = dict(v14_protocol["analysis_gates"])
    analysis_gates["predesignated_candidate"] = "hold-quarter"
    protocol = {
        "schema_version": "transit-h2oplus-v15-hold-calibration-protocol-v1",
        "created_utc": now,
        "study_stage": "development",
        "status": "frozen-before-any-v15-outcome",
        "formal_held_out_outcomes_read": False,
        "v15_development_evaluation_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration_path),
        "rationale": descriptor(rationale_path),
        "parent_v14_protocol": descriptor(v14_protocol_path),
        "hold_calibration": {
            "parent_algorithm": "h2oplus-v14-terminal-quarter",
            "parent_budget_online_events": 25_000,
            "activation_budget_online_events": 25_000,
            "predesignated_candidate": "hold-quarter",
            "candidate_hold_scale": 0.25,
            "diagnostic_hold_scales": [0.125, 0.375],
            "unit": "physical_hold_seconds",
            "zero_hold_preserved_exactly": True,
            "raw_action_multiplication_forbidden": True,
            "actor_or_temperature_tensor_modification_forbidden": True,
            "checkpoint_selection_by_behavior": False,
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
            "candidate_25000_source": "new v15 hold-quarter results only",
            "diagnostic_25000_sources": ["hold-eighth", "hold-three-eighths"],
            "formula_source": "frozen v14 AULC, endpoint, seed-win, completion and demand-scale gates",
            "diagnostics_cannot_replace_candidate": True,
        },
        "analysis_gates": analysis_gates,
        "frozen_v14_evidence": frozen_v14,
    }
    write_once(args.protocol_output, protocol)
    protocol_path = args.protocol_output.expanduser().resolve(strict=True)

    parent_rows = {
        int(row["training_seed"]): row
        for row in v14_methods["methods"]
        if row.get("algorithm") == "h2oplus-v14-terminal-quarter"
    }
    if set(parent_rows) != set(SEEDS):
        raise ValueError("sealed v14 terminal-quarter parent grid is incomplete")
    methods_output = args.methods_output.expanduser().resolve()
    methods, aliases, identities = [], [], []
    behavior_keys: set[tuple[int, str, float]] = set()
    for seed in SEEDS:
        parent = parent_rows[seed]
        checkpoint = Path(parent["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = v14_methods_path.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if sha256(checkpoint) != parent["checkpoint_sha256"]:
            raise ValueError(f"sealed v14 parent checkpoint drifted: seed {seed}")
        seed_rows = []
        for variant in VARIANTS:
            scale = HOLD_SCALES[variant]
            algorithm = f"h2oplus-v15-hold-{variant}"
            name = f"{algorithm}-seed{seed}-events000025000"
            key = (seed, parent["actor_tensor_sha256"], scale)
            if key in behavior_keys:
                raise ValueError(f"duplicate v15 physical behavior: {key}")
            behavior_keys.add(key)
            row = dict(parent)
            row.update(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "variant": variant,
                    "checkpoint": os.path.relpath(checkpoint, methods_output.parent),
                    "artifact_lineage": "v15-hold-calibration",
                    "parent_artifact_lineage": "v14-terminal-taper",
                    "parent_method": parent["name"],
                    "hold_scale": scale,
                    "predesignated_candidate": variant == "quarter",
                    "source_protocol_sha256": EXPECTED["v14_protocol"],
                    "v15_protocol_sha256": sha256(protocol_path),
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
                    "hold_scale": scale,
                    "parent_method": parent["name"],
                    "predesignated_candidate": variant == "quarter",
                }
            )
            seed_rows.append(
                {
                    "variant": variant,
                    "hold_scale": scale,
                    "physical_method": name,
                    "behavior_identity": [seed, parent["actor_tensor_sha256"], scale],
                }
            )
        identities.append(
            {
                "training_seed": seed,
                "parent_method": parent["name"],
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "actor_tensor_sha256": parent["actor_tensor_sha256"],
                "temperature_tensor_sha256": parent["temperature_tensor_sha256"],
                "calibrated_physical_behaviors": seed_rows,
            }
        )

    methods_payload = {
        "schema_version": "transit-evaluation-methods-v15-hold-calibration-v1",
        "created_utc": now,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v15_development_evaluation_outcomes_read": False,
        "prior_development_outcomes_used_for_design": True,
        "selection_rule": METHODS_SELECTION_RULE,
        "predesignated_candidate": "h2oplus-v15-hold-quarter",
        "diagnostic_variants": [
            "h2oplus-v15-hold-eighth",
            "h2oplus-v15-hold-three-eighths",
        ],
        "physical_method_count": 15,
        "logical_method_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "v8_protocol_sha256": v14_methods["v8_protocol_sha256"],
        "v10_protocol_sha256": v14_methods["v10_protocol_sha256"],
        "v11_protocol_sha256": v14_methods["v11_protocol_sha256"],
        "v12_protocol_sha256": v14_methods["v12_protocol_sha256"],
        "v13_protocol_sha256": v14_methods["v13_protocol_sha256"],
        "v14_protocol_sha256": v14_methods["v14_protocol_sha256"],
        "v15_protocol_sha256": sha256(protocol_path),
        "bindings": {
            "rationale": descriptor(rationale_path, methods_output),
            "registration": descriptor(registration_path, methods_output),
            "protocol": descriptor(protocol_path, methods_output),
            "v14_methods": descriptor(v14_methods_path, methods_output),
            "v14_analysis": descriptor(v14_analysis_path, methods_output),
            "v14_evaluation_closeout": descriptor(v14_closeout_path, methods_output),
            "v14_evaluation_verification": descriptor(v14_verification_path, methods_output),
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
