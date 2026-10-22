#!/usr/bin/env python3
"""Audit and analyze the preregistered v10 delayed-transition development screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256

REPORT_SCHEMA = "transit-h2oplus-v10-delayed-analysis-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v10-delayed-v1"
REGISTRATION_SCHEMA = "transit-h2oplus-v10-delayed-registration-v4"
PROTOCOL_SCHEMA = "transit-h2oplus-v10-delayed-protocol-v4"
CLOSEOUT_SCHEMA = "transit-h2oplus-v10-delayed-branch-closeout-v4"
VERIFICATION_SCHEMA = "transit-h2oplus-v10-delayed-branch-closeout-verification-v4"
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-(?:unit-control|"
    r"delayed-replay|delayed-actor|delayed-order|delayed-combined))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SEEDS = tuple(range(6201, 6206))
BUDGETS = (0, 5_000, 10_000, 25_000)
ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-unit-control",
    "h2oplus-v10-delayed-replay",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v10-delayed-order",
    "h2oplus-v10-delayed-combined",
)
SINGLE_FACTORS = (
    "h2oplus-v10-delayed-replay",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v10-delayed-order",
)

# This operationalization is frozen before any v10 evaluation outcome is read.
# It treats attribution as available only when every single-factor contrast is
# complete and at least one factor improves the unit control in >=3/5 seeds.
SINGLE_FACTOR_MINIMUM_SEED_WINS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--branch-closeout", type=Path, required=True)
    parser.add_argument("--branch-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _nested(payload: dict, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def _resolve_output(root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    local = root / candidate.name
    if local.is_file():
        return local.resolve()
    matches = list(root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve matrix output {raw!r} under {root}")


def _load_control(path: Path, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema drift for {resolved}")
    return resolved, payload


def _load_methods(path: Path) -> tuple[Path, dict, dict[str, list[dict]]]:
    resolved, payload = _load_control(path, METHODS_SCHEMA)
    methods = payload.get("methods", [])
    aliases = payload.get("aliases", [])
    if (
        payload.get("study_stage") != "development"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 60
        or payload.get("logical_method_budget_alias_count") != 140
        or payload.get("expected_remote_sumo_episodes") != 540
        or len(methods) != 60
        or len(aliases) != 140
    ):
        raise ValueError("v10 methods manifest scope drifted")
    physical = {row.get("name"): row for row in methods}
    if len(physical) != 60:
        raise ValueError("duplicate physical method names")
    by_physical: dict[str, list[dict]] = {name: [] for name in physical}
    logical_names = set()
    for alias in aliases:
        match = METHOD.fullmatch(alias.get("name", ""))
        physical_name = alias.get("physical_method")
        if (
            match is None
            or alias["name"] in logical_names
            or physical_name not in physical
            or alias.get("algorithm") != match.group("algorithm")
            or alias.get("training_seed") != int(match.group("seed"))
            or alias.get("online_events") != int(match.group("events"))
            or int(match.group("seed")) not in SEEDS
            or int(match.group("events")) not in BUDGETS
            or alias.get("actor_tensor_sha256")
            != physical[physical_name].get("actor_tensor_sha256")
        ):
            raise ValueError(f"malformed v10 logical alias: {alias!r}")
        logical_names.add(alias["name"])
        by_physical[physical_name].append(alias)
    expected = {
        f"{algorithm}-seed{seed}-events{events:09d}"
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for events in BUDGETS
    }
    if logical_names != expected or any(not rows for rows in by_physical.values()):
        raise ValueError("v10 logical alias grid is incomplete")
    return resolved, payload, by_physical


def _validate_training_metadata(payload: dict, method: dict, protocol_hash: str) -> None:
    training = payload.get("policy", {}).get("training_metadata", {})
    common = {
        "training_seed": method["training_seed"],
        "run_signature": method["training_run_signature"],
        "training_checkpoint_sha256": method["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (training.get(key), value)
        for key, value in common.items()
        if training.get(key) != value
    }
    admission = training.get("training_admission", {})
    if method["artifact_lineage"] == "v8-fewshot":
        specific = {
            "method": method["algorithm"],
            "fewshot_online_events_nominal": method["online_events"],
            "fewshot_online_events_actual": method["actual_online_events"],
            "fewshot_learner_update_steps": method["learner_update_steps"],
            "fewshot_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        artifact_protocol = training.get("fewshot_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v8-fewshot-development-v1"
    elif method["artifact_lineage"] == "v10-delayed":
        variant = method["algorithm"][len("h2oplus-v10-") :]
        specific = {
            "method": "h2oplus",
            "variant": variant,
            "v10_online_events_nominal": method["online_events"],
            "v10_online_events_actual": method["actual_online_events"],
            "v10_learner_update_steps": method["learner_update_steps"],
            "v10_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        artifact_protocol = training.get("v10_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v10-delayed-development-v4"
        intervention = training.get("v10_intervention", {})
        if intervention.get("variant") != variant or intervention.get("unit_ratio_weights") is not True:
            raise ValueError("v10 result intervention provenance drifted")
    else:
        raise ValueError("unsupported physical actor lineage")
    mismatches.update(
        {
            key: (training.get(key), value)
            for key, value in specific.items()
            if training.get(key) != value
        }
    )
    if (
        mismatches
        or admission.get("schema_version") != admission_schema
        or admission.get("sha256") != method["training_admission_sha256"]
        or artifact_protocol != method["source_protocol_sha256"]
        or method["artifact_lineage"] == "v10-delayed"
        and artifact_protocol != protocol_hash
    ):
        raise ValueError(f"evaluation result policy provenance drifted: {mismatches}")


def _load_rows(
    roots: list[Path],
    methods_payload: dict,
    aliases_by_physical: dict[str, list[dict]],
    protocol_hash: str,
) -> tuple[list[dict], list[dict]]:
    if len(roots) != 3:
        raise ValueError("v10 analysis requires exactly three merged development blocks")
    physical_methods = {row["name"]: row for row in methods_payload["methods"]}
    rows, manifests, seen = [], [], set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256")
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256")
            != sha256(Path(methods_payload["manifest_path"]))
            or len(jobs) != 180
            or not isinstance(source_hashes, dict)
        ):
            raise ValueError(f"v10 matrix block is incomplete: {root}")
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "jobs": len(jobs),
                "source_file_count": len(source_hashes),
            }
        )
        for job in jobs:
            physical_name = job.get("method", {}).get("name")
            if physical_name not in physical_methods:
                raise ValueError("matrix contains an unregistered physical actor")
            method = physical_methods[physical_name]
            output = _resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != job["signature"]
                or payload.get("method") != physical_name
                or payload.get("source_sha256") != source_hashes
            ):
                raise ValueError(f"v10 physical result binding failed: {output}")
            _validate_training_metadata(payload, method, protocol_hash)
            if any(
                abs(float(value)) > 1e-9
                for value in payload.get("passenger_welfare", {})
                .get("chronological_coverage_errors", {})
                .values()
            ):
                raise ValueError(f"v10 welfare ledger failed: {output}")
            block = str(job["environment_block"])
            scale = float(job["scenario"]["demand_scale"])
            primary = _finite(
                _nested(
                    payload,
                    "passenger_welfare",
                    "accrued_generalized_seconds_per_departed_passenger",
                ),
                "primary endpoint",
            )
            completion = _finite(
                _nested(payload, "passengers", "completion_rate"),
                "completion rate",
            )
            for alias in aliases_by_physical[physical_name]:
                key = (
                    alias["algorithm"],
                    alias["training_seed"],
                    alias["online_events"],
                    block,
                    scale,
                )
                if key in seen:
                    raise ValueError(f"duplicate v10 logical result: {key}")
                seen.add(key)
                rows.append(
                    {
                        "algorithm": alias["algorithm"],
                        "seed": alias["training_seed"],
                        "events": alias["online_events"],
                        "physical_method": physical_name,
                        "actor_tensor_sha256": alias["actor_tensor_sha256"],
                        "block": block,
                        "scale": scale,
                        "primary": primary,
                        "completion": completion,
                        "output": str(output),
                        "sha256": sha256(output),
                    }
                )
    if len(rows) != 1260:
        raise ValueError(f"v10 analysis requires 1260 logical rows, got {len(rows)}")
    return rows, manifests


def _cell_means(rows: list[dict]) -> dict[tuple[str, int, int], dict[str, float]]:
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["seed"], row["events"]), []).append(row)
    result = {}
    for key, values in grouped.items():
        if len(values) != 9:
            raise ValueError(f"logical policy does not cover nine development cells: {key}")
        result[key] = {
            "primary": float(np.mean([item["primary"] for item in values])),
            "completion": float(np.mean([item["completion"] for item in values])),
        }
    if len(result) != 140:
        raise ValueError("logical policy grid is incomplete")
    return result


def _aulc(means: dict, algorithm: str, seed: int, metric: str) -> float:
    values = [means[(algorithm, seed, events)][metric] for events in BUDGETS]
    return float(np.trapz(values, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def _algorithm_aulcs(means: dict, algorithm: str, metric: str) -> np.ndarray:
    return np.asarray([_aulc(means, algorithm, seed, metric) for seed in SEEDS])


def _scale_aulc(rows: list[dict], algorithm: str, scale: float) -> float:
    points = []
    for events in BUDGETS:
        values = [
            row["primary"]
            for row in rows
            if row["algorithm"] == algorithm
            and row["events"] == events
            and math.isclose(row["scale"], scale)
        ]
        if len(values) != 15:
            raise ValueError("v10 demand-scale AULC coverage drifted")
        points.append(float(np.mean(values)))
    return float(np.trapz(points, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def _comparison(means: dict, candidate: str, control: str) -> dict:
    candidate_values = _algorithm_aulcs(means, candidate, "primary")
    control_values = _algorithm_aulcs(means, control, "primary")
    seed_wins = int(np.sum(candidate_values < control_values))
    return {
        "candidate": candidate,
        "control": control,
        "candidate_mean_aulc": float(candidate_values.mean()),
        "control_mean_aulc": float(control_values.mean()),
        "relative_improvement": float(
            (control_values.mean() - candidate_values.mean()) / control_values.mean()
        ),
        "seed_wins": seed_wins,
        "mean_lower": bool(candidate_values.mean() < control_values.mean()),
        "per_seed": [
            {
                "seed": seed,
                "candidate_aulc": float(candidate_values[index]),
                "control_aulc": float(control_values[index]),
                "candidate_lower": bool(candidate_values[index] < control_values[index]),
            }
            for index, seed in enumerate(SEEDS)
        ],
    }


def main() -> int:
    args = parse_args()
    registration_path, registration = _load_control(args.registration, REGISTRATION_SCHEMA)
    protocol_path, protocol = _load_control(args.protocol, PROTOCOL_SCHEMA)
    closeout_path, closeout = _load_control(args.branch_closeout, CLOSEOUT_SCHEMA)
    verification_path, verification = _load_control(
        args.branch_verification, VERIFICATION_SCHEMA
    )
    methods_path, methods_payload, aliases_by_physical = _load_methods(args.methods_manifest)
    methods_payload = dict(methods_payload)
    methods_payload["manifest_path"] = str(methods_path)
    protocol_hash = sha256(protocol_path)
    if (
        registration.get("study_stage") != "development"
        or registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("development_evaluation", {}).get("expected_unique_actor_tensors") != 60
        or registration.get("development_evaluation", {}).get("expected_unique_remote_sumo_episodes") != 540
        or protocol.get("formal_held_out_outcomes_read") is not False
        or methods_payload.get("v10_protocol_sha256") != protocol_hash
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
    ):
        raise ValueError("v10 immutable analysis controls drifted")
    rows, matrix_evidence = _load_rows(
        args.matrix_root, methods_payload, aliases_by_physical, protocol_hash
    )
    means = _cell_means(rows)
    thresholds = registration["screening_gates"]
    combined = "h2oplus-v10-delayed-combined"
    versus_sac = _comparison(means, combined, "sac-online")
    versus_fixed = _comparison(means, combined, "h2oplus")

    budget_results = {}
    for events in (10_000, 25_000):
        values = {
            algorithm: float(
                np.mean([means[(algorithm, seed, events)]["primary"] for seed in SEEDS])
            )
            for algorithm in ALGORITHMS
        }
        budget_results[str(events)] = {
            "means": values,
            "combined_lower_than_sac": values[combined] < values["sac-online"],
            "combined_lower_than_fixed_h2o": values[combined] < values["h2oplus"],
        }

    completion = {}
    combined_completion = float(_algorithm_aulcs(means, combined, "completion").mean())
    for control in ("sac-online", "h2oplus"):
        control_value = float(_algorithm_aulcs(means, control, "completion").mean())
        completion[control] = {
            "combined_aulc": combined_completion,
            "control_aulc": control_value,
            "delta": combined_completion - control_value,
        }

    scale_rows = []
    for scale in (0.85, 1.0, 1.15):
        candidate = _scale_aulc(rows, combined, scale)
        control = _scale_aulc(rows, "sac-online", scale)
        scale_rows.append(
            {
                "demand_scale": scale,
                "delayed_combined_aulc": candidate,
                "sac_online_aulc": control,
                "relative_degradation": (candidate - control) / control,
            }
        )

    factor_rows = []
    for factor in SINGLE_FACTORS:
        comparison = _comparison(means, factor, "h2oplus-v10-unit-control")
        comparison["attribution_positive"] = bool(
            comparison["mean_lower"]
            and comparison["seed_wins"] >= SINGLE_FACTOR_MINIMUM_SEED_WINS
        )
        factor_rows.append(comparison)
    single_factor_attribution = {
        "operationalization_frozen_before_outcomes": True,
        "required_complete_contrasts": list(SINGLE_FACTORS),
        "minimum_seed_wins_for_positive_factor": SINGLE_FACTOR_MINIMUM_SEED_WINS,
        "contrasts": factor_rows,
        "passed": any(row["attribution_positive"] for row in factor_rows),
    }

    stability_passed = closeout.get("all_30_training_runs_stability_gate_passed") is True
    checks = {
        "all_30_training_runs_stability": stability_passed,
        "all_branches_share_exact_seed_matched_10k_state": closeout.get(
            "exact_seed_matched_handoffs"
        )
        == 25,
        "combined_lower_than_sac_at_10000": budget_results["10000"][
            "combined_lower_than_sac"
        ],
        "combined_lower_than_sac_at_25000": budget_results["25000"][
            "combined_lower_than_sac"
        ],
        "combined_relative_aulc_improvement_over_sac": versus_sac[
            "relative_improvement"
        ]
        >= float(thresholds["delayed_combined_relative_aulc_improvement_over_sac_at_least"]),
        "combined_relative_aulc_improvement_over_fixed_h2o": versus_fixed[
            "relative_improvement"
        ]
        >= float(
            thresholds[
                "delayed_combined_relative_aulc_improvement_over_fixed_h2o_at_least"
            ]
        ),
        "combined_seed_wins_over_sac": versus_sac["seed_wins"]
        >= int(thresholds["delayed_combined_seed_wins_over_sac_at_least"]),
        "combined_seed_wins_over_fixed_h2o": versus_fixed["seed_wins"]
        >= int(thresholds["delayed_combined_seed_wins_over_fixed_h2o_at_least"]),
        "completion_rate_noninferiority": all(
            item["delta"] >= -float(thresholds["completion_rate_noninferiority_margin"])
            for item in completion.values()
        ),
        "demand_scale_robustness": all(
            item["relative_degradation"]
            <= float(
                thresholds[
                    "no_demand_scale_relative_aulc_degradation_over_sac_greater_than"
                ]
            )
            for item in scale_rows
        ),
        "single_factor_attribution": single_factor_attribution["passed"],
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 540,
            "logical_evaluation_rows": len(rows),
            "logical_policy_budget_aliases": 140,
            "matrix_blocks": len(matrix_evidence),
        },
        "training_scientific_stability": {
            "passed": stability_passed,
            "failure_runs": closeout.get("stability_failure_runs", []),
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "screening_gate_passed": gate_passed,
            "checks": checks,
            "comparison_over_sac": versus_sac,
            "comparison_over_fixed_h2o": versus_fixed,
            "budget_results": budget_results,
            "completion_aulc": completion,
            "demand_scale_aulc": scale_rows,
            "single_factor_attribution": single_factor_attribution,
            "all_algorithm_mean_aulc": {
                algorithm: float(_algorithm_aulcs(means, algorithm, "primary").mean())
                for algorithm in ALGORITHMS
            },
        },
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
        "methods_manifest": {"path": str(methods_path), "sha256": sha256(methods_path)},
        "branch_closeout": {"path": str(closeout_path), "sha256": sha256(closeout_path)},
        "branch_verification": {
            "path": str(verification_path),
            "sha256": sha256(verification_path),
        },
        "matrix_evidence": matrix_evidence,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": gate_passed,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(report["development_scientific_result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
