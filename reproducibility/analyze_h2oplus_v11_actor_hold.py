#!/usr/bin/env python3
"""Audit and analyze the preregistered v11 actor-hold development screen."""

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


REPORT_SCHEMA = "transit-h2oplus-v11-actor-hold-analysis-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v11-actor-hold-v1"
REGISTRATION_SCHEMA = "transit-h2oplus-v11-actor-hold-registration-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v11-actor-hold-protocol-v2"
CLOSEOUT_SCHEMA = "transit-h2oplus-v11-actor-hold-training-closeout-v2"
VERIFICATION_SCHEMA = (
    "transit-h2oplus-v11-actor-hold-training-closeout-verification-v2"
)
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-delayed-actor|"
    r"h2oplus-v11-actor-hold)-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SEEDS = tuple(range(6201, 6206))
BUDGETS = (0, 5_000, 10_000, 25_000)
SCALES = (0.85, 1.0, 1.15)
ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
)
CANDIDATE = "h2oplus-v11-actor-hold"
CONTROLS = ("sac-online", "h2oplus", "h2oplus-v10-delayed-actor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-closeout", type=Path, required=True)
    parser.add_argument("--training-verification", type=Path, required=True)
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
        or payload.get("physical_method_count") != 45
        or payload.get("logical_method_budget_alias_count") != 80
        or payload.get("expected_remote_sumo_episodes") != 405
        or len(methods) != 45
        or len(aliases) != 80
    ):
        raise ValueError("v11 methods manifest scope drifted")
    physical = {row.get("name"): row for row in methods}
    if len(physical) != 45:
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
            raise ValueError(f"malformed v11 logical alias: {alias!r}")
        if alias["algorithm"] == CANDIDATE:
            if (
                not isinstance(alias.get("source_checkpoint_sha256"), str)
                or len(alias["source_checkpoint_sha256"]) != 64
                or not isinstance(alias.get("source_temperature_tensor_sha256"), str)
                or len(alias["source_temperature_tensor_sha256"]) != 64
                or alias.get("source_artifact_lineage") != "v11-actor-hold"
            ):
                raise ValueError(f"candidate alias source evidence drifted: {alias!r}")
        logical_names.add(alias["name"])
        by_physical[physical_name].append(alias)
    expected = {
        f"{algorithm}-seed{seed}-events{events:09d}"
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for events in BUDGETS
    }
    identities = payload.get("identity_evidence", [])
    if (
        logical_names != expected
        or any(not rows for rows in by_physical.values())
        or len(identities) != 5
        or any(
            row.get("logical_aliases") != 16
            or row.get("physical_actor_tensors") != 9
            or row.get("v11_zero_5k_10k_equal_v8_h2oplus_zero") is not True
            or row.get("v11_zero_5k_10k_temperature_equal") is not True
            for row in identities
        )
    ):
        raise ValueError("v11 logical alias grid or hold identity is incomplete")
    return resolved, payload, by_physical


def _validate_training_metadata(
    payload: dict,
    method: dict,
    v10_protocol_hash: str,
    v11_protocol_hash: str,
) -> None:
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
    lineage = method["artifact_lineage"]
    if lineage == "v8-fewshot":
        specific = {
            "method": method["algorithm"],
            "fewshot_online_events_nominal": method["online_events"],
            "fewshot_online_events_actual": method["actual_online_events"],
            "fewshot_learner_update_steps": method["learner_update_steps"],
            "fewshot_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        artifact_protocol = training.get("fewshot_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v8-fewshot-development-v1"
    elif lineage == "v10-delayed":
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
    elif lineage == "v11-actor-hold":
        specific = {
            "method": "h2oplus",
            "variant": "actor-hold-online-only",
            "stage": "candidate",
            "v11_online_events_nominal": method["online_events"],
            "v11_online_events_actual": method["actual_online_events"],
            "v11_learner_update_steps": method["learner_update_steps"],
            "v11_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        artifact_protocol = training.get("v11_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v11-actor-hold-development-v2"
        intervention = training.get("v11_intervention", {})
        required = {
            "variant": "actor-hold-online-only",
            "unit_ratio_weights": True,
            "checkpoint_selection_by_behavior": False,
        }
        if any(intervention.get(key) != value for key, value in required.items()):
            raise ValueError("v11 result intervention provenance drifted")
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
        or lineage == "v10-delayed"
        and artifact_protocol != v10_protocol_hash
        or lineage == "v11-actor-hold"
        and artifact_protocol != v11_protocol_hash
    ):
        raise ValueError(f"evaluation result policy provenance drifted: {mismatches}")


def _load_rows(
    roots: list[Path],
    methods_payload: dict,
    aliases_by_physical: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    if len(roots) != 3:
        raise ValueError("v11 analysis requires exactly three merged development blocks")
    physical_methods = {row["name"]: row for row in methods_payload["methods"]}
    v10_protocol_hash = methods_payload["v10_protocol_sha256"]
    v11_protocol_hash = methods_payload["v11_protocol_sha256"]
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
            or len(jobs) != 135
            or not isinstance(source_hashes, dict)
        ):
            raise ValueError(f"v11 matrix block is incomplete: {root}")
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
                raise ValueError(f"v11 physical result binding failed: {output}")
            _validate_training_metadata(
                payload, method, v10_protocol_hash, v11_protocol_hash
            )
            if any(
                abs(float(value)) > 1e-9
                for value in payload.get("passenger_welfare", {})
                .get("chronological_coverage_errors", {})
                .values()
            ):
                raise ValueError(f"v11 welfare ledger failed: {output}")
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
                    raise ValueError(f"duplicate v11 logical result: {key}")
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
    if len(rows) != 720:
        raise ValueError(f"v11 analysis requires 720 logical rows, got {len(rows)}")
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
    if len(result) != 80:
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
            raise ValueError("v11 demand-scale AULC coverage drifted")
        points.append(float(np.mean(values)))
    return float(np.trapz(points, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def _comparison(means: dict, control: str) -> dict:
    candidate_values = _algorithm_aulcs(means, CANDIDATE, "primary")
    control_values = _algorithm_aulcs(means, control, "primary")
    return {
        "candidate": CANDIDATE,
        "control": control,
        "candidate_mean_aulc": float(candidate_values.mean()),
        "control_mean_aulc": float(control_values.mean()),
        "relative_improvement": float(
            (control_values.mean() - candidate_values.mean()) / control_values.mean()
        ),
        "seed_wins": int(np.sum(candidate_values < control_values)),
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
    closeout_path, closeout = _load_control(args.training_closeout, CLOSEOUT_SCHEMA)
    verification_path, verification = _load_control(
        args.training_verification, VERIFICATION_SCHEMA
    )
    methods_path, methods_payload, aliases_by_physical = _load_methods(
        args.methods_manifest
    )
    methods_payload = dict(methods_payload)
    methods_payload["manifest_path"] = str(methods_path)
    protocol_hash = sha256(protocol_path)
    development = registration.get("development_evaluation", {})
    if (
        registration.get("study_stage") != "development"
        or registration.get("formal_held_out_outcomes_read") is not False
        or development.get("logical_policy_aliases") != 80
        or development.get("maximum_unique_remote_sumo_episodes") != 720
        or protocol.get("formal_held_out_outcomes_read") is not False
        or methods_payload.get("v11_protocol_sha256") != protocol_hash
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("all_5_training_runs_stability_gate_passed") is not True
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
        or verification.get("artifact_integrity_verified") is not True
    ):
        raise ValueError("v11 immutable analysis controls drifted")
    rows, matrix_evidence = _load_rows(
        args.matrix_root, methods_payload, aliases_by_physical
    )
    means = _cell_means(rows)
    thresholds = registration["screening_gates"]
    comparisons = {control: _comparison(means, control) for control in CONTROLS}

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
            "candidate_lower_than_sac": values[CANDIDATE] < values["sac-online"],
            "candidate_lower_than_fixed_h2o": values[CANDIDATE] < values["h2oplus"],
        }

    completion = {}
    candidate_completion = float(_algorithm_aulcs(means, CANDIDATE, "completion").mean())
    for control in CONTROLS:
        control_value = float(_algorithm_aulcs(means, control, "completion").mean())
        completion[control] = {
            "candidate_aulc": candidate_completion,
            "control_aulc": control_value,
            "delta": candidate_completion - control_value,
        }

    scale_rows = []
    for scale in SCALES:
        candidate = _scale_aulc(rows, CANDIDATE, scale)
        control = _scale_aulc(rows, "sac-online", scale)
        scale_rows.append(
            {
                "demand_scale": scale,
                "candidate_aulc": candidate,
                "sac_online_aulc": control,
                "relative_degradation": (candidate - control) / control,
            }
        )

    hold_identity_passed = all(
        item.get("v11_zero_5k_10k_equal_v8_h2oplus_zero") is True
        and item.get("v11_zero_5k_10k_temperature_equal") is True
        for item in methods_payload["identity_evidence"]
    )
    checks = {
        "all_5_training_runs_stability": closeout.get(
            "all_5_training_runs_stability_gate_passed"
        )
        is True,
        "zero_5k_10k_actor_and_temperature_hashes_identical": hold_identity_passed,
        "candidate_lower_than_sac_at_10000": budget_results["10000"][
            "candidate_lower_than_sac"
        ],
        "candidate_lower_than_sac_at_25000": budget_results["25000"][
            "candidate_lower_than_sac"
        ],
        "candidate_lower_than_fixed_h2o_at_10000": budget_results["10000"][
            "candidate_lower_than_fixed_h2o"
        ],
        "candidate_lower_than_fixed_h2o_at_25000": budget_results["25000"][
            "candidate_lower_than_fixed_h2o"
        ],
        "relative_aulc_improvement_over_sac": comparisons["sac-online"][
            "relative_improvement"
        ]
        >= float(thresholds["candidate_relative_aulc_improvement_over_sac_at_least"]),
        "relative_aulc_improvement_over_fixed_h2o": comparisons["h2oplus"][
            "relative_improvement"
        ]
        >= float(
            thresholds["candidate_relative_aulc_improvement_over_fixed_h2o_at_least"]
        ),
        "relative_aulc_improvement_over_v10_delayed_actor": comparisons[
            "h2oplus-v10-delayed-actor"
        ]["relative_improvement"]
        >= float(
            thresholds[
                "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least"
            ]
        ),
        "seed_wins_over_sac": comparisons["sac-online"]["seed_wins"]
        >= int(thresholds["candidate_seed_wins_over_sac_at_least"]),
        "seed_wins_over_fixed_h2o": comparisons["h2oplus"]["seed_wins"]
        >= int(thresholds["candidate_seed_wins_over_fixed_h2o_at_least"]),
        "seed_wins_over_v10_delayed_actor": comparisons[
            "h2oplus-v10-delayed-actor"
        ]["seed_wins"]
        >= int(thresholds["candidate_seed_wins_over_v10_delayed_actor_at_least"]),
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
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 405,
            "logical_evaluation_rows": len(rows),
            "logical_policy_budget_aliases": 80,
            "matrix_blocks": len(matrix_evidence),
        },
        "training_scientific_stability": {
            "passed": True,
            "failure_runs": [],
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "screening_gate_passed": gate_passed,
            "checks": checks,
            "comparisons": comparisons,
            "budget_results": budget_results,
            "completion_aulc": completion,
            "demand_scale_aulc": scale_rows,
            "all_algorithm_mean_aulc": {
                algorithm: float(_algorithm_aulcs(means, algorithm, "primary").mean())
                for algorithm in ALGORITHMS
            },
        },
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
        "methods_manifest": {"path": str(methods_path), "sha256": sha256(methods_path)},
        "training_closeout": {"path": str(closeout_path), "sha256": sha256(closeout_path)},
        "training_verification": {
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
