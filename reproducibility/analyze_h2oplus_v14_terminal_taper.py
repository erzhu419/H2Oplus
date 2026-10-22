#!/usr/bin/env python3
"""Audit and analyze the preregistered v14 terminal-taper development screen."""

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


REPORT_SCHEMA = "transit-h2oplus-v14-terminal-taper-analysis-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v14-terminal-taper-v1"
REGISTRATION_SCHEMA = "transit-h2oplus-v14-terminal-taper-registration-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v14-terminal-taper-protocol-v1"
CLOSEOUT_SCHEMA = "transit-h2oplus-v14-terminal-taper-branch-closeout-v1"
VERIFICATION_SCHEMA = (
    "transit-h2oplus-v14-terminal-taper-branch-closeout-verification-v1"
)
EXPECTED_SHA256 = {
    "registration": "6be53642facea16013dc90f84f60a1d861c112e08a9b748a61011eb985dd2a41",
    "protocol": "ba275fc4c134feadb564b49548699571ca5bf6bfd1aee529759d6ce791df12c5",
    "closeout": "5d609f76f6d139dc8982a0ce4f8c9e95a4f0e5f4c4d9655b3545fa15270d2ffc",
    "verification": "56cf282615364dc1b93bbf118cd1f3101b20fa2398ca87e08b202322c833c15b",
    "methods": "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b",
}
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-delayed-actor|"
    r"h2oplus-v11-actor-hold|h2oplus-v12-actor-ramp|"
    r"h2oplus-v13-proximal-strong|"
    r"h2oplus-v14-terminal-(?:half|quarter|zero))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SEEDS = tuple(range(6201, 6206))
BUDGETS = (0, 5_000, 10_000, 25_000)
SCALES = (0.85, 1.0, 1.15)
BLOCKS = tuple(f"v12-staged-release-development-0{index}" for index in range(1, 4))
ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
    "h2oplus-v14-terminal-half",
    "h2oplus-v14-terminal-quarter",
    "h2oplus-v14-terminal-zero",
)
CANDIDATE = "h2oplus-v14-terminal-quarter"
CONTROLS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
)
DIAGNOSTICS = (
    "h2oplus-v14-terminal-half",
    "h2oplus-v14-terminal-zero",
)
TERMINAL_MULTIPLIERS = {
    "terminal-half": 0.5,
    "terminal-quarter": 0.25,
    "terminal-zero": 0.0,
}
FULL_RATE_EQUIVALENT_UPDATES = {
    "terminal-half": 10249.449999999913,
    "terminal-quarter": 8999.449999999915,
    "terminal-zero": 7749.449999999926,
}


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


def _load_control(path: Path, expected_sha: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected_sha:
        raise ValueError(f"frozen identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema drifted: {resolved}")
    return resolved, payload


def _load_methods(path: Path) -> tuple[Path, dict, dict[str, list[dict]]]:
    resolved, payload = _load_control(path, EXPECTED_SHA256["methods"], METHODS_SCHEMA)
    methods = payload.get("methods", [])
    aliases = payload.get("aliases", [])
    if (
        payload.get("study_stage") != "development"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 70
        or payload.get("logical_method_budget_alias_count") != 180
        or payload.get("expected_remote_sumo_episodes") != 630
        or payload.get("registered_maximum_remote_sumo_episodes") != 1620
        or payload.get("predesignated_candidate") != CANDIDATE
        or tuple(payload.get("control_algorithms", [])) != CONTROLS
        or tuple(payload.get("diagnostic_variants", [])) != DIAGNOSTICS
        or len(methods) != 70
        or len(aliases) != 180
    ):
        raise ValueError("v14 methods manifest scope drifted")
    physical = {row.get("name"): row for row in methods}
    if len(physical) != 70:
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
            raise ValueError(f"malformed v14 logical alias: {alias!r}")
        if alias["algorithm"].startswith("h2oplus-v14-"):
            variant = alias["algorithm"][len("h2oplus-v14-") :]
            if (
                alias.get("source_artifact_lineage") != "v14-terminal-taper"
                or alias.get("source_variant") != variant
                or alias.get("predesignated_candidate")
                is not (alias["algorithm"] == CANDIDATE)
                or not isinstance(alias.get("source_checkpoint_sha256"), str)
                or len(alias["source_checkpoint_sha256"]) != 64
                or not isinstance(alias.get("source_temperature_tensor_sha256"), str)
                or len(alias["source_temperature_tensor_sha256"]) != 64
            ):
                raise ValueError(f"v14 alias source evidence drifted: {alias!r}")
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
            row.get("v14_logical_aliases") != 12
            or row.get("prefix_zero_5k_10k_equal_fixed_h2oplus_zero") is not True
            or row.get("exact_seed_matched_v12_10k_handoff") is not True
            for row in identities
        )
    ):
        raise ValueError("v14 logical alias grid or prefix identity is incomplete")
    return resolved, payload, by_physical


def _validate_training_metadata(
    payload: dict,
    method: dict,
    protocol_hashes: dict[str, str],
) -> None:
    training = payload.get("policy", {}).get("training_metadata", {})
    common = {
        "training_seed": method["training_seed"],
        "run_signature": method["training_run_signature"],
        "training_checkpoint_sha256": method["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    expected: dict[str, Any]
    lineage = method["artifact_lineage"]
    admission = training.get("training_admission", {})
    intervention: dict[str, Any] = {}
    if lineage == "v8-fewshot":
        expected = {
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
        expected = {
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
        expected = {
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
        if any(
            intervention.get(key) != value
            for key, value in {
                "variant": "actor-hold-online-only",
                "unit_ratio_weights": True,
                "checkpoint_selection_by_behavior": False,
            }.items()
        ):
            raise ValueError("v11 result intervention provenance drifted")
    elif lineage == "v12-staged-release":
        variant = method["algorithm"][len("h2oplus-v12-") :]
        expected = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v12_online_events_nominal": method["online_events"],
            "v12_online_events_actual": method["actual_online_events"],
            "v12_learner_update_steps": method["learner_update_steps"],
            "v12_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v12_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v12_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        artifact_protocol = training.get("v12_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v12-staged-release-development-v2"
        intervention = training.get("v12_intervention", {})
        if (
            variant != "actor-ramp"
            or intervention.get("variant") != variant
            or intervention.get("actor_learning_rate_ramp") is not True
            or intervention.get("temperature_learning_rate_ramp") is not False
            or intervention.get("actor_state_anchor") is not False
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError("v12 actor-ramp provenance drifted")
    elif lineage == "v13-proximal-release":
        variant = method["algorithm"][len("h2oplus-v13-") :]
        rate = 5e-5
        expected = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v13_online_events_nominal": method["online_events"],
            "v13_online_events_actual": method["actual_online_events"],
            "v13_learner_update_steps": method["learner_update_steps"],
            "v13_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v13_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v13_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        artifact_protocol = training.get("v13_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v13-proximal-release-development-v2"
        intervention = training.get("v13_intervention", {})
        retention = float(intervention.get("no_gradient_anchor_retention_at_25k", math.nan))
        if (
            variant != "proximal-strong"
            or intervention.get("variant") != variant
            or intervention.get("proximal_rate_per_actor_update") != rate
            or intervention.get("registered_post_10k_actor_updates") != 15_000
            or not math.isclose(retention, (1.0 - rate) ** 15_000, rel_tol=0.0, abs_tol=1e-15)
            or intervention.get("predesignated_candidate") is not False
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError("v13 strong-control provenance drifted")
    elif lineage == "v14-terminal-taper":
        variant = method["algorithm"][len("h2oplus-v14-") :]
        multiplier = TERMINAL_MULTIPLIERS[variant]
        expected = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v14_online_events_nominal": method["online_events"],
            "v14_online_events_actual": method["actual_online_events"],
            "v14_learner_update_steps": method["learner_update_steps"],
            "v14_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v14_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v14_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        artifact_protocol = training.get("v14_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v14-terminal-taper-development-v1"
        intervention = training.get("v14_intervention", {})
        retention = float(intervention.get("no_gradient_anchor_retention_at_25k", math.nan))
        if (
            intervention.get("variant") != variant
            or intervention.get("proximal_rate_per_actor_update") != 5e-5
            or intervention.get("registered_post_10k_actor_updates") != 15_000
            or not math.isclose(retention, (1.0 - 5e-5) ** 15_000, rel_tol=0.0, abs_tol=1e-15)
            or intervention.get("predesignated_candidate") is not (method["algorithm"] == CANDIDATE)
            or intervention.get("terminal_actor_lr_taper_events") != [15_000, 25_000]
            or intervention.get("terminal_actor_lr_taper_critic_updates")
            != [10_000, 19_999]
            or intervention.get("terminal_actor_lr_multiplier") != multiplier
            or not math.isclose(
                float(intervention.get("full_rate_equivalent_actor_updates", math.nan)),
                FULL_RATE_EQUIVALENT_UPDATES[variant],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError("v14 result intervention provenance drifted")
    else:
        raise ValueError(f"unsupported physical actor lineage: {lineage}")
    mismatches = {
        key: (training.get(key), value)
        for key, value in {**common, **expected}.items()
        if training.get(key) != value
    }
    if (
        mismatches
        or admission.get("schema_version") != admission_schema
        or admission.get("sha256") != method["training_admission_sha256"]
        or artifact_protocol != method["source_protocol_sha256"]
        or lineage in protocol_hashes
        and artifact_protocol != protocol_hashes[lineage]
    ):
        raise ValueError(f"evaluation result policy provenance drifted: {mismatches}")


def _load_rows(
    roots: list[Path],
    methods_path: Path,
    methods_payload: dict,
    aliases_by_physical: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    if len(roots) != 3:
        raise ValueError("v14 analysis requires exactly three merged development blocks")
    physical_methods = {row["name"]: row for row in methods_payload["methods"]}
    protocol_hashes = {
        "v10-delayed": methods_payload["v10_protocol_sha256"],
        "v11-actor-hold": methods_payload["v11_protocol_sha256"],
        "v12-staged-release": methods_payload["v12_protocol_sha256"],
        "v13-proximal-release": methods_payload["v13_protocol_sha256"],
        "v14-terminal-taper": methods_payload["v14_protocol_sha256"],
    }
    rows, manifests, seen, blocks = [], [], set(), set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256")
        block_names = {str(job.get("environment_block")) for job in jobs}
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != sha256(methods_path)
            or len(jobs) != 210
            or len(block_names) != 1
            or not isinstance(source_hashes, dict)
            or len(source_hashes) != 47
        ):
            raise ValueError(f"v14 matrix block is incomplete: {root}")
        block = next(iter(block_names))
        if block not in BLOCKS or block in blocks:
            raise ValueError(f"unexpected or duplicate development block: {block}")
        blocks.add(block)
        manifests.append(
            {
                "block": block,
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
            scale = float(job["scenario"]["demand_scale"])
            signature = job.get("signature")
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != signature
                or payload.get("method") != physical_name
                or payload.get("source_sha256") != source_hashes
                or scale not in SCALES
                or signature in seen
            ):
                raise ValueError(f"v14 physical result binding failed: {output}")
            seen.add(signature)
            _validate_training_metadata(payload, method, protocol_hashes)
            if any(
                abs(float(value)) > 1e-9
                for value in payload.get("passenger_welfare", {})
                .get("chronological_coverage_errors", {})
                .values()
            ):
                raise ValueError(f"v14 welfare ledger failed: {output}")
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
                    raise ValueError(f"duplicate v14 logical result: {key}")
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
    if blocks != set(BLOCKS) or len(rows) != 1620:
        raise ValueError(f"v14 analysis grid is incomplete: {len(rows)} logical rows")
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
    if len(result) != 180:
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
            raise ValueError("v14 demand-scale AULC coverage drifted")
        points.append(float(np.mean(values)))
    return float(np.trapz(points, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def _comparison(means: dict, other: str) -> dict:
    candidate_values = _algorithm_aulcs(means, CANDIDATE, "primary")
    other_values = _algorithm_aulcs(means, other, "primary")
    return {
        "candidate": CANDIDATE,
        "other": other,
        "candidate_mean_aulc": float(candidate_values.mean()),
        "other_mean_aulc": float(other_values.mean()),
        "relative_improvement": float(
            (other_values.mean() - candidate_values.mean()) / other_values.mean()
        ),
        "seed_wins": int(np.sum(candidate_values < other_values)),
        "mean_lower": bool(candidate_values.mean() < other_values.mean()),
        "per_seed": [
            {
                "seed": seed,
                "candidate_aulc": float(candidate_values[index]),
                "other_aulc": float(other_values[index]),
                "candidate_lower": bool(candidate_values[index] < other_values[index]),
            }
            for index, seed in enumerate(SEEDS)
        ],
    }


def main() -> int:
    args = parse_args()
    registration_path, registration = _load_control(
        args.registration, EXPECTED_SHA256["registration"], REGISTRATION_SCHEMA
    )
    protocol_path, protocol = _load_control(
        args.protocol, EXPECTED_SHA256["protocol"], PROTOCOL_SCHEMA
    )
    closeout_path, closeout = _load_control(
        args.training_closeout, EXPECTED_SHA256["closeout"], CLOSEOUT_SCHEMA
    )
    verification_path, verification = _load_control(
        args.training_verification, EXPECTED_SHA256["verification"], VERIFICATION_SCHEMA
    )
    methods_path, methods_payload, aliases_by_physical = _load_methods(
        args.methods_manifest
    )
    if (
        registration.get("study_stage") != "development"
        or registration.get("predesignated_candidate") != "terminal-quarter"
        or registration.get("new_training_runs") != 15
        or registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("development_evaluation_submitted") is not False
        or registration.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or methods_payload.get("v14_protocol_sha256") != sha256(protocol_path)
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_15_branches") is not True
        or closeout.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or closeout.get("development_evaluation_gate_open") is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
        or verification.get("all_15_scientific_stability_passed") is not True
        or verification.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or verification.get("development_evaluation_gate_open") is not True
        or verification.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v14 immutable analysis controls drifted")
    rows, matrix_evidence = _load_rows(
        args.matrix_root, methods_path, methods_payload, aliases_by_physical
    )
    means = _cell_means(rows)
    thresholds = protocol["analysis_gates"]
    comparisons = {control: _comparison(means, control) for control in CONTROLS}
    diagnostic_comparisons = {
        variant: _comparison(means, variant) for variant in DIAGNOSTICS
    }

    budget_means = {
        algorithm: float(
            np.mean([means[(algorithm, seed, 25_000)]["primary"] for seed in SEEDS])
        )
        for algorithm in ALGORITHMS
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
    diagnostic_completion = {}
    for variant in DIAGNOSTICS:
        value = float(_algorithm_aulcs(means, variant, "completion").mean())
        diagnostic_completion[variant] = {
            "candidate_aulc": candidate_completion,
            "diagnostic_aulc": value,
            "delta": candidate_completion - value,
        }

    scale_rows = []
    for scale in SCALES:
        candidate = _scale_aulc(rows, CANDIDATE, scale)
        sac = _scale_aulc(rows, "sac-online", scale)
        scale_rows.append(
            {
                "demand_scale": scale,
                "candidate_aulc": candidate,
                "sac_online_aulc": sac,
                "relative_degradation": (candidate - sac) / sac,
            }
        )

    relative_thresholds = {
        "h2oplus": "candidate_relative_aulc_improvement_over_fixed_h2o_at_least",
        "sac-online": "candidate_relative_aulc_improvement_over_sac_at_least",
        "h2oplus-v10-delayed-actor": (
            "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least"
        ),
        "h2oplus-v11-actor-hold": (
            "candidate_relative_aulc_improvement_over_v11_at_least"
        ),
        "h2oplus-v12-actor-ramp": (
            "candidate_relative_aulc_improvement_over_v12_actor_ramp_at_least"
        ),
        "h2oplus-v13-proximal-strong": (
            "candidate_relative_aulc_improvement_over_v13_strong_at_least"
        ),
    }
    win_thresholds = {
        "h2oplus": "candidate_seed_wins_over_fixed_h2o_at_least",
        "sac-online": "candidate_seed_wins_over_sac_at_least",
        "h2oplus-v10-delayed-actor": "candidate_seed_wins_over_v10_delayed_actor_at_least",
        "h2oplus-v11-actor-hold": "candidate_seed_wins_over_v11_at_least",
        "h2oplus-v12-actor-ramp": "candidate_seed_wins_over_v12_actor_ramp_at_least",
        "h2oplus-v13-proximal-strong": "candidate_seed_wins_over_v13_strong_at_least",
    }
    checks = {
        "all_15_training_runs_stability": closeout.get(
            "scientific_stability_passed_for_all_15_branches"
        )
        is True,
        "all_branches_share_exact_seed_matched_v12_10k_state": closeout.get(
            "exact_seed_matched_v12_10k_handoffs_passed"
        )
        is True,
        "candidate_mean_lower_than_all_controls_at_25000": all(
            budget_means[CANDIDATE] < budget_means[control] for control in CONTROLS
        ),
        "relative_aulc_improvement_over_all_controls": all(
            comparisons[control]["relative_improvement"]
            >= float(thresholds[key])
            for control, key in relative_thresholds.items()
        ),
        "seed_wins_over_all_controls": all(
            comparisons[control]["seed_wins"] >= int(thresholds[key])
            for control, key in win_thresholds.items()
        ),
        "completion_rate_noninferiority_against_all_controls": all(
            item["delta"] >= -float(thresholds["completion_rate_noninferiority_margin"])
            for item in completion.values()
        ),
        "demand_scale_robustness_against_sac": all(
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
            "physical_evaluation_results": 630,
            "logical_evaluation_rows": len(rows),
            "logical_policy_budget_aliases": 180,
            "matrix_blocks": len(matrix_evidence),
        },
        "training_scientific_stability": {
            "passed": True,
            "registered_runs": 15,
            "failure_runs": [],
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "screening_gate_passed": gate_passed,
            "predesignated_candidate": CANDIDATE,
            "checks": checks,
            "control_comparisons": comparisons,
            "mean_primary_at_25000": budget_means,
            "completion_aulc_against_controls": completion,
            "demand_scale_aulc_against_sac": scale_rows,
            "all_algorithm_mean_aulc": {
                algorithm: float(_algorithm_aulcs(means, algorithm, "primary").mean())
                for algorithm in ALGORITHMS
            },
        },
        "diagnostic_results_not_used_as_selection_gates": {
            "comparisons": diagnostic_comparisons,
            "completion_aulc": diagnostic_completion,
        },
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
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
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(report["development_scientific_result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
