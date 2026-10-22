#!/usr/bin/env python3
"""Audit data and runtime factors that can affect H2O+ convergence or validity.

This is a local, read-only audit. It does not import SUMO, execute a simulator,
train a policy, or inspect formal held-out evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DATASET = ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
SUMMARY = DATASET.with_suffix(".h5.summary.json")
DATA_AUDIT = DATASET.with_suffix(".audit.json")
CREDIT_AUDIT = DATASET.with_suffix(".credit_audit.json")
REWARD_AUDIT = DATASET.parent / "reward_conditioning_v7_formal_v2.json"
ENVIRONMENT = REPRO / "h2oplus_v12_staged_release_environment_v2.json"
RUNTIME_LOCK = REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
V16_ANALYSIS = REPRO / "h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json"
V17_INCIDENT = (
    REPRO
    / "h2oplus_v17_tail_service_reserve_results/evaluation_incidents"
    / "v1_python38_removeprefix/incident_closeout_v1.json"
)
V14_BRANCH_ROOT = (
    REPRO / "h2oplus_v14_terminal_taper_results/branches_v1"
)
V14_SUMMARIES = tuple(
    V14_BRANCH_ROOT
    / f"h2oplus_v14_terminal_quarter_seed{seed}/summary.json"
    for seed in range(6201, 6206)
)
OUTPUT = REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json"

RAW_PREFIX_WIDTH = 5
RAW_SLOT_WIDTH = 15
RAW_CURRENT_DECISION_INDEX = 2
RAW_DECISION_TIME_SECONDS = 0
LOCAL_EVENT_INDEX = {
    "line_id": 0,
    "station_index": 1,
    "forward_headway_seconds": 2,
    "backward_headway_seconds": 3,
    "target_forward_headway_seconds": 4,
    "target_backward_headway_seconds": 5,
    "eligible_stranded_passengers": 6,
    "onboard_passengers": 7,
    "eligible_arrival_rate_per_second": 8,
    "base_dwell_seconds": 9,
    "forward_bus_present": 10,
    "backward_bus_present": 11,
    "bus_capacity": 12,
}
HORIZON_SECONDS = 24_000.0
MAX_HEADWAY_RATIO = 5.0
MAX_QUEUE_CAPACITY_RATIO = 2.0
MAX_DEMAND_CAPACITY_RATIO = 2.0
MAX_SYSTEM_PASSENGERS = 25_000.0
V17_START = 0.5
V17_END = 0.75
V17_MAX_ONBOARD_FRACTION = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.resolve(strict=True).read_text())


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    q = np.quantile(values, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "q0": float(q[0]),
        "q01": float(q[1]),
        "q05": float(q[2]),
        "q50": float(q[3]),
        "q95": float(q[4]),
        "q99": float(q[5]),
        "q100": float(q[6]),
    }


def checked_existing_evidence() -> dict:
    summary = load_json(SUMMARY)
    audit = load_json(DATA_AUDIT)
    credit = load_json(CREDIT_AUDIT)
    reward = load_json(REWARD_AUDIT)
    if (
        summary.get("dataset_sha256") != DATASET_SHA256
        or audit.get("dataset_sha256") != DATASET_SHA256
        or credit.get("dataset_sha256") != DATASET_SHA256
        or reward.get("dataset_sha256") != DATASET_SHA256
        or audit.get("passed") is not True
        or credit.get("passed") is not True
        or audit.get("errors") != []
        or credit.get("errors") != []
    ):
        raise ValueError("existing v7 training-data evidence failed closed")
    if audit.get("reconstruction_errors") != {
        "discount": 0.0,
        "duration": 0.0,
        "in_vehicle_cost": 0.0,
        "next_observations": 0.0,
        "observations": 0.0,
        "reward": 0.0,
        "waiting_cost": 0.0,
    }:
        raise ValueError("v7 reconstruction evidence drifted")
    return {
        "summary": descriptor(SUMMARY),
        "dataset_audit": descriptor(DATA_AUDIT),
        "credit_audit": descriptor(CREDIT_AUDIT),
        "reward_conditioning": descriptor(REWARD_AUDIT),
        "existing_checks_passed": True,
        "reconstruction_maximum_error": 0.0,
        "bellman_credit_maximum_error": float(
            credit["bellman_credit"]["maximum_absolute_slot_target_error"]
        ),
    }


def scan_dataset(chunk_rows: int) -> dict:
    guard_holds = []
    guard_by_policy = Counter()
    guard_by_demand = Counter()
    component_counts = Counter()
    clip_counts = Counter()
    ledger_max = 0.0
    duration_max = 0.0
    row_count = 0
    guard_count = 0
    zero_duration_count = 0
    terminal_count = 0
    nonfinite_count = 0

    with h5py.File(DATASET, "r") as handle:
        total = int(handle["actions"].shape[0])
        required = {
            "actions",
            "delta_t_seconds",
            "demand_scale",
            "discounts",
            "episode_id",
            "in_vehicle_cost",
            "policy_id",
            "raw_observations",
            "rewards",
            "source_shard_id",
            "sumo_seed",
            "demand_realization_seed",
            "terminals",
            "trajectory_id",
            "waiting_cost",
        }
        if not required.issubset(handle.keys()):
            raise ValueError("v7 dataset is missing required audit fields")
        policy_names_value = json.loads(handle.attrs["policy_names"])
        if isinstance(policy_names_value, dict):
            policy_names = {
                int(identifier): str(name)
                for name, identifier in policy_names_value.items()
            }
        elif isinstance(policy_names_value, list):
            policy_names = {
                int(identifier): str(name)
                for identifier, name in enumerate(policy_names_value)
            }
        else:
            raise ValueError("unrecognized policy_names dataset attribute")

        for start in range(0, total, chunk_rows):
            end = min(total, start + chunk_rows)
            raw = np.asarray(handle["raw_observations"][start:end], dtype=np.float64)
            actions = np.asarray(handle["actions"][start:end, 0], dtype=np.float64)
            rewards = np.asarray(handle["rewards"][start:end], dtype=np.float64)
            waiting = np.asarray(handle["waiting_cost"][start:end], dtype=np.float64)
            in_vehicle = np.asarray(handle["in_vehicle_cost"][start:end], dtype=np.float64)
            durations = np.asarray(handle["delta_t_seconds"][start:end], dtype=np.float64)
            terminals = np.asarray(handle["terminals"][start:end], dtype=np.uint8)
            discounts = np.asarray(handle["discounts"][start:end], dtype=np.float64)
            policy_ids = np.asarray(handle["policy_id"][start:end], dtype=np.int64)
            demand_scales = np.asarray(handle["demand_scale"][start:end], dtype=np.float64)
            rows = raw.shape[0]
            row_count += rows

            finite = (
                np.isfinite(raw).all(axis=1)
                & np.isfinite(actions)
                & np.isfinite(rewards)
                & np.isfinite(waiting)
                & np.isfinite(in_vehicle)
                & np.isfinite(durations)
            )
            nonfinite_count += int(np.count_nonzero(~finite))
            ledger_max = max(
                ledger_max,
                float(np.max(np.abs(rewards + waiting + in_vehicle), initial=0.0)),
            )
            duration_max = max(duration_max, float(np.max(durations, initial=0.0)))
            zero_duration_count += int(np.count_nonzero(durations == 0.0))
            terminal_count += int(np.count_nonzero(terminals))
            if np.any(discounts != (1.0 - terminals.astype(np.float64))):
                raise ValueError("terminal/discount contract drifted")

            current = np.rint(raw[:, RAW_CURRENT_DECISION_INDEX]).astype(np.int64)
            if np.any(current < 0) or np.any(current >= 16):
                raise ValueError("current decision index escaped v7 schema")
            base = RAW_PREFIX_WIDTH + current * RAW_SLOT_WIDTH + 1
            index = np.arange(rows)

            def local(name: str) -> np.ndarray:
                return raw[index, base + LOCAL_EVENT_INDEX[name]]

            progress = np.clip(
                raw[:, RAW_DECISION_TIME_SECONDS] / HORIZON_SECONDS, 0.0, 1.0
            )
            capacity = local("bus_capacity")
            onboard = local("onboard_passengers")
            arrivals = local("eligible_arrival_rate_per_second")
            forward_present = np.rint(local("forward_bus_present")).astype(np.int64)
            backward_present = np.rint(local("backward_bus_present")).astype(np.int64)
            in_window = (progress >= V17_START) & (progress < V17_END)
            forward_ok = forward_present == 1
            backward_ok = backward_present == 0
            arrivals_ok = arrivals > 0.0
            low_load = (capacity > 0.0) & (
                onboard / np.maximum(capacity, np.finfo(np.float64).tiny)
                <= V17_MAX_ONBOARD_FRACTION
            )
            guard = in_window & forward_ok & backward_ok & arrivals_ok & low_load
            count = int(np.count_nonzero(guard))
            guard_count += count
            component_counts.update(
                {
                    "activation_window": int(np.count_nonzero(in_window)),
                    "forward_bus_present": int(np.count_nonzero(forward_ok)),
                    "backward_bus_absent": int(np.count_nonzero(backward_ok)),
                    "positive_arrival_rate": int(np.count_nonzero(arrivals_ok)),
                    "low_onboard_load": int(np.count_nonzero(low_load)),
                }
            )
            if count:
                holds = np.where(actions[guard] <= -1.0, 0.0, (actions[guard] + 1.0) * 30.0)
                guard_holds.append(holds)
                for value, amount in zip(*np.unique(policy_ids[guard], return_counts=True)):
                    guard_by_policy[policy_names[int(value)]] += int(amount)
                rounded = np.round(demand_scales[guard], 2)
                for value, amount in zip(*np.unique(rounded, return_counts=True)):
                    guard_by_demand[f"{float(value):.2f}"] += int(amount)

            target_forward = local("target_forward_headway_seconds")
            target_backward = local("target_backward_headway_seconds")
            if np.any(capacity <= 0.0) or np.any(target_forward <= 0.0) or np.any(target_backward <= 0.0):
                raise ValueError("positive physical denominator contract drifted")
            forward_ratio = np.where(
                forward_ok,
                local("forward_headway_seconds") / target_forward,
                1.0,
            )
            backward_ratio = np.where(
                backward_present == 1,
                local("backward_headway_seconds") / target_backward,
                1.0,
            )
            queue_ratio = np.maximum(local("eligible_stranded_passengers"), 0.0) / capacity
            demand_ratio = (
                np.maximum(arrivals, 0.0)
                * 0.5
                * (target_forward + target_backward)
                / capacity
            )
            clip_counts.update(
                {
                    "forward_headway_ratio_above_5": int(np.count_nonzero(forward_ratio > MAX_HEADWAY_RATIO)),
                    "backward_headway_ratio_above_5": int(np.count_nonzero(backward_ratio > MAX_HEADWAY_RATIO)),
                    "queue_capacity_ratio_above_2": int(np.count_nonzero(queue_ratio > MAX_QUEUE_CAPACITY_RATIO)),
                    "demand_capacity_ratio_above_2": int(np.count_nonzero(demand_ratio > MAX_DEMAND_CAPACITY_RATIO)),
                    "onboard_capacity_ratio_above_1": int(np.count_nonzero(onboard > capacity)),
                    "system_waiting_above_scale_cap": int(np.count_nonzero(raw[:, 3] > MAX_SYSTEM_PASSENGERS)),
                    "system_in_vehicle_above_scale_cap": int(np.count_nonzero(raw[:, 4] > MAX_SYSTEM_PASSENGERS)),
                }
            )

        episode_ids = np.asarray(handle["episode_id"][:], dtype=np.int64)
        unique_episodes, episode_first = np.unique(episode_ids, return_index=True)
        environment_cells = set(
            zip(
                np.asarray(handle["sumo_seed"][:], dtype=np.int64)[episode_first].tolist(),
                np.asarray(handle["demand_realization_seed"][:], dtype=np.int64)[episode_first].tolist(),
                np.round(np.asarray(handle["demand_scale"][:], dtype=np.float64)[episode_first], 2).tolist(),
            )
        )
        unique_shards = np.unique(np.asarray(handle["source_shard_id"][:], dtype=np.int64))
        unique_trajectories = np.unique(np.asarray(handle["trajectory_id"][:], dtype=np.int64))

    all_guard_holds = np.concatenate(guard_holds) if guard_holds else np.empty(0)
    clip = {
        key: {
            "rows": int(value),
            "fraction": float(value / row_count),
        }
        for key, value in sorted(clip_counts.items())
    }
    return {
        "rows": row_count,
        "episodes": int(unique_episodes.size),
        "trajectories": int(unique_trajectories.size),
        "source_shards": int(unique_shards.size),
        "independent_environment_cells": len(environment_cells),
        "policies_per_environment_cell": float(unique_episodes.size / len(environment_cells)),
        "nonfinite_rows": nonfinite_count,
        "terminal_rows": terminal_count,
        "zero_duration_rows": zero_duration_count,
        "maximum_duration_seconds": duration_max,
        "maximum_reward_ledger_error": ledger_max,
        "v17_guard_training_support": {
            "guard_rows": guard_count,
            "guard_fraction": float(guard_count / row_count),
            "component_true_rows": dict(sorted(component_counts.items())),
            "behavior_hold_seconds_on_guard": quantiles(all_guard_holds),
            "behavior_hold_at_least_20_seconds_fraction": float(np.mean(all_guard_holds >= 20.0)),
            "behavior_hold_at_least_30_seconds_fraction": float(np.mean(all_guard_holds >= 30.0)),
            "behavior_hold_at_least_40_seconds_fraction": float(np.mean(all_guard_holds >= 40.0)),
            "by_policy": dict(sorted(guard_by_policy.items())),
            "by_demand_scale": dict(sorted(guard_by_demand.items())),
        },
        "canonical_feature_clipping_exposure": clip,
    }


def parent_training_evidence() -> dict:
    summaries = [load_json(path) for path in V14_SUMMARIES]
    if any(
        int(summary.get("simulator_decision_events", -1))
        + int(summary.get("simulator_decision_event_underfill", -1))
        != 25_000
        or summary.get("total_updates") != 70_000
        or int(summary.get("simulator_decision_event_underfill", -1)) not in (0, 1)
        or int(summary.get("final_blocked_simultaneous_batch_size", -1)) not in (0, 2)
        for summary in summaries
    ):
        raise ValueError("sealed v14 parent training accounting drifted")
    validation_auc = [
        float(summary["final_ratio_validation"]["ratio/validation_joint_auc"])
        for summary in summaries
    ]
    validation_raw_ess = [
        float(summary["final_ratio_validation"]["ratio/validation_raw_ess_fraction"])
        for summary in summaries
    ]
    training_auc = [float(summary["ratio_metrics"]["ratio/joint_auc"]) for summary in summaries]
    return {
        "status": "sealed-parent-training-complete",
        "new_training_runs": 0,
        "seeds": [int(summary["seed"]) for summary in summaries],
        "event_accounting": [
            {
                "seed": int(summary["seed"]),
                "completed_events": int(summary["simulator_decision_events"]),
                "registered_underfill": int(summary["simulator_decision_event_underfill"]),
                "blocked_final_simultaneous_batch_size": int(
                    summary["final_blocked_simultaneous_batch_size"]
                ),
            }
            for summary in summaries
        ],
        "all_event_caps_or_registered_batch_underfill_exact": True,
        "all_total_updates_exact": True,
        "ratio_weights_used_by_v14": "unit weights; ratio classifiers remained diagnostic",
        "ratio_classifier_training_joint_auc": quantiles(np.asarray(training_auc)),
        "ratio_classifier_validation_joint_auc": quantiles(np.asarray(validation_auc)),
        "ratio_classifier_validation_raw_ess_fraction": quantiles(np.asarray(validation_raw_ess)),
        "evidence": [descriptor(path) for path in V14_SUMMARIES],
    }


def environment_evidence() -> dict:
    summary = load_json(SUMMARY)
    environment = load_json(ENVIRONMENT)
    runtime = load_json(RUNTIME_LOCK)
    v16 = load_json(V16_ANALYSIS)
    incident = load_json(V17_INCIDENT)
    collection = summary["collection_matrix"]
    blocks = environment.get("blocks", [])
    completion = v16["development_scientific_result"]["completion_aulc_against_controls"]
    return {
        "training_collection": {
            "sumo_version": collection["sumo_version"],
            "python_version": summary["build_provenance"]["runtime"]["python"],
            "independent_realization_blocks": collection["independent_realization_blocks"],
            "environment_cells": collection["environment_cells"],
            "episodes": summary["episodes"],
            "policies": collection["policies"],
            "demand_scales": collection["environment_design"]["demand_scales"],
        },
        "development_evaluation": {
            "sumo_version": runtime["runtime_fingerprint"]["runtime_version"],
            "python_version": incident["remote_python_version"],
            "runtime_signature": runtime["runtime_signature"],
            "development_blocks": [block["name"] for block in blocks],
            "environment_cells": environment["environment_cells"],
            "demand_scales": [
                float(value) for value in environment["demand_scales"]
            ],
            "same_cells_reused_by_v14_v15_v16_and_v17_design": True,
        },
        "v16_completion_aulc_deltas": {
            name: float(values["delta"]) for name, values in sorted(completion.items())
        },
        "evidence": {
            "environment_design": descriptor(ENVIRONMENT),
            "runtime_lock": descriptor(RUNTIME_LOCK),
            "v16_analysis": descriptor(V16_ANALYSIS),
            "v17_python38_incident": descriptor(V17_INCIDENT),
        },
    }


def risk_register(scan: dict, parent: dict, environment: dict) -> list[dict]:
    guard = scan["v17_guard_training_support"]
    training = environment["training_collection"]
    development = environment["development_evaluation"]
    return [
        {
            "id": "objective-constraint-mismatch",
            "severity": "high",
            "affects": ["effectiveness", "development-gate-pass-rate"],
            "finding": (
                "The optimized reward is generalized passenger time, while completion rate is "
                "only an evaluation constraint. A policy can improve welfare and still fail the "
                "completion noninferiority boundary."
            ),
            "evidence": {
                "reward_objective": "undiscounted negative waiting and in-vehicle passenger-seconds",
                "v16_completion_aulc_deltas": environment["v16_completion_aulc_deltas"],
            },
            "remediation": (
                "For a future training generation, preregister a completion-aware constrained "
                "objective or terminal service-shortfall cost. Do not retrofit this into v17-v2."
            ),
        },
        {
            "id": "adaptive-development-cell-reuse",
            "severity": "high",
            "affects": ["validity", "selection-bias"],
            "finding": (
                "The same nine development cells informed v14-v17 design. Repeated tuning can "
                "overfit these cells even when each individual analysis is preregistered."
            ),
            "evidence": {
                "development_cells": development["environment_cells"],
                "reused_rounds": ["v14", "v15", "v16", "v17"],
            },
            "remediation": (
                "Treat v17-v2 as the final operational correction of the frozen v17 design. "
                "Any later scientific redesign should use fresh, untouched development blocks "
                "or nested development selection."
            ),
        },
        {
            "id": "sumo-runtime-version-shift",
            "severity": "medium-high",
            "affects": ["sim-to-evaluation-transfer", "effectiveness"],
            "finding": (
                "Training data were collected with SUMO 1.27.1, whereas remote development "
                "evaluation is locked to SUMO/libsumo 1.25.0."
            ),
            "evidence": {
                "training_sumo": training["sumo_version"],
                "evaluation_sumo": development["sumo_version"],
            },
            "remediation": (
                "Before any formal confirmation, preregister a paired version-sensitivity audit "
                "on fresh development cells. Do not change the frozen v17-v2 runtime."
            ),
        },
        {
            "id": "transition-count-overstates-independent-sample-size",
            "severity": "medium",
            "affects": ["convergence-uncertainty", "generalization"],
            "finding": (
                "The dataset has millions of AEC-token transitions but only 30 independent "
                "exogenous environment cells, each reused by eight behavior policies."
            ),
            "evidence": {
                "rows": scan["rows"],
                "episodes": scan["episodes"],
                "independent_environment_cells": scan["independent_environment_cells"],
                "policies_per_environment_cell": scan["policies_per_environment_cell"],
            },
            "remediation": (
                "Report uncertainty by exogenous block/episode rather than transition; future "
                "training data should prioritize additional independent cells over denser tokens."
            ),
        },
        {
            "id": "sparse-positive-action-support",
            "severity": "medium",
            "affects": ["offline-action-support", "tail-overlay-validity"],
            "finding": (
                "Most offline actions are exact zero and only a minority of guard states contain "
                "behavior holds at the v17 floor magnitudes."
            ),
            "evidence": {
                "dataset_exact_zero_action_rate": load_json(SUMMARY)["exact_zero_action_rate"],
                "v17_guard_fraction": guard["guard_fraction"],
                "guard_hold_at_least_30_seconds_fraction": guard[
                    "behavior_hold_at_least_30_seconds_fraction"
                ],
            },
            "remediation": (
                "Keep v17 as a bounded physical overlay rather than claiming learned support. "
                "A future learned completion controller needs explicit tail-state/action coverage."
            ),
        },
        {
            "id": "ratio-diagnostic-generalization",
            "severity": "medium",
            "affects": ["domain-shift-diagnosis"],
            "finding": (
                "The ratio classifier separates training batches but generalizes near chance, "
                "with low raw validation ESS. Unit weights prevented this diagnostic failure "
                "from directly perturbing the sealed v14 optimizer."
            ),
            "evidence": {
                "training_joint_auc": parent["ratio_classifier_training_joint_auc"],
                "validation_joint_auc": parent["ratio_classifier_validation_joint_auc"],
                "validation_raw_ess_fraction": parent[
                    "ratio_classifier_validation_raw_ess_fraction"
                ],
            },
            "remediation": (
                "Do not enable learned ratio weights post hoc. If revisited, use block-held-out "
                "validation and calibration gates before optimizer use."
            ),
        },
        {
            "id": "canonical-feature-clipping",
            "severity": "low-to-medium",
            "affects": ["state-aliasing", "tail-state-resolution"],
            "finding": (
                "Physical values beyond canonical caps are aliased. The measured exposure below "
                "determines whether this is negligible or material."
            ),
            "evidence": scan["canonical_feature_clipping_exposure"],
            "remediation": (
                "Retain current frozen normalization for v17-v2. For future training, inspect "
                "clipped tails by environment block before changing scales."
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=32_768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    if args.chunk_rows < 1024:
        raise ValueError("chunk rows is unreasonably small")
    if sha256(DATASET.resolve(strict=True)) != DATASET_SHA256:
        raise ValueError("v7 training dataset identity drifted")

    existing = checked_existing_evidence()
    scan = scan_dataset(args.chunk_rows)
    if (
        scan["rows"] != 3_284_718
        or scan["episodes"] != 240
        or scan["trajectories"] != 240
        or scan["source_shards"] != 240
        or scan["independent_environment_cells"] != 30
        or scan["nonfinite_rows"] != 0
        or scan["terminal_rows"] != 240
        or scan["maximum_reward_ledger_error"] != 0.0
    ):
        raise ValueError("independent dataset scan failed closed")
    parent = parent_training_evidence()
    environment = environment_evidence()
    risks = risk_register(scan, parent, environment)
    payload = {
        "schema_version": "transit-h2oplus-v17-data-environment-convergence-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "training data, sealed parent training, and development runtime only",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_authorized_or_submitted": False,
        "local_training_or_sumo_executed": False,
        "v17_incident_numeric_outcomes_inspected": False,
        "dataset": descriptor(DATASET),
        "existing_data_integrity": existing,
        "independent_dataset_scan": scan,
        "sealed_parent_training": parent,
        "environment_comparison": environment,
        "risk_register": risks,
        "conclusions": {
            "artifact_integrity": "passed",
            "sealed_parent_training_accounting": "passed",
            "algorithm_convergence_failure_proven": False,
            "material_data_or_environment_factors_present": True,
            "highest_priority_factors": [
                "objective-constraint-mismatch",
                "adaptive-development-cell-reuse",
                "sumo-runtime-version-shift",
            ],
            "v17_v2_allowed_change": (
                "Python 3.8 compatibility and fresh evidence identity only; scientific design "
                "and frozen runtime remain unchanged"
            ),
            "formal_effect_claim_supported": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha256(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
