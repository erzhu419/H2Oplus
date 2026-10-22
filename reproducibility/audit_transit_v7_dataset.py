#!/usr/bin/env python3
"""Independently audit a chronological transit-smdp-v7 HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V7,
    HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS,
    LOCAL_EVENT_STATE_DIM,
    LOCAL_EVENT_STATE_INDEX,
    MAX_SIMULTANEOUS_DECISIONS,
    RAW_PREFIX_NAMES,
    RAW_SLOT_WIDTH,
    RAW_STATE_DIM,
    RAW_STATE_INDEX,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v7,
    raw_state_to_canonical_v7,
    transition_duration_seconds_v7,
)
from reproducibility.audit_transit_v4_dataset import (
    _maximum_absolute_error,
    logical_content_sha256,
    numeric_summary,
    sha256,
)


SHARD_SCHEMA = "transit-smdp-v7-shard-v1"
AUDIT_SCHEMA = "transit-smdp-v7-independent-audit-v1"
DEFAULT_CHUNK_ROWS = 8_192
POSITIVE_HOLD_BIN_EDGES_SECONDS = (0.0, 15.0, 30.0, 45.0, 60.0)
REGISTERED_BEHAVIOR_POLICIES = {
    "zero",
    "random-hurdle-p0.5-beta1-3",
    "random-hurdle-p0.5-uniform",
    "headway-gain-0.05",
    "headway-gain-0.1",
    "headway-gain-0.25",
    "headway-gain-0.5",
    "myopic-welfare-v2",
}
STATE_DATASETS = {
    "observations",
    "next_observations",
    "raw_observations",
    "raw_next_observations",
}
REQUIRED_DATASETS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "discounts",
    "terminals",
    "raw_observations",
    "raw_next_observations",
    "wait_passenger_seconds",
    "in_vehicle_passenger_seconds",
    "waiting_cost",
    "in_vehicle_cost",
    "delta_t_seconds",
    "event_bus_code",
    "event_line_index",
    "event_station_index",
    "event_decision_time_seconds",
    "event_base_dwell_seconds",
    "policy_id",
    "policy_seed",
    "sumo_seed",
    "split_id",
    "episode_id",
    "trajectory_id",
    "scenario_id",
    "demand_realization_seed",
    "demand_scale",
    "source_shard_id",
)


def dataset_audit_source_hashes() -> dict[str, str]:
    """Hash every local source that defines this audit's calculations."""

    sources = (
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "audit_transit_v4_dataset.py",
        ROOT / "bus_h2o" / "mdp_contract_v7.py",
        ROOT / "bus_h2o" / "mdp_contract_v5.py",
        ROOT / "bus_h2o" / "passenger_time.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sources
    }


def _slot_start(slot: int) -> int:
    return len(RAW_PREFIX_NAMES) + int(slot) * RAW_SLOT_WIDTH


def _correct_packed(values: np.ndarray) -> np.ndarray:
    corrected = np.asarray(values, dtype=np.float32).copy()
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        start = _slot_start(slot)
        selected = corrected[:, start] > 0.5
        if not np.any(selected):
            continue
        selected_rows = np.flatnonzero(selected)
        for name in ("forward_headway_seconds", "backward_headway_seconds"):
            index = start + 1 + LOCAL_EVENT_STATE_INDEX[name]
            slot_values = corrected[selected_rows, index]
            if np.any(
                slot_values < -HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS
            ):
                raise ValueError(
                    f"packed {name} is below the measurement-jitter tolerance"
                )
            corrected[selected_rows[slot_values < 0.0], index] = 0.0
    return corrected


def _audit_sources(
    descriptors: list[dict],
    handle: h5py.File,
    source_shard_ids: np.ndarray,
    verify: bool,
) -> dict[str, Any]:
    if not verify:
        return {"verified": False, "reason": "source verification disabled"}
    errors = []
    summaries = []
    for source_id, descriptor in enumerate(descriptors):
        path = Path(descriptor["path"])
        if not path.is_file() or sha256(path) != descriptor["sha256"]:
            errors.append(f"missing or changed source shard: {path}")
            continue
        summary_path = path.with_suffix(path.suffix + ".json")
        if not summary_path.is_file():
            errors.append(f"missing source summary: {summary_path}")
            continue
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        if summary.get("schema_version") != SHARD_SCHEMA:
            errors.append(f"wrong source schema: {path}")
        if summary.get("shard_sha256") != descriptor["sha256"]:
            errors.append(f"source summary hash mismatch: {path}")
        if descriptor.get("event_bus_code_to_id") != summary.get(
            "event_identity", {}
        ).get("bus_code_to_id"):
            errors.append(f"source event-bus mapping mismatch: {path}")
        if descriptor.get("event_line_index_to_id") != summary.get(
            "event_identity", {}
        ).get("line_index_to_id"):
            errors.append(f"source event-line mapping mismatch: {path}")
        count_audit = summary.get("ledger_audit", {}).get(
            "count_source_audit", {}
        )
        if count_audit.get("max_waiting_count_error") != 0 or count_audit.get(
            "max_in_vehicle_count_error"
        ) != 0:
            errors.append(f"source passenger-count audit failed: {path}")
        coverage = summary.get("training_reward", {}).get(
            "chronological_coverage_errors", {}
        )
        if any(abs(float(value)) > 1e-9 for value in coverage.values()):
            errors.append(f"source chronological coverage failed: {path}")

        selected_rows = np.flatnonzero(source_shard_ids == source_id)
        if not len(selected_rows):
            errors.append(f"source shard has no HDF5 rows: {path}")
            continue
        if len(selected_rows) > 1 and np.any(np.diff(selected_rows) != 1):
            errors.append(f"source shard rows are not contiguous: {path}")
            continue
        selected = slice(int(selected_rows[0]), int(selected_rows[-1]) + 1)
        with np.load(path, allow_pickle=False) as source:
            comparisons = {
                "raw_observations": _correct_packed(source["observations"]),
                "raw_next_observations": _correct_packed(
                    source["next_observations"]
                ),
                "actions": np.asarray(source["actions"]),
                "terminals": np.asarray(source["terminals"]),
                "wait_passenger_seconds": np.asarray(
                    source["wait_passenger_seconds"]
                ),
                "in_vehicle_passenger_seconds": np.asarray(
                    source["in_vehicle_passenger_seconds"]
                ),
                "delta_t_seconds": np.asarray(source["delta_t_seconds"]),
                "event_bus_code": np.asarray(source["event_bus_code"]),
                "event_line_index": np.asarray(source["event_line_index"]),
                "event_station_index": np.asarray(
                    source["event_station_index"]
                ),
                "event_decision_time_seconds": np.asarray(
                    source["event_decision_time_seconds"]
                ),
                "event_base_dwell_seconds": np.asarray(
                    source["event_base_dwell_seconds"]
                ),
            }
        for name, expected in comparisons.items():
            actual = np.asarray(handle[name][selected])
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                errors.append(f"derived {name} differs from source shard: {path}")
    return {
        "verified": not errors,
        "declared_shards": len(descriptors),
        "errors": errors,
        "episode_reward_sums": [
            float(summary["training_reward"]["reward_sum"])
            for summary in summaries
        ],
    }


def _update_maximum_error(
    errors: dict[str, float],
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    errors[name] = max(
        errors[name], _maximum_absolute_error(actual, expected)
    )


def _action_audit(
    actions: np.ndarray,
    policy_ids: np.ndarray,
    episode_ids: np.ndarray,
    policy_names: dict[str, int],
) -> dict[str, Any]:
    scalar = np.asarray(actions, dtype=np.float32).reshape(-1)
    finite = np.isfinite(scalar)

    def counts(selected: np.ndarray) -> dict[str, Any]:
        values = scalar[selected]
        rows = int(values.size)
        finite_values = values[np.isfinite(values)]
        hold_seconds = (
            0.5 * (finite_values + 1.0) * DEFAULT_CONFIG_V7.max_hold_seconds
        )
        positive_holds = hold_seconds[hold_seconds > 0.0]
        hold_bin_rows = np.histogram(
            positive_holds,
            bins=np.asarray(POSITIVE_HOLD_BIN_EDGES_SECONDS, dtype=np.float64),
        )[0]
        positive_rows = int(positive_holds.size)
        return {
            "rows": rows,
            "episodes": int(np.unique(episode_ids[selected]).size),
            "exact_zero_hold_rows": int(np.count_nonzero(values == -1.0)),
            "exact_zero_hold_fraction": (
                float(np.mean(values == -1.0)) if rows else None
            ),
            "exact_max_hold_rows": int(np.count_nonzero(values == 1.0)),
            "exact_max_hold_fraction": (
                float(np.mean(values == 1.0)) if rows else None
            ),
            "strictly_positive_hold_rows": positive_rows,
            "normalized_action": (
                numeric_summary(finite_values) if finite_values.size else None
            ),
            "hold_seconds": (
                numeric_summary(hold_seconds) if hold_seconds.size else None
            ),
            "positive_hold_seconds": (
                numeric_summary(positive_holds) if positive_holds.size else None
            ),
            "positive_hold_bin_edges_seconds": list(
                POSITIVE_HOLD_BIN_EDGES_SECONDS
            ),
            "positive_hold_bin_rows": [int(value) for value in hold_bin_rows],
            "positive_hold_bin_fractions": [
                float(value / positive_rows) if positive_rows else 0.0
                for value in hold_bin_rows
            ],
        }

    by_policy = {
        name: counts(policy_ids == int(identifier))
        for name, identifier in sorted(policy_names.items(), key=lambda item: item[1])
    }
    coverage_applicable = REGISTERED_BEHAVIOR_POLICIES.issubset(by_policy)
    coverage_checks: dict[str, bool] = {}
    if coverage_applicable:
        zero = by_policy["zero"]
        uniform = by_policy["random-hurdle-p0.5-uniform"]
        beta = by_policy["random-hurdle-p0.5-beta1-3"]
        coverage_checks = {
            "zero_policy_is_exact_atom": (
                zero["exact_zero_hold_rows"] == zero["rows"]
            ),
            "uniform_hurdle_zero_rate_registered": (
                0.45 <= uniform["exact_zero_hold_fraction"] <= 0.55
            ),
            "beta_hurdle_zero_rate_registered": (
                0.45 <= beta["exact_zero_hold_fraction"] <= 0.55
            ),
            "uniform_positive_supports_every_hold_quartile": (
                min(uniform["positive_hold_bin_fractions"]) >= 0.20
            ),
            "every_nonzero_policy_has_positive_hold_support": all(
                report["strictly_positive_hold_rows"] > 0
                for name, report in by_policy.items()
                if name != "zero"
            ),
        }
    coverage_gate = {
        "applicable": coverage_applicable,
        "registered_policy_names": sorted(REGISTERED_BEHAVIOR_POLICIES),
        "checks": coverage_checks,
        "passed": bool(coverage_applicable and all(coverage_checks.values())),
        "scope": (
            "descriptive for partial datasets; fail-closed when all registered "
            "behavior policies are present"
        ),
    }
    return {
        **counts(np.ones(scalar.shape, dtype=bool)),
        "minimum": float(np.min(scalar[finite])) if np.any(finite) else None,
        "maximum": float(np.max(scalar[finite])) if np.any(finite) else None,
        "nonfinite_rows": int(np.count_nonzero(~finite)),
        "outside_closed_action_space_rows": int(
            np.count_nonzero(finite & ((scalar < -1.0) | (scalar > 1.0)))
        ),
        "by_policy": by_policy,
        "coverage_gate": coverage_gate,
    }


def audit(
    path: Path,
    *,
    verify_source_files: bool = True,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    errors = []
    disclosures = []
    with h5py.File(path, "r") as handle:
        missing = sorted(set(REQUIRED_DATASETS).difference(handle.keys()))
        if missing:
            raise KeyError(f"dataset is missing arrays {missing}")
        attrs = {name: handle.attrs[name] for name in handle.attrs}
        n = int(handle["rewards"].shape[0])
        if n <= 0:
            raise ValueError("v7 dataset is empty")
        if any(int(handle[name].shape[0]) != n for name in REQUIRED_DATASETS):
            errors.append("HDF5 arrays have inconsistent row counts")
        if attrs.get("schema_version") != SCHEMA_VERSION:
            errors.append("HDF5 schema is not transit-smdp-v7")
        config = json.loads(attrs.get("config", "{}"))
        state_names = json.loads(attrs.get("state_names", "[]"))
        raw_state_names = json.loads(attrs.get("raw_state_names", "[]"))
        station_max = np.asarray(
            json.loads(attrs.get("station_max_by_line", "[]")), np.float32
        )
        if config != DEFAULT_CONFIG_V7.to_dict():
            errors.append("v7 configuration differs from the executable contract")
        expected_discount_semantics = (
            "finite-horizon undiscounted objective: nonterminal discount is 1, "
            "including at zero duration; terminal discount is 0"
        )
        if attrs.get("discount_semantics") != expected_discount_semantics:
            errors.append("v7 discount semantics metadata is inconsistent")
        if state_names != list(CANONICAL_STATE_NAMES):
            errors.append("v7 canonical state names/order differ from the contract")
        if raw_state_names != list(RAW_STATE_NAMES):
            errors.append("v7 raw state names/order differ from the contract")
        expected_shapes = {
            "observations": (n, len(CANONICAL_STATE_NAMES)),
            "next_observations": (n, len(CANONICAL_STATE_NAMES)),
            "raw_observations": (n, RAW_STATE_DIM),
            "raw_next_observations": (n, RAW_STATE_DIM),
            "actions": (n, 1),
        }
        for name, expected in expected_shapes.items():
            if handle[name].shape != expected:
                errors.append(f"v7 {name} shape is invalid")

        policy_names = json.loads(attrs["policy_names"])
        split_names = json.loads(attrs["split_names"])
        scenario_names = json.loads(attrs["scenario_manifest_hashes"])
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "state_names": state_names,
            "raw_state_names": raw_state_names,
            "policy_names": policy_names,
            "split_names": split_names,
            "scenario_manifest_hashes": scenario_names,
            "station_max_by_line": station_max.tolist(),
        }
        if attrs.get("content_signature") != logical_content_sha256(
            handle, metadata, chunk_rows=chunk_rows
        ):
            errors.append("v7 logical content signature is invalid")

        arrays = {
            name: np.asarray(handle[name])
            for name in REQUIRED_DATASETS
            if name not in STATE_DATASETS
        }
        terminals = arrays["terminals"].astype(bool)
        episode_ids = np.asarray(arrays["episode_id"], dtype=np.int64)
        trajectory_ids = np.asarray(arrays["trajectory_id"], dtype=np.int64)
        computed_durations = np.empty(n, dtype=np.float64)
        reconstruction_errors = {
            name: 0.0
            for name in (
                "observations",
                "next_observations",
                "duration",
                "discount",
                "waiting_cost",
                "in_vehicle_cost",
                "reward",
            )
        }
        counters = {
            "negative_duration": 0,
            "zero_reward": 0,
            "zero_discount": 0,
            "terminal_discount": 0,
            "noninteger_count_or_index": 0,
            "count": 0,
            "index": 0,
            "event_metadata": int(np.count_nonzero(arrays["event_bus_code"] < 0))
            + int(np.count_nonzero(arrays["event_base_dwell_seconds"] < 0.0)),
            "mask": 0,
            "action_history": 0,
            "chain": 0,
        }
        maximum_count = 0
        previous_raw_next = None
        previous_episode = None
        same_batch_indices = [
            RAW_STATE_INDEX[name]
            for name in (
                "decision_time_seconds",
                "simultaneous_decision_count",
                "system_waiting_passengers",
                "system_in_vehicle_passengers",
            )
        ]

        for start_row in range(0, n, chunk_rows):
            stop_row = min(start_row + chunk_rows, n)
            selected = slice(start_row, stop_row)
            raw = np.asarray(handle["raw_observations"][selected])
            raw_next = np.asarray(handle["raw_next_observations"][selected])
            canonical = raw_state_to_canonical_v7(raw, station_max)
            canonical_next = raw_state_to_canonical_v7(raw_next, station_max)
            durations = np.asarray(
                transition_duration_seconds_v7(raw, raw_next), np.float64
            )
            computed_durations[selected] = durations
            selected_terminals = terminals[selected]
            discounts = np.asarray(
                continuous_time_discount_v7(durations), np.float32
            )
            discounts[selected_terminals] = 0.0
            waiting_cost = (
                DEFAULT_CONFIG_V7.waiting_time_value
                * arrays["wait_passenger_seconds"][selected]
                / DEFAULT_CONFIG_V7.reward_unit_passenger_seconds
            )
            in_vehicle_cost = (
                DEFAULT_CONFIG_V7.in_vehicle_time_value
                * arrays["in_vehicle_passenger_seconds"][selected]
                / DEFAULT_CONFIG_V7.reward_unit_passenger_seconds
            )
            expected = {
                "observations": canonical,
                "next_observations": canonical_next,
                "duration": durations,
                "discount": discounts,
                "waiting_cost": waiting_cost,
                "in_vehicle_cost": in_vehicle_cost,
                "reward": -(waiting_cost + in_vehicle_cost),
            }
            actual = {
                "observations": np.asarray(handle["observations"][selected]),
                "next_observations": np.asarray(
                    handle["next_observations"][selected]
                ),
                "duration": arrays["delta_t_seconds"][selected],
                "discount": arrays["discounts"][selected],
                "waiting_cost": arrays["waiting_cost"][selected],
                "in_vehicle_cost": arrays["in_vehicle_cost"][selected],
                "reward": arrays["rewards"][selected],
            }
            for name in reconstruction_errors:
                _update_maximum_error(
                    reconstruction_errors, name, actual[name], expected[name]
                )

            zero = (durations == 0.0) & ~selected_terminals
            counters["negative_duration"] += int(np.count_nonzero(durations < 0.0))
            counters["zero_reward"] += int(
                np.count_nonzero(arrays["rewards"][selected][zero] != 0.0)
            )
            counters["zero_discount"] += int(
                np.count_nonzero(arrays["discounts"][selected][zero] != 1.0)
            )
            counters["terminal_discount"] += int(
                np.count_nonzero(
                    arrays["discounts"][selected][selected_terminals] != 0.0
                )
            )

            count_values = raw[
                :, RAW_STATE_INDEX["simultaneous_decision_count"]
            ]
            index_values = raw[:, RAW_STATE_INDEX["current_decision_index"]]
            counts = np.rint(count_values).astype(np.int64)
            indices = np.rint(index_values).astype(np.int64)
            next_indices = np.rint(
                raw_next[:, RAW_STATE_INDEX["current_decision_index"]]
            ).astype(np.int64)
            counters["noninteger_count_or_index"] += int(
                np.count_nonzero(count_values != counts)
                + np.count_nonzero(index_values != indices)
            )
            valid_counts = (counts >= 1) & (
                counts <= MAX_SIMULTANEOUS_DECISIONS
            )
            valid_indices = valid_counts & (indices >= 0) & (indices < counts)
            counters["count"] += int(np.count_nonzero(~valid_counts))
            counters["index"] += int(np.count_nonzero(~valid_indices))
            if np.any(valid_counts):
                maximum_count = max(maximum_count, int(counts[valid_counts].max()))

            valid_rows = np.flatnonzero(valid_indices)
            if len(valid_rows):
                slot_starts = len(RAW_PREFIX_NAMES) + indices[valid_rows] * RAW_SLOT_WIDTH
                event_expectations = {
                    "event_line_index": raw[
                        valid_rows,
                        slot_starts + 1 + LOCAL_EVENT_STATE_INDEX["line_id"],
                    ],
                    "event_station_index": raw[
                        valid_rows,
                        slot_starts
                        + 1
                        + LOCAL_EVENT_STATE_INDEX["station_index"],
                    ],
                    "event_decision_time_seconds": raw[
                        valid_rows,
                        RAW_STATE_INDEX["decision_time_seconds"],
                    ],
                    "event_base_dwell_seconds": raw[
                        valid_rows,
                        slot_starts
                        + 1
                        + LOCAL_EVENT_STATE_INDEX["base_dwell_seconds"],
                    ],
                }
                for name, values in event_expectations.items():
                    counters["event_metadata"] += int(
                        np.count_nonzero(arrays[name][selected][valid_rows] != values)
                    )

            for slot in range(MAX_SIMULTANEOUS_DECISIONS):
                slot_start = _slot_start(slot)
                counters["mask"] += int(
                    np.count_nonzero(raw[:, slot_start] != (slot < counts))
                )
                prior = raw[:, slot_start + RAW_SLOT_WIDTH - 1]
                counters["action_history"] += int(
                    np.count_nonzero((slot >= indices) & (prior != -1.0))
                )
                same_time_slot = zero & valid_indices & (slot < counts)
                if np.any(same_time_slot):
                    local = slice(
                        slot_start + 1,
                        slot_start + 1 + LOCAL_EVENT_STATE_DIM,
                    )
                    counters["action_history"] += int(
                        np.count_nonzero(
                            np.any(
                                raw[same_time_slot, local]
                                != raw_next[same_time_slot, local],
                                axis=1,
                            )
                        )
                    )

            zero_valid = zero & valid_indices
            counters["action_history"] += int(
                np.count_nonzero(next_indices[zero_valid] != indices[zero_valid] + 1)
            )
            if np.any(zero_valid):
                counters["action_history"] += int(
                    np.count_nonzero(
                        np.any(
                            raw[zero_valid][:, same_batch_indices]
                            != raw_next[zero_valid][:, same_batch_indices],
                            axis=1,
                        )
                    )
                )
                zero_rows = np.flatnonzero(zero_valid)
                action_positions = (
                    len(RAW_PREFIX_NAMES)
                    + indices[zero_rows] * RAW_SLOT_WIDTH
                    + RAW_SLOT_WIDTH
                    - 1
                )
                counters["action_history"] += int(
                    np.count_nonzero(
                        raw_next[zero_rows, action_positions]
                        != arrays["actions"][selected][zero_rows, 0]
                    )
                )

            selected_episodes = episode_ids[selected]
            if previous_raw_next is not None and previous_episode == int(
                selected_episodes[0]
            ):
                counters["chain"] += int(
                    not np.array_equal(previous_raw_next, raw[0])
                )
            if len(raw) > 1:
                adjacent = selected_episodes[:-1] == selected_episodes[1:]
                if np.any(adjacent):
                    counters["chain"] += int(
                        np.count_nonzero(
                            np.any(raw_next[:-1][adjacent] != raw[1:][adjacent], axis=1)
                        )
                    )
            previous_raw_next = raw_next[-1].copy()
            previous_episode = int(selected_episodes[-1])

        for name, error in reconstruction_errors.items():
            if error != 0.0:
                errors.append(f"v7 {name} reconstruction error is {error}")
        if counters["negative_duration"]:
            errors.append("v7 time moves backwards")
        if counters["zero_reward"]:
            errors.append("zero-time v7 rows have nonzero reward")
        if counters["zero_discount"]:
            errors.append("zero-time v7 rows do not have unit discount")
        if counters["terminal_discount"]:
            errors.append("terminal v7 rows have nonzero discount")
        if counters["noninteger_count_or_index"]:
            errors.append("v7 decision count/index values are not integers")
        if counters["count"]:
            errors.append("v7 simultaneous decision count exceeds its schema")
        if counters["index"]:
            errors.append("v7 current decision indices are invalid")
        if counters["event_metadata"]:
            errors.append(
                f"v7 controlled-event metadata has {counters['event_metadata']} errors"
            )
        if counters["mask"]:
            errors.append(f"v7 packed active masks have {counters['mask']} errors")
        if counters["action_history"]:
            errors.append(
                "v7 serialized action history has "
                f"{counters['action_history']} errors"
            )
        if counters["chain"]:
            errors.append(
                f"v7 chronological chain has {counters['chain']} broken links"
            )

        boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), boundaries))
        ends = np.concatenate((boundaries, np.asarray([n])))
        labels = episode_ids[starts]
        duplicate_episode_spans = len(np.unique(labels)) != len(labels)
        if duplicate_episode_spans:
            errors.append("v7 episode ids occur in multiple disjoint spans")
        terminal_errors = 0
        chronological_reward_sums = {}
        if duplicate_episode_spans:
            episode_spans = (
                (int(episode), np.flatnonzero(episode_ids == episode))
                for episode in np.unique(episode_ids)
            )
        else:
            episode_spans = (
                (int(label), np.arange(start, end, dtype=np.int64))
                for label, start, end in sorted(
                    zip(labels, starts, ends), key=lambda item: int(item[0])
                )
            )
        for episode, selected_rows in episode_spans:
            terminal_errors += int(np.sum(terminals[selected_rows]) != 1)
            terminal_errors += int(not terminals[selected_rows[-1]])
            terminal_errors += int(
                not np.all(
                    trajectory_ids[selected_rows] == trajectory_ids[selected_rows[0]]
                )
            )
            chronological_reward_sums[str(episode)] = float(
                arrays["rewards"][selected_rows].sum()
            )
        if terminal_errors:
            errors.append(
                f"v7 episode terminal structure has {terminal_errors} errors"
            )

        action_report = _action_audit(
            arrays["actions"], arrays["policy_id"], episode_ids, policy_names
        )
        if action_report["nonfinite_rows"]:
            errors.append("v7 actions contain non-finite values")
        if action_report["outside_closed_action_space_rows"]:
            errors.append("v7 actions fall outside [-1, 1]")
        if (
            action_report["coverage_gate"]["applicable"]
            and not action_report["coverage_gate"]["passed"]
        ):
            errors.append("v7 registered behavior mixture lacks action support")

        descriptors = json.loads(attrs["source_shards"])
        source_shard_ids = np.asarray(arrays["source_shard_id"], dtype=np.int64)
        for source_id, descriptor in enumerate(descriptors):
            source_rows = source_shard_ids == source_id
            mapping = descriptor.get("event_bus_code_to_id")
            line_mapping = descriptor.get("event_line_index_to_id")
            if not isinstance(mapping, list) or not mapping:
                errors.append(f"source shard {source_id} lacks an event-bus mapping")
                continue
            if np.any(arrays["event_bus_code"][source_rows] >= len(mapping)):
                errors.append(f"source shard {source_id} has an invalid event-bus code")
            if not isinstance(line_mapping, list) or not line_mapping:
                errors.append(f"source shard {source_id} lacks an event-line mapping")
            elif np.any(
                arrays["event_line_index"][source_rows] >= len(line_mapping)
            ):
                errors.append(f"source shard {source_id} has an invalid event-line index")
        source_integrity = _audit_sources(
            descriptors, handle, source_shard_ids, verify_source_files
        )
        if verify_source_files and not source_integrity["verified"]:
            errors.append("one or more v7 source shards failed verification")
        if verify_source_files:
            declared_sums = source_integrity["episode_reward_sums"]
            if len(declared_sums) != len(descriptors):
                errors.append("v7 source reward declarations are incomplete")
            elif any(
                str(index) not in chronological_reward_sums
                for index in range(len(declared_sums))
            ):
                errors.append("v7 episode ids do not match source declaration order")
            else:
                observed_sums = [
                    chronological_reward_sums[str(index)]
                    for index in range(len(declared_sums))
                ]
                if not np.allclose(
                    observed_sums, declared_sums, rtol=0.0, atol=1e-12
                ):
                    errors.append(
                        "v7 episode reward sums differ from source declarations"
                    )

    disclosures.append(
        "Pre-first-decision passenger time is excluded from training rewards but "
        "retained in episode welfare; it is invariant to the holding policy."
    )
    report = {
        "schema_version": AUDIT_SCHEMA,
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "source_sha256": dataset_audit_source_hashes(),
        "passed": not errors,
        "errors": errors,
        "disclosures": disclosures,
        "rows": n,
        "episodes": int(np.unique(episode_ids).size),
        "terminal_rows": int(terminals.sum()),
        "zero_duration_rows": int(
            np.count_nonzero((computed_durations == 0.0) & ~terminals)
        ),
        "maximum_simultaneous_decisions_observed": maximum_count,
        "reconstruction_errors": reconstruction_errors,
        "chronological_reward_sums": chronological_reward_sums,
        "duration_seconds": numeric_summary(computed_durations),
        "reward": numeric_summary(arrays["rewards"]),
        "actions": action_report,
        "source_integrity": source_integrity,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-source-files", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()
    report = audit(
        args.dataset,
        verify_source_files=not args.skip_source_files,
        chunk_rows=args.chunk_rows,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(encoded)
        os.replace(temporary, output)
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
