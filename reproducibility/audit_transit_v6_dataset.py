#!/usr/bin/env python3
"""Independently audit a chronological transit-smdp-v6 HDF5 dataset."""

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

from bus_h2o.mdp_contract_v4 import (
    RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4,
    correct_headway_measurement_jitter_v4,
)
from bus_h2o.mdp_contract_v6 import (
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V6,
    MAX_SIMULTANEOUS_DECISIONS,
    RAW_EVENT_STATE_DIM,
    RAW_SLOT_WIDTH,
    RAW_STATE_DIM,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v6,
    raw_state_to_canonical_v6,
    transition_duration_seconds_v6,
)
from collect_policy.build_transit_v4_dataset import dataset_content_sha256
from reproducibility.audit_transit_v4_dataset import (
    _maximum_absolute_error,
    numeric_summary,
    sha256,
)


SHARD_SCHEMA = "transit-smdp-v6-shard-v1"
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


def _slot_start(slot: int) -> int:
    return 3 + int(slot) * RAW_SLOT_WIDTH


def _correct_packed(values: np.ndarray) -> np.ndarray:
    corrected = np.asarray(values, dtype=np.float32).copy()
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        start = _slot_start(slot)
        selected = corrected[:, start] > 0.5
        if not np.any(selected):
            continue
        local = corrected[
            selected, start + 1 : start + 1 + RAW_EVENT_STATE_DIM
        ].copy()
        local[:, :RAW_EVENT_STATE_DIM_V4] = correct_headway_measurement_jitter_v4(
            local[:, :RAW_EVENT_STATE_DIM_V4]
        )
        corrected[
            selected, start + 1 : start + 1 + RAW_EVENT_STATE_DIM
        ] = local
    return corrected


def _audit_sources(
    descriptors: list[dict], arrays: dict[str, np.ndarray], verify: bool
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

        selected = arrays["source_shard_id"] == source_id
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
            }
        for name, expected in comparisons.items():
            actual = arrays[name][selected]
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


def audit(path: Path, *, verify_source_files: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    errors = []
    disclosures = []
    with h5py.File(path, "r") as handle:
        missing = sorted(set(REQUIRED_DATASETS).difference(handle.keys()))
        if missing:
            raise KeyError(f"dataset is missing arrays {missing}")
        arrays = {name: np.asarray(handle[name]) for name in REQUIRED_DATASETS}
        attrs = {name: handle.attrs[name] for name in handle.attrs}

    n = len(arrays["rewards"])
    if not all(values.shape[0] == n for values in arrays.values()):
        errors.append("HDF5 arrays have inconsistent row counts")
    if attrs.get("schema_version") != SCHEMA_VERSION:
        errors.append("HDF5 schema is not transit-smdp-v6")
    config = json.loads(attrs.get("config", "{}"))
    state_names = json.loads(attrs.get("state_names", "[]"))
    raw_state_names = json.loads(attrs.get("raw_state_names", "[]"))
    station_max = np.asarray(
        json.loads(attrs.get("station_max_by_line", "[]")), np.float32
    )
    if config != DEFAULT_CONFIG_V6.to_dict():
        errors.append("v6 configuration differs from the executable contract")
    expected_discount_semantics = (
        "finite-horizon undiscounted objective: nonterminal discount is 1, "
        "including at zero duration; terminal discount is 0"
    )
    if attrs.get("discount_semantics") != expected_discount_semantics:
        errors.append("v6 discount semantics metadata is inconsistent")
    if state_names != list(CANONICAL_STATE_NAMES):
        errors.append("v6 canonical state names/order differ from the contract")
    if raw_state_names != list(RAW_STATE_NAMES):
        errors.append("v6 raw state names/order differ from the contract")
    if arrays["raw_observations"].shape != (n, RAW_STATE_DIM):
        errors.append("v6 raw observation width is invalid")

    policy_names = json.loads(attrs["policy_names"])
    split_names = json.loads(attrs["split_names"])
    scenario_names = json.loads(attrs["scenario_manifest_hashes"])
    recomputed_content = dataset_content_sha256(
        arrays,
        {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "state_names": state_names,
            "raw_state_names": raw_state_names,
            "policy_names": policy_names,
            "split_names": split_names,
            "scenario_manifest_hashes": scenario_names,
            "station_max_by_line": station_max.tolist(),
        },
    )
    if attrs.get("content_signature") != recomputed_content:
        errors.append("v6 logical content signature is invalid")

    raw = arrays["raw_observations"]
    raw_next = arrays["raw_next_observations"]
    terminals = arrays["terminals"].astype(bool)
    canonical = raw_state_to_canonical_v6(raw, station_max)
    canonical_next = raw_state_to_canonical_v6(raw_next, station_max)
    durations = np.asarray(
        transition_duration_seconds_v6(raw, raw_next), np.float64
    )
    discounts = np.asarray(continuous_time_discount_v6(durations), np.float32)
    discounts[terminals] = 0.0
    waiting_cost = (
        DEFAULT_CONFIG_V6.waiting_time_value
        * arrays["wait_passenger_seconds"]
        / DEFAULT_CONFIG_V6.reward_unit_passenger_seconds
    )
    in_vehicle_cost = (
        DEFAULT_CONFIG_V6.in_vehicle_time_value
        * arrays["in_vehicle_passenger_seconds"]
        / DEFAULT_CONFIG_V6.reward_unit_passenger_seconds
    )
    reconstruction_errors = {
        "observations": _maximum_absolute_error(
            arrays["observations"], canonical
        ),
        "next_observations": _maximum_absolute_error(
            arrays["next_observations"], canonical_next
        ),
        "duration": _maximum_absolute_error(
            arrays["delta_t_seconds"], durations
        ),
        "discount": _maximum_absolute_error(arrays["discounts"], discounts),
        "waiting_cost": _maximum_absolute_error(
            arrays["waiting_cost"], waiting_cost
        ),
        "in_vehicle_cost": _maximum_absolute_error(
            arrays["in_vehicle_cost"], in_vehicle_cost
        ),
        "reward": _maximum_absolute_error(
            arrays["rewards"], -(waiting_cost + in_vehicle_cost)
        ),
    }
    for name, error in reconstruction_errors.items():
        if error != 0.0:
            errors.append(f"v6 {name} reconstruction error is {error}")

    if np.any(durations < 0.0):
        errors.append("v6 time moves backwards")
    zero = (durations == 0.0) & ~terminals
    if np.any(arrays["rewards"][zero] != 0.0):
        errors.append("zero-time v6 rows have nonzero reward")
    if np.any(arrays["discounts"][zero] != 1.0):
        errors.append("zero-time v6 rows do not have unit discount")
    if np.any(arrays["discounts"][terminals] != 0.0):
        errors.append("terminal v6 rows have nonzero discount")

    counts = np.rint(raw[:, 1]).astype(np.int64)
    indices = np.rint(raw[:, 2]).astype(np.int64)
    next_indices = np.rint(raw_next[:, 2]).astype(np.int64)
    if np.any(counts < 1) or np.any(counts > MAX_SIMULTANEOUS_DECISIONS):
        errors.append("v6 simultaneous decision count exceeds its schema")
    if np.any(indices < 0) or np.any(indices >= counts):
        errors.append("v6 current decision indices are invalid")
    action_history_errors = 0
    mask_errors = 0
    for row in range(n):
        for slot in range(MAX_SIMULTANEOUS_DECISIONS):
            start = _slot_start(slot)
            mask_errors += int(raw[row, start] != float(slot < counts[row]))
            prior = raw[row, start + RAW_SLOT_WIDTH - 1]
            action_history_errors += int(slot >= indices[row] and prior != -1.0)
        if zero[row]:
            action_history_errors += int(next_indices[row] != indices[row] + 1)
            action_position = _slot_start(indices[row]) + RAW_SLOT_WIDTH - 1
            action_history_errors += int(
                raw_next[row, action_position] != arrays["actions"][row, 0]
            )
            for slot in range(counts[row]):
                start = _slot_start(slot)
                action_history_errors += int(
                    not np.array_equal(
                        raw[
                            row,
                            start + 1 : start + 1 + RAW_EVENT_STATE_DIM,
                        ],
                        raw_next[
                            row,
                            start + 1 : start + 1 + RAW_EVENT_STATE_DIM,
                        ],
                    )
                )
    if mask_errors:
        errors.append(f"v6 packed active masks have {mask_errors} errors")
    if action_history_errors:
        errors.append(
            f"v6 serialized action history has {action_history_errors} errors"
        )

    episode_ids = arrays["episode_id"]
    trajectory_ids = arrays["trajectory_id"]
    chain_errors = 0
    terminal_errors = 0
    chronological_reward_sums = {}
    for episode in np.unique(episode_ids):
        selected = np.flatnonzero(episode_ids == episode)
        terminal_errors += int(np.sum(terminals[selected]) != 1)
        terminal_errors += int(not terminals[selected[-1]])
        terminal_errors += int(np.unique(trajectory_ids[selected]).size != 1)
        for left, right in zip(selected[:-1], selected[1:]):
            chain_errors += int(not np.array_equal(raw_next[left], raw[right]))
        chronological_reward_sums[str(int(episode))] = float(
            arrays["rewards"][selected].sum()
        )
    if chain_errors:
        errors.append(f"v6 chronological chain has {chain_errors} broken links")
    if terminal_errors:
        errors.append(f"v6 episode terminal structure has {terminal_errors} errors")

    descriptors = json.loads(attrs["source_shards"])
    source_integrity = _audit_sources(
        descriptors, arrays, verify_source_files
    )
    if verify_source_files and not source_integrity["verified"]:
        errors.append("one or more v6 source shards failed verification")
    if verify_source_files:
        declared_sums = source_integrity["episode_reward_sums"]
        observed_sums = [
            chronological_reward_sums[str(index)]
            for index in range(len(declared_sums))
        ]
        if not np.allclose(observed_sums, declared_sums, rtol=0.0, atol=1e-12):
            errors.append("v6 episode reward sums differ from source declarations")

    disclosures.append(
        "Pre-first-decision passenger time is excluded from training rewards but "
        "retained in episode welfare; it is invariant to the holding policy."
    )
    report = {
        "schema_version": "transit-smdp-v6-independent-audit-v1",
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "passed": not errors,
        "errors": errors,
        "disclosures": disclosures,
        "rows": n,
        "episodes": int(np.unique(episode_ids).size),
        "terminal_rows": int(terminals.sum()),
        "zero_duration_rows": int(zero.sum()),
        "maximum_simultaneous_decisions_observed": int(counts.max()),
        "reconstruction_errors": reconstruction_errors,
        "chronological_reward_sums": chronological_reward_sums,
        "duration_seconds": numeric_summary(durations),
        "reward": numeric_summary(arrays["rewards"]),
        "source_integrity": source_integrity,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-source-files", action="store_true")
    args = parser.parse_args()
    report = audit(args.dataset, verify_source_files=not args.skip_source_files)
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
