#!/usr/bin/env python3
"""Independently audit a derived transit-smdp-v5 HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
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
from bus_h2o.mdp_contract_v5 import (
    CANONICAL_STATE_INDEX,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V5,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v5,
    raw_event_to_canonical_v5,
    transition_duration_seconds_v5,
)
from reproducibility.audit_transit_v4_dataset import (
    _maximum_absolute_error,
    logical_content_sha256,
    numeric_summary,
    sha256,
)


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
    "demand_scale",
    "source_shard_id",
)


def _correct_raw(values: np.ndarray) -> np.ndarray:
    corrected = np.asarray(values, dtype=np.float32).copy()
    corrected[:, :RAW_EVENT_STATE_DIM_V4] = correct_headway_measurement_jitter_v4(
        corrected[:, :RAW_EVENT_STATE_DIM_V4]
    )
    return corrected


def _source_integrity(
    source_descriptors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    verify_source_files: bool,
) -> dict[str, Any]:
    if not verify_source_files:
        return {
            "verified": False,
            "declared_shards": len(source_descriptors),
            "reason": "source-file verification disabled",
        }

    missing = []
    hash_mismatches = []
    metadata_mismatches = []
    content_mismatches = []
    matrix_cells = []
    ledger_errors = []
    welfare_by_policy: dict[str, list[float]] = defaultdict(list)
    source_arrays = (
        "actions",
        "terminals",
        "wait_passenger_seconds",
        "in_vehicle_passenger_seconds",
    )
    for source_id, descriptor in enumerate(source_descriptors):
        path = Path(descriptor["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        actual_hash = sha256(path)
        if actual_hash != descriptor["sha256"]:
            hash_mismatches.append(
                {
                    "path": str(path),
                    "declared": descriptor["sha256"],
                    "actual": actual_hash,
                }
            )
            continue
        summary_path = path.with_suffix(path.suffix + ".json")
        if not summary_path.is_file():
            metadata_mismatches.append(
                {"path": str(path), "reason": "missing summary"}
            )
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("shard_sha256") != actual_hash:
            metadata_mismatches.append(
                {"path": str(path), "reason": "summary hash mismatch"}
            )
        if summary.get("schema_version") != "transit-smdp-v5-shard-v2":
            metadata_mismatches.append(
                {"path": str(path), "reason": "summary schema mismatch"}
            )
        ledger_audit = summary.get("ledger_audit", {})
        count_audit = ledger_audit.get("count_source_audit", {})
        wait_error = float(
            count_audit.get("max_waiting_count_error", float("nan"))
        )
        in_vehicle_error = float(
            count_audit.get("max_in_vehicle_count_error", float("nan"))
        )
        if ledger_audit.get(
            "legacy_sampled_trip_histories_used_as_ground_truth"
        ) is not False:
            metadata_mismatches.append(
                {"path": str(path), "reason": "non-exact welfare source"}
            )
        reward_error = float(
            summary.get("training_reward", {}).get(
                "max_reward_reconstruction_error", float("nan")
            )
        )
        ledger_errors.append(
            {
                "path": str(path),
                "waiting_count": wait_error,
                "in_vehicle_count": in_vehicle_error,
                "reward": reward_error,
            }
        )
        matrix_cells.append(
            (
                summary.get("policy_name"),
                summary.get("scenario", {}).get("manifest_sha256"),
                summary.get("sumo_seed"),
            )
        )
        welfare = summary.get("passenger_welfare", {}).get(
            "accrued_generalized_seconds_per_departed_passenger"
        )
        if welfare is not None:
            welfare_by_policy[str(summary.get("policy_name"))].append(float(welfare))

        selected = arrays["source_shard_id"] == source_id
        with np.load(path, allow_pickle=False) as source:
            expected_rows = int(np.asarray(source["rewards"]).shape[0])
            if int(np.sum(selected)) != expected_rows:
                content_mismatches.append(
                    {"path": str(path), "array": "row_count"}
                )
                continue
            comparisons = {
                "raw_observations": _correct_raw(source["observations"]),
                "raw_next_observations": _correct_raw(
                    source["next_observations"]
                ),
                **{name: np.asarray(source[name]) for name in source_arrays},
            }
            for name, expected in comparisons.items():
                actual = arrays[name][selected]
                if actual.shape != expected.shape or not np.array_equal(
                    actual, expected
                ):
                    content_mismatches.append(
                        {
                            "path": str(path),
                            "array": name,
                            "maximum_absolute_error": (
                                _maximum_absolute_error(actual, expected)
                                if actual.shape == expected.shape
                                else None
                            ),
                        }
                    )

    finite_zero_ledger = all(
        all(np.isfinite(row[name]) and abs(row[name]) <= 1e-6 for name in (
            "waiting_count",
            "in_vehicle_count",
            "reward",
        ))
        for row in ledger_errors
    )
    return {
        "verified": not (
            missing
            or hash_mismatches
            or metadata_mismatches
            or content_mismatches
        )
        and finite_zero_ledger,
        "declared_shards": len(source_descriptors),
        "missing": missing,
        "hash_mismatches": hash_mismatches,
        "metadata_mismatches": metadata_mismatches,
        "content_mismatches": content_mismatches,
        "ledger_errors": ledger_errors,
        "all_ledger_errors_within_tolerance": finite_zero_ledger,
        "unique_matrix_cells": len(set(matrix_cells)),
        "duplicate_matrix_cells": len(matrix_cells) - len(set(matrix_cells)),
        "episode_welfare_by_policy": {
            policy: numeric_summary(values)
            for policy, values in sorted(welfare_by_policy.items())
        },
    }


def audit(path: Path, *, verify_source_files: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    errors: list[str] = []
    disclosures: list[str] = [
        "Agent-centric reward intervals overlap across buses; transition rewards "
        "must not be summed to estimate episode passenger welfare."
    ]

    with h5py.File(path, "r") as handle:
        attrs = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in handle.attrs.items()
        }
        missing = sorted(set(REQUIRED_DATASETS).difference(handle.keys()))
        extra = sorted(set(handle.keys()).difference(REQUIRED_DATASETS))
        if missing:
            raise KeyError(f"dataset is missing required arrays: {missing}")
        arrays = {name: handle[name][:] for name in REQUIRED_DATASETS}
        shapes = {
            name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in sorted(handle.items())
        }
        config = json.loads(attrs["config"])
        state_names = json.loads(attrs["state_names"])
        raw_state_names = json.loads(attrs["raw_state_names"])
        policy_name_to_id = json.loads(attrs["policy_names"])
        split_names = json.loads(attrs["split_names"])
        scenario_to_id = json.loads(attrs["scenario_manifest_hashes"])
        station_max = json.loads(attrs["station_max_by_line"])
        content_metadata = {
            "schema_version": attrs["schema_version"],
            "config": config,
            "state_names": state_names,
            "raw_state_names": raw_state_names,
            "policy_names": policy_name_to_id,
            "split_names": split_names,
            "scenario_manifest_hashes": scenario_to_id,
            "station_max_by_line": station_max,
        }
        recomputed_content_signature = logical_content_sha256(
            handle, content_metadata
        )

    n_rows = arrays["rewards"].shape[0]
    for name, values in arrays.items():
        if values.shape[0] != n_rows:
            errors.append(f"{name} has {values.shape[0]} rows instead of {n_rows}")
        if not np.isfinite(values).all():
            errors.append(f"{name} contains non-finite values")
    if extra:
        errors.append(f"unexpected HDF5 arrays are not content-hashed: {extra}")
    if attrs.get("schema_version") != SCHEMA_VERSION:
        errors.append("HDF5 schema does not match the executable v5 contract")
    if config != DEFAULT_CONFIG_V5.to_dict():
        errors.append("HDF5 configuration does not match the executable default")
    if state_names != list(CANONICAL_STATE_NAMES):
        errors.append("canonical state names or order do not match the contract")
    if raw_state_names != list(RAW_EVENT_STATE_NAMES):
        errors.append("raw state names or order do not match the contract")
    if attrs.get("content_signature") != recomputed_content_signature:
        errors.append("logical content signature does not match arrays and semantics")
    if "collection_matrix" not in attrs:
        errors.append("dataset lacks a provenance-locked collection matrix")

    raw = arrays["raw_observations"]
    raw_next = arrays["raw_next_observations"]
    actions = arrays["actions"]
    terminals = arrays["terminals"].astype(bool)
    if actions.shape != (n_rows, 1) or np.any(np.abs(actions) > 1.00001):
        errors.append("actions do not have shape (N, 1) within [-1, 1]")
    if np.any((arrays["terminals"] != 0) & (arrays["terminals"] != 1)):
        errors.append("terminal flags are not binary")
    for endpoint, observations in (("raw", raw), ("raw_next", raw_next)):
        for field in ("forward_headway_seconds", "backward_headway_seconds"):
            if np.any(observations[:, RAW_EVENT_STATE_INDEX[field]] < 0.0):
                errors.append(f"{endpoint}.{field} remains negative")
        for field in (
            "target_forward_headway_seconds",
            "target_backward_headway_seconds",
            "bus_capacity",
        ):
            if np.any(observations[:, RAW_EVENT_STATE_INDEX[field]] <= 0.0):
                errors.append(f"{endpoint}.{field} contains non-positive values")
        for field in ("forward_bus_present", "backward_bus_present"):
            values = observations[:, RAW_EVENT_STATE_INDEX[field]]
            if np.any((values != 0.0) & (values != 1.0)):
                errors.append(f"{endpoint}.{field} is not binary")
        for field in (
            "system_waiting_passengers",
            "system_in_vehicle_passengers",
        ):
            if np.any(observations[:, RAW_EVENT_STATE_INDEX[field]] < 0.0):
                errors.append(f"{endpoint}.{field} is negative")

    canonical = raw_event_to_canonical_v5(raw, station_max)
    canonical_next = raw_event_to_canonical_v5(raw_next, station_max)
    durations = np.asarray(transition_duration_seconds_v5(raw, raw_next))
    discounts = np.asarray(continuous_time_discount_v5(durations), np.float32)
    discounts[terminals] = 0.0
    waiting_cost = (
        DEFAULT_CONFIG_V5.waiting_time_value
        * arrays["wait_passenger_seconds"]
        / DEFAULT_CONFIG_V5.reward_unit_passenger_seconds
    )
    in_vehicle_cost = (
        DEFAULT_CONFIG_V5.in_vehicle_time_value
        * arrays["in_vehicle_passenger_seconds"]
        / DEFAULT_CONFIG_V5.reward_unit_passenger_seconds
    )
    reconstruction_errors = {
        "observations": _maximum_absolute_error(arrays["observations"], canonical),
        "next_observations": _maximum_absolute_error(
            arrays["next_observations"], canonical_next
        ),
        "waiting_cost": _maximum_absolute_error(
            arrays["waiting_cost"], waiting_cost
        ),
        "in_vehicle_cost": _maximum_absolute_error(
            arrays["in_vehicle_cost"], in_vehicle_cost
        ),
        "reward_identity": _maximum_absolute_error(
            arrays["rewards"], -(waiting_cost + in_vehicle_cost)
        ),
        "delta_t_seconds": _maximum_absolute_error(
            arrays["delta_t_seconds"], durations
        ),
        "discounts": _maximum_absolute_error(arrays["discounts"], discounts),
    }
    for name, error in reconstruction_errors.items():
        if error != 0.0:
            errors.append(f"{name} reconstruction error is {error}")
    if np.any(arrays["wait_passenger_seconds"] < 0.0) or np.any(
        arrays["in_vehicle_passenger_seconds"] < 0.0
    ):
        errors.append("passenger-time components contain negative values")
    if np.any(durations <= 0.0):
        errors.append("one or more transition durations are not positive")

    raw_line = np.rint(raw[:, RAW_EVENT_STATE_INDEX["line_id"]]).astype(np.int64)
    next_line = np.rint(raw_next[:, RAW_EVENT_STATE_INDEX["line_id"]]).astype(
        np.int64
    )
    raw_station = raw[:, RAW_EVENT_STATE_INDEX["station_index"]]
    next_station = raw_next[:, RAW_EVENT_STATE_INDEX["station_index"]]
    if np.any(raw_line != next_line):
        errors.append("one or more transitions cross directional lines")
    if np.any(next_station[~terminals] <= raw_station[~terminals]):
        errors.append("one or more nonterminal transitions are not downstream")
    if np.any(next_station[terminals] < raw_station[terminals]):
        errors.append("one or more terminal transitions move backwards")
    if np.any(arrays["discounts"][terminals] != 0.0):
        errors.append("one or more terminal transitions have nonzero discount")

    trajectory_ids = arrays["trajectory_id"]
    episode_ids = arrays["episode_id"]
    policy_ids = arrays["policy_id"]
    source_ids = arrays["source_shard_id"]
    trajectory_group_conflicts = 0
    terminal_count_errors = 0
    for trajectory_id in np.unique(trajectory_ids):
        selected = trajectory_ids == trajectory_id
        trajectory_group_conflicts += int(np.unique(episode_ids[selected]).size != 1)
        trajectory_group_conflicts += int(np.unique(policy_ids[selected]).size != 1)
        trajectory_group_conflicts += int(np.unique(source_ids[selected]).size != 1)
        terminal_count_errors += int(np.sum(terminals[selected]) != 1)
    if trajectory_group_conflicts:
        errors.append("trajectory IDs cross episode, policy, or source groups")
    if terminal_count_errors:
        errors.append(
            f"{terminal_count_errors} trajectories do not contain exactly one terminal row"
        )

    source_descriptors = json.loads(attrs["source_shards"])
    source_integrity = _source_integrity(
        source_descriptors,
        arrays,
        verify_source_files=verify_source_files,
    )
    if verify_source_files and not source_integrity["verified"]:
        errors.append("one or more immutable source shards failed verification")
    if verify_source_files and source_integrity["duplicate_matrix_cells"]:
        errors.append("source shards repeat policy-scenario-SUMO-seed cells")

    unique_splits = sorted(map(int, np.unique(arrays["split_id"])))
    if unique_splits == [int(split_names["train"])]:
        disclosures.append(
            "The dataset is training-only; selection and testing require disjoint "
            "demand realizations and SUMO seeds."
        )
    policy_id_to_name = {
        int(identifier): name for name, identifier in policy_name_to_id.items()
    }
    by_policy = {}
    for policy_id in sorted(np.unique(policy_ids)):
        selected = policy_ids == policy_id
        policy_actions = actions[selected, 0]
        by_policy[policy_id_to_name[int(policy_id)]] = {
            "rows": int(np.sum(selected)),
            "episodes": int(np.unique(episode_ids[selected]).size),
            "exact_no_hold_rate": float(np.mean(policy_actions == -1.0)),
            "hold_seconds": numeric_summary(30.0 * (policy_actions + 1.0)),
            "reward": numeric_summary(arrays["rewards"][selected]),
            "wait_passenger_seconds": numeric_summary(
                arrays["wait_passenger_seconds"][selected]
            ),
            "in_vehicle_passenger_seconds": numeric_summary(
                arrays["in_vehicle_passenger_seconds"][selected]
            ),
        }

    state = {
        name: {
            **numeric_summary(arrays["observations"][:, index]),
            "exact_zero_rate": float(
                np.mean(arrays["observations"][:, index] == 0.0)
            ),
        }
        for index, name in enumerate(state_names)
    }
    for name in ("system_waiting_fraction", "system_in_vehicle_fraction"):
        at_cap = int(
            np.sum(
                arrays["observations"][:, CANONICAL_STATE_INDEX[name]]
                == DEFAULT_CONFIG_V5.max_system_capacity_fraction
            )
        )
        state[name]["at_configured_cap"] = at_cap
        if at_cap:
            disclosures.append(
                f"Canonical feature {name} reaches its configured cap in {at_cap} rows."
            )

    return {
        "schema_version": "transit-smdp-v5-audit-v1",
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "logical_content_signature": recomputed_content_signature,
        "logical_content_signature_matches": (
            attrs.get("content_signature") == recomputed_content_signature
        ),
        "rows": n_rows,
        "episodes": int(np.unique(episode_ids).size),
        "trajectories": int(np.unique(trajectory_ids).size),
        "trajectory_group_conflicts": trajectory_group_conflicts,
        "trajectory_terminal_count_errors": terminal_count_errors,
        "attributes": attrs,
        "datasets": shapes,
        "reconstruction_max_absolute_error": reconstruction_errors,
        "source_integrity": source_integrity,
        "matrix": {
            "policies": len(policy_name_to_id),
            "scenarios": len(scenario_to_id),
            "sumo_seeds": sorted(map(int, np.unique(arrays["sumo_seed"]))),
            "splits": unique_splits,
        },
        "actions": {
            "exact_no_hold_rate": float(np.mean(actions[:, 0] == -1.0)),
            "exact_max_hold_rate": float(np.mean(actions[:, 0] == 1.0)),
            "hold_seconds": numeric_summary(30.0 * (actions[:, 0] + 1.0)),
        },
        "rewards": numeric_summary(arrays["rewards"]),
        "wait_passenger_seconds": numeric_summary(
            arrays["wait_passenger_seconds"]
        ),
        "in_vehicle_passenger_seconds": numeric_summary(
            arrays["in_vehicle_passenger_seconds"]
        ),
        "delta_t_seconds": numeric_summary(arrays["delta_t_seconds"]),
        "state": state,
        "by_policy": by_policy,
        "errors": errors,
        "disclosures": disclosures,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-source-files", action="store_true")
    args = parser.parse_args()
    report = audit(args.dataset, verify_source_files=not args.skip_source_files)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "errors": report["errors"],
                "disclosures": report["disclosures"],
            },
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
