#!/usr/bin/env python3
"""Validate and merge chronological SUMO v7 shards into immutable HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (
    ACTION_DIM,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V7,
    HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS,
    LOCAL_EVENT_STATE_DIM,
    LOCAL_EVENT_STATE_INDEX,
    LOCAL_EVENT_STATE_NAMES,
    MAX_SIMULTANEOUS_DECISIONS,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_PREFIX_NAMES,
    RAW_SLOT_WIDTH,
    RAW_STATE_DIM,
    RAW_STATE_INDEX,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v7,
    infer_station_max_v7,
    raw_state_to_canonical_v7,
    transition_duration_seconds_v7,
)
from collect_policy.build_transit_v4_dataset import (
    SPLIT_IDS,
    canonical_sha256,
    discover_shards,
    sha256,
)


SHARD_SCHEMA = "transit-smdp-v7-shard-v1"
MATRIX_SCHEMA = "transit-smdp-v7-collection-matrix-v1"
MATRIX_SCHEMA_V2 = "transit-smdp-v7-collection-matrix-v2"
MATRIX_SCHEMAS = {MATRIX_SCHEMA, MATRIX_SCHEMA_V2}
BUILD_TRANSFORM_VERSION = "transit-smdp-v7-derived-transform-v1"
REQUIRED_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "discounts",
    "terminals",
    "delta_t_seconds",
    "trajectory_local_id",
    "wait_passenger_seconds",
    "in_vehicle_passenger_seconds",
    "event_bus_code",
    "event_line_index",
    "event_station_index",
    "event_decision_time_seconds",
    "event_base_dwell_seconds",
)


def shard_paths_from_manifest(manifest_path: Path) -> list[Path]:
    """Resolve the exact shard set registered by a collection manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") not in MATRIX_SCHEMAS:
        raise ValueError(f"unsupported v7 collection matrix: {manifest_path}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("v7 collection matrix contains no jobs")
    paths = [Path(job["output"]).expanduser().resolve() for job in jobs]
    if len(set(paths)) != len(paths):
        raise ValueError("v7 collection matrix repeats an output path")
    return sorted(paths)


def _slot_start(slot: int) -> int:
    return len(RAW_PREFIX_NAMES) + int(slot) * RAW_SLOT_WIDTH


def active_event_observations(raw_states: np.ndarray) -> np.ndarray:
    states = np.asarray(raw_states, dtype=np.float32)
    events = []
    local_event_indices = [
        RAW_EVENT_STATE_INDEX[name] for name in LOCAL_EVENT_STATE_NAMES
    ]
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        start = _slot_start(slot)
        selected = states[:, start] > 0.5
        if np.any(selected):
            event_batch = np.zeros(
                (int(np.sum(selected)), RAW_EVENT_STATE_DIM), np.float32
            )
            event_batch[:, RAW_EVENT_STATE_INDEX["sim_time_seconds"]] = states[
                selected, RAW_STATE_INDEX["decision_time_seconds"]
            ]
            event_batch[
                :, RAW_EVENT_STATE_INDEX["system_waiting_passengers"]
            ] = states[selected, RAW_STATE_INDEX["system_waiting_passengers"]]
            event_batch[
                :, RAW_EVENT_STATE_INDEX["system_in_vehicle_passengers"]
            ] = states[
                selected, RAW_STATE_INDEX["system_in_vehicle_passengers"]
            ]
            event_batch[:, local_event_indices] = states[
                selected,
                start + 1 : start + 1 + LOCAL_EVENT_STATE_DIM,
            ]
            events.append(event_batch)
    if not events:
        raise ValueError("packed v7 states contain no active event slots")
    return np.concatenate(events)


def correct_packed_headway_jitter(raw_states: np.ndarray) -> np.ndarray:
    corrected = np.asarray(raw_states, dtype=np.float32).copy()
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        start = _slot_start(slot)
        selected = corrected[:, start] > 0.5
        if not np.any(selected):
            continue
        selected_rows = np.flatnonzero(selected)
        for name in ("forward_headway_seconds", "backward_headway_seconds"):
            index = start + 1 + LOCAL_EVENT_STATE_INDEX[name]
            values = corrected[selected_rows, index]
            if np.any(
                values < -HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS
            ):
                raise ValueError(
                    f"packed {name} is below the measurement-jitter tolerance"
                )
            corrected[selected_rows[values < 0.0], index] = 0.0
    return corrected


def validate_collection_matrix(
    manifest_path: Path, shard_paths: list[Path]
) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    matrix_schema = manifest.get("schema_version")
    if matrix_schema not in MATRIX_SCHEMAS:
        raise ValueError(f"unsupported v7 collection matrix: {manifest_path}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("v7 collection matrix contains no jobs")
    design = manifest.get("environment_design", {})
    if matrix_schema == MATRIX_SCHEMA_V2 and not isinstance(design, dict):
        raise ValueError("v7 collection matrix v2 requires an environment design")
    if isinstance(design, dict) and "path" in design:
        design_path = Path(design["path"]).expanduser().resolve()
        if design.get("sha256") != sha256(design_path):
            raise ValueError("v7 collection environment design changed after collection")
    jobs_by_output = {
        Path(job["output"]).expanduser().resolve(): job for job in jobs
    }
    if len(jobs_by_output) != len(jobs):
        raise ValueError("v7 collection matrix repeats an output path")
    actual = {path.expanduser().resolve() for path in shard_paths}
    if actual != set(jobs_by_output):
        raise ValueError("v7 shards do not exactly match the collection matrix")

    cells = set()
    block_environments = {}
    realization_environments = {}
    policies = set()
    for output, job in jobs_by_output.items():
        summary = json.loads(output.with_suffix(output.suffix + ".json").read_text())
        expected = {
            "schema_version": SHARD_SCHEMA,
            "job_signature": job["signature"],
            "policy_name": job["policy"]["policy_name"],
            "policy_seed": int(job["policy_seed"]),
            "sumo_seed": int(job["sumo_seed"]),
            "split": job["split"],
            "diagnostic": bool(job["diagnostic"]),
            "training_eligible": not bool(job["diagnostic"]),
        }
        mismatch = {
            key: (summary.get(key), value)
            for key, value in expected.items()
            if summary.get(key) != value
        }
        if summary.get("source_sha256") != manifest.get("source_sha256"):
            mismatch["source_sha256"] = "matrix mismatch"
        scenario = summary.get("scenario", {})
        if scenario.get("manifest_sha256") != job["scenario"]["sha256"]:
            mismatch["scenario"] = "manifest mismatch"
        if mismatch:
            raise ValueError(f"v7 shard disagrees with matrix: {output}: {mismatch}")
        block_id = int(job["block_id"])
        realization_id = (
            int(job["realization_block_id"])
            if matrix_schema == MATRIX_SCHEMA_V2
            else block_id
        )
        environment = (job["scenario"]["sha256"], int(job["sumo_seed"]))
        if block_environments.setdefault(block_id, environment) != environment:
            raise ValueError(f"v7 block {block_id} maps to multiple environments")
        policy = job["policy"]["policy_name"]
        policies.add(policy)
        cells.add((block_id, policy))
        realization_environments.setdefault(realization_id, set()).add(
            (
                int(job["scenario"]["selection_seed"]),
                float(job["scenario"]["demand_scale"]),
                int(job["sumo_seed"]),
                policy,
            )
        )

    blocks = set(block_environments)
    expected_cells = {(block, policy) for block in blocks for policy in policies}
    if cells != expected_cells:
        raise ValueError("not every v7 environment block contains every policy")
    declared_environment_cells = manifest.get(
        "environment_cells", manifest.get("independent_environment_blocks", -1)
    )
    if len(blocks) != int(declared_environment_cells):
        raise ValueError("v7 matrix environment-cell count is inconsistent")
    realization_ids = {
        int(job["realization_block_id"])
        if matrix_schema == MATRIX_SCHEMA_V2
        else int(job["block_id"])
        for job in jobs
    }
    declared_realization_blocks = manifest.get(
        "independent_realization_blocks",
        manifest.get("independent_environment_blocks", -1),
    )
    if len(realization_ids) != int(declared_realization_blocks):
        raise ValueError("v7 matrix realization-block count is inconsistent")
    expected_scales = {
        float(value)
        for value in (
            design.get("demand_scales", []) if isinstance(design, dict) else []
        )
    }
    if matrix_schema == MATRIX_SCHEMA_V2 and not expected_scales:
        raise ValueError("v7 collection matrix v2 lacks declared demand scales")
    if expected_scales:
        seen_demand_seeds = set()
        seen_sumo_seeds = set()
        for realization_id, rows in realization_environments.items():
            demand_seeds = {row[0] for row in rows}
            sumo_seeds = {row[2] for row in rows}
            conditions = {(row[1], row[3]) for row in rows}
            expected_conditions = {
                (scale, policy) for scale in expected_scales for policy in policies
            }
            if (
                len(demand_seeds) != 1
                or len(sumo_seeds) != 1
                or conditions != expected_conditions
            ):
                raise ValueError(
                    f"v7 realization block {realization_id} is incomplete or mixed"
                )
            demand_seed = next(iter(demand_seeds))
            sumo_seed = next(iter(sumo_seeds))
            if demand_seed in seen_demand_seeds or sumo_seed in seen_sumo_seeds:
                raise ValueError(
                    "v7 independent realization blocks reuse a demand or SUMO seed"
                )
            seen_demand_seeds.add(demand_seed)
            seen_sumo_seeds.add(sumo_seed)
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "schema_version": matrix_schema,
        "jobs": len(jobs),
        "study_stage": manifest.get("study_stage"),
        "environment_design": manifest.get("environment_design"),
        "independent_realization_blocks": len(realization_ids),
        "environment_cells": len(blocks),
        "policies": sorted(policies),
        "collector_sha256": manifest.get("collector_sha256"),
        "runner_sha256": manifest.get("runner_sha256"),
        "source_sha256": manifest.get("source_sha256"),
        "sumo_version": manifest.get("sumo_version"),
    }


def validate_serialized_sequence(arrays: dict[str, np.ndarray], path: Path) -> None:
    observations = arrays["observations"]
    next_observations = arrays["next_observations"]
    actions = arrays["actions"][:, 0]
    terminals = arrays["terminals"].astype(bool)
    durations = np.asarray(
        transition_duration_seconds_v7(observations, next_observations),
        dtype=np.float64,
    )
    if not np.array_equal(durations, arrays["delta_t_seconds"]):
        raise ValueError(f"v7 state clocks disagree with stored durations: {path}")
    if np.any(durations < 0.0):
        raise ValueError(f"v7 transitions move backwards in time: {path}")
    if int(terminals.sum()) != 1 or not terminals[-1]:
        raise ValueError(f"v7 shard must end with exactly one terminal row: {path}")
    if np.any(terminals[:-1]):
        raise ValueError(f"v7 shard terminates before its final row: {path}")

    for endpoint, states in (
        ("observation", observations),
        ("next_observation", next_observations),
    ):
        counts = np.rint(
            states[:, RAW_STATE_INDEX["simultaneous_decision_count"]]
        ).astype(np.int64)
        indices = np.rint(
            states[:, RAW_STATE_INDEX["current_decision_index"]]
        ).astype(np.int64)
        if np.any(counts < 1) or np.any(counts > MAX_SIMULTANEOUS_DECISIONS):
            raise ValueError(f"{endpoint} has an invalid v7 batch count: {path}")
        if np.any(indices < 0) or np.any(indices >= counts):
            raise ValueError(f"{endpoint} has an invalid current slot: {path}")
        for row, (count, current) in enumerate(zip(counts, indices)):
            for slot in range(MAX_SIMULTANEOUS_DECISIONS):
                start = _slot_start(slot)
                expected_active = float(slot < count)
                if states[row, start] != expected_active:
                    raise ValueError(f"{endpoint} active mask is not packed: {path}")
                prior = states[row, start + RAW_SLOT_WIDTH - 1]
                if slot >= current and prior != -1.0:
                    raise ValueError(f"{endpoint} contains a future action: {path}")

    current_indices = np.rint(
        observations[:, RAW_STATE_INDEX["current_decision_index"]]
    ).astype(np.int64)
    event_rows = np.arange(len(observations), dtype=np.int64)
    event_slot_starts = (
        len(RAW_PREFIX_NAMES) + current_indices * RAW_SLOT_WIDTH
    )
    if np.any(arrays["event_bus_code"] < 0):
        raise ValueError(f"v7 event bus codes are negative: {path}")
    event_checks = {
        "line index": (
            arrays["event_line_index"],
            observations[
                event_rows,
                event_slot_starts
                + 1
                + LOCAL_EVENT_STATE_INDEX["line_id"],
            ],
        ),
        "station index": (
            arrays["event_station_index"],
            observations[
                event_rows,
                event_slot_starts
                + 1
                + LOCAL_EVENT_STATE_INDEX["station_index"],
            ],
        ),
        "decision time": (
            arrays["event_decision_time_seconds"],
            observations[:, RAW_STATE_INDEX["decision_time_seconds"]],
        ),
        "base dwell": (
            arrays["event_base_dwell_seconds"],
            observations[
                event_rows,
                event_slot_starts
                + 1
                + LOCAL_EVENT_STATE_INDEX["base_dwell_seconds"],
            ],
        ),
    }
    for name, (actual, expected) in event_checks.items():
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"v7 event {name} metadata disagrees with state: {path}"
            )

    zero = (durations == 0.0) & ~terminals
    if np.any(arrays["rewards"][zero] != 0.0):
        raise ValueError(f"zero-time v7 rows have nonzero reward: {path}")
    if np.any(arrays["discounts"][zero] != 1.0):
        raise ValueError(f"zero-time v7 rows do not have unit discount: {path}")
    for row in np.flatnonzero(zero):
        current = int(
            round(
                float(
                    observations[
                        row, RAW_STATE_INDEX["current_decision_index"]
                    ]
                )
            )
        )
        next_current = int(
            round(
                float(
                    next_observations[
                        row, RAW_STATE_INDEX["current_decision_index"]
                    ]
                )
            )
        )
        if next_current != current + 1:
            raise ValueError(f"zero-time v7 row skips a batch token: {path}")
        same_batch_names = (
            "decision_time_seconds",
            "simultaneous_decision_count",
            "system_waiting_passengers",
            "system_in_vehicle_passengers",
        )
        same_batch_indices = [RAW_STATE_INDEX[name] for name in same_batch_names]
        np.testing.assert_array_equal(
            observations[row, same_batch_indices],
            next_observations[row, same_batch_indices],
            err_msg=f"same-time global batch state changed in {path}",
        )
        count = int(
            round(
                float(
                    observations[
                        row, RAW_STATE_INDEX["simultaneous_decision_count"]
                    ]
                )
            )
        )
        for slot in range(count):
            start = _slot_start(slot)
            np.testing.assert_array_equal(
                observations[
                    row, start + 1 : start + 1 + LOCAL_EVENT_STATE_DIM
                ],
                next_observations[
                    row, start + 1 : start + 1 + LOCAL_EVENT_STATE_DIM
                ],
                err_msg=f"same-time event batch changed in {path}",
            )
        action_position = _slot_start(current) + RAW_SLOT_WIDTH - 1
        if next_observations[row, action_position] != actions[row]:
            raise ValueError(f"v7 next token lost the preceding action: {path}")

    positive_nonterminal = (durations > 0.0) & ~terminals
    if np.any(
        next_observations[
            positive_nonterminal,
            RAW_STATE_INDEX["current_decision_index"],
        ]
        != 0.0
    ):
        raise ValueError(f"a new v7 event batch does not start at slot zero: {path}")


def load_shard(path: Path, *, allow_partial: bool) -> tuple[dict, dict]:
    path = path.expanduser().resolve()
    summary_path = path.with_suffix(path.suffix + ".json")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != SHARD_SCHEMA:
        raise ValueError(f"unsupported v7 shard schema: {path}")
    if summary.get("contract_schema") != SCHEMA_VERSION:
        raise ValueError(f"v7 shard contract mismatch: {path}")
    if summary.get("contract") != DEFAULT_CONFIG_V7.to_dict():
        raise ValueError(f"v7 shard configuration mismatch: {path}")
    if summary.get("raw_state_names") != list(RAW_STATE_NAMES):
        raise ValueError(f"v7 raw-state semantics mismatch: {path}")
    bus_ids = summary.get("event_identity", {}).get("bus_code_to_id")
    line_ids = summary.get("event_identity", {}).get("line_index_to_id")
    if (
        not isinstance(bus_ids, list)
        or not bus_ids
        or len(set(bus_ids)) != len(bus_ids)
        or any(not isinstance(value, str) or not value for value in bus_ids)
    ):
        raise ValueError(f"v7 event-bus identity mapping is invalid: {path}")
    if (
        not isinstance(line_ids, list)
        or not line_ids
        or len(set(line_ids)) != len(line_ids)
        or any(not isinstance(value, str) or not value for value in line_ids)
    ):
        raise ValueError(f"v7 event-line identity mapping is invalid: {path}")
    if summary.get("shard_sha256") != sha256(path):
        raise ValueError(f"v7 shard hash mismatch: {path}")
    count_audit = summary.get("ledger_audit", {}).get("count_source_audit", {})
    if count_audit.get("max_waiting_count_error") != 0 or count_audit.get(
        "max_in_vehicle_count_error"
    ) != 0:
        raise ValueError(f"v7 passenger-count audit failed: {path}")
    coverage = summary.get("training_reward", {}).get(
        "chronological_coverage_errors", {}
    )
    if set(coverage) != {
        "wait_passenger_seconds",
        "in_vehicle_passenger_seconds",
        "reward",
    } or any(abs(float(value)) > 1e-9 for value in coverage.values()):
        raise ValueError(f"v7 chronological welfare identity failed: {path}")
    episode = summary["episode"]
    if not allow_partial:
        if summary.get("diagnostic") or not summary.get("training_eligible"):
            raise ValueError(f"diagnostic v7 shard is not training eligible: {path}")
        if episode["configured_seconds"] != DEFAULT_CONFIG_V7.horizon_seconds:
            raise ValueError(f"v7 shard uses a non-contract horizon: {path}")
        if episode["observed_end_time_seconds"] != episode["configured_seconds"]:
            raise ValueError(f"v7 shard did not reach its horizon: {path}")
        service = summary["service"]
        if service["completed_trips"] != service["departed_trips"]:
            raise ValueError(f"not every v7 SUMO trip completed: {path}")

    with np.load(path, allow_pickle=False) as handle:
        missing = sorted(set(REQUIRED_ARRAYS).difference(handle.files))
        if missing:
            raise KeyError(f"{path} is missing arrays {missing}")
        arrays = {name: np.asarray(handle[name]) for name in REQUIRED_ARRAYS}
    n = len(arrays["rewards"])
    expected_shapes = {
        "observations": (n, RAW_STATE_DIM),
        "actions": (n, ACTION_DIM),
        "rewards": (n,),
        "next_observations": (n, RAW_STATE_DIM),
        "discounts": (n,),
        "terminals": (n,),
        "delta_t_seconds": (n,),
        "trajectory_local_id": (n,),
        "wait_passenger_seconds": (n,),
        "in_vehicle_passenger_seconds": (n,),
        "event_bus_code": (n,),
        "event_line_index": (n,),
        "event_station_index": (n,),
        "event_decision_time_seconds": (n,),
        "event_base_dwell_seconds": (n,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}:{name} has invalid shape or values")
    if n != episode["transitions"] or n != episode["control_decisions"]:
        raise ValueError(f"v7 row/decision counts disagree: {path}")
    if np.any(np.abs(arrays["actions"]) > 1.00001):
        raise ValueError(f"v7 actions lie outside [-1,1]: {path}")
    if np.any(arrays["event_bus_code"] >= len(bus_ids)):
        raise ValueError(f"v7 event bus code exceeds its identity mapping: {path}")
    if np.any(arrays["event_line_index"] < 0) or np.any(
        arrays["event_line_index"] >= len(line_ids)
    ):
        raise ValueError(f"v7 event line index exceeds its identity mapping: {path}")
    if np.any(arrays["wait_passenger_seconds"] < 0.0) or np.any(
        arrays["in_vehicle_passenger_seconds"] < 0.0
    ):
        raise ValueError(f"v7 passenger-time components are negative: {path}")
    validate_serialized_sequence(arrays, path)

    reward = -(
        DEFAULT_CONFIG_V7.waiting_time_value
        * arrays["wait_passenger_seconds"]
        + DEFAULT_CONFIG_V7.in_vehicle_time_value
        * arrays["in_vehicle_passenger_seconds"]
    ) / DEFAULT_CONFIG_V7.reward_unit_passenger_seconds
    discount = np.asarray(
        continuous_time_discount_v7(arrays["delta_t_seconds"]), np.float32
    )
    discount[arrays["terminals"].astype(bool)] = 0.0
    if not np.array_equal(reward, arrays["rewards"]):
        raise ValueError(f"v7 rewards do not reconstruct exactly: {path}")
    if not np.array_equal(discount, arrays["discounts"]):
        raise ValueError(f"v7 discounts do not reconstruct exactly: {path}")
    return summary, arrays


def _create_empty_dataset(
    handle: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> h5py.Dataset:
    rows = max(int(shape[0]), 1)
    chunks = (min(rows, 8192),) + tuple(shape[1:])
    return handle.create_dataset(
        name,
        shape=shape,
        dtype=np.dtype(dtype),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=chunks,
    )


def _dataset_content_sha256_hdf5(
    handle: h5py.File,
    metadata: dict,
    *,
    chunk_rows: int = 8192,
) -> str:
    """Hash logical HDF5 contents without materializing the merged arrays."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for name in sorted(handle.keys()):
        dataset = handle[name]
        header = {
            "name": name,
            "dtype": dataset.dtype.str,
            "shape": dataset.shape,
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        for start in range(0, int(dataset.shape[0]), int(chunk_rows)):
            values = np.ascontiguousarray(dataset[start : start + chunk_rows])
            digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _derived_shard_arrays(
    arrays: dict[str, np.ndarray], station_max: np.ndarray
) -> dict[str, np.ndarray]:
    raw_observations = correct_packed_headway_jitter(arrays["observations"])
    raw_next_observations = correct_packed_headway_jitter(
        arrays["next_observations"]
    )
    wait_seconds = np.asarray(
        arrays["wait_passenger_seconds"], dtype=np.float64
    )
    in_vehicle_seconds = np.asarray(
        arrays["in_vehicle_passenger_seconds"], dtype=np.float64
    )
    waiting_cost = (
        DEFAULT_CONFIG_V7.waiting_time_value
        * wait_seconds
        / DEFAULT_CONFIG_V7.reward_unit_passenger_seconds
    )
    in_vehicle_cost = (
        DEFAULT_CONFIG_V7.in_vehicle_time_value
        * in_vehicle_seconds
        / DEFAULT_CONFIG_V7.reward_unit_passenger_seconds
    )
    delta_t = np.asarray(
        transition_duration_seconds_v7(
            raw_observations, raw_next_observations
        ),
        dtype=np.float64,
    )
    terminals = np.asarray(arrays["terminals"], dtype=np.uint8)
    discounts = np.asarray(continuous_time_discount_v7(delta_t), np.float32)
    discounts[terminals.astype(bool)] = 0.0
    return {
        "observations": raw_state_to_canonical_v7(
            raw_observations, station_max
        ),
        "actions": np.asarray(arrays["actions"], dtype=np.float32),
        "rewards": -(waiting_cost + in_vehicle_cost),
        "next_observations": raw_state_to_canonical_v7(
            raw_next_observations, station_max
        ),
        "discounts": discounts,
        "terminals": terminals,
        "raw_observations": raw_observations,
        "raw_next_observations": raw_next_observations,
        "wait_passenger_seconds": wait_seconds,
        "in_vehicle_passenger_seconds": in_vehicle_seconds,
        "waiting_cost": waiting_cost,
        "in_vehicle_cost": in_vehicle_cost,
        "delta_t_seconds": delta_t,
        "event_bus_code": np.asarray(
            arrays["event_bus_code"], dtype=np.int32
        ),
        "event_line_index": np.asarray(
            arrays["event_line_index"], dtype=np.int16
        ),
        "event_station_index": np.asarray(
            arrays["event_station_index"], dtype=np.int16
        ),
        "event_decision_time_seconds": np.asarray(
            arrays["event_decision_time_seconds"], dtype=np.float64
        ),
        "event_base_dwell_seconds": np.asarray(
            arrays["event_base_dwell_seconds"], dtype=np.float64
        ),
    }


def _dataset_specs(n_rows: int) -> dict[str, tuple[tuple[int, ...], type]]:
    vector = (int(n_rows),)
    return {
        "observations": ((n_rows, len(CANONICAL_STATE_NAMES)), np.float32),
        "actions": ((n_rows, ACTION_DIM), np.float32),
        "rewards": (vector, np.float64),
        "next_observations": (
            (n_rows, len(CANONICAL_STATE_NAMES)),
            np.float32,
        ),
        "discounts": (vector, np.float32),
        "terminals": (vector, np.uint8),
        "raw_observations": ((n_rows, RAW_STATE_DIM), np.float32),
        "raw_next_observations": ((n_rows, RAW_STATE_DIM), np.float32),
        "wait_passenger_seconds": (vector, np.float64),
        "in_vehicle_passenger_seconds": (vector, np.float64),
        "waiting_cost": (vector, np.float64),
        "in_vehicle_cost": (vector, np.float64),
        "delta_t_seconds": (vector, np.float64),
        "event_bus_code": (vector, np.int32),
        "event_line_index": (vector, np.int16),
        "event_station_index": (vector, np.int16),
        "event_decision_time_seconds": (vector, np.float64),
        "event_base_dwell_seconds": (vector, np.float64),
        "policy_id": (vector, np.int8),
        "policy_seed": (vector, np.int32),
        "sumo_seed": (vector, np.int32),
        "split_id": (vector, np.uint8),
        "episode_id": (vector, np.int64),
        "trajectory_id": (vector, np.int64),
        "scenario_id": (vector, np.int16),
        "demand_realization_seed": (vector, np.int32),
        "demand_scale": (vector, np.float32),
        "source_shard_id": (vector, np.int16),
    }


def build_dataset(
    shard_paths: list[Path],
    output: Path,
    *,
    allow_partial: bool = False,
    matrix_manifest: Path | None = None,
) -> dict:
    if not shard_paths:
        raise ValueError("at least one v7 shard is required")
    matrix = (
        validate_collection_matrix(matrix_manifest, shard_paths)
        if matrix_manifest is not None
        else None
    )
    records = []
    station_max = None
    for source_path in sorted(Path(path).resolve() for path in shard_paths):
        summary, arrays = load_shard(
            source_path, allow_partial=allow_partial
        )
        endpoint_maxima = []
        for endpoint in ("observations", "next_observations"):
            corrected = correct_packed_headway_jitter(arrays[endpoint])
            endpoint_maxima.append(
                infer_station_max_v7(active_event_observations(corrected))
            )
        shard_max = np.maximum(endpoint_maxima[0], endpoint_maxima[1])
        station_max = (
            shard_max
            if station_max is None
            else np.maximum(station_max, shard_max)
        )
        records.append((source_path, summary, len(arrays["rewards"])))
        del arrays, corrected, endpoint_maxima, shard_max

    if station_max is None:
        raise ValueError("merged dataset would be empty")
    n_rows = sum(rows for _, _, rows in records)
    if n_rows == 0:
        raise ValueError("merged dataset would be empty")
    if len(records) > np.iinfo(np.int16).max + 1:
        raise ValueError("v7 source-shard identifiers exceed int16 capacity")

    policy_names = sorted({summary["policy_name"] for _, summary, _ in records})
    policy_to_id = {name: index for index, name in enumerate(policy_names)}
    if len(policy_to_id) > np.iinfo(np.int8).max + 1:
        raise ValueError("v7 policy identifiers exceed int8 capacity")
    scenario_hashes = sorted(
        {summary["scenario"]["manifest_sha256"] for _, summary, _ in records}
    )
    scenario_to_id = {
        digest: index for index, digest in enumerate(scenario_hashes)
    }
    if len(scenario_to_id) > np.iinfo(np.int16).max + 1:
        raise ValueError("v7 scenario identifiers exceed int16 capacity")

    source_shards = [
        {
            "path": str(path),
            "sha256": summary["shard_sha256"],
            "job_signature": summary.get("job_signature"),
            "event_bus_code_to_id": summary["event_identity"][
                "bus_code_to_id"
            ],
            "event_line_index_to_id": summary["event_identity"][
                "line_index_to_id"
            ],
        }
        for path, summary, _ in records
    ]
    provenance = {
        "transform_version": BUILD_TRANSFORM_VERSION,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "contract": {
            "path": str(ROOT / "bus_h2o" / "mdp_contract_v7.py"),
            "sha256": sha256(ROOT / "bus_h2o" / "mdp_contract_v7.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "assembly": {
            "strategy": "two-pass streaming",
            "source_shards_loaded_concurrently": 1,
            "logical_content_hash": "streamed from HDF5 datasets",
        },
        "source_shard_sha256": [item["sha256"] for item in source_shards],
        "collection_matrix": matrix,
    }
    derivation_signature = canonical_sha256(provenance)
    provenance["derivation_signature"] = derivation_signature
    content_metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": DEFAULT_CONFIG_V7.to_dict(),
        "state_names": list(CANONICAL_STATE_NAMES),
        "raw_state_names": list(RAW_STATE_NAMES),
        "policy_names": policy_to_id,
        "split_names": SPLIT_IDS,
        "scenario_manifest_hashes": scenario_to_id,
        "station_max_by_line": station_max.tolist(),
    }

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    terminal_rows = 0
    zero_duration_rows = 0
    exact_zero_actions = 0
    reward_sum = 0.0
    reward_min = float("inf")
    reward_max = float("-inf")
    try:
        with h5py.File(temporary, "w") as handle:
            for name, (shape, dtype) in _dataset_specs(n_rows).items():
                _create_empty_dataset(handle, name, shape, dtype)

            offset = 0
            for episode_id, (path, summary, expected_rows) in enumerate(records):
                _, arrays = load_shard(path, allow_partial=allow_partial)
                derived = _derived_shard_arrays(arrays, station_max)
                if any(
                    values.shape[0] != expected_rows
                    for values in derived.values()
                ):
                    raise RuntimeError(
                        f"derived v7 shard arrays have inconsistent rows: {path}"
                    )
                row_slice = slice(offset, offset + expected_rows)
                for name, values in derived.items():
                    handle[name][row_slice] = values

                scenario_hash = summary["scenario"]["manifest_sha256"]
                scalar_metadata = {
                    "policy_id": policy_to_id[summary["policy_name"]],
                    "policy_seed": int(summary["policy_seed"]),
                    "sumo_seed": int(summary["sumo_seed"]),
                    "split_id": SPLIT_IDS[summary["split"]],
                    "episode_id": episode_id,
                    "trajectory_id": episode_id,
                    "scenario_id": scenario_to_id[scenario_hash],
                    "demand_realization_seed": int(
                        summary["scenario"]["selection_seed"]
                    ),
                    "demand_scale": float(
                        summary["scenario"]["demand_scale"]
                    ),
                    "source_shard_id": episode_id,
                }
                for name, value in scalar_metadata.items():
                    handle[name][row_slice] = value

                rewards = derived["rewards"]
                terminal_rows += int(np.sum(derived["terminals"]))
                zero_duration_rows += int(
                    np.sum(derived["delta_t_seconds"] == 0.0)
                )
                exact_zero_actions += int(
                    np.sum(derived["actions"][:, 0] == -1.0)
                )
                reward_sum += float(np.sum(rewards, dtype=np.float64))
                reward_min = min(reward_min, float(np.min(rewards)))
                reward_max = max(reward_max, float(np.max(rewards)))
                offset += expected_rows
                del arrays, derived, rewards

            if offset != n_rows:
                raise RuntimeError("streamed v7 row count changed during build")
            handle.flush()
            content_signature = _dataset_content_sha256_hdf5(
                handle, content_metadata
            )

            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["content_signature"] = content_signature
            handle.attrs["derivation_signature"] = derivation_signature
            handle.attrs["build_provenance"] = json.dumps(
                provenance, sort_keys=True
            )
            handle.attrs["config"] = json.dumps(
                DEFAULT_CONFIG_V7.to_dict(), sort_keys=True
            )
            handle.attrs["state_dim"] = len(CANONICAL_STATE_NAMES)
            handle.attrs["state_names"] = json.dumps(
                list(CANONICAL_STATE_NAMES)
            )
            handle.attrs["raw_state_dim"] = RAW_STATE_DIM
            handle.attrs["raw_state_names"] = json.dumps(list(RAW_STATE_NAMES))
            handle.attrs["action_dim"] = ACTION_DIM
            handle.attrs["action_semantics"] = (
                "one AEC token action: a=-1 is exact zero hold; -1<a<=1 maps "
                "linearly to (0,60] seconds"
            )
            handle.attrs["reward_semantics"] = (
                "negative non-overlapping generalized passenger-seconds between "
                "chronologically consecutive centralized decision tokens; "
                "training scale is external to the dataset"
            )
            handle.attrs["transition_semantics"] = (
                "fixed-width simultaneous event batch serialized as AEC tokens; "
                "same-time intermediate rows preserve prior actions"
            )
            handle.attrs["discount_semantics"] = (
                "finite-horizon undiscounted objective: nonterminal discount is 1, "
                "including at zero duration; terminal discount is 0"
            )
            handle.attrs["station_max_by_line"] = json.dumps(
                station_max.tolist()
            )
            handle.attrs["policy_names"] = json.dumps(
                policy_to_id, sort_keys=True
            )
            handle.attrs["split_names"] = json.dumps(SPLIT_IDS, sort_keys=True)
            handle.attrs["scenario_manifest_hashes"] = json.dumps(
                scenario_to_id, sort_keys=True
            )
            handle.attrs["n_transitions"] = n_rows
            handle.attrs["source_shards"] = json.dumps(
                source_shards, sort_keys=True
            )
            if matrix is not None:
                handle.attrs["collection_matrix"] = json.dumps(
                    matrix, sort_keys=True
                )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    summary = {
        "schema_version": "transit-smdp-v7-dataset-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output),
        "dataset_sha256": sha256(output),
        "content_signature": content_signature,
        "derivation_signature": derivation_signature,
        "rows": n_rows,
        "episodes": len(records),
        "trajectories": len(records),
        "state_dim": len(CANONICAL_STATE_NAMES),
        "raw_state_dim": RAW_STATE_DIM,
        "terminal_rows": terminal_rows,
        "zero_duration_rows": zero_duration_rows,
        "exact_zero_action_rate": exact_zero_actions / n_rows,
        "reward": {
            "mean": reward_sum / n_rows,
            "min": reward_min,
            "max": reward_max,
            "sum": reward_sum,
        },
        "station_max_by_line": station_max.tolist(),
        "source_shards": len(records),
        "formal_full_horizon_only": not allow_partial,
        "collection_matrix": matrix,
        "build_provenance": provenance,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument("--allow-unmanifested", action="store_true")
    args = parser.parse_args()
    if args.shard_root is None and args.matrix_manifest is None:
        raise ValueError("provide --matrix-manifest or --shard-root")
    default_manifest = (
        args.shard_root.expanduser().resolve() / "matrix_manifest.json"
        if args.shard_root is not None
        else None
    )
    matrix_manifest = args.matrix_manifest or default_manifest
    if matrix_manifest is None or not matrix_manifest.is_file():
        if not args.allow_unmanifested:
            raise FileNotFoundError("formal v7 builds require a matrix manifest")
        if args.shard_root is None:
            raise ValueError("unmanifested v7 builds require --shard-root")
        matrix_manifest = None
    shard_paths = (
        shard_paths_from_manifest(matrix_manifest)
        if matrix_manifest is not None
        else discover_shards(args.shard_root)
    )
    summary = build_dataset(
        shard_paths,
        args.output,
        allow_partial=args.allow_partial,
        matrix_manifest=matrix_manifest,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
