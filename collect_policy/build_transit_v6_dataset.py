#!/usr/bin/env python3
"""Validate and merge chronological SUMO v6 shards into immutable HDF5."""

from __future__ import annotations

import argparse
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

from bus_h2o.mdp_contract_v4 import (
    RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4,
    correct_headway_measurement_jitter_v4,
    infer_station_max_v4,
)
from bus_h2o.mdp_contract_v6 import (
    ACTION_DIM,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V6,
    MAX_SIMULTANEOUS_DECISIONS,
    RAW_EVENT_STATE_DIM,
    RAW_SLOT_WIDTH,
    RAW_STATE_DIM,
    RAW_STATE_INDEX,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v6,
    raw_state_to_canonical_v6,
    transition_duration_seconds_v6,
)
from collect_policy.build_transit_v4_dataset import (
    SPLIT_IDS,
    _create_dataset,
    canonical_sha256,
    dataset_content_sha256,
    discover_shards,
    sha256,
)


SHARD_SCHEMA = "transit-smdp-v6-shard-v1"
MATRIX_SCHEMA = "transit-smdp-v6-collection-matrix-v1"
MATRIX_SCHEMA_V2 = "transit-smdp-v6-collection-matrix-v2"
MATRIX_SCHEMAS = {MATRIX_SCHEMA, MATRIX_SCHEMA_V2}
BUILD_TRANSFORM_VERSION = "transit-smdp-v6-derived-transform-v1"
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
)


def _slot_start(slot: int) -> int:
    return 3 + int(slot) * RAW_SLOT_WIDTH


def active_event_observations(raw_states: np.ndarray) -> np.ndarray:
    states = np.asarray(raw_states, dtype=np.float32)
    events = []
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        start = _slot_start(slot)
        selected = states[:, start] > 0.5
        if np.any(selected):
            events.append(
                states[
                    selected,
                    start + 1 : start + 1 + RAW_EVENT_STATE_DIM,
                ]
            )
    if not events:
        raise ValueError("packed v6 states contain no active event slots")
    return np.concatenate(events)


def correct_packed_headway_jitter(raw_states: np.ndarray) -> np.ndarray:
    corrected = np.asarray(raw_states, dtype=np.float32).copy()
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


def validate_collection_matrix(
    manifest_path: Path, shard_paths: list[Path]
) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    matrix_schema = manifest.get("schema_version")
    if matrix_schema not in MATRIX_SCHEMAS:
        raise ValueError(f"unsupported v6 collection matrix: {manifest_path}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("v6 collection matrix contains no jobs")
    design = manifest.get("environment_design", {})
    if matrix_schema == MATRIX_SCHEMA_V2 and not isinstance(design, dict):
        raise ValueError("v6 collection matrix v2 requires an environment design")
    if isinstance(design, dict) and "path" in design:
        design_path = Path(design["path"]).expanduser().resolve()
        if design.get("sha256") != sha256(design_path):
            raise ValueError("v6 collection environment design changed after collection")
    jobs_by_output = {
        Path(job["output"]).expanduser().resolve(): job for job in jobs
    }
    if len(jobs_by_output) != len(jobs):
        raise ValueError("v6 collection matrix repeats an output path")
    actual = {path.expanduser().resolve() for path in shard_paths}
    if actual != set(jobs_by_output):
        raise ValueError("v6 shards do not exactly match the collection matrix")

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
            raise ValueError(f"v6 shard disagrees with matrix: {output}: {mismatch}")
        block_id = int(job["block_id"])
        realization_id = (
            int(job["realization_block_id"])
            if matrix_schema == MATRIX_SCHEMA_V2
            else block_id
        )
        environment = (job["scenario"]["sha256"], int(job["sumo_seed"]))
        if block_environments.setdefault(block_id, environment) != environment:
            raise ValueError(f"v6 block {block_id} maps to multiple environments")
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
        raise ValueError("not every v6 environment block contains every policy")
    declared_environment_cells = manifest.get(
        "environment_cells", manifest.get("independent_environment_blocks", -1)
    )
    if len(blocks) != int(declared_environment_cells):
        raise ValueError("v6 matrix environment-cell count is inconsistent")
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
        raise ValueError("v6 matrix realization-block count is inconsistent")
    expected_scales = {
        float(value)
        for value in (
            design.get("demand_scales", []) if isinstance(design, dict) else []
        )
    }
    if matrix_schema == MATRIX_SCHEMA_V2 and not expected_scales:
        raise ValueError("v6 collection matrix v2 lacks declared demand scales")
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
                    f"v6 realization block {realization_id} is incomplete or mixed"
                )
            demand_seed = next(iter(demand_seeds))
            sumo_seed = next(iter(sumo_seeds))
            if demand_seed in seen_demand_seeds or sumo_seed in seen_sumo_seeds:
                raise ValueError(
                    "v6 independent realization blocks reuse a demand or SUMO seed"
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
        transition_duration_seconds_v6(observations, next_observations),
        dtype=np.float64,
    )
    if not np.array_equal(durations, arrays["delta_t_seconds"]):
        raise ValueError(f"v6 state clocks disagree with stored durations: {path}")
    if np.any(durations < 0.0):
        raise ValueError(f"v6 transitions move backwards in time: {path}")
    if int(terminals.sum()) != 1 or not terminals[-1]:
        raise ValueError(f"v6 shard must end with exactly one terminal row: {path}")
    if np.any(terminals[:-1]):
        raise ValueError(f"v6 shard terminates before its final row: {path}")

    for endpoint, states in (
        ("observation", observations),
        ("next_observation", next_observations),
    ):
        counts = np.rint(states[:, 1]).astype(np.int64)
        indices = np.rint(states[:, 2]).astype(np.int64)
        if np.any(counts < 1) or np.any(counts > MAX_SIMULTANEOUS_DECISIONS):
            raise ValueError(f"{endpoint} has an invalid v6 batch count: {path}")
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

    zero = (durations == 0.0) & ~terminals
    if np.any(arrays["rewards"][zero] != 0.0):
        raise ValueError(f"zero-time v6 rows have nonzero reward: {path}")
    if np.any(arrays["discounts"][zero] != 1.0):
        raise ValueError(f"zero-time v6 rows do not have unit discount: {path}")
    for row in np.flatnonzero(zero):
        current = int(round(float(observations[row, 2])))
        next_current = int(round(float(next_observations[row, 2])))
        if next_current != current + 1:
            raise ValueError(f"zero-time v6 row skips a batch token: {path}")
        count = int(round(float(observations[row, 1])))
        for slot in range(count):
            start = _slot_start(slot)
            np.testing.assert_array_equal(
                observations[
                    row, start + 1 : start + 1 + RAW_EVENT_STATE_DIM
                ],
                next_observations[
                    row, start + 1 : start + 1 + RAW_EVENT_STATE_DIM
                ],
                err_msg=f"same-time event batch changed in {path}",
            )
        action_position = _slot_start(current) + RAW_SLOT_WIDTH - 1
        if next_observations[row, action_position] != actions[row]:
            raise ValueError(f"v6 next token lost the preceding action: {path}")

    positive_nonterminal = (durations > 0.0) & ~terminals
    if np.any(next_observations[positive_nonterminal, 2] != 0.0):
        raise ValueError(f"a new v6 event batch does not start at slot zero: {path}")


def load_shard(path: Path, *, allow_partial: bool) -> tuple[dict, dict]:
    path = path.expanduser().resolve()
    summary_path = path.with_suffix(path.suffix + ".json")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != SHARD_SCHEMA:
        raise ValueError(f"unsupported v6 shard schema: {path}")
    if summary.get("contract_schema") != SCHEMA_VERSION:
        raise ValueError(f"v6 shard contract mismatch: {path}")
    if summary.get("contract") != DEFAULT_CONFIG_V6.to_dict():
        raise ValueError(f"v6 shard configuration mismatch: {path}")
    if summary.get("raw_state_names") != list(RAW_STATE_NAMES):
        raise ValueError(f"v6 raw-state semantics mismatch: {path}")
    if summary.get("shard_sha256") != sha256(path):
        raise ValueError(f"v6 shard hash mismatch: {path}")
    count_audit = summary.get("ledger_audit", {}).get("count_source_audit", {})
    if count_audit.get("max_waiting_count_error") != 0 or count_audit.get(
        "max_in_vehicle_count_error"
    ) != 0:
        raise ValueError(f"v6 passenger-count audit failed: {path}")
    coverage = summary.get("training_reward", {}).get(
        "chronological_coverage_errors", {}
    )
    if set(coverage) != {
        "wait_passenger_seconds",
        "in_vehicle_passenger_seconds",
        "reward",
    } or any(abs(float(value)) > 1e-9 for value in coverage.values()):
        raise ValueError(f"v6 chronological welfare identity failed: {path}")
    episode = summary["episode"]
    if not allow_partial:
        if summary.get("diagnostic") or not summary.get("training_eligible"):
            raise ValueError(f"diagnostic v6 shard is not training eligible: {path}")
        if episode["configured_seconds"] != DEFAULT_CONFIG_V6.episode_seconds:
            raise ValueError(f"v6 shard uses a non-contract horizon: {path}")
        if episode["observed_end_time_seconds"] != episode["configured_seconds"]:
            raise ValueError(f"v6 shard did not reach its horizon: {path}")
        service = summary["service"]
        if service["completed_trips"] != service["departed_trips"]:
            raise ValueError(f"not every v6 SUMO trip completed: {path}")

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
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}:{name} has invalid shape or values")
    if n != episode["transitions"] or n != episode["control_decisions"]:
        raise ValueError(f"v6 row/decision counts disagree: {path}")
    if np.any(np.abs(arrays["actions"]) > 1.00001):
        raise ValueError(f"v6 actions lie outside [-1,1]: {path}")
    if np.any(arrays["wait_passenger_seconds"] < 0.0) or np.any(
        arrays["in_vehicle_passenger_seconds"] < 0.0
    ):
        raise ValueError(f"v6 passenger-time components are negative: {path}")
    validate_serialized_sequence(arrays, path)

    reward = -(
        DEFAULT_CONFIG_V6.waiting_time_value
        * arrays["wait_passenger_seconds"]
        + DEFAULT_CONFIG_V6.in_vehicle_time_value
        * arrays["in_vehicle_passenger_seconds"]
    ) / DEFAULT_CONFIG_V6.reward_unit_passenger_seconds
    discount = np.asarray(
        continuous_time_discount_v6(arrays["delta_t_seconds"]), np.float32
    )
    discount[arrays["terminals"].astype(bool)] = 0.0
    if not np.array_equal(reward, arrays["rewards"]):
        raise ValueError(f"v6 rewards do not reconstruct exactly: {path}")
    if not np.array_equal(discount, arrays["discounts"]):
        raise ValueError(f"v6 discounts do not reconstruct exactly: {path}")
    return summary, arrays


def build_dataset(
    shard_paths: list[Path],
    output: Path,
    *,
    allow_partial: bool = False,
    matrix_manifest: Path | None = None,
) -> dict:
    if not shard_paths:
        raise ValueError("at least one v6 shard is required")
    matrix = (
        validate_collection_matrix(matrix_manifest, shard_paths)
        if matrix_manifest is not None
        else None
    )
    loaded = [
        (path.resolve(), *load_shard(path, allow_partial=allow_partial))
        for path in sorted(shard_paths)
    ]
    raw_observations = correct_packed_headway_jitter(
        np.concatenate([arrays["observations"] for _, _, arrays in loaded])
    )
    raw_next_observations = correct_packed_headway_jitter(
        np.concatenate([arrays["next_observations"] for _, _, arrays in loaded])
    )
    local_events = np.concatenate(
        (
            active_event_observations(raw_observations),
            active_event_observations(raw_next_observations),
        )
    )
    station_max = infer_station_max_v4(
        local_events[:, :RAW_EVENT_STATE_DIM_V4]
    )
    observations = raw_state_to_canonical_v6(raw_observations, station_max)
    next_observations = raw_state_to_canonical_v6(
        raw_next_observations, station_max
    )
    actions = np.concatenate([arrays["actions"] for _, _, arrays in loaded]).astype(
        np.float32, copy=False
    )
    wait_seconds = np.concatenate(
        [arrays["wait_passenger_seconds"] for _, _, arrays in loaded]
    ).astype(np.float64, copy=False)
    in_vehicle_seconds = np.concatenate(
        [arrays["in_vehicle_passenger_seconds"] for _, _, arrays in loaded]
    ).astype(np.float64, copy=False)
    waiting_cost = (
        DEFAULT_CONFIG_V6.waiting_time_value
        * wait_seconds
        / DEFAULT_CONFIG_V6.reward_unit_passenger_seconds
    )
    in_vehicle_cost = (
        DEFAULT_CONFIG_V6.in_vehicle_time_value
        * in_vehicle_seconds
        / DEFAULT_CONFIG_V6.reward_unit_passenger_seconds
    )
    rewards = -(waiting_cost + in_vehicle_cost)
    delta_t = np.asarray(
        transition_duration_seconds_v6(raw_observations, raw_next_observations),
        np.float64,
    )
    terminals = np.concatenate(
        [arrays["terminals"] for _, _, arrays in loaded]
    ).astype(np.uint8, copy=False)
    discounts = np.asarray(continuous_time_discount_v6(delta_t), np.float32)
    discounts[terminals.astype(bool)] = 0.0

    policy_names = sorted({summary["policy_name"] for _, summary, _ in loaded})
    policy_to_id = {name: index for index, name in enumerate(policy_names)}
    scenario_hashes = sorted(
        {summary["scenario"]["manifest_sha256"] for _, summary, _ in loaded}
    )
    scenario_to_id = {
        digest: index for index, digest in enumerate(scenario_hashes)
    }
    metadata = {name: [] for name in (
        "policy_id", "policy_seed", "sumo_seed", "split_id", "episode_id",
        "trajectory_id", "scenario_id", "demand_realization_seed",
        "demand_scale", "source_shard_id",
    )}
    for episode_id, (_, summary, shard) in enumerate(loaded):
        n = len(shard["rewards"])
        metadata["policy_id"].append(
            np.full(n, policy_to_id[summary["policy_name"]], np.int8)
        )
        metadata["policy_seed"].append(
            np.full(n, int(summary["policy_seed"]), np.int32)
        )
        metadata["sumo_seed"].append(
            np.full(n, int(summary["sumo_seed"]), np.int32)
        )
        metadata["split_id"].append(
            np.full(n, SPLIT_IDS[summary["split"]], np.uint8)
        )
        metadata["episode_id"].append(np.full(n, episode_id, np.int64))
        metadata["trajectory_id"].append(np.full(n, episode_id, np.int64))
        scenario_hash = summary["scenario"]["manifest_sha256"]
        metadata["scenario_id"].append(
            np.full(n, scenario_to_id[scenario_hash], np.int16)
        )
        metadata["demand_realization_seed"].append(
            np.full(
                n,
                int(summary["scenario"]["selection_seed"]),
                np.int32,
            )
        )
        metadata["demand_scale"].append(
            np.full(n, float(summary["scenario"]["demand_scale"]), np.float32)
        )
        metadata["source_shard_id"].append(np.full(n, episode_id, np.int16))

    arrays_to_write = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "discounts": discounts,
        "terminals": terminals,
        "raw_observations": raw_observations,
        "raw_next_observations": raw_next_observations,
        "wait_passenger_seconds": wait_seconds,
        "in_vehicle_passenger_seconds": in_vehicle_seconds,
        "waiting_cost": waiting_cost,
        "in_vehicle_cost": in_vehicle_cost,
        "delta_t_seconds": delta_t,
        **{name: np.concatenate(parts) for name, parts in metadata.items()},
    }
    n_rows = len(rewards)
    source_shards = [
        {
            "path": str(path),
            "sha256": summary["shard_sha256"],
            "job_signature": summary.get("job_signature"),
        }
        for path, summary, _ in loaded
    ]
    provenance = {
        "transform_version": BUILD_TRANSFORM_VERSION,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "contract": {"path": str(ROOT / "bus_h2o" / "mdp_contract_v6.py"), "sha256": sha256(ROOT / "bus_h2o" / "mdp_contract_v6.py")},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__},
        "source_shard_sha256": [item["sha256"] for item in source_shards],
        "collection_matrix": matrix,
    }
    derivation_signature = canonical_sha256(provenance)
    provenance["derivation_signature"] = derivation_signature
    content_signature = dataset_content_sha256(
        arrays_to_write,
        {
            "schema_version": SCHEMA_VERSION,
            "config": DEFAULT_CONFIG_V6.to_dict(),
            "state_names": list(CANONICAL_STATE_NAMES),
            "raw_state_names": list(RAW_STATE_NAMES),
            "policy_names": policy_to_id,
            "split_names": SPLIT_IDS,
            "scenario_manifest_hashes": scenario_to_id,
            "station_max_by_line": station_max.tolist(),
        },
    )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["content_signature"] = content_signature
        handle.attrs["derivation_signature"] = derivation_signature
        handle.attrs["build_provenance"] = json.dumps(provenance, sort_keys=True)
        handle.attrs["config"] = json.dumps(DEFAULT_CONFIG_V6.to_dict(), sort_keys=True)
        handle.attrs["state_dim"] = observations.shape[1]
        handle.attrs["state_names"] = json.dumps(list(CANONICAL_STATE_NAMES))
        handle.attrs["raw_state_dim"] = RAW_STATE_DIM
        handle.attrs["raw_state_names"] = json.dumps(list(RAW_STATE_NAMES))
        handle.attrs["action_dim"] = ACTION_DIM
        handle.attrs["action_semantics"] = (
            "one AEC token action: a=-1 is exact zero hold; -1<a<=1 maps "
            "linearly to (0,60] seconds"
        )
        handle.attrs["reward_semantics"] = (
            "non-overlapping system generalized passenger time between "
            "chronologically consecutive centralized decision tokens"
        )
        handle.attrs["transition_semantics"] = (
            "fixed-width simultaneous event batch serialized as AEC tokens; "
            "same-time intermediate rows preserve prior actions"
        )
        handle.attrs["discount_semantics"] = (
            "finite-horizon undiscounted objective: nonterminal discount is 1, "
            "including at zero duration; terminal discount is 0"
        )
        handle.attrs["station_max_by_line"] = json.dumps(station_max.tolist())
        handle.attrs["policy_names"] = json.dumps(policy_to_id, sort_keys=True)
        handle.attrs["split_names"] = json.dumps(SPLIT_IDS, sort_keys=True)
        handle.attrs["scenario_manifest_hashes"] = json.dumps(scenario_to_id, sort_keys=True)
        handle.attrs["n_transitions"] = n_rows
        handle.attrs["source_shards"] = json.dumps(source_shards, sort_keys=True)
        if matrix is not None:
            handle.attrs["collection_matrix"] = json.dumps(matrix, sort_keys=True)
        for name, values in arrays_to_write.items():
            _create_dataset(handle, name, values)
    os.replace(temporary, output)

    summary = {
        "schema_version": "transit-smdp-v6-dataset-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output),
        "dataset_sha256": sha256(output),
        "content_signature": content_signature,
        "derivation_signature": derivation_signature,
        "rows": n_rows,
        "episodes": len(loaded),
        "trajectories": len(loaded),
        "state_dim": int(observations.shape[1]),
        "raw_state_dim": RAW_STATE_DIM,
        "terminal_rows": int(terminals.sum()),
        "zero_duration_rows": int(np.sum(delta_t == 0.0)),
        "exact_zero_action_rate": float(np.mean(actions[:, 0] == -1.0)),
        "reward": {"mean": float(rewards.mean()), "min": float(rewards.min()), "max": float(rewards.max()), "sum": float(rewards.sum())},
        "station_max_by_line": station_max.tolist(),
        "source_shards": len(loaded),
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
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument("--allow-unmanifested", action="store_true")
    args = parser.parse_args()
    default_manifest = args.shard_root.expanduser().resolve() / "matrix_manifest.json"
    matrix_manifest = args.matrix_manifest or default_manifest
    if not matrix_manifest.is_file():
        if not args.allow_unmanifested:
            raise FileNotFoundError("formal v6 builds require a matrix manifest")
        matrix_manifest = None
    summary = build_dataset(
        discover_shards(args.shard_root),
        args.output,
        allow_partial=args.allow_partial,
        matrix_manifest=matrix_manifest,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
