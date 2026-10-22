#!/usr/bin/env python3
"""Validate and merge true-welfare SUMO v5 shards into immutable HDF5."""

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
    HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS,
    RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4,
    correct_headway_measurement_jitter_v4,
    infer_station_max_v4,
)
from bus_h2o.mdp_contract_v5 import (
    ACTION_DIM,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V5,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v5,
    raw_event_to_canonical_v5,
    transition_duration_seconds_v5,
)
from collect_policy.build_transit_v4_dataset import (
    SPLIT_IDS,
    _create_dataset,
    canonical_sha256,
    dataset_content_sha256,
    discover_shards,
    measurement_correction_report,
    sha256,
)


BUILD_TRANSFORM_VERSION = "transit-smdp-v5-derived-transform-v1"
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


def validate_collection_matrix(
    manifest_path: Path, shard_paths: list[Path]
) -> dict:
    """Require exact agreement between the declared v5 jobs and their shards."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"collection matrix manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-smdp-v5-collection-matrix-v1":
        raise ValueError(f"unsupported collection matrix schema: {manifest_path}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("collection matrix contains no jobs")

    jobs_by_output = {}
    signatures = set()
    logical_jobs = set()
    for job in jobs:
        output = Path(job["output"]).expanduser().resolve()
        job_signature = job.get("signature")
        if not isinstance(job_signature, str) or not job_signature:
            raise ValueError(f"collection job lacks a signature: {output}")
        if output in jobs_by_output:
            raise ValueError(f"collection matrix repeats output path: {output}")
        if job_signature in signatures:
            raise ValueError(f"collection matrix repeats job signature: {job_signature}")
        logical_key = (
            job["policy"]["policy_name"],
            job["scenario"]["sha256"],
            int(job["sumo_seed"]),
        )
        if logical_key in logical_jobs:
            raise ValueError(f"collection matrix repeats logical job: {logical_key}")
        jobs_by_output[output] = job
        signatures.add(job_signature)
        logical_jobs.add(logical_key)

    actual = {path.expanduser().resolve() for path in shard_paths}
    expected = set(jobs_by_output)
    if actual != expected:
        missing = sorted(str(path) for path in expected.difference(actual))
        extra = sorted(str(path) for path in actual.difference(expected))
        raise ValueError(
            "shards do not exactly match the collection matrix: "
            f"missing={missing}, extra={extra}"
        )

    source_sha256 = manifest.get("source_sha256")
    sumo_version_value = manifest.get("sumo_version")
    block_ids = set()
    block_environments = {}
    for output, job in jobs_by_output.items():
        summary_path = output.with_suffix(output.suffix + ".json")
        if not summary_path.is_file():
            raise FileNotFoundError(f"shard summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text())
        expected_values = {
            "schema_version": "transit-smdp-v5-shard-v2",
            "job_signature": job["signature"],
            "policy_name": job["policy"]["policy_name"],
            "policy_seed": int(job["policy_seed"]),
            "sumo_seed": int(job["sumo_seed"]),
            "split": job["split"],
            "diagnostic": bool(job["diagnostic"]),
            "training_eligible": not bool(job["diagnostic"]),
        }
        mismatches = {
            name: (summary.get(name), value)
            for name, value in expected_values.items()
            if summary.get(name) != value
        }
        scenario_summary = summary.get("scenario", {})
        scenario_job = job["scenario"]
        for summary_name, job_name in (
            ("manifest_sha256", "sha256"),
            ("runtime_input_sha256", "runtime_input_sha256"),
        ):
            if scenario_summary.get(summary_name) != scenario_job[job_name]:
                mismatches[f"scenario_{summary_name}"] = (
                    scenario_summary.get(summary_name),
                    scenario_job[job_name],
                )
        if summary.get("source_sha256") != source_sha256:
            mismatches["source_sha256"] = (
                summary.get("source_sha256"),
                source_sha256,
            )
        if summary.get("runtime", {}).get("sumo_version") != sumo_version_value:
            mismatches["sumo_version"] = (
                summary.get("runtime", {}).get("sumo_version"),
                sumo_version_value,
            )
        if mismatches:
            raise ValueError(
                f"shard disagrees with collection matrix: {output}: {mismatches}"
            )

        block_id = int(job["block_id"])
        environment = (scenario_job["sha256"], int(job["sumo_seed"]))
        previous_environment = block_environments.setdefault(block_id, environment)
        if previous_environment != environment:
            raise ValueError(
                f"block {block_id} maps to multiple environments: "
                f"{previous_environment} and {environment}"
            )
        block_ids.add(block_id)

    declared_blocks = int(manifest.get("independent_environment_blocks", -1))
    if declared_blocks != len(block_ids):
        raise ValueError(
            "independent environment block count disagrees with matrix jobs: "
            f"declared={declared_blocks}, observed={len(block_ids)}"
        )
    policy_names = {job["policy"]["policy_name"] for job in jobs}
    expected_pairs = {
        (block_id, policy_name)
        for block_id in block_ids
        for policy_name in policy_names
    }
    observed_pairs = {
        (int(job["block_id"]), job["policy"]["policy_name"]) for job in jobs
    }
    if observed_pairs != expected_pairs:
        raise ValueError("not every environment block contains every declared policy")

    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "jobs": len(jobs),
        "study_stage": manifest.get("study_stage"),
        "environment_design": manifest.get("environment_design"),
        "independent_environment_blocks": declared_blocks,
        "policies": sorted(policy_names),
        "collector_sha256": manifest.get("collector_sha256"),
        "runner_sha256": manifest.get("runner_sha256"),
        "source_sha256": source_sha256,
        "sumo_version": sumo_version_value,
    }


def _correct_measurement_jitter(raw_observations: np.ndarray) -> np.ndarray:
    corrected_local = np.asarray(
        correct_headway_measurement_jitter_v4(
            raw_observations[:, :RAW_EVENT_STATE_DIM_V4]
        ),
        dtype=np.float32,
    )
    corrected = np.asarray(raw_observations, dtype=np.float32).copy()
    corrected[:, :RAW_EVENT_STATE_DIM_V4] = corrected_local
    return corrected


def _load_shard(
    path: Path, *, allow_partial: bool
) -> tuple[dict, dict[str, np.ndarray]]:
    summary_path = path.with_suffix(path.suffix + ".json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"shard summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != "transit-smdp-v5-shard-v2":
        raise ValueError(f"unsupported shard summary schema: {summary_path}")
    if summary.get("contract_schema") != SCHEMA_VERSION:
        raise ValueError(f"shard contract mismatch: {summary_path}")
    if summary.get("contract") != DEFAULT_CONFIG_V5.to_dict():
        raise ValueError(f"shard MDP configuration mismatch: {summary_path}")
    if summary.get("raw_state_names") != list(RAW_EVENT_STATE_NAMES):
        raise ValueError(f"shard raw-state semantics mismatch: {summary_path}")
    if summary.get("split") not in SPLIT_IDS:
        raise ValueError(f"unknown shard split in {summary_path}")
    if summary.get("shard_sha256") != sha256(path):
        raise ValueError(f"shard hash mismatch: {path}")
    episode = summary["episode"]
    count_audit = summary.get("ledger_audit", {}).get("count_source_audit", {})
    if summary.get("ledger_audit", {}).get(
        "legacy_sampled_trip_histories_used_as_ground_truth"
    ) is not False:
        raise ValueError(f"shard welfare source is not exact SUMO counts: {path}")
    if count_audit.get("max_waiting_count_error") != 0 or count_audit.get(
        "max_in_vehicle_count_error"
    ) != 0:
        raise ValueError(f"shard passenger-count audit failed: {path}")
    if not allow_partial:
        if summary.get("diagnostic") or not summary.get("training_eligible"):
            raise ValueError(f"diagnostic shard is not training eligible: {path}")
        if float(episode["configured_seconds"]) != float(
            DEFAULT_CONFIG_V5.episode_seconds
        ):
            raise ValueError(f"shard uses a non-contract horizon: {path}")
        if float(episode["observed_end_time_seconds"]) != float(
            episode["configured_seconds"]
        ):
            raise ValueError(f"shard did not reach its configured horizon: {path}")
        service = summary.get("service", {})
        if service.get("completed_trips") != service.get("departed_trips"):
            raise ValueError(f"not every departed SUMO trip completed: {path}")
        if episode.get("completed_trip_contexts") != service.get(
            "completed_trips"
        ):
            raise ValueError(f"trip-completion contexts are incomplete: {path}")
        if episode.get("horizon_closures") != 0:
            raise ValueError(f"completed trips were closed at the horizon: {path}")

    with np.load(path, allow_pickle=False) as handle:
        missing = sorted(set(REQUIRED_ARRAYS).difference(handle.files))
        if missing:
            raise KeyError(f"{path} is missing arrays {missing}")
        arrays = {name: np.asarray(handle[name]) for name in REQUIRED_ARRAYS}
    n = arrays["rewards"].shape[0]
    expected_shapes = {
        "observations": (n, RAW_EVENT_STATE_DIM),
        "actions": (n, ACTION_DIM),
        "rewards": (n,),
        "next_observations": (n, RAW_EVENT_STATE_DIM),
        "discounts": (n,),
        "terminals": (n,),
        "delta_t_seconds": (n,),
        "trajectory_local_id": (n,),
        "wait_passenger_seconds": (n,),
        "in_vehicle_passenger_seconds": (n,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"{path}:{name} has shape {arrays[name].shape}, expected {expected}"
            )
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}:{name} contains non-finite values")
    if n != int(episode["transitions"]):
        raise ValueError(f"shard row count disagrees with summary: {path}")
    if np.any(np.abs(arrays["actions"]) > 1.00001):
        raise ValueError(f"shard actions lie outside [-1,1]: {path}")
    if np.any((arrays["terminals"] != 0) & (arrays["terminals"] != 1)):
        raise ValueError(f"shard terminals are not binary: {path}")
    if np.any(arrays["wait_passenger_seconds"] < 0.0) or np.any(
        arrays["in_vehicle_passenger_seconds"] < 0.0
    ):
        raise ValueError(f"shard passenger-time components are negative: {path}")
    if not np.all(
        arrays["observations"][:, 0] == arrays["next_observations"][:, 0]
    ):
        raise ValueError(f"shard contains cross-line transitions: {path}")
    if np.any(arrays["observations"][:, -2:] < 0.0) or np.any(
        arrays["next_observations"][:, -2:] < 0.0
    ):
        raise ValueError(f"shard contains negative system passenger counts: {path}")

    station_change = (
        arrays["next_observations"][:, RAW_EVENT_STATE_INDEX["station_index"]]
        - arrays["observations"][:, RAW_EVENT_STATE_INDEX["station_index"]]
    )
    terminals = arrays["terminals"].astype(bool)
    if np.any(station_change[~terminals] <= 0.0):
        raise ValueError(f"non-terminal shard row is not strictly downstream: {path}")
    if np.any(station_change[terminals] < 0.0):
        raise ValueError(f"terminal shard row moves backwards: {path}")

    duration = np.asarray(
        transition_duration_seconds_v5(
            arrays["observations"], arrays["next_observations"]
        ),
        dtype=np.float32,
    )
    reward = -(
        DEFAULT_CONFIG_V5.waiting_time_value
        * arrays["wait_passenger_seconds"]
        + DEFAULT_CONFIG_V5.in_vehicle_time_value
        * arrays["in_vehicle_passenger_seconds"]
    ) / DEFAULT_CONFIG_V5.reward_unit_passenger_seconds
    discount = np.asarray(
        continuous_time_discount_v5(duration, DEFAULT_CONFIG_V5),
        dtype=np.float32,
    )
    discount[terminals] = 0.0
    recomputed = {
        "delta_t_seconds": duration,
        "rewards": reward,
        "discounts": discount,
    }
    for name, values in recomputed.items():
        stored = arrays[name]
        maximum = float(
            np.max(np.abs(np.asarray(values, dtype=np.float64) - stored))
        )
        if maximum > 1e-6:
            raise ValueError(f"{path}:{name} recomputation differs by {maximum}")
    if np.any(arrays["discounts"][terminals] != 0.0):
        raise ValueError(f"terminal shard rows have nonzero discounts: {path}")
    return summary, arrays


def build_dataset(
    shard_paths: list[Path],
    output: Path,
    *,
    allow_partial: bool = False,
    matrix_manifest: Path | None = None,
) -> dict:
    if not shard_paths:
        raise ValueError("at least one shard is required")
    matrix_provenance = (
        validate_collection_matrix(matrix_manifest, shard_paths)
        if matrix_manifest is not None
        else None
    )
    loaded = [
        (path.resolve(), *_load_shard(path.resolve(), allow_partial=allow_partial))
        for path in sorted(shard_paths)
    ]
    source_raw_observations = np.concatenate(
        [arrays["observations"] for _, _, arrays in loaded]
    ).astype(np.float32, copy=False)
    source_raw_next_observations = np.concatenate(
        [arrays["next_observations"] for _, _, arrays in loaded]
    ).astype(np.float32, copy=False)
    correction_report = measurement_correction_report(
        ("observations", source_raw_observations),
        ("next_observations", source_raw_next_observations),
    )
    raw_observations = _correct_measurement_jitter(source_raw_observations)
    raw_next_observations = _correct_measurement_jitter(
        source_raw_next_observations
    )
    station_max = infer_station_max_v4(
        np.concatenate(
            (
                raw_observations[:, :RAW_EVENT_STATE_DIM_V4],
                raw_next_observations[:, :RAW_EVENT_STATE_DIM_V4],
            ),
            axis=0,
        )
    )
    observations = raw_event_to_canonical_v5(
        raw_observations, station_max, DEFAULT_CONFIG_V5
    )
    next_observations = raw_event_to_canonical_v5(
        raw_next_observations, station_max, DEFAULT_CONFIG_V5
    )
    actions = np.concatenate(
        [arrays["actions"] for _, _, arrays in loaded]
    ).astype(np.float32, copy=False)
    wait_seconds = np.concatenate(
        [arrays["wait_passenger_seconds"] for _, _, arrays in loaded]
    ).astype(np.float64, copy=False)
    in_vehicle_seconds = np.concatenate(
        [arrays["in_vehicle_passenger_seconds"] for _, _, arrays in loaded]
    ).astype(np.float64, copy=False)
    waiting_cost = (
        DEFAULT_CONFIG_V5.waiting_time_value
        * wait_seconds
        / DEFAULT_CONFIG_V5.reward_unit_passenger_seconds
    ).astype(np.float64, copy=False)
    in_vehicle_cost = (
        DEFAULT_CONFIG_V5.in_vehicle_time_value
        * in_vehicle_seconds
        / DEFAULT_CONFIG_V5.reward_unit_passenger_seconds
    ).astype(np.float64, copy=False)
    rewards = -(waiting_cost + in_vehicle_cost)
    delta_t = np.asarray(
        transition_duration_seconds_v5(raw_observations, raw_next_observations),
        dtype=np.float32,
    )
    terminals = np.concatenate(
        [arrays["terminals"] for _, _, arrays in loaded]
    ).astype(np.uint8, copy=False)
    discounts = np.asarray(
        continuous_time_discount_v5(delta_t, DEFAULT_CONFIG_V5),
        dtype=np.float32,
    )
    discounts[terminals.astype(bool)] = 0.0

    policy_names = sorted({summary["policy_name"] for _, summary, _ in loaded})
    policy_name_to_id = {name: index for index, name in enumerate(policy_names)}
    scenario_hashes = sorted(
        {summary["scenario"]["manifest_sha256"] for _, summary, _ in loaded}
    )
    scenario_to_id = {value: index for index, value in enumerate(scenario_hashes)}
    metadata_arrays = {
        name: []
        for name in (
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
    }
    trajectory_base = 0
    for episode_id, (_, summary, shard_arrays) in enumerate(loaded):
        n = len(shard_arrays["rewards"])
        local_trajectory = shard_arrays["trajectory_local_id"].astype(np.int64)
        metadata_arrays["policy_id"].append(
            np.full(n, policy_name_to_id[summary["policy_name"]], np.int8)
        )
        metadata_arrays["policy_seed"].append(
            np.full(n, int(summary["policy_seed"]), np.int32)
        )
        metadata_arrays["sumo_seed"].append(
            np.full(n, int(summary["sumo_seed"]), np.int32)
        )
        metadata_arrays["split_id"].append(
            np.full(n, SPLIT_IDS[summary["split"]], np.uint8)
        )
        metadata_arrays["episode_id"].append(
            np.full(n, episode_id, np.int64)
        )
        metadata_arrays["trajectory_id"].append(
            local_trajectory + trajectory_base
        )
        if n:
            trajectory_base += int(local_trajectory.max()) + 1
        scenario_hash = summary["scenario"]["manifest_sha256"]
        metadata_arrays["scenario_id"].append(
            np.full(n, scenario_to_id[scenario_hash], np.int16)
        )
        metadata_arrays["demand_scale"].append(
            np.full(n, float(summary["scenario"]["demand_scale"]), np.float32)
        )
        metadata_arrays["source_shard_id"].append(
            np.full(n, episode_id, np.int16)
        )

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
        **{
            name: np.concatenate(parts)
            for name, parts in metadata_arrays.items()
        },
    }
    n_rows = len(rewards)
    if n_rows == 0:
        raise ValueError("merged dataset would be empty")
    if not all(values.shape[0] == n_rows for values in arrays_to_write.values()):
        raise RuntimeError("merged arrays have inconsistent row counts")

    source_hashes = [summary["shard_sha256"] for _, summary, _ in loaded]
    build_provenance = {
        "transform_version": BUILD_TRANSFORM_VERSION,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "mdp_contract": {
            "path": str(ROOT / "bus_h2o" / "mdp_contract_v5.py"),
            "sha256": sha256(ROOT / "bus_h2o" / "mdp_contract_v5.py"),
        },
        "passenger_time": {
            "path": str(ROOT / "bus_h2o" / "passenger_time.py"),
            "sha256": sha256(ROOT / "bus_h2o" / "passenger_time.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "source_shard_sha256": source_hashes,
        "measurement_resolution_correction": correction_report,
    }
    derivation_signature = canonical_sha256(
        {
            **build_provenance,
            "collection": (
                {
                    "jobs": matrix_provenance["jobs"],
                    "study_stage": matrix_provenance["study_stage"],
                    "environment_design": matrix_provenance[
                        "environment_design"
                    ],
                    "independent_environment_blocks": matrix_provenance[
                        "independent_environment_blocks"
                    ],
                    "collector_sha256": matrix_provenance[
                        "collector_sha256"
                    ],
                    "runner_sha256": matrix_provenance["runner_sha256"],
                    "source_file_hashes": sorted(
                        matrix_provenance["source_sha256"].values()
                    ),
                    "sumo_version": matrix_provenance["sumo_version"],
                }
                if matrix_provenance is not None
                else None
            ),
        }
    )
    build_provenance["derivation_signature"] = derivation_signature
    content_signature = dataset_content_sha256(
        arrays_to_write,
        {
            "schema_version": SCHEMA_VERSION,
            "config": DEFAULT_CONFIG_V5.to_dict(),
            "state_names": list(CANONICAL_STATE_NAMES),
            "raw_state_names": list(RAW_EVENT_STATE_NAMES),
            "policy_names": policy_name_to_id,
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
        handle.attrs["build_provenance"] = json.dumps(
            build_provenance, sort_keys=True
        )
        handle.attrs["config"] = json.dumps(
            DEFAULT_CONFIG_V5.to_dict(), sort_keys=True
        )
        handle.attrs["state_dim"] = observations.shape[1]
        handle.attrs["state_names"] = json.dumps(list(CANONICAL_STATE_NAMES))
        handle.attrs["raw_state_dim"] = RAW_EVENT_STATE_DIM
        handle.attrs["raw_state_names"] = json.dumps(list(RAW_EVENT_STATE_NAMES))
        handle.attrs["action_dim"] = ACTION_DIM
        handle.attrs["action_semantics"] = (
            "mixed measure: a=-1 is an exact zero-hold atom; -1<a<=1 maps "
            "linearly to (0,60] holding seconds; speed ratio is fixed at 1"
        )
        handle.attrs["reward_semantics"] = (
            "negative system-wide generalized passenger time accrued between "
            "successive decision epochs of the same bus, divided by 100000 "
            "passenger-seconds"
        )
        handle.attrs["reward_aggregation_warning"] = (
            "agent-centric time intervals overlap; row costs must not be summed "
            "to estimate episode passenger welfare"
        )
        handle.attrs["transition_semantics"] = (
            "same bus at its next downstream decision; terminal rows end at the "
            "route terminal or fixed evaluation horizon"
        )
        handle.attrs["discount_semantics"] = (
            "nonterminal: 0.99 ** (delta_t_seconds / 60); terminal: 0"
        )
        handle.attrs["station_max_by_line"] = json.dumps(station_max.tolist())
        handle.attrs["policy_names"] = json.dumps(policy_name_to_id, sort_keys=True)
        handle.attrs["split_names"] = json.dumps(SPLIT_IDS, sort_keys=True)
        handle.attrs["scenario_manifest_hashes"] = json.dumps(
            scenario_to_id, sort_keys=True
        )
        handle.attrs["n_transitions"] = n_rows
        handle.attrs["source_shards"] = json.dumps(
            [
                {
                    "path": str(path),
                    "sha256": summary["shard_sha256"],
                    "job_signature": summary.get("job_signature"),
                }
                for path, summary, _ in loaded
            ],
            sort_keys=True,
        )
        if matrix_provenance is not None:
            handle.attrs["collection_matrix"] = json.dumps(
                matrix_provenance, sort_keys=True
            )
        for name, values in arrays_to_write.items():
            _create_dataset(handle, name, values)
    os.replace(temporary, output)

    summary = {
        "schema_version": "transit-smdp-v5-dataset-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output),
        "dataset_sha256": sha256(output),
        "content_signature": content_signature,
        "derivation_signature": derivation_signature,
        "rows": n_rows,
        "episodes": len(loaded),
        "trajectories": trajectory_base,
        "state_dim": int(observations.shape[1]),
        "raw_state_dim": RAW_EVENT_STATE_DIM,
        "terminal_rows": int(terminals.sum()),
        "policy_counts": {
            name: int(np.sum(arrays_to_write["policy_id"] == index))
            for name, index in policy_name_to_id.items()
        },
        "demand_scale_counts": {
            str(float(scale)): int(
                np.sum(arrays_to_write["demand_scale"] == scale)
            )
            for scale in np.unique(arrays_to_write["demand_scale"])
        },
        "exact_zero_action_rate": float(np.mean(actions[:, 0] == -1.0)),
        "reward": {
            "mean": float(rewards.mean()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
        },
        "wait_passenger_seconds_mean": float(wait_seconds.mean()),
        "in_vehicle_passenger_seconds_mean": float(in_vehicle_seconds.mean()),
        "station_max_by_line": station_max.tolist(),
        "source_shards": len(loaded),
        "formal_full_horizon_only": not allow_partial,
        "collection_matrix": matrix_provenance,
        "build_provenance": build_provenance,
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
    parser.add_argument(
        "--allow-unmanifested",
        action="store_true",
        help="permit a diagnostic build without a collection matrix",
    )
    args = parser.parse_args()
    default_manifest = args.shard_root.expanduser().resolve() / "matrix_manifest.json"
    matrix_manifest = args.matrix_manifest or default_manifest
    if not matrix_manifest.is_file():
        if not args.allow_unmanifested:
            raise FileNotFoundError(
                "formal builds require --matrix-manifest (or matrix_manifest.json "
                "under --shard-root); use --allow-unmanifested only for diagnostics"
            )
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
