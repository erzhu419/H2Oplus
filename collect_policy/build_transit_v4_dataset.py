#!/usr/bin/env python3
"""Validate and merge SUMO v4 shards into one immutable HDF5 dataset."""

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

from bus_h2o.mdp_contract_v4 import (
    ACTION_DIM,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V4,
    HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v4,
    correct_headway_measurement_jitter_v4,
    infer_station_max_v4,
    raw_event_to_canonical_v4,
    transition_cost_components_v4,
    transition_duration_seconds_v4,
    transition_reward_v4,
)


SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
BUILD_TRANSFORM_VERSION = "transit-smdp-v4-derived-transform-v2"
REQUIRED_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "discounts",
    "terminals",
    "delta_t_seconds",
    "trajectory_local_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dataset_content_sha256(arrays: dict[str, np.ndarray], metadata: dict) -> str:
    """Hash logical dataset contents independently of HDF5 container bytes."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name])
        header = {
            "name": name,
            "dtype": values.dtype.str,
            "shape": values.shape,
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def measurement_correction_report(
    *raw_batches: tuple[str, np.ndarray],
) -> dict:
    fields = {}
    for endpoint, observations in raw_batches:
        for name in ("forward_headway_seconds", "backward_headway_seconds"):
            values = observations[:, RAW_EVENT_STATE_INDEX[name]]
            negative = values < 0.0
            fields[f"{endpoint}.{name}"] = {
                "corrected_values": int(np.sum(negative)),
                "minimum_source_seconds": float(np.min(values)),
            }
    return {
        "rule": (
            "negative headways within the SUMO one-second measurement resolution "
            "are simultaneous events and are mapped to zero; lower values fail"
        ),
        "tolerance_seconds": HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS,
        "fields": fields,
        "total_corrected_values": sum(
            field["corrected_values"] for field in fields.values()
        ),
    }


def discover_shards(root: Path) -> list[Path]:
    shards = sorted(root.expanduser().resolve().rglob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no .npz shards found under {root}")
    return shards


def validate_collection_matrix(
    manifest_path: Path, shard_paths: list[Path]
) -> dict:
    """Require an exact one-to-one match with the declared collection jobs."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"collection matrix manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-smdp-v4-collection-matrix-v1":
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
    sumo_version = manifest.get("sumo_version")
    for output, job in jobs_by_output.items():
        summary_path = output.with_suffix(output.suffix + ".json")
        if not summary_path.is_file():
            raise FileNotFoundError(f"shard summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text())
        expected_values = {
            "job_signature": job["signature"],
            "policy_name": job["policy"]["policy_name"],
            "policy_seed": int(job["policy_seed"]),
            "sumo_seed": int(job["sumo_seed"]),
        }
        mismatches = {
            name: (summary.get(name), value)
            for name, value in expected_values.items()
            if summary.get(name) != value
        }
        if summary.get("scenario", {}).get("manifest_sha256") != job["scenario"][
            "sha256"
        ]:
            mismatches["scenario_manifest_sha256"] = (
                summary.get("scenario", {}).get("manifest_sha256"),
                job["scenario"]["sha256"],
            )
        if summary.get("scenario", {}).get("runtime_input_sha256") != job[
            "scenario"
        ]["runtime_input_sha256"]:
            mismatches["scenario_runtime_input_sha256"] = (
                summary.get("scenario", {}).get("runtime_input_sha256"),
                job["scenario"]["runtime_input_sha256"],
            )
        if summary.get("source_sha256") != source_sha256:
            mismatches["source_sha256"] = (summary.get("source_sha256"), source_sha256)
        if summary.get("runtime", {}).get("sumo_version") != sumo_version:
            mismatches["sumo_version"] = (
                summary.get("runtime", {}).get("sumo_version"),
                sumo_version,
            )
        if mismatches:
            raise ValueError(f"shard disagrees with collection matrix: {output}: {mismatches}")

    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "jobs": len(jobs),
        "collector_sha256": manifest.get("collector_sha256"),
        "runner_sha256": manifest.get("runner_sha256"),
        "source_sha256": source_sha256,
        "sumo_version": sumo_version,
    }


def _load_shard(path: Path, *, allow_partial: bool) -> tuple[dict, dict[str, np.ndarray]]:
    summary_path = path.with_suffix(path.suffix + ".json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"shard summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != "transit-smdp-v4-shard-v1":
        raise ValueError(f"unsupported shard summary schema: {summary_path}")
    if summary.get("contract_schema") != SCHEMA_VERSION:
        raise ValueError(f"shard contract mismatch: {summary_path}")
    if summary.get("contract") != DEFAULT_CONFIG_V4.to_dict():
        raise ValueError(f"shard MDP configuration mismatch: {summary_path}")
    if summary.get("raw_state_names") != list(RAW_EVENT_STATE_NAMES):
        raise ValueError(f"shard raw-state semantics mismatch: {summary_path}")
    actual_hash = sha256(path)
    if summary.get("shard_sha256") != actual_hash:
        raise ValueError(f"shard hash mismatch: {path}")
    episode = summary["episode"]
    if not allow_partial:
        if int(episode.get("max_events", 0)) != 0:
            raise ValueError(f"partial event-limited shard is not formal data: {path}")
        if float(episode["observed_end_time_seconds"]) != float(
            episode["configured_seconds"]
        ):
            raise ValueError(f"shard did not reach its configured horizon: {path}")
    if summary.get("split") not in SPLIT_IDS:
        raise ValueError(f"unknown shard split in {summary_path}")

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
    if np.any(arrays["terminals"] != 0):
        raise ValueError(f"continuing same-trip shard has terminal rows: {path}")
    if not np.all(arrays["observations"][:, 0] == arrays["next_observations"][:, 0]):
        raise ValueError(f"shard contains cross-line transitions: {path}")
    if not np.all(arrays["next_observations"][:, 1] > arrays["observations"][:, 1]):
        raise ValueError(f"shard contains non-forward station transitions: {path}")

    duration = transition_duration_seconds_v4(
        arrays["observations"], arrays["next_observations"]
    )
    reward = transition_reward_v4(
        arrays["observations"],
        arrays["actions"],
        arrays["next_observations"],
        correct_measurement_jitter=False,
    )
    discount = continuous_time_discount_v4(duration)
    for name, recomputed, stored in (
        ("delta_t_seconds", duration, arrays["delta_t_seconds"]),
        ("rewards", reward, arrays["rewards"]),
        ("discounts", discount, arrays["discounts"]),
    ):
        if not np.array_equal(np.asarray(recomputed, dtype=stored.dtype), stored):
            maximum = float(np.max(np.abs(np.asarray(recomputed) - stored)))
            raise ValueError(f"{path}:{name} recomputation differs by {maximum}")
    return summary, arrays


def _create_dataset(handle: h5py.File, name: str, values: np.ndarray) -> None:
    rows = max(int(values.shape[0]), 1)
    chunk_rows = min(rows, 8192)
    chunks = (chunk_rows,) + values.shape[1:]
    handle.create_dataset(
        name,
        data=values,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=chunks,
    )


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
    raw_observations = np.asarray(
        correct_headway_measurement_jitter_v4(source_raw_observations),
        dtype=np.float32,
    )
    raw_next_observations = np.asarray(
        correct_headway_measurement_jitter_v4(source_raw_next_observations),
        dtype=np.float32,
    )
    station_max = infer_station_max_v4(
        np.concatenate((raw_observations, raw_next_observations), axis=0)
    )
    observations = raw_event_to_canonical_v4(raw_observations, station_max)
    next_observations = raw_event_to_canonical_v4(
        raw_next_observations, station_max
    )
    actions = np.concatenate([arrays["actions"] for _, _, arrays in loaded]).astype(
        np.float32, copy=False
    )
    components = transition_cost_components_v4(
        raw_observations, actions, raw_next_observations
    )
    rewards = -np.asarray(components["total_cost"], dtype=np.float32)
    delta_t = transition_duration_seconds_v4(
        raw_observations, raw_next_observations
    ).astype(np.float32, copy=False)
    discounts = continuous_time_discount_v4(delta_t).astype(np.float32, copy=False)
    terminals = np.concatenate(
        [arrays["terminals"] for _, _, arrays in loaded]
    ).astype(np.uint8, copy=False)

    policy_names = sorted({summary["policy_name"] for _, summary, _ in loaded})
    policy_name_to_id = {name: index for index, name in enumerate(policy_names)}
    scenario_hashes = sorted(
        {summary["scenario"]["manifest_sha256"] for _, summary, _ in loaded}
    )
    scenario_to_id = {value: index for index, value in enumerate(scenario_hashes)}
    policy_ids = []
    policy_seeds = []
    sumo_seeds = []
    split_ids = []
    episode_ids = []
    trajectory_ids = []
    scenario_ids = []
    demand_scales = []
    source_shard_ids = []
    trajectory_base = 0
    for episode_id, (_, summary, arrays) in enumerate(loaded):
        n = len(arrays["rewards"])
        local_trajectory = arrays["trajectory_local_id"].astype(np.int64)
        policy_ids.append(np.full(n, policy_name_to_id[summary["policy_name"]], np.int8))
        policy_seeds.append(np.full(n, int(summary["policy_seed"]), np.int32))
        sumo_seeds.append(np.full(n, int(summary["sumo_seed"]), np.int32))
        split_ids.append(np.full(n, SPLIT_IDS[summary["split"]], np.uint8))
        episode_ids.append(np.full(n, episode_id, np.int64))
        trajectory_ids.append(local_trajectory + trajectory_base)
        if n:
            trajectory_base += int(local_trajectory.max()) + 1
        scenario_hash = summary["scenario"]["manifest_sha256"]
        scenario_ids.append(np.full(n, scenario_to_id[scenario_hash], np.int16))
        demand_scales.append(
            np.full(n, float(summary["scenario"]["demand_scale"]), np.float32)
        )
        source_shard_ids.append(np.full(n, episode_id, np.int16))

    arrays_to_write = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "discounts": discounts,
        "terminals": terminals,
        "raw_observations": raw_observations,
        "raw_next_observations": raw_next_observations,
        "waiting_cost": np.asarray(components["waiting_cost"], dtype=np.float32),
        "holding_cost": np.asarray(components["holding_cost"], dtype=np.float32),
        "delta_t_seconds": delta_t,
        "policy_id": np.concatenate(policy_ids),
        "policy_seed": np.concatenate(policy_seeds),
        "sumo_seed": np.concatenate(sumo_seeds),
        "split_id": np.concatenate(split_ids),
        "episode_id": np.concatenate(episode_ids),
        "trajectory_id": np.concatenate(trajectory_ids),
        "scenario_id": np.concatenate(scenario_ids),
        "demand_scale": np.concatenate(demand_scales),
        "source_shard_id": np.concatenate(source_shard_ids),
    }
    n_rows = len(rewards)
    if n_rows == 0:
        raise ValueError("merged dataset would be empty")
    if not all(values.shape[0] == n_rows for values in arrays_to_write.values()):
        raise RuntimeError("merged arrays have inconsistent row counts")

    build_provenance = {
        "transform_version": BUILD_TRANSFORM_VERSION,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "mdp_contract": {
            "path": str(ROOT / "bus_h2o" / "mdp_contract_v4.py"),
            "sha256": sha256(ROOT / "bus_h2o" / "mdp_contract_v4.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "source_shard_sha256": [summary["shard_sha256"] for _, summary, _ in loaded],
        "measurement_resolution_correction": correction_report,
    }
    derivation_signature = canonical_sha256(
        {
            "transform_version": build_provenance["transform_version"],
            "builder_sha256": build_provenance["builder"]["sha256"],
            "mdp_contract_sha256": build_provenance["mdp_contract"]["sha256"],
            "runtime": build_provenance["runtime"],
            "source_shard_sha256": build_provenance["source_shard_sha256"],
            "measurement_resolution_correction": correction_report,
            "collection": (
                {
                    "jobs": matrix_provenance["jobs"],
                    "collector_sha256": matrix_provenance["collector_sha256"],
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
            "config": DEFAULT_CONFIG_V4.to_dict(),
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
        handle.attrs["build_provenance"] = json.dumps(build_provenance, sort_keys=True)
        handle.attrs["config"] = json.dumps(DEFAULT_CONFIG_V4.to_dict(), sort_keys=True)
        handle.attrs["state_dim"] = observations.shape[1]
        handle.attrs["state_names"] = json.dumps(list(CANONICAL_STATE_NAMES))
        handle.attrs["raw_state_dim"] = RAW_EVENT_STATE_DIM
        handle.attrs["raw_state_names"] = json.dumps(list(RAW_EVENT_STATE_NAMES))
        handle.attrs["action_dim"] = ACTION_DIM
        handle.attrs["action_semantics"] = (
            "mixed measure: a=-1 is an exact zero-hold atom; -1<a<=1 maps to "
            "hold_seconds=30*(a+1); speed_ratio is fixed at 1"
        )
        handle.attrs["reward_semantics"] = (
            "negative generalized passenger-minutes; waiting and holding terms "
            "are stored separately, preserve physical cross-line welfare weights, "
            "and are computed exactly once from raw endpoints"
        )
        handle.attrs["transition_semantics"] = (
            "same SUMO trip at its next strictly downstream station decision event"
        )
        handle.attrs["discount_semantics"] = "0.99 ** (delta_t_seconds / 60)"
        handle.attrs["station_max_by_line"] = json.dumps(station_max.tolist())
        handle.attrs["policy_names"] = json.dumps(policy_name_to_id, sort_keys=True)
        handle.attrs["split_names"] = json.dumps(SPLIT_IDS, sort_keys=True)
        handle.attrs["scenario_manifest_hashes"] = json.dumps(scenario_to_id, sort_keys=True)
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
    output_hash = sha256(output)
    summary = {
        "schema_version": "transit-smdp-v4-dataset-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output),
        "dataset_sha256": output_hash,
        "content_signature": content_signature,
        "derivation_signature": derivation_signature,
        "rows": n_rows,
        "episodes": len(loaded),
        "trajectories": trajectory_base,
        "state_dim": int(observations.shape[1]),
        "raw_state_dim": RAW_EVENT_STATE_DIM,
        "policy_counts": {
            name: int(np.sum(arrays_to_write["policy_id"] == index))
            for name, index in policy_name_to_id.items()
        },
        "demand_scale_counts": {
            str(float(scale)): int(np.sum(arrays_to_write["demand_scale"] == scale))
            for scale in np.unique(arrays_to_write["demand_scale"])
        },
        "exact_zero_action_rate": float(np.mean(actions[:, 0] == -1.0)),
        "reward": {
            "mean": float(rewards.mean()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
        },
        "waiting_cost_mean": float(arrays_to_write["waiting_cost"].mean()),
        "holding_cost_mean": float(arrays_to_write["holding_cost"].mean()),
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
    parser.add_argument("--allow-unmanifested", action="store_true")
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
