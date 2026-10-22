#!/usr/bin/env python3
"""Convert salvageable v2 SUMO transitions into the canonical v3 SMDP dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract import (
    ACTION_DIM,
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_NAMES,
    RAW_V2_OBS_DIM,
    SCHEMA_VERSION,
    TransitMDPConfig,
    continuous_time_discount,
    raw_v2_to_canonical,
    transition_duration_seconds,
    transition_reward_from_v2,
    validate_v2_transition_batch,
)


POLICY_FALLBACK = {
    "heuristic_best": 0,
    "heuristic_weak": 1,
    "random": 2,
    "sac": 3,
    "zero": 4,
}
SPLIT_NAMES = {"train": 0}
DETERMINISTIC_POLICY_NAMES = ("heuristic_best", "sac", "zero")
CANONICAL_DETERMINISTIC_POLICY_SEED = 42
EPISODE_RESET_FRACTION = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_station_max_stream(source: h5py.File, n_rows: int, chunk_rows: int) -> np.ndarray:
    maxima = np.ones(12, dtype=np.float32)
    observations = source["observations"]
    for start in range(0, n_rows, chunk_rows):
        batch = observations[start : min(start + chunk_rows, n_rows)]
        line_ids = np.rint(batch[:, 0]).astype(np.int64)
        for line_id in range(len(maxima)):
            stations = batch[line_ids == line_id, 2]
            if stations.size:
                maxima[line_id] = max(maxima[line_id], float(stations.max()))
    return maxima


def retained_legacy_rows(
    policies: np.ndarray,
    policy_seeds: np.ndarray,
    deterministic_policy_ids: set[int],
) -> np.ndarray:
    """Drop exact cross-file copies caused by the unpropagated SUMO seed."""

    deterministic = np.isin(policies, tuple(sorted(deterministic_policy_ids)))
    return (~deterministic) | (
        policy_seeds == CANONICAL_DETERMINISTIC_POLICY_SEED
    )


def infer_source_groups(
    source: h5py.File,
    n_rows: int,
    config: TransitMDPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover source-file, episode and same-bus trajectory provenance.

    Legacy collection kept rows in source-file order but omitted episode IDs.
    A reset is unambiguous because event time falls from roughly the end of a
    service day to its first event. Small decreases are normal when several
    buses produce transitions during one decision batch and are not resets.
    """

    policies = source["policy_id"][:n_rows].astype(np.int64, copy=False)
    seeds = source["seed"][:n_rows].astype(np.int64, copy=False)
    if "snap_file_id" in source:
        source_files = source["snap_file_id"][:n_rows].astype(np.int32, copy=False)
    else:
        source_files = np.zeros(n_rows, dtype=np.int32)
        if n_rows > 1:
            changed = (policies[1:] != policies[:-1]) | (seeds[1:] != seeds[:-1])
            source_files[1:] = np.cumsum(changed, dtype=np.int32)

    times = source["observations"][:n_rows, 10].astype(np.float64, copy=False)
    if not np.isfinite(times).all():
        raise ValueError("legacy observations contain non-finite event times")
    new_file = np.zeros(n_rows, dtype=bool)
    if n_rows:
        new_file[0] = True
    if n_rows > 1:
        new_file[1:] = source_files[1:] != source_files[:-1]
    reset = np.zeros(n_rows, dtype=bool)
    if n_rows > 1:
        reset[1:] = (
            (times[1:] < times[:-1] - config.episode_seconds * EPISODE_RESET_FRACTION)
            & ~new_file[1:]
        )
    new_episode = new_file | reset
    episode_ids = np.cumsum(new_episode, dtype=np.int64) - 1

    raw_line_ids = np.rint(source["observations"][:n_rows, 0]).astype(np.int64)
    raw_bus_ids = np.rint(source["observations"][:n_rows, 1]).astype(np.int64)
    if np.any(raw_line_ids < 0) or np.any(raw_line_ids >= 100):
        raise ValueError("legacy line IDs cannot be encoded into trajectory provenance")
    if np.any(raw_bus_ids < 0) or np.any(raw_bus_ids >= 10_000):
        raise ValueError("legacy bus IDs cannot be encoded into trajectory provenance")
    trajectory_ids = episode_ids * 1_000_000 + raw_line_ids * 10_000 + raw_bus_ids
    return source_files, episode_ids, trajectory_ids


def _dataset(
    target: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    chunk_rows: int,
) -> h5py.Dataset:
    chunks = (min(chunk_rows, shape[0]),) + shape[1:]
    return target.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def convert(
    input_path: Path,
    output_path: Path,
    config: TransitMDPConfig,
    chunk_rows: int = 100_000,
    limit: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    diagnostics: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    source_hash = sha256(input_path)

    try:
        with h5py.File(input_path, "r") as source:
            required = {"observations", "actions", "next_observations", "seed", "policy_id"}
            missing = sorted(required.difference(source.keys()))
            if missing:
                raise KeyError(f"v2 input is missing required datasets: {missing}")
            if source["observations"].shape[1] != RAW_V2_OBS_DIM:
                raise ValueError("v2 observation width does not match the executable contract")

            available = int(source["observations"].shape[0])
            n_rows = min(available, limit) if limit > 0 else available
            policy_names = json.loads(
                source.attrs.get("policy_names", json.dumps(POLICY_FALLBACK))
            )
            deterministic_policy_ids = {
                int(policy_names[name])
                for name in DETERMINISTIC_POLICY_NAMES
                if name in policy_names
            }
            output_rows = 0
            for start in range(0, n_rows, chunk_rows):
                stop = min(start + chunk_rows, n_rows)
                output_rows += int(
                    retained_legacy_rows(
                        source["policy_id"][start:stop],
                        source["seed"][start:stop],
                        deterministic_policy_ids,
                    ).sum()
                )
            station_max = infer_station_max_stream(source, n_rows, chunk_rows)
            source_files, episode_ids, trajectory_ids = infer_source_groups(
                source, n_rows, config
            )
            retained = retained_legacy_rows(
                source["policy_id"][:n_rows],
                source["seed"][:n_rows],
                deterministic_policy_ids,
            )
            retained_episode_count = int(np.unique(episode_ids[retained]).size)
            retained_trajectory_count = int(
                np.unique(trajectory_ids[retained]).size
            )
            retained_source_file_count = int(
                np.unique(source_files[retained]).size
            )

            with h5py.File(temporary, "w") as target:
                observations_out = _dataset(
                    target, "observations", (output_rows, CANONICAL_STATE_DIM), "float32", chunk_rows
                )
                next_observations_out = _dataset(
                    target,
                    "next_observations",
                    (output_rows, CANONICAL_STATE_DIM),
                    "float32",
                    chunk_rows,
                )
                actions_out = _dataset(target, "actions", (output_rows, ACTION_DIM), "float32", chunk_rows)
                rewards_out = _dataset(target, "rewards", (output_rows,), "float32", chunk_rows)
                discounts_out = _dataset(target, "discounts", (output_rows,), "float32", chunk_rows)
                durations_out = _dataset(target, "delta_t_seconds", (output_rows,), "float32", chunk_rows)
                terminals_out = _dataset(target, "terminals", (output_rows,), "uint8", chunk_rows)
                policy_out = _dataset(target, "policy_id", (output_rows,), "int8", chunk_rows)
                policy_seed_out = _dataset(target, "policy_seed", (output_rows,), "int32", chunk_rows)
                sumo_seed_out = _dataset(target, "sumo_seed", (output_rows,), "int32", chunk_rows)
                split_out = _dataset(target, "split_id", (output_rows,), "uint8", chunk_rows)
                source_index_out = _dataset(target, "source_index", (output_rows,), "int64", chunk_rows)
                source_file_out = _dataset(target, "source_file_id", (output_rows,), "int32", chunk_rows)
                episode_out = _dataset(target, "episode_id", (output_rows,), "int64", chunk_rows)
                trajectory_out = _dataset(target, "trajectory_id", (output_rows,), "int64", chunk_rows)

                write_at = 0
                for start in range(0, n_rows, chunk_rows):
                    stop = min(start + chunk_rows, n_rows)
                    policy_seeds = source["seed"][start:stop].astype(np.int32, copy=False)
                    policies = source["policy_id"][start:stop].astype(np.int8, copy=False)
                    keep = retained_legacy_rows(
                        policies, policy_seeds, deterministic_policy_ids
                    )
                    if not keep.any():
                        continue
                    raw_obs = source["observations"][start:stop][keep]
                    raw_next = source["next_observations"][start:stop][keep]
                    raw_actions = source["actions"][start:stop][keep]
                    policy_seeds = policy_seeds[keep]
                    policies = policies[keep]

                    batch_diagnostics = validate_v2_transition_batch(
                        raw_obs, raw_actions, raw_next
                    )
                    for key, value in batch_diagnostics.items():
                        if key != "rows":
                            diagnostics[key] += value
                    beyond_horizon = int(
                        (raw_next[:, 10] > config.episode_seconds).sum()
                    )
                    diagnostics["beyond_episode_horizon"] += beyond_horizon

                    fatal = (
                        batch_diagnostics["different_line"]
                        + batch_diagnostics["different_bus"]
                        + batch_diagnostics["nonpositive_duration"]
                        + batch_diagnostics["nonforward_station"]
                        + batch_diagnostics["action_out_of_range"]
                        + beyond_horizon
                    )
                    if fatal:
                        raise ValueError(
                            f"non-salvageable v2 transitions in rows {start}:{stop}: "
                            f"{batch_diagnostics}"
                        )

                    actions = raw_actions[:, :1].astype(np.float32, copy=False)
                    durations = transition_duration_seconds(raw_obs, raw_next)
                    splits = np.zeros(len(raw_obs), dtype=np.uint8)
                    output_stop = write_at + len(raw_obs)
                    output_slice = slice(write_at, output_stop)

                    observations_out[output_slice] = raw_v2_to_canonical(
                        raw_obs, station_max, config
                    )
                    next_observations_out[output_slice] = raw_v2_to_canonical(
                        raw_next, station_max, config
                    )
                    actions_out[output_slice] = actions
                    rewards_out[output_slice] = transition_reward_from_v2(
                        raw_next, actions, config
                    )
                    discounts_out[output_slice] = continuous_time_discount(durations, config)
                    durations_out[output_slice] = durations
                    terminals_out[output_slice] = 0
                    policy_out[output_slice] = policies
                    policy_seed_out[output_slice] = policy_seeds
                    sumo_seed_out[output_slice] = -1
                    split_out[output_slice] = splits
                    source_index_out[output_slice] = np.arange(start, stop, dtype=np.int64)[keep]
                    source_file_out[output_slice] = source_files[start:stop][keep]
                    episode_out[output_slice] = episode_ids[start:stop][keep]
                    trajectory_out[output_slice] = trajectory_ids[start:stop][keep]

                    split_counts.update(map(str, splits.tolist()))
                    policy_counts.update(map(str, policies.tolist()))
                    write_at = output_stop

                if write_at != output_rows:
                    raise RuntimeError("legacy deduplication wrote an inconsistent row count")

                target.attrs.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
                        "source_path": str(input_path.resolve()),
                        "source_sha256": source_hash,
                        "source_rows": available,
                        "source_rows_examined": n_rows,
                        "n_transitions": output_rows,
                        "state_dim": CANONICAL_STATE_DIM,
                        "action_dim": ACTION_DIM,
                        "state_names": json.dumps(CANONICAL_STATE_NAMES),
                        "action_semantics": "a_hold in [-1,1]; hold_seconds=30*(a_hold+1); speed_ratio=1.0",
                        "transition_semantics": "same bus at its next station-arrival event",
                        "reward_semantics": "computed exactly once from next-event normalized headways and prior hold",
                        "discount_semantics": "gamma_per_minute ** (delta_t_seconds/60)",
                        "episode_horizon_semantics": (
                            "24000 s covers departures through 18000 s and all observed route completions"
                        ),
                        "config": json.dumps(config.to_dict(), sort_keys=True),
                        "station_max_by_line": json.dumps(station_max.tolist()),
                        "policy_names": json.dumps(policy_names, sort_keys=True),
                        "split_names": json.dumps(SPLIT_NAMES, sort_keys=True),
                        "split_semantics": "legacy data are training-only; evaluation requires fresh SUMO seeds",
                        "policy_seed_semantics": "legacy --seed affected policy RNG only",
                        "sumo_seed_semantics": "-1 means unknown: legacy collector did not propagate --seed to SUMO",
                        "deduplication": (
                            "for deterministic policies heuristic_best, sac and zero, retain only "
                            f"policy_seed={CANONICAL_DETERMINISTIC_POLICY_SEED}; exact cross-seed copies removed"
                        ),
                        "source_rows_dropped_as_exact_seed_copies": n_rows - output_rows,
                        "n_source_files": retained_source_file_count,
                        "n_episodes": retained_episode_count,
                        "n_bus_trajectories": retained_trajectory_count,
                        "episode_id_semantics": (
                            "source-file-local service days recovered from event-time drops greater than "
                            f"{config.episode_seconds * EPISODE_RESET_FRACTION:g} seconds"
                        ),
                        "trajectory_id_semantics": "unique source episode, line and physical bus/trip identifier",
                        "legacy_second_action": "discarded: v2 SUMO collector did not apply speed action",
                        "legacy_segment_speed": (
                            "discarded: SUMO lane mean speed and low-fidelity bus running speed "
                            "were not semantically equivalent"
                        ),
                        "legacy_reward": "discarded: v3 rewards are recomputed; no loader-side correction allowed",
                    }
                )
                target.flush()
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": source_hash,
        "output": str(output_path.resolve()),
        "source_rows_examined": n_rows,
        "rows": output_rows,
        "rows_dropped_as_exact_seed_copies": n_rows - output_rows,
        "source_files": retained_source_file_count,
        "episodes": retained_episode_count,
        "bus_trajectories": retained_trajectory_count,
        "diagnostics": dict(diagnostics),
        "split_counts": dict(sorted(split_counts.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "station_max_by_line": station_max.tolist(),
        "config": config.to_dict(),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=0, help="Convert only the first N rows")
    parser.add_argument("--reward-hold-weight", type=float, default=0.05)
    parser.add_argument("--reward-balance-weight", type=float, default=0.10)
    parser.add_argument("--gamma-per-minute", type=float, default=0.99)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    config = TransitMDPConfig(
        reward_hold_weight=args.reward_hold_weight,
        reward_balance_weight=args.reward_balance_weight,
        discount_gamma_per_minute=args.gamma_per_minute,
    )
    print(
        json.dumps(
            convert(
                input_path,
                output_path,
                config,
                chunk_rows=args.chunk_rows,
                limit=args.limit,
                overwrite=args.overwrite,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
