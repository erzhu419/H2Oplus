#!/usr/bin/env python3
"""Produce a machine-readable audit of the canonical transit dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


STATE_BOUNDS = (
    (0.0, 11.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 5.0),
    (0.0, 5.0),
    (0.0, 5.0),
    (0.0, 1.5),
    (0.0, 2.0),
    (0.0, 5.0),
    (0.0, 5.0),
    (0.0, 2.0),
)
PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": int(len(values)), "finite": 0}
    quantiles = np.percentile(finite, PERCENTILES)
    return {
        "n": int(len(values)),
        "finite": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(PERCENTILES, quantiles)
        },
    }


def trajectory_validation_mask(trajectory_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectory_ids, dtype=np.uint64)
    mixed = values + np.uint64(0x9E3779B97F4A7C15)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed % np.uint64(5) == 0


def group_lengths(group_ids: np.ndarray) -> np.ndarray:
    _, counts = np.unique(group_ids, return_counts=True)
    return counts


def action_support(actions: np.ndarray) -> dict[str, Any]:
    edges = np.linspace(-1.0, 1.0, 21)
    counts, _ = np.histogram(actions, bins=edges)
    return {
        "summary": numeric_summary(actions),
        "exact_no_hold_fraction": float(np.mean(actions == -1.0)),
        "exact_max_hold_fraction": float(np.mean(actions == 1.0)),
        "bin_edges": edges.tolist(),
        "bin_counts": counts.tolist(),
        "occupied_bins": int(np.count_nonzero(counts)),
    }


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    return 0.5 * quadratic**2 + delta * (absolute - quadratic)


def audit(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        attrs = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in handle.attrs.items()
        }
        state_names = json.loads(attrs["state_names"])
        policy_name_to_id = json.loads(attrs["policy_names"])
        policy_id_to_name = {int(value): key for key, value in policy_name_to_id.items()}
        config = json.loads(attrs["config"])
        arrays = {
            name: handle[name][:]
            for name in (
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "discounts",
                "delta_t_seconds",
                "terminals",
                "policy_id",
                "policy_seed",
                "sumo_seed",
                "split_id",
                "source_file_id",
                "episode_id",
                "trajectory_id",
                "source_index",
            )
        }
        shapes = {
            name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in handle.items()
        }

    observations = arrays["observations"]
    next_observations = arrays["next_observations"]
    actions = arrays["actions"][:, 0]
    rewards = arrays["rewards"]
    durations = arrays["delta_t_seconds"]
    trajectory_ids = arrays["trajectory_id"]
    episode_ids = arrays["episode_id"]
    policy_ids = arrays["policy_id"]
    n_rows = len(actions)

    state = {}
    for index, (name, bounds) in enumerate(zip(state_names, STATE_BOUNDS)):
        values = observations[:, index]
        lower, upper = bounds
        state[name] = {
            **numeric_summary(values),
            "configured_bounds": [lower, upper],
            "at_lower_bound_fraction": float(np.mean(values == lower)),
            "at_upper_bound_fraction": float(np.mean(values == upper)),
            "outside_bounds": int(np.sum((values < lower) | (values > upper))),
        }

    delta = float(config["reward_huber_delta"])
    forward_error = next_observations[:, 3] - 1.0
    backward_error = next_observations[:, 4] - 1.0
    headway_cost = 0.5 * (_huber(forward_error, delta) + _huber(backward_error, delta))
    balance_cost = float(config["reward_balance_weight"]) * np.abs(
        next_observations[:, 3] - next_observations[:, 4]
    )
    hold_cost = float(config["reward_hold_weight"]) * np.clip(
        0.5 * (actions + 1.0), 0.0, 1.0
    )
    reconstructed_rewards = -(headway_cost + balance_cost + hold_cost)
    reconstructed_discounts = np.power(
        float(config["discount_gamma_per_minute"]), durations / 60.0
    )

    by_policy = {}
    for policy_id in sorted(np.unique(policy_ids)):
        selected = policy_ids == policy_id
        line_ids = np.rint(observations[selected, 0]).astype(np.int64)
        by_policy[policy_id_to_name.get(int(policy_id), str(int(policy_id)))] = {
            "policy_id": int(policy_id),
            "rows": int(selected.sum()),
            "source_files": int(np.unique(arrays["source_file_id"][selected]).size),
            "episodes": int(np.unique(episode_ids[selected]).size),
            "bus_trajectories": int(np.unique(trajectory_ids[selected]).size),
            "policy_seeds": sorted(map(int, np.unique(arrays["policy_seed"][selected]))),
            "line_counts": {
                str(int(line_id)): int(np.sum(line_ids == line_id))
                for line_id in np.unique(line_ids)
            },
            "actions": action_support(actions[selected]),
            "rewards": numeric_summary(rewards[selected]),
            "delta_t_seconds": numeric_summary(durations[selected]),
        }

    by_line = {}
    line_ids = np.rint(observations[:, 0]).astype(np.int64)
    for line_id in sorted(np.unique(line_ids)):
        selected = line_ids == line_id
        by_line[str(int(line_id))] = {
            "rows": int(selected.sum()),
            "episodes": int(np.unique(episode_ids[selected]).size),
            "bus_trajectories": int(np.unique(trajectory_ids[selected]).size),
            "actions": action_support(actions[selected]),
            "rewards": numeric_summary(rewards[selected]),
            "policy_counts": {
                policy_id_to_name.get(int(policy_id), str(int(policy_id))): int(
                    np.sum(policy_ids[selected] == policy_id)
                )
                for policy_id in np.unique(policy_ids[selected])
            },
        }

    unique_trajectories = np.unique(trajectory_ids)
    validation_trajectories = trajectory_validation_mask(unique_trajectories)
    validation_rows = trajectory_validation_mask(trajectory_ids)
    trajectory_order = np.argsort(trajectory_ids, kind="stable")
    ordered_trajectories = trajectory_ids[trajectory_order]
    starts = np.r_[0, np.flatnonzero(ordered_trajectories[1:] != ordered_trajectories[:-1]) + 1]
    group_episode = episode_ids[trajectory_order][starts]
    group_policy = policy_ids[trajectory_order][starts]
    expanded_group = np.searchsorted(ordered_trajectories[starts], ordered_trajectories)
    trajectory_group_conflicts = int(
        np.sum(episode_ids[trajectory_order] != group_episode[expanded_group])
        + np.sum(policy_ids[trajectory_order] != group_policy[expanded_group])
    )

    issues = []
    if np.all(arrays["sumo_seed"] == -1):
        issues.append(
            "Legacy collector did not record or propagate SUMO seeds; all offline rows have sumo_seed=-1."
        )
    if set(map(int, np.unique(arrays["split_id"]))) == {0}:
        issues.append(
            "The salvaged offline dataset is training-only; all evaluation must use fresh SUMO runs."
        )
    for name, details in state.items():
        if details["at_upper_bound_fraction"] > 0.01 and name not in {
            "line_id",
            "station_progress",
            "episode_progress",
        }:
            issues.append(
                f"State feature {name} is clipped at its upper bound in "
                f"{details['at_upper_bound_fraction']:.2%} of rows."
            )

    return {
        "schema_version": "transit-dataset-audit-v1",
        "dataset": str(path.resolve()),
        "sha256": sha256(path),
        "rows": n_rows,
        "attributes": attrs,
        "datasets": shapes,
        "source_units": {
            "source_files": int(np.unique(arrays["source_file_id"]).size),
            "episodes": int(np.unique(episode_ids).size),
            "bus_trajectories": int(len(unique_trajectories)),
            "rows_per_episode": numeric_summary(group_lengths(episode_ids)),
            "rows_per_bus_trajectory": numeric_summary(group_lengths(trajectory_ids)),
            "trajectory_group_conflicts": trajectory_group_conflicts,
            "unique_source_indices": int(np.unique(arrays["source_index"]).size),
        },
        "ratio_holdout": {
            "unit": "same-bus trajectory",
            "training_trajectories": int((~validation_trajectories).sum()),
            "validation_trajectories": int(validation_trajectories.sum()),
            "training_rows": int((~validation_rows).sum()),
            "validation_rows": int(validation_rows.sum()),
            "overlap": 0,
        },
        "state": state,
        "actions": action_support(actions),
        "rewards": numeric_summary(rewards),
        "delta_t_seconds": numeric_summary(durations),
        "discounts": numeric_summary(arrays["discounts"]),
        "objective_components": {
            "headway_cost": numeric_summary(headway_cost),
            "balance_cost": numeric_summary(balance_cost),
            "hold_cost": numeric_summary(hold_cost),
            "max_reward_reconstruction_error": float(
                np.max(np.abs(rewards - reconstructed_rewards))
            ),
            "max_discount_reconstruction_error": float(
                np.max(np.abs(arrays["discounts"] - reconstructed_discounts))
            ),
        },
        "by_policy": by_policy,
        "by_line_id": by_line,
        "issues_requiring_disclosure": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
