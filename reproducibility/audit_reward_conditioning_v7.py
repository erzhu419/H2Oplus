#!/usr/bin/env python3
"""Audit physical-reward scaling candidates without selecting on evaluation data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION
from bus_h2o.sumo_provenance import sha256


AUDIT_SCHEMA = "transit-smdp-v7-reward-conditioning-audit-v1"
DEFAULT_REWARD_SCALES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
DEFAULT_HUBER_DELTAS = (1.0,)
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def reward_audit_source_hashes() -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "bus_h2o" / "mdp_contract_v7.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sources
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--reward-scales",
        type=float,
        nargs="+",
        default=DEFAULT_REWARD_SCALES,
    )
    parser.add_argument(
        "--huber-deltas",
        type=float,
        nargs="+",
        default=DEFAULT_HUBER_DELTAS,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validated_candidates(values: list[float], name: str) -> list[float]:
    result = sorted(set(float(value) for value in values))
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot summarize an empty array")
    quantiles = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "quantiles": {
            f"q{quantile:g}": float(value)
            for quantile, value in zip(QUANTILES, quantiles)
        },
    }


def _safe_fraction(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _huber(errors: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(errors)
    return np.where(
        absolute <= delta,
        0.5 * np.square(errors),
        delta * (absolute - 0.5 * delta),
    )


def _loss_contribution(
    elementwise: np.ndarray, terminals: np.ndarray
) -> dict[str, Any]:
    total = float(np.sum(elementwise, dtype=np.float64))
    terminal = float(np.sum(elementwise[terminals], dtype=np.float64))
    nonterminal = total - terminal
    return {
        "total": total,
        "mean": float(np.mean(elementwise, dtype=np.float64)),
        "terminal": terminal,
        "nonterminal": nonterminal,
        "terminal_fraction": _safe_fraction(terminal, total),
    }


def _episode_returns(
    rewards: np.ndarray, episode_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    episodes, inverse = np.unique(episode_ids, return_inverse=True)
    returns = np.zeros(len(episodes), dtype=np.float64)
    np.add.at(returns, inverse, rewards)
    return episodes, returns


def _group_summary(
    rewards: np.ndarray,
    terminals: np.ndarray,
    episode_ids: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    selected_episodes = np.unique(episode_ids[selected])
    episode_selected = np.isin(episode_ids, selected_episodes)
    _, returns = _episode_returns(rewards[episode_selected], episode_ids[episode_selected])
    return {
        "rows": int(np.sum(selected)),
        "episodes": int(len(selected_episodes)),
        "terminal_rows": int(np.sum(terminals[selected])),
        "reward": _summary(rewards[selected]),
        "episode_return": _summary(returns),
    }


def audit(
    dataset: Path,
    *,
    reward_scales: list[float],
    huber_deltas: list[float],
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    with h5py.File(dataset, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("reward audit requires a transit-smdp-v7 dataset")
        required = {
            "rewards",
            "terminals",
            "episode_id",
            "policy_id",
            "demand_scale",
            "delta_t_seconds",
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"dataset is missing reward-audit arrays: {missing}")
        rewards = np.asarray(handle["rewards"], dtype=np.float64)
        terminals = np.asarray(handle["terminals"], dtype=bool)
        episode_ids = np.asarray(handle["episode_id"], dtype=np.int64)
        policy_ids = np.asarray(handle["policy_id"], dtype=np.int64)
        demand_scales = np.asarray(handle["demand_scale"], dtype=np.float64)
        durations = np.asarray(handle["delta_t_seconds"], dtype=np.float64)
        policy_mapping = json.loads(handle.attrs["policy_names"])
        content_signature = str(handle.attrs["content_signature"])

    n = rewards.size
    if n == 0 or any(
        values.shape != (n,)
        for values in (terminals, episode_ids, policy_ids, demand_scales, durations)
    ):
        raise ValueError("reward-audit arrays are empty or have inconsistent shapes")
    if not np.isfinite(rewards).all() or np.any(rewards > 1e-12):
        raise ValueError("v7 physical costs must produce finite nonpositive rewards")
    if np.any(durations < 0.0):
        raise ValueError("v7 transition durations must be nonnegative")
    episodes, episode_returns = _episode_returns(rewards, episode_ids)
    if int(np.sum(terminals)) != len(episodes):
        raise ValueError("each v7 episode must contain exactly one terminal row")

    policy_by_id = {int(identifier): name for name, identifier in policy_mapping.items()}
    unknown_policy_ids = sorted(set(np.unique(policy_ids)).difference(policy_by_id))
    if unknown_policy_ids:
        raise ValueError(f"unknown policy identifiers: {unknown_policy_ids}")

    groups = {
        "by_policy": {
            policy_by_id[identifier]: _group_summary(
                rewards,
                terminals,
                episode_ids,
                policy_ids == identifier,
            )
            for identifier in sorted(np.unique(policy_ids))
        },
        "by_demand_scale": {
            f"{scale:g}": _group_summary(
                rewards,
                terminals,
                episode_ids,
                demand_scales == scale,
            )
            for scale in sorted(np.unique(demand_scales))
        },
    }

    candidates: dict[str, Any] = {}
    for scale in reward_scales:
        scaled = rewards * scale
        absolute = np.abs(scaled)
        mse = np.square(scaled)
        candidate = {
            "reward_scale": scale,
            "scaled_reward": _summary(scaled),
            "absolute_scaled_reward": _summary(absolute),
            "scaled_episode_return": _summary(episode_returns * scale),
            "reward_only_initial_td_proxy": {
                "meaning": (
                    "prediction and bootstrap are set to zero; terminal TD is exact "
                    "at initialization only up to the critic prediction"
                ),
                "mse": _loss_contribution(mse, terminals),
                "huber": {},
            },
        }
        for delta in huber_deltas:
            candidate["reward_only_initial_td_proxy"]["huber"][f"delta_{delta:g}"] = (
                _loss_contribution(_huber(scaled, delta), terminals)
            )
        candidates[f"{scale:.12g}"] = candidate

    zero_duration = durations == 0.0
    return {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "dataset_content_signature": content_signature,
        "source_sha256": reward_audit_source_hashes(),
        "selection_boundary": (
            "descriptive offline-data audit only; it must not inspect reserved "
            "evaluation returns or select the final method ranking"
        ),
        "rows": n,
        "episodes": int(len(episodes)),
        "terminal_rows": int(np.sum(terminals)),
        "zero_duration_rows": int(np.sum(zero_duration)),
        "zero_reward_rows": int(np.sum(rewards == 0.0)),
        "physical_reward": _summary(rewards),
        "physical_episode_return": _summary(episode_returns),
        "groups": groups,
        "reward_scale_candidates": candidates,
        "huber_delta_candidates": huber_deltas,
    }


def main() -> None:
    args = parse_args()
    reward_scales = _validated_candidates(args.reward_scales, "reward-scales")
    huber_deltas = _validated_candidates(args.huber_deltas, "huber-deltas")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reward audit {output}")
    report = audit(
        args.dataset,
        reward_scales=reward_scales,
        huber_deltas=huber_deltas,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": report["rows"],
                "episodes": report["episodes"],
                "reward_scale_candidates": reward_scales,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
