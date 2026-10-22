#!/usr/bin/env python3
"""Analyze formal v7 H2O+ mechanisms without making performance claims."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.train_transit_v3 import sha256


SCHEMA_VERSION = "transit-h2oplus-mechanism-diagnostic-v7-v1"
SEEDS = tuple(range(5101, 5106))
METHODS = (
    "h2oplus",
    "h2oplus-source",
    "h2oplus-no-ratio",
    "h2oplus-no-warmstart",
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
)
TAIL_EVENTS = 20_000
EXPECTED_EPOCHS = {method: 190 for method in METHODS}
EXPECTED_EPOCHS["h2oplus-source"] = 198


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return False


def tail_event_weighted_reward(
    rows: list[dict[str, Any]], event_budget: int = TAIL_EVENTS
) -> dict[str, float | int]:
    if event_budget <= 0:
        raise ValueError("tail event budget must be positive")
    remaining = event_budget
    weighted_sum = 0.0
    used = 0
    for row in reversed(rows):
        rollout = row.get("rollout", {})
        events = int(rollout.get("events", 0))
        reward = rollout.get("mean_reward")
        if events <= 0 or not isinstance(reward, (int, float)):
            continue
        take = min(events, remaining)
        weighted_sum += take * float(reward)
        used += take
        remaining -= take
        if remaining == 0:
            break
    if used != event_budget:
        raise ValueError(f"run contains only {used} usable tail events")
    return {"events": used, "mean_reward_per_event": weighted_sum / used}


def _series(
    rows: list[dict[str, Any]], section: str, key: str
) -> list[tuple[int, float]]:
    values = []
    for row in rows:
        value = row.get(section, {}).get(key)
        if isinstance(value, (int, float)):
            values.append((int(row["epoch"]), float(value)))
    return values


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "last": values[-1],
    }


def run_diagnostics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    method = summary.get("method")
    expected_epochs = EXPECTED_EPOCHS.get(method)
    if expected_epochs is None:
        raise ValueError(f"unsupported formal H2O+ method: {method!r}")
    if len(rows) != expected_epochs or summary.get("completed_epochs") != expected_epochs:
        raise ValueError(f"incomplete formal H2O+ run: {run_dir}")
    if not (_finite(rows) and _finite(summary) and _finite(metadata)):
        raise ValueError(f"nonfinite formal H2O+ artifacts: {run_dir}")

    critic = _series(rows, "train", "train/critic_grad_norm")
    hold = _series(rows, "train", "train/hold_probability")
    action = _series(rows, "train", "train/positive_action_mean")
    ratio_keys = (
        "ratio/validation_joint_auc",
        "ratio/validation_state_action_auc",
        "ratio/validation_weight_mean",
        "ratio/validation_raw_ess_fraction",
        "ratio/validation_ess_fraction",
        "ratio/validation_high_clip_fraction",
        "ratio/validation_low_clip_fraction",
    )
    ratio = {
        key.removeprefix("ratio/validation_"): _summary(
            [value for _, value in _series(rows, "ratio_validation", key)]
        )
        for key in ratio_keys
    }
    ratio = {key: value for key, value in ratio.items() if value is not None}
    return {
        "method": summary["method"],
        "seed": int(summary["seed"]),
        "run_signature": summary["run_signature"],
        "completed_epochs": summary["completed_epochs"],
        "total_updates": summary["total_updates"],
        "simulator_decision_events": summary["simulator_decision_events"],
        "tail_training_simulator_reward": tail_event_weighted_reward(rows),
        "critic_gradient": {
            "maximum": max(value for _, value in critic),
            "maximum_epoch": max(critic, key=lambda item: item[1])[0],
        },
        "policy": {
            "minimum_hold_probability": min(value for _, value in hold),
            "maximum_hold_probability": max(value for _, value in hold),
            "maximum_absolute_positive_action": max(abs(value) for _, value in action),
        },
        "ratio_validation": ratio or None,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            "metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
        },
    }


def _find_run(roots: list[Path], name: str) -> Path:
    matches = [root / name for root in roots if (root / name).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one synchronized run {name}, found {matches}")
    return matches[0]


def _validated_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    if not methods:
        raise ValueError("at least one H2O+ method is required")
    if len(methods) != len(set(methods)):
        raise ValueError("H2O+ mechanism methods must be unique")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unsupported H2O+ mechanism methods: {unknown}")
    if "h2oplus" not in methods:
        raise ValueError("H2O+ mechanism analysis requires the h2oplus reference")
    return methods


def analyze(
    roots: list[Path], methods: tuple[str, ...] = METHODS
) -> dict[str, Any]:
    roots = [root.expanduser().resolve() for root in roots]
    methods = _validated_methods(tuple(methods))
    runs = [
        run_diagnostics(_find_run(roots, f"{method}_seed{seed}"))
        for method in methods
        for seed in SEEDS
    ]
    by_method = {
        method: sorted(
            (run for run in runs if run["method"] == method),
            key=lambda run: run["seed"],
        )
        for method in methods
    }
    core_tail = {
        run["seed"]: run["tail_training_simulator_reward"]["mean_reward_per_event"]
        for run in by_method["h2oplus"]
    }
    comparisons = {}
    for method in methods:
        if method == "h2oplus":
            continue
        deltas = [
            run["tail_training_simulator_reward"]["mean_reward_per_event"]
            - core_tail[run["seed"]]
            for run in by_method[method]
        ]
        comparisons[method] = {
            "paired_seed_deltas_vs_h2oplus": dict(zip(SEEDS, deltas)),
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "seeds_with_higher_training_simulator_reward": sum(
                delta > 0 for delta in deltas
            ),
        }

    aggregates = {}
    for method, method_runs in by_method.items():
        rewards = [
            run["tail_training_simulator_reward"]["mean_reward_per_event"]
            for run in method_runs
        ]
        aggregates[method] = {
            "seeds": list(SEEDS),
            "tail_training_simulator_reward_mean": statistics.fmean(rewards),
            "tail_training_simulator_reward_sd": statistics.stdev(rewards),
            "maximum_critic_gradient": max(
                run["critic_gradient"]["maximum"] for run in method_runs
            ),
        }

    ratio_runs = by_method["h2oplus"]
    high_clip = [
        run["ratio_validation"]["high_clip_fraction"]["mean"]
        for run in ratio_runs
    ]
    raw_ess = [
        run["ratio_validation"]["raw_ess_fraction"]["mean"]
        for run in ratio_runs
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory training-internal mechanism diagnosis",
        "scientific_interpretation": {
            "supports_held_out_model_effect_claim": False,
            "supports_formal_method_selection": False,
            "reason": (
                "simulator training rewards and optimizer diagnostics are not held-out "
                "target-SUMO outcomes"
            ),
        },
        "roots": [str(root) for root in roots],
        "methods": list(methods),
        "seeds": list(SEEDS),
        "runs": runs,
        "aggregates": aggregates,
        "paired_training_simulator_comparisons": comparisons,
        "ratio_diagnostic": {
            "mean_validation_high_clip_fraction": statistics.fmean(high_clip),
            "mean_validation_raw_ess_fraction": statistics.fmean(raw_ess),
            "post_hoc_engineering_flag": bool(
                statistics.fmean(high_clip) >= 0.25
                or statistics.fmean(raw_ess) <= 0.5
            ),
            "flag_is_not_a_preregistered_scientific_gate": True,
        },
        "required_next_step": (
            "evaluate any improvement candidate on development scenarios disjoint from "
            "the frozen formal held-out environment before formal confirmation"
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=METHODS,
        help=(
            "method to include; repeat for a stage-local analysis. Defaults to "
            "the complete core, ablation, and sensitivity set"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args.matrix_root, tuple(args.methods or METHODS))
    _write_json(args.output, report)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
