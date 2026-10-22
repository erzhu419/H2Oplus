#!/usr/bin/env python3
"""Audit preregistered RLPD runs against the registered numerical pilot gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_hybrid_entropy_pilot_v7 import (
    HOLD_PROBABILITY_RANGE,
    NUMERICAL_LIMIT,
)
from SimpleSAC.train_transit_v3 import sha256


SCHEMA_VERSION = "transit-v7-rlpd-long-horizon-stability-audit-v1"
POSITIVE_ACTION_ABSOLUTE_LIMIT = 0.98
EXPECTED_SEEDS = tuple(range(5101, 5106))


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


def _metric(rows: list[dict[str, Any]], name: str) -> list[tuple[int, float]]:
    values = []
    for row in rows:
        value = row.get("train", {}).get(name)
        if isinstance(value, (int, float)):
            values.append((int(row["epoch"]), float(value)))
    return values


def _first(values: list[tuple[int, float]], predicate: Callable[[float], bool]) -> dict[str, Any] | None:
    for epoch, value in values:
        if predicate(value):
            return {"epoch": epoch, "value": value}
    return None


def audit_run(run_dir: Path) -> dict[str, Any]:
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
    critic = _metric(rows, "train/critic_grad_norm")
    hold = _metric(rows, "train/hold_probability")
    action = _metric(rows, "train/positive_action_mean")
    if len(rows) != 190 or not critic or not hold or not action:
        raise ValueError(f"incomplete RLPD diagnostic series: {run_dir}")

    critic_crossing = _first(critic, lambda value: value > NUMERICAL_LIMIT)
    hold_crossing = _first(
        hold,
        lambda value: not HOLD_PROBABILITY_RANGE[0]
        <= value
        <= HOLD_PROBABILITY_RANGE[1],
    )
    action_crossing = _first(
        action, lambda value: abs(value) >= POSITIVE_ACTION_ABSOLUTE_LIMIT
    )
    policy_crossing_epoch = min(
        (
            record["epoch"]
            for record in (hold_crossing, action_crossing)
            if record is not None
        ),
        default=None,
    )
    recovered_policy = False
    recovery_epoch = None
    if policy_crossing_epoch is not None:
        hold_by_epoch = dict(hold)
        action_by_epoch = dict(action)
        for epoch in sorted(set(hold_by_epoch) & set(action_by_epoch)):
            if epoch <= policy_crossing_epoch:
                continue
            if (
                HOLD_PROBABILITY_RANGE[0]
                <= hold_by_epoch[epoch]
                <= HOLD_PROBABILITY_RANGE[1]
                and abs(action_by_epoch[epoch]) < POSITIVE_ACTION_ABSOLUTE_LIMIT
            ):
                recovered_policy = True
                recovery_epoch = epoch
                break

    all_finite = _finite(rows) and _finite(summary) and _finite(metadata)
    passed = bool(
        all_finite
        and critic_crossing is None
        and hold_crossing is None
        and action_crossing is None
    )
    return {
        "seed": int(summary["seed"]),
        "run_signature": summary["run_signature"],
        "passed": passed,
        "all_reported_metrics_finite": all_finite,
        "epochs": len(rows),
        "total_updates": summary.get("total_updates"),
        "critic_gradient": {
            "maximum": max(value for _, value in critic),
            "maximum_epoch": max(critic, key=lambda item: item[1])[0],
            "first_limit_crossing": critic_crossing,
        },
        "policy_nondegeneracy": {
            "minimum_hold_probability": min(value for _, value in hold),
            "minimum_hold_probability_epoch": min(hold, key=lambda item: item[1])[0],
            "maximum_hold_probability": max(value for _, value in hold),
            "maximum_absolute_positive_action": max(abs(value) for _, value in action),
            "maximum_absolute_positive_action_epoch": max(
                action, key=lambda item: abs(item[1])
            )[0],
            "first_hold_probability_crossing": hold_crossing,
            "first_positive_action_crossing": action_crossing,
            "first_crossing_epoch": policy_crossing_epoch,
            "recovered_inside_bounds_after_first_crossing": recovered_policy,
            "first_recovery_epoch": recovery_epoch,
        },
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            "metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
        },
    }


def audit(matrix_root: Path, plan_path: Path) -> dict[str, Any]:
    matrix_root = matrix_root.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text())
    expected = {
        int(job["seed"]): job["name"]
        for job in plan.get("jobs", [])
        if job.get("method") == "rlpd"
    }
    if tuple(sorted(expected)) != EXPECTED_SEEDS:
        raise ValueError("training plan does not contain the five formal RLPD seeds")
    runs = [audit_run(matrix_root / expected[seed]) for seed in EXPECTED_SEEDS]
    failures = [run["seed"] for run in runs if not run["passed"]]
    threshold_source = Path(__file__).with_name("run_hybrid_entropy_pilot_v7.py")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "scientific long-horizon stability; separate from artifact integrity",
        "selection_policy": (
            "all preregistered seeds are retained; a threshold failure, recovery, or "
            "unfavorable outcome cannot remove or replace a seed"
        ),
        "thresholds": {
            "critic_gradient_norm_maximum": NUMERICAL_LIMIT,
            "hold_probability_range_inclusive": list(HOLD_PROBABILITY_RANGE),
            "positive_action_mean_absolute_maximum_exclusive": (
                POSITIVE_ACTION_ABSOLUTE_LIMIT
            ),
            "source": {
                "path": str(threshold_source.resolve()),
                "sha256": sha256(threshold_source),
            },
        },
        "training_plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "matrix_root": str(matrix_root),
        "seeds_expected": list(EXPECTED_SEEDS),
        "seeds_audited": [run["seed"] for run in runs],
        "failed_seeds": failures,
        "passed_seeds": [run["seed"] for run in runs if run["passed"]],
        "failed_seed_count": len(failures),
        "scientific_stability_passed": not failures,
        "runs": runs,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.matrix_root, args.plan)
    _write_json(args.output, report)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
