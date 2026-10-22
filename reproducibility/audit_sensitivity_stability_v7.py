#!/usr/bin/env python3
"""Audit registered H2O+ lambda runs against the frozen pilot stability gates."""

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
from reproducibility.run_training_matrix_v4 import load_plan
from SimpleSAC.train_transit_v3 import sha256


SCHEMA_VERSION = "transit-v7-h2oplus-sensitivity-stability-audit-v1"
METHODS = (
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
)
SEEDS = tuple(range(5101, 5106))
EXPECTED_JOBS = {
    f"{method}_seed{seed}" for method in METHODS for seed in SEEDS
}
POSITIVE_ACTION_ABSOLUTE_LIMIT = 0.98


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
    return [
        (int(row["epoch"]), float(value))
        for row in rows
        if isinstance((value := row.get("train", {}).get(name)), (int, float))
    ]


def _violations(
    values: list[tuple[int, float]], predicate: Callable[[float], bool]
) -> list[dict[str, Any]]:
    return [
        {"epoch": epoch, "value": value}
        for epoch, value in values
        if predicate(value)
    ]


def _episodes(
    epochs: list[int], valid_by_epoch: dict[int, bool]
) -> list[dict[str, Any]]:
    if not epochs:
        return []
    groups: list[list[int]] = [[epochs[0]]]
    for epoch in epochs[1:]:
        if epoch == groups[-1][-1] + 1:
            groups[-1].append(epoch)
        else:
            groups.append([epoch])
    available = sorted(valid_by_epoch)
    episodes = []
    for group in groups:
        recovery = next(
            (
                epoch
                for epoch in available
                if epoch > group[-1] and valid_by_epoch[epoch]
            ),
            None,
        )
        episodes.append(
            {
                "first_epoch": group[0],
                "last_epoch": group[-1],
                "epochs": group,
                "first_recovery_epoch": recovery,
                "recovered": recovery is not None,
            }
        )
    return episodes


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def audit_run(run_dir: Path, expected_signature: str | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
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
    if len(rows) != 190 or any(len(series) != 190 for series in (critic, hold, action)):
        raise ValueError(f"incomplete sensitivity diagnostic series: {run_dir}")

    run_signature = summary.get("run_signature")
    if metadata.get("run_signature") != run_signature:
        raise ValueError(f"summary/metadata signature mismatch: {run_dir.name}")
    if expected_signature is not None and run_signature != expected_signature:
        raise ValueError(f"registered signature mismatch: {run_dir.name}")

    critic_violations = _violations(critic, lambda value: value > NUMERICAL_LIMIT)
    hold_violations = _violations(
        hold,
        lambda value: not HOLD_PROBABILITY_RANGE[0]
        <= value
        <= HOLD_PROBABILITY_RANGE[1],
    )
    action_violations = _violations(
        action, lambda value: abs(value) >= POSITIVE_ACTION_ABSOLUTE_LIMIT
    )
    hold_by_epoch = dict(hold)
    action_by_epoch = dict(action)
    policy_valid = {
        epoch: (
            HOLD_PROBABILITY_RANGE[0]
            <= hold_by_epoch[epoch]
            <= HOLD_PROBABILITY_RANGE[1]
            and abs(action_by_epoch[epoch]) < POSITIVE_ACTION_ABSOLUTE_LIMIT
        )
        for epoch in sorted(set(hold_by_epoch) & set(action_by_epoch))
    }
    policy_violation_epochs = sorted(
        {
            record["epoch"]
            for record in hold_violations + action_violations
        }
    )
    all_finite = _finite(rows) and _finite(summary) and _finite(metadata)
    passed = bool(
        all_finite
        and not critic_violations
        and not hold_violations
        and not action_violations
    )

    return {
        "name": run_dir.name,
        "method": summary["method"],
        "seed": int(summary["seed"]),
        "run_signature": run_signature,
        "passed": passed,
        "all_reported_metrics_finite": all_finite,
        "epochs": len(rows),
        "total_updates": summary.get("total_updates"),
        "critic_gradient": {
            "maximum": max(value for _, value in critic),
            "maximum_epoch": max(critic, key=lambda item: item[1])[0],
            "violations": critic_violations,
        },
        "policy_nondegeneracy": {
            "minimum_hold_probability": min(value for _, value in hold),
            "minimum_hold_probability_epoch": min(hold, key=lambda item: item[1])[0],
            "maximum_hold_probability": max(value for _, value in hold),
            "maximum_hold_probability_epoch": max(hold, key=lambda item: item[1])[0],
            "maximum_absolute_positive_action": max(abs(value) for _, value in action),
            "maximum_absolute_positive_action_epoch": max(
                action, key=lambda item: abs(item[1])
            )[0],
            "hold_probability_violations": hold_violations,
            "positive_action_violations": action_violations,
            "violation_epochs": policy_violation_epochs,
            "violation_episodes": _episodes(policy_violation_epochs, policy_valid),
            "final": {
                "epoch": hold[-1][0],
                "hold_probability": hold[-1][1],
                "positive_action_mean": action[-1][1],
            },
        },
        "artifacts": {
            "metrics": _artifact(metrics_path),
            "summary": _artifact(summary_path),
            "metadata": _artifact(metadata_path),
        },
    }


def audit(matrix_root: Path, plan_path: Path) -> dict[str, Any]:
    matrix_root = matrix_root.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    jobs, evidence = load_plan(plan_path)
    by_name = {job["name"]: job for job in jobs}
    if set(by_name) != EXPECTED_JOBS or len(jobs) != len(EXPECTED_JOBS):
        raise ValueError("training plan is not the exact registered 20-job lambda matrix")
    runs = [
        audit_run(matrix_root / name, by_name[name]["signature"])
        for name in sorted(EXPECTED_JOBS)
    ]
    failed = [run["name"] for run in runs if not run["passed"]]
    threshold_source = Path(__file__).with_name("run_hybrid_entropy_pilot_v7.py")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "scientific training-internal stability; separate from artifact integrity",
        "artifact_integrity_implication": "none",
        "supports_held_out_model_effect_claim": False,
        "supports_formal_method_selection": False,
        "selection_policy": (
            "all registered methods and seeds are retained; a threshold failure, "
            "recovery, or unfavorable outcome cannot remove or replace a run"
        ),
        "thresholds": {
            "critic_gradient_norm_maximum_inclusive": NUMERICAL_LIMIT,
            "hold_probability_range_inclusive": list(HOLD_PROBABILITY_RANGE),
            "positive_action_mean_absolute_maximum_exclusive": (
                POSITIVE_ACTION_ABSOLUTE_LIMIT
            ),
            "source": _artifact(threshold_source),
        },
        "training_plan": {
            "path": str(plan_path),
            "sha256": evidence["plan_sha256"],
        },
        "matrix_root": str(matrix_root),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "runs_expected": sorted(EXPECTED_JOBS),
        "runs_audited": [run["name"] for run in runs],
        "failed_runs": failed,
        "passed_runs": [run["name"] for run in runs if run["passed"]],
        "failed_run_count": len(failed),
        "scientific_stability_passed": not failed,
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
