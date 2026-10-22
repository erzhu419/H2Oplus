#!/usr/bin/env python3
"""Analyze paired episode welfare from a completed SUMO v5 collection matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect_policy.build_transit_v5_dataset import (
    discover_shards,
    sha256,
    validate_collection_matrix,
)


LOWER_IS_BETTER_METRICS = {
    "generalized_seconds_per_departed_passenger": (
        "passenger_welfare",
        "accrued_generalized_seconds_per_departed_passenger",
    ),
    "wait_seconds_per_departed_passenger": (
        "passengers",
        "accrued_wait_seconds_per_departed_passenger",
    ),
    "in_vehicle_seconds_per_departed_passenger": (
        "passengers",
        "accrued_in_vehicle_seconds_per_departed_passenger",
    ),
}


def _nested(payload: dict, path: tuple[str, ...]) -> float:
    value = payload
    for name in path:
        value = value[name]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite metric at {'.'.join(path)}")
    return result


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _paired_summary(
    method_values: list[float], baseline_values: list[float]
) -> dict:
    method = np.asarray(method_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if method.shape != baseline.shape or not method.size:
        raise ValueError("paired metric vectors must be non-empty and aligned")
    difference = method - baseline
    relative = difference / baseline
    rng = np.random.default_rng(20_260_804)
    bootstrap_means = np.mean(
        difference[rng.integers(0, difference.size, size=(10_000, difference.size))],
        axis=1,
    )
    return {
        "difference_semantics": "method minus zero; negative favors the method",
        "absolute_difference": _summary(difference.tolist()),
        "relative_difference": _summary(relative.tolist()),
        "wins": int(np.sum(difference < 0.0)),
        "ties": int(np.sum(difference == 0.0)),
        "losses": int(np.sum(difference > 0.0)),
        "bootstrap_mean_difference_95_percent_interval": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "inferential_warning": (
            "fewer than five independent blocks; use for engineering triage only"
            if difference.size < 5
            else None
        ),
    }


def analyze(matrix_manifest: Path) -> dict:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    matrix = json.loads(matrix_manifest.read_text())
    output_root = matrix_manifest.parent
    shard_paths = discover_shards(output_root)
    provenance = validate_collection_matrix(matrix_manifest, shard_paths)

    jobs = matrix["jobs"]
    rows = []
    by_cell = {}
    for job in jobs:
        shard = Path(job["output"]).expanduser().resolve()
        summary = json.loads(shard.with_suffix(shard.suffix + ".json").read_text())
        block_id = int(job["block_id"])
        policy_name = job["policy"]["policy_name"]
        key = (block_id, policy_name)
        if key in by_cell:
            raise ValueError(f"duplicate policy/block result {key}")
        metrics = {
            name: _nested(summary, path)
            for name, path in LOWER_IS_BETTER_METRICS.items()
        }
        row = {
            "block_id": block_id,
            "policy_name": policy_name,
            "scenario_manifest_sha256": job["scenario"]["sha256"],
            "demand_realization_seed": job["scenario"]["selection_seed"],
            "sumo_seed": int(job["sumo_seed"]),
            "metrics": metrics,
            "departed_passengers": int(summary["passengers"]["departed_passengers"]),
            "completed_passengers": int(summary["passengers"]["completed_passengers"]),
            "completion_rate": float(summary["passengers"]["completion_rate"]),
            "mean_hold_seconds": float(summary["actions"]["mean_hold_seconds"]),
            "exact_zero_rate": float(summary["actions"]["exact_zero_rate"]),
            "wall_time_seconds": float(summary["runtime"]["wall_time_seconds"]),
        }
        by_cell[key] = row
        rows.append(row)

    policies = sorted({row["policy_name"] for row in rows})
    block_ids = sorted({row["block_id"] for row in rows})
    if "zero" not in policies:
        raise ValueError("paired analysis requires the zero policy")
    expected = {(block_id, policy) for block_id in block_ids for policy in policies}
    if set(by_cell) != expected:
        raise ValueError("the policy-by-block result matrix is incomplete")

    aggregate = {}
    for policy in policies:
        policy_rows = [by_cell[(block_id, policy)] for block_id in block_ids]
        metrics = {
            metric: _summary([row["metrics"][metric] for row in policy_rows])
            for metric in LOWER_IS_BETTER_METRICS
        }
        aggregate[policy] = {
            "metrics": metrics,
            "completion_rate": _summary(
                [row["completion_rate"] for row in policy_rows]
            ),
            "mean_hold_seconds": _summary(
                [row["mean_hold_seconds"] for row in policy_rows]
            ),
            "exact_zero_rate": _summary(
                [row["exact_zero_rate"] for row in policy_rows]
            ),
            "wall_time_seconds": _summary(
                [row["wall_time_seconds"] for row in policy_rows]
            ),
        }

    paired_vs_zero = {}
    for policy in policies:
        if policy == "zero":
            continue
        paired_vs_zero[policy] = {}
        for metric in LOWER_IS_BETTER_METRICS:
            method_values = [
                by_cell[(block_id, policy)]["metrics"][metric]
                for block_id in block_ids
            ]
            baseline_values = [
                by_cell[(block_id, "zero")]["metrics"][metric]
                for block_id in block_ids
            ]
            paired_vs_zero[policy][metric] = _paired_summary(
                method_values, baseline_values
            )

    primary = "generalized_seconds_per_departed_passenger"
    ranking = sorted(
        policies,
        key=lambda policy: aggregate[policy]["metrics"][primary]["mean"],
    )
    return {
        "schema_version": "transit-smdp-v5-collection-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_manifest": {
            "path": str(matrix_manifest),
            "sha256": sha256(matrix_manifest),
            "provenance": provenance,
        },
        "primary_metric": primary,
        "primary_metric_direction": "lower is better",
        "objective_guardrail": (
            "episode ledger welfare only; overlapping agent-centric transition "
            "rewards are not summed"
        ),
        "independent_environment_blocks": len(block_ids),
        "policies": policies,
        "ranking": ranking,
        "aggregate": aggregate,
        "paired_vs_zero": paired_vs_zero,
        "rows": sorted(rows, key=lambda row: (row["block_id"], row["policy_name"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.matrix_manifest)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "ranking": report["ranking"],
                "independent_environment_blocks": report[
                    "independent_environment_blocks"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
