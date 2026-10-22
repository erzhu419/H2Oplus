#!/usr/bin/env python3
"""Analyze v7 behavior data without treating correlated demand scales as replicas."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect_policy.build_transit_v4_dataset import discover_shards, sha256
from collect_policy.build_transit_v7_dataset import validate_collection_matrix
from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7


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
PRIMARY_METRIC = "generalized_seconds_per_departed_passenger"


def _nested(payload: dict, path: tuple[str, ...]) -> float:
    value = payload
    for name in path:
        value = value[name]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite metric at {'.'.join(path)}")
    return result


def _summary(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        raise ValueError("cannot summarize an empty vector")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
    return extreme / total


def clustered_paired_summary(
    cell_differences: dict[int, tuple[int, float]],
    *,
    seed: int = 20_260_808,
    bootstrap_replicates: int = 20_000,
    allow_single_realization: bool = False,
) -> dict:
    """Aggregate demand-scale cells within each independent realization first."""

    by_realization: dict[int, list[float]] = {}
    for realization_id, difference in cell_differences.values():
        by_realization.setdefault(int(realization_id), []).append(float(difference))
    if not by_realization:
        raise ValueError("clustered comparison has no realizations")
    if len(by_realization) < 2 and not allow_single_realization:
        raise ValueError("clustered comparison requires at least two realizations")
    condition_counts = {len(values) for values in by_realization.values()}
    if len(condition_counts) != 1:
        raise ValueError("realizations contain unequal demand-condition counts")
    cluster_means = np.asarray(
        [np.mean(by_realization[key]) for key in sorted(by_realization)],
        dtype=np.float64,
    )
    inferential = len(cluster_means) >= 2
    report = {
        "difference_semantics": "method minus zero; negative favors the method",
        "environment_cell_differences": _summary(
            [difference for _, difference in cell_differences.values()]
        ),
        "realization_block_mean_differences": _summary(cluster_means),
        "realization_block_wins": int(np.sum(cluster_means < 0.0)),
        "realization_block_ties": int(np.sum(cluster_means == 0.0)),
        "realization_block_losses": int(np.sum(cluster_means > 0.0)),
        "independent_resampling_units": len(cluster_means),
        "demand_conditions_per_unit": next(iter(condition_counts)),
        "inferential_statistics_estimable": inferential,
    }
    if inferential:
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0,
            len(cluster_means),
            size=(bootstrap_replicates, len(cluster_means)),
        )
        bootstrap = cluster_means[indices].mean(axis=1)
        report.update(
            {
                "cluster_bootstrap_mean_difference_95_percent_interval": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
                "exact_two_sided_sign_flip_pvalue": _exact_sign_flip_pvalue(
                    cluster_means
                ),
            }
        )
    else:
        report.update(
            {
                "cluster_bootstrap_mean_difference_95_percent_interval": None,
                "exact_two_sided_sign_flip_pvalue": None,
                "inference_disclosure": (
                    "pilot has one independent realization; uncertainty is not "
                    "estimated and these values are not publication evidence"
                ),
            }
        )
    return report


def analyze(matrix_manifest: Path) -> dict:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    matrix = json.loads(matrix_manifest.read_text())
    study_stage = matrix.get("study_stage")
    if study_stage not in {"pilot", "formal"}:
        raise ValueError("v7 matrix has an invalid study stage")
    shard_paths = discover_shards(matrix_manifest.parent)
    provenance = validate_collection_matrix(matrix_manifest, shard_paths)

    rows = []
    by_cell = {}
    for job in matrix["jobs"]:
        shard = Path(job["output"]).expanduser().resolve()
        summary = json.loads(shard.with_suffix(shard.suffix + ".json").read_text())
        block_id = int(job["block_id"])
        realization_id = int(job["realization_block_id"])
        policy_name = job["policy"]["policy_name"]
        key = (block_id, policy_name)
        if key in by_cell:
            raise ValueError(f"duplicate policy/environment result {key}")
        row = {
            "block_id": block_id,
            "realization_block_id": realization_id,
            "policy_name": policy_name,
            "demand_scale": float(job["scenario"]["demand_scale"]),
            "demand_realization_seed": int(job["scenario"]["selection_seed"]),
            "sumo_seed": int(job["sumo_seed"]),
            "metrics": {
                name: _nested(summary, path)
                for name, path in LOWER_IS_BETTER_METRICS.items()
            },
            "completion_rate": float(summary["passengers"]["completion_rate"]),
            "mean_hold_seconds": float(summary["actions"]["mean_hold_seconds"]),
            "p50_hold_seconds": float(summary["actions"]["p50_hold_seconds"]),
            "p95_hold_seconds": float(summary["actions"]["p95_hold_seconds"]),
            "max_hold_seconds": float(summary["actions"]["max_hold_seconds"]),
            "exact_zero_rate": float(summary["actions"]["exact_zero_rate"]),
            "wall_time_seconds": float(summary["runtime"]["wall_time_seconds"]),
        }
        by_cell[key] = row
        rows.append(row)

    policies = sorted({row["policy_name"] for row in rows})
    block_ids = sorted({row["block_id"] for row in rows})
    if "zero" not in policies:
        raise ValueError("paired behavior analysis requires the zero policy")
    expected = {(block_id, policy) for block_id in block_ids for policy in policies}
    if set(by_cell) != expected:
        raise ValueError("the policy-by-environment matrix is incomplete")

    aggregate = {}
    demand_scales = sorted({row["demand_scale"] for row in rows})
    for policy in policies:
        policy_rows = [by_cell[(block_id, policy)] for block_id in block_ids]
        aggregate[policy] = {
            "metrics": {
                metric: _summary(row["metrics"][metric] for row in policy_rows)
                for metric in LOWER_IS_BETTER_METRICS
            },
            "metrics_by_demand_scale": {
                str(scale): {
                    metric: _summary(
                        row["metrics"][metric]
                        for row in policy_rows
                        if row["demand_scale"] == scale
                    )
                    for metric in LOWER_IS_BETTER_METRICS
                }
                for scale in demand_scales
            },
            "completion_rate": _summary(row["completion_rate"] for row in policy_rows),
            "mean_hold_seconds": _summary(
                row["mean_hold_seconds"] for row in policy_rows
            ),
            "p50_hold_seconds": _summary(
                row["p50_hold_seconds"] for row in policy_rows
            ),
            "p95_hold_seconds": _summary(
                row["p95_hold_seconds"] for row in policy_rows
            ),
            "max_hold_seconds": _summary(
                row["max_hold_seconds"] for row in policy_rows
            ),
            "exact_zero_rate": _summary(row["exact_zero_rate"] for row in policy_rows),
            "wall_time_seconds": _summary(
                row["wall_time_seconds"] for row in policy_rows
            ),
        }

    paired_vs_zero = {}
    for policy_index, policy in enumerate(policies):
        if policy == "zero":
            continue
        paired_vs_zero[policy] = {}
        for metric_index, metric in enumerate(LOWER_IS_BETTER_METRICS):
            differences = {
                block_id: (
                    by_cell[(block_id, policy)]["realization_block_id"],
                    by_cell[(block_id, policy)]["metrics"][metric]
                    - by_cell[(block_id, "zero")]["metrics"][metric],
                )
                for block_id in block_ids
            }
            paired_vs_zero[policy][metric] = clustered_paired_summary(
                differences,
                seed=20_260_808 + 100 * policy_index + metric_index,
                allow_single_realization=study_stage == "pilot",
            )

    ranking = sorted(
        policies,
        key=lambda policy: aggregate[policy]["metrics"][PRIMARY_METRIC]["mean"],
    )
    realization_ids = sorted({row["realization_block_id"] for row in rows})
    broad_policy_name = "random-hurdle-p0.5-uniform"
    broad_rows = [row for row in rows if row["policy_name"] == broad_policy_name]
    minimum_p95 = 0.75 * DEFAULT_CONFIG_V7.max_hold_seconds
    minimum_max = 0.95 * DEFAULT_CONFIG_V7.max_hold_seconds
    support_checks = {
        "policy_present": bool(broad_rows),
        "all_environment_cells_present": len(broad_rows) == len(block_ids),
        "all_cell_p95_at_least_75_percent_of_action_range": bool(broad_rows)
        and all(row["p95_hold_seconds"] >= minimum_p95 for row in broad_rows),
        "all_cell_max_at_least_95_percent_of_action_range": bool(broad_rows)
        and all(row["max_hold_seconds"] >= minimum_max for row in broad_rows),
        "all_cell_zero_rates_between_0.4_and_0.6": bool(broad_rows)
        and all(0.4 <= row["exact_zero_rate"] <= 0.6 for row in broad_rows),
    }
    action_support_gate = {
        "policy_name": broad_policy_name,
        "purpose": "logged support across zero and the full positive-hold range",
        "thresholds_seconds": {
            "minimum_cell_p95": minimum_p95,
            "minimum_cell_max": minimum_max,
        },
        "checks": support_checks,
        "passed": all(support_checks.values()),
    }
    return {
        "schema_version": "transit-smdp-v7-collection-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": study_stage,
        "matrix_manifest": {
            "path": str(matrix_manifest),
            "sha256": sha256(matrix_manifest),
            "provenance": provenance,
        },
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_direction": "lower is better",
        "resampling_unit": (
            "independent demand-realization/SUMO-seed block; all demand scales "
            "remain clustered"
        ),
        "independent_realization_blocks": len(realization_ids),
        "environment_cells": len(block_ids),
        "demand_scales": demand_scales,
        "policies": policies,
        "ranking": ranking,
        "action_support_gate": action_support_gate,
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
                "independent_realization_blocks": report[
                    "independent_realization_blocks"
                ],
                "environment_cells": report["environment_cells"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
