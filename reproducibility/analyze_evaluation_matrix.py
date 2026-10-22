#!/usr/bin/env python3
"""Aggregate canonical evaluations with hierarchical CIs and paired tests."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "passenger_wait_seconds": (
        ("passengers", "accrued_wait_seconds_per_departed_passenger"),
        "lower",
    ),
    "headway_normalized_absolute_error": (
        ("operations", "headway", "normalized_absolute_error", "mean"),
        "lower",
    ),
    "bunching_rate": (("operations", "headway", "bunching_rate"), "lower"),
    "large_gap_rate": (("operations", "headway", "large_gap_rate"), "lower"),
    "completed_journey_seconds": (
        ("passengers", "completed_journey_seconds", "mean"),
        "lower",
    ),
    "passenger_completion_rate": (("passengers", "completion_rate"), "higher"),
    "canonical_smdp_return": (("operations", "cumulative_smdp_reward"), "higher"),
    "total_hold_seconds": (("operations", "total_hold_seconds"), "lower"),
}


def _nested(payload: dict, path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _algorithm_and_seed(payload: dict) -> tuple[str, int | None]:
    policy = payload.get("policy", {})
    metadata = policy.get("training_metadata", {})
    algorithm = metadata.get("method")
    training_seed = metadata.get("training_seed")
    if algorithm is None:
        algorithm = payload["method"]
    return str(algorithm), None if training_seed is None else int(training_seed)


def load_rows(root: Path, allow_partial: bool) -> tuple[list[dict], dict]:
    rows = []
    dataset_hashes = set()
    contracts = set()
    skipped_partial = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("schema_version") != "transit-evaluation-v3":
            continue
        if payload.get("episode", {}).get("stopped_by_event_limit") and not allow_partial:
            skipped_partial.append(str(path))
            continue
        algorithm, training_seed = _algorithm_and_seed(payload)
        scenario = payload.get("scenario", {})
        demand_scale = float(scenario.get("demand_scale", 1.0))
        row = {
            "path": str(path.resolve()),
            "algorithm": algorithm,
            "training_seed": training_seed,
            "sumo_seed": int(payload["sumo_seed"]),
            "demand_scale": demand_scale,
        }
        for metric, (metric_path, _) in METRICS.items():
            row[metric] = _nested(payload, metric_path)
        rows.append(row)
        dataset_hashes.add(payload.get("dataset", {}).get("sha256"))
        contracts.add(json.dumps(payload.get("contract", {}), sort_keys=True))
    if not rows:
        raise ValueError(f"no canonical evaluation JSON files found under {root}")
    if len(dataset_hashes) != 1:
        raise ValueError(f"evaluation files use multiple dataset hashes: {dataset_hashes}")
    if len(contracts) != 1:
        raise ValueError("evaluation files use multiple MDP contracts")
    evidence = {
        "dataset_sha256": next(iter(dataset_hashes)),
        "contract": json.loads(next(iter(contracts))),
        "skipped_partial": skipped_partial,
    }
    return rows, evidence


def _matrix(rows: list[dict], metric: str) -> tuple[np.ndarray, list, list]:
    seeds = sorted(
        {row["training_seed"] for row in rows},
        key=lambda value: -1 if value is None else value,
    )
    cells = sorted({(row["demand_scale"], row["sumo_seed"]) for row in rows})
    values = np.full((len(seeds), len(cells)), np.nan, dtype=np.float64)
    seed_index = {value: index for index, value in enumerate(seeds)}
    cell_index = {value: index for index, value in enumerate(cells)}
    seen = set()
    for row in rows:
        key = (row["training_seed"], row["demand_scale"], row["sumo_seed"])
        if key in seen:
            raise ValueError(f"duplicate evaluation cell {key} for {row['algorithm']}")
        seen.add(key)
        value = row[metric]
        if value is not None:
            values[
                seed_index[row["training_seed"]],
                cell_index[(row["demand_scale"], row["sumo_seed"])],
            ] = value
    return values, seeds, cells


def _hierarchical_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_rows = rng.integers(0, values.shape[0], values.shape[0])
        sampled_cols = rng.integers(0, values.shape[1], values.shape[1])
        estimates[index] = np.nanmean(values[np.ix_(sampled_rows, sampled_cols)])
    return tuple(map(float, np.percentile(estimates, (2.5, 97.5))))


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _paired_permutation_p(differences: np.ndarray) -> float:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return float("nan")
    observed = abs(float(differences.mean()))
    if len(differences) <= 20:
        means = (
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        )
        extreme = sum(value >= observed - 1e-12 for value in means)
        return extreme / float(2 ** len(differences))
    rng = np.random.default_rng(20_260_803)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted


def analyze(rows: list[dict], reference: str, bootstrap_replicates: int) -> dict:
    by_algorithm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_algorithm[row["algorithm"]].append(row)
    if reference not in by_algorithm:
        raise ValueError(f"reference algorithm {reference!r} is absent")
    rng = np.random.default_rng(20_260_803)
    summaries = {}
    matrices = {}
    for algorithm, algorithm_rows in sorted(by_algorithm.items()):
        algorithm_summary = {
            "evaluation_runs": len(algorithm_rows),
            "training_seeds": sorted(
                seed
                for seed in {row["training_seed"] for row in algorithm_rows}
                if seed is not None
            ),
            "metrics": {},
        }
        matrices[algorithm] = {}
        for metric, (_, direction) in METRICS.items():
            values, seeds, cells = _matrix(algorithm_rows, metric)
            matrices[algorithm][metric] = (values, cells)
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            ci_low, ci_high = _hierarchical_ci(values, rng, bootstrap_replicates)
            algorithm_summary["metrics"][metric] = {
                "direction": direction,
                "mean": float(np.mean(finite)),
                "std_across_runs": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "hierarchical_bootstrap_ci95": [ci_low, ci_high],
                "n_training_seeds": len(seeds),
                "n_scenario_cells": len(cells),
                "n_runs": len(finite),
            }
        summaries[algorithm] = algorithm_summary

    comparisons = []
    for metric, (_, direction) in METRICS.items():
        reference_values, reference_cells = matrices[reference][metric]
        reference_by_cell = {
            cell: _finite_mean(reference_values[:, index])
            for index, cell in enumerate(reference_cells)
        }
        metric_comparisons = []
        for algorithm in sorted(by_algorithm):
            if algorithm == reference:
                continue
            values, cells = matrices[algorithm][metric]
            algorithm_by_cell = {
                cell: _finite_mean(values[:, index])
                for index, cell in enumerate(cells)
            }
            common = sorted(set(reference_by_cell) & set(algorithm_by_cell))
            differences = np.asarray(
                [algorithm_by_cell[cell] - reference_by_cell[cell] for cell in common],
                dtype=np.float64,
            )
            differences = differences[np.isfinite(differences)]
            if not len(differences):
                continue
            bootstrap = np.asarray(
                [
                    rng.choice(differences, size=len(differences), replace=True).mean()
                    for _ in range(bootstrap_replicates)
                ]
            )
            sd = differences.std(ddof=1) if len(differences) > 1 else 0.0
            metric_comparisons.append(
                {
                    "algorithm": algorithm,
                    "reference": reference,
                    "metric": metric,
                    "direction": direction,
                    "mean_difference_algorithm_minus_reference": float(differences.mean()),
                    "paired_bootstrap_ci95": list(map(float, np.percentile(bootstrap, (2.5, 97.5)))),
                    "paired_cohen_dz": float(differences.mean() / sd) if sd > 0 else None,
                    "paired_permutation_p_raw": _paired_permutation_p(differences),
                    "n_common_scenario_cells": len(differences),
                }
            )
        adjusted = _holm_adjust(
            [entry["paired_permutation_p_raw"] for entry in metric_comparisons]
        )
        for entry, p_adjusted in zip(metric_comparisons, adjusted):
            entry["paired_permutation_p_holm"] = p_adjusted
        comparisons.extend(metric_comparisons)
    return {
        "schema_version": "transit-statistical-analysis-v1",
        "primary_metrics": [
            "passenger_wait_seconds",
            "headway_normalized_absolute_error",
        ],
        "multiplicity": "Holm adjustment within each metric across comparisons to the prespecified reference",
        "algorithms": summaries,
        "comparisons": comparisons,
    }


def write_flat_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "algorithm",
        "training_seed",
        "sumo_seed",
        "demand_scale",
        *METRICS,
        "path",
    ]
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rows, evidence = load_rows(args.input_root.resolve(), args.allow_partial)
    report = analyze(rows, args.reference, args.bootstrap_replicates)
    report["evidence"] = evidence
    report["input_root"] = str(args.input_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    write_flat_csv(args.output.with_suffix(".csv"), rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
