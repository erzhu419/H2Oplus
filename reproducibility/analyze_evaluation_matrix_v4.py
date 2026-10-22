#!/usr/bin/env python3
"""Analyze a complete v4/v5/v6/v7 matrix with hierarchical CIs and paired tests."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bus_h2o.sumo_provenance import sha256


METRICS = {
    "generalized_seconds_per_passenger": (
        "passenger_welfare",
        "accrued_generalized_seconds_per_departed_passenger",
    ),
    "wait_seconds_per_passenger": (
        "passengers",
        "accrued_wait_seconds_per_departed_passenger",
    ),
    "in_vehicle_seconds_per_passenger": (
        "passengers",
        "accrued_in_vehicle_seconds_per_departed_passenger",
    ),
    "passenger_completion_rate": ("passengers", "completion_rate"),
    "headway_normalized_absolute_error": (
        "operations",
        "headway",
        "normalized_absolute_error",
        "mean",
    ),
    "mean_hold_seconds": ("operations", "hold_seconds", "mean"),
    "nonzero_hold_rate": ("operations", "nonzero_hold_rate"),
    "proxy_cost_passenger_minutes": ("objective", "cumulative_total_cost"),
}
METRICS_V5 = {
    name: path
    for name, path in METRICS.items()
    if name != "proxy_cost_passenger_minutes"
}
METRICS_V6 = dict(METRICS_V5)
METRICS_V7 = dict(METRICS_V5)
PRIMARY_METRIC = "generalized_seconds_per_passenger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--permutation-replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_803)
    return parser.parse_args()


def _nested(payload: dict, path: tuple[str, ...]):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def load_rows(matrix_root: Path) -> tuple[list[dict], dict]:
    root = matrix_root.expanduser().resolve()
    manifest_path = root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_schema = manifest.get("schema_version")
    schema_contracts = {
        "transit-evaluation-matrix-v4": (
            "v4",
            "transit-evaluation-v4",
            METRICS,
        ),
        "transit-evaluation-matrix-v5": (
            "v5",
            "transit-evaluation-v5",
            METRICS_V5,
        ),
        "transit-evaluation-matrix-v6": (
            "v6",
            "transit-evaluation-v6",
            METRICS_V6,
        ),
        "transit-evaluation-matrix-v7": (
            "v7",
            "transit-evaluation-v7",
            METRICS_V7,
        ),
    }
    if manifest_schema not in schema_contracts:
        raise ValueError("matrix manifest has an unsupported schema")
    mdp_version, evaluation_schema, metric_paths = schema_contracts[manifest_schema]
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("matrix manifest has no jobs")

    rows = []
    dataset_hashes = set()
    contracts = set()
    source_sets = set()
    output_hashes = {}
    seen = set()
    for job in jobs:
        output = Path(job["output"]).expanduser().resolve()
        if not output.is_file():
            raise FileNotFoundError(f"evaluation output is missing: {output}")
        payload = json.loads(output.read_text())
        if payload.get("schema_version") != evaluation_schema:
            raise ValueError(f"wrong evaluation schema in {output}")
        output_hashes[str(output)] = sha256(output)
        if payload.get("evaluation_job", {}).get("signature") != job["signature"]:
            raise ValueError(f"job signature mismatch in {output}")
        if payload.get("method") != job["method"]["name"]:
            raise ValueError(f"method name mismatch in {output}")
        if int(payload.get("sumo_seed")) != int(job["sumo_seed"]):
            raise ValueError(f"SUMO seed mismatch in {output}")
        if payload.get("dataset", {}).get("sha256") != job.get("dataset_sha256"):
            raise ValueError(f"dataset hash mismatch in {output}")
        if payload.get("source_sha256") != job.get("source_sha256"):
            raise ValueError(f"evaluator source hash mismatch in {output}")
        if payload.get("runtime", {}).get("sumo_version") != job.get("sumo_version"):
            raise ValueError(f"SUMO version mismatch in {output}")
        if payload.get("scenario", {}).get("runtime_input_sha256") != job.get(
            "scenario", {}
        ).get("runtime_input_sha256"):
            raise ValueError(f"scenario runtime hash mismatch in {output}")
        episode = payload.get("episode", {})
        if episode.get("stopped_by_event_limit") or episode.get(
            "stopped_by_time_limit"
        ):
            raise ValueError(f"partial diagnostic evaluation in formal matrix: {output}")
        configured = _finite_float(episode.get("configured_seconds"), "configured horizon")
        observed = _finite_float(
            episode.get("observed_end_time_seconds"), "observed horizon"
        )
        if observed + 1e-6 < configured:
            raise ValueError(f"evaluation ended before its contract horizon: {output}")
        if mdp_version in {"v6", "v7"}:
            coverage = payload.get("passenger_welfare", {}).get(
                "chronological_coverage_errors", {}
            )
            count_audit = payload.get("ledger_audit", {}).get(
                "count_source_audit", {}
            )
            service = payload.get("service", {})
            passengers = payload.get("passengers", {})
            if (
                not coverage
                or any(abs(float(value)) > 1e-9 for value in coverage.values())
                or int(count_audit.get("max_waiting_count_error", -1)) != 0
                or int(count_audit.get("max_in_vehicle_count_error", -1)) != 0
                or int(count_audit.get("subscription_fallback_queries", -1)) != 0
                or int(passengers.get("active_state_count_balance_error", -1)) != 0
                or int(service.get("active_trips_at_horizon", -1)) != 0
                or int(service.get("completed_trips", -1))
                != int(service.get("departed_trips", -2))
                or int(episode.get("terminal_transitions", -1)) != 1
                or int(episode.get("control_decisions", -1))
                != int(episode.get("transitions", -2))
            ):
                raise ValueError(
                    f"{mdp_version} accounting or service audit failed in {output}"
                )
        if mdp_version == "v7":
            dataset = payload.get("dataset", {})
            if (
                dataset.get("size_bytes") != job.get("dataset_size_bytes")
                or dataset.get("mtime_ns") != job.get("dataset_mtime_ns")
                or dataset.get("verification")
                != "parent-matrix-sha256-plus-child-stat-lock"
                or payload.get("bus_service_semantics_audit", {}).get("passed")
                is not True
            ):
                raise ValueError(f"v7 integrity or service audit failed in {output}")
            method = job["method"]
            if method["type"] == "checkpoint":
                policy = payload.get("policy", {})
                training = policy.get("training_metadata", {})
                if (
                    policy.get("sha256") != method.get("checkpoint_sha256")
                    or policy.get("artifact_schema")
                    != "transit-policy-artifact-v7-v1"
                    or training.get("phase") != "complete"
                    or training.get("method") != method.get("algorithm")
                    or training.get("training_seed") != method.get("training_seed")
                    or training.get("run_signature")
                    != method.get("training_run_signature")
                    or training.get("training_checkpoint_sha256")
                    != method.get("training_checkpoint_sha256")
                    or training.get("training_admission", {}).get("schema_version")
                    != "transit-training-admission-v7-v2"
                    or training.get("training_admission", {}).get("sha256")
                    != method.get("training_admission_sha256")
                ):
                    raise ValueError(f"v7 policy provenance failed in {output}")

        dataset_hash = payload.get("dataset", {}).get("sha256")
        dataset_hashes.add(dataset_hash)
        contracts.add(json.dumps(payload.get("contract"), sort_keys=True))
        source_sets.add(json.dumps(payload.get("source_sha256"), sort_keys=True))
        method = job["method"]
        algorithm = method.get("algorithm", method["name"])
        training_seed = method.get("training_seed")
        replicate = (
            f"seed-{training_seed}"
            if training_seed is not None
            else f"method-{method['name']}"
        )
        scenario = job["scenario"]
        cell = (str(scenario["tag"]), int(job["sumo_seed"]))
        environment_block = str(
            job.get("environment_block", f"sumo-seed-{int(job['sumo_seed'])}")
        )
        key = (algorithm, replicate, cell)
        if key in seen:
            raise ValueError(f"duplicate algorithm/replicate/cell {key}")
        seen.add(key)
        metrics = {
            name: _finite_float(_nested(payload, path), name)
            for name, path in metric_paths.items()
        }
        demand_realization_seed = payload.get("scenario", {}).get("selection_seed")
        if demand_realization_seed is not None:
            if isinstance(demand_realization_seed, bool):
                raise ValueError(f"invalid demand-realization seed in {output}")
            demand_realization_seed = int(demand_realization_seed)
        if mdp_version in {"v6", "v7"} and demand_realization_seed is None:
            raise ValueError(
                f"{mdp_version} evaluation lacks a demand-realization seed: {output}"
            )
        rows.append(
            {
                "algorithm": algorithm,
                "method": method["name"],
                "training_seed": training_seed,
                "replicate": replicate,
                "scenario": str(scenario["tag"]),
                "demand_scale": _finite_float(
                    payload["scenario"]["demand_scale"], "demand scale"
                ),
                "sumo_seed": int(job["sumo_seed"]),
                "demand_realization_seed": demand_realization_seed,
                "environment_block": environment_block,
                "cell": cell,
                "metrics": metrics,
                "output": str(output),
            }
        )
    if len(dataset_hashes) != 1:
        raise ValueError(f"matrix uses multiple datasets: {dataset_hashes}")
    if len(contracts) != 1:
        raise ValueError("matrix uses multiple MDP contracts")
    if len(source_sets) != 1:
        raise ValueError("matrix uses multiple evaluator source states")
    evidence = {
        "mdp_version": mdp_version,
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": sha256(manifest_path),
        "dataset_sha256": next(iter(dataset_hashes)),
        "contract": json.loads(next(iter(contracts))),
        "source_sha256": json.loads(next(iter(source_sets))),
        "evaluation_outputs_sha256": output_hashes,
        "evaluation_outputs_aggregate_sha256": hashlib.sha256(
            json.dumps(
                output_hashes, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    return rows, evidence


def build_cubes(rows: list[dict]) -> tuple[dict, list[tuple[str, int]], dict]:
    if not rows:
        raise ValueError("evaluation rows are empty")
    metric_names = tuple(rows[0]["metrics"])
    if any(tuple(row["metrics"]) != metric_names for row in rows):
        raise ValueError("evaluation rows expose inconsistent metric sets")
    cells = sorted({row["cell"] for row in rows})
    cell_metadata = {}
    for row in rows:
        existing = cell_metadata.setdefault(
            row["cell"],
            {
                "scenario": row["scenario"],
                "demand_scale": row["demand_scale"],
                "sumo_seed": row["sumo_seed"],
                "demand_realization_seed": row.get("demand_realization_seed"),
                "environment_block": row.get(
                    "environment_block", f"sumo-seed-{row['sumo_seed']}"
                ),
            },
        )
        if existing["demand_scale"] != row["demand_scale"]:
            raise ValueError(f"cell {row['cell']} has inconsistent demand scale")
        if existing["environment_block"] != row.get(
            "environment_block", f"sumo-seed-{row['sumo_seed']}"
        ):
            raise ValueError(
                f"cell {row['cell']} has inconsistent environment blocks"
            )
        if existing["demand_realization_seed"] != row.get(
            "demand_realization_seed"
        ):
            raise ValueError(
                f"cell {row['cell']} has inconsistent demand-realization seeds"
            )

    grouped: dict[str, dict[str, dict[tuple[str, int], dict]]] = {}
    seeds: dict[str, dict[str, int | None]] = {}
    for row in rows:
        algorithm = row["algorithm"]
        replicate = row["replicate"]
        grouped.setdefault(algorithm, {}).setdefault(replicate, {})[
            row["cell"]
        ] = row["metrics"]
        seeds.setdefault(algorithm, {})[replicate] = row["training_seed"]

    cubes = {}
    expected_cells = set(cells)
    for algorithm, replicates in sorted(grouped.items()):
        replicate_names = sorted(replicates)
        for replicate in replicate_names:
            present = set(replicates[replicate])
            if present != expected_cells:
                missing = sorted(expected_cells - present)
                extra = sorted(present - expected_cells)
                raise ValueError(
                    f"incomplete cells for {algorithm}/{replicate}: "
                    f"missing={missing}, extra={extra}"
                )
        metric_arrays = {
            metric: np.asarray(
                [
                    [replicates[replicate][cell][metric] for cell in cells]
                    for replicate in replicate_names
                ],
                dtype=np.float64,
            )
            for metric in metric_names
        }
        cubes[algorithm] = {
            "replicates": replicate_names,
            "training_seeds": [seeds[algorithm][name] for name in replicate_names],
            "metrics": metric_arrays,
        }
    return cubes, cells, cell_metadata


def _environment_block_clusters(
    cells: list[tuple[str, int]],
    cell_metadata: dict,
) -> tuple[list[str], list[np.ndarray]]:
    """Keep all demand conditions from one independent environment block together."""

    by_block: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        block = str(cell_metadata[cell]["environment_block"])
        by_block.setdefault(block, []).append(index)
    blocks = sorted(by_block)
    if not blocks:
        raise ValueError("evaluation matrix has no environment blocks")
    expected_scales = None
    declared_realization_seeds: set[int] = set()
    for block in blocks:
        indices = by_block[block]
        scales = tuple(
            sorted(cell_metadata[cells[index]]["demand_scale"] for index in indices)
        )
        if len(set(scales)) != len(scales):
            raise ValueError(
                f"environment block {block!r} repeats a demand-scale condition"
            )
        if expected_scales is None:
            expected_scales = scales
        elif scales != expected_scales:
            raise ValueError(
                f"environment block {block!r} has demand scales {scales}; "
                f"expected {expected_scales}"
            )
        sumo_seeds = {
            cell_metadata[cells[index]]["sumo_seed"] for index in indices
        }
        if len(sumo_seeds) != 1:
            raise ValueError(
                f"environment block {block!r} mixes SUMO seeds {sorted(sumo_seeds)}"
            )
        realization_seeds = {
            cell_metadata[cells[index]].get("demand_realization_seed")
            for index in indices
        }
        if realization_seeds != {None}:
            if None in realization_seeds or len(realization_seeds) != 1:
                raise ValueError(
                    f"environment block {block!r} mixes demand-realization seeds "
                    f"{sorted(str(seed) for seed in realization_seeds)}"
                )
            realization_seed = next(iter(realization_seeds))
            if realization_seed in declared_realization_seeds:
                raise ValueError(
                    f"demand-realization seed {realization_seed} is reused across "
                    "environment blocks"
                )
            declared_realization_seeds.add(realization_seed)
    return blocks, [
        np.asarray(by_block[block], dtype=np.int64) for block in blocks
    ]


def _resampled_cell_indices(
    clusters: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    selected = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[index] for index in selected])


def _hierarchical_bootstrap(
    values: np.ndarray,
    environment_clusters: list[np.ndarray],
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_training, n_cells = values.shape
    if sum(len(cluster) for cluster in environment_clusters) != n_cells:
        raise ValueError("environment clusters do not cover the metric cube")
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        training_indices = rng.integers(0, n_training, size=n_training)
        cell_indices = _resampled_cell_indices(environment_clusters, rng)
        estimates[index] = values[np.ix_(training_indices, cell_indices)].mean()
    return estimates


def _paired_hierarchical_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    environment_clusters: list[np.ndarray],
    candidate_training_seeds: list[int | None],
    reference_training_seeds: list[int | None],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if candidate.shape[1] != reference.shape[1]:
        raise ValueError("paired algorithms do not share environment cells")
    paired_training = (
        len(candidate_training_seeds) == len(reference_training_seeds)
        and len(candidate_training_seeds) > 1
        and all(seed is not None for seed in candidate_training_seeds)
        and set(candidate_training_seeds) == set(reference_training_seeds)
        and len(set(candidate_training_seeds)) == len(candidate_training_seeds)
    )
    if paired_training:
        candidate_by_seed = {
            seed: index for index, seed in enumerate(candidate_training_seeds)
        }
        reference_by_seed = {
            seed: index for index, seed in enumerate(reference_training_seeds)
        }
        common_seeds = sorted(candidate_by_seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        cell_indices = _resampled_cell_indices(environment_clusters, rng)
        if paired_training:
            sampled_seeds = rng.choice(
                common_seeds, size=len(common_seeds), replace=True
            )
            candidate_training = np.asarray(
                [candidate_by_seed[seed] for seed in sampled_seeds], dtype=np.int64
            )
            reference_training = np.asarray(
                [reference_by_seed[seed] for seed in sampled_seeds], dtype=np.int64
            )
        else:
            candidate_training = rng.integers(
                0, candidate.shape[0], size=candidate.shape[0]
            )
            reference_training = rng.integers(
                0, reference.shape[0], size=reference.shape[0]
            )
        candidate_mean = candidate[
            np.ix_(candidate_training, cell_indices)
        ].mean()
        reference_mean = reference[
            np.ix_(reference_training, cell_indices)
        ].mean()
        estimates[index] = reference_mean - candidate_mean
    return estimates, "paired" if paired_training else "independent"


def _sign_flip_pvalue(
    differences: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, str, int]:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    n = len(differences)
    if n <= 20:
        extreme = 0
        total = 2**n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            statistic = abs(float(np.mean(differences * np.asarray(signs))))
            extreme += statistic >= observed - 1e-15
        return extreme / total, "exact-sign-flip", total
    signs = rng.choice((-1.0, 1.0), size=(replicates, n))
    statistics = np.abs((signs * differences).mean(axis=1))
    pvalue = (1.0 + float(np.sum(statistics >= observed - 1e-15))) / (
        replicates + 1.0
    )
    return pvalue, "monte-carlo-sign-flip", replicates


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda name: pvalues[name])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def analyze(
    rows: list[dict],
    *,
    reference: str,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
    mdp_version: str = "v4",
) -> dict:
    if mdp_version not in {"v4", "v5", "v6", "v7"}:
        raise ValueError("mdp_version must be 'v4', 'v5', 'v6', or 'v7'")
    if bootstrap_replicates <= 0 or permutation_replicates <= 0:
        raise ValueError("resampling replicate counts must be positive")
    cubes, cells, cell_metadata = build_cubes(rows)
    environment_blocks, environment_clusters = _environment_block_clusters(
        cells, cell_metadata
    )
    evaluation_seeds = sorted(
        {metadata["sumo_seed"] for metadata in cell_metadata.values()}
    )
    if reference not in cubes:
        raise ValueError(f"reference algorithm {reference!r} is absent")
    rng = np.random.default_rng(seed)
    summaries = {}
    for algorithm, cube in sorted(cubes.items()):
        metrics = {}
        for metric, values in cube["metrics"].items():
            bootstrap = _hierarchical_bootstrap(
                values, environment_clusters, bootstrap_replicates, rng
            )
            cell_means = values.mean(axis=0)
            training_means = values.mean(axis=1)
            environment_block_means = np.asarray(
                [values[:, cluster].mean() for cluster in environment_clusters]
            )
            metrics[metric] = {
                "mean": float(values.mean()),
                "ci95": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
                "environment_cell_sd": (
                    float(cell_means.std(ddof=1)) if len(cell_means) > 1 else 0.0
                ),
                "environment_block_sd": (
                    float(environment_block_means.std(ddof=1))
                    if len(environment_block_means) > 1
                    else 0.0
                ),
                "training_seed_sd": (
                    float(training_means.std(ddof=1))
                    if len(training_means) > 1
                    else 0.0
                ),
            }
        primary = cube["metrics"][PRIMARY_METRIC]
        by_demand = {}
        for demand_scale in sorted(
            {metadata["demand_scale"] for metadata in cell_metadata.values()}
        ):
            indices = [
                index
                for index, cell in enumerate(cells)
                if cell_metadata[cell]["demand_scale"] == demand_scale
            ]
            selected = primary[:, indices]
            by_demand[str(demand_scale)] = {
                "mean": float(selected.mean()),
                "environment_cells": len(indices),
            }
        summaries[algorithm] = {
            "training_replicates": len(cube["replicates"]),
            "training_seeds": cube["training_seeds"],
            "environment_cells": len(cells),
            "environment_blocks": len(environment_blocks),
            "evaluation_seeds": len(evaluation_seeds),
            "runs": int(primary.size),
            "metrics": metrics,
            "primary_by_demand_scale": by_demand,
        }

    comparisons = {}
    raw_pvalues = {}
    reference_values = cubes[reference]["metrics"][PRIMARY_METRIC]
    for algorithm, cube in sorted(cubes.items()):
        if algorithm == reference:
            continue
        candidate = cube["metrics"][PRIMARY_METRIC]
        differences = np.asarray(
            [
                reference_values[:, cluster].mean()
                - candidate[:, cluster].mean()
                for cluster in environment_clusters
            ],
            dtype=np.float64,
        )
        bootstrap, training_resampling = _paired_hierarchical_bootstrap(
            candidate,
            reference_values,
            environment_clusters,
            cube["training_seeds"],
            cubes[reference]["training_seeds"],
            bootstrap_replicates,
            rng,
        )
        pvalue, test_name, test_replicates = _sign_flip_pvalue(
            differences, permutation_replicates, rng
        )
        raw_pvalues[algorithm] = pvalue
        reference_mean = float(reference_values.mean())
        improvement = float(reference_values.mean() - candidate.mean())
        comparisons[algorithm] = {
            "reference": reference,
            "improvement_seconds_per_passenger": improvement,
            "relative_improvement_percent": (
                100.0 * improvement / reference_mean if reference_mean else None
            ),
            "ci95": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
            "bootstrap_probability_improvement": float(np.mean(bootstrap > 0.0)),
            "training_seed_resampling": training_resampling,
            "paired_environment_block_win_rate": float(np.mean(differences > 0.0)),
            "paired_environment_block_mean": float(differences.mean()),
            "paired_environment_block_sd": (
                float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
            ),
            "pvalue_two_sided": pvalue,
            "test": f"{test_name}-clustered-by-environment-block",
            "test_replicates": test_replicates,
            "pvalue_scope": (
                "paired independent environment-block aggregates after averaging "
                "demand conditions and training replicates"
            ),
        }
    adjusted = _holm_adjust(raw_pvalues)
    for algorithm, value in adjusted.items():
        comparisons[algorithm]["pvalue_holm"] = value

    return {
        "schema_version": f"transit-evaluation-analysis-{mdp_version}",
        "mdp_version": mdp_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "primary_direction": "lower-is-better",
        "reference": reference,
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
        "resampling_seed": seed,
        "evaluation_seeds": evaluation_seeds,
        "environment_blocks": environment_blocks,
        "environment_resampling_unit": (
            "explicit paired environment block with all demand conditions retained"
        ),
        "environment_cells": [
            {
                "scenario": cell[0],
                "sumo_seed": cell[1],
                "environment_block": cell_metadata[cell]["environment_block"],
                "demand_scale": cell_metadata[cell]["demand_scale"],
                "demand_realization_seed": cell_metadata[cell].get(
                    "demand_realization_seed"
                ),
            }
            for cell in cells
        ],
        "algorithms": summaries,
        "comparisons_to_reference": comparisons,
    }


def render_markdown(analysis: dict) -> str:
    algorithms = analysis["algorithms"]
    comparisons = analysis["comparisons_to_reference"]
    reference = analysis["reference"]
    ordered = sorted(
        algorithms,
        key=lambda name: algorithms[name]["metrics"][PRIMARY_METRIC]["mean"],
    )
    lines = [
        f"# Transit {analysis.get('mdp_version', 'v4')} evaluation",
        "",
        (
            "Primary endpoint: accrued generalized passenger seconds per departed "
            "passenger (lower is better). Confidence intervals use a hierarchical "
            "bootstrap over training replicates and paired environment blocks; all "
            "demand conditions in a block remain in the same resampling cluster."
        ),
        "",
        "| Algorithm | Train reps | Runs | Welfare s/pax (95% CI) | Improvement vs ref | Holm p | Wait s/pax | In-vehicle s/pax | Headway NAE | Hold s | Completion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ordered:
        summary = algorithms[algorithm]
        metrics = summary["metrics"]
        primary = metrics[PRIMARY_METRIC]
        if algorithm == reference:
            improvement = "reference"
            holm = "-"
        else:
            comparison = comparisons[algorithm]
            improvement = f"{comparison['improvement_seconds_per_passenger']:+.2f}"
            holm = f"{comparison['pvalue_holm']:.4g}"
        lines.append(
            "| "
            + " | ".join(
                (
                    algorithm,
                    str(summary["training_replicates"]),
                    str(summary["runs"]),
                    f"{primary['mean']:.2f} [{primary['ci95'][0]:.2f}, {primary['ci95'][1]:.2f}]",
                    improvement,
                    holm,
                    f"{metrics['wait_seconds_per_passenger']['mean']:.2f}",
                    f"{metrics['in_vehicle_seconds_per_passenger']['mean']:.2f}",
                    f"{metrics['headway_normalized_absolute_error']['mean']:.4f}",
                    f"{metrics['mean_hold_seconds']['mean']:.2f}",
                    f"{metrics['passenger_completion_rate']['mean']:.4f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Reference algorithm: `{reference}`.",
            "Positive improvement means fewer generalized passenger seconds than the reference.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows, evidence = load_rows(args.matrix_root)
    analysis = analyze(
        rows,
        reference=args.reference,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        seed=args.seed,
        mdp_version=evidence["mdp_version"],
    )
    analysis["evidence"] = evidence
    analysis["analysis_source_sha256"] = sha256(Path(__file__).resolve())
    args.output_json = args.output_json.expanduser().resolve()
    args.output_markdown = args.output_markdown.expanduser().resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = args.output_json.with_suffix(args.output_json.suffix + ".partial")
    markdown_temporary = args.output_markdown.with_suffix(
        args.output_markdown.suffix + ".partial"
    )
    json_temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    markdown_temporary.write_text(render_markdown(analysis))
    os.replace(json_temporary, args.output_json)
    os.replace(markdown_temporary, args.output_markdown)
    print(args.output_json)
    print(args.output_markdown)


if __name__ == "__main__":
    main()
