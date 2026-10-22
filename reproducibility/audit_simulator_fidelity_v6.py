#!/usr/bin/env python3
"""Audit the calibrated low-fidelity simulator against zero-hold SUMO blocks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
from bus_h2o.passenger_time import aggregate_passenger_time_snapshots
from bus_h2o.sumo_provenance import sha256
from SimpleSAC.train_transit_v3 import (
    collect_runtime_fingerprint,
    simulator_input_hashes,
    source_hashes,
)
from SimpleSAC.transit_v3.evaluation import numeric_summary


SCHEMA_VERSION = "transit-simulator-fidelity-audit-v6-formal-v6"
DEFAULT_DEMAND_SCALES = (0.75, 1.0, 1.25)
DEFAULT_SIMULATOR_SEEDS = tuple(range(9901, 9911))
FIDELITY_THRESHOLDS = {
    "demand_relative_error": 0.03,
    "decision_relative_gap": 0.05,
    "generalized_welfare_relative_gap": 0.15,
    "wait_time_relative_gap": 0.15,
    "in_vehicle_time_relative_gap": 0.10,
    "completion_rate_absolute_gap": 0.03,
    "trip_duration_relative_gap": 0.05,
}


def _relative_gap(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("fidelity comparison values must be finite and positive")
    return (value - reference) / reference


def _active_agents(state: dict) -> list[object]:
    return sorted(
        (agent for agent, observations in state.items() if observations),
        key=lambda agent: (str(agent[0]), int(agent[1])),
    )


def run_zero_hold_episode(
    sim_env: Path,
    *,
    seed: int,
    episode_seconds: float,
    demand_scale: float,
) -> dict[str, Any]:
    """Run one fixed-horizon simulator episode and reconcile both ledgers."""

    random.seed(seed)
    np.random.seed(seed)
    env = AllLinesSimEnv(
        str(sim_env),
        episode_seconds=episode_seconds,
        demand_scale=demand_scale,
        observation_schema="v6",
    )
    env.reset()
    started = time.perf_counter()
    decisions = 0
    batches = 0
    maximum_batch = 0
    if not _active_agents(env.state):
        env.step_to_event({})
    while not env.done:
        active = _active_agents(env.state)
        if active:
            decisions += len(active)
            batches += 1
            maximum_batch = max(maximum_batch, len(active))
            env.step_to_event({agent: [0.0, 1.0] for agent in active})
        else:
            env.step_to_event({})

    ledger = aggregate_passenger_time_snapshots(
        line.passenger_time_ledger.snapshot() for line in env.line_map.values()
    )
    passengers = [
        passenger
        for line in env.line_map.values()
        for passenger in line.last_episode_passengers
    ]
    service_rows = [
        row
        for line in env.line_map.values()
        for row in line.last_episode_bus_summaries
    ]
    waiting = sum(passenger.boarding_time is None for passenger in passengers)
    onboard = sum(
        passenger.boarding_time is not None and passenger.arrive_time is None
        for passenger in passengers
    )
    completed = sum(passenger.arrive_time is not None for passenger in passengers)
    journeys = env.last_episode_journey_summary
    if not isinstance(journeys, dict) or journeys.get("unit") != (
        "causally ordered source-person journeys"
    ):
        raise RuntimeError("fidelity audit requires causal source-person journeys")
    completed_service = [row for row in service_rows if row["completed"]]
    trip_durations = [
        float(row["terminal_arrival_seconds"] - row["launch_time_seconds"])
        for row in completed_service
        if row["terminal_arrival_seconds"] is not None
    ]
    count_errors = {
        "waiting": int(ledger.waiting_passengers - waiting),
        "in_vehicle": int(ledger.in_vehicle_passengers - onboard),
    }
    generalized = (
        2.0 * ledger.accrued_wait_seconds + ledger.accrued_in_vehicle_seconds
    )
    return {
        "seed": int(seed),
        "demand_scale": float(demand_scale),
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "episode": {
            "configured_seconds": float(episode_seconds),
            "observed_end_time_seconds": float(env.current_time),
            "ledger_end_time_seconds": float(ledger.time_seconds),
            "control_decisions": int(decisions),
            "simultaneous_batches": int(batches),
            "maximum_simultaneous_decisions": int(maximum_batch),
        },
        "passengers": {
            "unit": journeys["unit"],
            "generated_journeys": int(journeys["generated_journeys"]),
            "completed_journeys": int(journeys["completed_journeys"]),
            "unfinished_journeys": int(journeys["unfinished_journeys"]),
            "completion_rate": float(journeys["completion_rate"]),
            "journey_state_balance_error": int(
                journeys["journey_state_balance_error"]
            ),
            "generated_ride_legs": int(len(passengers)),
            "completed_ride_legs": int(completed),
            "unfinished_ride_legs": int(len(passengers) - completed),
            "created_transfer_ride_legs": int(
                journeys["created_transfer_ride_legs"]
            ),
            "state_counts_at_horizon": {
                "waiting": int(waiting),
                "in_vehicle": int(onboard),
                "completed_ride_legs": int(completed),
                "between_transfer_legs": int(
                    journeys["state_counts_at_horizon"]["between_transfer_legs"]
                ),
            },
            "ledger_count_errors": count_errors,
            "accrued_wait_seconds_total": float(ledger.accrued_wait_seconds),
            "accrued_in_vehicle_seconds_total": float(
                ledger.accrued_in_vehicle_seconds
            ),
            "accrued_generalized_passenger_seconds": float(generalized),
        },
        "service": {
            "departed_trips": int(len(service_rows)),
            "completed_trips": int(len(completed_service)),
            "completion_rate": (
                len(completed_service) / len(service_rows) if service_rows else None
            ),
            "completed_trip_duration_seconds": numeric_summary(trip_durations),
        },
    }


def load_zero_hold_sumo_summaries(shard_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(shard_root.glob("zero/*/*.npz.json")):
        summary = json.loads(path.read_text())
        shard = path.with_suffix("")
        if summary.get("schema_version") != "transit-smdp-v6-shard-v1":
            raise ValueError(f"unsupported SUMO shard summary: {path}")
        if summary.get("policy_name") != "zero":
            raise ValueError(f"nonzero policy found under zero directory: {path}")
        if not summary.get("training_eligible"):
            raise ValueError(f"diagnostic SUMO shard cannot calibrate fidelity: {path}")
        if not shard.is_file() or sha256(shard) != summary.get("shard_sha256"):
            raise ValueError(f"SUMO shard hash mismatch: {shard}")
        summary["_fidelity_source"] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
        summaries.append(summary)
    if len(summaries) < 2:
        raise ValueError("fidelity audit requires at least two zero-hold SUMO blocks")
    return summaries


def _sumo_metrics(summaries: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "departed_persons": [
            float(row["passengers"]["departed_passengers"]) for row in summaries
        ],
        "control_decisions": [
            float(row["episode"]["control_decisions"]) for row in summaries
        ],
        "generalized_seconds_per_person": [
            float(
                row["passenger_welfare"][
                    "accrued_generalized_seconds_per_departed_passenger"
                ]
            )
            for row in summaries
        ],
        "wait_seconds_per_person": [
            float(row["passengers"]["accrued_wait_seconds_per_departed_passenger"])
            for row in summaries
        ],
        "in_vehicle_seconds_per_person": [
            float(
                row["passengers"][
                    "accrued_in_vehicle_seconds_per_departed_passenger"
                ]
            )
            for row in summaries
        ],
        "completion_rate": [
            float(row["passengers"]["completion_rate"]) for row in summaries
        ],
        "trip_duration_seconds": [
            float(row["service"]["completed_trip_duration_seconds"]["mean"])
            for row in summaries
        ],
    }


def _bootstrap_gap_interval(
    simulator_values: list[float],
    reference_values: list[float] | None,
    *,
    constant_reference: float | None = None,
    absolute_gap: bool = False,
    samples: int = 10_000,
    seed: int,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    simulator = np.asarray(simulator_values, dtype=np.float64)
    if simulator.size == 0:
        raise ValueError("bootstrap simulator sample cannot be empty")
    if (reference_values is None) == (constant_reference is None):
        raise ValueError("bootstrap gap requires exactly one reference form")
    rng = np.random.default_rng(seed)
    simulator_means = simulator[
        rng.integers(0, simulator.size, size=(samples, simulator.size))
    ].mean(axis=1)
    if reference_values is not None:
        reference = np.asarray(reference_values, dtype=np.float64)
        if reference.size == 0:
            raise ValueError("bootstrap reference sample cannot be empty")
        reference_means = reference[
            rng.integers(0, reference.size, size=(samples, reference.size))
        ].mean(axis=1)
    else:
        reference_means = np.full(samples, float(constant_reference))
    if np.any(reference_means <= 0.0):
        raise ValueError("bootstrap references must be positive")
    gaps = (
        simulator_means - reference_means
        if absolute_gap
        else (simulator_means - reference_means) / reference_means
    )
    return [
        float(np.quantile(gaps, 0.025)),
        float(np.quantile(gaps, 0.975)),
    ]


def compare_fidelity(
    simulator_episodes: list[dict[str, Any]],
    sumo_summaries: list[dict[str, Any]],
    *,
    expected_source_persons: float,
    episode_seconds: float,
    thresholds: dict[str, float],
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Compare like-for-like source-person journey outcomes."""

    if expected_source_persons <= 0:
        raise ValueError("expected source-person demand must be positive")
    sumo = _sumo_metrics(sumo_summaries)
    sim: dict[str, list[float]] = {
        "generated_journeys": [],
        "created_ride_legs": [],
        "control_decisions": [],
        "generalized_seconds_per_person": [],
        "wait_seconds_per_person": [],
        "in_vehicle_seconds_per_person": [],
        "completion_rate": [],
        "trip_duration_seconds": [],
    }
    invariants = []
    for row in simulator_episodes:
        passengers = row["passengers"]
        generated = float(passengers["generated_journeys"])
        if generated <= 0.0:
            raise ValueError("simulator episode generated no source-person journeys")
        sim["generated_journeys"].append(generated)
        sim["created_ride_legs"].append(
            float(passengers["generated_ride_legs"])
        )
        sim["control_decisions"].append(float(row["episode"]["control_decisions"]))
        for source_name, target_name in (
            (
                "accrued_generalized_passenger_seconds",
                "generalized_seconds_per_person",
            ),
            ("accrued_wait_seconds_total", "wait_seconds_per_person"),
            (
                "accrued_in_vehicle_seconds_total",
                "in_vehicle_seconds_per_person",
            ),
        ):
            sim[target_name].append(float(passengers[source_name]) / generated)
        sim["completion_rate"].append(float(passengers["completion_rate"]))
        sim["trip_duration_seconds"].append(
            float(row["service"]["completed_trip_duration_seconds"]["mean"])
        )
        invariants.append(
            {
                "seed": row["seed"],
                "fixed_horizon": bool(
                    row["episode"]["observed_end_time_seconds"] == episode_seconds
                    and row["episode"]["ledger_end_time_seconds"] == episode_seconds
                ),
                "ledger_counts_reconciled": not any(
                    row["passengers"]["ledger_count_errors"].values()
                ),
                "journey_states_reconciled": bool(
                    row["passengers"]["journey_state_balance_error"] == 0
                ),
                "all_trips_completed": bool(
                    row["service"]["departed_trips"]
                    == row["service"]["completed_trips"]
                    == 389
                ),
            }
        )

    mean = lambda values: float(np.mean(np.asarray(values, dtype=np.float64)))
    gaps = {
        "demand_relative_error": _relative_gap(
            mean(sim["generated_journeys"]), mean(sumo["departed_persons"])
        ),
        "decision_relative_gap": _relative_gap(
            mean(sim["control_decisions"]), mean(sumo["control_decisions"])
        ),
        "generalized_welfare_relative_gap": _relative_gap(
            mean(sim["generalized_seconds_per_person"]),
            mean(sumo["generalized_seconds_per_person"]),
        ),
        "wait_time_relative_gap": _relative_gap(
            mean(sim["wait_seconds_per_person"]),
            mean(sumo["wait_seconds_per_person"]),
        ),
        "in_vehicle_time_relative_gap": _relative_gap(
            mean(sim["in_vehicle_seconds_per_person"]),
            mean(sumo["in_vehicle_seconds_per_person"]),
        ),
        "completion_rate_absolute_gap": mean(sim["completion_rate"])
        - mean(sumo["completion_rate"]),
        "trip_duration_relative_gap": _relative_gap(
            mean(sim["trip_duration_seconds"]), mean(sumo["trip_duration_seconds"])
        ),
    }
    confidence_intervals = {
        "demand_relative_error": _bootstrap_gap_interval(
            sim["generated_journeys"],
            sumo["departed_persons"],
            samples=bootstrap_samples,
            seed=6101,
        ),
        "decision_relative_gap": _bootstrap_gap_interval(
            sim["control_decisions"],
            sumo["control_decisions"],
            samples=bootstrap_samples,
            seed=6102,
        ),
        "generalized_welfare_relative_gap": _bootstrap_gap_interval(
            sim["generalized_seconds_per_person"],
            sumo["generalized_seconds_per_person"],
            samples=bootstrap_samples,
            seed=6103,
        ),
        "wait_time_relative_gap": _bootstrap_gap_interval(
            sim["wait_seconds_per_person"],
            sumo["wait_seconds_per_person"],
            samples=bootstrap_samples,
            seed=6104,
        ),
        "in_vehicle_time_relative_gap": _bootstrap_gap_interval(
            sim["in_vehicle_seconds_per_person"],
            sumo["in_vehicle_seconds_per_person"],
            samples=bootstrap_samples,
            seed=6105,
        ),
        "completion_rate_absolute_gap": _bootstrap_gap_interval(
            sim["completion_rate"],
            sumo["completion_rate"],
            absolute_gap=True,
            samples=bootstrap_samples,
            seed=6106,
        ),
        "trip_duration_relative_gap": _bootstrap_gap_interval(
            sim["trip_duration_seconds"],
            sumo["trip_duration_seconds"],
            samples=bootstrap_samples,
            seed=6107,
        ),
    }
    interval_checks = {
        name: max(abs(bound) for bound in interval) <= thresholds[name]
        for name, interval in confidence_intervals.items()
    }
    checks = {
        "all_invariants": all(
            item["fixed_horizon"]
            and item["ledger_counts_reconciled"]
            and item["journey_states_reconciled"]
            and item["all_trips_completed"]
            for item in invariants
        ),
        "demand": abs(gaps["demand_relative_error"])
        <= thresholds["demand_relative_error"],
        "decisions": abs(gaps["decision_relative_gap"])
        <= thresholds["decision_relative_gap"],
        "generalized_welfare": abs(gaps["generalized_welfare_relative_gap"])
        <= thresholds["generalized_welfare_relative_gap"],
        "wait_time": abs(gaps["wait_time_relative_gap"])
        <= thresholds["wait_time_relative_gap"],
        "in_vehicle_time": abs(gaps["in_vehicle_time_relative_gap"])
        <= thresholds["in_vehicle_time_relative_gap"],
        "completion_rate": abs(gaps["completion_rate_absolute_gap"])
        <= thresholds["completion_rate_absolute_gap"],
        "trip_duration": abs(gaps["trip_duration_relative_gap"])
        <= thresholds["trip_duration_relative_gap"],
        "all_gap_confidence_intervals": all(interval_checks.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "invariants": invariants,
        "gaps": gaps,
        "gap_bootstrap_95_ci": confidence_intervals,
        "gap_bootstrap_interval_checks": interval_checks,
        "bootstrap_samples": bootstrap_samples,
        "thresholds": thresholds,
        "simulator": {name: numeric_summary(values) for name, values in sim.items()},
        "sumo": {name: numeric_summary(values) for name, values in sumo.items()},
        "normalization": {
            "expected_source_persons": expected_source_persons,
            "formula": "simulator total / generated source-person journeys",
            "disclosure": (
                "sim-core and SUMO are both normalized by source-person journeys; "
                "later ride legs are created only after the preceding leg alights"
            ),
        },
    }


def _run_episode_task(task: tuple[str, int, float, float]) -> dict[str, Any]:
    sim_env, seed, episode_seconds, demand_scale = task
    return run_zero_hold_episode(
        Path(sim_env),
        seed=seed,
        episode_seconds=episode_seconds,
        demand_scale=demand_scale,
    )


def _scale_key(scale: float) -> str:
    return format(float(scale), ".6g")


def _sumo_source_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": row["_fidelity_source"],
        "shard_sha256": row["shard_sha256"],
        "sumo_seed": row["sumo_seed"],
        "demand_scale": float(row["scenario"]["demand_scale"]),
        "scaled_total_persons": int(row["scenario"]["scaled_total_persons"]),
        "scenario_manifest_sha256": row["scenario"]["manifest_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sumo-shard-root",
        type=Path,
        default=ROOT / "bus_h2o" / "datasets_v6" / "shards_formal_v1",
    )
    parser.add_argument(
        "--sim-env",
        type=Path,
        default=ROOT / "bus_h2o" / "calibrated_env",
    )
    parser.add_argument(
        "--sim-seeds", type=int, nargs="+", default=list(DEFAULT_SIMULATOR_SEEDS)
    )
    parser.add_argument(
        "--demand-scales", type=float, nargs="+", default=list(DEFAULT_DEMAND_SCALES)
    )
    parser.add_argument("--episode-seconds", type=float, default=24_000.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--study-stage", choices=("pilot", "formal"), default="formal"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sumo_shard_root = args.sumo_shard_root.expanduser().resolve()
    args.sim_env = args.sim_env.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite fidelity audit {args.output}")
    if len(set(args.sim_seeds)) != len(args.sim_seeds):
        raise ValueError("simulator seeds must be unique")
    if len(set(args.demand_scales)) != len(args.demand_scales):
        raise ValueError("demand scales must be unique")
    if args.episode_seconds <= 0.0 or any(
        scale <= 0.0 for scale in args.demand_scales
    ):
        raise ValueError("episode seconds and demand scales must be positive")
    if args.workers <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("workers and bootstrap samples must be positive")

    calibration_path = args.sim_env / "simulator_demand_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("schema_version") != (
        "transit-simulator-demand-calibration-v3"
    ) or not calibration.get("journey_execution", {}).get(
        "causal_transfer_chain"
    ):
        raise ValueError("formal fidelity audit requires causal demand calibration v3")
    base_source_persons = float(calibration["source_passengers"])
    sumo_summaries = load_zero_hold_sumo_summaries(args.sumo_shard_root)
    sumo_by_scale = {
        scale: [
            row
            for row in sumo_summaries
            if np.isclose(float(row["scenario"]["demand_scale"]), scale)
        ]
        for scale in args.demand_scales
    }
    insufficient = {
        _scale_key(scale): len(rows)
        for scale, rows in sumo_by_scale.items()
        if len(rows) < 2
    }
    if insufficient:
        raise ValueError(
            f"fidelity strata require at least two SUMO blocks: {insufficient}"
        )
    tasks = [
        (str(args.sim_env), seed, args.episode_seconds, scale)
        for scale in args.demand_scales
        for seed in args.sim_seeds
    ]
    if args.workers == 1:
        episodes = [_run_episode_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            episodes = list(executor.map(_run_episode_task, tasks))
    strata = {}
    for scale in args.demand_scales:
        scale_episodes = [
            row
            for row in episodes
            if np.isclose(float(row["demand_scale"]), scale)
        ]
        scale_sumo = sumo_by_scale[scale]
        comparison = compare_fidelity(
            scale_episodes,
            scale_sumo,
            expected_source_persons=base_source_persons * scale,
            episode_seconds=args.episode_seconds,
            thresholds=FIDELITY_THRESHOLDS,
            bootstrap_samples=args.bootstrap_samples,
        )
        strata[_scale_key(scale)] = {
            "demand_scale": float(scale),
            "comparison": comparison,
            "simulator_seeds": [int(row["seed"]) for row in scale_episodes],
            "sumo_sources": [
                _sumo_source_descriptor(row) for row in scale_sumo
            ],
            "sumo_scaled_person_counts": numeric_summary(
                [float(row["scenario"]["scaled_total_persons"]) for row in scale_sumo]
            ),
        }
    source = source_hashes("v6")
    source[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(
        Path(__file__).resolve()
    )
    import torch

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": args.study_stage,
        "passed": all(
            stratum["comparison"]["passed"] for stratum in strata.values()
        ),
        "design": {
            "demand_scales": [float(value) for value in args.demand_scales],
            "simulator_seeds": [int(value) for value in args.sim_seeds],
            "sumo_blocks_per_scale": {
                _scale_key(scale): len(rows)
                for scale, rows in sumo_by_scale.items()
            },
            "episode_seconds": float(args.episode_seconds),
            "bootstrap_samples": int(args.bootstrap_samples),
            "thresholds": FIDELITY_THRESHOLDS,
            "zero_hold_policy": [0.0, 1.0],
        },
        "strata": strata,
        "simulator_episodes": episodes,
        "sumo_sources": [
            _sumo_source_descriptor(row)
            for row in sumo_summaries
            if any(
                np.isclose(float(row["scenario"]["demand_scale"]), scale)
                for scale in args.demand_scales
            )
        ],
        "simulator_inputs": simulator_input_hashes(args.sim_env),
        "demand_calibration": {
            "path": str(calibration_path),
            "sha256": sha256(calibration_path),
        },
        "source_sha256": source,
        "runtime_fingerprint": collect_runtime_fingerprint(torch, "cpu"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "strata": {
                    key: {
                        "passed": value["comparison"]["passed"],
                        "checks": value["comparison"]["checks"],
                        "gaps": value["comparison"]["gaps"],
                        "gap_bootstrap_95_ci": value["comparison"][
                            "gap_bootstrap_95_ci"
                        ],
                    }
                    for key, value in strata.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
