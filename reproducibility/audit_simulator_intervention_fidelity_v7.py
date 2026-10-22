#!/usr/bin/env python3
"""Audit low-fidelity response to registered holding interventions against SUMO."""

from __future__ import annotations

import argparse
import hashlib
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
from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4
from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7, policy_action_to_hold_seconds_v7
from bus_h2o.passenger_time import aggregate_passenger_time_snapshots
from bus_h2o.sumo_provenance import sha256
from reproducibility.compose_collection_manifests_v7 import COMPOSITION_SCHEMA
from reproducibility.relocatable_paths_v7 import (
    ArtifactPathRemap,
    resolve_evidence_path,
)
from reproducibility.run_collection_matrix_v7 import MATRIX_SCHEMA, POLICIES
from SimpleSAC.train_transit_v3 import simulator_input_hashes
from SimpleSAC.transit_v3.evaluation import numeric_summary
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4,
    hurdle_random_actions_v4,
    myopic_welfare_hold_seconds_v4,
)


SCHEMA_VERSION = "transit-simulator-intervention-fidelity-v7-formal-v1"
SHARD_SCHEMA = "transit-smdp-v7-shard-v1"
DEMAND_SCALES = (0.75, 1.0, 1.25)
SIMULATOR_SEEDS = tuple(range(9921, 9931))
BOOTSTRAP_SAMPLES = 10_000
EPISODE_SECONDS = 24_000.0
THRESHOLDS = {
    "mean_hold_seconds_relative_gap": 0.35,
    "exact_zero_rate_absolute_gap": 0.10,
    "control_decision_response_normalized_gap": 0.05,
    "generalized_welfare_response_normalized_gap": 0.10,
    "wait_time_response_normalized_gap": 0.10,
    "in_vehicle_time_response_normalized_gap": 0.10,
    "completion_rate_response_absolute_gap": 0.03,
    "trip_duration_response_normalized_gap": 0.05,
}


def intervention_source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "compose_collection_manifests_v7.py",
        ROOT / "reproducibility" / "run_collection_matrix_v7.py",
        ROOT / "reproducibility" / "relocatable_paths_v7.py",
        ROOT / "SimpleSAC" / "transit_v3" / "evaluation.py",
        ROOT / "SimpleSAC" / "transit_v4" / "policies.py",
        ROOT / "bus_h2o" / "envs" / "all_lines_sim_env.py",
        ROOT / "bus_h2o" / "mdp_contract_v4.py",
        ROOT / "bus_h2o" / "mdp_contract_v5.py",
        ROOT / "bus_h2o" / "mdp_contract_v7.py",
        ROOT / "bus_h2o" / "passenger_time.py",
        ROOT / "bus_h2o" / "sumo_provenance.py",
        *sorted((ROOT / "bus_h2o" / "sim_core").glob("*.py")),
    ]
    resolved = sorted({path.expanduser().resolve(strict=True) for path in paths})
    return {
        path.relative_to(ROOT).as_posix(): sha256(path) for path in resolved
    }


METRIC_SPECS = {
    "control_decisions": (
        "control_decision_response_normalized_gap",
        "baseline-normalized-response",
    ),
    "generalized_seconds_per_person": (
        "generalized_welfare_response_normalized_gap",
        "baseline-normalized-response",
    ),
    "wait_seconds_per_person": (
        "wait_time_response_normalized_gap",
        "baseline-normalized-response",
    ),
    "in_vehicle_seconds_per_person": (
        "in_vehicle_time_response_normalized_gap",
        "baseline-normalized-response",
    ),
    "completion_rate": (
        "completion_rate_response_absolute_gap",
        "absolute-response",
    ),
    "trip_duration_seconds": (
        "trip_duration_response_normalized_gap",
        "baseline-normalized-response",
    ),
}


def _active_agents(state: dict) -> list[object]:
    return sorted(
        (agent for agent, observations in state.items() if observations),
        key=lambda agent: (str(agent[0]), int(agent[1])),
    )


def _raw_observation(value: Any) -> np.ndarray:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError("low-fidelity v7 policy observation is invalid")
    return result


def _policy_seed(policy_name: str, simulator_seed: int, demand_scale: float) -> int:
    payload = f"{policy_name}:{simulator_seed}:{demand_scale:.6f}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31 - 1
    )


def _hold_seconds(
    policy: dict[str, Any], raw: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    config = DEFAULT_CONFIG_V7
    policy_type = policy["policy"]
    if policy_type == "zero":
        return np.zeros(len(raw), dtype=np.float32)
    if policy_type == "random-hurdle":
        actions = hurdle_random_actions_v4(
            len(raw),
            rng,
            zero_probability=float(policy["random_zero_probability"]),
            beta_alpha=float(policy["random_beta_alpha"]),
            beta_beta=float(policy["random_beta_beta"]),
            config=config,
        )
        return np.asarray(
            policy_action_to_hold_seconds_v7(actions[:, 0], config),
            dtype=np.float32,
        )
    if policy_type == "headway-balance":
        return np.asarray(
            headway_balance_hold_seconds_v4(
                raw[:, :RAW_EVENT_STATE_DIM_V4],
                float(policy["headway_gain"]),
                config,
            ),
            dtype=np.float32,
        )
    if policy_type == "myopic-welfare":
        return np.asarray(
            myopic_welfare_hold_seconds_v4(
                raw[:, :RAW_EVENT_STATE_DIM_V4], config
            ),
            dtype=np.float32,
        )
    raise ValueError(f"unknown registered behavior policy {policy_type!r}")


def run_behavior_episode(
    sim_env: Path,
    *,
    policy: dict[str, Any],
    simulator_seed: int,
    demand_scale: float,
) -> dict[str, Any]:
    random.seed(simulator_seed)
    np.random.seed(simulator_seed)
    rng = np.random.default_rng(
        _policy_seed(policy["policy_name"], simulator_seed, demand_scale)
    )
    env = AllLinesSimEnv(
        str(sim_env),
        episode_seconds=EPISODE_SECONDS,
        demand_scale=demand_scale,
        observation_schema="v7",
        observation_speed_seed=simulator_seed,
    )
    env.reset()
    started = time.perf_counter()
    holds: list[float] = []
    batches = 0
    maximum_batch = 0
    if not _active_agents(env.state):
        env.step_to_event({})
    while not env.done:
        active = _active_agents(env.state)
        if not active:
            env.step_to_event({})
            continue
        raw = np.stack([_raw_observation(env.state[agent][-1]) for agent in active])
        batch_holds = _hold_seconds(policy, raw, rng)
        if batch_holds.shape != (len(active),) or np.any(
            (batch_holds < 0.0) | (batch_holds > DEFAULT_CONFIG_V7.max_hold_seconds)
        ):
            raise ValueError("registered low-fidelity policy emitted invalid holds")
        holds.extend(float(value) for value in batch_holds)
        batches += 1
        maximum_batch = max(maximum_batch, len(active))
        env.step_to_event(
            {
                agent: [float(hold), 1.0]
                for agent, hold in zip(active, batch_holds)
            }
        )

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
    journeys = env.last_episode_journey_summary
    completed_service = [row for row in service_rows if row["completed"]]
    trip_durations = [
        float(row["terminal_arrival_seconds"] - row["launch_time_seconds"])
        for row in completed_service
    ]
    generated = int(journeys["generated_journeys"])
    if generated <= 0:
        raise RuntimeError("low-fidelity intervention episode generated no journeys")
    hold_array = np.asarray(holds, dtype=np.float64)
    generalized = (
        DEFAULT_CONFIG_V7.waiting_time_value * ledger.accrued_wait_seconds
        + DEFAULT_CONFIG_V7.in_vehicle_time_value
        * ledger.accrued_in_vehicle_seconds
    )
    invariants = {
        "fixed_horizon": bool(
            env.current_time == EPISODE_SECONDS
            and ledger.time_seconds == EPISODE_SECONDS
        ),
        "ledger_counts_reconciled": bool(
            ledger.waiting_passengers == waiting
            and ledger.in_vehicle_passengers == onboard
        ),
        "journey_states_reconciled": bool(
            journeys["journey_state_balance_error"] == 0
        ),
        "all_trips_completed": bool(
            len(service_rows) == len(completed_service) == 389
        ),
    }
    return {
        "policy_name": policy["policy_name"],
        "simulator_seed": int(simulator_seed),
        "demand_scale": float(demand_scale),
        "observation_speed_profile": env.current_observation_speed_profile,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "invariants": invariants,
        "metrics": {
            "mean_hold_seconds": float(hold_array.mean()),
            "exact_zero_rate": float(np.mean(hold_array == 0.0)),
            "control_decisions": float(len(hold_array)),
            "generalized_seconds_per_person": float(generalized / generated),
            "wait_seconds_per_person": float(
                ledger.accrued_wait_seconds / generated
            ),
            "in_vehicle_seconds_per_person": float(
                ledger.accrued_in_vehicle_seconds / generated
            ),
            "completion_rate": float(journeys["completion_rate"]),
            "trip_duration_seconds": float(
                np.mean(trip_durations, dtype=np.float64)
            ),
        },
        "diagnostics": {
            "generated_journeys": generated,
            "generated_ride_legs": len(passengers),
            "simultaneous_batches": batches,
            "maximum_simultaneous_decisions": maximum_batch,
            "hold_seconds": numeric_summary(hold_array),
        },
    }


def _sumo_metrics(summary: dict[str, Any]) -> dict[str, float]:
    passengers = summary["passengers"]
    return {
        "mean_hold_seconds": float(summary["actions"]["mean_hold_seconds"]),
        "exact_zero_rate": float(summary["actions"]["exact_zero_rate"]),
        "control_decisions": float(summary["episode"]["control_decisions"]),
        "generalized_seconds_per_person": float(
            summary["passenger_welfare"][
                "accrued_generalized_seconds_per_departed_passenger"
            ]
        ),
        "wait_seconds_per_person": float(
            passengers["accrued_wait_seconds_per_departed_passenger"]
        ),
        "in_vehicle_seconds_per_person": float(
            passengers["accrued_in_vehicle_seconds_per_departed_passenger"]
        ),
        "completion_rate": float(passengers["completion_rate"]),
        "trip_duration_seconds": float(
            summary["service"]["completed_trip_duration_seconds"]["mean"]
        ),
    }


def load_sumo_rows(
    matrix_manifest: Path,
    *,
    path_remap: ArtifactPathRemap | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    matrix = json.loads(matrix_manifest.read_text())
    expected_policies = {policy["policy_name"] for policy in POLICIES}
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("intervention fidelity requires a v7 matrix-v2 manifest")
    if matrix.get("study_stage") != "formal":
        raise ValueError("intervention fidelity requires formal SUMO data")
    if set(matrix.get("behavior_mixture", {}).get("selected_policy_names", [])) != (
        expected_policies
    ):
        raise ValueError("intervention fidelity requires all registered policies")
    if (
        int(matrix.get("independent_realization_blocks", -1)) != 10
        or int(matrix.get("environment_cells", -1)) != 30
        or {
            float(value)
            for value in matrix.get("environment_design", {}).get(
                "demand_scales", []
            )
        }
        != set(DEMAND_SCALES)
    ):
        raise ValueError("intervention fidelity matrix design is incomplete")
    composition = matrix.get("composition", {})
    components = composition.get("components", [])
    if (
        composition.get("schema_version") != COMPOSITION_SCHEMA
        or len(components) != 2
    ):
        raise ValueError("intervention fidelity requires two frozen components")
    for descriptor in components:
        component = resolve_evidence_path(descriptor["path"], path_remap)
        audit_path = resolve_evidence_path(descriptor["audit_path"], path_remap)
        if (
            not component.is_file()
            or not audit_path.is_file()
            or sha256(component) != descriptor.get("sha256")
            or sha256(audit_path) != descriptor.get("audit_sha256")
        ):
            raise ValueError("intervention component provenance changed")
        audit = json.loads(audit_path.read_text())
        frozen = audit.get("frozen_manifest", {})
        if (
            audit.get("passed") is not True
            or resolve_evidence_path(frozen.get("path", ""), path_remap)
            != component
            or frozen.get("sha256") != descriptor.get("sha256")
        ):
            raise ValueError("intervention component freeze audit is invalid")
    jobs = matrix.get("jobs", [])
    expected_cells = 10 * len(DEMAND_SCALES) * len(POLICIES)
    if len(jobs) != expected_cells:
        raise ValueError(
            f"formal intervention matrix has {len(jobs)} jobs; expected {expected_cells}"
        )

    rows = []
    cells = set()
    for job in jobs:
        block = int(job["realization_block_id"])
        scale = float(job["scenario"]["demand_scale"])
        policy_name = job["policy"]["policy_name"]
        cell = (block, scale, policy_name)
        if cell in cells:
            raise ValueError(f"duplicate intervention SUMO cell {cell}")
        cells.add(cell)
        shard = resolve_evidence_path(job["output"], path_remap)
        summary_path = shard.with_suffix(shard.suffix + ".json")
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("schema_version") != SHARD_SCHEMA
            or summary.get("job_signature") != job.get("signature")
            or summary.get("policy_name") != policy_name
            or summary.get("training_eligible") is not True
            or summary.get("bus_service_semantics_audit", {}).get("passed") is not True
            or not shard.is_file()
            or sha256(shard) != summary.get("shard_sha256")
        ):
            raise ValueError(f"invalid intervention SUMO artifact {shard}")
        rows.append(
            {
                "realization_block_id": block,
                "demand_scale": scale,
                "policy_name": policy_name,
                "metrics": _sumo_metrics(summary),
                "source": {
                    "shard": str(shard),
                    "shard_sha256": summary["shard_sha256"],
                    "summary": str(summary_path),
                    "summary_sha256": sha256(summary_path),
                },
            }
        )
    expected = {
        (block, scale, policy)
        for block in range(10)
        for scale in DEMAND_SCALES
        for policy in expected_policies
    }
    if cells != expected:
        raise ValueError("formal intervention SUMO cells are incomplete")
    return matrix, rows


def _indexed(rows: list[dict[str, Any]], unit_key: str) -> dict[tuple, dict[str, float]]:
    result = {}
    for row in rows:
        key = (
            int(row[unit_key]),
            float(row["demand_scale"]),
            row["policy_name"],
        )
        if key in result:
            raise ValueError(f"duplicate intervention result {key}")
        result[key] = row["metrics"]
    return result


def _bootstrap_difference(
    simulator: np.ndarray,
    sumo: np.ndarray,
    *,
    denominator: float,
    absolute: bool,
    seed: int,
) -> list[float]:
    if denominator <= 0.0:
        raise ValueError("intervention bootstrap denominator must be positive")
    rng = np.random.default_rng(seed)
    sim_means = simulator[
        rng.integers(0, len(simulator), size=(BOOTSTRAP_SAMPLES, len(simulator)))
    ].mean(axis=1)
    sumo_means = sumo[
        rng.integers(0, len(sumo), size=(BOOTSTRAP_SAMPLES, len(sumo)))
    ].mean(axis=1)
    differences = sim_means - sumo_means
    if not absolute:
        differences = differences / denominator
    return [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ]


def compare_interventions(
    sumo_rows: list[dict[str, Any]], simulator_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    sumo = _indexed(sumo_rows, "realization_block_id")
    simulator = _indexed(simulator_rows, "simulator_seed")
    policies = [policy["policy_name"] for policy in POLICIES if policy["policy"] != "zero"]
    comparisons = {}
    all_checks = []
    bootstrap_seed = 7300
    for scale in DEMAND_SCALES:
        scale_result = {}
        sumo_units = sorted({key[0] for key in sumo if key[1] == scale})
        simulator_units = sorted(
            {key[0] for key in simulator if key[1] == scale}
        )
        for policy_name in policies:
            result: dict[str, Any] = {"metrics": {}}
            sumo_zero = {
                name: np.asarray(
                    [sumo[(unit, scale, "zero")][name] for unit in sumo_units],
                    dtype=np.float64,
                )
                for name in METRIC_SPECS
            }
            simulator_zero = {
                name: np.asarray(
                    [simulator[(unit, scale, "zero")][name] for unit in simulator_units],
                    dtype=np.float64,
                )
                for name in METRIC_SPECS
            }
            for name, (threshold_name, mode) in METRIC_SPECS.items():
                sumo_policy = np.asarray(
                    [sumo[(unit, scale, policy_name)][name] for unit in sumo_units],
                    dtype=np.float64,
                )
                simulator_policy = np.asarray(
                    [simulator[(unit, scale, policy_name)][name] for unit in simulator_units],
                    dtype=np.float64,
                )
                sumo_response = sumo_policy - sumo_zero[name]
                simulator_response = simulator_policy - simulator_zero[name]
                denominator = (
                    1.0
                    if mode == "absolute-response"
                    else abs(float(sumo_zero[name].mean()))
                )
                point_gap = float(
                    simulator_response.mean() - sumo_response.mean()
                )
                normalized_gap = point_gap if mode == "absolute-response" else (
                    point_gap / denominator
                )
                bootstrap_seed += 1
                interval = _bootstrap_difference(
                    simulator_response,
                    sumo_response,
                    denominator=denominator,
                    absolute=mode == "absolute-response",
                    seed=bootstrap_seed,
                )
                threshold = THRESHOLDS[threshold_name]
                point_passed = abs(normalized_gap) <= threshold
                interval_passed = max(abs(bound) for bound in interval) <= threshold
                all_checks.extend((point_passed, interval_passed))
                result["metrics"][name] = {
                    "mode": mode,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "sumo_zero_mean": float(sumo_zero[name].mean()),
                    "simulator_zero_mean": float(simulator_zero[name].mean()),
                    "sumo_response_mean": float(sumo_response.mean()),
                    "simulator_response_mean": float(simulator_response.mean()),
                    "response_gap": point_gap,
                    "normalized_response_gap": normalized_gap,
                    "gap_bootstrap_95_ci": interval,
                    "point_passed": point_passed,
                    "interval_passed": interval_passed,
                }

            for name, mode, threshold_name in (
                ("mean_hold_seconds", "relative", "mean_hold_seconds_relative_gap"),
                ("exact_zero_rate", "absolute", "exact_zero_rate_absolute_gap"),
            ):
                sumo_values = np.asarray(
                    [sumo[(unit, scale, policy_name)][name] for unit in sumo_units],
                    dtype=np.float64,
                )
                simulator_values = np.asarray(
                    [simulator[(unit, scale, policy_name)][name] for unit in simulator_units],
                    dtype=np.float64,
                )
                denominator = 1.0 if mode == "absolute" else abs(float(sumo_values.mean()))
                if denominator <= 1e-12:
                    raise ValueError(
                        f"policy {policy_name} has no measurable SUMO hold intervention"
                    )
                gap = float(simulator_values.mean() - sumo_values.mean())
                normalized_gap = gap if mode == "absolute" else gap / denominator
                bootstrap_seed += 1
                interval = _bootstrap_difference(
                    simulator_values,
                    sumo_values,
                    denominator=denominator,
                    absolute=mode == "absolute",
                    seed=bootstrap_seed,
                )
                threshold = THRESHOLDS[threshold_name]
                point_passed = abs(normalized_gap) <= threshold
                interval_passed = max(abs(bound) for bound in interval) <= threshold
                all_checks.extend((point_passed, interval_passed))
                result["metrics"][name] = {
                    "mode": f"{mode}-policy-level",
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "sumo_policy_mean": float(sumo_values.mean()),
                    "simulator_policy_mean": float(simulator_values.mean()),
                    "gap": gap,
                    "normalized_gap": normalized_gap,
                    "gap_bootstrap_95_ci": interval,
                    "point_passed": point_passed,
                    "interval_passed": interval_passed,
                }
            result["passed"] = all(
                metric["point_passed"] and metric["interval_passed"]
                for metric in result["metrics"].values()
            )
            scale_result[policy_name] = result
        comparisons[f"{scale:g}"] = scale_result
    return {"passed": all(all_checks), "comparisons": comparisons}


def _episode_task(task: tuple[str, dict, int, float]) -> dict[str, Any]:
    sim_env, policy, simulator_seed, demand_scale = task
    return run_behavior_episode(
        Path(sim_env),
        policy=policy,
        simulator_seed=simulator_seed,
        demand_scale=demand_scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument(
        "--sim-env", type=Path, default=ROOT / "bus_h2o" / "calibrated_env"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--path-remap-from", type=Path)
    parser.add_argument("--path-remap-to", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix_manifest.expanduser().resolve()
    sim_env = args.sim_env.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite intervention audit {output}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    path_remap = ArtifactPathRemap.build(
        args.path_remap_from, args.path_remap_to
    )
    matrix, sumo_rows = load_sumo_rows(matrix_path, path_remap=path_remap)
    tasks = [
        (str(sim_env), dict(policy), seed, scale)
        for scale in DEMAND_SCALES
        for seed in SIMULATOR_SEEDS
        for policy in POLICIES
    ]
    if args.workers == 1:
        simulator_rows = [_episode_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            simulator_rows = list(executor.map(_episode_task, tasks))
    invariants_passed = all(
        all(row["invariants"].values()) for row in simulator_rows
    )
    comparison = compare_interventions(sumo_rows, simulator_rows)
    source = intervention_source_hashes()
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal",
        "passed": bool(invariants_passed and comparison["passed"]),
        "design": {
            "demand_scales": list(DEMAND_SCALES),
            "simulator_seeds": list(SIMULATOR_SEEDS),
            "policies": [dict(policy) for policy in POLICIES],
            "episode_seconds": EPISODE_SECONDS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "thresholds": THRESHOLDS,
            "estimand": (
                "cross-simulator gap in each policy-minus-zero response; "
                "normalization uses the SUMO zero-policy mean"
            ),
        },
        "matrix_manifest": {
            "path": str(matrix_path),
            "sha256": sha256(matrix_path),
            "jobs": len(matrix["jobs"]),
        },
        "simulator": {
            "path": str(sim_env),
            "inputs": simulator_input_hashes(
                sim_env, runtime_calibration_version="v7_v4"
            ),
        },
        "invariants_passed": invariants_passed,
        "comparison": comparison,
        "sumo_rows": sumo_rows,
        "simulator_episodes": simulator_rows,
        "source_sha256": source,
        "path_remap": (
            {
                "source_root": str(path_remap.source_root),
                "target_root": str(path_remap.target_root),
            }
            if path_remap is not None
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {"output": str(output), "passed": result["passed"]}, sort_keys=True
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
