#!/usr/bin/env python3
"""Audit v7 simulator fidelity on SUMO blocks excluded from calibration."""

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
from collect_policy.build_transit_v7_dataset import load_shard
from reproducibility.audit_simulator_fidelity_v6 import (
    FIDELITY_THRESHOLDS,
    _scale_key,
    compare_fidelity,
)
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FROZEN_ZERO_AUDIT_SCHEMA,
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
    FORMAL_FIDELITY_REALIZATION_BLOCKS,
    MATRIX_SCHEMA,
)
from reproducibility.build_simulator_runtime_calibration_v7_v4 import (
    CALIBRATION_SCHEMA_V4 as CALIBRATION_SCHEMA,
)
from reproducibility.relocatable_paths_v7 import (
    ArtifactPathRemap,
    resolve_evidence_path,
)
from SimpleSAC.train_transit_v3 import (
    collect_runtime_fingerprint,
    simulator_input_hashes,
)
from SimpleSAC.transit_v3.evaluation import numeric_summary


SCHEMA_VERSION = "transit-simulator-fidelity-audit-v7-formal-v2"
DEFAULT_DEMAND_SCALES = (0.75, 1.0, 1.25)
DEFAULT_SIMULATOR_SEEDS = tuple(range(9901, 9911))


def fidelity_source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "audit_simulator_fidelity_v6.py",
        ROOT / "reproducibility" / "build_simulator_runtime_calibration_v7.py",
        ROOT / "reproducibility" / "build_simulator_runtime_calibration_v7_v2.py",
        ROOT / "reproducibility" / "build_simulator_runtime_calibration_v7_v3.py",
        ROOT / "reproducibility" / "build_simulator_runtime_calibration_v7_v4.py",
        ROOT / "reproducibility" / "relocatable_paths_v7.py",
        ROOT / "collect_policy" / "build_transit_v7_dataset.py",
        ROOT / "bus_h2o" / "envs" / "all_lines_sim_env.py",
        ROOT / "bus_h2o" / "mdp_contract_v4.py",
        ROOT / "bus_h2o" / "mdp_contract_v5.py",
        ROOT / "bus_h2o" / "mdp_contract_v7.py",
        ROOT / "bus_h2o" / "passenger_time.py",
        ROOT / "bus_h2o" / "sumo_provenance.py",
        ROOT / "SimpleSAC" / "transit_v3" / "evaluation.py",
        *sorted((ROOT / "bus_h2o" / "sim_core").glob("*.py")),
    ]
    resolved = sorted({path.expanduser().resolve(strict=True) for path in paths})
    return {
        path.relative_to(ROOT).as_posix(): sha256(path) for path in resolved
    }


def _active_agents(state: dict) -> list[object]:
    return sorted(
        (agent for agent, observations in state.items() if observations),
        key=lambda agent: (str(agent[0]), int(agent[1])),
    )


def run_zero_hold_episode_v7(
    sim_env: Path,
    *,
    seed: int,
    episode_seconds: float,
    demand_scale: float,
) -> dict[str, Any]:
    """Run one fixed-horizon v7 simulator episode and reconcile ledgers."""

    random.seed(seed)
    np.random.seed(seed)
    env = AllLinesSimEnv(
        str(sim_env),
        episode_seconds=episode_seconds,
        demand_scale=demand_scale,
        observation_schema="v7",
        observation_speed_seed=seed,
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
        raise RuntimeError("v7 fidelity audit requires causal source journeys")
    completed_service = [row for row in service_rows if row["completed"]]
    trip_durations = [
        float(row["terminal_arrival_seconds"] - row["launch_time_seconds"])
        for row in completed_service
        if row["terminal_arrival_seconds"] is not None
    ]
    generalized = (
        2.0 * ledger.accrued_wait_seconds + ledger.accrued_in_vehicle_seconds
    )
    return {
        "seed": int(seed),
        "demand_scale": float(demand_scale),
        "observation_speed_profile": env.current_observation_speed_profile,
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
            "ledger_count_errors": {
                "waiting": int(ledger.waiting_passengers - waiting),
                "in_vehicle": int(ledger.in_vehicle_passengers - onboard),
            },
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
                len(completed_service) / len(service_rows)
                if service_rows
                else None
            ),
            "completed_trip_duration_seconds": numeric_summary(trip_durations),
        },
    }


def _reserved_jobs(matrix: dict, calibration: dict) -> list[dict]:
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("v7 fidelity audit requires collection matrix v2")
    if matrix.get("study_stage") != "formal":
        raise ValueError("v7 formal fidelity requires a formal collection matrix")
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("v7 runtime calibration schema is invalid")
    if calibration.get("coverage_gate", {}).get("passed") is not True:
        raise ValueError("v7 runtime calibration did not pass its coverage gate")
    partition = calibration.get("data_partition", {})
    if partition.get("calibration_realization_blocks") != list(
        FORMAL_CALIBRATION_REALIZATION_BLOCKS
    ) or partition.get("reserved_fidelity_realization_blocks") != list(
        FORMAL_FIDELITY_REALIZATION_BLOCKS
    ):
        raise ValueError("v7 runtime calibration partition is not registered")
    if set(partition["calibration_realization_blocks"]) & set(
        partition["reserved_fidelity_realization_blocks"]
    ):
        raise ValueError("v7 calibration and fidelity blocks overlap")

    reserved = set(FORMAL_FIDELITY_REALIZATION_BLOCKS)
    jobs = [
        job
        for job in matrix.get("jobs", [])
        if job.get("policy", {}).get("policy_name") == "zero"
        and int(job["realization_block_id"]) in reserved
    ]
    expected_cells = {
        (block, scale)
        for block in FORMAL_FIDELITY_REALIZATION_BLOCKS
        for scale in DEFAULT_DEMAND_SCALES
    }
    actual_cells = [
        (
            int(job["realization_block_id"]),
            float(job["scenario"]["demand_scale"]),
        )
        for job in jobs
    ]
    if len(actual_cells) != len(expected_cells) or set(actual_cells) != expected_cells:
        raise ValueError("reserved v7 fidelity cells are missing or duplicated")
    return jobs


def load_reserved_sumo_summaries(
    matrix_manifest: Path,
    calibration_path: Path,
    *,
    path_remap: ArtifactPathRemap | None = None,
) -> tuple[dict, dict, list[dict[str, Any]]]:
    matrix = json.loads(matrix_manifest.read_text())
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("provenance", {}).get(
        "collection_matrix_sha256"
    ) != sha256(matrix_manifest):
        raise ValueError("v7 calibration does not match the collection matrix")
    freeze_descriptor = calibration.get("provenance", {}).get(
        "frozen_zero_matrix_audit", {}
    )
    freeze_path = resolve_evidence_path(
        freeze_descriptor.get("path", ""), path_remap
    )
    if (
        freeze_descriptor.get("schema_version") != FROZEN_ZERO_AUDIT_SCHEMA
        or not freeze_path.is_file()
        or freeze_descriptor.get("sha256") != sha256(freeze_path)
    ):
        raise ValueError("v7 calibration freeze-audit provenance is missing or stale")
    freeze_audit = json.loads(freeze_path.read_text())
    if (
        freeze_audit.get("schema_version") != FROZEN_ZERO_AUDIT_SCHEMA
        or freeze_audit.get("passed") is not True
        or freeze_audit.get("artifact_set", {}).get("exact_match") is not True
        or freeze_audit.get("frozen_manifest", {}).get("sha256")
        != sha256(matrix_manifest)
        or resolve_evidence_path(
            freeze_audit.get("frozen_manifest", {}).get("path", ""),
            path_remap,
        )
        != matrix_manifest.resolve()
    ):
        raise ValueError("v7 fidelity input is not bound by a passing freeze audit")
    jobs = _reserved_jobs(matrix, calibration)
    summaries = []
    for job in sorted(jobs, key=lambda value: value["output"]):
        shard = resolve_evidence_path(job["output"], path_remap)
        summary, _ = load_shard(shard, allow_partial=False)
        if summary.get("job_signature") != job.get("signature"):
            raise ValueError(f"reserved SUMO shard signature mismatch: {shard}")
        if summary.get("source_sha256") != matrix.get("source_sha256"):
            raise ValueError(f"reserved SUMO shard source mismatch: {shard}")
        if summary.get("policy_name") != "zero" or not summary.get(
            "training_eligible"
        ):
            raise ValueError(f"ineligible SUMO fidelity source: {shard}")
        summary["_fidelity_source"] = {
            "summary": str(shard.with_suffix(shard.suffix + ".json")),
            "summary_sha256": sha256(shard.with_suffix(shard.suffix + ".json")),
            "shard": str(shard),
            "shard_sha256": summary["shard_sha256"],
            "realization_block_id": int(job["realization_block_id"]),
        }
        summaries.append(summary)
    return matrix, calibration, summaries


def _sumo_source_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row["_fidelity_source"],
        "sumo_seed": int(row["sumo_seed"]),
        "demand_scale": float(row["scenario"]["demand_scale"]),
        "scaled_total_persons": int(row["scenario"]["scaled_total_persons"]),
        "scenario_manifest_sha256": row["scenario"]["manifest_sha256"],
    }


def _run_episode_task(task: tuple[str, int, float, float]) -> dict[str, Any]:
    sim_env, seed, episode_seconds, demand_scale = task
    return run_zero_hold_episode_v7(
        Path(sim_env),
        seed=seed,
        episode_seconds=episode_seconds,
        demand_scale=demand_scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=(
            ROOT
            / "bus_h2o"
            / "datasets_v7"
            / "formal_v1"
            / "zero_matrix_manifest.json"
        ),
    )
    parser.add_argument(
        "--runtime-calibration",
        type=Path,
        default=ROOT / "bus_h2o" / "calibrated_env" / "segment_runtime_calibration_v7_v4.json",
    )
    parser.add_argument(
        "--sim-env", type=Path, default=ROOT / "bus_h2o" / "calibrated_env"
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
    parser.add_argument("--path-remap-from", type=Path)
    parser.add_argument("--path-remap-to", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.matrix_manifest = args.matrix_manifest.expanduser().resolve()
    args.runtime_calibration = args.runtime_calibration.expanduser().resolve()
    args.sim_env = args.sim_env.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite fidelity audit {args.output}")
    if tuple(args.sim_seeds) != DEFAULT_SIMULATOR_SEEDS:
        raise ValueError("formal v7 fidelity simulator seeds are registered")
    if tuple(args.demand_scales) != DEFAULT_DEMAND_SCALES:
        raise ValueError("formal v7 fidelity demand scales are registered")
    if args.episode_seconds != 24_000.0:
        raise ValueError("formal v7 fidelity horizon is registered at 24000 seconds")
    if args.workers <= 0 or args.bootstrap_samples != 10_000:
        raise ValueError("formal v7 workers must be positive and bootstrap count is fixed")
    path_remap = ArtifactPathRemap.build(
        args.path_remap_from, args.path_remap_to
    )

    matrix, calibration, sumo_summaries = load_reserved_sumo_summaries(
        args.matrix_manifest,
        args.runtime_calibration,
        path_remap=path_remap,
    )
    calibration_demand = args.sim_env / "simulator_demand_calibration.json"
    demand_report = json.loads(calibration_demand.read_text())
    if demand_report.get("schema_version") != (
        "transit-simulator-demand-calibration-v3"
    ) or not demand_report.get("journey_execution", {}).get(
        "causal_transfer_chain"
    ):
        raise ValueError("v7 fidelity requires causal demand calibration v3")
    base_source_persons = float(demand_report["source_passengers"])
    sumo_by_scale = {
        scale: [
            row
            for row in sumo_summaries
            if np.isclose(float(row["scenario"]["demand_scale"]), scale)
        ]
        for scale in args.demand_scales
    }
    if any(len(rows) != len(FORMAL_FIDELITY_REALIZATION_BLOCKS) for rows in sumo_by_scale.values()):
        raise ValueError("each v7 fidelity stratum requires five reserved SUMO blocks")

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
            row for row in episodes if np.isclose(row["demand_scale"], scale)
        ]
        comparison = compare_fidelity(
            scale_episodes,
            sumo_by_scale[scale],
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
                _sumo_source_descriptor(row) for row in sumo_by_scale[scale]
            ],
            "sumo_scaled_person_counts": numeric_summary(
                [
                    float(row["scenario"]["scaled_total_persons"])
                    for row in sumo_by_scale[scale]
                ]
            ),
        }

    source = fidelity_source_hashes()
    import torch

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal",
        "passed": all(row["comparison"]["passed"] for row in strata.values()),
        "design": {
            "demand_scales": list(DEFAULT_DEMAND_SCALES),
            "simulator_seeds": list(DEFAULT_SIMULATOR_SEEDS),
            "calibration_realization_blocks": list(
                FORMAL_CALIBRATION_REALIZATION_BLOCKS
            ),
            "reserved_fidelity_realization_blocks": list(
                FORMAL_FIDELITY_REALIZATION_BLOCKS
            ),
            "sumo_blocks_per_scale": 5,
            "episode_seconds": 24_000.0,
            "bootstrap_samples": 10_000,
            "thresholds": FIDELITY_THRESHOLDS,
            "zero_hold_policy": [0.0, 1.0],
        },
        "strata": strata,
        "simulator_episodes": episodes,
        "sumo_sources": [
            _sumo_source_descriptor(row) for row in sumo_summaries
        ],
        "collection_matrix": {
            "path": str(args.matrix_manifest),
            "sha256": sha256(args.matrix_manifest),
            "source_sha256": matrix.get("source_sha256"),
        },
        "runtime_calibration": {
            "path": str(args.runtime_calibration),
            "sha256": sha256(args.runtime_calibration),
            "data_partition": calibration["data_partition"],
        },
        "simulator_inputs": simulator_input_hashes(
            args.sim_env, runtime_calibration_version="v7_v4"
        ),
        "demand_calibration": {
            "path": str(calibration_demand),
            "sha256": sha256(calibration_demand),
        },
        "source_sha256": source,
        "runtime_fingerprint": collect_runtime_fingerprint(torch, "cpu"),
        "path_remap": (
            {
                "source_root": str(path_remap.source_root),
                "target_root": str(path_remap.target_root),
            }
            if path_remap is not None
            else None
        ),
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
                        "passed": row["comparison"]["passed"],
                        "checks": row["comparison"]["checks"],
                        "gaps": row["comparison"]["gaps"],
                    }
                    for key, row in strata.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
