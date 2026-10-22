#!/usr/bin/env python3
"""Evaluate one v4 policy in the calibrated low-fidelity simulator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
from bus_h2o.mdp_contract_v4 import (
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v4,
    policy_action_to_hold_seconds_v4,
    transition_cost_components_v4,
    transition_duration_seconds_v4,
    transition_reward_v4,
)
from SimpleSAC.evaluate_transit_v4 import (
    _event_record,
    build_policy,
    load_dataset_contract,
)
from SimpleSAC.train_transit_v3 import sha256, simulator_input_hashes, source_hashes
from SimpleSAC.transit_v3.evaluation import numeric_summary, summarize_operations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--zero-hold", action="store_true")
    policy.add_argument("--headway-balance-gain", type=float)
    policy.add_argument("--myopic-welfare", action="store_true")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--sim-seed", type=int, required=True)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "bus_h2o" / "datasets_v4" / "transit_smdp_v4.h5",
    )
    parser.add_argument(
        "--sim-env",
        type=Path,
        default=ROOT / "bus_h2o" / "calibrated_env",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def active_observations(state: dict) -> list[tuple[object, np.ndarray]]:
    active = []
    for agent, values in state.items():
        if not values:
            continue
        observation = values[-1]
        while isinstance(observation, list) and len(observation) == 1:
            observation = observation[0]
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        if raw.shape != (RAW_EVENT_STATE_DIM,):
            raise ValueError(
                f"simulator emitted raw v4 shape {raw.shape}; "
                f"expected ({RAW_EVENT_STATE_DIM},)"
            )
        if not np.isfinite(raw).all():
            raise ValueError("simulator emitted a non-finite v4 event")
        active.append((agent, raw.copy()))
    return active


def simulator_passengers(env: AllLinesSimEnv, horizon: float) -> dict:
    passengers = []
    for line_env in env.line_map.values():
        captured = tuple(getattr(line_env, "last_episode_passengers", ()))
        if captured:
            passengers.extend(captured)
        else:
            for station in line_env.stations:
                passengers.extend(station.total_passenger)

    accrued_wait = 0.0
    accrued_in_vehicle = 0.0
    completed_wait = []
    completed_in_vehicle = []
    completed_total = []
    states: Counter[str] = Counter()
    for passenger in passengers:
        appear = float(passenger.appear_time)
        board = passenger.boarding_time
        arrive = passenger.arrive_time
        if board is None:
            accrued_wait += max(horizon - appear, 0.0)
            states["waiting"] += 1
            continue
        board = float(board)
        wait = max(board - appear, 0.0)
        accrued_wait += wait
        if arrive is None:
            accrued_in_vehicle += max(horizon - board, 0.0)
            states["onboard"] += 1
            continue
        arrive = float(arrive)
        in_vehicle = max(arrive - board, 0.0)
        accrued_in_vehicle += in_vehicle
        completed_wait.append(wait)
        completed_in_vehicle.append(in_vehicle)
        completed_total.append(max(arrive - appear, 0.0))
        states["completed"] += 1

    generated = len(passengers)
    completed = len(completed_total)
    return {
        "unit": "independently simulated passenger ride legs",
        "generated_ride_legs": generated,
        "completed_ride_legs": completed,
        "completion_rate": completed / generated if generated else None,
        "unfinished_ride_legs": generated - completed,
        "state_counts_at_horizon": dict(sorted(states.items())),
        "completed_leg_wait_seconds": numeric_summary(completed_wait),
        "completed_leg_in_vehicle_seconds": numeric_summary(completed_in_vehicle),
        "completed_leg_total_seconds": numeric_summary(completed_total),
        "accrued_wait_seconds_total": accrued_wait,
        "accrued_in_vehicle_seconds_total": accrued_in_vehicle,
        "accrued_wait_seconds_per_generated_leg": (
            accrued_wait / generated if generated else None
        ),
        "accrued_in_vehicle_seconds_per_generated_leg": (
            accrued_in_vehicle / generated if generated else None
        ),
    }


def simulator_service(env: AllLinesSimEnv) -> dict:
    rows = [
        row
        for line_env in env.line_map.values()
        for row in getattr(line_env, "last_episode_bus_summaries", ())
    ]
    completed = [row for row in rows if row["completed"]]
    durations = [
        row["terminal_arrival_seconds"] - row["launch_time_seconds"]
        for row in completed
        if row["terminal_arrival_seconds"] is not None
    ]
    return {
        "departed_trips": len(rows),
        "completed_trips": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else None,
        "completed_trip_duration_seconds": numeric_summary(durations),
    }


def main() -> None:
    args = parse_args()
    if args.demand_scale <= 0.0 or not np.isfinite(args.demand_scale):
        raise ValueError("demand-scale must be finite and positive")
    args.dataset = args.dataset.expanduser().resolve()
    args.sim_env = args.sim_env.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite evaluation {args.output}")

    random.seed(args.sim_seed)
    np.random.seed(args.sim_seed)
    torch.manual_seed(args.sim_seed)
    torch.set_num_threads(1)
    config, station_max, dataset_hash = load_dataset_contract(args.dataset)
    policy, policy_metadata = build_policy(
        args, config, station_max, dataset_hash
    )
    simulator_inputs = simulator_input_hashes(args.sim_env)
    source = source_hashes("v4")
    source[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(
        Path(__file__).resolve()
    )
    env = AllLinesSimEnv(
        str(args.sim_env),
        episode_seconds=config.episode_seconds,
        demand_scale=args.demand_scale,
        observation_schema="v4",
    )

    pending = {}
    event_records = []
    transition_records = []
    station_wraparounds = 0
    started = time.perf_counter()
    env.reset()
    if not active_observations(env.state):
        env.step_to_event({})
    while not env.done:
        active = active_observations(env.state)
        if not active:
            _, _, done = env.step_to_event({})
            if done:
                break
            continue
        raw_batch = np.stack([raw for _, raw in active])
        action_batch = np.asarray(policy(raw_batch), dtype=np.float32)
        if action_batch.shape != (len(active), 1):
            raise ValueError(
                f"policy returned {action_batch.shape}; expected {(len(active), 1)}"
            )
        if not np.isfinite(action_batch).all() or np.any(
            np.abs(action_batch) > 1.00001
        ):
            raise ValueError("policy emitted an invalid action")

        env_actions = {}
        for (agent, raw), action in zip(active, action_batch):
            line_id, bus_id = agent
            station_index = int(
                round(float(raw[RAW_EVENT_STATE_INDEX["station_index"]]))
            )
            hold_seconds = float(
                policy_action_to_hold_seconds_v4(action[0], config)
            )
            event = SimpleNamespace(
                sim_time=float(raw[RAW_EVENT_STATE_INDEX["sim_time_seconds"]]),
                line_id=line_id,
                bus_id=bus_id,
            )
            record = _event_record(event, raw, hold_seconds)
            record["action"] = action.copy()
            event_records.append(record)

            previous = pending.pop(agent, None)
            if previous is not None and station_index > previous["station_index"]:
                duration = float(
                    transition_duration_seconds_v4(previous["raw"], raw)
                )
                transition_records.append(
                    {
                        "line_id": str(line_id),
                        "raw_observation": previous["raw"],
                        "action": previous["action"],
                        "raw_next_observation": raw.copy(),
                        "reward": float(
                            transition_reward_v4(
                                previous["raw"], previous["action"], raw, config
                            )
                        ),
                        "delta_t_seconds": duration,
                        "discount": float(
                            continuous_time_discount_v4(duration, config)
                        ),
                    }
                )
            elif previous is not None:
                station_wraparounds += 1
            pending[agent] = {
                "raw": raw.copy(),
                "action": action.copy(),
                "station_index": station_index,
            }
            env_actions[agent] = [hold_seconds, 1.0]
        env.step_to_event(env_actions)

    if not transition_records:
        raise RuntimeError("simulator evaluation produced no same-bus transitions")
    observations = np.asarray(
        [row["raw_observation"] for row in transition_records], dtype=np.float32
    )
    actions = np.asarray(
        [row["action"] for row in transition_records], dtype=np.float32
    ).reshape(len(transition_records), 1)
    next_observations = np.asarray(
        [row["raw_next_observation"] for row in transition_records],
        dtype=np.float32,
    )
    costs = transition_cost_components_v4(
        observations, actions, next_observations, config
    )
    passengers = simulator_passengers(env, config.episode_seconds)
    generalized_total = (
        config.waiting_time_value * passengers["accrued_wait_seconds_total"]
        + config.in_vehicle_time_value
        * passengers["accrued_in_vehicle_seconds_total"]
    )
    generated = passengers["generated_ride_legs"]
    result = {
        "schema_version": "transit-simulator-evaluation-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method_name,
        "contract_schema": SCHEMA_VERSION,
        "contract": config.to_dict(),
        "raw_state_names": list(RAW_EVENT_STATE_NAMES),
        "dataset": {"path": str(args.dataset), "sha256": dataset_hash},
        "policy": policy_metadata,
        "sim_seed": args.sim_seed,
        "demand_scale": args.demand_scale,
        "episode": {
            "configured_seconds": config.episode_seconds,
            "observed_end_time_seconds": float(env.current_time),
            "events": len(event_records),
            "transitions": len(transition_records),
            "discarded_pending": len(pending),
            "station_wraparounds": station_wraparounds,
        },
        "operations": summarize_operations(event_records, transition_records),
        "objective": {
            "units": "generalized passenger-minutes",
            "waiting_cost": numeric_summary(costs["waiting_cost"]),
            "holding_cost": numeric_summary(costs["holding_cost"]),
            "total_cost": numeric_summary(costs["total_cost"]),
            "cumulative_waiting_cost": float(np.sum(costs["waiting_cost"])),
            "cumulative_holding_cost": float(np.sum(costs["holding_cost"])),
            "cumulative_total_cost": float(np.sum(costs["total_cost"])),
        },
        "passengers": passengers,
        "passenger_welfare": {
            "denominator": "generated simulator ride legs",
            "waiting_time_value": config.waiting_time_value,
            "in_vehicle_time_value": config.in_vehicle_time_value,
            "accrued_generalized_passenger_seconds": generalized_total,
            "accrued_generalized_seconds_per_generated_leg": (
                generalized_total / generated if generated else None
            ),
        },
        "service": simulator_service(env),
        "simulator_inputs": simulator_inputs,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": args.device,
        },
        "source_sha256": source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
