#!/usr/bin/env python3
"""Collect one provenance-locked true-welfare SUMO shard for transit-smdp-v5."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4
from bus_h2o.mdp_contract_v5 import (
    DEFAULT_CONFIG_V5,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v5,
    hold_seconds_to_policy_action_v5,
    passenger_time_snapshot_v5,
    policy_action_to_hold_seconds_v5,
    transition_cost_components_v5,
    transition_reward_v5,
)
from bus_h2o.sumo_provenance import (
    collection_source_paths,
    hash_files,
    load_validated_scenario,
    sha256,
    sumo_version,
)
from SimpleSAC.transit_v3.evaluation import (
    numeric_summary,
    summarize_service,
)
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4,
    hurdle_random_actions_v4,
    myopic_welfare_hold_seconds_v4,
)
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        required=True,
        choices=("zero", "random-hurdle", "headway-balance", "myopic-welfare"),
    )
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--sumo-seed", type=int, required=True)
    parser.add_argument("--headway-gain", type=float, default=0.1)
    parser.add_argument("--random-zero-probability", type=float, default=0.5)
    parser.add_argument("--random-beta-alpha", type=float, default=1.0)
    parser.add_argument("--random-beta-beta", type=float, default=3.0)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-signature")
    parser.add_argument("--episode-seconds", type=int, default=24_000)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow a shortened horizon or event limit; output is not training eligible",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="diagnostic limit on non-terminal control decisions; zero is full horizon",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sumo-dir",
        type=Path,
        default=ROOT.parent / "SUMO_ruiguang" / "online_control",
    )
    parser.add_argument(
        "--edge-xml",
        type=Path,
        default=ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml",
    )
    return parser.parse_args()


def raw_observation(env: AllLinesSumoEnv, bus_id) -> np.ndarray:
    observation = env.state[bus_id][-1]
    while isinstance(observation, list) and len(observation) == 1:
        observation = observation[0]
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    if array.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"SUMO emitted raw v5 shape {array.shape}; "
            f"expected ({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(array).all():
        raise ValueError("SUMO emitted a non-finite v5 event")
    return array


def build_policy(args: argparse.Namespace):
    config = DEFAULT_CONFIG_V5
    rng = np.random.default_rng(args.policy_seed)
    if args.policy == "zero":
        return (
            lambda raw: np.full((len(raw), 1), -1.0, dtype=np.float32),
            {"type": "zero", "exact_zero_probability": 1.0},
        )
    if args.policy == "random-hurdle":
        return (
            lambda raw: hurdle_random_actions_v4(
                len(raw),
                rng,
                zero_probability=args.random_zero_probability,
                beta_alpha=args.random_beta_alpha,
                beta_beta=args.random_beta_beta,
                config=config,
            ),
            {
                "type": "random-hurdle",
                "zero_probability": args.random_zero_probability,
                "positive_hold_fraction": {
                    "distribution": "beta",
                    "alpha": args.random_beta_alpha,
                    "beta": args.random_beta_beta,
                },
            },
        )
    if args.policy == "headway-balance":
        if args.headway_gain < 0.0:
            raise ValueError("headway gain must be non-negative")

        def policy(raw):
            holds = headway_balance_hold_seconds_v4(
                raw[:, :RAW_EVENT_STATE_DIM_V4], args.headway_gain, config
            )
            return np.asarray(
                hold_seconds_to_policy_action_v5(holds, config), dtype=np.float32
            )[:, None]

        return policy, {
            "type": "schedule-adjusted-headway-balance",
            "gain": args.headway_gain,
            "hold_seconds": "clip(gain*((H_b-T_b)-(H_f-T_f))/2, 0, 60)",
        }

    def policy(raw):
        holds = myopic_welfare_hold_seconds_v4(
            raw[:, :RAW_EVENT_STATE_DIM_V4], config
        )
        return np.asarray(
            hold_seconds_to_policy_action_v5(holds, config), dtype=np.float32
        )[:, None]

    return policy, {
        "type": "closed-form-myopic-headway-surrogate",
        "note": "behavior coverage policy only; v5 learning reward is true passenger time",
    }


def _is_terminal_event(env: AllLinesSumoEnv, event) -> bool:
    line = env._bridge.line_obj_dic[event.line_id]
    return bool(
        event.stop_id == line.stop_id_l[-1]
        or int(event.stop_idx) >= len(line.stop_id_l) - 1
    )


def _append_transition(
    rows: dict[str, list],
    previous: dict,
    raw_next_observation: np.ndarray,
    passenger_time_end,
    *,
    terminal: bool,
) -> None:
    config = DEFAULT_CONFIG_V5
    start = previous["passenger_time"]
    end = passenger_time_snapshot_v5(passenger_time_end)
    duration = float(end.time_seconds - start.time_seconds)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"v5 transition has invalid duration {duration}")
    components = transition_cost_components_v5(start, end, config)
    reward = transition_reward_v5(start, end, config)
    rows["observations"].append(previous["raw"])
    rows["actions"].append(previous["action"])
    rows["rewards"].append(reward)
    rows["next_observations"].append(raw_next_observation.copy())
    rows["discounts"].append(
        0.0 if terminal else continuous_time_discount_v5(duration, config)
    )
    rows["terminals"].append(int(terminal))
    rows["delta_t_seconds"].append(duration)
    rows["trajectory_local_id"].append(previous["trajectory_id"])
    rows["wait_passenger_seconds"].append(
        components["wait_passenger_seconds"]
    )
    rows["in_vehicle_passenger_seconds"].append(
        components["in_vehicle_passenger_seconds"]
    )


def _horizon_observation(previous_raw: np.ndarray, passenger_time_end) -> np.ndarray:
    end = passenger_time_snapshot_v5(passenger_time_end)
    observation = previous_raw.copy()
    observation[RAW_EVENT_STATE_INDEX["sim_time_seconds"]] = end.time_seconds
    observation[
        RAW_EVENT_STATE_INDEX["system_waiting_passengers"]
    ] = end.waiting_passengers
    observation[
        RAW_EVENT_STATE_INDEX["system_in_vehicle_passengers"]
    ] = end.in_vehicle_passengers
    return observation


def _close_completed_trip_transitions(
    bridge, pending: dict, rows: dict[str, list]
) -> tuple[int, int]:
    """Close pending rows at each trip's exact SUMO completion instant."""

    contexts = bridge.consume_completed_trip_passenger_times()
    closures = 0
    for bus_id, passenger_time_end in contexts.items():
        previous = pending.pop(str(bus_id), None)
        if previous is None:
            continue
        _append_transition(
            rows,
            previous,
            _horizon_observation(previous["raw"], passenger_time_end),
            passenger_time_end,
            terminal=True,
        )
        closures += 1
    return len(contexts), closures


def save_shard(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return sha256(path)


def main() -> None:
    args = parse_args()
    if args.episode_seconds <= 0:
        raise ValueError("episode-seconds must be positive")
    if args.max_events < 0:
        raise ValueError("max-events must be non-negative")
    diagnostic_conditions = (
        args.episode_seconds != int(DEFAULT_CONFIG_V5.episode_seconds)
        or args.max_events > 0
    )
    if diagnostic_conditions and not args.diagnostic:
        raise ValueError(
            "shortened horizons and max-events require --diagnostic so the "
            "shard cannot be mistaken for training data"
        )
    args.output = args.output.expanduser().resolve()
    summary_path = args.output.with_suffix(args.output.suffix + ".json")
    if not args.force and (args.output.exists() or summary_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing shard {args.output}")

    scenario_runtime, scenario = load_validated_scenario(args.scenario_manifest)
    policy, policy_metadata = build_policy(args)
    config = DEFAULT_CONFIG_V5
    env = AllLinesSumoEnv(
        str(args.sumo_dir.expanduser().resolve()),
        str(args.edge_xml.expanduser().resolve()),
        episode_seconds=args.episode_seconds,
        sumo_seed=args.sumo_seed,
        sumo_cfg=scenario_runtime["sumo_cfg"],
        passenger_obj_xml=scenario_runtime["passenger_obj_xml"],
        observation_schema="v5",
    )
    rows = {
        name: []
        for name in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "discounts",
            "terminals",
            "delta_t_seconds",
            "trajectory_local_id",
            "wait_passenger_seconds",
            "in_vehicle_passenger_seconds",
        )
    }
    pending = {}
    trajectory_ids = {}
    control_decisions = 0
    terminal_endpoints = 0
    completed_trip_contexts = 0
    completion_closures = 0
    horizon_closures = 0
    station_wraparounds = 0
    holds = []
    started = time.perf_counter()
    try:
        env.reset()
        env.step_to_event({})
        seen_contexts, closed = _close_completed_trip_transitions(
            env._bridge, pending, rows
        )
        completed_trip_contexts += seen_contexts
        completion_closures += closed
        while not env.done:
            current_events = list(env._current_events)
            if not current_events:
                _, _, done = env.step_to_event({})
                seen_contexts, closed = _close_completed_trip_transitions(
                    env._bridge, pending, rows
                )
                completed_trip_contexts += seen_contexts
                completion_closures += closed
                if done:
                    break
                continue
            event_rows = []
            control_raw = []
            for event in current_events:
                raw = raw_observation(env, event.bus_id)
                passenger_time = passenger_time_snapshot_v5(
                    event.metadata["passenger_time_at_decision"]
                )
                terminal = _is_terminal_event(env, event)
                event_rows.append((event, raw, passenger_time, terminal))
                if not terminal:
                    control_raw.append(raw)

            if control_raw:
                action_batch = np.asarray(
                    policy(np.stack(control_raw)), dtype=np.float32
                )
                if action_batch.shape != (len(control_raw), 1):
                    raise ValueError(
                        f"behavior policy returned {action_batch.shape}; "
                        f"expected {(len(control_raw), 1)}"
                    )
                if not np.isfinite(action_batch).all() or np.any(
                    np.abs(action_batch) > 1.00001
                ):
                    raise ValueError("behavior policy emitted an invalid action")
                control_actions = iter(action_batch)
            else:
                control_actions = iter(())

            env_actions = {}
            for event, raw, passenger_time, terminal in event_rows:
                bus_id = str(event.bus_id)
                station_index = int(
                    round(float(raw[RAW_EVENT_STATE_INDEX["station_index"]]))
                )
                previous = pending.pop(bus_id, None)
                if previous is not None and station_index > previous["station_index"]:
                    _append_transition(
                        rows,
                        previous,
                        raw,
                        passenger_time,
                        terminal=terminal,
                    )
                elif previous is not None:
                    station_wraparounds += 1
                    _append_transition(
                        rows,
                        previous,
                        _horizon_observation(previous["raw"], passenger_time),
                        passenger_time,
                        terminal=True,
                    )

                if terminal:
                    terminal_endpoints += 1
                    env_actions[event.bus_id] = [0.0, 1.0]
                    continue

                action = next(control_actions)
                trajectory_id = trajectory_ids.setdefault(
                    bus_id, len(trajectory_ids)
                )
                hold_seconds = float(
                    policy_action_to_hold_seconds_v5(action[0], config)
                )
                control_decisions += 1
                holds.append(hold_seconds)
                pending[bus_id] = {
                    "raw": raw.copy(),
                    "action": action.copy(),
                    "station_index": station_index,
                    "trajectory_id": trajectory_id,
                    "passenger_time": passenger_time,
                }
                env_actions[event.bus_id] = [hold_seconds, 1.0]

            _, _, done = env.step_to_event(env_actions)
            seen_contexts, closed = _close_completed_trip_transitions(
                env._bridge, pending, rows
            )
            completed_trip_contexts += seen_contexts
            completion_closures += closed
            if done:
                break
            if args.max_events and control_decisions >= args.max_events:
                break

        bridge = env._bridge
        observed_end_time = float(bridge.current_time)
        passenger_count_audit = bridge.audit_passenger_time_counts()
        passenger_time_end = bridge.passenger_time_ledger.snapshot()
        for previous in tuple(pending.values()):
            if passenger_time_end.time_seconds <= previous["passenger_time"].time_seconds:
                continue
            _append_transition(
                rows,
                previous,
                _horizon_observation(previous["raw"], passenger_time_end),
                passenger_time_end,
                terminal=True,
            )
            horizon_closures += 1
        pending.clear()
        backend = bridge.sumo_binary
        passenger_summary = summarize_passengers_v5(bridge, passenger_time_end)
        service_summary = summarize_service(bridge)
    finally:
        env.close()

    if passenger_summary["active_state_count_balance_error"] < 0:
        raise RuntimeError(
            "SUMO passenger-state counts exceed the number of unfinished passengers"
        )
    if (
        not diagnostic_conditions
        and service_summary["active_trips_at_horizon"] == 0
        and horizon_closures
    ):
        raise RuntimeError(
            "a full episode closed pending trajectories at the horizon even though "
            "all SUMO trips had already completed"
        )

    n = len(rows["rewards"])
    if n == 0:
        raise RuntimeError("SUMO v5 shard contains no transitions")
    arrays = {
        "observations": np.asarray(rows["observations"], dtype=np.float32).reshape(
            n, RAW_EVENT_STATE_DIM
        ),
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, 1),
        "rewards": np.asarray(rows["rewards"], dtype=np.float64),
        "next_observations": np.asarray(
            rows["next_observations"], dtype=np.float32
        ).reshape(n, RAW_EVENT_STATE_DIM),
        "discounts": np.asarray(rows["discounts"], dtype=np.float32),
        "terminals": np.asarray(rows["terminals"], dtype=np.uint8),
        "delta_t_seconds": np.asarray(rows["delta_t_seconds"], dtype=np.float64),
        "trajectory_local_id": np.asarray(
            rows["trajectory_local_id"], dtype=np.int64
        ),
        "wait_passenger_seconds": np.asarray(
            rows["wait_passenger_seconds"], dtype=np.float64
        ),
        "in_vehicle_passenger_seconds": np.asarray(
            rows["in_vehicle_passenger_seconds"], dtype=np.float64
        ),
    }
    reward_reconstruction = -(
        config.waiting_time_value * arrays["wait_passenger_seconds"]
        + config.in_vehicle_time_value * arrays["in_vehicle_passenger_seconds"]
    ) / config.reward_unit_passenger_seconds
    reward_error = float(
        np.max(np.abs(arrays["rewards"] - reward_reconstruction))
    )
    if reward_error > 1e-6:
        raise RuntimeError(f"v5 reward reconstruction error is {reward_error}")

    shard_hash = save_shard(args.output, arrays)
    hold_array = np.asarray(holds, dtype=np.float64)
    generalized_total = (
        config.waiting_time_value * passenger_summary["accrued_wait_seconds_total"]
        + config.in_vehicle_time_value
        * passenger_summary["accrued_in_vehicle_seconds_total"]
    )
    departed = passenger_summary["departed_passengers"]
    source_paths = set(
        collection_source_paths(
            ROOT, args.sumo_dir, args.edge_xml, contract_version="v5"
        )
    )
    source_paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "bus_h2o" / "passenger_time.py",
            ROOT / "bus_h2o" / "mdp_contract_v5.py",
            ROOT / "SimpleSAC" / "transit_v5" / "evaluation.py",
        }
    )
    summary = {
        "schema_version": "transit-smdp-v5-shard-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_signature": args.job_signature,
        "shard": str(args.output),
        "shard_sha256": shard_hash,
        "contract_schema": SCHEMA_VERSION,
        "contract": config.to_dict(),
        "raw_state_names": list(RAW_EVENT_STATE_NAMES),
        "split": args.split,
        "diagnostic": bool(diagnostic_conditions),
        "training_eligible": not diagnostic_conditions,
        "policy_name": args.policy_name,
        "policy_seed": args.policy_seed,
        "policy": policy_metadata,
        "sumo_seed": args.sumo_seed,
        "scenario": scenario,
        "episode": {
            "configured_seconds": args.episode_seconds,
            "observed_end_time_seconds": observed_end_time,
            "max_control_decisions": args.max_events,
            "control_decisions": control_decisions,
            "terminal_endpoints": terminal_endpoints,
            "completed_trip_contexts": completed_trip_contexts,
            "completion_closures": completion_closures,
            "transitions": n,
            "terminal_transitions": int(arrays["terminals"].sum()),
            "horizon_closures": horizon_closures,
            "station_wraparounds": station_wraparounds,
        },
        "actions": {
            "exact_zero_rate": float(np.mean(hold_array == 0.0)),
            "mean_hold_seconds": float(hold_array.mean()),
            "p50_hold_seconds": float(np.percentile(hold_array, 50)),
            "p95_hold_seconds": float(np.percentile(hold_array, 95)),
            "max_hold_seconds": float(hold_array.max()),
        },
        "training_reward": {
            "semantics": (
                "system-wide generalized passenger time accrued between two "
                "decision epochs of the same bus"
            ),
            "aggregation_warning": (
                "agent-centric intervals overlap; summing transition costs across "
                "agents does not equal episode passenger time"
            ),
            "reward": numeric_summary(arrays["rewards"]),
            "wait_passenger_seconds": numeric_summary(
                arrays["wait_passenger_seconds"]
            ),
            "in_vehicle_passenger_seconds": numeric_summary(
                arrays["in_vehicle_passenger_seconds"]
            ),
            "max_reward_reconstruction_error": reward_error,
        },
        "passengers": passenger_summary,
        "passenger_welfare": {
            "waiting_time_value": config.waiting_time_value,
            "in_vehicle_time_value": config.in_vehicle_time_value,
            "accrued_generalized_passenger_seconds": generalized_total,
            "accrued_generalized_seconds_per_departed_passenger": (
                generalized_total / departed if departed else None
            ),
        },
        "ledger_audit": {
            "final_snapshot": passenger_time_end.to_dict(),
            "count_source_audit": passenger_count_audit,
            "legacy_sampled_trip_histories_used_as_ground_truth": False,
        },
        "service": service_summary,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "sumo_backend": backend,
            "sumo_version": sumo_version(),
        },
        "source_sha256": hash_files(tuple(sorted(source_paths))),
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
