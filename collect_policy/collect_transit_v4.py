#!/usr/bin/env python3
"""Collect one provenance-locked SUMO shard for transit-smdp-v4."""

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
from bus_h2o.mdp_contract_v4 import (
    DEFAULT_CONFIG_V4,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v4,
    hold_seconds_to_policy_action_v4,
    policy_action_to_hold_seconds_v4,
    transition_cost_components_v4,
    transition_duration_seconds_v4,
    transition_reward_v4,
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
    summarize_passengers,
    summarize_service,
)
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4,
    hurdle_random_actions_v4,
    myopic_welfare_hold_seconds_v4,
)


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
    parser.add_argument("--max-events", type=int, default=0)
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


def load_scenario(path: Path) -> tuple[dict, dict]:
    return load_validated_scenario(path)


def raw_observation(env: AllLinesSumoEnv, bus_id) -> np.ndarray:
    observation = env.state[bus_id][-1]
    while isinstance(observation, list) and len(observation) == 1:
        observation = observation[0]
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    if array.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"SUMO emitted raw v4 shape {array.shape}; expected ({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(array).all():
        raise ValueError("SUMO emitted a non-finite v4 event")
    return array


def build_policy(args: argparse.Namespace):
    config = DEFAULT_CONFIG_V4
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
                raw, args.headway_gain, config
            )
            return np.asarray(
                hold_seconds_to_policy_action_v4(holds, config), dtype=np.float32
            )[:, None]

        return policy, {
            "type": "schedule-adjusted-headway-balance",
            "gain": args.headway_gain,
            "hold_seconds": "clip(gain*((H_b-T_b)-(H_f-T_f))/2, 0, 60)",
        }

    def policy(raw):
        holds = myopic_welfare_hold_seconds_v4(raw, config)
        return np.asarray(
            hold_seconds_to_policy_action_v4(holds, config), dtype=np.float32
        )[:, None]

    return policy, {
        "type": "closed-form-myopic-generalized-passenger-time",
        "contract": config.to_dict(),
    }


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
    args.output = args.output.expanduser().resolve()
    summary_path = args.output.with_suffix(args.output.suffix + ".json")
    if not args.force and (args.output.exists() or summary_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing shard {args.output}")

    scenario_runtime, scenario = load_scenario(args.scenario_manifest)
    policy, policy_metadata = build_policy(args)
    config = DEFAULT_CONFIG_V4
    env = AllLinesSumoEnv(
        str(args.sumo_dir.expanduser().resolve()),
        str(args.edge_xml.expanduser().resolve()),
        episode_seconds=args.episode_seconds,
        sumo_seed=args.sumo_seed,
        sumo_cfg=scenario_runtime["sumo_cfg"],
        passenger_obj_xml=scenario_runtime["passenger_obj_xml"],
        observation_schema="v4",
    )

    rows = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "discounts": [],
        "terminals": [],
        "delta_t_seconds": [],
        "trajectory_local_id": [],
    }
    pending = {}
    trajectory_ids = {}
    events = 0
    station_wraparounds = 0
    holds = []
    raw_events = []
    started = time.perf_counter()
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            current_events = list(env._current_events)
            if not current_events:
                _, _, done = env.step_to_event({})
                if done:
                    break
                continue
            raw_batch = np.stack(
                [raw_observation(env, event.bus_id) for event in current_events]
            )
            action_batch = np.asarray(policy(raw_batch), dtype=np.float32)
            if action_batch.shape != (len(current_events), 1):
                raise ValueError(
                    f"behavior policy returned {action_batch.shape}; "
                    f"expected {(len(current_events), 1)}"
                )
            if not np.isfinite(action_batch).all() or np.any(
                np.abs(action_batch) > 1.00001
            ):
                raise ValueError("behavior policy emitted an invalid action")

            env_actions = {}
            for event, raw, action in zip(current_events, raw_batch, action_batch):
                bus_id = str(event.bus_id)
                events += 1
                trajectory_id = trajectory_ids.setdefault(bus_id, len(trajectory_ids))
                station_index = int(
                    round(float(raw[RAW_EVENT_STATE_INDEX["station_index"]]))
                )
                previous = pending.pop(bus_id, None)
                if previous is not None and station_index > previous["station_index"]:
                    duration = float(
                        transition_duration_seconds_v4(previous["raw"], raw)
                    )
                    reward = float(
                        transition_reward_v4(
                            previous["raw"], previous["action"], raw, config
                        )
                    )
                    rows["observations"].append(previous["raw"])
                    rows["actions"].append(previous["action"])
                    rows["rewards"].append(reward)
                    rows["next_observations"].append(raw.copy())
                    rows["discounts"].append(
                        continuous_time_discount_v4(duration, config)
                    )
                    rows["terminals"].append(0)
                    rows["delta_t_seconds"].append(duration)
                    rows["trajectory_local_id"].append(trajectory_id)
                elif previous is not None:
                    station_wraparounds += 1

                hold_seconds = float(
                    policy_action_to_hold_seconds_v4(action[0], config)
                )
                holds.append(hold_seconds)
                raw_events.append(raw.copy())
                pending[bus_id] = {
                    "raw": raw.copy(),
                    "action": action.copy(),
                    "station_index": station_index,
                }
                env_actions[event.bus_id] = [hold_seconds, 1.0]

            _, _, done = env.step_to_event(env_actions)
            if done:
                break
            if args.max_events and events >= args.max_events:
                break
        bridge = env._bridge
        observed_end_time = float(bridge.current_time)
        backend = bridge.sumo_binary
        passenger_summary = summarize_passengers(bridge, observed_end_time)
        service_summary = summarize_service(bridge)
    finally:
        env.close()

    n = len(rows["rewards"])
    if n == 0:
        raise RuntimeError("SUMO shard contains no same-bus transitions")
    arrays = {
        "observations": np.asarray(rows["observations"], dtype=np.float32).reshape(
            n, RAW_EVENT_STATE_DIM
        ),
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, 1),
        "rewards": np.asarray(rows["rewards"], dtype=np.float32),
        "next_observations": np.asarray(
            rows["next_observations"], dtype=np.float32
        ).reshape(n, RAW_EVENT_STATE_DIM),
        "discounts": np.asarray(rows["discounts"], dtype=np.float32),
        "terminals": np.asarray(rows["terminals"], dtype=np.uint8),
        "delta_t_seconds": np.asarray(rows["delta_t_seconds"], dtype=np.float32),
        "trajectory_local_id": np.asarray(
            rows["trajectory_local_id"], dtype=np.int64
        ),
    }
    shard_hash = save_shard(args.output, arrays)
    hold_array = np.asarray(holds, dtype=np.float64)
    raw_event_array = np.asarray(raw_events, dtype=np.float64).reshape(
        len(raw_events), RAW_EVENT_STATE_DIM
    )
    costs = transition_cost_components_v4(
        arrays["observations"],
        arrays["actions"],
        arrays["next_observations"],
        config,
    )
    forward_present = (
        raw_event_array[:, RAW_EVENT_STATE_INDEX["forward_bus_present"]] > 0.5
    )
    backward_present = (
        raw_event_array[:, RAW_EVENT_STATE_INDEX["backward_bus_present"]] > 0.5
    )
    forward_ratio = (
        raw_event_array[:, RAW_EVENT_STATE_INDEX["forward_headway_seconds"]]
        / np.maximum(
            raw_event_array[
                :, RAW_EVENT_STATE_INDEX["target_forward_headway_seconds"]
            ],
            1.0,
        )
    )
    backward_ratio = (
        raw_event_array[:, RAW_EVENT_STATE_INDEX["backward_headway_seconds"]]
        / np.maximum(
            raw_event_array[
                :, RAW_EVENT_STATE_INDEX["target_backward_headway_seconds"]
            ],
            1.0,
        )
    )
    valid_forward = forward_ratio[forward_present]
    valid_backward = backward_ratio[backward_present]
    headway_absolute_error = np.concatenate(
        (np.abs(valid_forward - 1.0), np.abs(valid_backward - 1.0))
    )
    summary = {
        "schema_version": "transit-smdp-v4-shard-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_signature": args.job_signature,
        "shard": str(args.output),
        "shard_sha256": shard_hash,
        "contract_schema": SCHEMA_VERSION,
        "contract": config.to_dict(),
        "raw_state_names": list(RAW_EVENT_STATE_NAMES),
        "split": args.split,
        "policy_name": args.policy_name,
        "policy_seed": args.policy_seed,
        "policy": policy_metadata,
        "sumo_seed": args.sumo_seed,
        "scenario": scenario,
        "episode": {
            "configured_seconds": args.episode_seconds,
            "observed_end_time_seconds": observed_end_time,
            "max_events": args.max_events,
            "events": events,
            "transitions": n,
            "discarded_pending": len(pending),
            "station_wraparounds": station_wraparounds,
        },
        "actions": {
            "exact_zero_rate": float(np.mean(hold_array == 0.0)),
            "mean_hold_seconds": float(hold_array.mean()),
            "p50_hold_seconds": float(np.percentile(hold_array, 50)),
            "p95_hold_seconds": float(np.percentile(hold_array, 95)),
            "max_hold_seconds": float(hold_array.max()),
        },
        "objective": {
            "units": "generalized passenger-minutes",
            "waiting_cost": numeric_summary(costs["waiting_cost"]),
            "holding_cost": numeric_summary(costs["holding_cost"]),
            "total_cost": numeric_summary(costs["total_cost"]),
            "cumulative_waiting_cost": float(np.sum(costs["waiting_cost"])),
            "cumulative_holding_cost": float(np.sum(costs["holding_cost"])),
            "cumulative_total_cost": float(np.sum(costs["total_cost"])),
        },
        "headway": {
            "normalized_absolute_error": numeric_summary(headway_absolute_error),
            "forward_ratio": numeric_summary(valid_forward),
            "backward_ratio": numeric_summary(valid_backward),
            "bunching_rate": (
                float(np.mean(valid_forward < 0.5)) if len(valid_forward) else None
            ),
            "large_gap_rate": (
                float(np.mean(valid_forward > 1.5)) if len(valid_forward) else None
            ),
        },
        "passengers": passenger_summary,
        "passenger_welfare": {
            "waiting_time_value": config.waiting_time_value,
            "in_vehicle_time_value": config.in_vehicle_time_value,
            "accrued_generalized_passenger_seconds": (
                config.waiting_time_value
                * passenger_summary["accrued_wait_seconds_total"]
                + config.in_vehicle_time_value
                * passenger_summary["accrued_in_vehicle_seconds_total"]
            ),
            "accrued_generalized_seconds_per_departed_passenger": (
                (
                    config.waiting_time_value
                    * passenger_summary["accrued_wait_seconds_total"]
                    + config.in_vehicle_time_value
                    * passenger_summary["accrued_in_vehicle_seconds_total"]
                )
                / passenger_summary["departed_passengers"]
                if passenger_summary["departed_passengers"]
                else None
            ),
        },
        "service": service_summary,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "sumo_backend": backend,
            "sumo_version": sumo_version(),
        },
        "source_sha256": hash_files(
            collection_source_paths(ROOT, args.sumo_dir, args.edge_xml)
        ),
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
