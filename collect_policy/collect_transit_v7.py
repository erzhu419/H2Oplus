#!/usr/bin/env python3
"""Collect one chronological, non-overlapping SUMO transit-v7 shard."""

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
from bus_h2o.full_service_trace_v7 import (
    FULL_SERVICE_TRACE_SCHEMA,
    extract_full_service_arrays,
    full_service_trace_path,
    save_full_service_trace,
    write_full_service_summary,
)
from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4
from bus_h2o.mdp_contract_v7 import (
    DEFAULT_CONFIG_V7,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_STATE_DIM,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v7,
    hold_seconds_to_policy_action_v7,
    pack_event_batch_v7,
    passenger_time_snapshot_v5,
    policy_action_to_hold_seconds_v7,
    terminal_raw_state_v7,
    transition_cost_components_v7,
    transition_duration_seconds_v7,
    transition_reward_v7,
)
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    collection_source_paths,
    hash_files,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
)
from SimpleSAC.transit_v3.evaluation import numeric_summary, summarize_service
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4,
    hurdle_random_actions_v4,
    myopic_welfare_hold_seconds_v4,
)
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5


SHARD_SCHEMA = "transit-smdp-v7-shard-v1"


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
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-signature")
    parser.add_argument(
        "--sumo-backend", choices=("libsumo", "traci"), required=True
    )
    parser.add_argument("--expected-sumo-runtime-version", required=True)
    parser.add_argument("--expected-sumo-runtime-signature", required=True)
    parser.add_argument("--sumo-runtime-lock", type=Path, required=True)
    parser.add_argument("--expected-sumo-runtime-lock-sha256", required=True)
    parser.add_argument("--episode-seconds", type=int, default=24_000)
    parser.add_argument("--diagnostic", action="store_true")
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


def raw_event_observation(env: AllLinesSumoEnv, bus_id) -> np.ndarray:
    observation = env.state[bus_id][-1]
    while isinstance(observation, list) and len(observation) == 1:
        observation = observation[0]
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    if array.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"SUMO emitted event width {array.shape}; expected "
            f"({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(array).all():
        raise ValueError("SUMO emitted a non-finite v7 event")
    return array


def is_terminal_event(env: AllLinesSumoEnv, event) -> bool:
    line = env._bridge.line_obj_dic[event.line_id]
    return bool(
        event.stop_id == line.stop_id_l[-1]
        or int(event.stop_idx) >= len(line.stop_id_l) - 1
    )


def event_sort_key(event) -> tuple:
    return (
        float(event.sim_time),
        str(event.line_id),
        str(event.bus_id),
        int(event.stop_idx),
    )


def build_policy(args: argparse.Namespace):
    config = DEFAULT_CONFIG_V7
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
                hold_seconds_to_policy_action_v7(holds, config), dtype=np.float32
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
            hold_seconds_to_policy_action_v7(holds, config), dtype=np.float32
        )[:, None]

    return policy, {
        "type": "closed-form-myopic-headway-surrogate",
        "note": "behavior coverage only; v7 reward is exact chronological welfare",
    }


def append_transition(
    rows: dict[str, list],
    previous: dict,
    raw_next_state: np.ndarray,
    passenger_time_end,
    *,
    terminal: bool,
) -> None:
    config = DEFAULT_CONFIG_V7
    start = previous["passenger_time"]
    duration = float(passenger_time_end.time_seconds - start.time_seconds)
    state_duration = float(
        transition_duration_seconds_v7(previous["raw"], raw_next_state)
    )
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError(f"v7 transition has invalid duration {duration}")
    if state_duration != duration:
        raise RuntimeError(
            "v7 passenger-time and packed-state clocks disagree: "
            f"{duration} versus {state_duration}"
        )
    components = transition_cost_components_v7(start, passenger_time_end, config)
    rows["observations"].append(previous["raw"])
    rows["actions"].append(previous["action"])
    rows["rewards"].append(
        transition_reward_v7(start, passenger_time_end, config)
    )
    rows["next_observations"].append(raw_next_state.copy())
    rows["discounts"].append(
        0.0 if terminal else continuous_time_discount_v7(duration, config)
    )
    rows["terminals"].append(int(terminal))
    rows["delta_t_seconds"].append(duration)
    rows["trajectory_local_id"].append(0)
    rows["wait_passenger_seconds"].append(
        components["wait_passenger_seconds"]
    )
    rows["in_vehicle_passenger_seconds"].append(
        components["in_vehicle_passenger_seconds"]
    )
    rows["event_bus_id"].append(previous["event_bus_id"])
    rows["event_line_index"].append(previous["event_line_index"])
    rows["event_station_index"].append(previous["event_station_index"])
    rows["event_decision_time_seconds"].append(
        previous["event_decision_time_seconds"]
    )
    rows["event_base_dwell_seconds"].append(
        previous["event_base_dwell_seconds"]
    )


def save_shard(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return sha256(path)


def main() -> None:
    args = parse_args()
    if (
        args.sumo_backend != FORMAL_V7_SUMO_BACKEND
        or args.expected_sumo_runtime_version != FORMAL_V7_SUMO_VERSION
    ):
        raise ValueError(
            "v7 collection requires the formal libsumo 1.25.0 runtime"
        )
    runtime_lock, runtime_lock_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_sumo_runtime_lock_sha256,
    )
    runtime_fingerprint = sumo_runtime_fingerprint(
        args.sumo_backend,
        expected_version=args.expected_sumo_runtime_version,
    )
    if (
        runtime_fingerprint["portable_signature"]
        != args.expected_sumo_runtime_signature
        or runtime_lock_identity["runtime_signature"]
        != args.expected_sumo_runtime_signature
        or runtime_lock["runtime_fingerprint"]["portable_identity"]
        != runtime_fingerprint["portable_identity"]
    ):
        raise RuntimeError("v7 collection runtime differs from its formal lock")
    if args.episode_seconds <= 0 or args.max_events < 0:
        raise ValueError("episode/max-event limits are invalid")
    diagnostic_conditions = (
        args.episode_seconds != int(DEFAULT_CONFIG_V7.horizon_seconds)
        or args.max_events > 0
    )
    if diagnostic_conditions != bool(args.diagnostic):
        raise ValueError(
            "--diagnostic must be present exactly for a shortened horizon/event cap"
        )
    args.output = args.output.expanduser().resolve()
    summary_path = args.output.with_suffix(args.output.suffix + ".json")
    if not args.force and (args.output.exists() or summary_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing shard {args.output}")

    scenario_runtime, scenario = load_validated_scenario(args.scenario_manifest)
    policy, policy_metadata = build_policy(args)
    config = DEFAULT_CONFIG_V7
    env = AllLinesSumoEnv(
        str(args.sumo_dir.expanduser().resolve()),
        str(args.edge_xml.expanduser().resolve()),
        episode_seconds=args.episode_seconds,
        sumo_seed=args.sumo_seed,
        sumo_cfg=scenario_runtime["sumo_cfg"],
        passenger_obj_xml=scenario_runtime["passenger_obj_xml"],
        observation_schema="v5",
        sumo_backend=args.sumo_backend,
    )
    line_index_to_id = [None] * len(env._line_idx_map)
    for line_id, line_index in env._line_idx_map.items():
        if (
            line_index < 0
            or line_index >= len(line_index_to_id)
            or line_index_to_id[line_index] is not None
        ):
            raise RuntimeError("SUMO line-index mapping is not contiguous")
        line_index_to_id[line_index] = str(line_id)
    if any(value is None for value in line_index_to_id):
        raise RuntimeError("SUMO line-index mapping contains a gap")
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
            "event_bus_id",
            "event_line_index",
            "event_station_index",
            "event_decision_time_seconds",
            "event_base_dwell_seconds",
        )
    }
    previous = None
    first_passenger_time = None
    control_decisions = 0
    terminal_endpoints = 0
    completed_trip_contexts = 0
    simultaneous_batches = 0
    simultaneous_batch_sizes = []
    holds = []
    full_service_arrays = None
    full_service_diagnostics = None
    stopped_by_event_limit = False
    started = time.perf_counter()
    try:
        env.reset()
        env.step_to_event({})
        completed_trip_contexts += len(
            env._bridge.consume_completed_trip_passenger_times()
        )
        while not env.done:
            events = sorted(env._current_events, key=event_sort_key)
            if not events:
                _, _, done = env.step_to_event({})
                completed_trip_contexts += len(
                    env._bridge.consume_completed_trip_passenger_times()
                )
                if done:
                    break
                continue

            event_rows = []
            for event in events:
                raw = raw_event_observation(env, event.bus_id)
                passenger_time = passenger_time_snapshot_v5(
                    event.metadata["passenger_time_at_decision"]
                )
                event_rows.append(
                    (event, raw, passenger_time, is_terminal_event(env, event))
                )
            controls = [row for row in event_rows if not row[3]]
            env_actions = {
                event.bus_id: [0.0, 1.0]
                for event, _, _, terminal in event_rows
                if terminal
            }
            terminal_endpoints += len(event_rows) - len(controls)

            if controls:
                local_batch = np.stack([row[1] for row in controls])
                snapshots = [row[2] for row in controls]
                reference = snapshots[0]
                if any(snapshot != reference for snapshot in snapshots[1:]):
                    raise RuntimeError(
                        "simultaneous SUMO events disagree on passenger-time state"
                    )
                action_batch = np.asarray(policy(local_batch), dtype=np.float32)
                if action_batch.shape != (len(controls), 1):
                    raise ValueError(
                        f"behavior policy returned {action_batch.shape}; expected "
                        f"{(len(controls), 1)}"
                    )
                if not np.isfinite(action_batch).all() or np.any(
                    np.abs(action_batch) > 1.00001
                ):
                    raise ValueError("behavior policy emitted an invalid action")

                simultaneous_batches += 1
                simultaneous_batch_sizes.append(len(controls))
                prior_actions = np.full(len(controls), -1.0, dtype=np.float32)
                for index, ((event, _, passenger_time, _), action) in enumerate(
                    zip(controls, action_batch)
                ):
                    raw_state = pack_event_batch_v7(
                        local_batch, index, prior_actions, config
                    )
                    if previous is not None:
                        append_transition(
                            rows,
                            previous,
                            raw_state,
                            passenger_time,
                            terminal=False,
                        )
                    else:
                        first_passenger_time = passenger_time

                    hold_seconds = float(
                        policy_action_to_hold_seconds_v7(action[0], config)
                    )
                    holds.append(hold_seconds)
                    control_decisions += 1
                    previous = {
                        "raw": raw_state.copy(),
                        "action": action.copy(),
                        "passenger_time": passenger_time,
                        "event_bus_id": str(event.bus_id),
                        "event_line_index": int(
                            round(
                                float(
                                    local_batch[
                                        index, RAW_EVENT_STATE_INDEX["line_id"]
                                    ]
                                )
                            )
                        ),
                        "event_station_index": int(
                            round(
                                float(
                                    local_batch[
                                        index,
                                        RAW_EVENT_STATE_INDEX["station_index"],
                                    ]
                                )
                            )
                        ),
                        "event_decision_time_seconds": float(
                            local_batch[
                                index,
                                RAW_EVENT_STATE_INDEX["sim_time_seconds"],
                            ]
                        ),
                        "event_base_dwell_seconds": float(
                            local_batch[
                                index,
                                RAW_EVENT_STATE_INDEX["base_dwell_seconds"],
                            ]
                        ),
                    }
                    prior_actions[index] = action[0]
                    env_actions[event.bus_id] = [hold_seconds, 1.0]

            _, _, done = env.step_to_event(env_actions)
            completed_trip_contexts += len(
                env._bridge.consume_completed_trip_passenger_times()
            )
            if done:
                break
            if args.max_events and control_decisions >= args.max_events:
                stopped_by_event_limit = True
                break

        bridge = env._bridge
        if bridge.sumo_backend != args.sumo_backend:
            raise RuntimeError("SUMO collection started the wrong backend")
        observed_end_time = float(bridge.current_time)
        passenger_count_audit = bridge.audit_passenger_time_counts()
        service_semantics_audit = bridge.audit_bus_service_semantics(
            require_all_departed_trips_completed=not diagnostic_conditions
        )
        if not service_semantics_audit["passed"]:
            raise RuntimeError(
                "SUMO v7 bus-service semantics audit failed: "
                f"checks={service_semantics_audit['checks']}, "
                f"violations={service_semantics_audit['violations']}"
            )
        passenger_time_end = bridge.passenger_time_ledger.snapshot()
        if previous is None or first_passenger_time is None:
            raise RuntimeError("SUMO v7 collection produced no controllable event")
        append_transition(
            rows,
            previous,
            terminal_raw_state_v7(previous["raw"], passenger_time_end),
            passenger_time_end,
            terminal=True,
        )
        backend = bridge.sumo_backend
        passenger_summary = summarize_passengers_v5(bridge, passenger_time_end)
        service_summary = summarize_service(bridge)
        if args.policy == "zero" and not diagnostic_conditions:
            full_service_arrays, full_service_diagnostics = (
                extract_full_service_arrays(bridge, line_index_to_id)
            )
    finally:
        env.close()

    if passenger_summary["active_state_count_balance_error"] < 0:
        raise RuntimeError(
            "SUMO passenger-state counts exceed the number of unfinished passengers"
        )
    if not diagnostic_conditions and service_summary["completed_trips"] != service_summary[
        "departed_trips"
    ]:
        raise RuntimeError("not every departed SUMO trip completed in the full episode")

    n = len(rows["rewards"])
    event_bus_ids = sorted(set(rows["event_bus_id"]))
    event_bus_to_code = {
        bus_id: index for index, bus_id in enumerate(event_bus_ids)
    }
    arrays = {
        "observations": np.asarray(rows["observations"], dtype=np.float32).reshape(
            n, RAW_STATE_DIM
        ),
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, 1),
        "rewards": np.asarray(rows["rewards"], dtype=np.float64),
        "next_observations": np.asarray(
            rows["next_observations"], dtype=np.float32
        ).reshape(n, RAW_STATE_DIM),
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
        "event_bus_code": np.asarray(
            [event_bus_to_code[value] for value in rows["event_bus_id"]],
            dtype=np.int32,
        ),
        "event_line_index": np.asarray(
            rows["event_line_index"], dtype=np.int16
        ),
        "event_station_index": np.asarray(
            rows["event_station_index"], dtype=np.int16
        ),
        "event_decision_time_seconds": np.asarray(
            rows["event_decision_time_seconds"], dtype=np.float64
        ),
        "event_base_dwell_seconds": np.asarray(
            rows["event_base_dwell_seconds"], dtype=np.float64
        ),
    }
    reward_reconstruction = -(
        config.waiting_time_value * arrays["wait_passenger_seconds"]
        + config.in_vehicle_time_value * arrays["in_vehicle_passenger_seconds"]
    ) / config.reward_unit_passenger_seconds
    reward_error = float(np.max(np.abs(arrays["rewards"] - reward_reconstruction)))
    covered_wait = (
        passenger_time_end.accrued_wait_seconds
        - first_passenger_time.accrued_wait_seconds
    )
    covered_in_vehicle = (
        passenger_time_end.accrued_in_vehicle_seconds
        - first_passenger_time.accrued_in_vehicle_seconds
    )
    coverage_errors = {
        "wait_passenger_seconds": float(
            arrays["wait_passenger_seconds"].sum() - covered_wait
        ),
        "in_vehicle_passenger_seconds": float(
            arrays["in_vehicle_passenger_seconds"].sum() - covered_in_vehicle
        ),
        "reward": float(
            arrays["rewards"].sum()
            + (
                config.waiting_time_value * covered_wait
                + config.in_vehicle_time_value * covered_in_vehicle
            )
            / config.reward_unit_passenger_seconds
        ),
    }
    if reward_error > 1e-12 or any(
        abs(value) > 1e-9 for value in coverage_errors.values()
    ):
        raise RuntimeError(
            f"v7 chronological welfare identity failed: reward={reward_error}, "
            f"coverage={coverage_errors}"
        )
    if int(arrays["terminals"].sum()) != 1:
        raise RuntimeError("a v7 episode must contain exactly one terminal row")

    shard_hash = save_shard(args.output, arrays)
    hold_array = np.asarray(holds, dtype=np.float64)
    generalized_total = (
        config.waiting_time_value * passenger_time_end.accrued_wait_seconds
        + config.in_vehicle_time_value
        * passenger_time_end.accrued_in_vehicle_seconds
    )
    covered_generalized = (
        config.waiting_time_value * covered_wait
        + config.in_vehicle_time_value * covered_in_vehicle
    )
    departed = passenger_summary["departed_passengers"]
    source_paths = set(
        collection_source_paths(
            ROOT, args.sumo_dir, args.edge_xml, contract_version="v7"
        )
    )
    source_paths.add(ROOT / "SimpleSAC" / "transit_v5" / "evaluation.py")
    source_sha256 = hash_files(tuple(sorted(source_paths)))
    full_service_descriptor = None
    if full_service_arrays is not None:
        trace_path = full_service_trace_path(args.output)
        trace_hash = save_full_service_trace(trace_path, full_service_arrays)
        trace_summary_path = trace_path.with_suffix(trace_path.suffix + ".json")
        trace_summary = {
            "schema_version": FULL_SERVICE_TRACE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "trace": str(trace_path),
            "trace_sha256": trace_hash,
            "study_stage": "formal",
            "policy_name": args.policy_name,
            "policy_seed": args.policy_seed,
            "exact_zero_hold": True,
            "sumo_seed": args.sumo_seed,
            "sumo_version": runtime_fingerprint["runtime_version"],
            "sumo_runtime_signature": runtime_fingerprint[
                "portable_signature"
            ],
            "sumo_runtime_lock": runtime_lock_identity,
            "scenario": scenario,
            "source_shard": {
                "path": str(args.output),
                "sha256": shard_hash,
                "job_signature": args.job_signature,
            },
            "collection_source_sha256": source_sha256,
            "bus_service_semantics_audit": service_semantics_audit,
            "full_service_trace": full_service_diagnostics,
        }
        write_full_service_summary(trace_summary_path, trace_summary)
        full_service_descriptor = {
            "schema_version": FULL_SERVICE_TRACE_SCHEMA,
            "path": str(trace_path),
            "sha256": trace_hash,
            "summary_path": str(trace_summary_path),
            "summary_sha256": sha256(trace_summary_path),
        }
    summary = {
        "schema_version": SHARD_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_signature": args.job_signature,
        "shard": str(args.output),
        "shard_sha256": shard_hash,
        "contract_schema": SCHEMA_VERSION,
        "contract": config.to_dict(),
        "raw_state_names": list(RAW_STATE_NAMES),
        "event_identity": {
            "bus_code_to_id": event_bus_ids,
            "line_index_to_id": line_index_to_id,
            "semantics": (
                "audit/calibration metadata for the controlled event; excluded "
                "from policy observations"
            ),
        },
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
            "transitions": n,
            "terminal_transitions": int(arrays["terminals"].sum()),
            "terminal_endpoints_excluded_from_control": terminal_endpoints,
            "completed_trip_contexts_observed": completed_trip_contexts,
            "simultaneous_batches": simultaneous_batches,
            "simultaneous_batch_size": numeric_summary(
                simultaneous_batch_sizes
            ),
            "zero_duration_transitions": int(
                np.sum(arrays["delta_t_seconds"] == 0.0)
            ),
            "stopped_by_event_limit": stopped_by_event_limit,
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
                "non-overlapping system generalized passenger time between "
                "chronologically consecutive centralized decision tokens"
            ),
            "simultaneous_transition_semantics": (
                "deterministically ordered AEC tokens; intermediate rows have "
                "zero duration, zero reward, and unit discount"
            ),
            "pre_first_decision_cost_excluded_as_policy_invariant": True,
            "first_decision_time_seconds": first_passenger_time.time_seconds,
            "covered_generalized_passenger_seconds": covered_generalized,
            "episode_coverage_fraction": (
                covered_generalized / generalized_total if generalized_total else None
            ),
            "reward_sum": float(arrays["rewards"].sum()),
            "reward": numeric_summary(arrays["rewards"]),
            "wait_passenger_seconds": numeric_summary(
                arrays["wait_passenger_seconds"]
            ),
            "in_vehicle_passenger_seconds": numeric_summary(
                arrays["in_vehicle_passenger_seconds"]
            ),
            "max_reward_reconstruction_error": reward_error,
            "chronological_coverage_errors": coverage_errors,
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
        "bus_service_semantics_audit": service_semantics_audit,
        "full_service_trace": full_service_descriptor,
        "service": service_summary,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "sumo_backend": backend,
            "sumo_version": runtime_fingerprint["runtime_version"],
            "sumo_runtime_signature": runtime_fingerprint[
                "portable_signature"
            ],
            "sumo_runtime_lock": runtime_lock_identity,
            "sumo_runtime_fingerprint": runtime_fingerprint,
        },
        "source_sha256": source_sha256,
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
