#!/usr/bin/env python3
"""Evaluate one transit-smdp-v5 policy in one provenance-locked SUMO episode."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4
from bus_h2o.mdp_contract_v5 import (
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    TransitMDPConfigV5,
    continuous_time_discount_v5,
    hold_seconds_to_policy_action_v5,
    passenger_time_snapshot_v5,
    policy_action_to_hold_seconds_v5,
    raw_event_to_canonical_v5,
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
    _headway_summary,
    numeric_summary,
    summarize_service,
)
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4,
    myopic_welfare_hold_seconds_v4,
)
from SimpleSAC.transit_v5.trainer import load_actor_checkpoint_v5
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--zero-hold", action="store_true")
    policy.add_argument("--headway-balance-gain", type=float)
    policy.add_argument("--myopic-welfare", action="store_true")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--sumo-seed", type=int, required=True)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--job-signature")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--max-sim-time-seconds",
        type=int,
        default=0,
        help="diagnostic horizon; zero evaluates the complete contract episode",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "bus_h2o" / "datasets_v5" / "transit_smdp_v5.h5",
    )
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_dataset_contract(
    path: Path,
) -> tuple[TransitMDPConfigV5, np.ndarray, str]:
    resolved = path.expanduser().resolve()
    dataset_hash = sha256(resolved)
    with h5py.File(resolved, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("evaluation dataset does not implement transit-smdp-v5")
        config = TransitMDPConfigV5(**json.loads(handle.attrs["config"]))
        station_max = np.asarray(
            json.loads(handle.attrs["station_max_by_line"]), dtype=np.float32
        )
    if station_max.shape != (12,) or np.any(station_max <= 0.0):
        raise ValueError("evaluation dataset has invalid station normalization")
    return config, station_max, dataset_hash


def raw_observation(env: AllLinesSumoEnv, bus_id) -> np.ndarray:
    observation = env.state[bus_id][-1]
    while isinstance(observation, list) and len(observation) == 1:
        observation = observation[0]
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    if values.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"SUMO emitted raw v5 shape {values.shape}; "
            f"expected ({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(values).all():
        raise ValueError("SUMO emitted a non-finite v5 event")
    return values


def build_policy(
    args: argparse.Namespace,
    config: TransitMDPConfigV5,
    station_max: np.ndarray,
    dataset_hash: str,
):
    if args.zero_hold:
        return (
            lambda raw: np.full((len(raw), 1), -1.0, dtype=np.float32),
            {"type": "zero", "exact_zero_probability": 1.0},
        )
    if args.headway_balance_gain is not None:
        gain = float(args.headway_balance_gain)
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("headway-balance gain must be finite and non-negative")

        def headway_policy(raw: np.ndarray) -> np.ndarray:
            holds = headway_balance_hold_seconds_v4(
                raw[:, :RAW_EVENT_STATE_DIM_V4], gain, config
            )
            actions = hold_seconds_to_policy_action_v5(holds, config)
            return np.asarray(actions, dtype=np.float32)[:, None]

        return headway_policy, {
            "type": "schedule-adjusted-headway-balance",
            "gain": gain,
            "hold_seconds": "clip(gain*((H_b-T_b)-(H_f-T_f))/2, 0, 60)",
        }
    if args.myopic_welfare:

        def myopic_policy(raw: np.ndarray) -> np.ndarray:
            holds = myopic_welfare_hold_seconds_v4(
                raw[:, :RAW_EVENT_STATE_DIM_V4], config
            )
            actions = hold_seconds_to_policy_action_v5(holds, config)
            return np.asarray(actions, dtype=np.float32)[:, None]

        return myopic_policy, {
            "type": "closed-form-myopic-headway-surrogate",
            "note": "evaluation baseline; primary welfare is the exact v5 ledger",
        }

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_hash = sha256(checkpoint)
    actor, extra = load_actor_checkpoint_v5(checkpoint, args.device)
    expected = {
        "dataset_sha256": dataset_hash,
        "contract_config": config.to_dict(),
        "mdp_version": "v5",
    }
    mismatches = {
        name: {"checkpoint": extra.get(name), "evaluation": value}
        for name, value in expected.items()
        if extra.get(name) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint/evaluation provenance mismatch: {mismatches}")

    def checkpoint_policy(raw: np.ndarray) -> np.ndarray:
        canonical = raw_event_to_canonical_v5(raw, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical, dtype=torch.float32, device=args.device
            )
            actions, _ = actor(tensor, deterministic=True)
        return actions.cpu().numpy()

    return checkpoint_policy, {
        "type": "checkpoint",
        "path": str(checkpoint),
        "sha256": checkpoint_hash,
        "training_metadata": extra,
    }


def _is_terminal_event(env: AllLinesSumoEnv, event) -> bool:
    line = env._bridge.line_obj_dic[event.line_id]
    return bool(
        event.stop_id == line.stop_id_l[-1]
        or int(event.stop_idx) >= len(line.stop_id_l) - 1
    )


def _event_record(event, raw: np.ndarray, action: np.ndarray, hold: float) -> dict:
    target_forward = max(
        float(raw[RAW_EVENT_STATE_INDEX["target_forward_headway_seconds"]]), 1.0
    )
    target_backward = max(
        float(raw[RAW_EVENT_STATE_INDEX["target_backward_headway_seconds"]]), 1.0
    )
    forward_present = bool(
        raw[RAW_EVENT_STATE_INDEX["forward_bus_present"]] > 0.5
    )
    backward_present = bool(
        raw[RAW_EVENT_STATE_INDEX["backward_bus_present"]] > 0.5
    )
    return {
        "sim_time": float(event.sim_time),
        "line_id": str(event.line_id),
        "bus_id": str(event.bus_id),
        "station_index": int(
            round(float(raw[RAW_EVENT_STATE_INDEX["station_index"]]))
        ),
        "scheduled_headway_seconds": 0.5 * (target_forward + target_backward),
        "forward_ratio": (
            float(raw[RAW_EVENT_STATE_INDEX["forward_headway_seconds"]])
            / target_forward
            if forward_present
            else None
        ),
        "backward_ratio": (
            float(raw[RAW_EVENT_STATE_INDEX["backward_headway_seconds"]])
            / target_backward
            if backward_present
            else None
        ),
        "action": action.copy(),
        "hold_seconds": hold,
    }


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


def _transition_record(
    previous: dict,
    raw_next: np.ndarray,
    passenger_time_end,
    config: TransitMDPConfigV5,
    *,
    terminal: bool,
) -> dict:
    end = passenger_time_snapshot_v5(passenger_time_end)
    start = previous["passenger_time"]
    duration = float(end.time_seconds - start.time_seconds)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"evaluation transition has invalid duration {duration}")
    components = transition_cost_components_v5(start, end, config)
    return {
        "line_id": previous["line_id"],
        "raw_observation": previous["raw"],
        "action": previous["action"],
        "raw_next_observation": raw_next.copy(),
        "reward": transition_reward_v5(start, end, config),
        "delta_t_seconds": duration,
        "discount": (
            0.0
            if terminal
            else float(continuous_time_discount_v5(duration, config))
        ),
        "terminal": int(terminal),
        "wait_passenger_seconds": components["wait_passenger_seconds"],
        "in_vehicle_passenger_seconds": components[
            "in_vehicle_passenger_seconds"
        ],
    }


def _close_completed_trip_transitions(
    bridge,
    pending: dict,
    transition_records: list[dict],
    config: TransitMDPConfigV5,
) -> tuple[int, int]:
    """Close pending rows at each trip's exact SUMO completion instant."""

    contexts = bridge.consume_completed_trip_passenger_times()
    closures = 0
    for bus_id, passenger_time_end in contexts.items():
        previous = pending.pop(str(bus_id), None)
        if previous is None:
            continue
        transition_records.append(
            _transition_record(
                previous,
                _horizon_observation(previous["raw"], passenger_time_end),
                passenger_time_end,
                config,
                terminal=True,
            )
        )
        closures += 1
    return len(contexts), closures


def summarize_operations_v5(
    event_records: list[dict], transition_records: list[dict]
) -> dict:
    by_line: dict[str, list[dict]] = defaultdict(list)
    for row in event_records:
        by_line[row["line_id"]].append(row)
    per_line = {}
    for line_id, rows in sorted(by_line.items()):
        holds = [row["hold_seconds"] for row in rows]
        per_line[line_id] = {
            "decision_events": len(rows),
            "headway": _headway_summary(rows),
            "hold_seconds": numeric_summary(holds),
            "total_hold_seconds": float(np.sum(holds)),
        }
    holds = [row["hold_seconds"] for row in event_records]
    return {
        "decision_events": len(event_records),
        "same_bus_transitions": len(transition_records),
        "terminal_transitions": int(
            sum(row["terminal"] for row in transition_records)
        ),
        "transition_reward": numeric_summary(
            row["reward"] for row in transition_records
        ),
        "reward_aggregation_warning": (
            "same-bus intervals overlap across agents; transition rewards are "
            "diagnostics and are not summed"
        ),
        "delta_t_seconds": numeric_summary(
            row["delta_t_seconds"] for row in transition_records
        ),
        "discount": numeric_summary(row["discount"] for row in transition_records),
        "headway": _headway_summary(event_records),
        "hold_seconds": numeric_summary(holds),
        "total_hold_seconds": float(np.sum(holds)),
        "nonzero_hold_rate": (
            float((np.asarray(holds) > 1e-6).mean()) if holds else None
        ),
        "per_line": per_line,
    }


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(
        collection_source_paths(
            ROOT, sumo_dir, edge_xml, contract_version="v5"
        )
    )
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "SimpleSAC" / "transit_v3" / "domain_ratio.py",
            ROOT / "SimpleSAC" / "transit_v3" / "networks.py",
            ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
            ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
            ROOT / "SimpleSAC" / "transit_v5" / "networks.py",
            ROOT / "SimpleSAC" / "transit_v5" / "trainer.py",
            ROOT / "SimpleSAC" / "transit_v5" / "evaluation.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def save_event_log(
    path: Path,
    raw_events: list[np.ndarray],
    event_records: list[dict],
    transition_records: list[dict],
) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            raw_events=np.asarray(raw_events, dtype=np.float32).reshape(
                len(raw_events), RAW_EVENT_STATE_DIM
            ),
            event_actions=np.asarray(
                [row["action"] for row in event_records], dtype=np.float32
            ).reshape(len(event_records), 1),
            event_hold_seconds=np.asarray(
                [row["hold_seconds"] for row in event_records], dtype=np.float32
            ),
            transition_observations=np.asarray(
                [row["raw_observation"] for row in transition_records],
                dtype=np.float32,
            ).reshape(len(transition_records), RAW_EVENT_STATE_DIM),
            transition_actions=np.asarray(
                [row["action"] for row in transition_records], dtype=np.float32
            ).reshape(len(transition_records), 1),
            transition_next_observations=np.asarray(
                [row["raw_next_observation"] for row in transition_records],
                dtype=np.float32,
            ).reshape(len(transition_records), RAW_EVENT_STATE_DIM),
            transition_rewards=np.asarray(
                [row["reward"] for row in transition_records], dtype=np.float64
            ),
            transition_delta_t_seconds=np.asarray(
                [row["delta_t_seconds"] for row in transition_records],
                dtype=np.float64,
            ),
            transition_discounts=np.asarray(
                [row["discount"] for row in transition_records], dtype=np.float32
            ),
            transition_terminals=np.asarray(
                [row["terminal"] for row in transition_records], dtype=np.uint8
            ),
            wait_passenger_seconds=np.asarray(
                [row["wait_passenger_seconds"] for row in transition_records],
                dtype=np.float64,
            ),
            in_vehicle_passenger_seconds=np.asarray(
                [row["in_vehicle_passenger_seconds"] for row in transition_records],
                dtype=np.float64,
            ),
        )
    os.replace(temporary, path)
    return sha256(path)


def main() -> None:
    args = parse_args()
    if args.max_events < 0 or args.max_sim_time_seconds < 0:
        raise ValueError("diagnostic limits cannot be negative")
    if args.max_events and args.max_sim_time_seconds:
        raise ValueError("max-events and max-sim-time-seconds are mutually exclusive")
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite evaluation {args.output}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config, station_max, dataset_hash = load_dataset_contract(args.dataset)
    if args.max_sim_time_seconds > config.episode_seconds:
        raise ValueError("diagnostic horizon exceeds the MDP contract")
    horizon = int(args.max_sim_time_seconds or config.episode_seconds)
    policy, policy_metadata = build_policy(args, config, station_max, dataset_hash)
    scenario_runtime, scenario = load_validated_scenario(args.scenario_manifest)
    sumo_dir = args.sumo_dir.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()
    source_hashes = hash_files(evaluation_source_paths(sumo_dir, edge_xml))
    env = AllLinesSumoEnv(
        str(sumo_dir),
        str(edge_xml),
        episode_seconds=horizon,
        sumo_seed=args.sumo_seed,
        sumo_cfg=scenario_runtime["sumo_cfg"],
        passenger_obj_xml=scenario_runtime["passenger_obj_xml"],
        observation_schema="v5",
    )

    pending = {}
    raw_events: list[np.ndarray] = []
    event_records: list[dict] = []
    transition_records: list[dict] = []
    inference_seconds = []
    station_wraparounds = 0
    terminal_endpoints = 0
    completed_trip_contexts = 0
    completion_closures = 0
    horizon_closures = 0
    stopped_by_event_limit = False
    started = time.perf_counter()
    try:
        env.reset()
        env.step_to_event({})
        seen_contexts, closed = _close_completed_trip_transitions(
            env._bridge, pending, transition_records, config
        )
        completed_trip_contexts += seen_contexts
        completion_closures += closed
        while not env.done:
            events = list(env._current_events)
            if not events:
                _, _, done = env.step_to_event({})
                seen_contexts, closed = _close_completed_trip_transitions(
                    env._bridge, pending, transition_records, config
                )
                completed_trip_contexts += seen_contexts
                completion_closures += closed
                if done:
                    break
                continue

            event_rows = []
            control_raw = []
            for event in events:
                raw = raw_observation(env, event.bus_id)
                passenger_time = passenger_time_snapshot_v5(
                    event.metadata["passenger_time_at_decision"]
                )
                terminal = _is_terminal_event(env, event)
                event_rows.append((event, raw, passenger_time, terminal))
                if not terminal:
                    control_raw.append(raw)

            if control_raw:
                inference_started = time.perf_counter()
                action_batch = np.asarray(
                    policy(np.stack(control_raw)), dtype=np.float32
                )
                inference_elapsed = time.perf_counter() - inference_started
                if action_batch.shape != (len(control_raw), 1):
                    raise ValueError(
                        f"policy returned {action_batch.shape}; "
                        f"expected {(len(control_raw), 1)}"
                    )
                if not np.isfinite(action_batch).all() or np.any(
                    np.abs(action_batch) > 1.00001
                ):
                    raise ValueError("policy emitted an invalid action")
                inference_seconds.extend(
                    [inference_elapsed / len(control_raw)] * len(control_raw)
                )
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
                    transition_records.append(
                        _transition_record(
                            previous,
                            raw,
                            passenger_time,
                            config,
                            terminal=terminal,
                        )
                    )
                elif previous is not None:
                    station_wraparounds += 1
                    transition_records.append(
                        _transition_record(
                            previous,
                            _horizon_observation(previous["raw"], passenger_time),
                            passenger_time,
                            config,
                            terminal=True,
                        )
                    )

                if terminal:
                    terminal_endpoints += 1
                    env_actions[event.bus_id] = [0.0, 1.0]
                    continue

                action = next(control_actions)
                hold_seconds = float(
                    policy_action_to_hold_seconds_v5(action[0], config)
                )
                raw_events.append(raw.copy())
                event_records.append(
                    _event_record(event, raw, action, hold_seconds)
                )
                pending[bus_id] = {
                    "raw": raw.copy(),
                    "action": action.copy(),
                    "station_index": station_index,
                    "passenger_time": passenger_time,
                    "line_id": str(event.line_id),
                }
                env_actions[event.bus_id] = [hold_seconds, 1.0]

            _, _, done = env.step_to_event(env_actions)
            seen_contexts, closed = _close_completed_trip_transitions(
                env._bridge, pending, transition_records, config
            )
            completed_trip_contexts += seen_contexts
            completion_closures += closed
            if done:
                break
            if args.max_events and len(event_records) >= args.max_events:
                stopped_by_event_limit = True
                break

        bridge = env._bridge
        observed_horizon = float(bridge.current_time)
        passenger_count_audit = bridge.audit_passenger_time_counts()
        passenger_time_end = bridge.passenger_time_ledger.snapshot()
        for previous in tuple(pending.values()):
            if passenger_time_end.time_seconds <= previous["passenger_time"].time_seconds:
                continue
            transition_records.append(
                _transition_record(
                    previous,
                    _horizon_observation(previous["raw"], passenger_time_end),
                    passenger_time_end,
                    config,
                    terminal=True,
                )
            )
            horizon_closures += 1
        pending.clear()
        passenger_summary = summarize_passengers_v5(bridge, passenger_time_end)
        service_summary = summarize_service(bridge)
        backend = bridge.sumo_binary
    finally:
        env.close()

    if not transition_records:
        raise RuntimeError("evaluation produced no same-bus transitions")
    if passenger_summary["active_state_count_balance_error"] < 0:
        raise RuntimeError(
            "SUMO passenger-state counts exceed the number of unfinished passengers"
        )
    if (
        not args.max_events
        and not args.max_sim_time_seconds
        and service_summary["active_trips_at_horizon"] == 0
        and horizon_closures
    ):
        raise RuntimeError(
            "a full evaluation closed pending trajectories at the horizon even "
            "though all SUMO trips had already completed"
        )
    generalized_total = (
        config.waiting_time_value * passenger_summary["accrued_wait_seconds_total"]
        + config.in_vehicle_time_value
        * passenger_summary["accrued_in_vehicle_seconds_total"]
    )
    departed = passenger_summary["departed_passengers"]
    event_log_hash = None
    if args.event_log is not None:
        event_log_hash = save_event_log(
            args.event_log, raw_events, event_records, transition_records
        )

    result = {
        "schema_version": "transit-evaluation-v5",
        "measurement_schema": "sumo-passenger-time-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method_name,
        "contract_schema": SCHEMA_VERSION,
        "contract": config.to_dict(),
        "raw_state_names": list(RAW_EVENT_STATE_NAMES),
        "dataset": {"path": str(args.dataset), "sha256": dataset_hash},
        "policy": policy_metadata,
        "sumo_seed": args.sumo_seed,
        "scenario": scenario,
        "evaluation_job": {
            "signature": args.job_signature,
            "max_events": args.max_events,
            "max_sim_time_seconds": args.max_sim_time_seconds,
        },
        "episode": {
            "configured_seconds": config.episode_seconds,
            "evaluation_horizon_seconds": horizon,
            "observed_end_time_seconds": observed_horizon,
            "control_decisions": len(event_records),
            "transitions": len(transition_records),
            "terminal_transitions": int(
                sum(row["terminal"] for row in transition_records)
            ),
            "terminal_endpoints": terminal_endpoints,
            "completed_trip_contexts": completed_trip_contexts,
            "completion_closures": completion_closures,
            "horizon_closures": horizon_closures,
            "station_wraparounds": station_wraparounds,
            "stopped_by_event_limit": stopped_by_event_limit,
            "stopped_by_time_limit": bool(args.max_sim_time_seconds),
        },
        "operations": summarize_operations_v5(event_records, transition_records),
        "passengers": passenger_summary,
        "passenger_welfare": {
            "primary": True,
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
        "inference_seconds_per_decision": numeric_summary(inference_seconds),
        "event_log": (
            {
                "path": str(args.event_log.expanduser().resolve()),
                "sha256": event_log_hash,
            }
            if args.event_log is not None
            else None
        ),
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "sumo_backend": backend,
            "sumo_version": sumo_version(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": str(args.device),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "source_sha256": source_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
