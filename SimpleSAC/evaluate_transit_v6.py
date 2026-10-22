#!/usr/bin/env python3
"""Evaluate one chronological transit-smdp-v6 policy in SUMO."""

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
from bus_h2o.mdp_contract_v6 import (
    ACTION_DIM,
    CANONICAL_STATE_DIM,
    DEFAULT_CONFIG_V6,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_SLOT_WIDTH,
    RAW_STATE_DIM,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    TransitMDPConfigV6,
    hold_seconds_to_policy_action_v6,
    pack_event_batch_v6,
    passenger_time_snapshot_v5,
    policy_action_to_hold_seconds_v6,
    raw_state_to_canonical_v6,
    terminal_raw_state_v6,
)
from bus_h2o.sumo_provenance import (
    collection_source_paths,
    hash_files,
    load_validated_scenario,
    sha256,
    sumo_version,
)
from collect_policy.collect_transit_v6 import (
    append_transition,
    event_sort_key,
    is_terminal_event,
    raw_event_observation,
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
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
from SimpleSAC.transit_v6.trainer import load_actor_checkpoint_v6


EVALUATION_SCHEMA = "transit-evaluation-v6"


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
        default=ROOT / "bus_h2o" / "datasets_v6" / "transit_smdp_v6.h5",
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
) -> tuple[TransitMDPConfigV6, np.ndarray, str]:
    resolved = path.expanduser().resolve()
    dataset_hash = sha256(resolved)
    with h5py.File(resolved, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("evaluation dataset does not implement transit-smdp-v6")
        config = TransitMDPConfigV6(**json.loads(handle.attrs["config"]))
        station_max = np.asarray(
            json.loads(handle.attrs["station_max_by_line"]), dtype=np.float32
        )
    if config != DEFAULT_CONFIG_V6:
        raise ValueError("evaluation dataset does not use the fixed v6 objective")
    if station_max.shape != (12,) or np.any(station_max <= 0.0):
        raise ValueError("evaluation dataset has invalid station normalization")
    return config, station_max, dataset_hash


def build_policy(
    args: argparse.Namespace,
    config: TransitMDPConfigV6,
    station_max: np.ndarray,
    dataset_hash: str,
):
    if args.zero_hold:
        return (
            lambda raw: np.asarray([-1.0], dtype=np.float32),
            {"type": "zero", "exact_zero_probability": 1.0},
        )
    if args.headway_balance_gain is not None:
        gain = float(args.headway_balance_gain)
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("headway-balance gain must be finite and non-negative")

        def headway_policy(raw_state: np.ndarray) -> np.ndarray:
            current = int(round(float(raw_state[2])))
            slot_start = 3 + current * RAW_SLOT_WIDTH
            local = raw_state[
                slot_start + 1 : slot_start + 1 + RAW_EVENT_STATE_DIM
            ][None, :]
            hold = headway_balance_hold_seconds_v4(
                local[:, :RAW_EVENT_STATE_DIM_V4], gain, config
            )[0]
            return np.asarray(
                [hold_seconds_to_policy_action_v6(hold, config)], np.float32
            )

        return headway_policy, {
            "type": "schedule-adjusted-headway-balance",
            "gain": gain,
            "hold_seconds": "clip(gain*((H_b-T_b)-(H_f-T_f))/2, 0, 60)",
        }
    if args.myopic_welfare:

        def myopic_policy(raw_state: np.ndarray) -> np.ndarray:
            current = int(round(float(raw_state[2])))
            slot_start = 3 + current * RAW_SLOT_WIDTH
            local = raw_state[
                slot_start + 1 : slot_start + 1 + RAW_EVENT_STATE_DIM
            ][None, :]
            hold = myopic_welfare_hold_seconds_v4(
                local[:, :RAW_EVENT_STATE_DIM_V4], config
            )[0]
            return np.asarray(
                [hold_seconds_to_policy_action_v6(hold, config)], np.float32
            )

        return myopic_policy, {
            "type": "closed-form-myopic-headway-surrogate",
            "note": "baseline only; primary welfare uses the exact v6 ledger",
        }

    checkpoint = args.checkpoint.expanduser().resolve()
    actor, extra = load_actor_checkpoint_v6(checkpoint, args.device)
    expected = {
        "dataset_sha256": dataset_hash,
        "contract_config": config.to_dict(),
        "mdp_version": "v6",
    }
    mismatches = {
        name: {"checkpoint": extra.get(name), "evaluation": value}
        for name, value in expected.items()
        if extra.get(name) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint/evaluation provenance mismatch: {mismatches}")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v6(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return action.cpu().numpy()[0]

    return checkpoint_policy, {
        "type": "checkpoint",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "selection_rule": "predeclared final training checkpoint; no test-set selection",
        "training_metadata": extra,
    }


def _event_record(event, local: np.ndarray, action: np.ndarray, hold: float) -> dict:
    target_forward = max(
        float(local[RAW_EVENT_STATE_INDEX["target_forward_headway_seconds"]]), 1.0
    )
    target_backward = max(
        float(local[RAW_EVENT_STATE_INDEX["target_backward_headway_seconds"]]), 1.0
    )
    return {
        "sim_time": float(event.sim_time),
        "line_id": str(event.line_id),
        "bus_id": str(event.bus_id),
        "station_index": int(
            round(float(local[RAW_EVENT_STATE_INDEX["station_index"]]))
        ),
        "scheduled_headway_seconds": 0.5 * (target_forward + target_backward),
        "forward_ratio": (
            float(local[RAW_EVENT_STATE_INDEX["forward_headway_seconds"]])
            / target_forward
            if local[RAW_EVENT_STATE_INDEX["forward_bus_present"]] > 0.5
            else None
        ),
        "backward_ratio": (
            float(local[RAW_EVENT_STATE_INDEX["backward_headway_seconds"]])
            / target_backward
            if local[RAW_EVENT_STATE_INDEX["backward_bus_present"]] > 0.5
            else None
        ),
        "action": action.copy(),
        "hold_seconds": hold,
    }


def summarize_operations(
    event_records: list[dict], rows: dict[str, list], batch_sizes: list[int]
) -> dict:
    by_line: dict[str, list[dict]] = defaultdict(list)
    for row in event_records:
        by_line[row["line_id"]].append(row)
    per_line = {}
    for line_id, line_rows in sorted(by_line.items()):
        line_holds = [row["hold_seconds"] for row in line_rows]
        per_line[line_id] = {
            "decision_events": len(line_rows),
            "headway": _headway_summary(line_rows),
            "hold_seconds": numeric_summary(line_holds),
            "total_hold_seconds": float(np.sum(line_holds)),
        }
    holds = [row["hold_seconds"] for row in event_records]
    return {
        "decision_events": len(event_records),
        "chronological_transitions": len(rows["rewards"]),
        "terminal_transitions": int(np.sum(rows["terminals"])),
        "simultaneous_batches": len(batch_sizes),
        "simultaneous_batch_size": numeric_summary(batch_sizes),
        "zero_duration_transitions": int(
            np.sum(np.asarray(rows["delta_t_seconds"]) == 0.0)
        ),
        "transition_reward": numeric_summary(rows["rewards"]),
        "delta_t_seconds": numeric_summary(rows["delta_t_seconds"]),
        "discount": numeric_summary(rows["discounts"]),
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
        collection_source_paths(ROOT, sumo_dir, edge_xml, contract_version="v6")
    )
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "collect_policy" / "collect_transit_v6.py",
            ROOT / "SimpleSAC" / "transit_v3" / "domain_ratio.py",
            ROOT / "SimpleSAC" / "transit_v3" / "networks.py",
            ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
            ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
            ROOT / "SimpleSAC" / "transit_v6" / "networks.py",
            ROOT / "SimpleSAC" / "transit_v6" / "trainer.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def save_event_log(path: Path, rows: dict[str, list], holds: list[float]) -> str:
    n = len(rows["rewards"])
    arrays = {
        "raw_observations": np.asarray(rows["observations"], np.float32).reshape(
            n, RAW_STATE_DIM
        ),
        "actions": np.asarray(rows["actions"], np.float32).reshape(n, ACTION_DIM),
        "rewards": np.asarray(rows["rewards"], np.float64),
        "raw_next_observations": np.asarray(
            rows["next_observations"], np.float32
        ).reshape(n, RAW_STATE_DIM),
        "discounts": np.asarray(rows["discounts"], np.float32),
        "terminals": np.asarray(rows["terminals"], np.uint8),
        "delta_t_seconds": np.asarray(rows["delta_t_seconds"], np.float64),
        "wait_passenger_seconds": np.asarray(
            rows["wait_passenger_seconds"], np.float64
        ),
        "in_vehicle_passenger_seconds": np.asarray(
            rows["in_vehicle_passenger_seconds"], np.float64
        ),
        "hold_seconds": np.asarray(holds, np.float32),
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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
        raise ValueError("diagnostic horizon exceeds the v6 contract")
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
    previous = None
    first_passenger_time = None
    event_records = []
    holds = []
    inference_seconds = []
    batch_sizes = []
    terminal_endpoints = 0
    completed_trip_contexts = 0
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
                local = raw_event_observation(env, event.bus_id)
                passenger_time = passenger_time_snapshot_v5(
                    event.metadata["passenger_time_at_decision"]
                )
                event_rows.append(
                    (event, local, passenger_time, is_terminal_event(env, event))
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
                batch_sizes.append(len(controls))
                prior_actions = np.full(len(controls), -1.0, np.float32)
                for index, (event, local, passenger_time, _) in enumerate(controls):
                    raw_state = pack_event_batch_v6(
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

                    inference_started = time.perf_counter()
                    action = np.asarray(policy(raw_state), np.float32).reshape(-1)
                    inference_seconds.append(time.perf_counter() - inference_started)
                    if action.shape != (ACTION_DIM,):
                        raise ValueError(
                            f"policy returned {action.shape}; expected {(ACTION_DIM,)}"
                        )
                    if not np.isfinite(action).all() or np.any(
                        np.abs(action) > 1.00001
                    ):
                        raise ValueError("policy emitted an invalid v6 action")
                    action = np.clip(action, -1.0, 1.0)
                    hold = float(policy_action_to_hold_seconds_v6(action[0], config))
                    event_records.append(_event_record(event, local, action, hold))
                    holds.append(hold)
                    previous = {
                        "raw": raw_state.copy(),
                        "action": action.copy(),
                        "passenger_time": passenger_time,
                    }
                    prior_actions[index] = action[0]
                    env_actions[event.bus_id] = [hold, 1.0]

            _, _, done = env.step_to_event(env_actions)
            completed_trip_contexts += len(
                env._bridge.consume_completed_trip_passenger_times()
            )
            if done:
                break
            if args.max_events and len(event_records) >= args.max_events:
                stopped_by_event_limit = True
                break

        bridge = env._bridge
        observed_end_time = float(bridge.current_time)
        passenger_count_audit = bridge.audit_passenger_time_counts()
        passenger_time_end = bridge.passenger_time_ledger.snapshot()
        if previous is None or first_passenger_time is None:
            raise RuntimeError("SUMO v6 evaluation produced no controllable event")
        append_transition(
            rows,
            previous,
            terminal_raw_state_v6(previous["raw"], passenger_time_end),
            passenger_time_end,
            terminal=True,
        )
        passenger_summary = summarize_passengers_v5(bridge, passenger_time_end)
        service_summary = summarize_service(bridge)
        backend = bridge.sumo_binary
    finally:
        env.close()

    diagnostic = bool(args.max_events or args.max_sim_time_seconds)
    if not diagnostic and observed_end_time != config.episode_seconds:
        raise RuntimeError("full v6 evaluation did not reach its fixed horizon")
    if not diagnostic and service_summary["completed_trips"] != service_summary[
        "departed_trips"
    ]:
        raise RuntimeError("not every departed SUMO trip completed")
    if passenger_summary["active_state_count_balance_error"] < 0:
        raise RuntimeError("SUMO passenger-state counts exceed unfinished passengers")
    if len(rows["rewards"]) != len(event_records):
        raise RuntimeError("v6 evaluation lost chronological decision rows")
    if int(np.sum(rows["terminals"])) != 1:
        raise RuntimeError("v6 evaluation must contain exactly one terminal row")

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
            np.sum(rows["wait_passenger_seconds"]) - covered_wait
        ),
        "in_vehicle_passenger_seconds": float(
            np.sum(rows["in_vehicle_passenger_seconds"]) - covered_in_vehicle
        ),
    }
    if any(abs(value) > 1e-9 for value in coverage_errors.values()):
        raise RuntimeError(f"v6 evaluation welfare chain is incomplete: {coverage_errors}")

    generalized_total = (
        config.waiting_time_value * passenger_summary["accrued_wait_seconds_total"]
        + config.in_vehicle_time_value
        * passenger_summary["accrued_in_vehicle_seconds_total"]
    )
    departed = passenger_summary["departed_passengers"]
    event_log_hash = (
        save_event_log(args.event_log, rows, holds)
        if args.event_log is not None
        else None
    )
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "measurement_schema": "sumo-passenger-time-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method_name,
        "contract_schema": SCHEMA_VERSION,
        "contract": config.to_dict(),
        "raw_state_names": list(RAW_STATE_NAMES),
        "dataset": {"path": str(args.dataset), "sha256": dataset_hash},
        "policy": policy_metadata,
        "sumo_seed": args.sumo_seed,
        "scenario": scenario,
        "evaluation_job": {
            "signature": args.job_signature,
            "max_events": args.max_events,
            "max_sim_time_seconds": args.max_sim_time_seconds,
            "diagnostic": diagnostic,
        },
        "episode": {
            "configured_seconds": config.episode_seconds,
            "evaluation_horizon_seconds": horizon,
            "observed_end_time_seconds": observed_end_time,
            "control_decisions": len(event_records),
            "transitions": len(rows["rewards"]),
            "terminal_transitions": int(np.sum(rows["terminals"])),
            "terminal_endpoints_excluded_from_control": terminal_endpoints,
            "completed_trip_contexts_observed": completed_trip_contexts,
            "stopped_by_event_limit": stopped_by_event_limit,
            "stopped_by_time_limit": bool(args.max_sim_time_seconds),
        },
        "operations": summarize_operations(event_records, rows, batch_sizes),
        "passengers": passenger_summary,
        "passenger_welfare": {
            "primary": True,
            "waiting_time_value": config.waiting_time_value,
            "in_vehicle_time_value": config.in_vehicle_time_value,
            "accrued_generalized_passenger_seconds": generalized_total,
            "accrued_generalized_seconds_per_departed_passenger": (
                generalized_total / departed if departed else None
            ),
            "pre_first_decision_cost_excluded_from_training_only": True,
            "first_decision_time_seconds": first_passenger_time.time_seconds,
            "chronological_coverage_errors": coverage_errors,
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
