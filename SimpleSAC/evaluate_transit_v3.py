#!/usr/bin/env python3
"""Evaluate one canonical policy on one explicitly seeded SUMO episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
from bus_h2o.mdp_contract import (
    RAW_V2_OBS_DIM,
    SCHEMA_VERSION,
    TransitMDPConfig,
    continuous_time_discount,
    hold_seconds_to_policy_action,
    policy_action_to_hold_seconds,
    raw_v2_to_canonical,
    transition_duration_seconds,
    transition_reward_from_v2,
)
from SimpleSAC.transit_v3.evaluation import (
    headway_balance_actions,
    numeric_summary,
    summarize_operations,
    summarize_passengers,
    summarize_service,
)
from SimpleSAC.transit_v3.trainer import load_actor_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--zero-hold", action="store_true")
    policy.add_argument("--constant-hold-seconds", type=float)
    policy.add_argument("--headway-balance-gain", type=float)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--sumo-seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--job-signature")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        help="Passenger-only demand scenario built by build_passenger_scenarios.py",
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--max-sim-time-seconds",
        type=int,
        default=0,
        help="diagnostic fixed simulation horizon; zero uses the contract horizon",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "bus_h2o" / "datasets_v3" / "transit_smdp_v3.h5",
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
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_observation(env: AllLinesSumoEnv, bus_id) -> np.ndarray:
    values = env.state[bus_id]
    observation = values[-1]
    while isinstance(observation, list) and len(observation) == 1:
        observation = observation[0]
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    if array.size == RAW_V2_OBS_DIM - 2:
        array = np.concatenate((array, np.zeros(2, dtype=np.float32)))
    if array.shape != (RAW_V2_OBS_DIM,):
        raise ValueError(f"SUMO emitted an observation with shape {array.shape}")
    return array


def load_dataset_contract(path: Path) -> tuple[TransitMDPConfig, np.ndarray, str]:
    dataset_hash = sha256(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("evaluation dataset does not implement transit-smdp-v3")
        config = TransitMDPConfig(**json.loads(handle.attrs["config"]))
        station_max = np.asarray(
            json.loads(handle.attrs["station_max_by_line"]), dtype=np.float32
        )
    return config, station_max, dataset_hash


def load_scenario(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        return {}, {"type": "base", "demand_scale": 1.0}
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-passenger-demand-v1":
        raise ValueError("scenario manifest has an unsupported schema")
    for raw_path, expected_hash in manifest.get("scenario_sha256", {}).items():
        scenario_file = Path(raw_path)
        if not scenario_file.is_file():
            raise FileNotFoundError(f"scenario file is missing: {scenario_file}")
        actual_hash = sha256(scenario_file)
        if actual_hash != expected_hash:
            raise ValueError(
                f"scenario file hash mismatch for {scenario_file}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    runtime = {
        "sumo_cfg": manifest["sumo_config"],
        "passenger_obj_xml": manifest["passenger_objects"],
    }
    evidence = {
        "type": "passenger-only-demand",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "demand_scale": float(manifest["demand_scale"]),
        "selection_seed": int(manifest["selection_seed"]),
        "source_total_persons": int(manifest["source_total_persons"]),
        "scaled_total_persons": int(manifest["scaled_total_persons"]),
        "scenario_sha256": manifest["scenario_sha256"],
    }
    return runtime, evidence


def build_policy(args, config: TransitMDPConfig, dataset_hash: str):
    if args.zero_hold:
        return (
            lambda observations: np.full((len(observations), 1), -1.0, np.float32),
            {"type": "zero_hold"},
        )
    if args.constant_hold_seconds is not None:
        if not 0.0 <= args.constant_hold_seconds <= config.max_hold_seconds:
            raise ValueError("constant hold lies outside the canonical action range")
        action = float(
            hold_seconds_to_policy_action(args.constant_hold_seconds, config)
        )
        return (
            lambda observations: np.full((len(observations), 1), action, np.float32),
            {"type": "constant_hold", "hold_seconds": args.constant_hold_seconds},
        )
    if args.headway_balance_gain is not None:
        if args.headway_balance_gain < 0:
            raise ValueError("headway-balance gain must be non-negative")
        return (
            lambda observations: headway_balance_actions(
                observations, args.headway_balance_gain, config
            ),
            {
                "type": "two-sided-headway-balance",
                "gain": args.headway_balance_gain,
                "hold_seconds": "clip(gain*(backward-forward)/2, 0, 60)",
            },
        )

    checkpoint_hash = sha256(args.checkpoint)
    actor, extra = load_actor_checkpoint(args.checkpoint, args.device)
    checkpoint_dataset_hash = extra.get("dataset_sha256")
    if checkpoint_dataset_hash != dataset_hash:
        raise ValueError(
            "checkpoint was not trained from the exact evaluation contract dataset: "
            f"checkpoint={checkpoint_dataset_hash!r}, current={dataset_hash!r}"
        )

    def policy(observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=args.device
            )
            actions, _ = actor(tensor, deterministic=True)
        return actions.cpu().numpy()

    return policy, {
        "type": "checkpoint",
        "path": str(args.checkpoint.resolve()),
        "sha256": checkpoint_hash,
        "training_metadata": extra,
    }


def save_event_log(path: Path, event_records: list[dict], transition_records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_keys = (
        "sim_time",
        "station_index",
        "scheduled_headway_seconds",
        "forward_ratio",
        "backward_ratio",
        "hold_seconds",
    )
    transition_keys = ("reward", "delta_t_seconds", "discount")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            event_line_id=np.asarray([row["line_id"] for row in event_records]),
            **{
                f"event_{key}": np.asarray(
                    [np.nan if row[key] is None else row[key] for row in event_records]
                )
                for key in event_keys
            },
            transition_line_id=np.asarray(
                [row["line_id"] for row in transition_records]
            ),
            **{
                f"transition_{key}": np.asarray(
                    [row[key] for row in transition_records], dtype=np.float64
                )
                for key in transition_keys
            },
        )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.max_events < 0 or args.max_sim_time_seconds < 0:
        raise ValueError("diagnostic limits cannot be negative")
    if args.max_events and args.max_sim_time_seconds:
        raise ValueError("max-events and max-sim-time-seconds are mutually exclusive")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config, station_max, dataset_hash = load_dataset_contract(args.dataset)
    if args.max_sim_time_seconds > config.episode_seconds:
        raise ValueError("diagnostic simulation horizon exceeds the contract horizon")
    evaluation_horizon = float(args.max_sim_time_seconds or config.episode_seconds)
    policy, policy_metadata = build_policy(args, config, dataset_hash)
    scenario_runtime, scenario_evidence = load_scenario(args.scenario_manifest)
    env = AllLinesSumoEnv(
        str(args.sumo_dir),
        str(args.edge_xml),
        episode_seconds=int(evaluation_horizon),
        sumo_seed=args.sumo_seed,
        **scenario_runtime,
    )
    pending = {}
    event_records: list[dict] = []
    transition_records: list[dict] = []
    inference_seconds = []
    station_wraparounds = 0
    started = time.perf_counter()
    stopped_by_event_limit = False
    stopped_by_time_limit = bool(args.max_sim_time_seconds)

    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            events = list(env._current_events)
            if not events:
                _, _, done = env.step_to_event({})
                if done:
                    break
                continue
            raw_batch = np.stack([raw_observation(env, event.bus_id) for event in events])
            canonical_batch = raw_v2_to_canonical(raw_batch, station_max, config)
            inference_started = time.perf_counter()
            action_batch = np.asarray(policy(canonical_batch), dtype=np.float32)
            inference_elapsed = time.perf_counter() - inference_started
            if action_batch.shape != (len(events), 1):
                raise ValueError(
                    f"policy returned {action_batch.shape}; expected {(len(events), 1)}"
                )
            action_batch = np.clip(action_batch, -1.0, 1.0)
            inference_seconds.extend([inference_elapsed / len(events)] * len(events))

            env_actions = {}
            for event, raw, action in zip(events, raw_batch, action_batch):
                bus_id = event.bus_id
                station_index = int(round(float(raw[2])))
                hold_seconds = float(
                    policy_action_to_hold_seconds(action[0], config)
                )
                scheduled = max(float(raw[8]), 1.0)
                event_records.append(
                    {
                        "sim_time": float(event.sim_time),
                        "line_id": str(event.line_id),
                        "bus_id": str(bus_id),
                        "station_index": station_index,
                        "scheduled_headway_seconds": scheduled,
                        "forward_ratio": (
                            float(event.forward_headway / scheduled)
                            if event.forward_bus_present
                            else None
                        ),
                        "backward_ratio": (
                            float(event.backward_headway / scheduled)
                            if event.backward_bus_present
                            else None
                        ),
                        "hold_seconds": hold_seconds,
                    }
                )

                previous = pending.pop(bus_id, None)
                if previous is not None and station_index > previous["station_index"]:
                    reward = float(
                        transition_reward_from_v2(raw, previous["action"], config)
                    )
                    duration = float(
                        transition_duration_seconds(previous["raw"], raw)
                    )
                    transition_records.append(
                        {
                            "line_id": str(event.line_id),
                            "reward": reward,
                            "delta_t_seconds": duration,
                            "discount": float(continuous_time_discount(duration, config)),
                        }
                    )
                elif previous is not None:
                    station_wraparounds += 1

                pending[bus_id] = {
                    "raw": raw.copy(),
                    "action": action.copy(),
                    "station_index": station_index,
                }
                env_actions[bus_id] = [hold_seconds, 1.0]

            _, _, done = env.step_to_event(env_actions)
            if done:
                break
            if args.max_events and len(event_records) >= args.max_events:
                stopped_by_event_limit = True
                break

        bridge = env._bridge
        horizon = float(bridge.current_time)
        result = {
            "schema_version": "transit-evaluation-v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": args.method_name,
            "sumo_seed": args.sumo_seed,
            "sumo_backend": bridge.sumo_binary,
            "sumo_version": subprocess.run(
                ["sumo", "--version"], capture_output=True, text=True, check=False
            ).stdout.splitlines()[0],
            "contract": asdict(config),
            "dataset": {
                "path": str(args.dataset.resolve()),
                "sha256": dataset_hash,
            },
            "policy": policy_metadata,
            "scenario": scenario_evidence,
            "episode": {
                "configured_horizon_seconds": config.episode_seconds,
                "evaluation_horizon_seconds": evaluation_horizon,
                "observed_end_time_seconds": horizon,
                "stopped_by_event_limit": stopped_by_event_limit,
                "stopped_by_time_limit": stopped_by_time_limit,
                "station_wraparounds": station_wraparounds,
                "wall_time_seconds": time.perf_counter() - started,
            },
            "evaluation_job": {
                "signature": args.job_signature,
                "max_events": int(args.max_events),
                "max_sim_time_seconds": int(args.max_sim_time_seconds),
            },
            "runtime": {
                "diagnostic_sumo_outputs_disabled": True,
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
            },
            "operations": summarize_operations(event_records, transition_records),
            "passengers": summarize_passengers(bridge, horizon),
            "service": summarize_service(bridge),
            "inference_seconds_per_decision": numeric_summary(inference_seconds),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
        if args.event_log:
            save_event_log(args.event_log, event_records, transition_records)
        if not args.quiet:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
