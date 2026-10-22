#!/usr/bin/env python3
"""Collect one zero-hold SUMO trace of decision-lane observation speeds."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
from bus_h2o.sumo_provenance import (
    collection_source_paths,
    hash_files,
    load_validated_scenario,
    sha256,
    sumo_version,
)
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
)


TRACE_SCHEMA = "transit-simulator-observation-speed-trace-v7-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--sumo-seed", type=int, required=True)
    parser.add_argument("--realization-block-id", type=int, required=True)
    parser.add_argument("--source-shard-summary", type=Path, required=True)
    parser.add_argument("--source-job-signature", required=True)
    parser.add_argument("--job-signature", required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _event_sort_key(event) -> tuple[float, str, str, int]:
    return (
        float(event.sim_time),
        str(event.line_id),
        str(event.bus_id),
        int(event.stop_idx),
    )


def _is_terminal_event(env: AllLinesSumoEnv, event) -> bool:
    line = env._bridge.line_obj_dic[event.line_id]
    return bool(
        event.stop_id == line.stop_id_l[-1]
        or int(event.stop_idx) >= len(line.stop_id_l) - 1
    )


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return sha256(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_source_summary(
    path: Path,
    *,
    scenario_sha256: str,
    sumo_seed_value: int,
    source_job_signature: str,
) -> dict:
    path = path.expanduser().resolve()
    summary = json.loads(path.read_text())
    checks = {
        "v7_shard_schema": summary.get("schema_version")
        == "transit-smdp-v7-shard-v1",
        "formal_full_horizon": bool(
            not summary.get("diagnostic")
            and summary.get("episode", {}).get("configured_seconds")
            == int(DEFAULT_CONFIG_V7.horizon_seconds)
            and summary.get("episode", {}).get("observed_end_time_seconds")
            == DEFAULT_CONFIG_V7.horizon_seconds
        ),
        "exact_zero_policy": bool(
            summary.get("policy_name") == "zero"
            and summary.get("policy", {}).get("type") == "zero"
            and summary.get("actions", {}).get("exact_zero_rate") == 1.0
        ),
        "scenario_matches": summary.get("scenario", {}).get("manifest_sha256")
        == scenario_sha256,
        "sumo_seed_matches": int(summary.get("sumo_seed", -1))
        == int(sumo_seed_value),
        "source_job_signature_matches": summary.get("job_signature")
        == source_job_signature,
    }
    if not all(checks.values()):
        raise ValueError(f"source zero-hold summary failed checks: {checks}")
    shard = Path(summary["shard"]).expanduser().resolve()
    if sha256(shard) != summary.get("shard_sha256"):
        raise ValueError("source zero-hold shard changed")
    return summary


def _codes(values: list[str]) -> tuple[np.ndarray, list[str]]:
    vocabulary = sorted(set(values))
    lookup = {value: index for index, value in enumerate(vocabulary)}
    return np.asarray([lookup[value] for value in values], dtype=np.int16), vocabulary


def main() -> None:
    args = parse_args()
    if args.realization_block_id not in FORMAL_CALIBRATION_REALIZATION_BLOCKS:
        raise ValueError("speed traces may use only formal calibration blocks 0 through 4")
    output = args.output.expanduser().resolve()
    summary_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite speed trace {output}")

    scenario_path = args.scenario_manifest.expanduser().resolve()
    scenario_runtime, scenario = load_validated_scenario(scenario_path)
    source_summary_path = args.source_shard_summary.expanduser().resolve()
    source_summary = _load_source_summary(
        source_summary_path,
        scenario_sha256=scenario["manifest_sha256"],
        sumo_seed_value=args.sumo_seed,
        source_job_signature=args.source_job_signature,
    )
    sumo_dir = args.sumo_dir.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()
    env = AllLinesSumoEnv(
        str(sumo_dir),
        str(edge_xml),
        episode_seconds=int(DEFAULT_CONFIG_V7.horizon_seconds),
        sumo_seed=args.sumo_seed,
        sumo_cfg=scenario_runtime["sumo_cfg"],
        passenger_obj_xml=scenario_runtime["passenger_obj_xml"],
        observation_schema="v5",
    )

    line_index = []
    station_index = []
    sim_time_seconds = []
    segment_mean_speed_mps = []
    forward_bus_present = []
    backward_bus_present = []
    stop_ids: list[str] = []
    lane_ids: list[str] = []
    bus_ids: list[str] = []
    terminal_events = 0
    batches = 0
    started = time.perf_counter()
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            events = sorted(env._current_events, key=_event_sort_key)
            if not events:
                env.step_to_event({})
                continue
            batches += 1
            actions = {}
            for event in events:
                actions[event.bus_id] = [0.0, 1.0]
                if _is_terminal_event(env, event):
                    terminal_events += 1
                    continue
                speed = float(event.segment_mean_speed)
                if not np.isfinite(speed) or speed < 1.0:
                    raise RuntimeError(
                        f"SUMO emitted invalid decision-lane speed {speed}"
                    )
                stop = env._bridge.stop_obj_dic.get(event.stop_id)
                lane_id = str(getattr(stop, "at_lane_s", ""))
                if not lane_id:
                    raise RuntimeError(f"SUMO event {event.stop_id} has no stop lane")
                line_index.append(int(env._line_idx_map[event.line_id]))
                station_index.append(int(event.stop_idx))
                sim_time_seconds.append(float(event.sim_time))
                segment_mean_speed_mps.append(speed)
                forward_bus_present.append(int(bool(event.forward_bus_present)))
                backward_bus_present.append(int(bool(event.backward_bus_present)))
                stop_ids.append(str(event.stop_id))
                lane_ids.append(lane_id)
                bus_ids.append(str(event.bus_id))
            env.step_to_event(actions)
    finally:
        env.close()

    stop_code, stop_vocabulary = _codes(stop_ids)
    lane_code, lane_vocabulary = _codes(lane_ids)
    bus_code, bus_vocabulary = _codes(bus_ids)
    arrays = {
        "line_index": np.asarray(line_index, dtype=np.int16),
        "station_index": np.asarray(station_index, dtype=np.int16),
        "sim_time_seconds": np.asarray(sim_time_seconds, dtype=np.float64),
        "segment_mean_speed_mps": np.asarray(
            segment_mean_speed_mps, dtype=np.float32
        ),
        "forward_bus_present": np.asarray(forward_bus_present, dtype=np.uint8),
        "backward_bus_present": np.asarray(backward_bus_present, dtype=np.uint8),
        "physical_stop_code": stop_code,
        "lane_code": lane_code,
        "bus_code": bus_code,
    }
    count = len(arrays["segment_mean_speed_mps"])
    expected_count = int(source_summary["episode"]["control_decisions"])
    checks = {
        "nonempty": count > 0,
        "control_decisions_match_frozen_zero_shard": count == expected_count,
        "all_arrays_aligned": all(len(values) == count for values in arrays.values()),
        "all_twelve_lines_observed": sorted(set(line_index)) == list(range(12)),
        "times_within_horizon": bool(
            count
            and np.all(arrays["sim_time_seconds"] >= 0.0)
            and np.all(
                arrays["sim_time_seconds"] <= DEFAULT_CONFIG_V7.horizon_seconds
            )
        ),
        "speeds_finite_and_at_least_one_mps": bool(
            count
            and np.isfinite(arrays["segment_mean_speed_mps"]).all()
            and np.all(arrays["segment_mean_speed_mps"] >= 1.0)
        ),
        "binary_presence_flags": bool(
            np.isin(arrays["forward_bus_present"], (0, 1)).all()
            and np.isin(arrays["backward_bus_present"], (0, 1)).all()
        ),
        "terminal_events_match_frozen_zero_shard": terminal_events
        == int(source_summary["episode"]["terminal_endpoints_excluded_from_control"]),
        "sumo_version_matches_frozen_zero_shard": sumo_version()
        == source_summary["runtime"]["sumo_version"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"observation-speed trace failed checks: {checks}")

    trace_hash = _atomic_save_npz(output, arrays)
    source_paths = set(
        collection_source_paths(ROOT, sumo_dir, edge_xml, contract_version="v7")
    )
    source_paths.add(Path(__file__).resolve())
    speeds = arrays["segment_mean_speed_mps"].astype(np.float64)
    summary = {
        "schema_version": TRACE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(output),
        "trace_sha256": trace_hash,
        "job_signature": args.job_signature,
        "study_stage": "formal-calibration",
        "policy": {"type": "zero", "hold_seconds": 0.0},
        "realization_block_id": args.realization_block_id,
        "demand_scale": float(scenario["demand_scale"]),
        "demand_realization_seed": int(scenario["selection_seed"]),
        "sumo_seed": args.sumo_seed,
        "sumo_version": sumo_version(),
        "scenario": {
            "manifest": str(scenario_path),
            "manifest_sha256": scenario["manifest_sha256"],
            "runtime_input_sha256": scenario["runtime_input_sha256"],
        },
        "source_zero_shard": {
            "path": source_summary["shard"],
            "sha256": source_summary["shard_sha256"],
            "summary_path": str(source_summary_path),
            "summary_sha256": sha256(source_summary_path),
            "job_signature": source_summary["job_signature"],
        },
        "event_identity": {
            "line_index_to_id": [
                line_id
                for line_id, _ in sorted(
                    env._line_idx_map.items(), key=lambda item: item[1]
                )
            ],
            "physical_stop_code_to_id": stop_vocabulary,
            "lane_code_to_id": lane_vocabulary,
            "bus_code_to_id": bus_vocabulary,
        },
        "events": {
            "control_decisions": count,
            "simultaneous_batches": batches,
            "terminal_endpoints_excluded": terminal_events,
            "backward_bus_present": int(arrays["backward_bus_present"].sum()),
        },
        "speed_mps": {
            "minimum": float(speeds.min()),
            "maximum": float(speeds.max()),
            "mean": float(speeds.mean()),
            "standard_deviation": float(speeds.std()),
            "quantiles": {
                str(quantile): float(np.quantile(speeds, quantile / 100.0))
                for quantile in (1, 5, 50, 95, 99)
            },
            "network_limit_fraction": float(np.mean(speeds >= 13.89 - 1e-6)),
            "bridge_fallback_10mps_fraction": float(np.mean(speeds == 10.0)),
        },
        "coverage_gate": {"passed": True, "checks": checks},
        "source_sha256": hash_files(tuple(sorted(source_paths))),
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    _atomic_write_json(summary_path, summary)
    print(json.dumps({"trace": str(output), "events": count, "passed": True}))


if __name__ == "__main__":
    main()
