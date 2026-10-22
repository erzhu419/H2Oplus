#!/usr/bin/env python3
"""Collect complete zero-hold SUMO service traces for v7 runtime calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_INDEX
from bus_h2o.sumo_provenance import (
    collection_source_paths,
    hash_files,
    load_validated_scenario,
    sha256,
    sumo_version,
)
from collect_policy.build_transit_v7_dataset import load_shard
from collect_policy.collect_transit_v7 import (
    event_sort_key,
    is_terminal_event,
    raw_event_observation,
)
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
    MATRIX_SCHEMA,
    _load_frozen_zero_audit,
)


TRACE_SCHEMA = "transit-smdp-v7-full-service-trace-v1"
TRACE_MATRIX_SCHEMA = "transit-smdp-v7-full-service-trace-matrix-v1"
TRACE_ARRAY_NAMES = (
    "stop_bus_code",
    "stop_line_index",
    "stop_station_index",
    "stop_arrival_time_seconds",
    "stop_service_completion_time_seconds",
    "stop_decision_ready_time_seconds",
    "stop_base_dwell_seconds",
    "stop_actual_departure_time_seconds",
    "segment_bus_code",
    "segment_line_index",
    "segment_origin_station_index",
    "segment_origin_decision_ready_time_seconds",
    "segment_destination_arrival_time_seconds",
    "segment_runtime_seconds",
)


def canonical_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def trace_job_signature(
    *,
    trace_collector_sha256: str,
    trace_runner_sha256: str,
    matrix_sha256: str,
    source_job: dict[str, Any],
    source_shard_sha256: str,
) -> str:
    return canonical_signature(
        {
            "trace_schema": TRACE_SCHEMA,
            "trace_matrix_schema": TRACE_MATRIX_SCHEMA,
            "trace_collector_sha256": trace_collector_sha256,
            "trace_runner_sha256": trace_runner_sha256,
            "matrix_sha256": matrix_sha256,
            "source_job_signature": source_job["signature"],
            "source_shard_sha256": source_shard_sha256,
            "realization_block_id": int(source_job["realization_block_id"]),
            "demand_scale": float(source_job["scenario"]["demand_scale"]),
            "scenario_manifest_sha256": source_job["scenario"]["sha256"],
            "sumo_seed": int(source_job["sumo_seed"]),
            "sumo_version": source_job["sumo_version"],
        }
    )


def _decision_ready_time(service_completion_time: float) -> float:
    if not np.isfinite(service_completion_time):
        raise ValueError("service-completion time must be finite")
    return float(math.ceil(float(service_completion_time) - 1e-9))


def extract_full_service_arrays(
    bridge: Any, line_index_to_id: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract every stop and segment from completed SUMO trip objects."""

    line_to_index = {line_id: index for index, line_id in enumerate(line_index_to_id)}
    if len(line_to_index) != len(line_index_to_id):
        raise ValueError("line-index mapping contains duplicates")
    bus_ids = sorted(str(value) for value in bridge.departed_trip_ids)
    if not bus_ids:
        raise ValueError("full trace contains no departed SUMO trips")
    if set(bus_ids) != set(str(value) for value in bridge.completed_trip_times):
        raise ValueError("full trace requires every departed SUMO trip to complete")
    bus_code_to_id = bus_ids
    bus_to_code = {bus_id: index for index, bus_id in enumerate(bus_code_to_id)}
    stop_rows: dict[str, list] = {
        name: [] for name in TRACE_ARRAY_NAMES if name.startswith("stop_")
    }
    segment_rows: dict[str, list] = {
        name: [] for name in TRACE_ARRAY_NAMES if name.startswith("segment_")
    }
    service_completion_quantization = []
    actual_departure_lag = []
    per_line_trips = Counter()
    per_line_segments = Counter()

    for bus_id in bus_ids:
        bus = bridge.bus_obj_dic[bus_id]
        line_id = str(bus.belong_line_id_s)
        if line_id not in line_to_index:
            raise ValueError(f"trace bus {bus_id} has unknown line {line_id}")
        line = bridge.line_obj_dic[line_id]
        stops = [str(value) for value in line.stop_id_l]
        if len(stops) < 2 or len(set(stops)) != len(stops):
            raise ValueError(f"trace line {line_id} has an invalid stop sequence")
        arrivals = bus.arriver_stop_time_d
        completions = bus.service_completion_history
        departures = bus.depart_stop_time_d
        missing_arrivals = [stop for stop in stops if stop not in arrivals]
        missing_departures = [stop for stop in stops if stop not in departures]
        invalid_completions = [
            stop for stop in stops if len(completions.get(stop, ())) != 1
        ]
        if missing_arrivals or missing_departures or invalid_completions:
            raise ValueError(
                f"trace trip {bus_id} is incomplete: arrivals={missing_arrivals}, "
                f"departures={missing_departures}, "
                f"service_completions={invalid_completions}"
            )

        trip_events = []
        for station_index, stop_id in enumerate(stops):
            arrival = float(arrivals[stop_id])
            service_completion = float(completions[stop_id][0])
            decision_ready = _decision_ready_time(service_completion)
            base_dwell = service_completion - arrival
            if (
                not np.isfinite(arrival)
                or not np.isfinite(base_dwell)
                or base_dwell < 0.0
                or decision_ready < service_completion
            ):
                raise ValueError(f"trace trip {bus_id} has invalid stop timing")
            actual_departure = float(departures[stop_id])
            if actual_departure < arrival:
                raise ValueError(f"trace trip {bus_id} departs before arrival")
            actual_departure_lag.append(actual_departure - decision_ready)
            service_completion_quantization.append(decision_ready - service_completion)
            trip_events.append(
                (station_index, arrival, service_completion, decision_ready, base_dwell)
            )
            stop_rows["stop_bus_code"].append(bus_to_code[bus_id])
            stop_rows["stop_line_index"].append(line_to_index[line_id])
            stop_rows["stop_station_index"].append(station_index)
            stop_rows["stop_arrival_time_seconds"].append(arrival)
            stop_rows["stop_service_completion_time_seconds"].append(
                service_completion
            )
            stop_rows["stop_decision_ready_time_seconds"].append(decision_ready)
            stop_rows["stop_base_dwell_seconds"].append(base_dwell)
            stop_rows["stop_actual_departure_time_seconds"].append(actual_departure)

        if any(
            right[1] <= left[1] or right[3] <= left[3]
            for left, right in zip(trip_events[:-1], trip_events[1:])
        ):
            raise ValueError(f"trace trip {bus_id} does not move forward in time")
        for left, right in zip(trip_events[:-1], trip_events[1:]):
            runtime = right[3] - left[3] - right[4]
            if not np.isfinite(runtime) or runtime <= 0.0:
                raise ValueError(f"trace trip {bus_id} has nonpositive segment runtime")
            segment_rows["segment_bus_code"].append(bus_to_code[bus_id])
            segment_rows["segment_line_index"].append(line_to_index[line_id])
            segment_rows["segment_origin_station_index"].append(left[0])
            segment_rows["segment_origin_decision_ready_time_seconds"].append(
                left[3]
            )
            segment_rows["segment_destination_arrival_time_seconds"].append(right[1])
            segment_rows["segment_runtime_seconds"].append(runtime)
            per_line_segments[line_id] += 1
        per_line_trips[line_id] += 1

    arrays = {
        "stop_bus_code": np.asarray(stop_rows["stop_bus_code"], dtype=np.int32),
        "stop_line_index": np.asarray(
            stop_rows["stop_line_index"], dtype=np.int16
        ),
        "stop_station_index": np.asarray(
            stop_rows["stop_station_index"], dtype=np.int16
        ),
        "stop_arrival_time_seconds": np.asarray(
            stop_rows["stop_arrival_time_seconds"], dtype=np.float64
        ),
        "stop_service_completion_time_seconds": np.asarray(
            stop_rows["stop_service_completion_time_seconds"], dtype=np.float64
        ),
        "stop_decision_ready_time_seconds": np.asarray(
            stop_rows["stop_decision_ready_time_seconds"], dtype=np.float64
        ),
        "stop_base_dwell_seconds": np.asarray(
            stop_rows["stop_base_dwell_seconds"], dtype=np.float64
        ),
        "stop_actual_departure_time_seconds": np.asarray(
            stop_rows["stop_actual_departure_time_seconds"], dtype=np.float64
        ),
        "segment_bus_code": np.asarray(
            segment_rows["segment_bus_code"], dtype=np.int32
        ),
        "segment_line_index": np.asarray(
            segment_rows["segment_line_index"], dtype=np.int16
        ),
        "segment_origin_station_index": np.asarray(
            segment_rows["segment_origin_station_index"], dtype=np.int16
        ),
        "segment_origin_decision_ready_time_seconds": np.asarray(
            segment_rows["segment_origin_decision_ready_time_seconds"],
            dtype=np.float64,
        ),
        "segment_destination_arrival_time_seconds": np.asarray(
            segment_rows["segment_destination_arrival_time_seconds"],
            dtype=np.float64,
        ),
        "segment_runtime_seconds": np.asarray(
            segment_rows["segment_runtime_seconds"], dtype=np.float64
        ),
    }
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("full service trace contains non-finite values")
    diagnostics = {
        "bus_code_to_id": bus_code_to_id,
        "line_index_to_id": list(line_index_to_id),
        "completed_trips": len(bus_ids),
        "stop_events": len(arrays["stop_bus_code"]),
        "segments": len(arrays["segment_bus_code"]),
        "per_line_trips": dict(sorted(per_line_trips.items())),
        "per_line_segments": dict(sorted(per_line_segments.items())),
        "decision_ready_minus_exact_completion_seconds": {
            "minimum": float(np.min(service_completion_quantization)),
            "maximum": float(np.max(service_completion_quantization)),
            "mean": float(np.mean(service_completion_quantization)),
        },
        "actual_departure_minus_decision_ready_seconds": {
            "n": len(actual_departure_lag),
            "minimum": float(np.min(actual_departure_lag)),
            "maximum": float(np.max(actual_departure_lag)),
            "mean": float(np.mean(actual_departure_lag)),
        },
    }
    return arrays, diagnostics


def verify_control_replay(
    observed: dict[str, list], source_summary: dict, source_arrays: dict
) -> dict[str, Any]:
    bus_ids = source_summary["event_identity"]["bus_code_to_id"]
    line_ids = source_summary["event_identity"]["line_index_to_id"]
    expected = {
        "bus_id": [bus_ids[int(value)] for value in source_arrays["event_bus_code"]],
        "line_id": [
            line_ids[int(value)] for value in source_arrays["event_line_index"]
        ],
        "station_index": source_arrays["event_station_index"].astype(int).tolist(),
        "decision_time_seconds": source_arrays[
            "event_decision_time_seconds"
        ].tolist(),
        "base_dwell_seconds": source_arrays["event_base_dwell_seconds"].tolist(),
    }
    mismatches = {}
    for name, expected_values in expected.items():
        observed_values = observed[name]
        if observed_values != expected_values:
            first = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(observed_values, expected_values)
                    )
                    if left != right
                ),
                min(len(observed_values), len(expected_values)),
            )
            mismatches[name] = {
                "observed_rows": len(observed_values),
                "expected_rows": len(expected_values),
                "first_mismatch_index": first,
                "observed": (
                    observed_values[first] if first < len(observed_values) else None
                ),
                "expected": (
                    expected_values[first] if first < len(expected_values) else None
                ),
            }
    if mismatches:
        raise ValueError(f"full trace control replay mismatch: {mismatches}")
    return {
        "passed": True,
        "control_events": len(expected["bus_id"]),
        "fields_exact": sorted(expected),
    }


def load_full_service_trace(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = path.expanduser().resolve()
    summary = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if summary.get("schema_version") != TRACE_SCHEMA:
        raise ValueError(f"unsupported full service trace: {path}")
    if summary.get("trace_sha256") != sha256(path):
        raise ValueError(f"full service trace hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as handle:
        missing = sorted(set(TRACE_ARRAY_NAMES) - set(handle.files))
        if missing:
            raise ValueError(f"full service trace is missing arrays {missing}: {path}")
        arrays = {name: np.asarray(handle[name]) for name in TRACE_ARRAY_NAMES}
    stop_n = len(arrays["stop_bus_code"])
    segment_n = len(arrays["segment_bus_code"])
    for name, values in arrays.items():
        expected_n = stop_n if name.startswith("stop_") else segment_n
        if values.shape != (expected_n,) or not np.isfinite(values).all():
            raise ValueError(f"full service trace array is invalid: {path}:{name}")
    if np.any(arrays["segment_runtime_seconds"] <= 0.0):
        raise ValueError(f"full service trace contains nonpositive runtimes: {path}")
    if summary.get("control_replay", {}).get("passed") is not True:
        raise ValueError(f"full service trace did not pass control replay: {path}")
    return summary, arrays


def _save_trace(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_trace_registration(
    trace_matrix_manifest: Path,
    *,
    matrix_manifest: Path,
    output: Path,
    block_id: int,
    trace_signature: str,
    trace_runner_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_matrix_manifest = trace_matrix_manifest.expanduser().resolve()
    manifest = json.loads(trace_matrix_manifest.read_text())
    registration = manifest.get("registration")
    if (
        manifest.get("schema_version") != TRACE_MATRIX_SCHEMA
        or not isinstance(registration, dict)
        or manifest.get("registration_sha256")
        != canonical_signature(registration)
    ):
        raise ValueError("full service trace matrix registration is invalid")
    frozen = registration.get("frozen_zero_matrix", {})
    if (
        Path(frozen.get("path", "")).expanduser().resolve()
        != matrix_manifest.resolve()
        or frozen.get("sha256") != sha256(matrix_manifest)
    ):
        raise ValueError("trace matrix does not bind the frozen zero matrix")
    collector = registration.get("trace_collector", {})
    runner = registration.get("trace_runner", {})
    if (
        Path(collector.get("path", "")).expanduser().resolve()
        != Path(__file__).resolve()
        or collector.get("sha256") != sha256(Path(__file__).resolve())
        or runner.get("sha256") != trace_runner_sha256
    ):
        raise ValueError("trace collector or runner provenance changed")
    matches = [
        job
        for job in registration.get("jobs", [])
        if job.get("trace_signature") == trace_signature
    ]
    if len(matches) != 1:
        raise ValueError("full service trace job is not uniquely registered")
    registered_job = matches[0]
    if (
        int(registered_job.get("block_id", -1)) != int(block_id)
        or Path(registered_job.get("output", "")).expanduser().resolve()
        != output.resolve()
    ):
        raise ValueError("full service trace invocation differs from registration")
    return manifest, registered_job


def collect_trace(
    matrix_manifest: Path,
    *,
    trace_matrix_manifest: Path,
    block_id: int,
    output: Path,
    trace_signature: str,
    trace_runner_sha256: str,
) -> dict[str, Any]:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    summary_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite full service trace {output}")
    trace_matrix, registered_job = _load_trace_registration(
        trace_matrix_manifest,
        matrix_manifest=matrix_manifest,
        output=output,
        block_id=block_id,
        trace_signature=trace_signature,
        trace_runner_sha256=trace_runner_sha256,
    )
    freeze_audit_path, _ = _load_frozen_zero_audit(matrix_manifest)
    matrix = json.loads(matrix_manifest.read_text())
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("full service trace requires collection matrix v2")
    matches = [job for job in matrix["jobs"] if int(job["block_id"]) == block_id]
    if len(matches) != 1:
        raise ValueError(f"full service trace block {block_id} is not unique")
    job = matches[0]
    if registered_job.get("source_job_signature") != job.get("signature"):
        raise ValueError("trace registration selects a different source shard")
    if (
        job.get("policy", {}).get("policy_name") != "zero"
        or int(job["realization_block_id"])
        not in FORMAL_CALIBRATION_REALIZATION_BLOCKS
    ):
        raise ValueError("full service traces are restricted to zero-hold fit blocks")

    source_shard = Path(job["output"]).expanduser().resolve()
    source_summary, source_arrays = load_shard(source_shard, allow_partial=False)
    collector_path = Path(__file__).resolve()
    expected_trace_signature = trace_job_signature(
        trace_collector_sha256=sha256(collector_path),
        trace_runner_sha256=trace_runner_sha256,
        matrix_sha256=sha256(matrix_manifest),
        source_job=job,
        source_shard_sha256=source_summary["shard_sha256"],
    )
    if trace_signature != expected_trace_signature:
        raise ValueError("full service trace job signature is stale")

    sumo_dir = Path(job["sumo_dir"]).expanduser().resolve()
    edge_xml = Path(job["edge_xml"]).expanduser().resolve()
    current_source = hash_files(
        collection_source_paths(ROOT, sumo_dir, edge_xml, contract_version="v7")
    )
    if current_source != matrix.get("source_sha256"):
        raise ValueError("full service trace collection sources changed")
    if sumo_version() != matrix.get("sumo_version"):
        raise ValueError("full service trace SUMO version changed")
    scenario_runtime, scenario = load_validated_scenario(Path(job["scenario"]["path"]))
    if (
        scenario.get("manifest_sha256") != job["scenario"]["sha256"]
        or scenario.get("runtime_input_sha256")
        != job["scenario"]["runtime_input_sha256"]
    ):
        raise ValueError("full service trace scenario provenance changed")

    env = AllLinesSumoEnv(
        str(sumo_dir),
        str(edge_xml),
        episode_seconds=int(job["episode_seconds"]),
        sumo_seed=int(job["sumo_seed"]),
        sumo_cfg=scenario_runtime["sumo_cfg"],
        passenger_obj_xml=scenario_runtime["passenger_obj_xml"],
        observation_schema="v5",
    )
    line_index_to_id = [None] * len(env._line_idx_map)
    for line_id, line_index in env._line_idx_map.items():
        if line_index_to_id[line_index] is not None:
            raise ValueError("full trace line mapping contains duplicate indices")
        line_index_to_id[line_index] = str(line_id)
    if any(value is None for value in line_index_to_id):
        raise ValueError("full trace line mapping contains gaps")

    observed_controls = {
        "bus_id": [],
        "line_id": [],
        "station_index": [],
        "decision_time_seconds": [],
        "base_dwell_seconds": [],
    }
    terminal_events = 0
    started = time.perf_counter()
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            events = sorted(env._current_events, key=event_sort_key)
            if not events:
                _, _, done = env.step_to_event({})
                if done:
                    break
                continue
            actions = {}
            for event in events:
                raw = raw_event_observation(env, event.bus_id)
                terminal = is_terminal_event(env, event)
                actions[event.bus_id] = [0.0, 1.0]
                if terminal:
                    terminal_events += 1
                    continue
                observed_controls["bus_id"].append(str(event.bus_id))
                observed_controls["line_id"].append(
                    line_index_to_id[
                        int(round(float(raw[RAW_EVENT_STATE_INDEX["line_id"]])))
                    ]
                )
                observed_controls["station_index"].append(
                    int(round(float(raw[RAW_EVENT_STATE_INDEX["station_index"]])))
                )
                observed_controls["decision_time_seconds"].append(
                    float(raw[RAW_EVENT_STATE_INDEX["sim_time_seconds"]])
                )
                observed_controls["base_dwell_seconds"].append(
                    float(raw[RAW_EVENT_STATE_INDEX["base_dwell_seconds"]])
                )
            _, _, done = env.step_to_event(actions)
            if done:
                break

        replay = verify_control_replay(
            observed_controls, source_summary, source_arrays
        )
        arrays, trace_diagnostics = extract_full_service_arrays(
            env._bridge, list(line_index_to_id)
        )
    finally:
        env.close()

    trace_hash = _save_trace(output, arrays)
    result = {
        "schema_version": TRACE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(output),
        "trace_sha256": trace_hash,
        "trace_signature": trace_signature,
        "study_stage": "formal",
        "policy": "exact-zero-hold",
        "realization_block_id": int(job["realization_block_id"]),
        "block_id": int(job["block_id"]),
        "demand_scale": float(job["scenario"]["demand_scale"]),
        "sumo_seed": int(job["sumo_seed"]),
        "sumo_version": matrix["sumo_version"],
        "scenario": job["scenario"],
        "control_replay": replay,
        "full_service_trace": trace_diagnostics,
        "terminal_events_emitted": terminal_events,
        "runtime": {"wall_time_seconds": time.perf_counter() - started},
        "provenance": {
            "frozen_collection_matrix": {
                "path": str(matrix_manifest),
                "sha256": sha256(matrix_manifest),
            },
            "trace_matrix": {
                "path": str(trace_matrix_manifest.expanduser().resolve()),
                "sha256": sha256(trace_matrix_manifest.expanduser().resolve()),
                "registration_sha256": trace_matrix["registration_sha256"],
            },
            "frozen_zero_matrix_audit": {
                "path": str(freeze_audit_path),
                "sha256": sha256(freeze_audit_path),
            },
            "source_shard": {
                "path": str(source_shard),
                "sha256": source_summary["shard_sha256"],
                "summary_sha256": sha256(
                    source_shard.with_suffix(source_shard.suffix + ".json")
                ),
                "job_signature": job["signature"],
            },
            "collection_source_sha256": matrix["source_sha256"],
            "trace_collector": {
                "path": str(collector_path),
                "sha256": sha256(collector_path),
            },
            "trace_runner_sha256": trace_runner_sha256,
        },
    }
    _write_json(summary_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--trace-matrix-manifest", type=Path, required=True)
    parser.add_argument("--block-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-signature", required=True)
    parser.add_argument("--trace-runner-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_trace(
        args.matrix_manifest,
        trace_matrix_manifest=args.trace_matrix_manifest,
        block_id=args.block_id,
        output=args.output,
        trace_signature=args.trace_signature,
        trace_runner_sha256=args.trace_runner_sha256,
    )
    print(
        json.dumps(
            {
                "trace": result["trace"],
                "trace_sha256": result["trace_sha256"],
                "control_replay": result["control_replay"],
                "full_service_trace": result["full_service_trace"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
