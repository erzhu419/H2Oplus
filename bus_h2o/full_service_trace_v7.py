"""Complete per-stop and per-segment SUMO timing evidence for transit v7."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from bus_h2o.sumo_provenance import sha256


FULL_SERVICE_TRACE_SCHEMA = "transit-smdp-v7-full-service-trace-v2"
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
    "segment_origin_actual_departure_time_seconds",
    "segment_destination_arrival_time_seconds",
    "segment_road_runtime_seconds",
    "segment_decision_clock_runtime_seconds",
    "segment_event_clock_minus_road_runtime_seconds",
)


def full_service_trace_path(shard: Path) -> Path:
    shard = shard.expanduser().resolve()
    return shard.with_name(shard.stem + ".full_service.npz")


def decision_ready_time(service_completion_time: float) -> float:
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
    bus_to_code = {bus_id: index for index, bus_id in enumerate(bus_ids)}
    stop_rows = {
        name: [] for name in TRACE_ARRAY_NAMES if name.startswith("stop_")
    }
    segment_rows = {
        name: [] for name in TRACE_ARRAY_NAMES if name.startswith("segment_")
    }
    completion_quantization = []
    actual_departure_lag = []
    event_clock_minus_road = []
    per_line_trips = Counter()
    per_line_segments = Counter()

    for bus_id in bus_ids:
        bus = bridge.bus_obj_dic[bus_id]
        line_id = str(bus.belong_line_id_s)
        if line_id not in line_to_index:
            raise ValueError(f"trace bus {bus_id} has unknown line {line_id}")
        stops = [
            str(value) for value in bridge.line_obj_dic[line_id].stop_id_l
        ]
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
            ready = decision_ready_time(service_completion)
            base_dwell = service_completion - arrival
            actual_departure = float(departures[stop_id])
            if (
                not np.isfinite(arrival)
                or not np.isfinite(base_dwell)
                or base_dwell < 0.0
                or ready + 1e-9 < service_completion
                or actual_departure + 1e-9 < ready
            ):
                raise ValueError(f"trace trip {bus_id} has invalid stop timing")
            completion_quantization.append(ready - service_completion)
            actual_departure_lag.append(actual_departure - ready)
            trip_events.append(
                (
                    station_index,
                    arrival,
                    service_completion,
                    ready,
                    base_dwell,
                    actual_departure,
                )
            )
            stop_rows["stop_bus_code"].append(bus_to_code[bus_id])
            stop_rows["stop_line_index"].append(line_to_index[line_id])
            stop_rows["stop_station_index"].append(station_index)
            stop_rows["stop_arrival_time_seconds"].append(arrival)
            stop_rows["stop_service_completion_time_seconds"].append(
                service_completion
            )
            stop_rows["stop_decision_ready_time_seconds"].append(ready)
            stop_rows["stop_base_dwell_seconds"].append(base_dwell)
            stop_rows["stop_actual_departure_time_seconds"].append(
                actual_departure
            )

        if any(
            right[1] <= left[1]
            or right[3] <= left[3]
            or right[5] <= left[5]
            for left, right in zip(trip_events[:-1], trip_events[1:])
        ):
            raise ValueError(f"trace trip {bus_id} does not move forward in time")
        for left, right in zip(trip_events[:-1], trip_events[1:]):
            road_runtime = right[1] - left[5]
            decision_clock_runtime = right[3] - left[3] - right[4]
            clock_difference = decision_clock_runtime - road_runtime
            if (
                not np.isfinite(road_runtime)
                or not np.isfinite(decision_clock_runtime)
                or road_runtime <= 0.0
                or decision_clock_runtime <= 0.0
            ):
                raise ValueError(f"trace trip {bus_id} has nonpositive segment runtime")
            segment_rows["segment_bus_code"].append(bus_to_code[bus_id])
            segment_rows["segment_line_index"].append(line_to_index[line_id])
            segment_rows["segment_origin_station_index"].append(left[0])
            segment_rows[
                "segment_origin_decision_ready_time_seconds"
            ].append(left[3])
            segment_rows[
                "segment_origin_actual_departure_time_seconds"
            ].append(left[5])
            segment_rows["segment_destination_arrival_time_seconds"].append(
                right[1]
            )
            segment_rows["segment_road_runtime_seconds"].append(road_runtime)
            segment_rows["segment_decision_clock_runtime_seconds"].append(
                decision_clock_runtime
            )
            segment_rows[
                "segment_event_clock_minus_road_runtime_seconds"
            ].append(clock_difference)
            event_clock_minus_road.append(clock_difference)
            per_line_segments[line_id] += 1
        per_line_trips[line_id] += 1

    integer_types = {
        "stop_bus_code": np.int32,
        "stop_line_index": np.int16,
        "stop_station_index": np.int16,
        "segment_bus_code": np.int32,
        "segment_line_index": np.int16,
        "segment_origin_station_index": np.int16,
    }
    arrays = {
        name: np.asarray(values, dtype=integer_types.get(name, np.float64))
        for name, values in {**stop_rows, **segment_rows}.items()
    }
    if set(arrays) != set(TRACE_ARRAY_NAMES):
        raise AssertionError("full service trace array schema is incomplete")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("full service trace contains non-finite values")
    diagnostics = {
        "bus_code_to_id": bus_ids,
        "line_index_to_id": list(line_index_to_id),
        "completed_trips": len(bus_ids),
        "stop_events": len(arrays["stop_bus_code"]),
        "segments": len(arrays["segment_bus_code"]),
        "per_line_trips": dict(sorted(per_line_trips.items())),
        "per_line_segments": dict(sorted(per_line_segments.items())),
        "decision_ready_minus_exact_completion_seconds": {
            "minimum": float(np.min(completion_quantization)),
            "maximum": float(np.max(completion_quantization)),
            "mean": float(np.mean(completion_quantization)),
        },
        "actual_departure_minus_decision_ready_seconds": {
            "minimum": float(np.min(actual_departure_lag)),
            "maximum": float(np.max(actual_departure_lag)),
            "mean": float(np.mean(actual_departure_lag)),
        },
        "decision_clock_minus_road_runtime_seconds": {
            "minimum": float(np.min(event_clock_minus_road)),
            "maximum": float(np.max(event_clock_minus_road)),
            "mean": float(np.mean(event_clock_minus_road)),
        },
    }
    return arrays, diagnostics


def save_full_service_trace(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path = path.expanduser().resolve()
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(f"refusing to overwrite full service trace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256(path)


def write_full_service_summary(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite full service summary {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_full_service_trace(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = path.expanduser().resolve()
    summary_path = path.with_suffix(path.suffix + ".json")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != FULL_SERVICE_TRACE_SCHEMA:
        raise ValueError(f"unsupported full service trace: {path}")
    if summary.get("trace_sha256") != sha256(path):
        raise ValueError(f"full service trace hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as handle:
        if set(handle.files) != set(TRACE_ARRAY_NAMES):
            raise ValueError(f"full service trace array schema changed: {path}")
        arrays = {name: np.asarray(handle[name]) for name in TRACE_ARRAY_NAMES}
    stop_n = len(arrays["stop_bus_code"])
    segment_n = len(arrays["segment_bus_code"])
    for name, values in arrays.items():
        expected_n = stop_n if name.startswith("stop_") else segment_n
        if values.shape != (expected_n,) or not np.isfinite(values).all():
            raise ValueError(f"full service trace array is invalid: {path}:{name}")
    if (
        np.any(arrays["segment_road_runtime_seconds"] <= 0.0)
        or np.any(arrays["segment_decision_clock_runtime_seconds"] <= 0.0)
    ):
        raise ValueError(f"full service trace contains nonpositive runtimes: {path}")
    diagnostics = summary.get("full_service_trace", {})
    if diagnostics.get("stop_events") != stop_n or diagnostics.get(
        "segments"
    ) != segment_n:
        raise ValueError(f"full service trace row counts changed: {path}")
    if summary.get("bus_service_semantics_audit", {}).get("passed") is not True:
        raise ValueError(f"full service trace lacks a passing service audit: {path}")
    return summary, arrays


__all__ = [
    "FULL_SERVICE_TRACE_SCHEMA",
    "TRACE_ARRAY_NAMES",
    "decision_ready_time",
    "extract_full_service_arrays",
    "full_service_trace_path",
    "load_full_service_trace",
    "save_full_service_trace",
    "write_full_service_summary",
]
