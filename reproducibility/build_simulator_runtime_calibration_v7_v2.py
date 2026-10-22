#!/usr/bin/env python3
"""Build v7 simulator runtimes from frozen inline full-service SUMO traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.full_service_trace_v7 import (
    FULL_SERVICE_TRACE_SCHEMA,
    decision_ready_time,
    full_service_trace_path,
    load_full_service_trace,
)
from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION
from bus_h2o.sumo_provenance import sha256
from collect_policy.build_transit_v7_dataset import load_shard
from reproducibility.build_simulator_runtime_calibration_v7 import (
    _load_frozen_zero_audit,
    _zero_jobs,
)


CALIBRATION_SCHEMA_V2 = "transit-simulator-runtime-calibration-v7-v2"
SIMULATOR_FIXED_TRANSITION_OVERHEAD_SECONDS = 3


def _scale_key(value: float) -> str:
    return format(float(value), ".12g")


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("runtime calibration samples must be finite and positive")
    array.sort(kind="mergesort")
    return {
        "n": int(len(array)),
        "mean_seconds": float(array.mean()),
        "std_seconds": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median_seconds": float(np.median(array)),
        "p05_seconds": float(np.percentile(array, 5)),
        "p95_seconds": float(np.percentile(array, 95)),
        "min_seconds": float(array[0]),
        "max_seconds": float(array[-1]),
        "samples_seconds": [round(float(value), 3) for value in array],
    }


def validate_trace_arrays(
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    stops_by_line: dict[str, list[str]],
) -> tuple[list[tuple[str, int, float, float, float]], dict[str, Any]]:
    """Reconstruct both road and decision-clock runtimes from stop rows."""

    diagnostics = summary.get("full_service_trace", {})
    line_mapping = tuple(diagnostics.get("line_index_to_id", ()))
    bus_mapping = tuple(diagnostics.get("bus_code_to_id", ()))
    if set(line_mapping) != set(stops_by_line) or not bus_mapping:
        raise ValueError("full-service trace identity mapping is incomplete")
    stop_codes = arrays["stop_bus_code"]
    segment_codes = arrays["segment_bus_code"]
    if set(np.unique(stop_codes).tolist()) != set(range(len(bus_mapping))):
        raise ValueError("full-service stop rows do not cover every bus")
    if not set(np.unique(segment_codes).tolist()) <= set(range(len(bus_mapping))):
        raise ValueError("full-service segment rows contain an unknown bus")

    samples = []
    trips_by_line = Counter()
    segments_by_line = Counter()
    maximum_decision_runtime_error = 0.0
    maximum_road_runtime_error = 0.0
    maximum_clock_difference_error = 0.0
    departure_lags = []
    clock_differences = []
    low_fidelity_clock_mapping_errors = []
    for bus_code, bus_id in enumerate(bus_mapping):
        stop_rows = np.flatnonzero(stop_codes == bus_code)
        if not len(stop_rows):
            raise ValueError(f"full-service trace has no stops for {bus_id}")
        line_indices = np.unique(arrays["stop_line_index"][stop_rows])
        if len(line_indices) != 1:
            raise ValueError(f"full-service bus changes line: {bus_id}")
        line_index = int(line_indices[0])
        if line_index < 0 or line_index >= len(line_mapping):
            raise ValueError(f"full-service bus has invalid line index: {bus_id}")
        line_id = line_mapping[line_index]
        stops = stops_by_line[line_id]
        stations = arrays["stop_station_index"][stop_rows]
        if not np.array_equal(stations, np.arange(len(stops))):
            raise ValueError(f"full-service bus has a station gap: {bus_id}")

        arrivals = arrays["stop_arrival_time_seconds"][stop_rows]
        completions = arrays["stop_service_completion_time_seconds"][stop_rows]
        ready = arrays["stop_decision_ready_time_seconds"][stop_rows]
        base_dwell = arrays["stop_base_dwell_seconds"][stop_rows]
        departures = arrays["stop_actual_departure_time_seconds"][stop_rows]
        expected_ready = np.asarray(
            [decision_ready_time(value) for value in completions],
            dtype=np.float64,
        )
        if (
            not np.allclose(completions - arrivals, base_dwell, rtol=0.0, atol=1e-9)
            or not np.array_equal(ready, expected_ready)
            or np.any(np.diff(arrivals) <= 0.0)
            or np.any(np.diff(ready) <= 0.0)
            or np.any(departures + 1e-9 < ready)
        ):
            raise ValueError(f"full-service stop timing identity failed: {bus_id}")
        departure_lags.extend((departures - ready).tolist())

        segment_rows = np.flatnonzero(segment_codes == bus_code)
        expected_stations = np.arange(len(stops) - 1)
        if (
            len(segment_rows) != len(expected_stations)
            or not np.array_equal(
                arrays["segment_origin_station_index"][segment_rows],
                expected_stations,
            )
            or not np.all(arrays["segment_line_index"][segment_rows] == line_index)
        ):
            raise ValueError(f"full-service segment sequence is incomplete: {bus_id}")

        decision_runtime = ready[1:] - ready[:-1] - base_dwell[1:]
        road_runtime = arrivals[1:] - departures[:-1]
        clock_difference = decision_runtime - road_runtime
        recorded_decision = arrays[
            "segment_decision_clock_runtime_seconds"
        ][segment_rows]
        recorded_road = arrays["segment_road_runtime_seconds"][segment_rows]
        recorded_difference = arrays[
            "segment_event_clock_minus_road_runtime_seconds"
        ][segment_rows]
        decision_errors = np.abs(recorded_decision - decision_runtime)
        road_errors = np.abs(recorded_road - road_runtime)
        difference_errors = np.abs(recorded_difference - clock_difference)
        maximum_decision_runtime_error = max(
            maximum_decision_runtime_error,
            float(decision_errors.max()) if len(decision_errors) else 0.0,
        )
        maximum_road_runtime_error = max(
            maximum_road_runtime_error,
            float(road_errors.max()) if len(road_errors) else 0.0,
        )
        maximum_clock_difference_error = max(
            maximum_clock_difference_error,
            float(difference_errors.max()) if len(difference_errors) else 0.0,
        )
        if (
            np.any(decision_runtime <= 0.0)
            or np.any(road_runtime <= 0.0)
            or not np.allclose(recorded_decision, decision_runtime, rtol=0.0, atol=1e-9)
            or not np.allclose(recorded_road, road_runtime, rtol=0.0, atol=1e-9)
            or not np.allclose(recorded_difference, clock_difference, rtol=0.0, atol=1e-9)
            or not np.array_equal(
                arrays["segment_origin_decision_ready_time_seconds"][segment_rows],
                ready[:-1],
            )
            or not np.array_equal(
                arrays["segment_origin_actual_departure_time_seconds"][segment_rows],
                departures[:-1],
            )
            or not np.array_equal(
                arrays["segment_destination_arrival_time_seconds"][segment_rows],
                arrivals[1:],
            )
        ):
            raise ValueError(f"full-service segment timing identity failed: {bus_id}")
        destination_quantization = ready[1:] - completions[1:]
        calibrated_travel_steps = np.maximum(
            np.rint(decision_runtime).astype(np.int64)
            - SIMULATOR_FIXED_TRANSITION_OVERHEAD_SECONDS,
            1,
        )
        reconstructed_event_runtime = (
            calibrated_travel_steps
            + SIMULATOR_FIXED_TRANSITION_OVERHEAD_SECONDS
            + destination_quantization
        )
        low_fidelity_clock_mapping_errors.extend(
            (reconstructed_event_runtime - decision_runtime).tolist()
        )
        for station, origin_time, decision_value, road_value in zip(
            expected_stations, ready[:-1], decision_runtime, road_runtime
        ):
            samples.append(
                (
                    line_id,
                    int(station),
                    float(origin_time),
                    float(decision_value),
                    float(road_value),
                )
            )
        clock_differences.extend(clock_difference.tolist())
        trips_by_line[line_id] += 1
        segments_by_line[line_id] += len(segment_rows)

    if len(samples) != len(segment_codes):
        raise ValueError("full-service trace contains unassigned segment rows")
    if diagnostics.get("completed_trips") != len(bus_mapping):
        raise ValueError("full-service summary trip count disagrees with arrays")
    if diagnostics.get("stop_events") != len(stop_codes) or diagnostics.get(
        "segments"
    ) != len(segment_codes):
        raise ValueError("full-service summary row counts disagree with arrays")
    return samples, {
        "completed_trips": len(bus_mapping),
        "stop_events": len(stop_codes),
        "segments": len(segment_codes),
        "trips_by_line": dict(sorted(trips_by_line.items())),
        "segments_by_line": dict(sorted(segments_by_line.items())),
        "maximum_decision_runtime_reconstruction_error_seconds": maximum_decision_runtime_error,
        "maximum_road_runtime_reconstruction_error_seconds": maximum_road_runtime_error,
        "maximum_clock_difference_reconstruction_error_seconds": maximum_clock_difference_error,
        "actual_departure_minus_decision_ready_seconds": {
            "minimum": float(np.min(departure_lags)),
            "maximum": float(np.max(departure_lags)),
            "mean": float(np.mean(departure_lags)),
        },
        "decision_clock_minus_road_runtime_seconds": {
            "minimum": float(np.min(clock_differences)),
            "maximum": float(np.max(clock_differences)),
            "mean": float(np.mean(clock_differences)),
        },
        "low_fidelity_event_clock_mapping_error_seconds": {
            "minimum": float(np.min(low_fidelity_clock_mapping_errors)),
            "maximum": float(np.max(low_fidelity_clock_mapping_errors)),
            "mean": float(np.mean(low_fidelity_clock_mapping_errors)),
            "mean_absolute": float(
                np.mean(np.abs(low_fidelity_clock_mapping_errors))
            ),
            "p95_absolute": float(
                np.percentile(np.abs(low_fidelity_clock_mapping_errors), 95)
            ),
            "maximum_absolute": float(
                np.max(np.abs(low_fidelity_clock_mapping_errors))
            ),
        },
    }


def build_calibration_v2(
    matrix_manifest: Path,
    simulator_data: Path,
    output: Path,
    *,
    minimum_samples_per_segment: int = 10,
) -> dict[str, Any]:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    simulator_data = simulator_data.expanduser().resolve()
    output = output.expanduser().resolve()
    if minimum_samples_per_segment <= 0:
        raise ValueError("minimum samples per segment must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to replace runtime calibration: {output}")

    freeze_audit_path, freeze_audit = _load_frozen_zero_audit(matrix_manifest)
    matrix, jobs, partition = _zero_jobs(matrix_manifest, allow_partial=False)
    artifact_set = freeze_audit.get("artifact_set", {})
    if (
        artifact_set.get("exact_match") is not True
        or int(artifact_set.get("registered_files", -1)) != 150
        or len(freeze_audit.get("shards", [])) != 30
    ):
        raise ValueError("frozen zero matrix does not bind the 150-file trace set")

    line_dirs = sorted(
        path
        for path in simulator_data.iterdir()
        if path.is_dir()
        and (path / "stop_news.xlsx").is_file()
        and (path / "route_news.xlsx").is_file()
    )
    stops_by_line = {
        path.name: [
            str(value)
            for value in pd.read_excel(path / "stop_news.xlsx")["stop_name"].tolist()
        ]
        for path in line_dirs
    }
    if not stops_by_line:
        raise ValueError("simulator data contains no line directories")

    samples = defaultdict(lambda: defaultdict(list))
    road_samples = defaultdict(lambda: defaultdict(list))
    samples_by_demand_scale = defaultdict(lambda: defaultdict(list))
    trace_sources = []
    reference_line_mapping = None
    total_trips = 0
    total_stops = 0
    total_segments = 0
    maximum_reconstruction_error = 0.0
    maximum_low_fidelity_clock_mapping_error = 0.0
    clock_differences = []
    for job in sorted(jobs, key=lambda value: int(value["block_id"])):
        shard = Path(job["output"]).expanduser().resolve()
        shard_summary, _ = load_shard(shard, allow_partial=False)
        trace = full_service_trace_path(shard)
        trace_summary, arrays = load_full_service_trace(trace)
        descriptor = shard_summary.get("full_service_trace", {})
        mismatches = {}
        if shard_summary.get("job_signature") != job.get("signature"):
            mismatches["job_signature"] = "matrix disagreement"
        if shard_summary.get("source_sha256") != matrix.get("source_sha256"):
            mismatches["source_sha256"] = "matrix disagreement"
        if shard_summary.get("bus_service_semantics_audit", {}).get("passed") is not True:
            mismatches["bus_service_semantics_audit"] = "did not pass"
        if (
            descriptor.get("schema_version") != FULL_SERVICE_TRACE_SCHEMA
            or Path(descriptor.get("path", "")).expanduser().resolve() != trace
            or descriptor.get("sha256") != sha256(trace)
            or descriptor.get("summary_sha256")
            != sha256(trace.with_suffix(trace.suffix + ".json"))
            or trace_summary.get("source_shard", {}).get("sha256")
            != shard_summary.get("shard_sha256")
            or trace_summary.get("source_shard", {}).get("job_signature")
            != job.get("signature")
            or trace_summary.get("collection_source_sha256")
            != matrix.get("source_sha256")
            or trace_summary.get("exact_zero_hold") is not True
        ):
            mismatches["full_service_trace"] = "trace/shard disagreement"
        if mismatches:
            raise ValueError(f"full-service trace provenance mismatch: {trace}: {mismatches}")

        trace_samples, diagnostics = validate_trace_arrays(
            trace_summary, arrays, stops_by_line
        )
        line_mapping = tuple(
            trace_summary["full_service_trace"]["line_index_to_id"]
        )
        if reference_line_mapping is None:
            reference_line_mapping = line_mapping
        elif line_mapping != reference_line_mapping:
            raise ValueError("full-service traces disagree on line-index semantics")
        demand_scale = float(job["scenario"]["demand_scale"])
        for line_id, station, origin_time, runtime, road_runtime in trace_samples:
            period = int(origin_time // 3600.0)
            samples[(line_id, station)][period].append(runtime)
            road_samples[(line_id, station)][period].append(road_runtime)
            samples_by_demand_scale[(line_id, station, demand_scale)][
                period
            ].append(runtime)
        total_trips += diagnostics["completed_trips"]
        total_stops += diagnostics["stop_events"]
        total_segments += diagnostics["segments"]
        maximum_reconstruction_error = max(
            maximum_reconstruction_error,
            diagnostics[
                "maximum_decision_runtime_reconstruction_error_seconds"
            ],
            diagnostics["maximum_road_runtime_reconstruction_error_seconds"],
            diagnostics[
                "maximum_clock_difference_reconstruction_error_seconds"
            ],
        )
        maximum_low_fidelity_clock_mapping_error = max(
            maximum_low_fidelity_clock_mapping_error,
            diagnostics[
                "low_fidelity_event_clock_mapping_error_seconds"
            ]["maximum_absolute"],
        )
        clock_differences.append(
            diagnostics["decision_clock_minus_road_runtime_seconds"]
        )
        trace_sources.append(
            {
                "path": str(trace),
                "sha256": trace_summary["trace_sha256"],
                "summary_sha256": sha256(
                    trace.with_suffix(trace.suffix + ".json")
                ),
                "source_shard": str(shard),
                "source_shard_sha256": shard_summary["shard_sha256"],
                "job_signature": job["signature"],
                "realization_block_id": int(job["realization_block_id"]),
                "demand_scale": demand_scale,
                **diagnostics,
            }
        )

    lines = {line_id: {"segments": {}} for line_id in sorted(stops_by_line)}
    missing_segments = {}
    below_minimum_segments = {}
    missing_demand_segments = {}
    below_minimum_demand_segments = {}
    expected_segment_count = 0
    total_samples = 0
    for line_id, stops in sorted(stops_by_line.items()):
        expected_segment_count += len(stops) - 1
        missing = []
        below = {}
        for station in range(len(stops) - 1):
            segment = f"{stops[station]}->{stops[station + 1]}"
            by_hour = samples.get((line_id, station), {})
            road_by_hour = road_samples.get((line_id, station), {})
            combined = [value for values in by_hour.values() for value in values]
            combined_road = [
                value for values in road_by_hour.values() for value in values
            ]
            if not combined or len(combined) != len(combined_road):
                missing.append(segment)
                continue
            distribution = _distribution(combined)
            total_samples += distribution["n"]
            lines[line_id]["segments"][segment] = {
                "source_station_index": station,
                "runtime_semantics": "decision-clock runtime used by the low-fidelity state machine",
                "all_periods": distribution,
                "by_service_hour": {
                    str(period): _distribution(values)
                    for period, values in sorted(by_hour.items())
                },
                "road_runtime_audit": _distribution(combined_road),
                "by_demand_scale": {},
            }
            for demand_scale in partition["demand_scales"]:
                scale = float(demand_scale)
                key = _scale_key(scale)
                scale_by_hour = samples_by_demand_scale.get(
                    (line_id, station, scale), {}
                )
                scale_combined = [
                    value for values in scale_by_hour.values() for value in values
                ]
                if not scale_combined:
                    missing_demand_segments.setdefault(key, {}).setdefault(
                        line_id, []
                    ).append(segment)
                    continue
                scale_distribution = _distribution(scale_combined)
                lines[line_id]["segments"][segment]["by_demand_scale"][key] = {
                    "all_periods": scale_distribution,
                    "by_service_hour": {
                        str(period): _distribution(values)
                        for period, values in sorted(scale_by_hour.items())
                    },
                }
                if scale_distribution["n"] < minimum_samples_per_segment:
                    below_minimum_demand_segments.setdefault(key, {}).setdefault(
                        line_id, {}
                    )[segment] = scale_distribution["n"]
            if distribution["n"] < minimum_samples_per_segment:
                below[segment] = distribution["n"]
        if missing:
            missing_segments[line_id] = missing
        if below:
            below_minimum_segments[line_id] = below

    checks = {
        "frozen_zero_artifact_set_is_exact": artifact_set.get("exact_match") is True,
        "one_trace_per_calibration_cell": len(trace_sources)
        == int(partition["calibration_environment_cells"]),
        "calibration_and_fidelity_blocks_disjoint": not (
            set(partition["calibration_realization_blocks"])
            & set(partition["reserved_fidelity_realization_blocks"])
        ),
        "all_trace_timing_identities_reconstructed": maximum_reconstruction_error
        <= 1e-9,
        "low_fidelity_event_clock_mapping_within_one_step": (
            maximum_low_fidelity_clock_mapping_error <= 1.0
        ),
        "all_simulator_segments_calibrated": not missing_segments,
        "minimum_samples_per_segment_met": not below_minimum_segments,
        "all_registered_demand_scales_calibrated": not missing_demand_segments,
        "minimum_samples_per_demand_scale_met": not below_minimum_demand_segments,
        "sample_count_matches_full_service_segments": total_samples
        == total_segments,
    }
    coverage_gate = {
        "minimum_samples_per_segment": minimum_samples_per_segment,
        "expected_segments": expected_segment_count,
        "calibrated_segments": sum(
            len(value["segments"]) for value in lines.values()
        ),
        "missing_segments": missing_segments,
        "below_minimum_segments": below_minimum_segments,
        "missing_demand_conditioned_segments": missing_demand_segments,
        "below_minimum_demand_conditioned_segments": below_minimum_demand_segments,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result = {
        "schema_version": CALIBRATION_SCHEMA_V2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_schema": SCHEMA_VERSION,
        "semantics": (
            "destination decision-ready time minus origin decision-ready time "
            "minus destination base passenger-exchange dwell, reconstructed for "
            "every consecutive stop pair of every completed zero-hold SUMO trip"
        ),
        "road_runtime_audit_semantics": (
            "destination arrival time minus origin actual departure time; retained "
            "as an independent physical audit and not substituted for event-clock runtime"
        ),
        "decision_ready_time_semantics": (
            "ceil(exact service-completion time - 1e-9), matching the integer SUMO event clock"
        ),
        "simulator_fixed_transition_overhead_seconds": SIMULATOR_FIXED_TRANSITION_OVERHEAD_SECONDS,
        "simulator_fixed_transition_overhead_semantics": (
            "the low-fidelity one-second state machine consumes one action tick, one "
            "departure tick, and one arrival-detection tick outside its calibrated travel counter"
        ),
        "sampling": (
            "sample with replacement by line, segment, registered demand scale, and "
            "service hour; sparse hours fall back to demand-scale all-periods and "
            "unregistered demand scales fall back to pooled samples"
        ),
        "line_index_order": list(reference_line_mapping or ()),
        "data_partition": partition,
        "lines": lines,
        "artifact_set": artifact_set,
        "coverage_gate": coverage_gate,
        "diagnostics": {
            "full_service_traces": len(trace_sources),
            "completed_bus_trajectories": total_trips,
            "stop_events": total_stops,
            "runtime_samples": total_samples,
            "maximum_timing_reconstruction_error_seconds": maximum_reconstruction_error,
            "maximum_low_fidelity_event_clock_mapping_error_seconds": (
                maximum_low_fidelity_clock_mapping_error
            ),
            "per_trace_decision_clock_minus_road_runtime_seconds": clock_differences,
        },
        "provenance": {
            "collection_matrix": str(matrix_manifest),
            "collection_matrix_sha256": sha256(matrix_manifest),
            "frozen_zero_matrix_audit": {
                "path": str(freeze_audit_path),
                "sha256": sha256(freeze_audit_path),
                "schema_version": freeze_audit["schema_version"],
            },
            "collection_source_sha256": matrix.get("source_sha256"),
            "sumo_version": matrix.get("sumo_version"),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "trace_sources": trace_sources,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--simulator-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-samples-per-segment", type=int, default=10)
    args = parser.parse_args()
    result = build_calibration_v2(
        args.matrix_manifest,
        args.simulator_data,
        args.output,
        minimum_samples_per_segment=args.minimum_samples_per_segment,
    )
    print(json.dumps(result["diagnostics"], sort_keys=True))
    if not result["coverage_gate"]["passed"]:
        raise RuntimeError(
            "formal v7 v2 runtime calibration failed its coverage gate; "
            f"inspect {args.output.expanduser().resolve()}"
        )


if __name__ == "__main__":
    main()
