#!/usr/bin/env python3
"""Build the formal v7 runtime model with correlated SUMO speed profiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
)
from reproducibility.build_simulator_runtime_calibration_v7_v3 import (
    CALIBRATION_SCHEMA_V3,
)
from reproducibility.collect_observation_speed_trace_v7 import TRACE_SCHEMA
from reproducibility.run_observation_speed_calibration_v7 import MATRIX_SCHEMA


CALIBRATION_SCHEMA_V4 = "transit-simulator-runtime-calibration-v7-v4"
PROFILE_SCHEMA = "transit-simulator-observation-speed-profiles-v7-v1"
EXPECTED_DEMAND_SCALES = (0.75, 1.0, 1.25)


def _logical_array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with fixed member metadata so its SHA is reproducible."""

    temporary = path.with_suffix(path.suffix + ".partial")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, values in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(values), allow_pickle=False
            )
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validated_parent(path: Path) -> dict[str, Any]:
    parent = json.loads(path.read_text())
    if (
        parent.get("schema_version") != CALIBRATION_SCHEMA_V3
        or parent.get("coverage_gate", {}).get("passed") is not True
    ):
        raise ValueError("v7-v4 requires a passing v7-v3 parent calibration")
    observation = parent.get("backward_headway_observation", {})
    if observation.get("coverage_gate", {}).get("passed") is not True:
        raise ValueError("v7-v3 backward-headway calibration is not passing")
    return parent


def _validated_trace(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    trace = Path(job["output"]).expanduser().resolve()
    summary_path = trace.with_suffix(trace.suffix + ".json")
    if not trace.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"speed-calibration cell is incomplete: {trace}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("schema_version") != TRACE_SCHEMA
        or summary.get("job_signature") != job.get("signature")
        or summary.get("trace_sha256") != sha256(trace)
        or summary.get("coverage_gate", {}).get("passed") is not True
        or int(summary.get("realization_block_id", -1))
        != int(job.get("realization_block_id", -2))
        or not np.isclose(
            float(summary.get("demand_scale", np.nan)),
            float(job.get("demand_scale", np.nan)),
        )
    ):
        raise ValueError(f"speed-calibration summary is stale or invalid: {summary_path}")
    with np.load(trace, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {
        "line_index",
        "station_index",
        "sim_time_seconds",
        "segment_mean_speed_mps",
        "physical_stop_code",
    }
    if not required.issubset(arrays):
        raise ValueError(f"speed trace lacks required arrays: {trace}")
    count = len(arrays["segment_mean_speed_mps"])
    if count <= 0 or any(len(values) != count for values in arrays.values()):
        raise ValueError(f"speed trace arrays are empty or misaligned: {trace}")
    return summary, arrays


def _profile_context(
    *,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    line_ids: list[str],
    expected_stops: dict[str, list[dict[str, Any]]],
) -> list[tuple[int, int, str, np.ndarray, np.ndarray]]:
    observed_line_ids = summary["event_identity"]["line_index_to_id"]
    if observed_line_ids != line_ids:
        raise ValueError("speed-trace line-index vocabulary changed")
    physical_vocabulary = summary["event_identity"]["physical_stop_code_to_id"]
    contexts = []
    for line_index, line_id in enumerate(line_ids):
        station_values = arrays["station_index"][arrays["line_index"] == line_index]
        expected_station_indices = list(range(len(expected_stops[line_id]) - 1))
        if sorted(set(int(value) for value in station_values)) != expected_station_indices:
            raise ValueError(
                f"speed trace does not cover every nonterminal stop of {line_id}"
            )
        for station_index in expected_station_indices:
            selected = (arrays["line_index"] == line_index) & (
                arrays["station_index"] == station_index
            )
            physical_ids = {
                physical_vocabulary[int(code)]
                for code in arrays["physical_stop_code"][selected]
            }
            expected_physical = expected_stops[line_id][station_index][
                "physical_stop_id"
            ]
            if physical_ids != {expected_physical}:
                raise ValueError(
                    f"speed trace physical stop changed for {line_id}:{station_index}"
                )
            times = arrays["sim_time_seconds"][selected].astype(np.float64)
            speeds = arrays["segment_mean_speed_mps"][selected].astype(np.float32)
            order = np.argsort(times, kind="stable")
            times = times[order]
            speeds = speeds[order]
            unique_times, first_indices, counts = np.unique(
                times, return_index=True, return_counts=True
            )
            if np.any(counts > 1):
                for start, count in zip(first_indices[counts > 1], counts[counts > 1]):
                    duplicate_speeds = speeds[start : start + count]
                    if np.ptp(duplicate_speeds.astype(np.float64)) > 1e-6:
                        raise ValueError(
                            f"simultaneous lane speeds disagree for "
                            f"{line_id}:{station_index}"
                        )
                speeds = speeds[first_indices]
                times = unique_times
            if (
                not len(times)
                or not np.isfinite(times).all()
                or not np.isfinite(speeds).all()
                or np.any(np.diff(times) <= 0.0)
                or np.any(speeds < 1.0)
            ):
                raise ValueError(
                    f"invalid observation-speed series for {line_id}:{station_index}"
                )
            contexts.append(
                (line_index, station_index, expected_physical, times, speeds)
            )
    return contexts


def build_calibration_v4(
    parent_calibration: Path,
    speed_matrix_manifest: Path,
    output: Path,
    profile_output: Path,
) -> dict[str, Any]:
    parent_calibration = parent_calibration.expanduser().resolve()
    speed_matrix_manifest = speed_matrix_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    profile_output = profile_output.expanduser().resolve()
    for path in (output, profile_output):
        if path.exists():
            raise FileExistsError(f"refusing to replace formal calibration output: {path}")
    if not parent_calibration.is_file() or not speed_matrix_manifest.is_file():
        raise FileNotFoundError("v7-v4 calibration inputs are missing")
    if output.parent != profile_output.parent:
        raise ValueError("runtime descriptor and speed profiles must share a directory")

    parent = _validated_parent(parent_calibration)
    matrix = json.loads(speed_matrix_manifest.read_text())
    expected_blocks = list(FORMAL_CALIBRATION_REALIZATION_BLOCKS)
    if (
        matrix.get("schema_version") != MATRIX_SCHEMA
        or matrix.get("study_stage") != "formal-calibration"
        or matrix.get("calibration_realization_blocks") != expected_blocks
        or sorted(float(value) for value in matrix.get("demand_scales", []))
        != list(EXPECTED_DEMAND_SCALES)
        or int(matrix.get("environment_cells", -1)) != 15
        or len(matrix.get("jobs", [])) != 15
    ):
        raise ValueError("observation-speed matrix has an invalid formal design")

    parent_lines = parent["backward_headway_observation"]["lines"]
    line_ids = sorted(parent_lines)
    expected_stops = {
        line_id: parent_lines[line_id]["stops"] for line_id in line_ids
    }
    global_physical_stops = sorted(
        {
            stop["physical_stop_id"]
            for stops in expected_stops.values()
            for stop in stops
        }
    )
    physical_stop_codes = {
        value: index for index, value in enumerate(global_physical_stops)
    }

    jobs = sorted(
        matrix["jobs"],
        key=lambda row: (
            int(row["realization_block_id"]),
            float(row["demand_scale"]),
        ),
    )
    observed_cells = [
        (int(job["realization_block_id"]), float(job["demand_scale"]))
        for job in jobs
    ]
    expected_cells = [
        (block, scale) for block in expected_blocks for scale in EXPECTED_DEMAND_SCALES
    ]
    if observed_cells != expected_cells:
        raise ValueError("observation-speed matrix does not contain the exact 5-by-3 grid")

    context_profile_index = []
    context_line_index = []
    context_station_index = []
    context_physical_stop_code = []
    context_offsets = [0]
    all_times = []
    all_speeds = []
    profiles = []
    trace_event_count = 0
    duplicate_events_removed = 0
    for profile_index, job in enumerate(jobs):
        summary, arrays = _validated_trace(job)
        contexts = _profile_context(
            arrays=arrays,
            summary=summary,
            line_ids=line_ids,
            expected_stops=expected_stops,
        )
        context_start = len(context_profile_index)
        retained_events = 0
        for line_index, station_index, physical_stop_id, times, speeds in contexts:
            context_profile_index.append(profile_index)
            context_line_index.append(line_index)
            context_station_index.append(station_index)
            context_physical_stop_code.append(physical_stop_codes[physical_stop_id])
            all_times.append(times)
            all_speeds.append(speeds)
            retained_events += len(times)
            context_offsets.append(context_offsets[-1] + len(times))
        raw_events = int(summary["events"]["control_decisions"])
        trace_event_count += raw_events
        duplicate_events_removed += raw_events - retained_events
        trace = Path(job["output"]).expanduser().resolve()
        summary_path = trace.with_suffix(trace.suffix + ".json")
        profiles.append(
            {
                "profile_index": profile_index,
                "realization_block_id": int(job["realization_block_id"]),
                "demand_scale": float(job["demand_scale"]),
                "context_start": context_start,
                "context_stop": len(context_profile_index),
                "contexts": len(contexts),
                "raw_control_decisions": raw_events,
                "retained_unique_context_times": retained_events,
                "source_trace": {
                    "path": str(trace),
                    "sha256": sha256(trace),
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256(summary_path),
                    "job_signature": job["signature"],
                },
            }
        )

    arrays = {
        "context_profile_index": np.asarray(context_profile_index, dtype=np.int16),
        "context_line_index": np.asarray(context_line_index, dtype=np.int16),
        "context_station_index": np.asarray(context_station_index, dtype=np.int16),
        "context_physical_stop_code": np.asarray(
            context_physical_stop_code, dtype=np.int16
        ),
        "context_offsets": np.asarray(context_offsets, dtype=np.int64),
        "sim_time_seconds": np.concatenate(all_times).astype(np.float64),
        "speed_mps": np.concatenate(all_speeds).astype(np.float32),
    }
    expected_contexts_per_profile = sum(
        len(stops) - 1 for stops in expected_stops.values()
    )
    checks = {
        "parent_runtime_calibration_passed": True,
        "exact_five_by_three_profile_grid": len(profiles) == 15,
        "all_profiles_cover_every_nonterminal_line_stop": all(
            profile["contexts"] == expected_contexts_per_profile
            for profile in profiles
        ),
        "packed_context_offsets_are_valid": bool(
            len(arrays["context_offsets"])
            == len(arrays["context_profile_index"]) + 1
            and arrays["context_offsets"][0] == 0
            and arrays["context_offsets"][-1] == len(arrays["speed_mps"])
            and np.all(np.diff(arrays["context_offsets"]) > 0)
        ),
        "all_profile_times_are_finite": bool(
            np.isfinite(arrays["sim_time_seconds"]).all()
        ),
        "all_profile_speeds_are_finite_and_at_least_one_mps": bool(
            np.isfinite(arrays["speed_mps"]).all()
            and np.all(arrays["speed_mps"] >= 1.0)
        ),
        "profile_line_vocabulary_matches_parent": line_ids == sorted(parent_lines),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v7-v4 speed-profile coverage failed: {checks}")

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(profile_output, arrays)
    profile_sha256 = sha256(profile_output)
    profile_logical_sha256 = _logical_array_hash(arrays)
    speeds64 = arrays["speed_mps"].astype(np.float64)
    profile_descriptor = {
        "schema_version": PROFILE_SCHEMA,
        "artifact": profile_output.name,
        "sha256": profile_sha256,
        "logical_array_sha256": profile_logical_sha256,
        "selection_semantics": (
            "condition on the exact demand scale; select one calibration "
            "realization per low-fidelity episode; within that realization query "
            "the nearest-in-time SUMO lane speed for the same line and physical stop"
        ),
        "line_index_to_id": line_ids,
        "physical_stop_code_to_id": global_physical_stops,
        "profiles": profiles,
        "arrays": {
            name: {"dtype": str(values.dtype), "shape": list(values.shape)}
            for name, values in arrays.items()
        },
        "coverage_gate": {
            "passed": True,
            "checks": checks,
            "profiles": len(profiles),
            "contexts_per_profile": expected_contexts_per_profile,
            "packed_contexts": len(arrays["context_profile_index"]),
            "raw_control_decisions": trace_event_count,
            "retained_unique_context_times": len(arrays["speed_mps"]),
            "duplicate_same-context-time_events_removed": duplicate_events_removed,
            "speed_mps": {
                "minimum": float(speeds64.min()),
                "maximum": float(speeds64.max()),
                "mean": float(speeds64.mean()),
                "standard_deviation": float(speeds64.std()),
                "quantiles": {
                    str(value): float(np.quantile(speeds64, value / 100.0))
                    for value in (1, 5, 50, 95, 99)
                },
                "network_limit_fraction": float(
                    np.mean(speeds64 >= 13.89 - 1e-6)
                ),
                "bridge_fallback_10mps_fraction": float(
                    np.mean(speeds64 == 10.0)
                ),
            },
        },
        "provenance": {
            "matrix_manifest": {
                "path": str(speed_matrix_manifest),
                "sha256": sha256(speed_matrix_manifest),
            },
            "zero_manifest": matrix["zero_manifest"],
            "zero_audit": matrix["zero_audit"],
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
    }

    result = copy.deepcopy(parent)
    result["schema_version"] = CALIBRATION_SCHEMA_V4
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["backward_headway_observation"]["low_fidelity_proxy"] = (
        "a hash-locked set of zero-hold SUMO observation-speed traces from the "
        "five registered calibration realizations, conditioned by demand scale, "
        "line, physical stop, and event time"
    )
    result["backward_headway_observation"]["dynamic_speed_profiles"] = (
        profile_descriptor
    )
    result["backward_headway_observation"]["coverage_gate"] = copy.deepcopy(
        profile_descriptor["coverage_gate"]
    )
    result["coverage_gate"] = copy.deepcopy(parent["coverage_gate"])
    result["coverage_gate"]["checks"] = {
        **dict(parent["coverage_gate"].get("checks", {})),
        "dynamic_observation_speed_profiles_passed": True,
    }
    result["coverage_gate"]["dynamic_observation_speed_profiles"] = (
        profile_descriptor["coverage_gate"]
    )
    result["coverage_gate"]["passed"] = all(
        result["coverage_gate"]["checks"].values()
    )
    result["provenance"] = {
        **dict(parent.get("provenance", {})),
        "parent_runtime_calibration": {
            "path": str(parent_calibration),
            "sha256": sha256(parent_calibration),
            "schema_version": CALIBRATION_SCHEMA_V3,
        },
        "observation_speed_matrix": profile_descriptor["provenance"],
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
    }
    _atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--speed-matrix-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path)
    args = parser.parse_args()
    profile_output = (
        args.profile_output
        if args.profile_output is not None
        else args.output.with_name("observation_speed_profiles_v7_v1.npz")
    )
    result = build_calibration_v4(
        args.parent_calibration,
        args.speed_matrix_manifest,
        args.output,
        profile_output,
    )
    print(
        json.dumps(
            result["backward_headway_observation"]["dynamic_speed_profiles"][
                "coverage_gate"
            ],
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
