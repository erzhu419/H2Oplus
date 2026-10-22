#!/usr/bin/env python3
"""Calibrate simulator runtimes from same-bus zero-hold SUMO v7 events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION
from bus_h2o.sumo_provenance import sha256
from collect_policy.build_transit_v7_dataset import load_shard


CALIBRATION_SCHEMA = "transit-simulator-runtime-calibration-v7"
MATRIX_SCHEMA = "transit-smdp-v7-collection-matrix-v2"
FROZEN_ZERO_AUDIT_SCHEMA = "transit-smdp-v7-frozen-zero-matrix-v2"
FORMAL_CALIBRATION_REALIZATION_BLOCKS = tuple(range(5))
FORMAL_FIDELITY_REALIZATION_BLOCKS = tuple(range(5, 10))


def _distribution(values: list[float]) -> dict:
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


def _zero_jobs(
    matrix_manifest: Path, *, allow_partial: bool
) -> tuple[dict, list[dict], dict]:
    manifest = json.loads(matrix_manifest.read_text())
    if manifest.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("v7 collection matrix has an unsupported schema")
    if not allow_partial and manifest.get("study_stage") != "formal":
        raise ValueError("formal calibration requires a formal v7 collection matrix")
    all_zero_jobs = [
        job
        for job in manifest.get("jobs", [])
        if job.get("policy", {}).get("policy_name") == "zero"
    ]
    if not all_zero_jobs:
        raise ValueError("collection matrix contains no zero-hold jobs")
    all_blocks = sorted(
        {int(job["realization_block_id"]) for job in all_zero_jobs}
    )
    if allow_partial:
        calibration_blocks = tuple(all_blocks)
        fidelity_blocks: tuple[int, ...] = ()
    else:
        if int(manifest.get("independent_realization_blocks", -1)) != 10:
            raise ValueError(
                "formal v7 calibration requires ten independent realization blocks"
            )
        expected_blocks = set(
            FORMAL_CALIBRATION_REALIZATION_BLOCKS
            + FORMAL_FIDELITY_REALIZATION_BLOCKS
        )
        if set(all_blocks) != expected_blocks:
            raise ValueError(
                "formal zero-hold collection does not cover the registered blocks"
            )
        calibration_blocks = FORMAL_CALIBRATION_REALIZATION_BLOCKS
        fidelity_blocks = FORMAL_FIDELITY_REALIZATION_BLOCKS
    if set(calibration_blocks) & set(fidelity_blocks):
        raise AssertionError("runtime calibration and fidelity blocks overlap")
    jobs = [
        job
        for job in all_zero_jobs
        if int(job["realization_block_id"]) in set(calibration_blocks)
    ]
    if not jobs:
        raise ValueError("runtime calibration partition contains no zero-hold jobs")
    outputs = [Path(job["output"]).expanduser().resolve() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("zero-hold collection jobs repeat an output path")
    signatures = [job["signature"] for job in jobs]
    if len(signatures) != len(set(signatures)):
        raise ValueError("zero-hold collection jobs repeat a signature")
    demand_scales = sorted(
        {float(job["scenario"]["demand_scale"]) for job in all_zero_jobs}
    )
    expected_calibration_cells = {
        (block, scale)
        for block in calibration_blocks
        for scale in demand_scales
    }
    actual_calibration_cells = [
        (
            int(job["realization_block_id"]),
            float(job["scenario"]["demand_scale"]),
        )
        for job in jobs
    ]
    if (
        len(actual_calibration_cells) != len(expected_calibration_cells)
        or set(actual_calibration_cells) != expected_calibration_cells
    ):
        raise ValueError(
            "runtime calibration partition is missing or duplicates an environment cell"
        )
    partition = {
        "calibration_realization_blocks": list(calibration_blocks),
        "reserved_fidelity_realization_blocks": list(fidelity_blocks),
        "demand_scales": demand_scales,
        "calibration_environment_cells": len(expected_calibration_cells),
        "selection_rule": (
            "formal blocks 0-4 fit segment runtimes; blocks 5-9 are excluded "
            "from fitting and reserved for the simulator fidelity acceptance audit"
            if not allow_partial
            else "pilot calibration uses every block present in the supplied matrix"
        ),
    }
    return manifest, jobs, partition


def _load_frozen_zero_audit(matrix_manifest: Path) -> tuple[Path, dict]:
    audit_path = matrix_manifest.with_name(matrix_manifest.stem + ".audit.json")
    if not audit_path.is_file():
        raise ValueError(
            "formal v7 calibration requires the frozen zero-matrix audit artifact"
        )
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema_version") != FROZEN_ZERO_AUDIT_SCHEMA
        or audit.get("passed") is not True
    ):
        raise ValueError("formal v7 zero-matrix freeze audit did not pass")
    descriptor = audit.get("frozen_manifest", {})
    if (
        Path(descriptor.get("path", "")).expanduser().resolve()
        != matrix_manifest.resolve()
        or descriptor.get("sha256") != sha256(matrix_manifest)
    ):
        raise ValueError("formal v7 freeze audit does not bind this matrix manifest")
    if audit.get("artifact_set", {}).get("exact_match") is not True:
        raise ValueError("formal v7 freeze audit did not close the zero artifact set")
    topology = audit.get("topology", {})
    if topology.get("calibration_realization_blocks") != list(
        FORMAL_CALIBRATION_REALIZATION_BLOCKS
    ) or topology.get("reserved_fidelity_realization_blocks") != list(
        FORMAL_FIDELITY_REALIZATION_BLOCKS
    ):
        raise ValueError("formal v7 freeze audit has an unregistered data partition")
    if len(audit.get("shards", [])) != 30:
        raise ValueError("formal v7 freeze audit does not bind all 30 zero-hold shards")
    return audit_path, audit


def build_calibration(
    matrix_manifest: Path,
    simulator_data: Path,
    output: Path,
    *,
    allow_partial: bool = False,
    minimum_samples_per_segment: int = 10,
) -> dict:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    simulator_data = simulator_data.expanduser().resolve()
    output = output.expanduser().resolve()
    if minimum_samples_per_segment <= 0:
        raise ValueError("minimum samples per segment must be positive")
    freeze_audit_path = None
    freeze_audit = None
    if not allow_partial:
        freeze_audit_path, freeze_audit = _load_frozen_zero_audit(matrix_manifest)
    manifest, jobs, data_partition = _zero_jobs(
        matrix_manifest, allow_partial=allow_partial
    )
    line_dirs = sorted(
        path
        for path in simulator_data.iterdir()
        if path.is_dir()
        and (path / "stop_news.xlsx").is_file()
        and (path / "route_news.xlsx").is_file()
    )
    line_names = [path.name for path in line_dirs]
    if not line_names:
        raise ValueError("simulator data contains no line directories")
    stops_by_line = {
        path.name: [
            str(value)
            for value in pd.read_excel(path / "stop_news.xlsx")["stop_name"].tolist()
        ]
        for path in line_dirs
    }

    samples = defaultdict(lambda: defaultdict(list))
    source_shards = []
    rejected_nonpositive = 0
    skipped_multisegment = 0
    adjacent_same_bus_pairs = 0
    observed_bus_trajectories = 0
    reference_line_mapping = None
    for job in sorted(jobs, key=lambda value: value["output"]):
        shard = Path(job["output"]).expanduser().resolve()
        summary, arrays = load_shard(shard, allow_partial=allow_partial)
        expected = {
            "job_signature": job["signature"],
            "policy_name": "zero",
            "policy_seed": int(job["policy_seed"]),
            "sumo_seed": int(job["sumo_seed"]),
        }
        mismatches = {
            name: (summary.get(name), value)
            for name, value in expected.items()
            if summary.get(name) != value
        }
        if summary.get("source_sha256") != manifest.get("source_sha256"):
            mismatches["source_sha256"] = "collection matrix disagreement"
        if summary.get("runtime", {}).get("sumo_version") != manifest.get(
            "sumo_version"
        ):
            mismatches["sumo_version"] = "collection matrix disagreement"
        if mismatches:
            raise ValueError(f"zero-hold shard provenance mismatch: {shard}: {mismatches}")
        if not np.all(arrays["actions"][:, 0] == -1.0):
            raise ValueError(f"zero-hold calibration shard has nonzero actions: {shard}")
        identity = summary["event_identity"]
        line_mapping = tuple(identity["line_index_to_id"])
        if reference_line_mapping is None:
            reference_line_mapping = line_mapping
        elif line_mapping != reference_line_mapping:
            raise ValueError("v7 zero-hold shards disagree on line-index semantics")
        if set(line_mapping) != set(line_names):
            raise ValueError(
                "SUMO and simulator line sets differ: "
                f"SUMO={sorted(line_mapping)}, simulator={line_names}"
            )

        for bus_code in np.unique(arrays["event_bus_code"]):
            selected = np.flatnonzero(arrays["event_bus_code"] == bus_code)
            if len(selected) < 2:
                continue
            observed_bus_trajectories += 1
            times = arrays["event_decision_time_seconds"][selected]
            if np.any(np.diff(times) <= 0.0):
                bus_id = identity["bus_code_to_id"][int(bus_code)]
                raise ValueError(
                    f"same-bus v7 event times are not strictly increasing: "
                    f"{shard}:{bus_id}"
                )
            for left, right in zip(selected[:-1], selected[1:]):
                adjacent_same_bus_pairs += 1
                line = int(arrays["event_line_index"][left])
                next_line = int(arrays["event_line_index"][right])
                station = int(arrays["event_station_index"][left])
                next_station = int(arrays["event_station_index"][right])
                if line != next_line:
                    raise ValueError(f"same SUMO bus changes line within {shard}")
                if next_station <= station:
                    raise ValueError(
                        f"same SUMO bus moves backward within {shard}: "
                        f"{station}->{next_station}"
                    )
                if next_station != station + 1:
                    skipped_multisegment += 1
                    continue

                line_name = line_mapping[line]
                if station < 0 or station + 1 >= len(stops_by_line[line_name]):
                    raise ValueError(
                        f"station {station} is invalid for line {line_name} "
                        f"in {shard}"
                    )
                elapsed = float(
                    arrays["event_decision_time_seconds"][right]
                    - arrays["event_decision_time_seconds"][left]
                )
                next_base_dwell = float(
                    arrays["event_base_dwell_seconds"][right]
                )
                travel = elapsed - next_base_dwell
                if travel <= 0.0 or not np.isfinite(travel):
                    rejected_nonpositive += 1
                    continue
                period = int(
                    arrays["event_decision_time_seconds"][left] // 3600.0
                )
                segment = (
                    f"{stops_by_line[line_name][station]}->"
                    f"{stops_by_line[line_name][station + 1]}"
                )
                samples[(line_name, segment, station)][period].append(travel)
        source_shards.append(
            {
                "path": str(shard),
                "sha256": summary["shard_sha256"],
                "job_signature": summary["job_signature"],
                "sumo_seed": int(summary["sumo_seed"]),
                "realization_block_id": int(job["realization_block_id"]),
                "demand_scale": float(job["scenario"]["demand_scale"]),
                "transitions": int(summary["episode"]["transitions"]),
                "event_bus_id_count": len(identity["bus_code_to_id"]),
                "line_index_to_id": list(line_mapping),
            }
        )

    lines = {line_name: {"segments": {}} for line_name in line_names}
    total_samples = 0
    for (line_name, segment, station), by_hour in sorted(samples.items()):
        combined = [value for values in by_hour.values() for value in values]
        total_samples += len(combined)
        lines[line_name]["segments"][segment] = {
            "source_station_index": station,
            "all_periods": _distribution(combined),
            "by_service_hour": {
                str(period): _distribution(values)
                for period, values in sorted(by_hour.items())
            },
        }

    missing_segments = {}
    below_minimum_segments = {}
    expected_segment_count = 0
    for line_name, stops in sorted(stops_by_line.items()):
        expected = [
            f"{stops[index]}->{stops[index + 1]}"
            for index in range(len(stops) - 1)
        ]
        expected_segment_count += len(expected)
        missing = sorted(set(expected) - set(lines[line_name]["segments"]))
        if missing:
            missing_segments[line_name] = missing
        below = {
            segment: int(lines[line_name]["segments"][segment]["all_periods"]["n"])
            for segment in expected
            if segment in lines[line_name]["segments"]
            and int(lines[line_name]["segments"][segment]["all_periods"]["n"])
            < minimum_samples_per_segment
        }
        if below:
            below_minimum_segments[line_name] = below
    coverage_checks = {
        "one_zero_hold_shard_per_calibration_cell": len(source_shards)
        == int(data_partition["calibration_environment_cells"]),
        "calibration_and_fidelity_blocks_disjoint": not (
            set(data_partition["calibration_realization_blocks"])
            & set(data_partition["reserved_fidelity_realization_blocks"])
        ),
        "all_simulator_segments_calibrated": not missing_segments,
        "minimum_samples_per_segment_met": not below_minimum_segments,
        "no_nonpositive_runtime_samples": rejected_nonpositive == 0,
        "no_same_bus_station_gaps": skipped_multisegment == 0,
    }
    coverage_gate = {
        "minimum_samples_per_segment": minimum_samples_per_segment,
        "expected_segments": expected_segment_count,
        "missing_segments": missing_segments,
        "below_minimum_segments": below_minimum_segments,
        "checks": coverage_checks,
        "passed": all(coverage_checks.values()),
    }

    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_schema": SCHEMA_VERSION,
        "semantics": (
            "for each identified SUMO bus, empirical service-completion time at "
            "station k+1 minus service-completion time at station k and minus "
            "the k+1 base passenger-exchange dwell; source actions are exact "
            "zero hold"
        ),
        "simulator_fixed_transition_overhead_seconds": 3,
        "simulator_fixed_transition_overhead_semantics": (
            "the low-fidelity one-second state machine consumes one action tick, "
            "one departure tick, and one arrival-detection tick outside its "
            "calibrated travel-step counter"
        ),
        "sampling": (
            "sample with replacement from the matching line/segment/service-hour "
            "empirical distribution; fall back to all periods for sparse or missing hours"
        ),
        "line_index_order": list(reference_line_mapping or ()),
        "data_partition": data_partition,
        "lines": lines,
        "coverage_gate": coverage_gate,
        "diagnostics": {
            "zero_hold_shards": len(source_shards),
            "observed_bus_trajectories": observed_bus_trajectories,
            "adjacent_same_bus_event_pairs": adjacent_same_bus_pairs,
            "runtime_samples": total_samples,
            "rejected_nonpositive_samples": rejected_nonpositive,
            "skipped_multisegment_transitions": skipped_multisegment,
            "calibrated_segments": sum(
                len(value["segments"]) for value in lines.values()
            ),
        },
        "provenance": {
            "collection_matrix": str(matrix_manifest),
            "collection_matrix_sha256": sha256(matrix_manifest),
            "frozen_zero_matrix_audit": (
                {
                    "path": str(freeze_audit_path),
                    "sha256": sha256(freeze_audit_path),
                    "schema_version": freeze_audit["schema_version"],
                }
                if freeze_audit_path is not None and freeze_audit is not None
                else None
            ),
            "collection_source_sha256": manifest.get("source_sha256"),
            "sumo_version": manifest.get("sumo_version"),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "source_shards": source_shards,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace runtime calibration: {output}")
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--simulator-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--minimum-samples-per-segment", type=int, default=10)
    args = parser.parse_args()
    result = build_calibration(
        args.matrix_manifest,
        args.simulator_data,
        args.output,
        allow_partial=args.allow_partial,
        minimum_samples_per_segment=args.minimum_samples_per_segment,
    )
    print(json.dumps(result["diagnostics"], sort_keys=True))
    if not args.allow_partial and not result["coverage_gate"]["passed"]:
        raise RuntimeError(
            "formal v7 runtime calibration failed its coverage gate; "
            f"inspect {args.output.expanduser().resolve()}"
        )


if __name__ == "__main__":
    main()
