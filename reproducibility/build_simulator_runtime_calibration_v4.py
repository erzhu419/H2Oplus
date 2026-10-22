#!/usr/bin/env python3
"""Calibrate simulator segment runtimes from full-horizon zero-hold SUMO shards."""

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

from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_INDEX, SCHEMA_VERSION
from bus_h2o.sumo_provenance import sha256
from collect_policy.build_transit_v4_dataset import _load_shard


CALIBRATION_SCHEMA = "transit-simulator-runtime-calibration-v4"


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


def _zero_jobs(matrix_manifest: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(matrix_manifest.read_text())
    if manifest.get("schema_version") != "transit-smdp-v4-collection-matrix-v1":
        raise ValueError("collection matrix has an unsupported schema")
    jobs = [
        job
        for job in manifest.get("jobs", [])
        if job.get("policy", {}).get("policy_name") == "zero"
    ]
    if not jobs:
        raise ValueError("collection matrix contains no zero-hold jobs")
    outputs = [Path(job["output"]).expanduser().resolve() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("zero-hold collection jobs repeat an output path")
    signatures = [job["signature"] for job in jobs]
    if len(signatures) != len(set(signatures)):
        raise ValueError("zero-hold collection jobs repeat a signature")
    return manifest, jobs


def build_calibration(
    matrix_manifest: Path, simulator_data: Path, output: Path
) -> dict:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    simulator_data = simulator_data.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest, jobs = _zero_jobs(matrix_manifest)
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
    for job in sorted(jobs, key=lambda value: value["output"]):
        shard = Path(job["output"]).expanduser().resolve()
        summary, arrays = _load_shard(shard, allow_partial=False)
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

        observations = arrays["observations"]
        next_observations = arrays["next_observations"]
        line_index = np.rint(
            observations[:, RAW_EVENT_STATE_INDEX["line_id"]]
        ).astype(np.int64)
        station_index = np.rint(
            observations[:, RAW_EVENT_STATE_INDEX["station_index"]]
        ).astype(np.int64)
        next_line_index = np.rint(
            next_observations[:, RAW_EVENT_STATE_INDEX["line_id"]]
        ).astype(np.int64)
        next_station_index = np.rint(
            next_observations[:, RAW_EVENT_STATE_INDEX["station_index"]]
        ).astype(np.int64)
        if np.any(line_index != next_line_index):
            raise ValueError(f"zero-hold shard crosses lines: {shard}")
        if np.any(next_station_index <= station_index):
            raise ValueError(f"zero-hold shard has a non-forward transition: {shard}")
        if np.any((line_index < 0) | (line_index >= len(line_names))):
            raise ValueError(f"zero-hold shard has an unknown line index: {shard}")

        next_base_dwell = next_observations[
            :, RAW_EVENT_STATE_INDEX["base_dwell_seconds"]
        ]
        travel_component = arrays["delta_t_seconds"] - next_base_dwell
        hour = np.floor(
            observations[:, RAW_EVENT_STATE_INDEX["sim_time_seconds"]] / 3600.0
        ).astype(np.int64)
        for line, station, next_station, period, travel in zip(
            line_index,
            station_index,
            next_station_index,
            hour,
            travel_component,
        ):
            if next_station != station + 1:
                skipped_multisegment += 1
                continue
            if travel <= 0.0 or not np.isfinite(travel):
                rejected_nonpositive += 1
                continue
            line_name = line_names[int(line)]
            if station < 0 or station + 1 >= len(stops_by_line[line_name]):
                raise ValueError(
                    f"station {station} is invalid for line {line_name} in {shard}"
                )
            segment = f"{stops_by_line[line_name][station]}->{stops_by_line[line_name][station + 1]}"
            samples[(line_name, segment, int(station))][int(period)].append(
                float(travel)
            )
        source_shards.append(
            {
                "path": str(shard),
                "sha256": summary["shard_sha256"],
                "job_signature": summary["job_signature"],
                "sumo_seed": int(summary["sumo_seed"]),
                "transitions": int(summary["episode"]["transitions"]),
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

    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_schema": SCHEMA_VERSION,
        "semantics": (
            "empirical service-completion-to-next-service-completion time minus "
            "the next stop's base passenger-exchange dwell; source actions are exact zero hold"
        ),
        "simulator_fixed_transition_overhead_seconds": 3,
        "sampling": (
            "sample with replacement from the matching line/segment/service-hour "
            "empirical distribution; fall back to all periods for sparse or missing hours"
        ),
        "line_index_order": line_names,
        "lines": lines,
        "diagnostics": {
            "zero_hold_shards": len(source_shards),
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
            "collection_source_sha256": manifest.get("source_sha256"),
            "sumo_version": manifest.get("sumo_version"),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "source_shards": source_shards,
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
    args = parser.parse_args()
    result = build_calibration(
        args.matrix_manifest, args.simulator_data, args.output
    )
    print(json.dumps(result["diagnostics"], sort_keys=True))


if __name__ == "__main__":
    main()
