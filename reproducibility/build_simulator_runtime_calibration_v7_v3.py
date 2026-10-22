#!/usr/bin/env python3
"""Add SUMO stop-lane observation speeds to the formal v7 runtime model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.build_simulator_runtime_calibration_v7_v2 import (
    CALIBRATION_SCHEMA_V2,
)


CALIBRATION_SCHEMA_V3 = "transit-simulator-runtime-calibration-v7-v3"


def _elements(root: ET.Element, local_name: str):
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == local_name:
            yield element


def _line_stop_sequences(line_stop_xml: Path) -> dict[str, list[dict[str, Any]]]:
    root = ET.parse(line_stop_xml).getroot()
    sequences: dict[str, list[dict[str, Any]]] = {}
    for busline in _elements(root, "busline"):
        line_id = str(busline.get("id", ""))
        if not line_id or line_id in sequences:
            raise ValueError("SUMO line-stop XML contains an invalid line ID")
        stops = []
        cumulative_distance = 0.0
        for element in list(busline):
            length = float(element.get("length", "0"))
            if not np.isfinite(length) or length < 0.0:
                raise ValueError(f"SUMO line {line_id} contains an invalid length")
            cumulative_distance += length
            if element.get("type") != "s":
                continue
            physical_stop_id = str(element.get("stop_id", ""))
            components = [
                value
                for value in physical_stop_id.split("_")
                if value.startswith(line_id)
                and value[len(line_id) :].isdigit()
            ]
            if len(components) != 1:
                raise ValueError(
                    f"cannot identify the {line_id} component of stop "
                    f"{physical_stop_id!r}"
                )
            stops.append(
                {
                    "station_name": components[0],
                    "physical_stop_id": physical_stop_id,
                    "route_coordinate_m": cumulative_distance,
                }
            )
        if len(stops) < 2:
            raise ValueError(f"SUMO line {line_id} contains fewer than two stops")
        sequences[line_id] = stops
    if not sequences:
        raise ValueError("SUMO line-stop XML contains no bus lines")
    return sequences


def _stop_lane_map(sumo_stops_xml: Path) -> dict[str, str]:
    result = {}
    for stop in _elements(ET.parse(sumo_stops_xml).getroot(), "busStop"):
        stop_id = str(stop.get("id", ""))
        lane_id = str(stop.get("lane", ""))
        if not stop_id or not lane_id or stop_id in result:
            raise ValueError("SUMO stop XML contains an invalid bus stop")
        result[stop_id] = lane_id
    if not result:
        raise ValueError("SUMO stop XML contains no bus stops")
    return result


def _lane_speed_map(sumo_network_xml: Path) -> dict[str, float]:
    result = {}
    for lane in _elements(ET.parse(sumo_network_xml).getroot(), "lane"):
        lane_id = str(lane.get("id", ""))
        raw_speed = lane.get("speed")
        if not lane_id or raw_speed is None or lane_id in result:
            raise ValueError("SUMO network XML contains an invalid lane")
        speed = float(raw_speed)
        if not np.isfinite(speed) or speed < 1.0:
            raise ValueError(f"SUMO lane {lane_id} has invalid speed {speed}")
        result[lane_id] = speed
    if not result:
        raise ValueError("SUMO network XML contains no lanes")
    return result


def build_calibration_v3(
    parent_calibration: Path,
    simulator_data: Path,
    line_stop_xml: Path,
    sumo_stops_xml: Path,
    sumo_network_xml: Path,
    output: Path,
) -> dict[str, Any]:
    parent_calibration = parent_calibration.expanduser().resolve()
    simulator_data = simulator_data.expanduser().resolve()
    line_stop_xml = line_stop_xml.expanduser().resolve()
    sumo_stops_xml = sumo_stops_xml.expanduser().resolve()
    sumo_network_xml = sumo_network_xml.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace runtime calibration: {output}")
    for required in (
        parent_calibration,
        line_stop_xml,
        sumo_stops_xml,
        sumo_network_xml,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"runtime calibration input is missing: {required}")
    if not simulator_data.is_dir():
        raise FileNotFoundError(f"simulator data directory is missing: {simulator_data}")

    parent = json.loads(parent_calibration.read_text())
    parent_gate = parent.get("coverage_gate", {})
    if (
        parent.get("schema_version") != CALIBRATION_SCHEMA_V2
        or parent_gate.get("passed") is not True
    ):
        raise ValueError("v7-v3 requires a passing v7-v2 parent calibration")

    line_sequences = _line_stop_sequences(line_stop_xml)
    stop_lanes = _stop_lane_map(sumo_stops_xml)
    lane_speeds = _lane_speed_map(sumo_network_xml)
    simulator_lines = sorted(
        path.name
        for path in simulator_data.iterdir()
        if path.is_dir() and (path / "stop_news.xlsx").is_file()
    )
    parent_lines = sorted(parent.get("lines", {}))
    if not simulator_lines or simulator_lines != parent_lines or simulator_lines != sorted(
        line_sequences
    ):
        raise ValueError(
            "v7-v3 line sets disagree among the simulator, parent calibration, "
            "and SUMO line-stop XML"
        )

    line_payload = {}
    observed_speeds = []
    physical_stops = set()
    lanes = set()
    for line_id in simulator_lines:
        expected_names = [
            str(value)
            for value in pd.read_excel(
                simulator_data / line_id / "stop_news.xlsx"
            )["stop_name"].tolist()
        ]
        sequence = line_sequences[line_id]
        actual_names = [row["station_name"] for row in sequence]
        if actual_names != expected_names:
            raise ValueError(
                f"simulator and SUMO stop orders disagree for line {line_id}"
            )
        rows = []
        base_coordinate = float(sequence[0]["route_coordinate_m"])
        for station_index, descriptor in enumerate(sequence):
            physical_stop_id = descriptor["physical_stop_id"]
            lane_id = stop_lanes.get(physical_stop_id)
            if lane_id is None:
                raise ValueError(f"SUMO stop {physical_stop_id} has no lane mapping")
            speed = lane_speeds.get(lane_id)
            if speed is None:
                raise ValueError(f"SUMO stop lane {lane_id} is absent from the network")
            observed_speeds.append(speed)
            physical_stops.add(physical_stop_id)
            lanes.add(lane_id)
            rows.append(
                {
                    "station_index": station_index,
                    "station_name": descriptor["station_name"],
                    "physical_stop_id": physical_stop_id,
                    "lane_id": lane_id,
                    "speed_mps": speed,
                    "route_coordinate_m": float(
                        descriptor["route_coordinate_m"] - base_coordinate
                    ),
                }
            )
        segments = []
        for station_index, (left, right) in enumerate(zip(rows[:-1], rows[1:])):
            distance = float(
                right["route_coordinate_m"] - left["route_coordinate_m"]
            )
            if not np.isfinite(distance) or distance <= 0.0:
                raise ValueError(
                    f"SUMO route coordinate is not increasing for {line_id}:"
                    f"{left['station_name']}->{right['station_name']}"
                )
            segments.append(
                {
                    "segment_index": station_index,
                    "start_stop": left["station_name"],
                    "end_stop": right["station_name"],
                    "distance_m": distance,
                }
            )
        line_payload[line_id] = {
            "stops": rows,
            "segments": segments,
            "route_distance_m": float(rows[-1]["route_coordinate_m"]),
        }

    speeds = np.asarray(observed_speeds, dtype=np.float64)
    observation_checks = {
        "line_sets_match": sorted(line_payload) == simulator_lines,
        "all_simulator_stops_match_sumo_line_order": sum(
            len(value["stops"]) for value in line_payload.values()
        )
        == sum(
            len(pd.read_excel(simulator_data / line / "stop_news.xlsx"))
            for line in simulator_lines
        ),
        "all_physical_stops_resolved": bool(physical_stops),
        "all_stop_lanes_resolved": bool(lanes),
        "all_observation_speeds_finite_and_at_least_one_mps": bool(
            len(speeds) and np.isfinite(speeds).all() and np.all(speeds >= 1.0)
        ),
        "all_spatial_segments_are_finite_and_positive": all(
            len(value["segments"]) == len(value["stops"]) - 1
            and all(
                np.isfinite(segment["distance_m"])
                and segment["distance_m"] > 0.0
                for segment in value["segments"]
            )
            for value in line_payload.values()
        ),
    }
    observation_gate = {
        "checks": observation_checks,
        "passed": all(observation_checks.values()),
        "line_count": len(line_payload),
        "line_stop_rows": int(len(speeds)),
        "unique_physical_stops": len(physical_stops),
        "unique_lanes": len(lanes),
        "spatial_segments": sum(
            len(value["segments"]) for value in line_payload.values()
        ),
        "speed_mps": {
            "minimum": float(speeds.min()),
            "maximum": float(speeds.max()),
            "mean": float(speeds.mean()),
            "unique": sorted(float(value) for value in np.unique(speeds)),
        },
    }
    if not observation_gate["passed"]:
        raise RuntimeError("v7-v3 stop-lane observation-speed coverage failed")

    result = copy.deepcopy(parent)
    result["schema_version"] = CALIBRATION_SCHEMA_V3
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["backward_headway_observation"] = {
        "semantics": (
            "convert the observable distance to the scheduled following bus into "
            "seconds using the current stop lane speed, matching the SUMO bridge's "
            "traci.lane.getLastStepMeanSpeed quantity without substituting the "
            "signal-delayed whole-segment travel speed"
        ),
        "low_fidelity_proxy": (
            "the immutable speed attribute of the current stop lane in the frozen "
            "SUMO network; no intervention or held-out fidelity outcomes are fitted"
        ),
        "route_coordinate_semantics": (
            "cumulative route length through each stop element in the frozen SUMO "
            "line decomposition; segment distances affect spatial observations but "
            "not the separately calibrated decision-clock runtime"
        ),
        "lines": line_payload,
        "coverage_gate": observation_gate,
    }
    coverage_gate = copy.deepcopy(parent_gate)
    coverage_gate["parent_runtime_calibration_passed"] = True
    coverage_gate["backward_headway_observation"] = observation_gate
    coverage_gate["checks"] = {
        **dict(parent_gate.get("checks", {})),
        "backward_headway_observation_coverage_passed": observation_gate["passed"],
    }
    coverage_gate["passed"] = all(coverage_gate["checks"].values())
    result["coverage_gate"] = coverage_gate
    result["provenance"] = {
        **dict(parent.get("provenance", {})),
        "parent_runtime_calibration": {
            "path": str(parent_calibration),
            "sha256": sha256(parent_calibration),
            "schema_version": CALIBRATION_SCHEMA_V2,
        },
        "backward_headway_network_inputs": {
            "line_stop_xml": {
                "path": str(line_stop_xml),
                "sha256": sha256(line_stop_xml),
            },
            "sumo_stops_xml": {
                "path": str(sumo_stops_xml),
                "sha256": sha256(sumo_stops_xml),
            },
            "sumo_network_xml": {
                "path": str(sumo_network_xml),
                "sha256": sha256(sumo_network_xml),
            },
        },
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--simulator-data", type=Path, required=True)
    parser.add_argument("--line-stop-xml", type=Path, required=True)
    parser.add_argument("--sumo-stops-xml", type=Path, required=True)
    parser.add_argument("--sumo-network-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_calibration_v3(
        args.parent_calibration,
        args.simulator_data,
        args.line_stop_xml,
        args.sumo_stops_xml,
        args.sumo_network_xml,
        args.output,
    )
    print(json.dumps(result["backward_headway_observation"]["coverage_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
