#!/usr/bin/env python3
"""Calibrate low-fidelity simulator OD rates from the canonical SUMO demand."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPORT_SCHEMA = "transit-simulator-demand-calibration-v3"
RELOCATABLE_REPORT_SCHEMA = "transit-simulator-demand-calibration-v2"
JOURNEY_SCHEMA = "transit-simulator-journeys-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_stop_component(physical_stop: str, line_id: str) -> str:
    pattern = re.compile(rf"^{re.escape(line_id)}\d+$")
    matches = [part for part in physical_stop.split("_") if pattern.match(part)]
    if len(matches) != 1:
        raise ValueError(
            f"physical stop {physical_stop!r} has {len(matches)} components for line {line_id!r}"
        )
    return matches[0]


def extract_journey_templates(
    route_paths: list[Path],
    line_trip_counts: dict[str, int],
) -> tuple[list[dict], dict]:
    """Parse person journeys without breaking the causal order of ride legs."""

    templates = []
    diagnostics = Counter()
    seen_ids = set()
    for route_path in route_paths:
        root = ET.parse(route_path).getroot()
        people = root.findall("person")
        if not people:
            raise ValueError(f"passenger route contains no people: {route_path}")
        departures = [float(person.get("depart", "0")) for person in people]
        upper_departure = max(departures)
        for person, departure in zip(people, departures):
            diagnostics["persons"] += 1
            person_id = person.get("id")
            if not person_id or person_id in seen_ids:
                raise ValueError(f"passenger IDs must be present and unique: {person_id!r}")
            seen_ids.add(person_id)
            current_stop = None
            legs = []
            for stage in person:
                if stage.tag == "walk" and stage.get("busStop"):
                    current_stop = stage.get("busStop")
                    continue
                if stage.tag != "ride":
                    continue
                if current_stop is None or not stage.get("busStop"):
                    raise ValueError(
                        f"person {person.get('id')} has a ride without stop anchors"
                    )
                destination_stop = stage.get("busStop")
                candidate_lines = [
                    line_id
                    for line_id in stage.get("lines", "").split()
                    if line_id in line_trip_counts
                ]
                if not candidate_lines:
                    raise ValueError(
                        f"person {person.get('id')} has no recognized ride line"
                    )
                total_frequency = sum(line_trip_counts[line_id] for line_id in candidate_lines)
                diagnostics["ride_legs"] += 1
                diagnostics["multi_line_ride_legs"] += len(candidate_lines) > 1
                candidates = []
                for line_id in candidate_lines:
                    weight = line_trip_counts[line_id] / total_frequency
                    origin = line_stop_component(current_stop, line_id)
                    destination = line_stop_component(destination_stop, line_id)
                    diagnostics["allocated_line_legs"] += weight
                    candidates.append(
                        {
                            "line_id": line_id,
                            "origin_stop": origin,
                            "destination_stop": destination,
                            "selection_weight": weight,
                        }
                    )
                legs.append(
                    {
                        "physical_origin_stop": current_stop,
                        "physical_destination_stop": destination_stop,
                        "candidates": candidates,
                    }
                )
                current_stop = destination_stop
            if not legs:
                raise ValueError(f"person {person_id} has no ride legs")
            diagnostics["transfer_journeys"] += len(legs) > 1
            templates.append(
                {
                    "template_id": person_id,
                    "departure_seconds": departure,
                    "departure_upper_bound_seconds": upper_departure,
                    "legs": legs,
                }
            )
    return templates, dict(diagnostics)


def _demand_from_templates(templates: list[dict]) -> dict:
    """Build the hourly scheduled-rate observation input from journey templates."""

    demand = defaultdict(float)
    for template in templates:
        departure = float(template["departure_seconds"])
        service_hour = min(19, max(6, 6 + int(departure // 3600)))
        for leg in template["legs"]:
            for candidate in leg["candidates"]:
                demand[
                    (
                        candidate["line_id"],
                        service_hour,
                        candidate["origin_stop"],
                        candidate["destination_stop"],
                    )
                ] += float(candidate["selection_weight"])
    return demand


def allocate_route_legs(
    route_paths: list[Path],
    line_trip_counts: dict[str, int],
) -> tuple[dict, dict]:
    """Compatibility API for the aggregate hourly OD calibration."""

    templates, diagnostics = extract_journey_templates(route_paths, line_trip_counts)
    return _demand_from_templates(templates), diagnostics


def _validated_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "transit-passenger-demand-v1":
        raise ValueError("scenario manifest does not implement transit-passenger-demand-v1")
    if float(payload.get("demand_scale", -1)) != 1.0:
        raise ValueError("simulator calibration requires the canonical 1.0 demand scenario")
    for raw_path, expected in payload.get("scenario_sha256", {}).items():
        artifact = Path(raw_path)
        if sha256(artifact) != expected:
            raise ValueError(f"scenario artifact hash mismatch: {artifact}")
    return payload


def build(
    manifest_path: Path,
    simulator_data: Path,
    bus_objects: Path,
    *,
    departure_jitter_seconds: float = 120.0,
) -> dict:
    if departure_jitter_seconds < 0:
        raise ValueError("departure jitter must be nonnegative")
    manifest = _validated_manifest(manifest_path)
    bus_root = ET.parse(bus_objects).getroot()
    departures = defaultdict(list)
    for bus in bus_root.findall(".//bus_obj"):
        line_id = bus.get("belong_line_id_s")
        if line_id:
            departures[line_id].append(float(bus.get("start_time_n", "0")))
    for values in departures.values():
        values.sort()

    line_directories = sorted(
        path for path in simulator_data.iterdir() if path.is_dir()
    )
    line_trip_counts = {}
    line_stops = {}
    for directory in line_directories:
        timetable = directory / "time_table.xlsx"
        stops = directory / "stop_news.xlsx"
        if not timetable.is_file() or not stops.is_file():
            continue
        line_id = directory.name
        if line_id not in departures:
            raise ValueError(f"SUMO bus objects contain no departures for {line_id}")
        line_trip_counts[line_id] = len(departures[line_id])
        line_stops[line_id] = [
            str(value) for value in pd.read_excel(stops)["stop_name"].tolist()
        ]
    if not line_trip_counts:
        raise ValueError(f"no simulator line data found under {simulator_data}")

    route_paths = [Path(value) for value in manifest["passenger_routes"]]
    journey_templates, diagnostics = extract_journey_templates(
        route_paths, line_trip_counts
    )
    demand = _demand_from_templates(journey_templates)
    report_root = simulator_data.parent.resolve()
    output_hashes = {}
    journey_output = simulator_data.parent / "simulator_journeys_v1.json"
    journey_payload = {
        "schema_version": JOURNEY_SCHEMA,
        "demand_scaling": (
            "floor(scale) copies of every source journey plus one Bernoulli copy "
            "with probability equal to the fractional part"
        ),
        "departure_jitter_seconds": float(departure_jitter_seconds),
        "transfer_semantics": (
            "only the first leg is scheduled initially; each later leg becomes "
            "eligible one simulator tick after the preceding leg actually alights"
        ),
        "source_passenger_routes_sha256": {
            path.name: sha256(path) for path in route_paths
        },
        "journeys": journey_templates,
    }
    journey_temporary = journey_output.with_suffix(journey_output.suffix + ".partial")
    journey_temporary.write_text(
        json.dumps(journey_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(journey_temporary, journey_output)
    output_hashes[
        journey_output.resolve().relative_to(report_root).as_posix()
    ] = sha256(journey_output)
    per_line = {}
    for line_id in sorted(line_trip_counts):
        stops = line_stops[line_id]
        stop_index = {stop: index for index, stop in enumerate(stops)}
        rows = []
        allocated = 0.0
        for hour in range(6, 20):
            for origin in stops:
                row = {"time_period": f"{hour:02d}:00:00", "stop_name": origin}
                for destination in stops:
                    value = demand.get((line_id, hour, origin, destination), 0.0)
                    if value and stop_index[destination] <= stop_index[origin]:
                        raise ValueError(
                            f"non-forward demand for {line_id}: {origin} -> {destination}"
                        )
                    row[destination] = round(value, 8)
                    allocated += value
                rows.append(row)
        output = simulator_data / line_id / "passenger_OD_sumo.xlsx"
        temporary = output.with_name(output.stem + ".partial" + output.suffix)
        pd.DataFrame(rows).to_excel(temporary, index=False)
        os.replace(temporary, output)
        output_hashes[output.resolve().relative_to(report_root).as_posix()] = sha256(
            output
        )
        timetable_output = simulator_data / line_id / "time_table_sumo.xlsx"
        timetable_temporary = timetable_output.with_name(
            timetable_output.stem + ".partial" + timetable_output.suffix
        )
        pd.DataFrame(
            {
                "launch_time": departures[line_id],
                "direction": [1] * len(departures[line_id]),
            }
        ).to_excel(timetable_temporary, index=False)
        os.replace(timetable_temporary, timetable_output)
        output_hashes[
            timetable_output.resolve().relative_to(report_root).as_posix()
        ] = sha256(timetable_output)
        per_line[line_id] = {
            "scheduled_trips": line_trip_counts[line_id],
            "first_departure_seconds": departures[line_id][0],
            "last_departure_seconds": departures[line_id][-1],
            "allocated_expected_ride_legs": allocated,
            "nonzero_od_hour_cells": int(
                sum(
                    value > 0
                    for (candidate, _hour, _origin, _destination), value in demand.items()
                    if candidate == line_id
                )
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA,
        "output_path_base": "report_directory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_bus_objects": str(bus_objects.resolve()),
        "source_bus_objects_sha256": sha256(bus_objects),
        "source_passengers": int(manifest["scaled_total_persons"]),
        "allocation": (
            "each SUMO ride leg is allocated across acceptable lines in proportion "
            "to their scheduled trip counts"
        ),
        "journey_execution": {
            "artifact": journey_output.name,
            "schema_version": JOURNEY_SCHEMA,
            "departure_jitter_seconds": float(departure_jitter_seconds),
            "causal_transfer_chain": True,
            "hourly_od_role": (
                "observation-rate calibration only; runtime passengers are generated "
                "from the causal journey artifact"
            ),
        },
        "diagnostics": diagnostics,
        "per_line": per_line,
        "output_sha256": output_hashes,
    }
    report_path = simulator_data.parent / "simulator_demand_calibration.json"
    temporary = report_path.with_suffix(report_path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    return report


def migrate_report_paths(report_path: Path) -> dict:
    """Convert a validated v1 absolute-path report into relocatable v2 evidence."""

    report_path = report_path.expanduser().resolve()
    report = json.loads(report_path.read_text())
    if report.get("schema_version") in {
        RELOCATABLE_REPORT_SCHEMA,
        REPORT_SCHEMA,
    }:
        return report
    if report.get("schema_version") != "transit-simulator-demand-calibration-v1":
        raise ValueError("only simulator demand calibration v1 can be migrated")
    report_root = report_path.parent
    migrated_outputs = {}
    for raw_path, expected_hash in report.get("output_sha256", {}).items():
        artifact = Path(raw_path).expanduser().resolve()
        if report_root not in artifact.parents:
            raise ValueError(
                f"calibrated output is outside the report directory: {artifact}"
            )
        if not artifact.is_file() or sha256(artifact) != expected_hash:
            raise ValueError(
                f"calibrated output cannot be migrated safely: {artifact}"
            )
        migrated_outputs[artifact.relative_to(report_root).as_posix()] = expected_hash
    if not migrated_outputs:
        raise ValueError("simulator calibration report declares no outputs")
    # Relocating paths does not create the causal journey artifact required by v3.
    report["schema_version"] = RELOCATABLE_REPORT_SCHEMA
    report["output_path_base"] = "report_directory"
    report["output_sha256"] = migrated_outputs
    temporary = report_path.with_suffix(report_path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--simulator-data", type=Path, required=True)
    parser.add_argument("--bus-objects", type=Path, required=True)
    parser.add_argument("--departure-jitter-seconds", type=float, default=120.0)
    args = parser.parse_args()
    report = build(
        args.scenario_manifest.expanduser().resolve(),
        args.simulator_data.expanduser().resolve(),
        args.bus_objects.expanduser().resolve(),
        departure_jitter_seconds=args.departure_jitter_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
