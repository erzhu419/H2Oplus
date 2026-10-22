#!/usr/bin/env python3
"""Build deterministic passenger-only demand scenarios for SUMO evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_uniform(seed: int, identifier: str) -> float:
    payload = f"{seed}:{identifier}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(2**64)


def _scale_tag(scale: float) -> str:
    return f"{scale:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def scaled_person_elements(
    route_path: Path,
    scale: float,
    seed: int,
    depart_jitter_seconds: float = 0.0,
) -> tuple[list[ET.Element], list[tuple[str, str, float]]]:
    """Return scaled person elements and `(new_id, source_id, depart)` records."""

    if scale <= 0:
        raise ValueError("demand scale must be positive")
    jitter = float(depart_jitter_seconds)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("departure jitter must be finite and non-negative")
    root = ET.parse(route_path).getroot()
    people = [element for element in root if element.tag == "person"]
    source_departures = [float(person.get("depart", "0")) for person in people]
    max_source_depart = max(source_departures, default=0.0)
    whole_copies = int(scale)
    fractional_copy = scale - whole_copies
    tag = _scale_tag(scale)
    generated: list[ET.Element] = []
    records: list[tuple[str, str, float]] = []
    for person in people:
        source_id = person.get("id")
        if not source_id:
            raise ValueError(f"person without id in {route_path}")
        copies = whole_copies
        if _stable_uniform(seed, source_id) < fractional_copy:
            copies += 1
        for copy_index in range(copies):
            cloned = copy.deepcopy(person)
            new_id = (
                source_id
                if copy_index == 0
                else f"{source_id}__demand_{tag}_{copy_index}"
            )
            cloned.set("id", new_id)
            source_depart = float(cloned.get("depart", "0"))
            if jitter:
                draw = _stable_uniform(seed, f"{source_id}:{copy_index}:depart")
                depart = min(
                    max_source_depart,
                    max(0.0, source_depart + jitter * (2.0 * draw - 1.0)),
                )
                cloned.set("depart", f"{depart:.6f}".rstrip("0").rstrip("."))
            else:
                depart = source_depart
            generated.append(cloned)
            records.append((new_id, source_id, depart))
    order = sorted(range(len(generated)), key=lambda index: (records[index][2], records[index][0]))
    return [generated[index] for index in order], [records[index] for index in order]


def _write_route(
    source_path: Path,
    target_path: Path,
    people: list[ET.Element],
) -> None:
    source_root = ET.parse(source_path).getroot()
    target_root = ET.Element(source_root.tag, source_root.attrib)
    for element in source_root:
        if element.tag != "person":
            target_root.append(copy.deepcopy(element))
    target_root.extend(people)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(target_path.suffix + ".partial")
    ET.ElementTree(target_root).write(
        temporary, encoding="utf-8", xml_declaration=True
    )
    os.replace(temporary, target_path)


def _write_passenger_objects(
    source_path: Path,
    target_path: Path,
    records: list[tuple[str, str, float]],
) -> None:
    source_root = ET.parse(source_path).getroot()
    source_objects = {
        element.get("passenger_id_s"): element
        for element in source_root
        if element.tag == "passenger_obj"
    }
    target_root = ET.Element(source_root.tag, source_root.attrib)
    for new_id, source_id, depart in sorted(records, key=lambda row: (row[2], row[0])):
        if source_id not in source_objects:
            raise KeyError(f"passenger {source_id!r} is missing from {source_path}")
        cloned = copy.deepcopy(source_objects[source_id])
        cloned.set("passenger_id_s", new_id)
        cloned.set("start_time_n", f"{depart:.6f}".rstrip("0").rstrip("."))
        target_root.append(cloned)
    temporary = target_path.with_suffix(target_path.suffix + ".partial")
    ET.ElementTree(target_root).write(
        temporary, encoding="utf-8", xml_declaration=True
    )
    os.replace(temporary, target_path)


def _resolved_values(config_path: Path, value: str) -> list[Path]:
    return [
        (config_path.parent / item.strip()).resolve()
        for item in value.split(",")
        if item.strip()
    ]


def _write_config(
    source_path: Path,
    target_path: Path,
    passenger_sources: tuple[Path, Path],
    passenger_targets: tuple[Path, Path],
) -> None:
    tree = ET.parse(source_path)
    root = tree.getroot()
    for section_name in ("output", "report", "fcd_device"):
        section = root.find(section_name)
        if section is not None:
            root.remove(section)
    input_element = root.find("input")
    if input_element is None:
        raise ValueError("SUMO configuration has no input section")
    for option_name in ("net-file", "additional-files"):
        option = input_element.find(option_name)
        if option is None:
            raise ValueError(f"SUMO configuration has no {option_name}")
        option.set(
            "value",
            ",".join(str(path) for path in _resolved_values(source_path, option.get("value", ""))),
        )
    routes = input_element.find("route-files")
    if routes is None:
        raise ValueError("SUMO configuration has no route-files")
    resolved_routes = _resolved_values(source_path, routes.get("value", ""))
    replacements = {
        source.resolve(): target.resolve()
        for source, target in zip(passenger_sources, passenger_targets)
    }
    replaced = [replacements.get(route.resolve(), route.resolve()) for route in resolved_routes]
    if sum(route in replacements for route in map(Path.resolve, resolved_routes)) != 2:
        raise ValueError("both passenger route files must occur in the base SUMO config")
    routes.set("value", ",".join(str(path) for path in replaced))
    temporary = target_path.with_suffix(target_path.suffix + ".partial")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, target_path)


def build_scenario(
    *,
    base_config: Path,
    passenger_routes: tuple[Path, Path],
    passenger_objects: Path,
    output_root: Path,
    scale: float,
    seed: int,
    depart_jitter_seconds: float = 0.0,
) -> Path:
    jitter = float(depart_jitter_seconds)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("departure jitter must be finite and non-negative")
    jitter_tag = (
        f"_jitter{_scale_tag(jitter)}s" if jitter > 0.0 else ""
    )
    scenario_dir = output_root / (
        f"demand_{_scale_tag(scale)}{jitter_tag}_seed{seed}"
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    route_targets = (
        scenario_dir / "passengers_direct.rou.xml",
        scenario_dir / "passengers_transfer.rou.xml",
    )
    all_records: list[tuple[str, str, float]] = []
    source_counts = []
    scaled_counts = []
    for source, target in zip(passenger_routes, route_targets):
        people, records = scaled_person_elements(
            source,
            scale,
            seed,
            depart_jitter_seconds=jitter,
        )
        _write_route(source, target, people)
        all_records.extend(records)
        source_counts.append(
            sum(element.tag == "person" for element in ET.parse(source).getroot())
        )
        scaled_counts.append(len(people))
    if len({record[0] for record in all_records}) != len(all_records):
        raise ValueError("scaled passenger identifiers are not unique")

    object_target = scenario_dir / "passenger_objects.add.xml"
    config_target = scenario_dir / "scenario.sumocfg"
    _write_passenger_objects(passenger_objects, object_target, all_records)
    _write_config(
        base_config,
        config_target,
        passenger_routes,
        route_targets,
    )
    files = (config_target, object_target, *route_targets)
    generation_config = {
        "demand_scale": scale,
        "selection_seed": seed,
        "depart_jitter_seconds": jitter,
        "departure_time_support": (
            "clipped to [0, maximum source departure] independently per route"
        ),
    }
    manifest = {
        "schema_version": "transit-passenger-demand-v1",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "generation_config": generation_config,
        "demand_scale": scale,
        "selection_seed": seed,
        "depart_jitter_seconds": jitter,
        "realization_design": (
            "deterministic source departures"
            if jitter == 0.0
            else (
                "independent stable-uniform departure-time perturbations "
                "clipped to each source route's original time support"
            )
        ),
        "scope": "passengers only; social traffic and bus timetable are unchanged",
        "source_person_counts": source_counts,
        "scaled_person_counts": scaled_counts,
        "source_total_persons": sum(source_counts),
        "scaled_total_persons": sum(scaled_counts),
        "sumo_config": str(config_target.resolve()),
        "passenger_objects": str(object_target.resolve()),
        "passenger_routes": [str(path.resolve()) for path in route_targets],
        "source_sha256": {
            str(path.resolve()): sha256(path)
            for path in (base_config, passenger_objects, *passenger_routes)
        },
        "scenario_sha256": {
            str(path.resolve()): sha256(path) for path in files
        },
    }
    manifest_path = scenario_dir / "manifest.json"
    created_utc = datetime.now(timezone.utc).isoformat()
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            previous_created = previous.pop("created_utc", None)
            if previous == manifest and isinstance(previous_created, str):
                created_utc = previous_created
        except Exception:
            pass
    manifest["created_utc"] = created_utc
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sumo_root = project_root.parent / "SUMO_ruiguang"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.75, 1.0, 1.25))
    parser.add_argument("--seed", type=int, default=20_260_803)
    parser.add_argument(
        "--depart-jitter-seconds",
        type=float,
        default=0.0,
        help="symmetric per-passenger departure-time perturbation; zero preserves source times",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "bus_h2o" / "evaluation_scenarios",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=sumo_root / "online_control" / "control_sim_traci_period.sumocfg",
    )
    parser.add_argument(
        "--direct-route",
        type=Path,
        default=sumo_root / "e_passenger_rou" / "3_modified_period_workday_passenger_VC_0805.rou.xml",
    )
    parser.add_argument(
        "--transfer-route",
        type=Path,
        default=sumo_root / "e_passenger_rou" / "3_modified_transfer_passenger_0805.rou.xml",
    )
    parser.add_argument(
        "--passenger-objects",
        type=Path,
        default=sumo_root / "online_control" / "initialize_obj" / "save_obj_passenger.add.xml",
    )
    args = parser.parse_args()
    manifests = [
        build_scenario(
            base_config=args.base_config.resolve(),
            passenger_routes=(args.direct_route.resolve(), args.transfer_route.resolve()),
            passenger_objects=args.passenger_objects.resolve(),
            output_root=args.output_root.resolve(),
            scale=scale,
            seed=args.seed,
            depart_jitter_seconds=args.depart_jitter_seconds,
        )
        for scale in args.scales
    ]
    print("\n".join(str(path.resolve()) for path in manifests))


if __name__ == "__main__":
    main()
