#!/usr/bin/env python3
"""Freeze path-portable v7 SUMO scenarios without changing physical demand."""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMO_ROOT = ROOT.parent / "SUMO_ruiguang"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    load_validated_scenario,
    sha256,
    sumo_input_paths,
)
from reproducibility.build_training_environment_design_v7 import (
    TRAINING_DESIGN_SCHEMA,
)
from reproducibility.run_evaluation_matrix_v4 import ENVIRONMENT_DESIGN_SCHEMA


PORTABLE_CONFIG_NAME = "scenario_portable_v7.sumocfg"
PORTABLE_MANIFEST_NAME = "manifest_portable_v7.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "evaluation_environment_design_v7_formal_v1.json",
    )
    parser.add_argument(
        "--training-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "training_environment_design_v7_formal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "evaluation_environment_design_v7_formal_v2.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _logical_source_name(path: Path) -> str:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        return "H2Oplus/" + resolved.relative_to(ROOT).as_posix()
    if resolved == SUMO_ROOT or SUMO_ROOT in resolved.parents:
        return "SUMO_ruiguang/" + resolved.relative_to(SUMO_ROOT).as_posix()
    return "external/" + resolved.name


def _write_json(path: Path, payload: dict, *, force: bool) -> None:
    if path.exists():
        current = json.loads(path.read_text())
        if current == payload:
            return
        if not force:
            raise FileExistsError(f"refusing to replace a different file: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _portable_config(source: Path, destination: Path, *, force: bool) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    input_block = root.find("input")
    if input_block is None:
        raise ValueError(f"SUMO configuration has no input block: {source}")
    for element in input_block:
        values = []
        for raw in element.get("value", "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = source.parent / path
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            values.append(Path(os.path.relpath(resolved, destination.parent)).as_posix())
        element.set("value", ",".join(values))
    temporary = destination.with_suffix(destination.suffix + ".partial")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    if destination.exists() and destination.read_bytes() == temporary.read_bytes():
        temporary.unlink()
        return
    if destination.exists() and not force:
        temporary.unlink()
        raise FileExistsError(f"refusing to replace {destination}")
    os.replace(temporary, destination)


def _portable_scenario(manifest_path: Path, *, force: bool) -> tuple[Path, dict]:
    original_path = manifest_path.resolve()
    original = json.loads(original_path.read_text())
    load_validated_scenario(original_path)
    scenario_dir = original_path.parent
    if ROOT / "bus_h2o" / "evaluation_scenarios" not in scenario_dir.parents:
        raise ValueError(f"scenario is outside the formal evaluation tree: {scenario_dir}")

    source_config = _resolve(scenario_dir, original["sumo_config"])
    portable_config = scenario_dir / PORTABLE_CONFIG_NAME
    _portable_config(source_config, portable_config, force=force)
    passenger_objects = _resolve(scenario_dir, original["passenger_objects"])
    passenger_routes = [
        _resolve(scenario_dir, value) for value in original["passenger_routes"]
    ]
    declared = [portable_config, passenger_objects, *passenger_routes]
    relative_declared = {
        path.relative_to(scenario_dir).as_posix(): sha256(path) for path in declared
    }
    source_content = {
        _logical_source_name(Path(raw)): digest
        for raw, digest in original.get("source_sha256", {}).items()
    }
    demand_generator_sha256 = original.get("generator_sha256")
    payload = {
        **{
            key: value
            for key, value in original.items()
            if key
            not in {
                "generator_sha256",
                "passenger_objects",
                "passenger_routes",
                "scenario_sha256",
                "source_sha256",
                "sumo_config",
            }
        },
        "sumo_config": PORTABLE_CONFIG_NAME,
        "passenger_objects": passenger_objects.name,
        "passenger_routes": [path.name for path in passenger_routes],
        "scenario_sha256": relative_declared,
        "source_content_sha256": source_content,
        "demand_generator_sha256": demand_generator_sha256,
        "portable_manifest_generator": Path(
            os.path.relpath(Path(__file__).resolve(), scenario_dir)
        ).as_posix(),
        "portable_manifest_generator_sha256": sha256(Path(__file__).resolve()),
        "path_semantics": (
            "all runtime paths are relative to this manifest/config; content "
            "identity is independent of host mount point"
        ),
    }
    destination = scenario_dir / PORTABLE_MANIFEST_NAME
    _write_json(destination, payload, force=force)
    _, evidence = load_validated_scenario(destination)
    return destination, evidence


def _runtime_content_key(evidence: dict) -> str:
    return json.dumps(
        sorted(evidence["runtime_input_sha256"].values()),
        separators=(",", ":"),
    )


def _design_runtime_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    keys = set()
    for block in payload.get("blocks", []):
        for raw in block.get("scenario_manifests", []):
            manifest = _resolve(path.parent, raw)
            _, evidence = load_validated_scenario(manifest)
            key = _runtime_content_key(evidence)
            if key in keys:
                raise ValueError(f"environment design reuses runtime content: {manifest}")
            keys.add(key)
    return keys


def build(args: argparse.Namespace) -> dict:
    source_path = args.source_design.expanduser().resolve()
    training_path = args.training_design.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source = json.loads(source_path.read_text())
    training = json.loads(training_path.read_text())
    if not (
        source.get("schema_version") == ENVIRONMENT_DESIGN_SCHEMA
        and source.get("mdp_version") == "v7"
        and source.get("study_stage") == "formal"
    ):
        raise ValueError("source evaluation design is not formal v7")
    if training.get("schema_version") != TRAINING_DESIGN_SCHEMA:
        raise ValueError("training design is not formal v7")
    source_generator = _resolve(source_path.parent, source["generator"])
    if sha256(source_generator) != source.get("generator_sha256"):
        raise ValueError("source evaluation design generator is stale")

    blocks = []
    descriptors = []
    seen_manifests = set()
    evaluation_runtime_keys = set()
    for block in source["blocks"]:
        portable_paths = []
        for raw in block["scenario_manifests"]:
            original = _resolve(source_path.parent, raw)
            portable, evidence = _portable_scenario(original, force=args.force)
            manifest_hash = evidence["manifest_sha256"]
            runtime_key = _runtime_content_key(evidence)
            if manifest_hash in seen_manifests or runtime_key in evaluation_runtime_keys:
                raise ValueError("portable evaluation scenario is duplicated")
            seen_manifests.add(manifest_hash)
            evaluation_runtime_keys.add(runtime_key)
            relative_manifest = Path(os.path.relpath(portable, output.parent)).as_posix()
            portable_paths.append(relative_manifest)
            descriptors.append(
                {
                    "path": relative_manifest,
                    "sha256": manifest_hash,
                    "demand_scale": float(evidence["demand_scale"]),
                    "runtime_content_sha256": sorted(
                        evidence["runtime_input_sha256"].values()
                    ),
                }
            )
        blocks.append(
            {
                "name": block["name"],
                "demand_realization_seed": int(block["demand_realization_seed"]),
                "sumo_seed": int(block["sumo_seed"]),
                "scenario_manifests": portable_paths,
            }
        )

    training_runtime_keys = _design_runtime_keys(training_path)
    overlap = sorted(evaluation_runtime_keys & training_runtime_keys)
    if overlap:
        raise ValueError("portable evaluation runtime overlaps training scenarios")
    generator = Path(__file__).resolve()
    payload = {
        "schema_version": ENVIRONMENT_DESIGN_SCHEMA,
        "mdp_version": "v7",
        "study_stage": "formal",
        "role": "held-out policy evaluation in SUMO",
        "design": source["design"],
        "independent_realization_blocks": len(blocks),
        "environment_cells": len(descriptors),
        "demand_scales": source["demand_scales"],
        "depart_jitter_seconds": source["depart_jitter_seconds"],
        "blocks": blocks,
        "scenario_descriptors": descriptors,
        "training_disjointness": {
            "verified_disjoint": True,
            "training_design": {
                "path": Path(os.path.relpath(training_path, output.parent)).as_posix(),
                "sha256": sha256(training_path),
                "schema_version": training["schema_version"],
            },
            "evaluation_demand_realization_seeds": [
                int(block["demand_realization_seed"]) for block in blocks
            ],
            "training_demand_realization_seeds": source["training_disjointness"][
                "training_demand_realization_seeds"
            ],
            "evaluation_sumo_seeds": [int(block["sumo_seed"]) for block in blocks],
            "training_sumo_seeds": source["training_disjointness"][
                "training_sumo_seeds"
            ],
            "scenario_manifest_hash_overlap": [],
            "scenario_runtime_hash_overlap": [],
            "runtime_identity": "sorted SHA-256 multiset; host paths excluded",
        },
        "physical_scenario_reuse": {
            "path": Path(os.path.relpath(source_path, output.parent)).as_posix(),
            "sha256": sha256(source_path),
            "schema_version": source["schema_version"],
            "reason": (
                "only path serialization and SUMO config path syntax changed; "
                "network, traffic, timetable, passengers, seeds, and demand remain fixed"
            ),
        },
        "generator": Path(os.path.relpath(generator, output.parent)).as_posix(),
        "generator_sha256": sha256(generator),
        "source_generator_sha256": source["generator_sha256"],
        "path_semantics": "relative to the file that declares each path",
    }
    if output.exists():
        current = json.loads(output.read_text())
        current_without_time = {k: v for k, v in current.items() if k != "created_utc"}
        if current_without_time == payload:
            payload["created_utc"] = current["created_utc"]
        elif not args.force:
            raise FileExistsError(f"refusing to replace a different design: {output}")
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    return payload


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    payload = build(args)
    _write_json(output, payload, force=args.force)
    print(output)


if __name__ == "__main__":
    main()
