#!/usr/bin/env python3
"""Freeze the held-out physical SUMO environment blocks for v7 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import load_validated_scenario, sha256
from reproducibility.build_training_environment_design_v7 import (
    TRAINING_DESIGN_SCHEMA,
)
from reproducibility.run_evaluation_matrix_v4 import ENVIRONMENT_DESIGN_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-design",
        type=Path,
        default=(
            ROOT
            / "reproducibility"
            / "evaluation_environment_design_v4_formal_v1.json"
        ),
    )
    parser.add_argument(
        "--training-design",
        type=Path,
        default=(
            ROOT
            / "reproducibility"
            / "training_environment_design_v7_formal_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reproducibility"
            / "evaluation_environment_design_v7_formal_v1.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    return resolved, json.loads(resolved.read_text())


def _validate_blocks(payload: dict, *, role: str) -> dict:
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{role} environment design has no blocks")
    expected_scales = tuple(float(value) for value in payload["demand_scales"])
    expected_jitter = float(payload["depart_jitter_seconds"])
    block_names = set()
    demand_seeds = set()
    sumo_seeds = set()
    manifest_hashes = set()
    runtime_hashes = set()
    normalized_blocks = []
    for block in blocks:
        name = str(block["name"])
        demand_seed = int(block["demand_realization_seed"])
        sumo_seed = int(block["sumo_seed"])
        if name in block_names:
            raise ValueError(f"{role} environment block name is reused: {name}")
        if demand_seed in demand_seeds:
            raise ValueError(f"{role} demand-realization seed is reused: {demand_seed}")
        if sumo_seed in sumo_seeds:
            raise ValueError(f"{role} SUMO seed is reused: {sumo_seed}")
        block_names.add(name)
        demand_seeds.add(demand_seed)
        sumo_seeds.add(sumo_seed)
        manifests = [
            Path(value).expanduser().resolve()
            for value in block.get("scenario_manifests", [])
        ]
        if len(manifests) != len(expected_scales):
            raise ValueError(f"{role} block {name} has the wrong condition count")
        observed_scales = []
        descriptors = []
        for manifest in manifests:
            _, scenario = load_validated_scenario(manifest)
            if int(scenario["selection_seed"]) != demand_seed:
                raise ValueError(f"{role} block {name} has a demand-seed mismatch")
            if float(scenario.get("depart_jitter_seconds", 0.0)) != expected_jitter:
                raise ValueError(f"{role} block {name} has a jitter mismatch")
            manifest_hash = scenario["manifest_sha256"]
            runtime_hash = json.dumps(
                scenario["runtime_input_sha256"],
                sort_keys=True,
                separators=(",", ":"),
            )
            if manifest_hash in manifest_hashes:
                raise ValueError(f"{role} scenario manifest is reused")
            if runtime_hash in runtime_hashes:
                raise ValueError(f"{role} runtime scenario is reused")
            manifest_hashes.add(manifest_hash)
            runtime_hashes.add(runtime_hash)
            observed_scales.append(float(scenario["demand_scale"]))
            descriptors.append(
                {
                    "path": str(manifest),
                    "sha256": manifest_hash,
                    "runtime_input_sha256": scenario["runtime_input_sha256"],
                    "demand_scale": float(scenario["demand_scale"]),
                }
            )
        if tuple(observed_scales) != expected_scales:
            raise ValueError(f"{role} block {name} has the wrong demand-scale order")
        normalized_blocks.append(
            {
                "name": name,
                "demand_realization_seed": demand_seed,
                "sumo_seed": sumo_seed,
                "scenario_manifests": [str(path) for path in manifests],
                "scenario_descriptors": descriptors,
            }
        )
    return {
        "blocks": normalized_blocks,
        "demand_seeds": sorted(demand_seeds),
        "sumo_seeds": sorted(sumo_seeds),
        "manifest_hashes": sorted(manifest_hashes),
        "runtime_hashes": sorted(runtime_hashes),
        "environment_cells": sum(len(block["scenario_manifests"]) for block in blocks),
        "demand_scales": list(expected_scales),
        "depart_jitter_seconds": expected_jitter,
    }


def build_design(args: argparse.Namespace) -> dict:
    source_path, source = _load_json(args.source_design)
    training_path, training = _load_json(args.training_design)
    if source.get("schema_version") != ENVIRONMENT_DESIGN_SCHEMA:
        raise ValueError("v7 evaluation requires the locked physical evaluation design")
    if training.get("schema_version") != TRAINING_DESIGN_SCHEMA:
        raise ValueError("v7 evaluation requires the locked v7 training design")
    source_generator = Path(source.get("generator", "")).expanduser().resolve()
    if source.get("generator_sha256") != sha256(source_generator):
        raise ValueError("physical evaluation-design generator changed")
    scenario_generator = (
        ROOT / "reproducibility" / "build_passenger_scenarios.py"
    ).resolve()
    if source.get("scenario_generator_sha256") != sha256(scenario_generator):
        raise ValueError("passenger-scenario generator changed")
    training_generator = Path(training.get("generator", "")).expanduser().resolve()
    if training.get("generator_sha256") != sha256(training_generator):
        raise ValueError("v7 training-design generator changed")

    evaluation_blocks = _validate_blocks(source, role="evaluation")
    training_blocks = _validate_blocks(training, role="training")
    demand_overlap = sorted(
        set(evaluation_blocks["demand_seeds"]) & set(training_blocks["demand_seeds"])
    )
    sumo_overlap = sorted(
        set(evaluation_blocks["sumo_seeds"]) & set(training_blocks["sumo_seeds"])
    )
    manifest_overlap = sorted(
        set(evaluation_blocks["manifest_hashes"])
        & set(training_blocks["manifest_hashes"])
    )
    runtime_overlap = sorted(
        set(evaluation_blocks["runtime_hashes"])
        & set(training_blocks["runtime_hashes"])
    )
    if demand_overlap or sumo_overlap or manifest_overlap or runtime_overlap:
        raise ValueError(
            "v7 training/evaluation environments overlap: "
            f"demand={demand_overlap}, SUMO={sumo_overlap}, "
            f"manifest={manifest_overlap}, runtime={runtime_overlap}"
        )
    declared = training.get("evaluation_disjointness", {})
    if not (
        declared.get("verified_disjoint") is True
        and Path(declared.get("path", "")).expanduser().resolve() == source_path
        and declared.get("sha256") == sha256(source_path)
        and declared.get("demand_realization_seeds")
        == evaluation_blocks["demand_seeds"]
        and declared.get("sumo_seeds") == evaluation_blocks["sumo_seeds"]
    ):
        raise ValueError("v7 training design does not bind the held-out environment set")

    payload = {
        "schema_version": ENVIRONMENT_DESIGN_SCHEMA,
        "mdp_version": "v7",
        "study_stage": "formal",
        "role": "held-out policy evaluation in SUMO",
        "design": (
            "independent demand-realization and SUMO-seed blocks; all demand "
            "conditions remain paired within each block"
        ),
        "independent_realization_blocks": len(evaluation_blocks["blocks"]),
        "environment_cells": evaluation_blocks["environment_cells"],
        "demand_scales": evaluation_blocks["demand_scales"],
        "depart_jitter_seconds": evaluation_blocks["depart_jitter_seconds"],
        "blocks": [
            {
                key: value
                for key, value in block.items()
                if key != "scenario_descriptors"
            }
            for block in evaluation_blocks["blocks"]
        ],
        "scenario_descriptors": [
            descriptor
            for block in evaluation_blocks["blocks"]
            for descriptor in block["scenario_descriptors"]
        ],
        "training_disjointness": {
            "verified_disjoint": True,
            "training_design": {
                "path": str(training_path),
                "sha256": sha256(training_path),
                "schema_version": training["schema_version"],
            },
            "evaluation_demand_realization_seeds": evaluation_blocks[
                "demand_seeds"
            ],
            "training_demand_realization_seeds": training_blocks["demand_seeds"],
            "evaluation_sumo_seeds": evaluation_blocks["sumo_seeds"],
            "training_sumo_seeds": training_blocks["sumo_seeds"],
            "scenario_manifest_hash_overlap": manifest_overlap,
            "scenario_runtime_hash_overlap": runtime_overlap,
        },
        "physical_scenario_reuse": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "schema_version": source["schema_version"],
            "reason": (
                "v7 changes policy state, reward, and learning semantics but not "
                "the predeclared held-out exogenous SUMO scenarios"
            ),
        },
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "scenario_generator_sha256": sha256(scenario_generator),
    }
    return payload


def write_design(path: Path, payload: dict, *, force: bool) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    if destination.is_file():
        previous = json.loads(destination.read_text())
        previous_created = previous.pop("created_utc", None)
        if previous == payload and isinstance(previous_created, str):
            created_utc = previous_created
        elif previous != payload and not force:
            raise FileExistsError(f"refusing to replace a different design: {destination}")
    encoded = {**payload, "created_utc": created_utc}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(encoded, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    payload = build_design(args)
    write_design(args.output, payload, force=args.force)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
