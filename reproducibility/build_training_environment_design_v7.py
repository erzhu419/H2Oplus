#!/usr/bin/env python3
"""Lock the physical v6 scenario set for formal transit-SMDP v7 training."""

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
from reproducibility.build_training_environment_design_v6 import (
    TRAINING_DESIGN_SCHEMA as SOURCE_DESIGN_SCHEMA,
)


TRAINING_DESIGN_SCHEMA = "transit-v7-training-environment-design-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "training_environment_design_v6_formal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "training_environment_design_v7_formal_v1.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validated_source_design(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != SOURCE_DESIGN_SCHEMA:
        raise ValueError("v7 requires the locked formal v6 physical design")

    source_generator = Path(payload.get("generator", "")).expanduser().resolve()
    expected_generator = (
        ROOT / "reproducibility" / "build_training_environment_design_v6.py"
    ).resolve()
    if source_generator != expected_generator:
        raise ValueError("the source environment-design generator path changed")
    if payload.get("generator_sha256") != sha256(source_generator):
        raise ValueError("the source environment-design generator changed")

    scenario_generator = (
        ROOT / "reproducibility" / "build_passenger_scenarios.py"
    ).resolve()
    if payload.get("scenario_generator_sha256") != sha256(scenario_generator):
        raise ValueError("the source passenger-scenario generator changed")

    for descriptor in payload.get("source_sha256", {}).values():
        source = Path(descriptor["path"]).expanduser().resolve()
        if descriptor.get("sha256") != sha256(source):
            raise ValueError(f"physical training source changed: {source}")

    evaluation = payload.get("evaluation_disjointness", {})
    evaluation_path = Path(evaluation.get("path", "")).expanduser().resolve()
    if (
        not evaluation.get("verified_disjoint")
        or evaluation.get("sha256") != sha256(evaluation_path)
    ):
        raise ValueError("training/evaluation disjointness evidence changed")

    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("the source design contains no realization blocks")
    if len({block["name"] for block in blocks}) != len(blocks):
        raise ValueError("training realization names are not unique")
    demand_seeds = [int(block["demand_realization_seed"]) for block in blocks]
    sumo_seeds = [int(block["sumo_seed"]) for block in blocks]
    if len(set(demand_seeds)) != len(demand_seeds):
        raise ValueError("training demand-realization seeds are reused")
    if len(set(sumo_seeds)) != len(sumo_seeds):
        raise ValueError("training SUMO seeds are reused")

    expected_scales = tuple(float(value) for value in payload["demand_scales"])
    expected_jitter = float(payload["depart_jitter_seconds"])
    seen_manifests = set()
    environment_cells = 0
    for block in blocks:
        manifests = block.get("scenario_manifests", [])
        if len(manifests) != len(expected_scales):
            raise ValueError(
                f"realization {block['name']} has the wrong condition count"
            )
        observed_scales = []
        for manifest_value in manifests:
            _, scenario = load_validated_scenario(
                Path(manifest_value).expanduser().resolve()
            )
            manifest_digest = scenario["manifest_sha256"]
            if manifest_digest in seen_manifests:
                raise ValueError("the physical design reuses a scenario manifest")
            seen_manifests.add(manifest_digest)
            if int(scenario["selection_seed"]) != int(
                block["demand_realization_seed"]
            ):
                raise ValueError(
                    f"realization {block['name']} has a demand-seed mismatch"
                )
            if float(scenario.get("depart_jitter_seconds", 0.0)) != expected_jitter:
                raise ValueError(
                    f"realization {block['name']} has a jitter mismatch"
                )
            if Path(scenario["manifest"]) != Path(manifest_value).expanduser().resolve():
                raise ValueError("scenario manifest resolution is inconsistent")
            observed_scales.append(float(scenario["demand_scale"]))
            environment_cells += 1
        if tuple(observed_scales) != expected_scales:
            raise ValueError(
                f"realization {block['name']} has the wrong demand-scale order"
            )

    if environment_cells != int(payload.get("environment_cells", -1)):
        raise ValueError("the source environment-cell count is inconsistent")
    if len(blocks) != int(payload.get("independent_realization_blocks", -1)):
        raise ValueError("the source realization-block count is inconsistent")
    return payload


def _write_design(path: Path, payload: dict, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        previous = json.loads(path.read_text())
        previous_created = previous.pop("created_utc", None)
        if previous == payload and isinstance(previous_created, str):
            created_utc = previous_created
        elif previous != payload and not force:
            raise FileExistsError(f"refusing to replace a different design: {path}")
    encoded = {**payload, "created_utc": created_utc}
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(encoded, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_design(args: argparse.Namespace) -> dict:
    source_path = args.source_design.expanduser().resolve()
    source = _validated_source_design(source_path)
    payload = {
        "schema_version": TRAINING_DESIGN_SCHEMA,
        "role": "offline real-domain training data",
        "design": source["design"],
        "independent_realization_blocks": source[
            "independent_realization_blocks"
        ],
        "environment_cells": source["environment_cells"],
        "demand_scales": source["demand_scales"],
        "depart_jitter_seconds": source["depart_jitter_seconds"],
        "blocks": source["blocks"],
        "evaluation_disjointness": source["evaluation_disjointness"],
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "scenario_generator_sha256": source["scenario_generator_sha256"],
        "source_sha256": source["source_sha256"],
        "physical_scenario_reuse": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "schema_version": source["schema_version"],
            "reason": (
                "v7 changes the decision-state and reward contract, not the "
                "exogenous passenger-demand realizations"
            ),
        },
    }
    _write_design(args.output.expanduser().resolve(), payload, force=args.force)
    return payload


def main() -> None:
    args = parse_args()
    build_design(args)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
