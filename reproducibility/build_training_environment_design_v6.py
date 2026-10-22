#!/usr/bin/env python3
"""Build disjoint multi-demand environment blocks for formal v6 training."""

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

from reproducibility.build_passenger_scenarios import build_scenario, sha256
from reproducibility.run_evaluation_matrix_v4 import ENVIRONMENT_DESIGN_SCHEMA


TRAINING_DESIGN_SCHEMA = "transit-v6-training-environment-design-v1"


def parse_args() -> argparse.Namespace:
    sumo_root = ROOT.parent / "SUMO_ruiguang"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sumo-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.75, 1.0, 1.25))
    parser.add_argument("--depart-jitter-seconds", type=float, default=120.0)
    parser.add_argument(
        "--evaluation-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "evaluation_environment_design_v4_formal_v1.json",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o" / "training_scenarios_v6" / "formal_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "training_environment_design_v6_formal_v1.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=sumo_root / "online_control" / "control_sim_traci_period.sumocfg",
    )
    parser.add_argument(
        "--direct-route",
        type=Path,
        default=sumo_root
        / "e_passenger_rou"
        / "3_modified_period_workday_passenger_VC_0805.rou.xml",
    )
    parser.add_argument(
        "--transfer-route",
        type=Path,
        default=sumo_root
        / "e_passenger_rou"
        / "3_modified_transfer_passenger_0805.rou.xml",
    )
    parser.add_argument(
        "--passenger-objects",
        type=Path,
        default=sumo_root
        / "online_control"
        / "initialize_obj"
        / "save_obj_passenger.add.xml",
    )
    return parser.parse_args()


def _validate_unique_inputs(
    demand_seeds: tuple[int, ...],
    sumo_seeds: tuple[int, ...],
    scales: tuple[float, ...],
) -> None:
    if not demand_seeds or len(demand_seeds) != len(sumo_seeds):
        raise ValueError("demand and SUMO seed lists must have equal positive length")
    if len(set(demand_seeds)) != len(demand_seeds):
        raise ValueError("training demand-realization seeds must be unique")
    if len(set(sumo_seeds)) != len(sumo_seeds):
        raise ValueError("training SUMO seeds must be unique")
    if not scales or len(set(scales)) != len(scales) or any(scale <= 0 for scale in scales):
        raise ValueError("training demand scales must be unique and positive")


def validate_evaluation_disjointness(
    evaluation_design: Path,
    demand_seeds: tuple[int, ...],
    sumo_seeds: tuple[int, ...],
) -> dict:
    path = evaluation_design.expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != ENVIRONMENT_DESIGN_SCHEMA:
        raise ValueError("formal evaluation design has an unsupported schema")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("formal evaluation design contains no blocks")
    evaluation_demand = {int(block["demand_realization_seed"]) for block in blocks}
    evaluation_sumo = {int(block["sumo_seed"]) for block in blocks}
    demand_overlap = sorted(set(demand_seeds) & evaluation_demand)
    sumo_overlap = sorted(set(sumo_seeds) & evaluation_sumo)
    if demand_overlap or sumo_overlap:
        raise ValueError(
            "training/evaluation seed overlap: "
            f"demand={demand_overlap}, SUMO={sumo_overlap}"
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "schema_version": payload["schema_version"],
        "demand_realization_seeds": sorted(evaluation_demand),
        "sumo_seeds": sorted(evaluation_sumo),
        "verified_disjoint": True,
    }


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
    demand_seeds = tuple(int(seed) for seed in args.demand_seeds)
    sumo_seeds = tuple(int(seed) for seed in args.sumo_seeds)
    scales = tuple(float(scale) for scale in args.scales)
    _validate_unique_inputs(demand_seeds, sumo_seeds, scales)
    if args.depart_jitter_seconds <= 0.0:
        raise ValueError("formal training requires positive departure-time jitter")
    evaluation = validate_evaluation_disjointness(
        args.evaluation_design, demand_seeds, sumo_seeds
    )

    source_paths = {
        "base_config": args.base_config.expanduser().resolve(),
        "direct_route": args.direct_route.expanduser().resolve(),
        "transfer_route": args.transfer_route.expanduser().resolve(),
        "passenger_objects": args.passenger_objects.expanduser().resolve(),
    }
    scenario_root = args.scenario_root.expanduser().resolve()
    blocks = []
    for index, (demand_seed, sumo_seed) in enumerate(
        zip(demand_seeds, sumo_seeds), start=1
    ):
        manifests = [
            build_scenario(
                base_config=source_paths["base_config"],
                passenger_routes=(
                    source_paths["direct_route"],
                    source_paths["transfer_route"],
                ),
                passenger_objects=source_paths["passenger_objects"],
                output_root=scenario_root,
                scale=scale,
                seed=demand_seed,
                depart_jitter_seconds=args.depart_jitter_seconds,
            )
            for scale in scales
        ]
        blocks.append(
            {
                "name": f"training-realization-{index:02d}",
                "demand_realization_seed": demand_seed,
                "sumo_seed": sumo_seed,
                "scenario_manifests": [str(path.resolve()) for path in manifests],
            }
        )

    payload = {
        "schema_version": TRAINING_DESIGN_SCHEMA,
        "role": "offline real-domain training data",
        "design": (
            "independent demand/SUMO realization blocks; correlated demand "
            "scales remain paired within each block"
        ),
        "independent_realization_blocks": len(blocks),
        "environment_cells": len(blocks) * len(scales),
        "demand_scales": list(scales),
        "depart_jitter_seconds": float(args.depart_jitter_seconds),
        "blocks": blocks,
        "evaluation_disjointness": evaluation,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "scenario_generator_sha256": sha256(
            ROOT / "reproducibility" / "build_passenger_scenarios.py"
        ),
        "source_sha256": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in source_paths.items()
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
