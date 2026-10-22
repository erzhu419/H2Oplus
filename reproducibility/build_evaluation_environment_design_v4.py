#!/usr/bin/env python3
"""Build independent paired demand/SUMO environment blocks for v4 evaluation."""

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


def parse_args() -> argparse.Namespace:
    sumo_root = ROOT.parent / "SUMO_ruiguang"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sumo-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.75, 1.0, 1.25))
    parser.add_argument("--depart-jitter-seconds", type=float, default=120.0)
    parser.add_argument("--forbidden-demand-seeds", type=int, nargs="*", default=())
    parser.add_argument("--forbidden-sumo-seeds", type=int, nargs="*", default=())
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o" / "evaluation_scenarios",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=sumo_root / "online_control" / "control_sim_traci_period.sumocfg",
    )
    parser.add_argument(
        "--direct-route",
        type=Path,
        default=(
            sumo_root
            / "e_passenger_rou"
            / "3_modified_period_workday_passenger_VC_0805.rou.xml"
        ),
    )
    parser.add_argument(
        "--transfer-route",
        type=Path,
        default=(
            sumo_root
            / "e_passenger_rou"
            / "3_modified_transfer_passenger_0805.rou.xml"
        ),
    )
    parser.add_argument(
        "--passenger-objects",
        type=Path,
        default=(
            sumo_root
            / "online_control"
            / "initialize_obj"
            / "save_obj_passenger.add.xml"
        ),
    )
    return parser.parse_args()


def _validate_design_inputs(
    demand_seeds: tuple[int, ...],
    sumo_seeds: tuple[int, ...],
    scales: tuple[float, ...],
    forbidden_demand_seeds: set[int],
    forbidden_sumo_seeds: set[int],
) -> None:
    if not demand_seeds or len(demand_seeds) != len(sumo_seeds):
        raise ValueError("demand and SUMO seed lists must have the same positive length")
    if len(set(demand_seeds)) != len(demand_seeds):
        raise ValueError("demand-realization seeds must be unique")
    if len(set(sumo_seeds)) != len(sumo_seeds):
        raise ValueError("SUMO seeds must be unique")
    if not scales or len(set(scales)) != len(scales) or any(scale <= 0 for scale in scales):
        raise ValueError("demand scales must be unique and positive")
    demand_overlap = sorted(set(demand_seeds) & forbidden_demand_seeds)
    if demand_overlap:
        raise ValueError(f"demand seeds overlap forbidden training seeds: {demand_overlap}")
    sumo_overlap = sorted(set(sumo_seeds) & forbidden_sumo_seeds)
    if sumo_overlap:
        raise ValueError(f"SUMO seeds overlap forbidden training seeds: {sumo_overlap}")


def _write_design(path: Path, payload: dict, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        previous = json.loads(path.read_text())
        previous_created = previous.pop("created_utc", None)
        if previous == payload:
            if isinstance(previous_created, str):
                created_utc = previous_created
        elif not force:
            raise FileExistsError(
                f"refusing to replace a different environment design: {path}"
            )
    payload = {**payload, "created_utc": created_utc}
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    demand_seeds = tuple(args.demand_seeds)
    sumo_seeds = tuple(args.sumo_seeds)
    scales = tuple(float(scale) for scale in args.scales)
    _validate_design_inputs(
        demand_seeds,
        sumo_seeds,
        scales,
        set(args.forbidden_demand_seeds),
        set(args.forbidden_sumo_seeds),
    )
    if args.depart_jitter_seconds <= 0.0:
        raise ValueError(
            "explicit unseen-demand blocks require positive departure-time jitter"
        )

    scenario_root = args.scenario_root.expanduser().resolve()
    blocks = []
    for index, (demand_seed, sumo_seed) in enumerate(
        zip(demand_seeds, sumo_seeds), start=1
    ):
        manifests = [
            build_scenario(
                base_config=args.base_config.expanduser().resolve(),
                passenger_routes=(
                    args.direct_route.expanduser().resolve(),
                    args.transfer_route.expanduser().resolve(),
                ),
                passenger_objects=args.passenger_objects.expanduser().resolve(),
                output_root=scenario_root,
                scale=scale,
                seed=demand_seed,
                depart_jitter_seconds=args.depart_jitter_seconds,
            )
            for scale in scales
        ]
        blocks.append(
            {
                "name": f"environment-{index:02d}",
                "demand_realization_seed": demand_seed,
                "sumo_seed": sumo_seed,
                "scenario_manifests": [str(path.resolve()) for path in manifests],
            }
        )

    payload = {
        "schema_version": ENVIRONMENT_DESIGN_SCHEMA,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "scenario_generator_sha256": sha256(
            ROOT / "reproducibility" / "build_passenger_scenarios.py"
        ),
        "design": (
            "one independent demand-realization seed and one independent SUMO "
            "seed per paired block; all demand scales retained within block"
        ),
        "depart_jitter_seconds": float(args.depart_jitter_seconds),
        "demand_scales": list(scales),
        "forbidden_demand_seeds": sorted(set(args.forbidden_demand_seeds)),
        "forbidden_sumo_seeds": sorted(set(args.forbidden_sumo_seeds)),
        "blocks": blocks,
    }
    output = args.output.expanduser().resolve()
    _write_design(output, payload, force=args.force)
    print(output)


if __name__ == "__main__":
    main()
