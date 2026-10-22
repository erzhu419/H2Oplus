#!/usr/bin/env python3
"""Evaluate frozen v18 behaviors with a collision-free parser contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC import evaluate_transit_v18_empty_tail_reserve as v1


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v1.evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT
            / "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v2.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


base = v1.v17.v16.v15.v14.v13.v12.v11.base
base.build_policy = v1.build_development_policy
base.evaluation_source_paths = evaluation_source_paths


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--expected-parent-tail-service-floor-seconds", type=float, required=True
    )
    extra_parser.add_argument(
        "--expected-empty-reserve-enabled", type=int, choices=(0, 1), required=True
    )
    extra_parser.add_argument(
        "--expected-empty-reserve-floor-seconds", type=float, required=True
    )
    extra_parser.add_argument(
        "--expected-empty-activation-start-fraction", type=float, required=True
    )
    extra_parser.add_argument(
        "--expected-empty-activation-end-fraction", type=float, required=True
    )
    extra_parser.add_argument(
        "--expected-empty-max-onboard-capacity-fraction",
        type=float,
        required=True,
    )
    extra_parser.add_argument("--expected-v17-parent-method-name", required=True)
    extra, remaining = extra_parser.parse_known_args()
    original_parse_args = base.parse_args

    def parse_args():
        args = original_parse_args()
        args.expected_parent_tail_service_floor_seconds = (
            extra.expected_parent_tail_service_floor_seconds
        )
        args.expected_empty_reserve_enabled = bool(extra.expected_empty_reserve_enabled)
        args.expected_empty_reserve_floor_seconds = (
            extra.expected_empty_reserve_floor_seconds
        )
        args.expected_activation_start_fraction = (
            extra.expected_empty_activation_start_fraction
        )
        args.expected_activation_end_fraction = (
            extra.expected_empty_activation_end_fraction
        )
        args.expected_max_onboard_capacity_fraction = (
            extra.expected_empty_max_onboard_capacity_fraction
        )
        args.expected_v17_parent_method_name = extra.expected_v17_parent_method_name
        return args

    base.parse_args = parse_args
    sys.argv = [sys.argv[0], *remaining]
    # Skip only the frozen v18-v1 outer parser; retain every parent parser.
    v1.v17.main()


if __name__ == "__main__":
    main()
