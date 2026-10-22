#!/usr/bin/env python3
"""Evaluate frozen v18 behaviors with final-boundary argument rebinding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC import evaluate_transit_v18_empty_tail_reserve as v1
from SimpleSAC import evaluate_transit_v18_empty_tail_reserve_v2 as v2


base = v1.v17.v16.v15.v14.v13.v12.v11.base


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v2.evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v3.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def _bind_empty_tail_expectations(args, expected):
    args.expected_parent_tail_service_floor_seconds = (
        expected.expected_parent_tail_service_floor_seconds
    )
    args.expected_empty_reserve_enabled = bool(
        expected.expected_empty_reserve_enabled
    )
    args.expected_empty_reserve_floor_seconds = (
        expected.expected_empty_reserve_floor_seconds
    )
    args.expected_activation_start_fraction = (
        expected.expected_empty_activation_start_fraction
    )
    args.expected_activation_end_fraction = (
        expected.expected_empty_activation_end_fraction
    )
    args.expected_max_onboard_capacity_fraction = (
        expected.expected_empty_max_onboard_capacity_fraction
    )
    args.expected_v17_parent_method_name = expected.expected_v17_parent_method_name
    return args


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
    expected, remaining = extra_parser.parse_known_args()

    def build_policy(args, config, station_max, dataset_hash):
        # Parent parsers run first; rebind only at the frozen v18 validator boundary.
        return v1.build_development_policy(
            _bind_empty_tail_expectations(args, expected),
            config,
            station_max,
            dataset_hash,
        )

    base.build_policy = build_policy
    base.evaluation_source_paths = evaluation_source_paths
    sys.argv = [sys.argv[0], *remaining]
    v1.v17.main()


if __name__ == "__main__":
    main()
