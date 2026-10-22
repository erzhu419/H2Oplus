#!/usr/bin/env python3
"""Merge the disjoint verified-host v7 H2O+ lambda sensitivity partitions."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility import merge_ablation_results_v7 as base


METHODS = (
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
)


def _configure() -> None:
    base.SCHEMA_VERSION = "transit-v7-sensitivity-partition-merge-v1"
    base.METHODS = METHODS
    base.EXPECTED_JOBS = {
        f"{method}_seed{seed}" for method in METHODS for seed in base.SEEDS
    }


def main() -> None:
    _configure()
    base.main()


if __name__ == "__main__":
    main()
