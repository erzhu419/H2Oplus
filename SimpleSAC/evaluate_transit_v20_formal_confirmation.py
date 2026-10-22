#!/usr/bin/env python3
"""Evaluate the frozen v20 formal confirmation matrix."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC import (  # noqa: E402
    evaluate_transit_v19_completion_safety_reserve as v19,
)


_V19_EVALUATION_SOURCE_PATHS = v19.evaluation_source_paths


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    """Add the formal-confirmation identity without changing policy behavior."""

    paths = set(_V19_EVALUATION_SOURCE_PATHS(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v20_formal_confirmation.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def main() -> None:
    v19.evaluation_source_paths = evaluation_source_paths
    v19.main()


if __name__ == "__main__":
    main()
