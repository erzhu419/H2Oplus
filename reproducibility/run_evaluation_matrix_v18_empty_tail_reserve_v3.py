#!/usr/bin/env python3
"""Run frozen v18 behaviors with final-boundary argument rebinding."""

from __future__ import annotations

from pathlib import Path

from reproducibility import run_evaluation_matrix_v18_empty_tail_reserve_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
METHODS_SCHEMA = v2.METHODS_SCHEMA
load_development_methods = v2.load_development_methods
job_command = v2.job_command


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v2.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


base = v2.base
base.EVALUATORS["v7"] = (
    ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py"
)
base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
base.load_methods = load_development_methods
base.evaluation_source_paths_v7 = evaluation_sources
base._job_command = job_command
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    base.main()
