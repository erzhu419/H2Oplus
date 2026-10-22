#!/usr/bin/env python3
"""Run frozen v18 behaviors with the corrected v18-v2 parser contract."""

from __future__ import annotations

from pathlib import Path

from reproducibility import run_evaluation_matrix_v18_empty_tail_reserve as v1


ROOT = Path(__file__).resolve().parents[1]
METHODS_SCHEMA = v1.METHODS_SCHEMA
load_development_methods = v1.load_development_methods


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v1.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    # Start from v17 so the three conflicting v18-v1 flags are never emitted.
    command = v1.v17.job_command(job)
    method = job["method"]
    command.extend(
        (
            "--expected-parent-tail-service-floor-seconds",
            str(method["tail_service_floor_seconds"]),
            "--expected-empty-reserve-enabled",
            "1" if method["empty_reserve_enabled"] else "0",
            "--expected-empty-reserve-floor-seconds",
            str(method["empty_reserve_floor_seconds"]),
            "--expected-empty-activation-start-fraction",
            str(method["empty_activation_start_fraction"]),
            "--expected-empty-activation-end-fraction",
            str(method["empty_activation_end_fraction"]),
            "--expected-empty-max-onboard-capacity-fraction",
            str(method["empty_max_onboard_capacity_fraction"]),
            "--expected-v17-parent-method-name",
            method["v17_parent_method"],
        )
    )
    return command


base = v1.v17.v16.v15.v14.v13.v12.v11.base
base.EVALUATORS["v7"] = (
    ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py"
)
base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
base.load_methods = load_development_methods
base.evaluation_source_paths_v7 = evaluation_sources
base._job_command = job_command
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    base.main()
