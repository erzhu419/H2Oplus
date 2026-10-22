#!/usr/bin/env python3
"""Bind the evidence that excludes the superseded v7 formal-v1 zero matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    collection_source_paths,
    hash_files,
    sha256,
)
from reproducibility.run_collection_matrix_v7 import MATRIX_SCHEMA


AUDIT_SCHEMA = "transit-smdp-v7-superseded-formal-v1-audit-v1"


def _descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def audit_superseded_matrix(
    matrix_path: Path,
    freeze_audit_path: Path,
    failed_calibration_path: Path,
    failed_replay_log_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    matrix_path = matrix_path.expanduser().resolve()
    freeze_audit_path = freeze_audit_path.expanduser().resolve()
    failed_calibration_path = failed_calibration_path.expanduser().resolve()
    failed_replay_log_path = failed_replay_log_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    matrix = json.loads(matrix_path.read_text())
    freeze_audit = json.loads(freeze_audit_path.read_text())
    calibration = json.loads(failed_calibration_path.read_text())
    replay_log = failed_replay_log_path.read_text(errors="replace")
    jobs = matrix.get("jobs", [])
    zero_jobs = [
        job
        for job in jobs
        if job.get("policy", {}).get("policy_name") == "zero"
    ]
    if matrix.get("schema_version") != MATRIX_SCHEMA or len(zero_jobs) != 30:
        raise ValueError("superseded audit requires the 30-cell formal-v1 zero matrix")

    summary_records = []
    service_audit_states: dict[str, int] = {}
    for job in zero_jobs:
        shard = Path(job["output"]).expanduser().resolve()
        summary_path = shard.with_suffix(shard.suffix + ".json")
        summary = json.loads(summary_path.read_text())
        service_audit = summary.get("bus_service_semantics_audit")
        state = (
            "missing"
            if service_audit is None
            else "passed"
            if service_audit.get("passed") is True
            else "failed"
        )
        service_audit_states[state] = service_audit_states.get(state, 0) + 1
        summary_records.append(
            {
                "block_id": int(job["block_id"]),
                "realization_block_id": int(job["realization_block_id"]),
                "demand_scale": float(job["scenario"]["demand_scale"]),
                "sumo_seed": int(job["sumo_seed"]),
                "job_signature": job["signature"],
                "shard": _descriptor(shard),
                "summary": _descriptor(summary_path),
                "service_semantics_audit": state,
            }
        )

    first_job = zero_jobs[0]
    current_sources = hash_files(
        collection_source_paths(
            ROOT,
            Path(first_job["sumo_dir"]),
            Path(first_job["edge_xml"]),
            contract_version="v7",
        )
    )
    registered_sources = matrix.get("source_sha256", {})
    source_changes = {
        path: {
            "registered_sha256": registered_sources.get(path),
            "current_sha256": current_sources.get(path),
        }
        for path in sorted(set(registered_sources) | set(current_sources))
        if registered_sources.get(path) != current_sources.get(path)
    }

    coverage = calibration.get("coverage_gate", {})
    coverage_checks = coverage.get("checks", {})
    replay_has_incomplete_service_error = (
        "is incomplete" in replay_log
        and "service_completions=" in replay_log
        and "ValueError:" in replay_log
    )
    checks = {
        "all_30_shards_have_passing_service_semantics_audits": service_audit_states
        == {"passed": 30},
        "runtime_calibration_coverage_passed": coverage.get("passed") is True,
        "all_447_simulator_segments_calibrated": (
            coverage.get("expected_segments") == 447
            and not coverage.get("missing_segments")
        ),
        "no_same_bus_station_gaps": coverage_checks.get(
            "no_same_bus_station_gaps"
        )
        is True,
        "full_service_replay_completed_without_error": not replay_has_incomplete_service_error,
        "collection_source_is_current": not source_changes,
    }
    if all(checks.values()):
        raise AssertionError("formal-v1 unexpectedly satisfies every exclusion check")
    result = {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": "superseded formal-v1 exact-zero v7 collection matrix",
        "verdict": {
            "eligible": False,
            "allowed_uses": ["forensic evidence of the superseded pipeline"],
            "prohibited_uses": [
                "offline or online training",
                "simulator runtime calibration",
                "model selection or evaluation",
                "paper tables, figures, or empirical claims",
            ],
            "reason": (
                "the matrix predates the bus-service semantics audit, its event-only "
                "runtime calibration misses terminal segments and contains station gaps, "
                "and a complete-service replay exposed missing service-completion records"
            ),
        },
        "eligibility_checks": checks,
        "service_semantics_evidence": {
            "summary_count": len(summary_records),
            "audit_states": service_audit_states,
            "summaries": summary_records,
        },
        "failed_runtime_calibration": {
            **_descriptor(failed_calibration_path),
            "schema_version": calibration.get("schema_version"),
            "coverage_gate": coverage,
            "diagnostics": calibration.get("diagnostics"),
        },
        "failed_full_service_replay": {
            **_descriptor(failed_replay_log_path),
            "contains_incomplete_service_error": replay_has_incomplete_service_error,
            "error_tail": replay_log.strip().splitlines()[-1],
        },
        "source_drift_after_semantics_repair": {
            "changed_file_count": len(source_changes),
            "changed_files": source_changes,
        },
        "provenance": {
            "matrix": _descriptor(matrix_path),
            "frozen_zero_matrix_audit": {
                **_descriptor(freeze_audit_path),
                "schema_version": freeze_audit.get("schema_version"),
                "passed": freeze_audit.get("passed"),
            },
            "auditor": _descriptor(Path(__file__).resolve()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "bus_h2o" / "datasets_v7" / "formal_v1" / "zero_matrix_manifest.json",
    )
    parser.add_argument(
        "--freeze-audit",
        type=Path,
        default=ROOT
        / "bus_h2o"
        / "datasets_v7"
        / "formal_v1"
        / "zero_matrix_manifest.audit.json",
    )
    parser.add_argument(
        "--failed-calibration",
        type=Path,
        default=ROOT
        / "bus_h2o"
        / "calibrated_env"
        / "segment_runtime_calibration_v7.json",
    )
    parser.add_argument(
        "--failed-replay-log",
        type=Path,
        default=ROOT
        / "bus_h2o"
        / "calibrated_env"
        / "runtime_traces_v7"
        / "formal_v1"
        / "realization_00"
        / "demand_0p75_jitter120s_seed1201"
        / "sumo_seed1101.full_service.npz.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "formal_v1_service_semantics_eligibility_audit_v7.json",
    )
    args = parser.parse_args()
    result = audit_superseded_matrix(
        args.matrix,
        args.freeze_audit,
        args.failed_calibration,
        args.failed_replay_log,
        args.output,
    )
    print(json.dumps(result["verdict"], sort_keys=True))


if __name__ == "__main__":
    main()
