#!/usr/bin/env python3
"""Create the final v12 staged-release development-evaluation closeout record."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v12_staged_release_results"
ANALYSIS = RESULTS / "staged_release_analysis_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
ANALYSIS_INCIDENT = (
    RESULTS / "evaluation_incidents/analysis_numpy_scalar_serialization_v1.json"
)
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v12_staged_release.py"
ANALYSIS_WRAPPER = (
    REPO / "reproducibility/run_frozen_h2oplus_v12_staged_release_analysis_v1.py"
)
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "a77ffa04f3e2f2bfc5c6e3495505cb70cace2326df54e6906f6fcca34a9aa4e4"
)
EXPECTED_FAILURE_RUNS: list[str] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(REPO)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build() -> dict:
    report = json.loads(ANALYSIS.read_text())
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    verification = json.loads(MANIFEST_VERIFICATION.read_text())
    scheduler = json.loads(SCHEDULER.read_text())
    incident = json.loads(ANALYSIS_INCIDENT.read_text())
    checks = report.get("development_scientific_result", {}).get("checks", {})
    stability = report.get("training_scientific_stability", {})
    screening_gate_passed = report.get("development_scientific_result", {}).get(
        "screening_gate_passed"
    )
    calculated_gate = bool(checks) and all(value is True for value in checks.values())
    if (
        sha256(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or report.get("schema_version") != "transit-h2oplus-v12-staged-release-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("physical_evaluation_results") != 630
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows") != 1620
        or report.get("artifact_integrity", {}).get("logical_policy_budget_aliases") != 180
        or screening_gate_passed is not calculated_gate
        or screening_gate_passed is not False
        or report.get("formal_confirmation_eligible") is not screening_gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or stability.get("passed") is not True
        or stability.get("failure_runs") != EXPECTED_FAILURE_RUNS
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 630
        or manifest.get("artifact_integrity", {}).get("result_count") != 630
        or verification.get("passed") is not True
        or verification.get("independent_artifact_audit", {}).get("result_count") != 630
        or verification.get("independent_artifact_audit", {}).get(
            "physical_policy_checkpoints_rehashed"
        )
        != 70
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t83115", "t83116"}
        or incident.get("schema_version")
        != "transit-h2oplus-v12-staged-release-analysis-incident-v1"
        or incident.get("analysis_output_created") is not False
        or incident.get("partial_output_created") is not False
        or incident.get("development_evaluation_outcomes_read") is not True
        or incident.get("formal_held_out_outcomes_read") is not False
        or incident.get("frozen_analyzer_modified") is not False
        or incident.get("sumo_or_training_run_started") is not False
        or incident.get("analyzer", {}).get("sha256")
        != EXPECTED_ANALYSIS_SOURCE_SHA256
    ):
        raise ValueError("v12 staged-release final closeout contract failed")
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    if not failed_checks:
        raise ValueError("v12 failed development screening gate lacks failed checks")
    return {
        "schema_version": "transit-h2oplus-v12-staged-release-evaluation-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed",
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 630,
            "logical_evaluation_rows": 1620,
            "logical_policy_budget_aliases": 180,
            "physical_policy_checkpoints_independently_rehashed": 70,
            "canonical_matrix_count": 3,
            "source_partition_count": 2,
            "evaluation_audit": descriptor(AUDIT),
            "evaluation_file_manifest": descriptor(MANIFEST),
            "evaluation_file_manifest_verification": descriptor(MANIFEST_VERIFICATION),
            "file_records_aggregate_sha256": manifest["file_records_aggregate_sha256"],
        },
        "scheduler_closeout": {
            "passed": True,
            "tasks": ["t83115", "t83116"],
            "token_bound_backend_exit0": True,
            "scheduler_evidence": descriptor(SCHEDULER),
        },
        "training_scientific_stability": {
            "passed": True,
            "failure_runs": EXPECTED_FAILURE_RUNS,
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "analysis": descriptor(ANALYSIS),
            "analysis_source": descriptor(ANALYSIS_SOURCE),
            "screening_gate_passed": screening_gate_passed,
            "checks": checks,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "comparisons": report["development_scientific_result"]["comparisons"],
            "budget_results": report["development_scientific_result"]["budget_results"],
            "completion_aulc": report["development_scientific_result"]["completion_aulc"],
            "demand_scale_aulc": report["development_scientific_result"]["demand_scale_aulc"],
        },
        "analysis_execution": {
            "first_attempt_incident": descriptor(ANALYSIS_INCIDENT),
            "frozen_analysis_source_modified": False,
            "numpy_scalar_serialization_wrapper": descriptor(ANALYSIS_WRAPPER),
            "scientific_computation_changed_by_wrapper": False,
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO / "reproducibility/verify_h2oplus_v12_staged_release_evaluation_closeout_v1.py"
            ),
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": screening_gate_passed,
        "formal_confirmation_eligibility_does_not_authorize_submission": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def main() -> int:
    payload = build()
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
