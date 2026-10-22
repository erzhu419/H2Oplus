#!/usr/bin/env python3
"""Create the final v13 proximal-release development-evaluation closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v13_proximal_release_results"
ANALYSIS = RESULTS / "proximal_release_analysis_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
ANALYSIS_INCIDENT = (
    RESULTS / "evaluation_incidents/analysis_block_label_contract_v1.json"
)
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v13_proximal_release.py"
ANALYSIS_WRAPPER = (
    REPO
    / "reproducibility/run_frozen_h2oplus_v13_proximal_release_analysis_v1.py"
)
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "d15a5aa4159f3a29ad9dd5fe28d9f0477be3209637cfe43754dab2790a7add88"
)
EXPECTED_ANALYSIS_WRAPPER_SHA256 = (
    "a17d7d0ff845c1761187ce53bba24147169524ef47c66e23c046c3f7feb3236b"
)
EXPECTED_INCIDENT_SHA256 = (
    "7a545bdba7db2f073bd825889d89cbdeff2ea5ed4c5a79d9ebe81ed7225b52c0"
)
EXPECTED_TASKS = {"t83401", "t83402"}
EXPECTED_CONTROLS = {
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
}
EXPECTED_ALGORITHMS = EXPECTED_CONTROLS | {
    "h2oplus-v13-proximal-weak",
    "h2oplus-v13-proximal-medium",
    "h2oplus-v13-proximal-strong",
}
EXPECTED_MATRIX_BLOCKS = {"sumo-seed-5101", "sumo-seed-5102", "sumo-seed-5103"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(REPO)),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    result = report.get("development_scientific_result", {})
    checks = result.get("checks", {})
    stability = report.get("training_scientific_stability", {})
    screening_gate_passed = result.get("screening_gate_passed")
    calculated_gate = bool(checks) and all(value is True for value in checks.values())
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    matrix_evidence = report.get("matrix_evidence", [])
    diagnostic = report.get("diagnostic_results_not_used_as_selection_gates", {})
    independent = verification.get("independent_artifact_audit", {})
    if (
        sha256(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or sha256(ANALYSIS_WRAPPER) != EXPECTED_ANALYSIS_WRAPPER_SHA256
        or sha256(ANALYSIS_INCIDENT) != EXPECTED_INCIDENT_SHA256
        or report.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("physical_evaluation_results")
        != 585
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows")
        != 1440
        or report.get("artifact_integrity", {}).get(
            "logical_policy_budget_aliases"
        )
        != 160
        or result.get("predesignated_candidate")
        != "h2oplus-v13-proximal-medium"
        or screening_gate_passed is not calculated_gate
        or screening_gate_passed is not False
        or set(result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or set(result.get("mean_primary_at_25000", {})) != EXPECTED_ALGORITHMS
        or set(diagnostic.get("comparisons", {}))
        != {"h2oplus-v13-proximal-weak", "h2oplus-v13-proximal-strong"}
        or len(matrix_evidence) != 3
        or {item.get("block") for item in matrix_evidence}
        != EXPECTED_MATRIX_BLOCKS
        or any(
            item.get("jobs") != 195 or item.get("source_file_count") != 45
            for item in matrix_evidence
        )
        or report.get("formal_confirmation_eligible") is not False
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or stability.get("passed") is not True
        or stability.get("registered_runs") != 15
        or stability.get("failure_runs") != []
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 585
        or audit.get("training_scientific_stability", {}).get(
            "all_15_training_branches_stability_gate_passed"
        )
        is not True
        or manifest.get("artifact_integrity", {}).get("result_count") != 585
        or verification.get("passed") is not True
        or independent.get("result_count") != 585
        or independent.get("physical_policy_checkpoints_rehashed") != 65
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != EXPECTED_TASKS
        or incident.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-analysis-incident-v1"
        or incident.get("analysis_output_created") is not False
        or incident.get("partial_output_created") is not False
        or incident.get("development_evaluation_result_payloads_read") is not False
        or incident.get("formal_held_out_outcomes_read") is not False
        or incident.get("frozen_analyzer_modified") is not False
        or incident.get("local_sumo_executed") is not False
        or incident.get("local_training_executed") is not False
        or incident.get("analyzer", {}).get("sha256")
        != EXPECTED_ANALYSIS_SOURCE_SHA256
        or incident.get("operational_resolution", {}).get(
            "performance_values_or_preregistered_gate_formulas_changed"
        )
        is not False
    ):
        raise ValueError("v13 proximal-release final closeout contract failed")
    if not failed_checks:
        raise ValueError("failed v13 development gate lacks failed checks")
    return {
        "schema_version": (
            "transit-h2oplus-v13-proximal-release-evaluation-closeout-v1"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed-development-screen-failed",
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 585,
            "logical_evaluation_rows": 1440,
            "logical_policy_budget_aliases": 160,
            "physical_policy_checkpoints_independently_rehashed": 65,
            "canonical_matrix_count": 3,
            "source_partition_count": 2,
            "evaluation_audit": descriptor(AUDIT),
            "evaluation_file_manifest": descriptor(MANIFEST),
            "evaluation_file_manifest_verification": descriptor(
                MANIFEST_VERIFICATION
            ),
            "file_records_aggregate_sha256": manifest[
                "file_records_aggregate_sha256"
            ],
        },
        "scheduler_closeout": {
            "passed": True,
            "tasks": sorted(EXPECTED_TASKS),
            "token_bound_backend_exit0": True,
            "scheduler_evidence": descriptor(SCHEDULER),
        },
        "training_scientific_stability": {
            "passed": True,
            "registered_runs": 15,
            "failure_runs": [],
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "analysis": descriptor(ANALYSIS),
            "analysis_source": descriptor(ANALYSIS_SOURCE),
            "predesignated_candidate": "h2oplus-v13-proximal-medium",
            "screening_gate_passed": False,
            "checks": checks,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "control_comparisons": result["control_comparisons"],
            "mean_primary_at_25000": result["mean_primary_at_25000"],
            "completion_aulc_against_controls": result[
                "completion_aulc_against_controls"
            ],
            "demand_scale_aulc_against_sac": result[
                "demand_scale_aulc_against_sac"
            ],
            "diagnostic_results_not_used_as_selection_gates": diagnostic,
        },
        "analysis_execution": {
            "first_attempt_incident": descriptor(ANALYSIS_INCIDENT),
            "frozen_analysis_source_modified": False,
            "block_label_adapter": descriptor(ANALYSIS_WRAPPER),
            "registered_to_matrix_block_mapping": incident[
                "operational_resolution"
            ]["mapping"],
            "performance_values_or_preregistered_gate_formulas_changed": False,
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO
                / "reproducibility/verify_h2oplus_v13_proximal_release_evaluation_closeout_v1.py"
            ),
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": False,
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
