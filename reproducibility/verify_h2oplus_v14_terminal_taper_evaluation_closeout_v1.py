#!/usr/bin/env python3
"""Independently verify the final v14 terminal-taper evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v14_terminal_taper_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v1.json"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "0df400805ac90e22be18a016bad42ce884cd946d576bdb95ad1d07afdbe0b297"
)
EXPECTED_TASKS = {"t83584", "t83585"}
EXPECTED_CONTROLS = {
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
}
EXPECTED_MAPPING = {
    "v12-staged-release-development-01": "sumo-seed-5101",
    "v12-staged-release-development-02": "sumo-seed-5102",
    "v12-staged-release-development-03": "sumo-seed-5103",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checked_descriptor(item: dict) -> Path:
    path = (REPO / item["path"]).resolve()
    if (
        not path.is_file()
        or REPO not in path.parents
        or path.stat().st_size != item.get("size_bytes")
        or sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"closeout descriptor failed: {item}")
    return path


def verify(expected_closeout: str, expected_wrapper: str) -> dict:
    if sha256(CLOSEOUT) != expected_closeout:
        raise ValueError("v14 closeout identity drifted")
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    result = closeout.get("development_scientific_result", {})
    stability = closeout.get("training_scientific_stability", {})
    scheduler = closeout.get("scheduler_closeout", {})
    execution = closeout.get("analysis_execution", {})
    screening_gate_passed = result.get("screening_gate_passed")
    expected_status = (
        "closed-development-screen-passed"
        if screening_gate_passed is True
        else "closed-development-screen-failed"
    )
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v14-terminal-taper-evaluation-closeout-v1"
        or closeout.get("status") != expected_status
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_evaluation_results") != 630
        or artifact.get("logical_evaluation_rows") != 1620
        or artifact.get("logical_policy_budget_aliases") != 180
        or artifact.get("physical_policy_checkpoints_independently_rehashed")
        != 70
        or scheduler.get("passed") is not True
        or set(scheduler.get("tasks", [])) != EXPECTED_TASKS
        or scheduler.get("token_bound_backend_exit0") is not True
        or stability.get("passed") is not True
        or stability.get("registered_runs") != 15
        or stability.get("failure_runs") != []
        or result.get("predesignated_candidate")
        != "h2oplus-v14-terminal-quarter"
        or not isinstance(screening_gate_passed, bool)
        or execution.get("frozen_analysis_source_modified") is not False
        or execution.get(
            "performance_values_or_preregistered_gate_formulas_changed"
        )
        is not False
        or execution.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_formal_held_out_model_effect_claim") is not False
        or closeout.get("formal_confirmation_eligible") is not screening_gate_passed
        or closeout.get(
            "formal_confirmation_eligibility_does_not_authorize_submission"
        )
        is not True
        or closeout.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
    ):
        raise ValueError("independent v14 closeout scope check failed")
    paths = {
        "audit": checked_descriptor(artifact["evaluation_audit"]),
        "manifest": checked_descriptor(artifact["evaluation_file_manifest"]),
        "manifest_verification": checked_descriptor(
            artifact["evaluation_file_manifest_verification"]
        ),
        "scheduler": checked_descriptor(scheduler["scheduler_evidence"]),
        "analysis": checked_descriptor(result["analysis"]),
        "analysis_source": checked_descriptor(result["analysis_source"]),
        "analysis_wrapper": checked_descriptor(execution["block_label_adapter"]),
        "closeout_builder": checked_descriptor(closeout["closeout_tools"]["builder"]),
        "closeout_verifier": checked_descriptor(
            closeout["closeout_tools"]["independent_verifier"]
        ),
    }
    report = json.loads(paths["analysis"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    manifest_verification = json.loads(paths["manifest_verification"].read_text())
    scheduler_evidence = json.loads(paths["scheduler"].read_text())
    report_result = report.get("development_scientific_result", {})
    checks = report_result.get("checks", {})
    failed = {key for key, value in checks.items() if value is not True}
    passed = {key for key, value in checks.items() if value is True}
    diagnostic = report.get("diagnostic_results_not_used_as_selection_gates", {})
    matrix_evidence = report.get("matrix_evidence", [])
    if (
        sha256(paths["analysis_source"]) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or sha256(paths["analysis_wrapper"]) != expected_wrapper
        or not checks
        or all(value is True for value in checks.values())
        is not screening_gate_passed
        or bool(failed) is screening_gate_passed
        or set(result.get("failed_checks", [])) != failed
        or set(result.get("passed_checks", [])) != passed
        or result.get("checks") != checks
        or report_result.get("screening_gate_passed") is not screening_gate_passed
        or report_result.get("predesignated_candidate")
        != "h2oplus-v14-terminal-quarter"
        or set(report_result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or set(diagnostic.get("comparisons", {}))
        != {"h2oplus-v14-terminal-half", "h2oplus-v14-terminal-zero"}
        or len(matrix_evidence) != 3
        or {item.get("block") for item in matrix_evidence}
        != set(EXPECTED_MAPPING.values())
        or any(
            item.get("jobs") != 210 or item.get("source_file_count") != 47
            for item in matrix_evidence
        )
        or report.get("formal_confirmation_eligible") is not screening_gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or report.get("artifact_integrity", {}).get(
            "physical_evaluation_results"
        )
        != 630
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows")
        != 1620
        or report.get("artifact_integrity", {}).get(
            "logical_policy_budget_aliases"
        )
        != 180
        or manifest.get("file_records_aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest.get("artifact_integrity", {}).get("result_count") != 630
        or manifest_verification.get("manifest", {}).get("sha256")
        != sha256(paths["manifest"])
        or manifest_verification.get("independent_artifact_audit", {}).get(
            "physical_policy_checkpoints_rehashed"
        )
        != 70
        or manifest_verification.get("independent_artifact_audit", {}).get(
            "result_count"
        )
        != 630
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or set(scheduler_evidence.get("tasks", {})) != EXPECTED_TASKS
    ):
        raise ValueError("independent v14 closeout identity or gate check failed")
    return {
        "schema_version": (
            "transit-h2oplus-v14-terminal-taper-evaluation-closeout-verification-v1"
        ),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": {
            "path": str(CLOSEOUT.relative_to(REPO)),
            "sha256": sha256(CLOSEOUT),
            "size_bytes": CLOSEOUT.stat().st_size,
        },
        "artifact_integrity_verified": True,
        "physical_evaluation_results_verified": 630,
        "logical_evaluation_rows_verified": 1620,
        "physical_policy_checkpoints_rehashed": 70,
        "training_scientific_stability_passed": True,
        "training_scientific_stability_verified_separately": True,
        "development_screening_gate_passed": screening_gate_passed,
        "development_screening_gate_value_verified": True,
        "failed_checks": sorted(failed),
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": screening_gate_passed,
        "formal_confirmation_eligibility_does_not_authorize_submission": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-closeout-sha256", required=True)
    parser.add_argument("--expected-analysis-wrapper-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = (args.expected_closeout_sha256, args.expected_analysis_wrapper_sha256)
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("invalid expected v14 closeout SHA256")
    payload = verify(*values)
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(payload["closeout"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
