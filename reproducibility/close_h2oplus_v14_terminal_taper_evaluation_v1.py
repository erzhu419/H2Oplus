#!/usr/bin/env python3
"""Create the final v14 terminal-taper development-evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v14_terminal_taper_results"
ANALYSIS = RESULTS / "terminal_taper_analysis_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v14_terminal_taper.py"
ANALYSIS_WRAPPER = (
    REPO
    / "reproducibility/run_frozen_h2oplus_v14_terminal_taper_analysis_v1.py"
)
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
EXPECTED_ALGORITHMS = EXPECTED_CONTROLS | {
    "h2oplus-v14-terminal-half",
    "h2oplus-v14-terminal-quarter",
    "h2oplus-v14-terminal-zero",
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


def build(expected: dict[str, str]) -> dict:
    report = json.loads(ANALYSIS.read_text())
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    verification = json.loads(MANIFEST_VERIFICATION.read_text())
    scheduler = json.loads(SCHEDULER.read_text())
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
        or sha256(ANALYSIS_WRAPPER) != expected["analysis_wrapper"]
        or sha256(ANALYSIS) != expected["analysis"]
        or sha256(AUDIT) != expected["audit"]
        or sha256(MANIFEST) != expected["manifest"]
        or sha256(MANIFEST_VERIFICATION) != expected["manifest_verification"]
        or sha256(SCHEDULER) != expected["scheduler"]
        or report.get("schema_version")
        != "transit-h2oplus-v14-terminal-taper-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("physical_evaluation_results")
        != 630
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows")
        != 1620
        or report.get("artifact_integrity", {}).get(
            "logical_policy_budget_aliases"
        )
        != 180
        or result.get("predesignated_candidate")
        != "h2oplus-v14-terminal-quarter"
        or screening_gate_passed is not calculated_gate
        or set(result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or set(result.get("mean_primary_at_25000", {})) != EXPECTED_ALGORITHMS
        or set(diagnostic.get("comparisons", {}))
        != {"h2oplus-v14-terminal-half", "h2oplus-v14-terminal-zero"}
        or len(matrix_evidence) != 3
        or {item.get("block") for item in matrix_evidence}
        != EXPECTED_MATRIX_BLOCKS
        or any(
            item.get("jobs") != 210 or item.get("source_file_count") != 47
            for item in matrix_evidence
        )
        or report.get("formal_confirmation_eligible") is not screening_gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or stability.get("passed") is not True
        or stability.get("registered_runs") != 15
        or stability.get("failure_runs") != []
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 630
        or audit.get("training_scientific_stability", {}).get(
            "all_15_training_branches_stability_gate_passed"
        )
        is not True
        or manifest.get("artifact_integrity", {}).get("result_count") != 630
        or verification.get("passed") is not True
        or independent.get("result_count") != 630
        or independent.get("physical_policy_checkpoints_rehashed") != 70
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != EXPECTED_TASKS
    ):
        raise ValueError("v14 terminal-taper final closeout contract failed")
    if bool(failed_checks) is screening_gate_passed:
        raise ValueError("v14 development gate/check consistency failed")
    return {
        "schema_version": (
            "transit-h2oplus-v14-terminal-taper-evaluation-closeout-v1"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "closed-development-screen-passed"
            if screening_gate_passed
            else "closed-development-screen-failed"
        ),
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
            "predesignated_candidate": "h2oplus-v14-terminal-quarter",
            "screening_gate_passed": screening_gate_passed,
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
            "frozen_analysis_source_modified": False,
            "block_label_adapter": descriptor(ANALYSIS_WRAPPER),
            "registered_to_matrix_block_mapping": {
                "v12-staged-release-development-01": "sumo-seed-5101",
                "v12-staged-release-development-02": "sumo-seed-5102",
                "v12-staged-release-development-03": "sumo-seed-5103",
            },
            "performance_values_or_preregistered_gate_formulas_changed": False,
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO
                / "reproducibility/verify_h2oplus_v14_terminal_taper_evaluation_closeout_v1.py"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "analysis",
        "analysis_wrapper",
        "audit",
        "manifest",
        "manifest_verification",
        "scheduler",
    ):
        parser.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256",
            required=True,
        )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = {
        name: getattr(args, "expected_" + name + "_sha256")
        for name in (
            "analysis",
            "analysis_wrapper",
            "audit",
            "manifest",
            "manifest_verification",
            "scheduler",
        )
    }
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in expected.values()
    ):
        raise ValueError("invalid expected closeout SHA256")
    payload = build(expected)
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
