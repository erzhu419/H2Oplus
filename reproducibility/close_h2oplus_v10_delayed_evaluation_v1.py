#!/usr/bin/env python3
"""Create the final v10 delayed development-evaluation closeout record."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v10_delayed_results"
ANALYSIS = RESULTS / "delayed_analysis_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v10_delayed.py"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "4542031ccfd73f78001de236e60e6b6df9df36054318669ffbc5558e5437d574"
)
EXPECTED_FAILURE_RUNS = ["h2oplus_v10c4_delayed_order_seed6205"]


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
    checks = report.get("development_scientific_result", {}).get("checks", {})
    stability = report.get("training_scientific_stability", {})
    if (
        sha256(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or report.get("schema_version") != "transit-h2oplus-v10-delayed-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("physical_evaluation_results") != 540
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows") != 1260
        or report.get("artifact_integrity", {}).get("logical_policy_budget_aliases") != 140
        or report.get("development_scientific_result", {}).get("screening_gate_passed") is not False
        or report.get("formal_confirmation_eligible") is not False
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or stability.get("passed") is not False
        or stability.get("failure_runs") != EXPECTED_FAILURE_RUNS
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or manifest.get("artifact_integrity", {}).get("result_count") != 540
        or verification.get("passed") is not True
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t82706", "t82707"}
    ):
        raise ValueError("v10 delayed final closeout contract failed")
    if not checks or all(checks.values()) or checks.get("all_30_training_runs_stability") is not False:
        raise ValueError("v10 failed screening gate was not preserved exactly")
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    return {
        "schema_version": "transit-h2oplus-v10-delayed-evaluation-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed",
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 540,
            "logical_evaluation_rows": 1260,
            "canonical_matrix_count": 3,
            "source_partition_count": 2,
            "evaluation_audit": descriptor(AUDIT),
            "evaluation_file_manifest": descriptor(MANIFEST),
            "evaluation_file_manifest_verification": descriptor(MANIFEST_VERIFICATION),
            "file_records_aggregate_sha256": manifest["file_records_aggregate_sha256"],
        },
        "scheduler_closeout": {
            "passed": True,
            "tasks": ["t82706", "t82707"],
            "token_bound_backend_exit0": True,
            "scheduler_evidence": descriptor(SCHEDULER),
        },
        "training_scientific_stability": {
            "passed": False,
            "failure_runs": EXPECTED_FAILURE_RUNS,
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "analysis": descriptor(ANALYSIS),
            "analysis_source": descriptor(ANALYSIS_SOURCE),
            "screening_gate_passed": False,
            "checks": checks,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "comparison_over_sac": report["development_scientific_result"]["comparison_over_sac"],
            "comparison_over_fixed_h2o": report["development_scientific_result"]["comparison_over_fixed_h2o"],
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO / "reproducibility/verify_h2oplus_v10_delayed_evaluation_closeout_v1.py"
            ),
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": False,
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
