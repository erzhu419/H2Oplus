#!/usr/bin/env python3
"""Create the final v11 actor-hold development-evaluation closeout record."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v11_actor_hold_results"
ANALYSIS = RESULTS / "actor_hold_analysis_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v11_actor_hold.py"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "9ae635c00ed3be4137a736fc1ad76efe4ded47173962ae8cd5254d4427c3ba4b"
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
    checks = report.get("development_scientific_result", {}).get("checks", {})
    stability = report.get("training_scientific_stability", {})
    if (
        sha256(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or report.get("schema_version") != "transit-h2oplus-v11-actor-hold-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("physical_evaluation_results") != 405
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows") != 720
        or report.get("artifact_integrity", {}).get("logical_policy_budget_aliases") != 80
        or report.get("development_scientific_result", {}).get("screening_gate_passed") is not True
        or report.get("formal_confirmation_eligible") is not True
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or stability.get("passed") is not True
        or stability.get("failure_runs") != EXPECTED_FAILURE_RUNS
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or manifest.get("artifact_integrity", {}).get("result_count") != 405
        or verification.get("passed") is not True
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t82929", "t82930"}
    ):
        raise ValueError("v11 actor-hold final closeout contract failed")
    if not checks or not all(checks.values()):
        raise ValueError("v11 passed development screening gate was not preserved exactly")
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    return {
        "schema_version": "transit-h2oplus-v11-actor-hold-evaluation-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed",
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 405,
            "logical_evaluation_rows": 720,
            "canonical_matrix_count": 3,
            "source_partition_count": 2,
            "evaluation_audit": descriptor(AUDIT),
            "evaluation_file_manifest": descriptor(MANIFEST),
            "evaluation_file_manifest_verification": descriptor(MANIFEST_VERIFICATION),
            "file_records_aggregate_sha256": manifest["file_records_aggregate_sha256"],
        },
        "scheduler_closeout": {
            "passed": True,
            "tasks": ["t82929", "t82930"],
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
            "screening_gate_passed": True,
            "checks": checks,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "comparisons": report["development_scientific_result"]["comparisons"],
            "budget_results": report["development_scientific_result"]["budget_results"],
            "completion_aulc": report["development_scientific_result"]["completion_aulc"],
            "demand_scale_aulc": report["development_scientific_result"]["demand_scale_aulc"],
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO / "reproducibility/verify_h2oplus_v11_actor_hold_evaluation_closeout_v1.py"
            ),
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": True,
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
