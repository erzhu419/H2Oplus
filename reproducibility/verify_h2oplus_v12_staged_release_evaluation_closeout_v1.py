#!/usr/bin/env python3
"""Independently verify the final v12 staged-release evaluation closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v12_staged_release_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v1.json"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "a77ffa04f3e2f2bfc5c6e3495505cb70cace2326df54e6906f6fcca34a9aa4e4"
)


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


def _checked_descriptor(item: dict) -> Path:
    path = (REPO / item["path"]).resolve()
    if (
        not path.is_file()
        or REPO not in path.parents
        or path.stat().st_size != item.get("size_bytes")
        or sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"closeout descriptor failed: {item}")
    return path


def verify() -> dict:
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    result = closeout.get("development_scientific_result", {})
    stability = closeout.get("training_scientific_stability", {})
    scheduler = closeout.get("scheduler_closeout", {})
    if (
        closeout.get("schema_version") != "transit-h2oplus-v12-staged-release-evaluation-closeout-v1"
        or closeout.get("status") != "closed"
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_evaluation_results") != 630
        or artifact.get("logical_evaluation_rows") != 1620
        or artifact.get("logical_policy_budget_aliases") != 180
        or artifact.get("physical_policy_checkpoints_independently_rehashed") != 70
        or scheduler.get("passed") is not True
        or scheduler.get("tasks") != ["t83115", "t83116"]
        or scheduler.get("token_bound_backend_exit0") is not True
        or stability.get("passed") is not True
        or stability.get("failure_runs") != []
        or result.get("screening_gate_passed") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_formal_held_out_model_effect_claim") is not False
        or closeout.get("formal_confirmation_eligible") is not False
        or closeout.get("formal_confirmation_eligibility_does_not_authorize_submission") is not True
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("independent v12 evaluation closeout scope check failed")
    paths = {
        "audit": _checked_descriptor(artifact["evaluation_audit"]),
        "manifest": _checked_descriptor(artifact["evaluation_file_manifest"]),
        "manifest_verification": _checked_descriptor(
            artifact["evaluation_file_manifest_verification"]
        ),
        "scheduler": _checked_descriptor(scheduler["scheduler_evidence"]),
        "analysis": _checked_descriptor(result["analysis"]),
        "analysis_source": _checked_descriptor(result["analysis_source"]),
        "analysis_incident": _checked_descriptor(
            closeout["analysis_execution"]["first_attempt_incident"]
        ),
        "analysis_wrapper": _checked_descriptor(
            closeout["analysis_execution"]["numpy_scalar_serialization_wrapper"]
        ),
        "closeout_builder": _checked_descriptor(closeout["closeout_tools"]["builder"]),
        "closeout_verifier": _checked_descriptor(
            closeout["closeout_tools"]["independent_verifier"]
        ),
    }
    report = json.loads(paths["analysis"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    manifest_verification = json.loads(paths["manifest_verification"].read_text())
    scheduler_evidence = json.loads(paths["scheduler"].read_text())
    incident = json.loads(paths["analysis_incident"].read_text())
    checks = report.get("development_scientific_result", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if value is not True)
    passed = sorted(key for key, value in checks.items() if value is True)
    if (
        sha256(paths["analysis_source"]) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or not checks
        or all(value is True for value in checks.values()) is not False
        or report.get("development_scientific_result", {}).get("screening_gate_passed") is not False
        or report.get("formal_confirmation_eligible") is not False
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or report.get("artifact_integrity", {}).get("physical_evaluation_results") != 630
        or report.get("artifact_integrity", {}).get("logical_evaluation_rows") != 1620
        or report.get("artifact_integrity", {}).get("logical_policy_budget_aliases") != 180
        or result.get("checks") != checks
        or result.get("failed_checks") != failed
        or result.get("passed_checks") != passed
        or not failed
        or manifest.get("file_records_aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest.get("artifact_integrity", {}).get("result_count") != 630
        or manifest_verification.get("manifest", {}).get("sha256")
        != sha256(paths["manifest"])
        or manifest_verification.get("independent_artifact_audit", {}).get(
            "physical_policy_checkpoints_rehashed"
        )
        != 70
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or set(scheduler_evidence.get("tasks", {})) != {"t83115", "t83116"}
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
        or closeout.get("analysis_execution", {}).get(
            "frozen_analysis_source_modified"
        )
        is not False
        or closeout.get("analysis_execution", {}).get(
            "scientific_computation_changed_by_wrapper"
        )
        is not False
    ):
        raise ValueError("independent v12 closeout identity or gate check failed")
    return {
        "schema_version": "transit-h2oplus-v12-staged-release-evaluation-closeout-verification-v1",
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
        "development_screening_gate_passed": False,
        "development_screening_gate_value_verified": True,
        "failed_checks": failed,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_eligibility_does_not_authorize_submission": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def main() -> int:
    payload = verify()
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(payload["closeout"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
