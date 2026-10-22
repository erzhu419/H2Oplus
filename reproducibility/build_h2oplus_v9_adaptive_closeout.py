#!/usr/bin/env python3
"""Bind sealed v9 adaptive artifacts to the preregistered development outcome."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v9_adaptive_results"
OUTPUT = RESULTS / "adaptive_closeout_v3.json"
FILE_MANIFEST = RESULTS / "adaptive_closeout_file_manifest_v3.json"
PATHS = {
    "registration": REPO / "reproducibility/h2oplus_v9_adaptive_registration_v2.json",
    "protocol": REPO / "reproducibility/h2oplus_v9_adaptive_protocol_v2.json",
    "source_freeze": REPO
    / "reproducibility/h2oplus_v9_adaptive_evaluation_source_freeze_v2.json",
    "methods": REPO / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v2.json",
    "training_audit": RESULTS / "training_audit.json",
    "training_closeout": RESULTS / "training_closeout.json",
    "training_file_manifest": RESULTS / "training_file_manifest.json",
    "training_verification": RESULTS / "training_file_manifest_verification.json",
    "evaluation_audit": RESULTS / "evaluation_audit_v3.json",
    "evaluation_file_manifest": RESULTS / "evaluation_file_manifest_v3.json",
    "evaluation_verification": RESULTS / "evaluation_file_manifest_verification_v3.json",
    "scheduler_evidence": RESULTS / "evidence/evaluation_v3/scheduler_evidence_v3.json",
    "analysis": RESULTS / "adaptive_analysis_v3.json",
    "analysis_serialization_correction": RESULTS
    / "analysis_serialization_correction_v1.json",
}
EXPECTED_CHECKS = {
    "combined_relative_aulc_over_fixed_h2o",
    "combined_relative_aulc_over_sac",
    "combined_seed_wins_over_fixed_h2o",
    "combined_seed_wins_over_sac",
    "combined_mean_lower_at_10000_and_25000",
    "completion_rate_noninferiority",
    "demand_scale_robustness",
    "intervention_attribution",
}


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


def records_aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_closeout() -> dict:
    payloads = {name: json.loads(path.read_text()) for name, path in PATHS.items()}
    registration = payloads["registration"]
    training = payloads["training_closeout"]
    audit = payloads["evaluation_audit"]
    evaluation_manifest = payloads["evaluation_file_manifest"]
    verification = payloads["evaluation_verification"]
    analysis = payloads["analysis"]
    correction = payloads["analysis_serialization_correction"]
    result = analysis.get("development_scientific_result", {})
    checks = result.get("checks", {})
    screening_passed = all(checks.values())
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 252
        or evaluation_manifest.get("artifact_integrity", {}).get("passed") is not True
        or verification.get("passed") is not True
        or verification.get("manifest", {}).get("sha256")
        != sha256(PATHS["evaluation_file_manifest"])
        or analysis.get("schema_version")
        != "transit-h2oplus-v9-adaptive-analysis-v1"
        or analysis.get("artifact_integrity", {}).get("passed") is not True
        or analysis.get("artifact_integrity", {}).get("physical_evaluation_rows")
        != 252
        or set(checks) != EXPECTED_CHECKS
        or result.get("screening_gate_passed") is not screening_passed
        or result.get("screening_gate_passed") is not False
        or checks.get("completion_rate_noninferiority") is not True
        or any(value is not False for key, value in checks.items() if key != "completion_rate_noninferiority")
        or training.get("ratio_intervention_ever_applied") is not False
        or training.get("intervention_attribution_available") is not False
        or training.get("scientific_stability_passed") is not False
        or correction.get("passed") is not True
        or correction.get("correction_scope", {}).get("analysis_metrics_changed")
        is not False
        or correction.get("report", {}).get("sha256") != sha256(PATHS["analysis"])
        or analysis.get("formal_held_out_outcomes_read") is not False
        or analysis.get("supports_formal_held_out_model_effect_claim") is not False
        or analysis.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("v9 adaptive closeout contract failed")

    return {
        "schema_version": "transit-h2oplus-v9-adaptive-closeout-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "training_runs": 9,
            "physical_evaluation_results": 252,
            "source_partition_count": 2,
            "canonical_matrix_count": 2,
            "corrected_source_file_count": 41,
            "evaluation_file_manifest_independently_verified": True,
        },
        "training_stability": {
            "scientific_stability_passed": False,
            "adaptation_seed6202_late_stability_passed": False,
            "ratio_intervention_ever_applied": False,
            "intervention_attribution_available": False,
            "separate_from_artifact_integrity": True,
        },
        "development_scientific_result": {
            "primary_endpoint": (
                "accrued generalized passenger seconds per departed passenger"
            ),
            "lower_is_better": True,
            "screening_gate_passed": False,
            "checks": checks,
            "mean_aulc": result["mean_aulc"],
            "combined_relative_aulc_improvement_over_fixed_h2o": result[
                "combined_relative_aulc_improvement_over_fixed_h2o"
            ],
            "combined_relative_aulc_improvement_over_sac": result[
                "combined_relative_aulc_improvement_over_sac"
            ],
            "combined_seed_wins_over_fixed_h2o": result[
                "combined_seed_wins_over_fixed_h2o"
            ],
            "combined_seed_wins_over_sac": result["combined_seed_wins_over_sac"],
            "early_budgets": result["early_budgets"],
            "demand_scale_aulc": result["demand_scale_aulc"],
            "factorial_mechanism_diagnostics": result[
                "factorial_mechanism_diagnostics"
            ],
            "interpretation": (
                "The preregistered development screening does not support the v9 "
                "combined adaptive method over fixed H2O+ and SAC-online."
            ),
        },
        "decision": {
            "development_campaign_closed": True,
            "adaptive_screening_supported": False,
            "formal_confirmation_eligible": False,
            "formal_confirmation_authorized_or_submitted": False,
            "automatic_follow_on_experiment_submitted": False,
        },
        "excluded_invalid_evaluation_lineages": {
            "v1_tasks": ["t82565", "t82566"],
            "v2_submitted": False,
            "used_as_scientific_evidence": False,
        },
        "analysis_serialization_correction": {
            "serialization_only": True,
            "frozen_analysis_source_modified": False,
            "preregistered_gate_logic_changed": False,
        },
        "evidence": {name: descriptor(path) for name, path in PATHS.items()},
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "passed": True,
    }


def build_file_manifest(closeout: dict) -> dict:
    files = set(PATHS.values()) | {
        OUTPUT,
        Path(__file__).resolve(),
        REPO / "reproducibility/analyze_h2oplus_v9_adaptive.py",
        REPO / "reproducibility/run_h2oplus_v9_adaptive_analysis_closeout.py",
        REPO / "reproducibility/build_h2oplus_v9_adaptive_evaluation_manifest.py",
        REPO / "reproducibility/verify_h2oplus_v9_adaptive_evaluation_manifest.py",
    }
    records = [descriptor(path) for path in sorted(files)]
    return {
        "schema_version": "transit-h2oplus-v9-adaptive-closeout-file-manifest-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": descriptor(OUTPUT),
        "file_count": len(records),
        "file_records_aggregate_sha256": records_aggregate(records),
        "files": records,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "artifact_integrity_passed": True,
        "development_screening_gate_passed": closeout[
            "development_scientific_result"
        ]["screening_gate_passed"],
        "formal_confirmation_eligible": False,
        "formal_held_out_outcomes_read": False,
        "passed": True,
    }


def main() -> int:
    closeout = build_closeout()
    atomic_json(OUTPUT, closeout)
    manifest = build_file_manifest(closeout)
    atomic_json(FILE_MANIFEST, manifest)
    print(OUTPUT)
    print(sha256(OUTPUT))
    print(FILE_MANIFEST)
    print(sha256(FILE_MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
