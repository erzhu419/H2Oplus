#!/usr/bin/env python3
"""Bind the sealed v8 few-shot artifacts to the preregistered gate outcome."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v8_fewshot_results"
OUTPUT = RESULTS / "fewshot_closeout_v3.json"
PATHS = {
    "registration": REPO / "reproducibility/h2oplus_v8_fewshot_registration_v1.json",
    "protocol": REPO / "reproducibility/h2oplus_v8_fewshot_protocol_v1.json",
    "evaluation_audit": RESULTS / "evaluation_audit_v3.json",
    "evaluation_file_manifest": RESULTS / "evaluation_file_manifest_v3.json",
    "evaluation_file_manifest_verification": RESULTS
    / "evaluation_file_manifest_verification_v3.json",
    "analysis": RESULTS / "fewshot_analysis_v3.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build() -> dict:
    registration = json.loads(PATHS["registration"].read_text())
    audit = json.loads(PATHS["evaluation_audit"].read_text())
    manifest = json.loads(PATHS["evaluation_file_manifest"].read_text())
    verification = json.loads(
        PATHS["evaluation_file_manifest_verification"].read_text()
    )
    analysis = json.loads(PATHS["analysis"].read_text())
    result = analysis.get("development_scientific_result", {})
    checks = result.get("checks", {})
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or manifest.get("artifact_integrity", {}).get("passed") is not True
        or verification.get("passed") is not True
        or verification.get("manifest", {}).get("sha256")
        != sha256(PATHS["evaluation_file_manifest"])
        or analysis.get("schema_version")
        != "transit-h2oplus-v8-fewshot-analysis-v1"
        or analysis.get("artifact_integrity", {}).get("passed") is not True
        or analysis.get("formal_held_out_outcomes_read") is not False
        or analysis.get("supports_formal_held_out_model_effect_claim") is not False
        or analysis.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or result.get("fewshot_gate_passed") is not all(checks.values())
        or result.get("fewshot_gate_passed") is not False
        or result.get("seed_wins") != 0
    ):
        raise ValueError("few-shot artifact or gate closeout contract failed")
    return {
        "schema_version": "transit-h2oplus-v8-fewshot-closeout-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "training_runs": 10,
            "evaluation_results": 630,
            "source_partition_count": 2,
            "canonical_matrix_count": 3,
            "corrected_source_file_count": 41,
            "file_manifest_independently_verified": True,
        },
        "development_scientific_result": {
            "primary_endpoint": (
                "accrued generalized passenger seconds per departed passenger"
            ),
            "lower_is_better": True,
            "fewshot_gate_passed": False,
            "checks": checks,
            "h2oplus_mean_aulc": result["h2oplus_mean_aulc"],
            "sac_online_mean_aulc": result["sac_online_mean_aulc"],
            "h2oplus_relative_aulc_improvement": result[
                "h2oplus_relative_aulc_improvement"
            ],
            "seed_wins": result["seed_wins"],
            "seed_count": 5,
            "exact_one_sided_sign_flip_p": result[
                "exact_one_sided_sign_flip_p"
            ],
            "early_budgets": result["early_budgets"],
            "final_100000_relative_degradation": result[
                "final_100000_relative_degradation"
            ],
            "completion_aulc_delta": result["completion_aulc_delta"],
            "interpretation": (
                "The preregistered development comparison does not support a "
                "few-shot advantage for H2O+ over SAC-online."
            ),
        },
        "decision": {
            "development_campaign_closed": True,
            "fewshot_advantage_supported": False,
            "formal_confirmation_eligible": False,
            "formal_confirmation_authorized_or_submitted": False,
            "automatic_follow_on_experiment_submitted": False,
        },
        "excluded_invalid_evaluation_lineages": {
            "v1_tasks": ["t82395", "t82396", "t82397", "t82398", "t82399"],
            "v2_tasks": ["t82400", "t82401"],
            "cancelled_retries": ["t82418", "t82447"],
            "used_as_scientific_evidence": False,
        },
        "evidence": {name: descriptor(path) for name, path in PATHS.items()},
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
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
