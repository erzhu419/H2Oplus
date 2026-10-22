#!/usr/bin/env python3
"""Build the verified v19-v2 development evaluation closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPRO = REPO / "reproducibility"
RESULTS = REPRO / "h2oplus_v19_completion_safety_results"
OUTPUT = RESULTS / "evaluation_closeout_v2.json"
PATHS = {
    "audit": RESULTS / "evaluation_audit_v2.json",
    "manifest": RESULTS / "evaluation_file_manifest_v2.json",
    "manifest_verification": RESULTS / "evaluation_file_manifest_verification_v2.json",
    "scheduler": RESULTS / "evidence/evaluation_v2/scheduler_evidence_v2.json",
    "source_sync": RESULTS / "evidence/evaluation_v2/source_sync_v2.json",
    "analysis": RESULTS / "completion_safety_analysis_v2.json",
    "analysis_execution": RESULTS / "analysis_execution_evidence_v2.json",
    "correction": REPRO / "h2oplus_v19_completion_safety_environment_binding_correction_v2.json",
    "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
}
EXPECTED = {
    "audit": "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    "manifest": "db80e286f39b7d8b1a984e6e1a5ffd6b1b0747e2bd61ac463ab663aa7298b125",
    "manifest_verification": "867a110f72010f51e35f55d237ecd1b58a35a8ed6bedc1d326960775d3abb70e",
    "scheduler": "fa02ece6ce209d672bd8b994cee615703b13fd57dd6a39855d0b1b2fef0264d0",
    "source_sync": "4d1d9fe6a6b92c83e8211b974d09980a5790c64ec3f169be140dc68266e5ff10",
    "analysis": "78a2fc88b81d3f171af5812c36bcf3e7b5cdab5d8660f4a8f1f151d00565565f",
    "analysis_execution": "fff91eb84e229426ae62d1d781296b54d3d242541f6e934f239199d8a384a336",
    "correction": "2aec76694036ad038d2d15b4197c1444db8587171898ed0abdb7bdc410b4c282",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
}


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
        "bytes": resolved.stat().st_size,
    }


def checked_json(name: str) -> dict:
    path = PATHS[name].resolve(strict=True)
    if sha256(path) != EXPECTED[name]:
        raise ValueError(f"v19 closeout input drifted: {name}")
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def build() -> dict:
    audit = checked_json("audit")
    manifest = checked_json("manifest")
    manifest_verification = checked_json("manifest_verification")
    scheduler = checked_json("scheduler")
    source_sync = checked_json("source_sync")
    analysis = checked_json("analysis")
    execution = checked_json("analysis_execution")
    correction = checked_json("correction")
    data_environment = checked_json("data_environment_audit")
    data_environment_verification = checked_json("data_environment_verification")
    scientific = analysis.get("development_scientific_result", {})
    checks = scientific.get("checks", {})
    passed_checks = sorted(name for name, value in checks.items() if value is True)
    failed_checks = sorted(name for name, value in checks.items() if value is not True)
    expected_checks = {
        "all_135_results_valid",
        "all_three_matrix_manifests_complete",
        "candidate_mean_completion_gain",
        "candidate_mean_primary_noninferiority",
        "candidate_completion_gain_positive_all_demand_scales",
        "candidate_completion_gain_positive_at_least_four_training_seeds",
        "parent_artifact_integrity_passed",
        "parent_training_stability_passed",
    }
    frontend = scheduler.get("scheduler_frontend_completion", {})
    retry = scheduler.get("scheduler_generated_retry_chain", {})
    artifact = audit.get("artifact_integrity", {})
    if (
        audit.get("passed") is not True
        or artifact.get("physical_result_count") != 135
        or artifact.get("physical_behavior_count") != 15
        or artifact.get("fresh_environment_cell_count") != 9
        or artifact.get("matrix_manifest_count") != 3
        or manifest.get("passed") is not True
        or manifest.get("file_count") != 316
        or manifest_verification.get("passed") is not True
        or scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or frontend.get("observed_status") != "failed"
        or frontend.get("administrative_classifier_false_negative") is not True
        or retry.get("task_ids") != ["t83858", "t83859", "t83860"]
        or retry.get("additional_results_counted") is not False
        or source_sync.get("passed") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("v19_v1_results_reused") is not False
        or execution.get("passed") is not True
        or execution.get("frozen_analyzer_exit_code") != 0
        or execution.get("result_files_or_manifests_modified") is not False
        or execution.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or set(checks) != expected_checks
        or not all(checks.values())
        or scientific.get("paired_mechanism_screen_passed") is not True
        or analysis.get("eligible_for_full_development_control_matrix") is not True
        or correction.get("execution_boundary", {}).get(
            "completed_scientific_result_count"
        )
        != 0
        or correction.get("execution_boundary", {}).get(
            "scientific_numeric_outcomes_read"
        )
        is not False
        or data_environment.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_environment_verification.get("passed") is not True
    ):
        raise ValueError("v19 closeout prerequisites failed")
    builder = Path(__file__).resolve()
    verifier = REPRO / "verify_h2oplus_v19_completion_safety_evaluation_closeout_v2.py"
    return {
        "schema_version": "transit-h2oplus-v19-completion-safety-evaluation-closeout-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed-development-paired-mechanism-screen-passed",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "passed": True,
        "passed_scope": "verified development closeout and scientific screen passage",
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 135,
            "unique_job_signatures": 135,
            "physical_behavior_count": 15,
            "fresh_environment_cell_count": 9,
            "matrix_manifest_count": 3,
            "matrix_manifests": artifact["matrix_manifests"],
            "evaluation_audit": descriptor(PATHS["audit"]),
            "evaluation_file_manifest": descriptor(PATHS["manifest"]),
            "evaluation_file_manifest_verification": descriptor(
                PATHS["manifest_verification"]
            ),
            "file_count": manifest["file_count"],
            "file_records_aggregate_sha256": manifest["aggregate_sha256"],
        },
        "scheduler_closeout": {
            "passed": True,
            "authorized_scientific_task": "t83842",
            "token_bound_backend_exit0": True,
            "scheduler_frontend_observed_status": frontend["observed_status"],
            "scheduler_frontend_completion_verified": False,
            "administrative_classifier_false_negative": True,
            "classifier_reason": frontend["classifier_reason"],
            "backend_and_host_evidence_override_frontend_classifier": True,
            "scheduler_generated_retry_chain": retry,
            "retry_tasks_contributed_scientific_results": False,
            "scheduler_evidence": descriptor(PATHS["scheduler"]),
            "source_sync": descriptor(PATHS["source_sync"]),
        },
        "parent_training_stability": analysis["parent_training_stability"],
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "fresh_development_environment_used": True,
            "fresh_development_environment_cell_count": 9,
            "remaining_runtime_shift_from_training": True,
            "algorithm_convergence_failure_proven": False,
            "highest_priority_factors": data_environment["conclusions"][
                "highest_priority_factors"
            ],
            "audit": descriptor(PATHS["data_environment_audit"]),
            "independent_verification": descriptor(
                PATHS["data_environment_verification"]
            ),
        },
        "predeployment_correction_isolation": {
            "correction": descriptor(PATHS["correction"]),
            "v19_v1_completed_scientific_result_count": 0,
            "v19_v1_scientific_numeric_outcomes_read": False,
            "v19_v1_results_reused": False,
            "change_analysis_or_gates": False,
            "change_policy_behavior": False,
        },
        "analysis_execution": {
            "frozen_analyzer": execution["frozen_analyzer"],
            "frozen_analyzer_modified": False,
            "frozen_analyzer_exit_code": execution["frozen_analyzer_exit_code"],
            "fail_closed_wrapper": descriptor(
                REPRO / "run_frozen_h2oplus_v19_completion_safety_analysis_v2.py"
            ),
            "execution_evidence": descriptor(PATHS["analysis_execution"]),
            "registered_to_matrix_block_mapping": execution[
                "registered_to_matrix_block_mapping"
            ],
            "source_freeze_schema_adapter_verified": True,
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
        },
        "development_scientific_result": {
            "analysis": descriptor(PATHS["analysis"]),
            "candidate": scientific["candidate"],
            "paired_baseline": scientific["paired_baseline"],
            "diagnostic": scientific["diagnostic"],
            "candidate_paired_mean": scientific["candidate_paired_mean"],
            "candidate_paired_by_demand_scale": scientific[
                "candidate_paired_by_demand_scale"
            ],
            "candidate_paired_by_training_seed": scientific[
                "candidate_paired_by_training_seed"
            ],
            "diagnostic_paired_mean": scientific["diagnostic_paired_mean"],
            "gate_specification": scientific["gate_specification"],
            "checks": checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "paired_mechanism_screen_passed": True,
            "diagnostic_cannot_replace_or_reinterpret_candidate": True,
        },
        "eligible_for_full_development_control_matrix": True,
        "full_development_control_matrix_authorized_or_submitted": False,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_v19_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized_or_submitted": False,
        "claim_boundary": (
            "The v19-v2 paired mechanism development screen passed and permits a "
            "fresh full development control matrix only. It does not establish a "
            "formal held-out effect and does not create or authorize formal confirmation."
        ),
        "closeout_tools": {
            "builder": descriptor(builder),
            "independent_verifier": descriptor(verifier),
        },
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v19-v2 closeout: {OUTPUT}")
    payload = build()
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
