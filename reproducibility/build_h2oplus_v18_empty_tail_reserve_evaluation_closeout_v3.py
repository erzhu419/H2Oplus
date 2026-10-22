#!/usr/bin/env python3
"""Build the verified v18-v3 empty-tail-reserve development closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v18_empty_tail_reserve_results"
OUTPUT = RESULTS / "evaluation_closeout_v3.json"

EXPECTED = {
    "analysis": "0fa89711f239b4eb798ad34d10dea79e00fd77a3274314a73d7d2e2bff745287",
    "adapter": "aadbcbc2f1a46c0f2bc624d95ca15d5addbd409bde1ff38db07e22fc38156fb4",
    "audit": "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    "manifest": "fab9283929d6bd42f5275d01cf998a7a71610594062b4ff8f3e315a9024cd986",
    "manifest_verification": "503784f1d414e637673e1fbe23358c3910f998709fc072fba6b867460dd8786e",
    "scheduler": "b514e9e354d15b919e1d2ed2f8562b1a3e94160adfcecb5968e95adaaec8a8ad",
    "source_sync": "da38e79455ee5c8d78ac7386c17d0cd38d04396efd8d8dbf2f3ba2a050e2af15",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "v1_incident": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "v1_incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
    "v2_incident": "c2cf9e59248887be24df96e53dcd4cb09a7f5c9fb5e5a8e8fd7746c6cf6d3568",
    "v2_incident_verification": "ee32840d6de1e10a7b3c9a622c1ff50ab33d63234b3cb3616d37af18a256f9b8",
}

PATHS = {
    "analysis": RESULTS / "empty_tail_reserve_analysis_v3.json",
    "adapter": RESULTS / "analysis_namespace_adapter_evidence_v3.json",
    "audit": RESULTS / "evaluation_audit_v3.json",
    "manifest": RESULTS / "evaluation_file_manifest_v3.json",
    "manifest_verification": RESULTS / "evaluation_file_manifest_verification_v3.json",
    "scheduler": RESULTS / "evidence/evaluation_v3/scheduler_evidence_v3.json",
    "source_sync": RESULTS / "evidence/evaluation_v3/source_sync_v3.json",
    "protocol": REPO / "reproducibility/h2oplus_v18_empty_tail_reserve_protocol_v1.json",
    "data_environment_audit": REPO / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPO / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
    "v1_incident": RESULTS / "evaluation_incidents/v1_parser_contract/incident_closeout_v1.json",
    "v1_incident_verification": RESULTS / "evaluation_incidents/v1_parser_contract/incident_verification_v1.json",
    "v2_incident": RESULTS / "evaluation_incidents/v2_namespace_binding/incident_closeout_v2.json",
    "v2_incident_verification": RESULTS / "evaluation_incidents/v2_namespace_binding/incident_verification_v2.json",
}

EXPECTED_CHECKS = {
    "all_135_results_valid": True,
    "all_three_matrix_manifests_complete": True,
    "candidate_mean_completion_gain": False,
    "candidate_mean_primary_noninferiority": True,
    "candidate_completion_gain_positive_all_demand_scales": True,
    "candidate_completion_gain_positive_at_least_four_training_seeds": True,
    "parent_artifact_integrity_passed": True,
    "parent_training_stability_passed": True,
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
        "bytes": path.stat().st_size,
    }


def checked_json(name: str) -> dict:
    path = PATHS[name]
    if not path.is_file() or sha256(path) != EXPECTED[name]:
        raise ValueError("v18-v3 closeout input identity drifted: {}".format(name))
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build() -> dict:
    values = {name: checked_json(name) for name in PATHS}
    analysis = values["analysis"]
    scientific = analysis.get("development_scientific_result", {})
    checks = scientific.get("checks", {})
    audit = values["audit"]
    artifact = audit.get("artifact_integrity", {})
    scheduler = values["scheduler"]
    frontend = scheduler.get("scheduler_frontend_completion", {})
    retries = scheduler.get("scheduler_generated_retry_chain", {})
    source_sync = values["source_sync"]
    manifest = values["manifest"]
    manifest_verification = values["manifest_verification"]
    adapter = values["adapter"]
    data_environment = values["data_environment_audit"]
    v1_incident = values["v1_incident"]
    v2_incident = values["v2_incident"]

    if (
        checks != EXPECTED_CHECKS
        or scientific.get("paired_mechanism_screen_passed") is not False
        or analysis.get("eligible_for_full_development_control_matrix") is not False
        or artifact.get("passed") is not True
        or artifact.get("physical_result_count") != 135
        or artifact.get("physical_behavior_count") != 15
        or artifact.get("fresh_environment_cell_count") != 9
        or artifact.get("matrix_manifest_count") != 3
        or manifest.get("passed") is not True
        or manifest.get("file_count") != 318
        or manifest_verification.get("passed") is not True
        or scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("task", {}).get("task_id") != "t83816"
        or frontend.get("administrative_classifier_false_negative") is not True
        or frontend.get("observed_status") != "failed"
        or retries.get("task_ids") != ["t83831", "t83832", "t83834"]
        or retries.get("additional_results_counted") is not False
        or source_sync.get("passed") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("v18_v1_or_v2_results_reused") is not False
        or adapter.get("passed") is not True
        or adapter.get("frozen_analyzer_exit_code") != 0
        or adapter.get("result_files_or_manifests_modified") is not False
        or adapter.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or v1_incident.get("completed_scientific_result_count") != 0
        or v1_incident.get("scientific_numeric_outcomes_read") is not False
        or v1_incident.get("v18_v1_results_reusable") is not False
        or v2_incident.get("completed_scientific_result_count") != 0
        or v2_incident.get("scientific_numeric_outcomes_read") is not False
        or v2_incident.get("v18_v2_results_reusable") is not False
        or data_environment.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or analysis.get("formal_held_out_outcomes_read") is not False
        or analysis.get("formal_confirmation_eligible") is not False
        or analysis.get("formal_confirmation_authorized") is not False
    ):
        raise ValueError("v18-v3 closeout preconditions failed")

    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    if failed_checks != ["candidate_mean_completion_gain"]:
        raise ValueError("unexpected v18-v3 scientific gate result")

    builder = Path(__file__).resolve()
    verifier = REPO / "reproducibility/verify_h2oplus_v18_empty_tail_reserve_evaluation_closeout_v3.py"
    if not verifier.is_file():
        raise FileNotFoundError(verifier)

    return {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-evaluation-closeout-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed-development-paired-mechanism-screen-failed",
        "passed": True,
        "passed_scope": "verified evaluation closeout completeness and integrity, not scientific gate passage",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 135,
            "physical_behavior_count": 15,
            "fresh_environment_cell_count": 9,
            "matrix_manifest_count": 3,
            "matrix_manifests": artifact["matrix_manifests"],
            "all_results_source_protocol_runtime_and_job_bound": True,
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
            "authorized_scientific_task": "t83816",
            "token_bound_backend_exit0": True,
            "scheduler_frontend_observed_status": frontend["observed_status"],
            "scheduler_frontend_completion_verified": False,
            "administrative_classifier_false_negative": True,
            "classifier_reason": frontend["classifier_reason"],
            "backend_and_host_evidence_override_frontend_classifier": True,
            "scheduler_generated_retry_chain": retries,
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
        "incident_isolation": {
            "v18_v1": descriptor(PATHS["v1_incident"]),
            "v18_v1_verification": descriptor(PATHS["v1_incident_verification"]),
            "v18_v2": descriptor(PATHS["v2_incident"]),
            "v18_v2_verification": descriptor(PATHS["v2_incident_verification"]),
            "v18_v1_or_v2_results_reused": False,
            "incident_scientific_numeric_outcomes_read": False,
            "incident_completed_scientific_results": 0,
        },
        "analysis_execution": {
            "frozen_analyzer": adapter["frozen_analyzer"],
            "frozen_analyzer_modified": False,
            "frozen_analyzer_exit_code": adapter["frozen_analyzer_exit_code"],
            "namespace_adapter": descriptor(
                REPO / "reproducibility/run_frozen_h2oplus_v18_empty_tail_reserve_analysis_v3.py"
            ),
            "namespace_adapter_evidence": descriptor(PATHS["adapter"]),
            "registered_to_matrix_block_mapping": adapter[
                "registered_to_matrix_block_mapping"
            ],
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
            "paired_mechanism_screen_passed": False,
            "diagnostic_cannot_replace_or_reinterpret_candidate": True,
        },
        "eligible_for_full_development_control_matrix": False,
        "full_development_control_matrix_authorized_or_submitted": False,
        "prior_v14_v15_v16_v17_development_outcomes_used_for_v18_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized_or_submitted": False,
        "claim_boundary": (
            "The v18-v3 paired mechanism development screen failed. It does not permit "
            "a full development control matrix, does not establish a formal held-out "
            "effect, and does not create or authorize formal confirmation."
        ),
        "closeout_tools": {
            "builder": descriptor(builder),
            "independent_verifier": descriptor(verifier),
        },
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite v18-v3 closeout: {}".format(OUTPUT))
    payload = build()
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
