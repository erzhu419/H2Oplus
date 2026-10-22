#!/usr/bin/env python3
"""Independently verify the final v17-v2 tail-service evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v2.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v2.json"
PROTOCOL = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "66cfff74c8d1b4988b314945aa8ea06d03fe1594db8cc0f76ca3396bfab37f1a"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790"
)
EXPECTED_DATA_ENV_AUDIT_SHA256 = (
    "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9"
)
EXPECTED_DATA_ENV_VERIFICATION_SHA256 = (
    "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390"
)
EXPECTED_CLASSIFIER_REASON = (
    "never entered training: lifetime 10834s but peak_vram=0 and no training "
    "markers in log tail (likely OOM/silent-kill mid-init)"
)
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
RELATIVE_THRESHOLD_KEYS = {
    "h2oplus": "candidate_relative_aulc_improvement_over_fixed_h2o_at_least",
    "sac-online": "candidate_relative_aulc_improvement_over_sac_at_least",
    "h2oplus-v10-delayed-actor": (
        "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least"
    ),
    "h2oplus-v11-actor-hold": (
        "candidate_relative_aulc_improvement_over_v11_at_least"
    ),
    "h2oplus-v12-actor-ramp": (
        "candidate_relative_aulc_improvement_over_v12_actor_ramp_at_least"
    ),
    "h2oplus-v13-proximal-strong": (
        "candidate_relative_aulc_improvement_over_v13_strong_at_least"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checked_descriptor(item: dict) -> Path:
    path = (REPO / item.get("path", "")).resolve()
    if (
        not path.is_file()
        or REPO not in path.parents
        or path.stat().st_size != item.get("size_bytes")
        or sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"v17 closeout descriptor failed: {item}")
    return path


def independent_relative_failures(result: dict, thresholds: dict) -> list[dict]:
    rows = []
    for control, key in RELATIVE_THRESHOLD_KEYS.items():
        observed = float(result["control_comparisons"][control]["relative_improvement"])
        required = float(thresholds[key])
        if observed < required:
            rows.append(
                {
                    "control": control,
                    "observed_relative_improvement": observed,
                    "required_at_least": required,
                    "shortfall": required - observed,
                }
            )
    return rows


def independent_completion_failures(result: dict, thresholds: dict) -> list[dict]:
    margin = float(thresholds["completion_rate_noninferiority_margin"])
    rows = []
    for control, item in result["completion_aulc_against_controls"].items():
        delta = float(item["delta"])
        if delta < -margin:
            rows.append(
                {
                    "control": control,
                    "observed_delta": delta,
                    "minimum_allowed_delta": -margin,
                    "shortfall": -margin - delta,
                }
            )
    return rows


def verify(expected_closeout: str) -> dict:
    if sha256(CLOSEOUT) != expected_closeout:
        raise ValueError("v17 closeout identity drifted")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v17 protocol identity drifted during closeout verification")
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    scheduler = closeout.get("scheduler_closeout", {})
    stability = closeout.get("parent_training_scientific_stability", {})
    result = closeout.get("development_scientific_result", {})
    execution = closeout.get("analysis_execution", {})
    risks = closeout.get("data_environment_validity_risks", {})
    gate_passed = result.get("screening_gate_passed")
    expected_status = (
        "closed-development-screen-passed"
        if gate_passed is True
        else "closed-development-screen-failed"
    )
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-evaluation-closeout-v2"
        or closeout.get("status") != expected_status
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_evaluation_results") != 135
        or artifact.get("physical_behaviors") != 15
        or artifact.get("unique_parent_checkpoints_independently_rehashed") != 5
        or artifact.get("matrix_count") != 3
        or artifact.get("source_file_count") != 53
        or scheduler.get("passed") is not True
        or scheduler.get("task") != "t83785"
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("scheduler_terminal_evidence_verified") is not True
        or scheduler.get("scheduler_frontend_completion_verified") is not False
        or scheduler.get("scheduler_frontend_observed_status") != "failed"
        or scheduler.get("administrative_classifier_false_negative") is not True
        or scheduler.get("backend_and_host_evidence_override_frontend_classifier")
        is not True
        or scheduler.get("classifier_reason") != EXPECTED_CLASSIFIER_REASON
        or stability.get("passed") is not True
        or stability.get("registered_runs") != 15
        or stability.get("new_v17_training_runs") != 0
        or result.get("predesignated_candidate")
        != "h2oplus-v17-tail-service-reserve-thirty"
        or not isinstance(gate_passed, bool)
        or execution.get("frozen_analysis_source_modified") is not False
        or execution.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
        or execution.get("result_files_or_manifests_modified") is not False
        or execution.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or execution.get("v17_v2_operational_identity_only") is not True
        or execution.get("scientific_matrix_sources_exactly_equal_to_v17_v1")
        is not True
        or execution.get("v17_v1_incident_results_reused") is not False
        or execution.get("v17_v1_incident_numeric_outcomes_inspected") is not False
        or risks.get("material_factors_present") is not True
        or risks.get("algorithm_convergence_failure_proven") is not False
        or risks.get("v17_v2_design_changed_after_audit") is not False
        or risks.get(
            "separate_from_artifact_integrity_parent_training_and_scientific_result"
        )
        is not True
        or closeout.get("design_reuse_disclosure", {}).get(
            "prior_v14_v15_v16_development_outcomes_used_for_v17_design"
        )
        is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_formal_held_out_model_effect_claim") is not False
        or closeout.get("formal_confirmation_eligible") is not gate_passed
        or closeout.get(
            "formal_confirmation_eligibility_does_not_authorize_submission"
        )
        is not True
        or closeout.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
    ):
        raise ValueError("independent v17 closeout scope check failed")

    paths = {
        "audit": checked_descriptor(artifact["evaluation_audit"]),
        "manifest": checked_descriptor(artifact["evaluation_file_manifest"]),
        "manifest_verification": checked_descriptor(
            artifact["evaluation_file_manifest_verification"]
        ),
        "scheduler": checked_descriptor(scheduler["scheduler_evidence"]),
        "source_sync": checked_descriptor(scheduler["source_sync"]),
        "analysis": checked_descriptor(result["analysis"]),
        "analysis_source": checked_descriptor(result["analysis_source"]),
        "analysis_wrapper": checked_descriptor(execution["block_label_adapter"]),
        "adapter_evidence": checked_descriptor(
            execution["block_label_adapter_evidence"]
        ),
        "data_environment_audit": checked_descriptor(risks["audit"]),
        "data_environment_verification": checked_descriptor(
            risks["independent_verification"]
        ),
        "closeout_builder": checked_descriptor(closeout["closeout_tools"]["builder"]),
        "closeout_verifier": checked_descriptor(
            closeout["closeout_tools"]["independent_verifier"]
        ),
    }
    report = json.loads(paths["analysis"].read_text())
    audit = json.loads(paths["audit"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    manifest_verification = json.loads(paths["manifest_verification"].read_text())
    scheduler_evidence = json.loads(paths["scheduler"].read_text())
    source_sync = json.loads(paths["source_sync"].read_text())
    adapter = json.loads(paths["adapter_evidence"].read_text())
    data_env = json.loads(paths["data_environment_audit"].read_text())
    data_env_verification = json.loads(
        paths["data_environment_verification"].read_text()
    )
    protocol = json.loads(PROTOCOL.read_text())
    report_result = report.get("development_scientific_result", {})
    checks = report_result.get("checks", {})
    failed = sorted(key for key, value in checks.items() if value is not True)
    passed = sorted(key for key, value in checks.items() if value is True)
    relative = independent_relative_failures(report_result, protocol["analysis_gates"])
    completion = independent_completion_failures(
        report_result, protocol["analysis_gates"]
    )
    if (
        sha256(paths["analysis_source"]) != EXPECTED_ANALYSIS_SOURCE_SHA256
        or not checks
        or all(value is True for value in checks.values()) is not gate_passed
        or bool(failed) is gate_passed
        or result.get("checks") != checks
        or result.get("failed_checks") != failed
        or result.get("passed_checks") != passed
        or result.get("relative_aulc_threshold_failures") != relative
        or result.get("completion_noninferiority_failures") != completion
        or report_result.get("screening_gate_passed") is not gate_passed
        or report_result.get("predesignated_candidate")
        != "h2oplus-v17-tail-service-reserve-thirty"
        or set(report_result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or report.get("formal_confirmation_eligible") is not gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or report.get("artifact_integrity", {}).get("new_v17_physical_results")
        != 135
        or report.get("parent_training_scientific_stability", {}).get("passed")
        is not True
        or report.get("design_reuse_disclosure", {}).get(
            "prior_v14_v15_v16_development_outcomes_used_for_v17_design"
        )
        is not True
        or adapter.get("passed") is not True
        or adapter.get("frozen_analyzer_exit_code") != (0 if gate_passed else 2)
        or adapter.get("analysis_output", {}).get("sha256")
        != sha256(paths["analysis"])
        or adapter.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
        or adapter.get("schema_version")
        != (
            "transit-h2oplus-v17-tail-service-reserve-analysis-block-label-"
            "adapter-evidence-v2"
        )
        or adapter.get("v17_v2_operational_identity_only") is not True
        or adapter.get("v17_v2_operationalization", {}).get(
            "scientific_matrix_sources_exactly_equal"
        )
        is not True
        or adapter.get("v17_v2_operationalization", {}).get(
            "v1_incident_results_reused"
        )
        is not False
        or audit.get("artifact_integrity", {}).get("result_count") != 135
        or manifest.get("file_records_aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest.get("artifact_integrity", {}).get("result_count") != 135
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("manifest", {}).get("sha256")
        != sha256(paths["manifest"])
        or manifest_verification.get("independent_artifact_audit", {}).get(
            "result_count"
        )
        != 135
        or manifest_verification.get("independent_artifact_audit", {}).get(
            "source_file_count"
        )
        != 53
        or manifest_verification.get("independent_artifact_audit", {}).get(
            "unique_parent_checkpoints_rehashed"
        )
        != 5
        or manifest_verification.get("scheduler_frontend_completion_verified")
        is not False
        or manifest_verification.get("scheduler_terminal_evidence_verified")
        is not True
        or manifest_verification.get("scheduler_classifier_false_negative_verified")
        is not True
        or manifest_verification.get("token_bound_backend_exit0_verified") is not True
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or scheduler_evidence.get("task", {}).get("task_id") != "t83785"
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "observed_status"
        )
        != "failed"
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "completion_verified_from_frontend_status"
        )
        is not False
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "administrative_classifier_false_negative"
        )
        is not True
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "backend_and_host_evidence_override_frontend_classifier"
        )
        is not True
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "classifier_reason"
        )
        != EXPECTED_CLASSIFIER_REASON
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "lifecycle_log_nonempty"
        )
        is not True
        or scheduler_evidence.get("scheduler_frontend_completion", {}).get(
            "scientific_result_status_inferred_from_scheduler_frontend_only"
        )
        is not False
        or source_sync.get("passed") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("partition", {}).get("file_count") != 281
        or source_sync.get("formal_held_out_outcomes_read") is not False
        or sha256(paths["data_environment_audit"])
        != EXPECTED_DATA_ENV_AUDIT_SHA256
        or sha256(paths["data_environment_verification"])
        != EXPECTED_DATA_ENV_VERIFICATION_SHA256
        or data_env.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_env.get("conclusions", {}).get("algorithm_convergence_failure_proven")
        is not False
        or data_env.get("formal_held_out_outcomes_read") is not False
        or data_env.get("v17_incident_numeric_outcomes_inspected") is not False
        or data_env.get("local_training_or_sumo_executed") is not False
        or data_env_verification.get("passed") is not True
        or data_env_verification.get("formal_held_out_outcomes_read") is not False
        or data_env_verification.get("local_training_or_sumo_executed") is not False
        or risks.get("highest_priority_factors")
        != data_env.get("conclusions", {}).get("highest_priority_factors")
        or risks.get("risk_ids")
        != [row.get("id") for row in data_env.get("risk_register", [])]
    ):
        raise ValueError("independent v17-v2 closeout identity or gate check failed")
    return {
        "schema_version": (
            "transit-h2oplus-v17-tail-service-reserve-evaluation-closeout-"
            "verification-v2"
        ),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": {
            "path": str(CLOSEOUT.relative_to(REPO)),
            "sha256": sha256(CLOSEOUT),
            "size_bytes": CLOSEOUT.stat().st_size,
        },
        "artifact_integrity_verified": True,
        "physical_evaluation_results_verified": 135,
        "physical_behaviors_verified": 15,
        "unique_parent_checkpoints_rehashed": 5,
        "parent_training_scientific_stability_passed": True,
        "parent_training_scientific_stability_verified_separately": True,
        "scheduler_terminal_evidence_verified": True,
        "scheduler_frontend_completion_verified": False,
        "scheduler_classifier_false_negative_verified": True,
        "data_environment_material_factors_verified": True,
        "data_environment_risks_verified_separately": True,
        "v17_v2_design_changed_after_data_environment_audit": False,
        "v17_v1_incident_results_reused": False,
        "v17_v1_incident_numeric_outcomes_inspected": False,
        "development_screening_gate_passed": gate_passed,
        "development_screening_gate_value_verified": True,
        "failed_checks": failed,
        "relative_aulc_threshold_failures": relative,
        "completion_noninferiority_failures": completion,
        "prior_v14_v15_v16_development_outcomes_used_for_v17_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": gate_passed,
        "formal_confirmation_eligibility_does_not_authorize_submission": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-closeout-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = args.expected_closeout_sha256
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid expected v17 closeout SHA256")
    payload = verify(value)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v17 closeout verification: {OUTPUT}")
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
