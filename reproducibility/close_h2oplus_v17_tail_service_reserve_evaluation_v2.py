#!/usr/bin/env python3
"""Create the final v17-v2 tail-service development-evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_results"
ANALYSIS = RESULTS / "tail_service_reserve_analysis_v2.json"
ADAPTER_EVIDENCE = RESULTS / "analysis_block_label_adapter_evidence_v2.json"
AUDIT = RESULTS / "evaluation_audit_v2.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v2.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v2.json"
SCHEDULER = RESULTS / "evidence/evaluation_v2/scheduler_evidence_v2.json"
SOURCE_SYNC = RESULTS / "evidence/evaluation_v2/source_sync_v2.json"
DATA_ENV_AUDIT = REPO / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json"
DATA_ENV_VERIFICATION = (
    REPO / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json"
)
PROTOCOL = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v17_tail_service_reserve.py"
ANALYSIS_WRAPPER = (
    REPO / "reproducibility/run_frozen_h2oplus_v17_tail_service_reserve_analysis_v2.py"
)
OUTPUT = RESULTS / "evaluation_closeout_v2.json"

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
EXPECTED_ALGORITHMS = EXPECTED_CONTROLS | {
    "h2oplus-v17-tail-service-reserve-twenty",
    "h2oplus-v17-tail-service-reserve-thirty",
    "h2oplus-v17-tail-service-reserve-forty",
}
EXPECTED_MAPPING = {
    "v12-staged-release-development-01": "sumo-seed-5101",
    "v12-staged-release-development-02": "sumo-seed-5102",
    "v12-staged-release-development-03": "sumo-seed-5103",
}
RELATIVE_THRESHOLDS = {
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


def relative_failures(result: dict, thresholds: dict) -> list[dict]:
    failures = []
    comparisons = result["control_comparisons"]
    for control, threshold_key in RELATIVE_THRESHOLDS.items():
        observed = float(comparisons[control]["relative_improvement"])
        required = float(thresholds[threshold_key])
        if observed < required:
            failures.append(
                {
                    "control": control,
                    "observed_relative_improvement": observed,
                    "required_at_least": required,
                    "shortfall": required - observed,
                }
            )
    return failures


def completion_failures(result: dict, thresholds: dict) -> list[dict]:
    margin = float(thresholds["completion_rate_noninferiority_margin"])
    failures = []
    for control, row in result["completion_aulc_against_controls"].items():
        delta = float(row["delta"])
        if delta < -margin:
            failures.append(
                {
                    "control": control,
                    "observed_delta": delta,
                    "minimum_allowed_delta": -margin,
                    "shortfall": -margin - delta,
                }
            )
    return failures


def build(expected: dict[str, str]) -> dict:
    for name, path in {
        "analysis": ANALYSIS,
        "adapter_evidence": ADAPTER_EVIDENCE,
        "audit": AUDIT,
        "manifest": MANIFEST,
        "manifest_verification": MANIFEST_VERIFICATION,
        "scheduler": SCHEDULER,
        "source_sync": SOURCE_SYNC,
        "analysis_wrapper": ANALYSIS_WRAPPER,
    }.items():
        actual = sha256(path)
        if actual != expected[name]:
            raise ValueError(f"v17 closeout input identity drifted for {name}: {actual}")
    if sha256(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256:
        raise ValueError("frozen v17 analysis source identity drifted")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen v17 protocol identity drifted")
    if sha256(DATA_ENV_AUDIT) != EXPECTED_DATA_ENV_AUDIT_SHA256:
        raise ValueError("v17 data/environment audit identity drifted")
    if sha256(DATA_ENV_VERIFICATION) != EXPECTED_DATA_ENV_VERIFICATION_SHA256:
        raise ValueError("v17 data/environment verification identity drifted")

    report = json.loads(ANALYSIS.read_text())
    adapter = json.loads(ADAPTER_EVIDENCE.read_text())
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    verification = json.loads(MANIFEST_VERIFICATION.read_text())
    scheduler = json.loads(SCHEDULER.read_text())
    source_sync = json.loads(SOURCE_SYNC.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    data_env = json.loads(DATA_ENV_AUDIT.read_text())
    data_env_verification = json.loads(DATA_ENV_VERIFICATION.read_text())
    result = report.get("development_scientific_result", {})
    checks = result.get("checks", {})
    gate_passed = result.get("screening_gate_passed")
    calculated_gate = bool(checks) and all(value is True for value in checks.values())
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    thresholds = protocol.get("analysis_gates", {})
    relative = relative_failures(result, thresholds)
    completion = completion_failures(result, thresholds)
    v17_matrix = report.get("evidence", {}).get("v17_matrix", [])
    diagnostics = report.get("diagnostic_results_not_used_as_selection_gates", {})
    independent = verification.get("independent_artifact_audit", {})
    if (
        report.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("new_v17_physical_results") != 135
        or report.get("artifact_integrity", {}).get("reused_frozen_v14_physical_results")
        != 630
        or report.get("artifact_integrity", {}).get("v17_physical_behaviors") != 15
        or report.get("parent_training_scientific_stability", {}).get("passed")
        is not True
        or report.get("parent_training_scientific_stability", {}).get(
            "new_training_runs"
        )
        != 0
        or result.get("predesignated_candidate")
        != "h2oplus-v17-tail-service-reserve-thirty"
        or gate_passed is not calculated_gate
        or set(result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or set(result.get("completion_aulc_against_controls", {})) != EXPECTED_CONTROLS
        or set(result.get("mean_primary_at_25000", {})) != EXPECTED_ALGORITHMS
        or set(diagnostics.get("algorithms", []))
        != {
            "h2oplus-v17-tail-service-reserve-twenty",
            "h2oplus-v17-tail-service-reserve-forty",
        }
        or len(v17_matrix) != 3
        or {row.get("block") for row in v17_matrix} != set(EXPECTED_MAPPING.values())
        or any(
            row.get("jobs") != 45 or row.get("source_file_count") != 53
            for row in v17_matrix
        )
        or report.get("formal_confirmation_eligible") is not gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or report.get("design_reuse_disclosure", {}).get(
            "prior_v14_v15_v16_development_outcomes_used_for_v17_design"
        )
        is not True
        or adapter.get("passed") is not True
        or adapter.get("schema_version")
        != (
            "transit-h2oplus-v17-tail-service-reserve-analysis-block-label-"
            "adapter-evidence-v2"
        )
        or adapter.get("frozen_analyzer", {}).get("sha256")
        != EXPECTED_ANALYSIS_SOURCE_SHA256
        or adapter.get("frozen_analyzer_exit_code") != (0 if gate_passed else 2)
        or adapter.get("analysis_output", {}).get("sha256") != sha256(ANALYSIS)
        or adapter.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
        or adapter.get("result_files_or_manifests_modified") is not False
        or adapter.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or adapter.get("v17_v2_operational_identity_only") is not True
        or adapter.get("v17_v2_operationalization", {}).get(
            "scientific_matrix_sources_exactly_equal"
        )
        is not True
        or adapter.get("v17_v2_operationalization", {}).get(
            "v1_incident_results_reused"
        )
        is not False
        or audit.get("passed") is not True
        or audit.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-evaluation-audit-v2"
        or audit.get("artifact_integrity", {}).get("result_count") != 135
        or audit.get("artifact_integrity", {}).get("unique_parent_checkpoint_count") != 5
        or manifest.get("artifact_integrity", {}).get("result_count") != 135
        or verification.get("passed") is not True
        or independent.get("result_count") != 135
        or independent.get("physical_behaviors_verified") != 15
        or independent.get("source_file_count") != 53
        or independent.get("unique_parent_checkpoints_rehashed") != 5
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or verification.get("scheduler_frontend_completion_verified") is not False
        or verification.get("scheduler_terminal_evidence_verified") is not True
        or verification.get("scheduler_classifier_false_negative_verified") is not True
        or verification.get("token_bound_backend_exit0_verified") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("task", {}).get("task_id") != "t83785"
        or scheduler.get("scheduler_frontend_completion", {}).get("observed_status")
        != "failed"
        or scheduler.get("scheduler_frontend_completion", {}).get(
            "completion_verified_from_frontend_status"
        )
        is not False
        or scheduler.get("scheduler_frontend_completion", {}).get(
            "administrative_classifier_false_negative"
        )
        is not True
        or scheduler.get("scheduler_frontend_completion", {}).get(
            "backend_and_host_evidence_override_frontend_classifier"
        )
        is not True
        or scheduler.get("scheduler_frontend_completion", {}).get(
            "classifier_reason"
        )
        != EXPECTED_CLASSIFIER_REASON
        or scheduler.get("scheduler_frontend_completion", {}).get(
            "lifecycle_log_nonempty"
        )
        is not True
        or scheduler.get("scheduler_frontend_completion", {}).get(
            "scientific_result_status_inferred_from_scheduler_frontend_only"
        )
        is not False
        or source_sync.get("passed") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("partition", {}).get("file_count") != 281
        or source_sync.get("local_sumo_executed") is not False
        or data_env.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_env.get("conclusions", {}).get("algorithm_convergence_failure_proven")
        is not False
        or data_env.get("v17_incident_numeric_outcomes_inspected") is not False
        or data_env.get("formal_held_out_outcomes_read") is not False
        or data_env.get("local_training_or_sumo_executed") is not False
        or data_env_verification.get("passed") is not True
        or data_env_verification.get("formal_held_out_outcomes_read") is not False
        or data_env_verification.get("local_training_or_sumo_executed") is not False
        or audit.get("data_environment_validity_risks", {}).get(
            "material_factors_present"
        )
        is not True
        or audit.get("data_environment_validity_risks", {}).get(
            "v17_v2_design_changed_after_audit"
        )
        is not False
    ):
        raise ValueError("v17-v2 final closeout contract failed")
    if bool(failed_checks) is gate_passed:
        raise ValueError("v17 development gate/check consistency failed")
    if checks.get("relative_aulc_improvement_over_all_controls") is (not bool(relative)):
        pass
    else:
        raise ValueError("v17 relative-improvement failure detail is inconsistent")
    if checks.get("completion_rate_noninferiority_against_all_controls") is (not bool(completion)):
        pass
    else:
        raise ValueError("v17 completion failure detail is inconsistent")

    return {
        "schema_version": "transit-h2oplus-v17-tail-service-reserve-evaluation-closeout-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "closed-development-screen-passed"
            if gate_passed
            else "closed-development-screen-failed"
        ),
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_results": 135,
            "physical_behaviors": 15,
            "unique_parent_checkpoints_independently_rehashed": 5,
            "matrix_count": 3,
            "source_file_count": 53,
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
            "task": "t83785",
            "token_bound_backend_exit0": True,
            "scheduler_terminal_evidence_verified": True,
            "scheduler_frontend_completion_verified": False,
            "scheduler_frontend_observed_status": "failed",
            "administrative_classifier_false_negative": True,
            "backend_and_host_evidence_override_frontend_classifier": True,
            "classifier_reason": EXPECTED_CLASSIFIER_REASON,
            "scheduler_evidence": descriptor(SCHEDULER),
            "source_sync": descriptor(SOURCE_SYNC),
        },
        "parent_training_scientific_stability": {
            "passed": True,
            "registered_runs": 15,
            "new_v17_training_runs": 0,
            "separate_from_v17_artifact_integrity": True,
        },
        "development_scientific_result": {
            "analysis": descriptor(ANALYSIS),
            "analysis_source": descriptor(ANALYSIS_SOURCE),
            "predesignated_candidate": "h2oplus-v17-tail-service-reserve-thirty",
            "screening_gate_passed": gate_passed,
            "checks": checks,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "relative_aulc_threshold_failures": relative,
            "completion_noninferiority_failures": completion,
            "control_comparisons": result["control_comparisons"],
            "mean_primary_at_25000": result["mean_primary_at_25000"],
            "completion_aulc_against_controls": result[
                "completion_aulc_against_controls"
            ],
            "demand_scale_aulc_against_sac": result[
                "demand_scale_aulc_against_sac"
            ],
            "diagnostic_results_not_used_as_selection_gates": diagnostics,
        },
        "analysis_execution": {
            "frozen_analysis_source_modified": False,
            "block_label_adapter": descriptor(ANALYSIS_WRAPPER),
            "block_label_adapter_evidence": descriptor(ADAPTER_EVIDENCE),
            "registered_to_matrix_block_mapping": EXPECTED_MAPPING,
            "v17_v2_operational_identity_only": True,
            "scientific_matrix_sources_exactly_equal_to_v17_v1": True,
            "v17_v1_incident_results_reused": False,
            "v17_v1_incident_numeric_outcomes_inspected": False,
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
        },
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "audit": descriptor(DATA_ENV_AUDIT),
            "independent_verification": descriptor(DATA_ENV_VERIFICATION),
            "highest_priority_factors": data_env["conclusions"][
                "highest_priority_factors"
            ],
            "risk_ids": [row["id"] for row in data_env["risk_register"]],
            "algorithm_convergence_failure_proven": False,
            "separate_from_artifact_integrity_parent_training_and_scientific_result": True,
            "v17_v2_design_changed_after_audit": False,
        },
        "design_reuse_disclosure": {
            "prior_v14_v15_v16_development_outcomes_used_for_v17_design": True,
            "v17_outcomes_used_before_freeze": False,
            "independent_confirmation": False,
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO
                / "reproducibility/verify_h2oplus_v17_tail_service_reserve_evaluation_closeout_v2.py"
            ),
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": gate_passed,
        "formal_confirmation_eligibility_does_not_authorize_submission": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "analysis",
        "adapter_evidence",
        "analysis_wrapper",
        "audit",
        "manifest",
        "manifest_verification",
        "scheduler",
        "source_sync",
    ):
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = (
        "analysis",
        "adapter_evidence",
        "analysis_wrapper",
        "audit",
        "manifest",
        "manifest_verification",
        "scheduler",
        "source_sync",
    )
    expected = {name: getattr(args, "expected_" + name + "_sha256") for name in names}
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in expected.values()
    ):
        raise ValueError("invalid expected v17 closeout SHA256")
    payload = build(expected)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v17 closeout: {OUTPUT}")
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
