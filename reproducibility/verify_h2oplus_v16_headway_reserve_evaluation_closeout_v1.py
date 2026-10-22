#!/usr/bin/env python3
"""Independently verify the final v16 headway-reserve evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v16_headway_reserve_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v1.json"
PROTOCOL = REPO / "reproducibility/h2oplus_v16_headway_reserve_protocol_v1.json"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "e46c04ffcab861ca045dafdacd8f7257da026213a82edf82bd8761fe15636834"
)
EXPECTED_PROTOCOL_SHA256 = (
    "978faf1b8ddf51128b5fac570552e8388c3d882e39244c4b6f0ec7615e79a43b"
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
        raise ValueError(f"v16 closeout descriptor failed: {item}")
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
        raise ValueError("v16 closeout identity drifted")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v16 protocol identity drifted during closeout verification")
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    scheduler = closeout.get("scheduler_closeout", {})
    stability = closeout.get("parent_training_scientific_stability", {})
    result = closeout.get("development_scientific_result", {})
    execution = closeout.get("analysis_execution", {})
    gate_passed = result.get("screening_gate_passed")
    expected_status = (
        "closed-development-screen-passed"
        if gate_passed is True
        else "closed-development-screen-failed"
    )
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-closeout-v1"
        or closeout.get("status") != expected_status
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_evaluation_results") != 135
        or artifact.get("physical_behaviors") != 15
        or artifact.get("unique_parent_checkpoints_independently_rehashed") != 5
        or artifact.get("matrix_count") != 3
        or artifact.get("source_file_count") != 51
        or scheduler.get("passed") is not True
        or scheduler.get("task") != "t83704"
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("scheduler_frontend_false_negative_preserved") is not True
        or stability.get("passed") is not True
        or stability.get("registered_runs") != 15
        or stability.get("new_v16_training_runs") != 0
        or result.get("predesignated_candidate")
        != "h2oplus-v16-headway-reserve-three-quarter"
        or not isinstance(gate_passed, bool)
        or execution.get("frozen_analysis_source_modified") is not False
        or execution.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
        or execution.get("result_files_or_manifests_modified") is not False
        or execution.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or closeout.get("design_reuse_disclosure", {}).get(
            "prior_v14_and_v15_development_outcomes_used_for_v16_design"
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
        raise ValueError("independent v16 closeout scope check failed")

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
        != "h2oplus-v16-headway-reserve-three-quarter"
        or set(report_result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or report.get("formal_confirmation_eligible") is not gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or report.get("artifact_integrity", {}).get("new_v16_physical_results")
        != 135
        or report.get("parent_training_scientific_stability", {}).get("passed")
        is not True
        or adapter.get("passed") is not True
        or adapter.get("frozen_analyzer_exit_code") != (0 if gate_passed else 2)
        or adapter.get("analysis_output", {}).get("sha256")
        != sha256(paths["analysis"])
        or adapter.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
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
            "unique_parent_checkpoints_rehashed"
        )
        != 5
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or scheduler_evidence.get("task", {}).get("task_id") != "t83704"
        or source_sync.get("passed") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("independent v16 closeout identity or gate check failed")
    return {
        "schema_version": (
            "transit-h2oplus-v16-headway-reserve-evaluation-closeout-"
            "verification-v1"
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
        "development_screening_gate_passed": gate_passed,
        "development_screening_gate_value_verified": True,
        "failed_checks": failed,
        "relative_aulc_threshold_failures": relative,
        "completion_noninferiority_failures": completion,
        "prior_v14_and_v15_development_outcomes_used_for_v16_design": True,
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
        raise ValueError("invalid expected v16 closeout SHA256")
    payload = verify(value)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v16 closeout verification: {OUTPUT}")
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
