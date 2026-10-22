#!/usr/bin/env python3
"""Create the final v16 headway-reserve development-evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v16_headway_reserve_results"
ANALYSIS = RESULTS / "headway_reserve_analysis_v1.json"
ADAPTER_EVIDENCE = RESULTS / "analysis_block_label_adapter_evidence_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
SOURCE_SYNC = RESULTS / "evidence/evaluation_v1/source_sync_v1.json"
PROTOCOL = REPO / "reproducibility/h2oplus_v16_headway_reserve_protocol_v1.json"
ANALYSIS_SOURCE = REPO / "reproducibility/analyze_h2oplus_v16_headway_reserve.py"
ANALYSIS_WRAPPER = (
    REPO / "reproducibility/run_frozen_h2oplus_v16_headway_reserve_analysis_v1.py"
)
OUTPUT = RESULTS / "evaluation_closeout_v1.json"

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
EXPECTED_ALGORITHMS = EXPECTED_CONTROLS | {
    "h2oplus-v16-headway-reserve-half",
    "h2oplus-v16-headway-reserve-three-quarter",
    "h2oplus-v16-headway-reserve-full",
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
            raise ValueError(f"v16 closeout input identity drifted for {name}: {actual}")
    if sha256(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256:
        raise ValueError("frozen v16 analysis source identity drifted")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen v16 protocol identity drifted")

    report = json.loads(ANALYSIS.read_text())
    adapter = json.loads(ADAPTER_EVIDENCE.read_text())
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    verification = json.loads(MANIFEST_VERIFICATION.read_text())
    scheduler = json.loads(SCHEDULER.read_text())
    source_sync = json.loads(SOURCE_SYNC.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    result = report.get("development_scientific_result", {})
    checks = result.get("checks", {})
    gate_passed = result.get("screening_gate_passed")
    calculated_gate = bool(checks) and all(value is True for value in checks.values())
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    passed_checks = sorted(key for key, value in checks.items() if value is True)
    thresholds = protocol.get("analysis_gates", {})
    relative = relative_failures(result, thresholds)
    completion = completion_failures(result, thresholds)
    v16_matrix = report.get("evidence", {}).get("v16_matrix", [])
    diagnostics = report.get("diagnostic_results_not_used_as_selection_gates", {})
    independent = verification.get("independent_artifact_audit", {})
    if (
        report.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-analysis-v1"
        or report.get("artifact_integrity", {}).get("passed") is not True
        or report.get("artifact_integrity", {}).get("new_v16_physical_results") != 135
        or report.get("artifact_integrity", {}).get("reused_frozen_v14_physical_results")
        != 630
        or report.get("artifact_integrity", {}).get("v16_physical_behaviors") != 15
        or report.get("parent_training_scientific_stability", {}).get("passed")
        is not True
        or report.get("parent_training_scientific_stability", {}).get(
            "new_training_runs"
        )
        != 0
        or result.get("predesignated_candidate")
        != "h2oplus-v16-headway-reserve-three-quarter"
        or gate_passed is not calculated_gate
        or set(result.get("control_comparisons", {})) != EXPECTED_CONTROLS
        or set(result.get("completion_aulc_against_controls", {})) != EXPECTED_CONTROLS
        or set(result.get("mean_primary_at_25000", {})) != EXPECTED_ALGORITHMS
        or set(diagnostics.get("algorithms", []))
        != {
            "h2oplus-v16-headway-reserve-half",
            "h2oplus-v16-headway-reserve-full",
        }
        or len(v16_matrix) != 3
        or {row.get("block") for row in v16_matrix} != set(EXPECTED_MAPPING.values())
        or any(
            row.get("jobs") != 45 or row.get("source_file_count") != 51
            for row in v16_matrix
        )
        or report.get("formal_confirmation_eligible") is not gate_passed
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or report.get("design_reuse_disclosure", {}).get(
            "prior_v14_and_v15_development_outcomes_used_for_v16_design"
        )
        is not True
        or adapter.get("passed") is not True
        or adapter.get("frozen_analyzer", {}).get("sha256")
        != EXPECTED_ANALYSIS_SOURCE_SHA256
        or adapter.get("frozen_analyzer_exit_code") != (0 if gate_passed else 2)
        or adapter.get("analysis_output", {}).get("sha256") != sha256(ANALYSIS)
        or adapter.get("registered_to_matrix_block_mapping") != EXPECTED_MAPPING
        or adapter.get("result_files_or_manifests_modified") is not False
        or adapter.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 135
        or audit.get("artifact_integrity", {}).get("unique_parent_checkpoint_count") != 5
        or manifest.get("artifact_integrity", {}).get("result_count") != 135
        or verification.get("passed") is not True
        or independent.get("result_count") != 135
        or independent.get("physical_behaviors_verified") != 15
        or independent.get("unique_parent_checkpoints_rehashed") != 5
        or verification.get("manifest", {}).get("sha256") != sha256(MANIFEST)
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("task", {}).get("task_id") != "t83704"
        or source_sync.get("passed") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("local_sumo_executed") is not False
    ):
        raise ValueError("v16 final closeout contract failed")
    if bool(failed_checks) is gate_passed:
        raise ValueError("v16 development gate/check consistency failed")
    if checks.get("relative_aulc_improvement_over_all_controls") is (not bool(relative)):
        pass
    else:
        raise ValueError("v16 relative-improvement failure detail is inconsistent")
    if checks.get("completion_rate_noninferiority_against_all_controls") is (not bool(completion)):
        pass
    else:
        raise ValueError("v16 completion failure detail is inconsistent")

    return {
        "schema_version": "transit-h2oplus-v16-headway-reserve-evaluation-closeout-v1",
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
            "source_file_count": 51,
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
            "task": "t83704",
            "token_bound_backend_exit0": True,
            "scheduler_frontend_false_negative_preserved": True,
            "scheduler_evidence": descriptor(SCHEDULER),
            "source_sync": descriptor(SOURCE_SYNC),
        },
        "parent_training_scientific_stability": {
            "passed": True,
            "registered_runs": 15,
            "new_v16_training_runs": 0,
            "separate_from_v16_artifact_integrity": True,
        },
        "development_scientific_result": {
            "analysis": descriptor(ANALYSIS),
            "analysis_source": descriptor(ANALYSIS_SOURCE),
            "predesignated_candidate": "h2oplus-v16-headway-reserve-three-quarter",
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
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
        },
        "design_reuse_disclosure": {
            "prior_v14_and_v15_development_outcomes_used_for_v16_design": True,
            "v16_outcomes_used_before_freeze": False,
            "independent_confirmation": False,
        },
        "closeout_tools": {
            "builder": descriptor(Path(__file__).resolve()),
            "independent_verifier": descriptor(
                REPO
                / "reproducibility/verify_h2oplus_v16_headway_reserve_evaluation_closeout_v1.py"
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
        raise ValueError("invalid expected v16 closeout SHA256")
    payload = build(expected)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v16 closeout: {OUTPUT}")
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
