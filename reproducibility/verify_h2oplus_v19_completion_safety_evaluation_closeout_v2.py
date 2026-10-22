#!/usr/bin/env python3
"""Independently verify the v19-v2 development evaluation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v19_completion_safety_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v2.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v2.json"
EXPECTED_CLASSIFIER_REASON = (
    "never entered training: lifetime 10947s but peak_vram=0 and no training "
    "markers in log tail (likely OOM/silent-kill mid-init)"
)
EXPECTED_MATRIX_SHA256 = [
    "09cee7c1c616a8b3740a854635716da0f74fff92e4b01d9968f677823de40683",
    "5830e5595a350a64df94b88a99f87a4e684dac183d8278a39a51aa5adedf38b0",
    "d06e017570c69cb5c432401a111ac680c1a7b38bce91b12ea95c65f5710be82c",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def checked_descriptor(item: dict) -> Path:
    path = Path(item.get("path", ""))
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if (
        not path.is_file()
        or REPO not in path.parents
        or path.stat().st_size != item.get("bytes")
        or sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"v19-v2 closeout descriptor failed: {item}")
    return path


def verify(expected_closeout_sha256: str) -> dict:
    if sha256(CLOSEOUT) != expected_closeout_sha256:
        raise ValueError("v19-v2 closeout identity drifted")
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    scheduler = closeout.get("scheduler_closeout", {})
    correction = closeout.get("predeployment_correction_isolation", {})
    execution = closeout.get("analysis_execution", {})
    scientific = closeout.get("development_scientific_result", {})
    risks = closeout.get("data_environment_validity_risks", {})
    checks = scientific.get("checks", {})
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-evaluation-closeout-v2"
        or closeout.get("status")
        != "closed-development-paired-mechanism-screen-passed"
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_result_count") != 135
        or artifact.get("physical_behavior_count") != 15
        or artifact.get("fresh_environment_cell_count") != 9
        or artifact.get("matrix_manifest_count") != 3
        or [row.get("sha256") for row in artifact.get("matrix_manifests", [])]
        != EXPECTED_MATRIX_SHA256
        or artifact.get("file_count") != 316
        or scheduler.get("passed") is not True
        or scheduler.get("authorized_scientific_task") != "t83842"
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("scheduler_frontend_observed_status") != "failed"
        or scheduler.get("scheduler_frontend_completion_verified") is not False
        or scheduler.get("administrative_classifier_false_negative") is not True
        or scheduler.get("classifier_reason") != EXPECTED_CLASSIFIER_REASON
        or scheduler.get("backend_and_host_evidence_override_frontend_classifier")
        is not True
        or scheduler.get("scheduler_generated_retry_chain", {}).get("task_ids")
        != ["t83858", "t83859", "t83860"]
        or scheduler.get("retry_tasks_contributed_scientific_results") is not False
        or closeout.get("parent_training_stability", {}).get("passed") is not True
        or risks.get("material_factors_present") is not True
        or risks.get("fresh_development_environment_cell_count") != 9
        or risks.get("algorithm_convergence_failure_proven") is not False
        or correction.get("v19_v1_completed_scientific_result_count") != 0
        or correction.get("v19_v1_scientific_numeric_outcomes_read") is not False
        or correction.get("v19_v1_results_reused") is not False
        or execution.get("frozen_analyzer_modified") is not False
        or execution.get("frozen_analyzer_exit_code") != 0
        or execution.get("source_freeze_schema_adapter_verified") is not True
        or execution.get("result_files_or_manifests_modified") is not False
        or execution.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or scientific.get("candidate")
        != "h2oplus-v19-completion-safety-quarter-load-sixty"
        or scientific.get("paired_baseline")
        != "h2oplus-v19-completion-safety-baseline"
        or scientific.get("diagnostic")
        != "h2oplus-v19-completion-safety-quarter-load-fifty"
        or scientific.get("diagnostic_cannot_replace_or_reinterpret_candidate")
        is not True
        or not checks
        or not all(checks.values())
        or scientific.get("failed_checks") != []
        or scientific.get("paired_mechanism_screen_passed") is not True
        or closeout.get("eligible_for_full_development_control_matrix") is not True
        or closeout.get("full_development_control_matrix_authorized_or_submitted")
        is not False
        or closeout.get(
            "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_v19_design"
        )
        is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_formal_held_out_effect_claim") is not False
        or closeout.get("formal_confirmation_eligible") is not False
        or closeout.get("formal_confirmation_authorized_or_submitted") is not False
    ):
        raise ValueError("independent v19-v2 closeout scope check failed")

    paths = {
        "audit": checked_descriptor(artifact["evaluation_audit"]),
        "manifest": checked_descriptor(artifact["evaluation_file_manifest"]),
        "manifest_verification": checked_descriptor(
            artifact["evaluation_file_manifest_verification"]
        ),
        "scheduler": checked_descriptor(scheduler["scheduler_evidence"]),
        "source_sync": checked_descriptor(scheduler["source_sync"]),
        "analysis": checked_descriptor(scientific["analysis"]),
        "analyzer": checked_descriptor(execution["frozen_analyzer"]),
        "execution": checked_descriptor(execution["execution_evidence"]),
        "correction": checked_descriptor(correction["correction"]),
        "data_environment_audit": checked_descriptor(risks["audit"]),
        "data_environment_verification": checked_descriptor(
            risks["independent_verification"]
        ),
        "builder": checked_descriptor(closeout["closeout_tools"]["builder"]),
        "verifier": checked_descriptor(
            closeout["closeout_tools"]["independent_verifier"]
        ),
    }
    audit = json.loads(paths["audit"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    manifest_verification = json.loads(paths["manifest_verification"].read_text())
    scheduler_evidence = json.loads(paths["scheduler"].read_text())
    source_sync = json.loads(paths["source_sync"].read_text())
    analysis = json.loads(paths["analysis"].read_text())
    execution_evidence = json.loads(paths["execution"].read_text())
    correction_evidence = json.loads(paths["correction"].read_text())
    data_environment = json.loads(paths["data_environment_audit"].read_text())
    data_verification = json.loads(
        paths["data_environment_verification"].read_text()
    )
    report = analysis.get("development_scientific_result", {})
    gates = report.get("gate_specification", {})
    mean = report.get("candidate_paired_mean", {})
    by_scale = report.get("candidate_paired_by_demand_scale", {})
    by_seed = report.get("candidate_paired_by_training_seed", {})
    recomputed = {
        "all_135_results_valid": (
            audit.get("passed") is True
            and audit.get("artifact_integrity", {}).get("physical_result_count") == 135
        ),
        "all_three_matrix_manifests_complete": (
            audit.get("artifact_integrity", {}).get("matrix_manifest_count") == 3
            and all(row.get("jobs") == 45 for row in artifact["matrix_manifests"])
        ),
        "candidate_mean_completion_gain": (
            float(mean["completion_gain"])
            >= float(
                gates[
                    "candidate_mean_completion_gain_over_paired_baseline_at_least"
                ]
            )
        ),
        "candidate_mean_primary_noninferiority": (
            float(mean["primary_increase_seconds"])
            <= float(
                gates[
                    "candidate_mean_primary_increase_over_paired_baseline_at_most_seconds"
                ]
            )
        ),
        "candidate_completion_gain_positive_all_demand_scales": (
            len(by_scale) == 3
            and all(float(row["completion_gain"]) > 0.0 for row in by_scale.values())
        ),
        "candidate_completion_gain_positive_at_least_four_training_seeds": (
            sum(float(row["completion_gain"]) > 0.0 for row in by_seed.values())
            >= int(
                gates[
                    "candidate_positive_completion_gain_training_seed_count_at_least"
                ]
            )
        ),
        "parent_artifact_integrity_passed": (
            analysis.get("artifact_integrity", {}).get("passed") is True
        ),
        "parent_training_stability_passed": (
            analysis.get("parent_training_stability", {}).get("passed") is True
        ),
    }
    if (
        checks != recomputed
        or report.get("checks") != recomputed
        or not all(recomputed.values())
        or manifest.get("passed") is not True
        or manifest.get("file_count") != 316
        or manifest.get("aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("independently_recomputed_file_count") != 316
        or scheduler_evidence.get("task", {}).get("task_id") != "t83842"
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("v19_v1_results_reused") is not False
        or execution_evidence.get("passed") is not True
        or execution_evidence.get("analysis_output", {}).get("sha256")
        != sha256(paths["analysis"])
        or execution_evidence.get("controls", {}).get(
            "v1_v2_scientific_matrix_sources_methods_protocol_and_analyzer_equal"
        )
        is not True
        or correction_evidence.get("execution_boundary", {}).get(
            "completed_scientific_result_count"
        )
        != 0
        or data_environment.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("independent v19-v2 identity or gate check failed")

    return {
        "schema_version": "transit-h2oplus-v19-completion-safety-evaluation-closeout-verification-v2",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": {
            "path": str(CLOSEOUT.relative_to(REPO)),
            "sha256": sha256(CLOSEOUT),
            "bytes": CLOSEOUT.stat().st_size,
        },
        "artifact_integrity_verified": True,
        "physical_result_count_verified": 135,
        "physical_behavior_count_verified": 15,
        "fresh_development_environment_cell_count_verified": 9,
        "scheduler_token_bound_backend_exit0_verified": True,
        "scheduler_frontend_classifier_false_negative_verified": True,
        "scheduler_generated_retry_chain_verified": ["t83858", "t83859", "t83860"],
        "scheduler_retry_results_counted": False,
        "v19_v1_results_reused": False,
        "parent_training_stability_verified": True,
        "material_data_environment_validity_factors_verified": True,
        "development_paired_mechanism_screen_passed": True,
        "failed_checks": [],
        "candidate_paired_mean": mean,
        "eligible_for_full_development_control_matrix": True,
        "full_development_control_matrix_authorized_or_submitted": False,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_v19_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized_or_submitted": False,
        "passed": True,
        "passed_scope": "independent verified development closeout",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-closeout-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = args.expected_closeout_sha256
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("invalid expected v19-v2 closeout SHA256")
    if OUTPUT.exists():
        raise FileExistsError(
            f"refusing to overwrite v19-v2 closeout verification: {OUTPUT}"
        )
    payload = verify(expected)
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
