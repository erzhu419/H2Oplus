#!/usr/bin/env python3
"""Build the verified v20 full development control-matrix closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_full_control_results"
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
PATHS = {
    "audit": RESULTS / "evaluation_audit_v1.json",
    "manifest": RESULTS / "evaluation_file_manifest_v1.json",
    "manifest_verification": RESULTS / "evaluation_file_manifest_verification_v1.json",
    "scheduler": RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json",
    "source_sync": RESULTS / "evidence/evaluation_v1/source_sync_v1.json",
    "analysis": RESULTS / "full_control_analysis_v1.json",
    "analysis_execution": RESULTS / "analysis_execution_evidence_v1.json",
    "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
}
EXPECTED = {
    "audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "manifest": "ea41597ee6b05b136ba16d5efeaaa50b9368542419913e18c79cb9ebc7bcb51b",
    "manifest_verification": "6f50f7de8b87519349d48805e2ac7f98a6ac7d9ed2df989378fd2448a95c5cb0",
    "scheduler": "63b5e39a9fa3edc2483f06ecb7685f41e3f24cd442fbeac899f63f2949d841dd",
    "source_sync": "02a63f240717b20d4e03d9fd6bd838ff13778f3d95340d8b8fb2ea3b83caa736",
    "analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "analysis_execution": "16b024a975a7a2df877c95215bd8e6d38fbabdcd126048b6953f37beb97b3beb",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
}
EXPECTED_CHECKS = {
    "all_405_results_valid",
    "all_three_135_job_matrix_manifests_complete",
    "candidate_completion_gain_positive_all_demand_scales",
    "candidate_completion_gain_positive_at_least_four_training_seeds",
    "candidate_completion_noninferior_to_every_comparison_control",
    "candidate_mean_completion_gain_over_direct_parent",
    "candidate_mean_primary_lower_than_every_comparison_control",
    "candidate_mean_primary_noninferiority_to_direct_parent",
    "candidate_primary_robust_at_all_demand_scales_vs_sac",
    "candidate_primary_seed_wins_against_every_comparison_control",
    "parent_artifact_integrity_passed",
    "parent_training_stability_passed",
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
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def checked_json(name: str) -> dict:
    path = PATHS[name].resolve(strict=True)
    if sha256(path) != EXPECTED[name]:
        raise ValueError(f"v20 closeout input drifted: {name}")
    return json.loads(path.read_text())


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v20 closeout: {OUTPUT}")
    audit = checked_json("audit")
    manifest = checked_json("manifest")
    manifest_verification = checked_json("manifest_verification")
    scheduler = checked_json("scheduler")
    source_sync = checked_json("source_sync")
    analysis = checked_json("analysis")
    execution = checked_json("analysis_execution")
    data_audit = checked_json("data_environment_audit")
    data_verification = checked_json("data_environment_verification")
    artifact = audit.get("artifact_integrity", {})
    scientific = analysis.get("development_scientific_result", {})
    checks = scientific.get("checks", {})
    if (
        audit.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_result_count") != 405
        or artifact.get("physical_behavior_count") != 45
        or artifact.get("fresh_environment_cell_count") != 9
        or artifact.get("matrix_manifest_count") != 3
        or manifest.get("passed") is not True
        or manifest.get("physical_result_count") != 405
        or manifest.get("file_count") != 847
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("physical_result_count") != 405
        or scheduler.get("passed") is not True
        or scheduler.get("task", {}).get("task_id") != "t83955"
        or scheduler.get("task", {}).get("status") != "done"
        or scheduler.get("token_bound_backend_exit0") is not True
        or source_sync.get("passed") is not True
        or source_sync.get("exact_remote_local_match") is not True
        or source_sync.get("physical_result_count") != 405
        or execution.get("passed") is not True
        or execution.get("frozen_analyzer_exit_code") != 0
        or execution.get("no_remote_sync_performed") is not True
        or execution.get("checkpoint_files_synchronized") is not False
        or set(checks) != EXPECTED_CHECKS
        or not all(checks.values())
        or scientific.get("full_development_control_matrix_passed") is not True
        or analysis.get("formal_confirmation_eligible") is not True
        or analysis.get("formal_confirmation_authorized") is not False
        or analysis.get("formal_confirmation_submitted") is not False
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("v20 closeout prerequisites failed")
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-evaluation-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed-development-full-control-matrix-passed",
        "study_stage": "development",
        "matrix_type": "full-development-control-matrix",
        "passed": True,
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 405,
            "physical_behavior_count": 45,
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
            "authorized_scientific_task": "t83955",
            "scheduler_status": "done",
            "token_bound_backend_exit0": True,
            "scheduler_evidence": descriptor(PATHS["scheduler"]),
            "source_sync": descriptor(PATHS["source_sync"]),
            "exact_remote_local_match": True,
        },
        "parent_training_stability": analysis["parent_training_stability"],
        "data_environment_validity_risks": analysis[
            "data_environment_validity_risks"
        ],
        "analysis_execution": {
            "frozen_analyzer": execution["frozen_analyzer"],
            "frozen_analyzer_modified": False,
            "frozen_analyzer_exit_code": 0,
            "runner": descriptor(
                REPRO / "run_frozen_h2oplus_v20_full_control_analysis_v1.py"
            ),
            "execution_evidence": descriptor(PATHS["analysis_execution"]),
            "registered_to_matrix_block_mapping": execution[
                "registered_to_matrix_block_mapping"
            ],
            "local_output_path_mapping": execution["local_output_path_mapping"],
            "result_files_or_manifests_modified": False,
            "no_remote_sync_performed": True,
            "checkpoint_files_synchronized": False,
            "local_sumo_executed": False,
        },
        "development_scientific_result": {
            "analysis": descriptor(PATHS["analysis"]),
            "candidate": scientific["candidate"],
            "direct_behavior_parent": scientific["direct_behavior_parent"],
            "comparison_controls": scientific["comparison_controls"],
            "algorithm_means": scientific["algorithm_means"],
            "candidate_vs_direct_parent_mean": scientific[
                "candidate_vs_direct_parent_mean"
            ],
            "candidate_vs_direct_parent_by_training_seed": scientific[
                "candidate_vs_direct_parent_by_training_seed"
            ],
            "candidate_vs_direct_parent_by_demand_scale": scientific[
                "candidate_vs_direct_parent_by_demand_scale"
            ],
            "candidate_control_comparisons": scientific[
                "candidate_control_comparisons"
            ],
            "candidate_relative_primary_change_vs_sac_by_demand_scale": scientific[
                "candidate_relative_primary_change_vs_sac_by_demand_scale"
            ],
            "gate_specification": scientific["gate_specification"],
            "checks": checks,
            "passed_checks": sorted(checks),
            "failed_checks": [],
            "full_development_control_matrix_passed": True,
        },
        "remote_sync_policy": {
            "minimal_result_and_metadata_sync": True,
            "checkpoint_optimizer_or_replay_artifacts_synchronized": False,
            "manifest_large_artifact_path_count": 0,
        },
        "prior_v14_through_v19_development_outcomes_used_for_v20_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_eligible": True,
        "formal_confirmation_authorized_or_submitted": False,
        "claim_boundary": (
            "The verified v20 full development control matrix passed and creates "
            "eligibility for a separately authorized formal confirmation only. It "
            "is not a formal held-out effect and does not authorize or submit one."
        ),
        "closeout_tools": {
            "builder": descriptor(Path(__file__)),
            "independent_verifier": descriptor(
                REPRO / "verify_h2oplus_v20_full_control_evaluation_closeout_v1.py"
            ),
        },
    }
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
