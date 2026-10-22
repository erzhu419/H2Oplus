#!/usr/bin/env python3
"""Independently verify the v20 full development control-matrix closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/h2oplus_v20_full_control_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v1.json"
EXPECTED_MATRIX_SHA256 = [
    "230e9ae6246bd4d6960f173c8072b4ac0d36ca9917f4dbad8e82168c11ed1d0e",
    "4e0954df68ec72a7cf0a14bad73f53bf739703c2b38a5d4874cd8ed4a474ed64",
    "4e08082078a24e1f32a7504bd68f7d1b979ed1bcdc8f3a57a7c0a54ec1509055",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_descriptor(item: dict) -> Path:
    path = Path(item.get("path", ""))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if (
        not path.is_file()
        or ROOT not in path.parents
        or path.stat().st_size != item.get("bytes")
        or sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"v20 closeout descriptor failed: {item}")
    return path


def main() -> int:
    args = parse_args()
    if sha256(CLOSEOUT) != args.expected_closeout_sha256:
        raise ValueError("v20 closeout identity drifted")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite v20 verification: {OUTPUT}")
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    scheduler = closeout.get("scheduler_closeout", {})
    execution = closeout.get("analysis_execution", {})
    scientific = closeout.get("development_scientific_result", {})
    sync_policy = closeout.get("remote_sync_policy", {})
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v20-full-control-evaluation-closeout-v1"
        or closeout.get("status") != "closed-development-full-control-matrix-passed"
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_result_count") != 405
        or artifact.get("physical_behavior_count") != 45
        or artifact.get("fresh_environment_cell_count") != 9
        or artifact.get("matrix_manifest_count") != 3
        or [row.get("sha256") for row in artifact.get("matrix_manifests", [])]
        != EXPECTED_MATRIX_SHA256
        or artifact.get("file_count") != 847
        or scheduler.get("authorized_scientific_task") != "t83955"
        or scheduler.get("scheduler_status") != "done"
        or scheduler.get("token_bound_backend_exit0") is not True
        or execution.get("frozen_analyzer_modified") is not False
        or execution.get("frozen_analyzer_exit_code") != 0
        or execution.get("result_files_or_manifests_modified") is not False
        or execution.get("no_remote_sync_performed") is not True
        or execution.get("checkpoint_files_synchronized") is not False
        or execution.get("local_sumo_executed") is not False
        or sync_policy.get("minimal_result_and_metadata_sync") is not True
        or sync_policy.get("checkpoint_optimizer_or_replay_artifacts_synchronized")
        is not False
        or sync_policy.get("manifest_large_artifact_path_count") != 0
        or scientific.get("candidate")
        != "h2oplus-v19-completion-safety-quarter-load-sixty"
        or scientific.get("direct_behavior_parent")
        != "h2oplus-v19-completion-safety-baseline"
        or scientific.get("failed_checks") != []
        or scientific.get("full_development_control_matrix_passed") is not True
        or closeout.get(
            "prior_v14_through_v19_development_outcomes_used_for_v20_design"
        )
        is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_formal_held_out_effect_claim") is not False
        or closeout.get("formal_confirmation_eligible") is not True
        or closeout.get("formal_confirmation_authorized_or_submitted") is not False
    ):
        raise ValueError("independent v20 closeout scope check failed")
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
        "runner": checked_descriptor(execution["runner"]),
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
    report = analysis["development_scientific_result"]
    gates = report["gate_specification"]
    direct = report["candidate_vs_direct_parent_mean"]
    by_seed = report["candidate_vs_direct_parent_by_training_seed"]
    by_scale = report["candidate_vs_direct_parent_by_demand_scale"]
    comparisons = report["candidate_control_comparisons"]
    sac_by_scale = report[
        "candidate_relative_primary_change_vs_sac_by_demand_scale"
    ]
    recomputed = {
        "all_405_results_valid": (
            audit.get("passed") is True
            and audit.get("artifact_integrity", {}).get("physical_result_count") == 405
        ),
        "all_three_135_job_matrix_manifests_complete": (
            audit.get("artifact_integrity", {}).get("matrix_manifest_count") == 3
            and all(row.get("jobs") == 135 for row in artifact["matrix_manifests"])
        ),
        "candidate_completion_gain_positive_all_demand_scales": (
            len(by_scale)
            >= int(gates["candidate_positive_completion_gain_demand_scale_count_at_least"])
            and all(float(row["completion_change"]) > 0.0 for row in by_scale.values())
        ),
        "candidate_completion_gain_positive_at_least_four_training_seeds": (
            sum(float(row["completion_change"]) > 0.0 for row in by_seed.values())
            >= int(gates["candidate_positive_completion_gain_training_seed_count_at_least"])
        ),
        "candidate_completion_noninferior_to_every_comparison_control": all(
            float(row["mean_completion_change"])
            >= -float(gates["candidate_completion_noninferiority_margin_per_control"])
            for row in comparisons.values()
        ),
        "candidate_mean_completion_gain_over_direct_parent": (
            float(direct["completion_change"])
            >= float(gates["candidate_mean_completion_gain_over_direct_parent_at_least"])
        ),
        "candidate_mean_primary_lower_than_every_comparison_control": all(
            float(row["mean_primary_change_seconds"]) < 0.0
            for row in comparisons.values()
        ),
        "candidate_mean_primary_noninferiority_to_direct_parent": (
            float(direct["primary_change_seconds"])
            <= float(gates["candidate_mean_primary_increase_over_direct_parent_at_most_seconds"])
        ),
        "candidate_primary_robust_at_all_demand_scales_vs_sac": all(
            float(value)
            <= float(gates["candidate_primary_demand_scale_degradation_vs_sac_at_most_fraction"])
            for value in sac_by_scale.values()
        ),
        "candidate_primary_seed_wins_against_every_comparison_control": all(
            int(row["primary_seed_wins"])
            >= int(gates["candidate_primary_seed_wins_per_comparison_control_at_least"])
            for row in comparisons.values()
        ),
        "parent_artifact_integrity_passed": (
            analysis.get("artifact_integrity", {}).get("passed") is True
        ),
        "parent_training_stability_passed": (
            analysis.get("parent_training_stability", {}).get("passed") is True
        ),
    }
    large_paths = [
        item["path"]
        for item in manifest["files"]
        if item["path"].lower().endswith((".pt", ".pth", ".ckpt", ".pkl", ".pickle"))
        or any(token in item["path"].lower() for token in ("replay_buffer", "optimizer"))
    ]
    if (
        scientific.get("checks") != recomputed
        or report.get("checks") != recomputed
        or not all(recomputed.values())
        or manifest.get("passed") is not True
        or manifest.get("file_count") != 847
        or manifest.get("aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("independently_recomputed_file_count") != 847
        or scheduler_evidence.get("task", {}).get("task_id") != "t83955"
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or source_sync.get("physical_result_count") != 405
        or source_sync.get("exact_remote_local_match") is not True
        or execution_evidence.get("passed") is not True
        or execution_evidence.get("analysis_output", {}).get("sha256")
        != sha256(paths["analysis"])
        or large_paths
    ):
        raise ValueError("independent v20 identity or gate check failed")
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-evaluation-closeout-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": {
            "path": str(CLOSEOUT.relative_to(ROOT)),
            "sha256": sha256(CLOSEOUT),
            "bytes": CLOSEOUT.stat().st_size,
        },
        "artifact_integrity_verified": True,
        "physical_result_count_verified": 405,
        "physical_behavior_count_verified": 45,
        "fresh_development_environment_cell_count_verified": 9,
        "scheduler_token_bound_backend_exit0_verified": True,
        "parent_training_stability_verified": True,
        "material_data_environment_validity_factors_verified": True,
        "full_development_control_matrix_passed": True,
        "failed_checks": [],
        "candidate_vs_direct_parent_mean": direct,
        "minimal_remote_sync_verified": True,
        "checkpoint_optimizer_or_replay_artifact_count": 0,
        "prior_v14_through_v19_development_outcomes_used_for_v20_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_eligible": True,
        "formal_confirmation_authorized_or_submitted": False,
        "passed": True,
        "passed_scope": "independent verified development closeout",
    }
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-closeout-sha256", required=True)
    args = parser.parse_args()
    value = args.expected_closeout_sha256
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("invalid expected v20 closeout SHA256")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
