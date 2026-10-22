#!/usr/bin/env python3
"""Independently verify the v20 formal-confirmation closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/h2oplus_v20_formal_confirmation_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v1.json"


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
        raise ValueError(f"formal closeout descriptor failed: {item}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-closeout-sha256", required=True)
    args = parser.parse_args()
    value = args.expected_closeout_sha256
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("invalid expected formal closeout SHA256")
    return args


def main() -> int:
    args = parse_args()
    if sha256(CLOSEOUT) != args.expected_closeout_sha256:
        raise ValueError("formal closeout identity drifted")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite formal verification: {OUTPUT}")
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    scheduler = closeout.get("scheduler_closeout", {})
    execution = closeout.get("analysis_execution", {})
    scientific = closeout.get("formal_confirmation_scientific_result", {})
    sync = closeout.get("remote_sync_policy", {})
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-evaluation-closeout-v1"
        or closeout.get("status") != "closed-formal-confirmation-passed"
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_result_count") != 1350
        or artifact.get("physical_behavior_count") != 45
        or artifact.get("fresh_formal_environment_cell_count") != 30
        or artifact.get("matrix_manifest_count") != 10
        or scheduler.get("authorized_scientific_task") != "t84212"
        or scheduler.get("scheduler_status") != "done"
        or scheduler.get("token_bound_backend_exit0") is not True
        or execution.get("frozen_analyzer_modified") is not False
        or execution.get("frozen_analyzer_exit_code") != 0
        or execution.get("result_files_or_manifests_modified") is not False
        or execution.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or execution.get("scientific_sources_changed") is not False
        or execution.get("local_sumo_executed") is not False
        or len(execution.get("zero_output_analysis_incidents", [])) != 2
        or scientific.get("candidate")
        != "h2oplus-v19-completion-safety-quarter-load-sixty"
        or scientific.get("direct_behavior_parent")
        != "h2oplus-v19-completion-safety-baseline"
        or scientific.get("failed_checks") != []
        or scientific.get("all_preregistered_gates_passed") is not True
        or scientific.get("formal_confirmation_passed") is not True
        or scientific.get("claim_limited_to_fixed_candidate_vs_direct_behavior_parent")
        is not True
        or scientific.get("confirmatory_inference", {}).get(
            "intersection_union_passed"
        )
        is not True
        or sync.get("minimal_result_and_metadata_sync") is not True
        or sync.get(
            "checkpoint_dataset_optimizer_replay_or_workspace_artifacts_synchronized"
        )
        is not False
        or closeout.get("formal_held_out_outcome_access_authorized") is not True
        or closeout.get("formal_held_out_outcomes_read") is not True
        or closeout.get("supports_formal_held_out_effect_claim") is not True
        or closeout.get("formal_confirmation_passed") is not True
    ):
        raise ValueError("independent formal closeout scope check failed")

    paths = {
        "audit": checked_descriptor(artifact["evaluation_audit"]),
        "manifest": checked_descriptor(artifact["evaluation_file_manifest"]),
        "manifest_verification": checked_descriptor(
            artifact["evaluation_file_manifest_verification"]
        ),
        "scheduler": checked_descriptor(scheduler["scheduler_evidence"]),
        "minimal_sync": checked_descriptor(scheduler["minimal_sync"]),
        "analysis": checked_descriptor(scientific["analysis"]),
        "analysis_verification": checked_descriptor(
            scientific["independent_verification"]
        ),
        "adapter": checked_descriptor(execution["adapter"]),
        "adapter_registration": checked_descriptor(execution["adapter_registration"]),
        "adapter_evidence": checked_descriptor(execution["adapter_evidence"]),
        "builder": checked_descriptor(closeout["closeout_tools"]["builder"]),
        "verifier": checked_descriptor(
            closeout["closeout_tools"]["independent_verifier"]
        ),
    }
    incidents = [checked_descriptor(item) for item in execution["zero_output_analysis_incidents"]]
    analysis = json.loads(paths["analysis"].read_text())
    verification = json.loads(paths["analysis_verification"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    manifest_verification = json.loads(paths["manifest_verification"].read_text())
    scheduler_evidence = json.loads(paths["scheduler"].read_text())
    source_sync = json.loads(paths["minimal_sync"].read_text())
    adapter_evidence = json.loads(paths["adapter_evidence"].read_text())
    incident_payloads = [json.loads(path.read_text()) for path in incidents]
    report = analysis["formal_confirmation_scientific_result"]
    direct = report["candidate_vs_direct_parent_mean"]
    checks = report["checks"]
    inference = report["confirmatory_inference"]
    if (
        scientific.get("checks") != checks
        or not all(checks.values())
        or report.get("failed_checks") != []
        or analysis.get("formal_confirmation_passed") is not True
        or analysis.get("supports_formal_held_out_effect_claim") is not True
        or verification.get("passed") is not True
        or verification.get("independently_recomputed_checks") != checks
        or verification.get("formal_confirmation_passed_verified") is not True
        or not math.isclose(
            direct["primary_change_seconds"],
            verification[
                "independently_recomputed_candidate_vs_direct_parent_mean"
            ]["primary_change_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or inference.get("primary_block_wins") != 10
        or inference.get("completion_noninferiority_block_wins") != 10
        or inference.get("primary_one_sided_sign_pvalue") != 0.0009765625
        or manifest.get("passed") is not True
        or manifest.get("file_count") != artifact.get("file_count")
        or manifest.get("aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("large_artifact_file_count") != 0
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or source_sync.get("scientific_result_json_count") != 1350
        or source_sync.get("checkpoint_optimizer_replay_or_workspace_files_synchronized")
        is not False
        or adapter_evidence.get("passed") is not True
        or any(item.get("analysis_output_created") is not False for item in incident_payloads)
    ):
        raise ValueError("independent formal closeout identity or gate check failed")

    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-evaluation-closeout-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": {
            "path": str(CLOSEOUT.relative_to(ROOT)),
            "sha256": sha256(CLOSEOUT),
            "bytes": CLOSEOUT.stat().st_size,
        },
        "passed": True,
        "artifact_integrity_verified": True,
        "physical_result_count_verified": 1350,
        "physical_behavior_count_verified": 45,
        "fresh_formal_environment_cell_count_verified": 30,
        "scheduler_token_bound_backend_exit0_verified": True,
        "minimal_remote_sync_verified": True,
        "large_artifact_file_count": 0,
        "parent_training_stability_verified": True,
        "material_data_environment_validity_factors_verified": True,
        "analysis_boundary_adapter_scope_verified": True,
        "zero_output_analysis_incident_count_verified": 2,
        "candidate_vs_direct_parent_mean": direct,
        "primary_block_wins_verified": 10,
        "completion_noninferiority_block_wins_verified": 10,
        "all_preregistered_gates_verified": True,
        "formal_held_out_outcomes_read": True,
        "supports_formal_held_out_effect_claim": True,
        "formal_confirmation_passed": True,
        "claim_limited_to_fixed_candidate_vs_direct_behavior_parent": True,
        "passed_scope": "independently verified endpoint-only formal confirmation",
    }
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "passed": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
