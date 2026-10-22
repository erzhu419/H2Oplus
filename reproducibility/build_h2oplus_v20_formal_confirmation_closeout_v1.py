#!/usr/bin/env python3
"""Build the verified v20 formal-confirmation closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
OUTPUT = RESULTS / "evaluation_closeout_v1.json"
PATHS = {
    "submission": REPRO / "h2oplus_v20_formal_confirmation_submission_v4.json",
    "scheduler": RESULTS / "evidence/evaluation_v4/scheduler_evidence_v4.json",
    "sync": REPRO / "h2oplus_v20_formal_confirmation_minimal_sync_v4.json",
    "audit": RESULTS / "evaluation_audit_v1.json",
    "manifest": RESULTS / "evaluation_file_manifest_v1.json",
    "manifest_verification": RESULTS / "evaluation_file_manifest_verification_v1.json",
    "compaction": RESULTS / "evaluation_v4_log_compaction_v1.json",
    "analysis": RESULTS / "formal_confirmation_analysis_v1.json",
    "analysis_verification": RESULTS / "formal_confirmation_analysis_verification_v1.json",
    "adapter_registration": REPRO / "h2oplus_v20_formal_confirmation_analysis_adapter_registration_v2.json",
    "adapter": REPRO / "run_frozen_h2oplus_v20_formal_confirmation_analysis_v2.py",
    "adapter_evidence": RESULTS / "analysis_boundary_adapter_evidence_v2.json",
    "direct_incident": RESULTS / "analysis_incidents/direct_block_label_mismatch_v1.json",
    "adapter_v1_incident": RESULTS / "analysis_incidents/adapter_v1_output_path_mismatch_v1.json",
    "execution_authorization": REPRO / "h2oplus_v20_formal_confirmation_execution_authorization_v1.json",
    "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
}
EXPECTED = {
    "submission": "6dea4c147d9ee987ef3f9bb970c0e9f5b79be6a9e85a8f8a92fc8e0365da35dc",
    "scheduler": "19efdd5b7942eda3be9ed122950eb30130364cc5fde56ca8252b7dac252df190",
    "sync": "c593de405c43cf2c28085072fff593a5cd913526f89b60e8ee61d9b8aa1eed88",
    "audit": "aa71cfa4060c22dc6413a500f0aa922b415a7fc09516231ff8b99401c15dd5d6",
    "manifest": "9b04c124fb584fbab0abe3abc0718a35f857d083288b4c7fc1610b900d81b705",
    "manifest_verification": "e439c25e4f8d73030f8d22cdb16033c2ef0ed4099436b6ee01064b79b82e018b",
    "compaction": "639c29f8e9ddf0f094e32c08d9d1d587d5601c5264b57e800e5d5f10545bb3b0",
    "analysis": "8042a1ca229959bbff7040c328626af75b1a87f5d0908d8f82d843f348a9a21e",
    "analysis_verification": "859c957296e553e1e58fad9283db7fd205087902ea567083c229b01690e063c7",
    "adapter_registration": "bca31361b8d26618809f1aa0011839adaeb86fca3f39796a5f0d8fac9113c8c6",
    "adapter": "349a3a6321b6da6130054b08ae6f2a8163cdb96be1672a6273a385513ada8ec2",
    "adapter_evidence": "6a3e04d7955707c60b1e224bfbc16d767a620c18ff32189464b44772c5ccb93e",
    "direct_incident": "b74fb97998c621dabfdd228c1ffe79b557ee3a504d3cf01596b88de97c81f4d9",
    "adapter_v1_incident": "b42ddcd0354cc58ce36e0a287b99cf0af419822ffe64fbc31478eee9c154dc7e",
    "execution_authorization": "c9e2cdfe10074e5ddded7f756c682b9c5a63d3e5fc6fc52c87498289407226d5",
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
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def checked_json(name: str) -> dict:
    path = PATHS[name].resolve(strict=True)
    if sha256(path) != EXPECTED[name]:
        raise ValueError(f"formal closeout input drifted: {name}")
    return json.loads(path.read_text())


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite formal closeout: {OUTPUT}")
    scheduler = checked_json("scheduler")
    sync = checked_json("sync")
    audit = checked_json("audit")
    manifest = checked_json("manifest")
    manifest_verification = checked_json("manifest_verification")
    compaction = checked_json("compaction")
    analysis = checked_json("analysis")
    verification = checked_json("analysis_verification")
    adapter = checked_json("adapter_evidence")
    direct_incident = checked_json("direct_incident")
    adapter_v1_incident = checked_json("adapter_v1_incident")
    data_audit = checked_json("data_environment_audit")
    data_verification = checked_json("data_environment_verification")
    scientific = analysis.get("formal_confirmation_scientific_result", {})
    checks = scientific.get("checks", {})
    direct = scientific.get("candidate_vs_direct_parent_mean", {})
    inference = scientific.get("confirmatory_inference", {})
    if (
        scheduler.get("passed") is not True
        or scheduler.get("authorized_scientific_task") != "t84212"
        or scheduler.get("task", {}).get("status") != "done"
        or scheduler.get("token_bound_backend_exit0") is not True
        or sync.get("passed") is not True
        or sync.get("scientific_result_json_count") != 1350
        or sync.get("checkpoint_optimizer_replay_or_workspace_files_synchronized")
        is not False
        or audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("physical_result_count") != 1350
        or audit.get("artifact_integrity", {}).get("physical_behavior_count") != 45
        or audit.get("artifact_integrity", {}).get(
            "fresh_formal_environment_cell_count"
        )
        != 30
        or manifest.get("passed") is not True
        or manifest.get("physical_result_count") != 1350
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("large_artifact_file_count") != 0
        or compaction.get("scientific_results_deleted") is not False
        or adapter.get("passed") is not True
        or adapter.get("result_files_or_manifests_modified") is not False
        or adapter.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or adapter.get("scientific_sources_changed") is not False
        or direct_incident.get("analysis_output_created") is not False
        or adapter_v1_incident.get("analysis_output_created") is not False
        or analysis.get("formal_confirmation_passed") is not True
        or analysis.get("supports_formal_held_out_effect_claim") is not True
        or scientific.get("all_preregistered_gates_passed") is not True
        or scientific.get("failed_checks") != []
        or not checks
        or not all(checks.values())
        or inference.get("intersection_union_passed") is not True
        or inference.get("claim_limited_to_direct_behavior_parent") is not True
        or verification.get("passed") is not True
        or verification.get("all_preregistered_gates_verified") is not True
        or verification.get("formal_confirmation_passed_verified") is not True
        or verification.get("supports_formal_held_out_effect_claim_verified")
        is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("formal closeout prerequisites failed")

    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-evaluation-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed-formal-confirmation-passed",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "passed": True,
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 1350,
            "physical_behavior_count": 45,
            "fresh_formal_environment_cell_count": 30,
            "matrix_manifest_count": 10,
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
            "authorized_scientific_task": "t84212",
            "scheduler_status": "done",
            "token_bound_backend_exit0": True,
            "scheduler_evidence": descriptor(PATHS["scheduler"]),
            "minimal_sync": descriptor(PATHS["sync"]),
        },
        "parent_training_stability": analysis["parent_training_stability"],
        "data_environment_validity_risks": analysis[
            "data_environment_validity_risks"
        ],
        "analysis_execution": {
            "frozen_analyzer": adapter["frozen_analyzer"],
            "frozen_analyzer_modified": False,
            "frozen_analyzer_exit_code": 0,
            "adapter": descriptor(PATHS["adapter"]),
            "adapter_registration": descriptor(PATHS["adapter_registration"]),
            "adapter_evidence": descriptor(PATHS["adapter_evidence"]),
            "adapter_scope": adapter["adapter_scope"],
            "zero_output_analysis_incidents": [
                descriptor(PATHS["direct_incident"]),
                descriptor(PATHS["adapter_v1_incident"]),
            ],
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
            "scientific_sources_changed": False,
            "local_sumo_executed": False,
        },
        "formal_confirmation_scientific_result": {
            "analysis": descriptor(PATHS["analysis"]),
            "independent_verification": descriptor(PATHS["analysis_verification"]),
            "candidate": scientific["candidate"],
            "direct_behavior_parent": scientific["direct_behavior_parent"],
            "candidate_vs_direct_parent_mean": direct,
            "candidate_vs_direct_parent_by_demand_scale": scientific[
                "candidate_vs_direct_parent_by_demand_scale"
            ],
            "confirmatory_inference": inference,
            "gate_specification": scientific["gate_specification"],
            "checks": checks,
            "failed_checks": [],
            "all_preregistered_gates_passed": True,
            "formal_confirmation_passed": True,
            "claim_limited_to_fixed_candidate_vs_direct_behavior_parent": True,
        },
        "remote_sync_policy": {
            "minimal_result_and_metadata_sync": True,
            "checkpoint_dataset_optimizer_replay_or_workspace_artifacts_synchronized": False,
            "redundant_per_job_logs_removed": 1350,
            "log_compaction": descriptor(PATHS["compaction"]),
        },
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_outcomes_read": True,
        "supports_formal_held_out_effect_claim": True,
        "formal_confirmation_passed": True,
        "claim_boundary": (
            "The verified formal held-out effect is limited to the fixed "
            "h2oplus-v19-completion-safety-quarter-load-sixty candidate versus "
            "its fixed h2oplus-v19-completion-safety-baseline direct behavior "
            "parent on the preregistered endpoint-only matrix."
        ),
        "limitations": [
            "Material data and environment validity factors recorded before formal confirmation remain.",
            "The claim is endpoint-only; no AULC effect was tested or claimed.",
            "Lineage-control comparisons are supportive and do not expand the confirmatory claim."
        ],
        "closeout_tools": {
            "builder": descriptor(Path(__file__)),
            "independent_verifier": descriptor(
                REPRO / "verify_h2oplus_v20_formal_confirmation_closeout_v1.py"
            ),
        },
    }
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "passed": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
