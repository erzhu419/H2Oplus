#!/usr/bin/env python3
"""Register the fresh Python 3.8-compatible v17-v2 evaluation successor."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT_ROOT = (
    REPRO
    / "h2oplus_v17_tail_service_reserve_results/evaluation_incidents"
    / "v1_python38_removeprefix"
)
OLD_DISPATCH = REPRO / "run_h2oplus_v17_tail_service_reserve_evaluation_dispatch.py"
NEW_DISPATCH = REPRO / "run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2.py"
AUTHORIZATION_OUTPUT = (
    REPRO / "h2oplus_v17_tail_service_reserve_successor_authorization_v2.json"
)
CORRECTION_OUTPUT = (
    REPRO / "h2oplus_v17_tail_service_reserve_python38_correction_v2.json"
)
REGISTRATION_OUTPUT = (
    REPRO / "h2oplus_v17_tail_service_reserve_successor_registration_v2.json"
)
EXPECTED = {
    "v1_authorization": "6871d458b23bbc796aa8f0a981eb21f865f6a5a0d61c3c427173573e44261e4a",
    "v1_registration": "d5465e677e29971eb1c6e88a926387bacd172de737f669f6fd84426bb7c68364",
    "v1_protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "v1_methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "v1_source_freeze": "30e4b23219dc01c96b7eeaf89e4a8564716a887b9e525e8bcf65164ca8026f95",
    "v1_submission": "37795ac948a19bb49667d4a54b7304be6c10952a39a79430da3b18644aeda804",
    "v1_deployment_manifest": "a4bb3706a8f401d347c0e403f7b37ce47f142401f3c4aac3a4507d6e1a64ef0d",
    "incident_closeout": "0d189243e83ee014c414fd154f26ebc52107c934dee51131a5aecaf9cd3e19ed",
    "incident_manifest": "7974f184003663aaea39c0090704be0a701bcb2465a1b35b0a2265a9b6ee4d73",
    "incident_verification": "922b514bbab7228da049eed036cb01e1f85b93c1e3de24bc79ad2a97b23150e0",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "old_dispatch": "e69525c9ebf79f5fe5a8e8826110ab465ae6bb3cf609fae251112f573e926867",
}
INPUTS = {
    "v1_authorization": REPRO / "h2oplus_v17_tail_service_reserve_authorization_v1.json",
    "v1_registration": REPRO / "h2oplus_v17_tail_service_reserve_registration_v1.json",
    "v1_protocol": REPRO / "h2oplus_v17_tail_service_reserve_protocol_v1.json",
    "v1_methods": REPRO / "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
    "v1_source_freeze": REPRO / "h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json",
    "v1_submission": REPRO / "h2oplus_v17_tail_service_reserve_evaluation_submission_v1.json",
    "v1_deployment_manifest": (
        REPRO
        / "deployments/h2oplus_v17_tail_service_reserve_evaluation_v1"
        / "workspace/deployment_manifest.json"
    ),
    "incident_closeout": INCIDENT_ROOT / "incident_closeout_v1.json",
    "incident_manifest": INCIDENT_ROOT / "incident_file_manifest_v1.json",
    "incident_verification": INCIDENT_ROOT / "incident_verification_v1.json",
    "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": (
        REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json"
    ),
    "old_dispatch": OLD_DISPATCH,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def future_descriptor(path: Path, payload: dict) -> dict:
    value = encode(payload)
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "bytes": len(value),
        "sha256": sha256_bytes(value),
    }


def validate_inputs() -> tuple[dict, dict, dict]:
    for name, path in INPUTS.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen successor input drifted: {name}")
    incident = json.loads(INPUTS["incident_closeout"].read_text())
    incident_verification = json.loads(INPUTS["incident_verification"].read_text())
    audit = json.loads(INPUTS["data_environment_audit"].read_text())
    audit_verification = json.loads(INPUTS["data_environment_verification"].read_text())
    if (
        incident.get("status") != "sealed-invalid-runtime-lineage"
        or incident.get("preserved_incident_result_count") != 45
        or incident.get("incident_results_reusable") is not False
        or incident.get("scientific_numeric_outcomes_inspected") is not False
        or incident.get("remote_python_version") != "3.8.20"
        or incident_verification.get("passed") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or audit.get("v17_incident_numeric_outcomes_inspected") is not False
        or audit.get("conclusions", {}).get("artifact_integrity") != "passed"
        or audit.get("conclusions", {}).get("material_data_or_environment_factors_present")
        is not True
        or audit_verification.get("passed") is not True
    ):
        raise ValueError("successor incident or audit boundary drifted")

    old_text = OLD_DISPATCH.read_text()
    new_text = NEW_DISPATCH.resolve(strict=True).read_text()
    before = 'variant = algorithm.removeprefix("h2oplus-v17-tail-service-reserve-")'
    after = 'variant = v17_algorithm_variant(algorithm)'
    if before not in old_text or before in new_text or after not in new_text:
        raise ValueError("Python 3.8 compatibility rewrite is missing")
    if ".removeprefix(" in new_text or ".removesuffix(" in new_text:
        raise ValueError("v17-v2 dispatch still uses a Python 3.9 string API")
    if (
        "def v17_algorithm_variant(algorithm: str) -> str:" not in new_text
        or 'if not algorithm.startswith(prefix):' not in new_text
        or "return algorithm[len(prefix):]" not in new_text
    ):
        raise ValueError("v17-v2 prefix slicing is not fail closed")
    return incident, audit, {
        "before": before,
        "after": after,
        "before_sha256": sha256(OLD_DISPATCH),
        "after_sha256": sha256(NEW_DISPATCH),
    }


def atomic_group(payloads: list[tuple[Path, dict]]) -> None:
    for path, _ in payloads:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite successor evidence: {path}")
    temporary = []
    try:
        for path, payload in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_bytes(encode(payload))
            temporary.append((partial, path))
        for partial, path in temporary:
            os.replace(partial, path)
    finally:
        for partial, _ in temporary:
            partial.unlink(missing_ok=True)


def main() -> int:
    incident, audit, rewrite = validate_inputs()
    created = datetime.now(timezone.utc).isoformat()
    authorization = {
        "schema_version": (
            "transit-h2oplus-v17-tail-service-reserve-successor-authorization-v2"
        ),
        "created_utc": created,
        "status": "authorized-before-any-v17-v2-development-outcome",
        "study_stage": "development",
        "request_summary": (
            "Register and freeze a fresh Python 3.8-compatible successor and analyze "
            "data/environment factors affecting convergence or effectiveness."
        ),
        "authorization_scope": {
            "remote_development_evaluation_authorized": True,
            "remote_host": "jtl110gpu",
            "remote_sumo_episode_cap": 135,
            "unique_task_count_cap": 1,
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
            "duplicate_submission_authorized": False,
            "force_dispatch_authorized": False,
            "move_host_authorized": False,
            "reuse_v1_incident_results_authorized": False,
            "local_sumo_authorized": False,
            "local_training_authorized": False,
            "new_training_authorized": False,
            "formal_held_out_access_authorized": False,
            "formal_confirmation_stage_authorized": False,
        },
        "v1_incident_episode_accounting": {
            "preserved_invalid_results": 45,
            "counted_as_v17_v2_results": 0,
            "reusable": False,
        },
        "formal_held_out_outcomes_read": False,
    }
    correction = {
        "schema_version": (
            "transit-h2oplus-v17-tail-service-reserve-python38-correction-v2"
        ),
        "created_utc": created,
        "status": "corrected-before-any-v17-v2-development-outcome",
        "study_stage": "development",
        "scope": "Python 3.8 compatibility and fresh evidence identity only",
        "reason": (
            "The frozen remote runtime is Python 3.8.20, where str.removeprefix is "
            "unavailable. Prefix validation plus slicing is semantically equivalent."
        ),
        "compatibility_rewrite": rewrite,
        "changed_source_paths": [
            "reproducibility/run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2.py"
        ],
        "scientific_source_paths_changed": [],
        "scientific_behavior_changed": False,
        "methods_protocol_or_aliases_changed": False,
        "training_or_policy_artifacts_changed": False,
        "runtime_changed": False,
        "v1_incident_results_reused": False,
        "v1_incident_numeric_outcomes_inspected": False,
        "formal_held_out_outcomes_read": False,
        "old_dispatch": descriptor(OLD_DISPATCH),
        "new_dispatch": descriptor(NEW_DISPATCH),
        "incident_closeout": descriptor(INPUTS["incident_closeout"]),
        "incident_verification": descriptor(INPUTS["incident_verification"]),
        "data_environment_audit": descriptor(INPUTS["data_environment_audit"]),
        "data_environment_verification": descriptor(
            INPUTS["data_environment_verification"]
        ),
    }
    registration = {
        "schema_version": (
            "transit-h2oplus-v17-tail-service-reserve-successor-registration-v2"
        ),
        "created_utc": created,
        "status": "registered-before-any-v17-v2-development-outcome",
        "study_stage": "development",
        "candidate": {
            "algorithm": "h2oplus-v17-tail-service-reserve-thirty",
            "tail_service_floor_seconds": 30.0,
        },
        "diagnostics_not_eligible_for_selection": [
            {
                "algorithm": "h2oplus-v17-tail-service-reserve-twenty",
                "tail_service_floor_seconds": 20.0,
            },
            {
                "algorithm": "h2oplus-v17-tail-service-reserve-forty",
                "tail_service_floor_seconds": 40.0,
            },
        ],
        "scientific_design_identity": {
            "unchanged_from_v17_v1": True,
            "protocol": descriptor(INPUTS["v1_protocol"]),
            "methods": descriptor(INPUTS["v1_methods"]),
            "physical_behavior_count": 15,
            "development_blocks": 3,
            "jobs_per_block": 45,
            "new_remote_sumo_episodes": 135,
        },
        "operational_successor_identity": {
            "fresh_registration": True,
            "fresh_source_freeze_required": True,
            "fresh_deployment_required": True,
            "fresh_output_root": (
                "reproducibility/deployments/"
                "h2oplus_v17_tail_service_reserve_evaluation_v2/remote_results_v2"
            ),
            "unique_submission_required": True,
        },
        "selection_rules": {
            "candidate_selection_after_v17_v2_outcomes": False,
            "diagnostics_cannot_replace_candidate": True,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "v1_incident_results_forbidden": True,
        },
        "authorization": future_descriptor(AUTHORIZATION_OUTPUT, authorization),
        "correction": future_descriptor(CORRECTION_OUTPUT, correction),
        "v1_incident": {
            "closeout": descriptor(INPUTS["incident_closeout"]),
            "verification": descriptor(INPUTS["incident_verification"]),
            "invalid_task_ids": incident["invalid_task_ids"],
            "preserved_result_count": incident["preserved_incident_result_count"],
            "result_reuse_forbidden": True,
        },
        "data_environment_audit": {
            "audit": descriptor(INPUTS["data_environment_audit"]),
            "verification": descriptor(INPUTS["data_environment_verification"]),
            "material_factors_present": audit["conclusions"][
                "material_data_or_environment_factors_present"
            ],
            "design_changed_post_audit": False,
        },
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "prior_v14_v15_v16_development_outcomes_used_for_design": True,
        "v17_v1_scientific_numeric_outcomes_read": False,
        "v17_v2_development_evaluation_outcomes_read": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "local_training_or_sumo_allowed": False,
    }
    atomic_group(
        [
            (AUTHORIZATION_OUTPUT, authorization),
            (CORRECTION_OUTPUT, correction),
            (REGISTRATION_OUTPUT, registration),
        ]
    )
    print(
        json.dumps(
            {
                "authorization": future_descriptor(AUTHORIZATION_OUTPUT, authorization),
                "correction": future_descriptor(CORRECTION_OUTPUT, correction),
                "registration": future_descriptor(REGISTRATION_OUTPUT, registration),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
