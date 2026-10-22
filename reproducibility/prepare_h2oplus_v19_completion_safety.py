#!/usr/bin/env python3
"""Preregister the v19 completion-safety paired development screen."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SUMO_ROOT = PROJECT_ROOT / "SUMO_ruiguang"
REPRO = ROOT / "reproducibility"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256  # noqa: E402
from reproducibility.build_passenger_scenarios import build_scenario  # noqa: E402
from reproducibility.prepare_h2oplus_v8_development_stage_a import (  # noqa: E402
    _design_inventory,
    _portable_scenario,
    _runtime_key,
)
from reproducibility.run_evaluation_matrix_v19_completion_safety import (  # noqa: E402
    ACTIVATION_END_FRACTION,
    ACTIVATION_START_FRACTION,
    METHODS_SCHEMA,
    PARENT_ACTIVATION_END_FRACTION,
    PARENT_ACTIVATION_START_FRACTION,
    PARENT_BASE_HOLD_SCALE,
    PARENT_HEADWAY_BALANCE_GAIN,
    PARENT_MAX_ONBOARD_CAPACITY_FRACTION,
    PARENT_RESERVE_HORIZON_FRACTION,
    PARENT_TAIL_SERVICE_FLOOR_SECONDS,
    SELECTION_RULE,
    VARIANT_CONFIG,
)


USER_AUTHORIZATION_TEXT = (
    "继续改进 仔细分析下到底什么问题 为什么之前GPT 5.5就能得到一个不错的结果"
)
SEEDS = tuple(range(6201, 6206))
VARIANTS = ("baseline", "quarter-load-fifty", "quarter-load-sixty")
BLOCKS = tuple(
    {
        "name": f"v19-completion-safety-development-{index:02d}",
        "sumo_seed": 7300 + index,
        "demand_realization_seed": 7400 + index,
    }
    for index in range(1, 4)
)
DEMAND_SCALES = (0.85, 1.0, 1.15)
DEPART_JITTER_SECONDS = 120.0
SCENARIO_ROOT = ROOT / (
    "bus_h2o/development_scenarios_v19/completion_safety_v1"
)

AUTHORIZATION_OUTPUT = REPRO / "h2oplus_v19_completion_safety_authorization_v1.json"
RATIONALE_OUTPUT = REPRO / "h2oplus_v19_completion_safety_rationale_v1.json"
REGISTRATION_OUTPUT = REPRO / "h2oplus_v19_completion_safety_registration_v1.json"
PROTOCOL_OUTPUT = REPRO / "h2oplus_v19_completion_safety_protocol_v1.json"
METHODS_OUTPUT = REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json"
ENVIRONMENT_OUTPUT = REPRO / "h2oplus_v19_completion_safety_environment_v1.json"

INPUTS = {
    "v17_methods": REPRO
    / "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
    "v17_protocol": REPRO / "h2oplus_v17_tail_service_reserve_protocol_v1.json",
    "v18_analysis": REPRO
    / "h2oplus_v18_empty_tail_reserve_results/empty_tail_reserve_analysis_v3.json",
    "v18_audit": REPRO
    / "h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
    "v18_closeout": REPRO
    / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
    "v18_verification": REPRO
    / (
        "h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_closeout_verification_v3.json"
    ),
    "design_audit": REPRO / "h2oplus_v19_completion_safety_design_audit_v2.json",
    "design_verification": REPRO
    / "h2oplus_v19_completion_safety_design_verification_v2.json",
    "data_environment_audit": REPRO
    / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPRO
    / "h2oplus_v17_data_environment_convergence_verification_v1.json",
    "training_environment": REPRO / "training_environment_design_v7_formal_v1.json",
    "formal_environment": REPRO / "evaluation_environment_design_v7_formal_v2.json",
}
EXPECTED = {
    "v17_methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "v17_protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "v18_analysis": "0fa89711f239b4eb798ad34d10dea79e00fd77a3274314a73d7d2e2bff745287",
    "v18_audit": "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    "v18_closeout": "e2a0285e9d95bc999cbe371ac261afb1f4526ff7a6722ba2d9d114f67c4c401c",
    "v18_verification": "6a08fa047a4616589fba04cd39eec41e9450ef1ce1fd100933039cb7354fa35a",
    "design_audit": "2238ceb015a456b80c2be5d4d06ebd96d2293fdbc88cadcfb08f087d2dafe1e8",
    "design_verification": "3f8c0e03514b2fb6e2ed46249cfe8260cfdc3c9a2872316f6575de1504543a78",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "training_environment": "12a26b1edfe9a3435153bf8dd9355e1cd5c39f1ae3653803e59b4f1f03bfb880",
    "formal_environment": "cf403a343db26e09f017d69575dceb0f87e99bfb0d931569ed18500991e04fbd",
}
PRIOR_DEVELOPMENT_ENVIRONMENTS = {
    "v8_stage_a": (
        REPRO / "h2oplus_v8_development_environment_stage_a_v1.json",
        "d4a40a900cf9f72043c1d7c2067d02d6c79f2081f856b38cb257ec2ad0d18f3c",
    ),
    "v8_fewshot": (
        REPRO / "h2oplus_v8_fewshot_environment_v1.json",
        "4ca7ac74ea9a4f327b03f88a1f7d8a32591852905fdf01142454737fa2dc71cd",
    ),
    "v9_adaptive": (
        REPRO / "h2oplus_v9_adaptive_environment_v2.json",
        "cf6faac3b684b4bddfd1391925d67d7c3d5202978bd2c6d02e6326a6bb5dd8f5",
    ),
    "v10_delayed": (
        REPRO / "h2oplus_v10_delayed_environment_v4.json",
        "ee3de3b3990e35c579f5fcfdbafe16168a83f5011f9e5b5ce44b59cea5365bb4",
    ),
    "v11_actor_hold": (
        REPRO / "h2oplus_v11_actor_hold_environment_v2.json",
        "0f005b94d5ecd361d7e10bac363be8dac5e6c9cdad7cbaf3f063744d280871de",
    ),
    "v12_through_v17": (
        REPRO / "h2oplus_v12_staged_release_environment_v2.json",
        "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    ),
    "v18": (
        REPRO / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    ),
}


def encode(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def descriptor(path: Path, owner: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            os.path.relpath(resolved, owner.parent.resolve())
            if owner is not None
            else str(resolved.relative_to(ROOT))
        ),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def future_descriptor(
    path: Path, payload: dict, owner: Path | None = None
) -> dict[str, Any]:
    value = encode(payload)
    resolved = path.resolve()
    return {
        "path": (
            os.path.relpath(resolved, owner.parent.resolve())
            if owner is not None
            else str(resolved.relative_to(ROOT))
        ),
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def load_frozen(name: str) -> tuple[Path, dict]:
    path = INPUTS[name].resolve(strict=True)
    actual = sha256(path)
    if actual != EXPECTED[name]:
        raise ValueError(f"frozen v19 input drifted: {name}: {actual}")
    payload = json.loads(path.read_text())
    if name not in {"training_environment", "formal_environment"} and payload.get(
        "formal_held_out_outcomes_read"
    ) is not False:
        raise ValueError(f"formal outcome boundary drifted in {name}")
    return path, payload


def atomic_group(payloads: list[tuple[Path, dict]]) -> None:
    existing = [str(path) for path, _ in payloads if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite v19 evidence: {existing}")
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


def validate_parent() -> dict[str, tuple[Path, dict]]:
    frozen = {name: load_frozen(name) for name in INPUTS}
    analysis = frozen["v18_analysis"][1]
    result = analysis.get("development_scientific_result", {})
    checks = result.get("checks", {})
    failed = sorted(name for name, value in checks.items() if value is False)
    audit = frozen["v18_audit"][1]
    closeout = frozen["v18_closeout"][1]
    verification = frozen["v18_verification"][1]
    design = frozen["design_audit"][1]
    design_verification = frozen["design_verification"][1]
    data_audit = frozen["data_environment_audit"][1]
    data_verification = frozen["data_environment_verification"][1]
    if (
        failed != ["candidate_mean_completion_gain"]
        or result.get("candidate_paired_mean", {}).get("completion_gain")
        != 0.00017333357950768895
        or result.get("diagnostic_paired_mean", {}).get("completion_gain")
        != 0.0003955748180052411
        or result.get("candidate_paired_mean", {}).get("primary_increase_seconds")
        != -8.26766015135699
        or analysis.get("formal_confirmation_eligible") is not False
        or audit.get("passed") is not True
        or closeout.get("passed") is not True
        or closeout.get("formal_confirmation_eligible") is not False
        or verification.get("passed") is not True
        or design.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-design-audit-v2"
        or design.get("claim_boundary", {}).get(
            "core_implementation_verified_without_sumo"
        )
        is not True
        or design.get("claim_boundary", {}).get("development_result_available")
        is not False
        or design.get("fixed_successor_design", {}).get(
            "predesignated_candidate"
        )
        != "quarter-load-sixty"
        or design_verification.get("passed") is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("v18 closeout or v19 design diagnosis drifted")
    return frozen


def build_environment_payload(created: str) -> dict:
    for name, (path, expected) in PRIOR_DEVELOPMENT_ENVIRONMENTS.items():
        if sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"prior development environment drifted: {name}")
    base_config = SUMO_ROOT / "online_control/control_sim_traci_period.sumocfg"
    routes = (
        SUMO_ROOT
        / "e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml",
        SUMO_ROOT / "e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml",
    )
    passengers = SUMO_ROOT / "online_control/initialize_obj/save_obj_passenger.add.xml"
    blocks = []
    scenarios = []
    manifest_hashes = set()
    runtime_hashes = set()
    for block in BLOCKS:
        manifests = []
        for scale in DEMAND_SCALES:
            original = build_scenario(
                base_config=base_config.resolve(strict=True),
                passenger_routes=tuple(path.resolve(strict=True) for path in routes),
                passenger_objects=passengers.resolve(strict=True),
                output_root=SCENARIO_ROOT,
                scale=scale,
                seed=block["demand_realization_seed"],
                depart_jitter_seconds=DEPART_JITTER_SECONDS,
            )
            portable, evidence = _portable_scenario(original, force=False)
            manifest_hash = evidence["manifest_sha256"]
            runtime_hash = _runtime_key(evidence)
            if manifest_hash in manifest_hashes or runtime_hash in runtime_hashes:
                raise ValueError("v19 fresh environment contains duplicate content")
            manifest_hashes.add(manifest_hash)
            runtime_hashes.add(runtime_hash)
            relative = os.path.relpath(portable, ENVIRONMENT_OUTPUT.parent)
            manifests.append(relative)
            scenarios.append(
                {
                    "block": block["name"],
                    "demand_scale": float(evidence["demand_scale"]),
                    "manifest": relative,
                    "manifest_sha256": manifest_hash,
                    "runtime_content_sha256": runtime_hash,
                }
            )
        blocks.append({**block, "scenario_manifests": manifests})

    inventory_paths = {
        "training": INPUTS["training_environment"],
        "formal_held_out_design_only": INPUTS["formal_environment"],
        **{
            name: path for name, (path, _) in PRIOR_DEVELOPMENT_ENVIRONMENTS.items()
        },
    }
    inventories = {name: _design_inventory(path) for name, path in inventory_paths.items()}
    demand_seeds = {block["demand_realization_seed"] for block in BLOCKS}
    sumo_seeds = {block["sumo_seed"] for block in BLOCKS}
    overlaps = {}
    for name, inventory in inventories.items():
        overlaps[f"{name}_demand_seed_overlap"] = sorted(
            demand_seeds & inventory["demand_seeds"]
        )
        overlaps[f"{name}_sumo_seed_overlap"] = sorted(
            sumo_seeds & inventory["sumo_seeds"]
        )
        overlaps[f"{name}_manifest_hash_overlap"] = sorted(
            manifest_hashes & inventory["manifest_hashes"]
        )
        overlaps[f"{name}_runtime_hash_overlap"] = sorted(
            runtime_hashes & inventory["runtime_hashes"]
        )
    if any(overlaps.values()):
        raise ValueError(f"v19 environment disjointness failed: {overlaps}")
    return {
        "schema_version": "transit-h2oplus-v19-completion-safety-environment-v1",
        "created_utc": created,
        "status": "frozen-before-any-v19-development-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "formal_held_out_design_metadata_read": True,
        "blocks": blocks,
        "demand_scales": list(DEMAND_SCALES),
        "depart_jitter_seconds": DEPART_JITTER_SECONDS,
        "environment_cells": len(scenarios),
        "independent_realization_blocks": len(blocks),
        "scenario_descriptors": scenarios,
        "disjointness": {
            "verified_disjoint": True,
            "formal_held_out_outcomes_read": False,
            **overlaps,
            **{
                f"{name}_design": descriptor(inventory["path"])
                for name, inventory in inventories.items()
            },
        },
        "generator": descriptor(Path(__file__).resolve()),
    }


def main() -> int:
    frozen = validate_parent()
    created = datetime.now(timezone.utc).isoformat()
    environment = build_environment_payload(created)
    authorization = {
        "schema_version": "transit-h2oplus-v19-completion-safety-authorization-v1",
        "created_utc": created,
        "status": "authorized-before-any-v19-development-outcome",
        "study_stage": "development",
        "authorization_channel": "explicit user message in current Codex task",
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode()
        ).hexdigest(),
        "authorization_scope": {
            "remote_development_evaluation_authorized": True,
            "remote_host": "jtl110gpu",
            "remote_sumo_episode_cap": 135,
            "unique_task_count_cap": 1,
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
            "new_training_authorized": False,
            "local_training_or_sumo_authorized": False,
            "duplicate_submission_authorized": False,
            "force_dispatch_authorized": False,
            "move_host_authorized": False,
            "checkpoint_selection_by_behavior_authorized": False,
            "formal_held_out_access_authorized": False,
            "formal_confirmation_stage_authorized": False,
        },
        "formal_held_out_outcomes_read": False,
    }
    authorization_desc = future_descriptor(AUTHORIZATION_OUTPUT, authorization)
    frozen_v18 = {
        name: descriptor(path)
        for name, (path, _) in frozen.items()
        if name.startswith("v18_")
    }
    design_evidence = {
        name: descriptor(frozen[name][0])
        for name in ("design_audit", "design_verification")
    }
    data_evidence = {
        name: descriptor(frozen[name][0])
        for name in ("data_environment_audit", "data_environment_verification")
    }
    rationale = {
        "schema_version": "transit-h2oplus-v19-completion-safety-rationale-v1",
        "created_utc": created,
        "status": "frozen-before-any-v19-development-outcome",
        "study_stage": "development",
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "formal_held_out_outcomes_read": False,
        "v19_development_evaluation_outcomes_read": False,
        "diagnosis": {
            "why_gpt55_looked_better": (
                "v11 changed the training trajectory and used a 0.005 completion "
                "margin; it would fail the current 0.003 completion margin"
            ),
            "primary_defect": (
                "the optimized welfare reward does not encode the evaluation-only "
                "completion noninferiority constraint"
            ),
            "material_validity_factors": [
                "3,284,718 transitions represent only 30 independent exogenous cells",
                "large positive holds are sparse in the offline dataset",
                "packed terminal state omits part of exact unfinished journey state",
                "training and development use different SUMO runtime versions",
                "v14-v18 development outcomes adaptively informed successor designs",
            ],
            "v18_signal": (
                "the narrow empty and one-seat guards improved completion too little; "
                "v19 restores the full registered quarter-load service guard"
            ),
        },
        "fixed_design": {
            "parent": "sealed v17 twenty-second 25k behavior",
            "paired_baseline": "baseline",
            "predesignated_candidate": "quarter-load-sixty",
            "diagnostic": "quarter-load-fifty",
            "activation_fraction": [
                ACTIVATION_START_FRACTION,
                ACTIVATION_END_FRACTION,
            ],
            "max_onboard_capacity_fraction": 0.25,
            "candidate_floor_seconds": 60.0,
            "diagnostic_floor_seconds": 50.0,
            "new_training_runs": 0,
            "new_remote_development_episodes": 135,
            "fresh_environment_cells": 9,
        },
        "uncertainty": {
            "global_response_projection": 0.0016743827420365436,
            "targeted_response_projection": 0.004059447344569947,
            "completion_gain_gate": 0.003,
            "success_assumed": False,
        },
        "claim_boundary": {
            "screen_can_authorize_full_development_control_matrix_only": True,
            "screen_can_establish_formal_confirmation_eligibility": False,
            "screen_can_establish_formal_held_out_effect": False,
        },
        "authorization": authorization_desc,
        "frozen_v18_evidence": frozen_v18,
        "v19_design_evidence": design_evidence,
        "data_environment_evidence": data_evidence,
    }
    rationale_desc = future_descriptor(RATIONALE_OUTPUT, rationale)
    registration = {
        "schema_version": "transit-h2oplus-v19-completion-safety-registration-v1",
        "created_utc": created,
        "status": "registered-before-any-v19-development-outcome",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "candidate": {
            "algorithm": "h2oplus-v19-completion-safety-quarter-load-sixty",
            "reserve_floor_seconds": 60.0,
            "max_onboard_capacity_fraction": 0.25,
        },
        "diagnostics_not_eligible_for_selection": [
            {"algorithm": "h2oplus-v19-completion-safety-baseline"},
            {
                "algorithm": "h2oplus-v19-completion-safety-quarter-load-fifty",
                "reserve_floor_seconds": 50.0,
            },
        ],
        "selection_rules": {
            "candidate_selection_after_v19_outcomes": False,
            "diagnostics_cannot_replace_candidate": True,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "fresh_environment_required": True,
        },
        "episode_accounting": {
            "physical_methods": 15,
            "development_cells_per_method": 9,
            "new_remote_sumo_episodes": 135,
            "reused_prior_result_episodes": 0,
        },
        "authorization": authorization_desc,
        "rationale": rationale_desc,
        "environment": future_descriptor(ENVIRONMENT_OUTPUT, environment),
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "v19_development_evaluation_outcomes_read": False,
    }
    registration_desc = future_descriptor(REGISTRATION_OUTPUT, registration)
    gates = {
        "candidate_mean_completion_gain_over_paired_baseline_at_least": 0.003,
        "candidate_mean_primary_increase_over_paired_baseline_at_most_seconds": 3.0,
        "candidate_positive_completion_gain_demand_scale_count_at_least": 3,
        "candidate_positive_completion_gain_training_seed_count_at_least": 4,
        "all_135_results_valid": True,
        "all_three_matrix_manifests_complete": True,
        "parent_artifact_integrity_passed": True,
        "parent_training_stability_passed": True,
    }
    protocol = {
        "schema_version": "transit-h2oplus-v19-completion-safety-protocol-v1",
        "created_utc": created,
        "status": "frozen-before-any-v19-development-outcome",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "v19_development_evaluation_outcomes_read": False,
        "registration": registration_desc,
        "authorization": authorization_desc,
        "rationale": rationale_desc,
        "parent_v17_protocol": descriptor(frozen["v17_protocol"][0]),
        "completion_safety_reserve": {
            "parent_algorithm": "h2oplus-v17-tail-service-reserve-twenty",
            "parent_actor_algorithm": "h2oplus-v14-terminal-quarter",
            "parent_tail_service_floor_seconds": PARENT_TAIL_SERVICE_FLOOR_SECONDS,
            "predesignated_candidate": "quarter-load-sixty",
            "diagnostic": "quarter-load-fifty",
            "candidate_floor_seconds": 60.0,
            "diagnostic_floor_seconds": 50.0,
            "activation_fraction": [
                ACTIVATION_START_FRACTION,
                ACTIVATION_END_FRACTION,
            ],
            "max_onboard_capacity_fraction": 0.25,
            "requires_forward_bus_present": True,
            "requires_backward_bus_absent": True,
            "requires_positive_eligible_arrival_rate": True,
            "parent_action_preserved_outside_guard": True,
            "physical_max_hold_preserved": True,
            "checkpoint_selection_by_behavior": False,
        },
        "evaluation": {
            "training_seeds": list(SEEDS),
            "development_blocks": [block["name"] for block in BLOCKS],
            "sumo_seeds": [block["sumo_seed"] for block in BLOCKS],
            "demand_realization_seeds": [
                block["demand_realization_seed"] for block in BLOCKS
            ],
            "demand_scales": list(DEMAND_SCALES),
            "new_physical_methods": 15,
            "new_remote_sumo_episodes": 135,
            "jobs_per_block": 45,
            "workers": 12,
            "host": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
        },
        "analysis": {
            "candidate": "h2oplus-v19-completion-safety-quarter-load-sixty",
            "paired_baseline": "h2oplus-v19-completion-safety-baseline",
            "diagnostic": "h2oplus-v19-completion-safety-quarter-load-fifty",
            "endpoint_only": True,
            "full_control_matrix_deferred": True,
            "formal_eligibility_assessment_forbidden": True,
        },
        "analysis_gates": gates,
        "environment": future_descriptor(ENVIRONMENT_OUTPUT, environment),
        "frozen_v18_evidence": frozen_v18,
        "v19_design_evidence": design_evidence,
        "data_environment_evidence": data_evidence,
    }
    protocol_desc = future_descriptor(PROTOCOL_OUTPUT, protocol)

    parent_methods = frozen["v17_methods"][1]
    parents = {
        int(row["training_seed"]): row
        for row in parent_methods["methods"]
        if row.get("algorithm") == "h2oplus-v17-tail-service-reserve-twenty"
    }
    if set(parents) != set(SEEDS):
        raise ValueError("sealed v17 twenty-second parent grid is incomplete")
    methods = []
    aliases = []
    identities = []
    behavior_keys = set()
    for seed in SEEDS:
        parent = parents[seed]
        checkpoint = Path(parent["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = frozen["v17_methods"][0].parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if sha256(checkpoint) != parent["checkpoint_sha256"]:
            raise ValueError(f"sealed v17 parent checkpoint drifted: seed {seed}")
        seed_rows = []
        for variant in VARIANTS:
            registered = VARIANT_CONFIG[variant]
            algorithm = f"h2oplus-v19-completion-safety-{variant}"
            name = f"{algorithm}-seed{seed}-events000025000"
            key = (
                seed,
                parent["actor_tensor_sha256"],
                variant,
                registered["enabled"],
                registered["floor_seconds"],
                ACTIVATION_START_FRACTION,
                ACTIVATION_END_FRACTION,
                registered["max_onboard_fraction"],
            )
            if key in behavior_keys:
                raise ValueError(f"duplicate v19 physical behavior: {key}")
            behavior_keys.add(key)
            row = dict(parent)
            row.update(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "variant": variant,
                    "checkpoint": os.path.relpath(checkpoint, METHODS_OUTPUT.parent),
                    "artifact_lineage": "v19-completion-safety",
                    "parent_artifact_lineage": "v17-tail-service-reserve",
                    "v16_parent_method": parent["parent_method"],
                    "parent_method": parent["name"],
                    "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
                    "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
                    "parent_reserve_horizon_fraction": (
                        PARENT_RESERVE_HORIZON_FRACTION
                    ),
                    "parent_tail_service_floor_seconds": (
                        PARENT_TAIL_SERVICE_FLOOR_SECONDS
                    ),
                    "parent_activation_start_fraction": (
                        PARENT_ACTIVATION_START_FRACTION
                    ),
                    "parent_activation_end_fraction": PARENT_ACTIVATION_END_FRACTION,
                    "parent_max_onboard_capacity_fraction": (
                        PARENT_MAX_ONBOARD_CAPACITY_FRACTION
                    ),
                    "completion_safety_reserve_enabled": registered["enabled"],
                    "completion_safety_reserve_floor_seconds": registered[
                        "floor_seconds"
                    ],
                    "activation_start_fraction": ACTIVATION_START_FRACTION,
                    "activation_end_fraction": ACTIVATION_END_FRACTION,
                    "max_onboard_capacity_fraction": registered[
                        "max_onboard_fraction"
                    ],
                    "predesignated_candidate": variant == "quarter-load-sixty",
                    "source_protocol_sha256": parent["source_protocol_sha256"],
                    "v17_protocol_sha256": EXPECTED["v17_protocol"],
                    "v19_protocol_sha256": protocol_desc["sha256"],
                }
            )
            methods.append(row)
            aliases.append(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "online_events": 25_000,
                    "physical_method": name,
                    "actor_tensor_sha256": parent["actor_tensor_sha256"],
                    "parent_method": parent["name"],
                    "completion_safety_reserve_enabled": registered["enabled"],
                    "completion_safety_reserve_floor_seconds": registered[
                        "floor_seconds"
                    ],
                    "activation_start_fraction": ACTIVATION_START_FRACTION,
                    "activation_end_fraction": ACTIVATION_END_FRACTION,
                    "max_onboard_capacity_fraction": registered[
                        "max_onboard_fraction"
                    ],
                    "predesignated_candidate": variant == "quarter-load-sixty",
                }
            )
            seed_rows.append(
                {
                    "variant": variant,
                    "physical_method": name,
                    "behavior_identity": list(key),
                }
            )
        identities.append(
            {
                "training_seed": seed,
                "parent_method": parent["name"],
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "actor_tensor_sha256": parent["actor_tensor_sha256"],
                "temperature_tensor_sha256": parent["temperature_tensor_sha256"],
                "completion_safety_physical_behaviors": seed_rows,
            }
        )
    methods_payload = {
        "schema_version": METHODS_SCHEMA,
        "created_utc": created,
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "v19_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "selection_rule": SELECTION_RULE,
        "predesignated_candidate": (
            "h2oplus-v19-completion-safety-quarter-load-sixty"
        ),
        "diagnostic_variants": [
            "h2oplus-v19-completion-safety-baseline",
            "h2oplus-v19-completion-safety-quarter-load-fifty",
        ],
        "physical_method_count": 15,
        "logical_method_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        **{
            key: parent_methods[key]
            for key in (
                "v8_protocol_sha256",
                "v10_protocol_sha256",
                "v11_protocol_sha256",
                "v12_protocol_sha256",
                "v13_protocol_sha256",
                "v14_protocol_sha256",
                "v15_protocol_sha256",
                "v16_protocol_sha256",
                "v17_protocol_sha256",
            )
        },
        "v19_protocol_sha256": protocol_desc["sha256"],
        "bindings": {
            "authorization": authorization_desc,
            "rationale": rationale_desc,
            "registration": registration_desc,
            "protocol": protocol_desc,
            "environment": future_descriptor(
                ENVIRONMENT_OUTPUT, environment, METHODS_OUTPUT
            ),
            "v17_methods": descriptor(frozen["v17_methods"][0], METHODS_OUTPUT),
            "v18_analysis": descriptor(frozen["v18_analysis"][0], METHODS_OUTPUT),
            "v18_closeout": descriptor(frozen["v18_closeout"][0], METHODS_OUTPUT),
            "v18_verification": descriptor(
                frozen["v18_verification"][0], METHODS_OUTPUT
            ),
            "design_audit": descriptor(frozen["design_audit"][0], METHODS_OUTPUT),
            "design_verification": descriptor(
                frozen["design_verification"][0], METHODS_OUTPUT
            ),
        },
        "identity_evidence": identities,
        "aliases": sorted(aliases, key=lambda item: item["name"]),
        "methods": sorted(methods, key=lambda item: item["name"]),
    }
    atomic_group(
        [
            (AUTHORIZATION_OUTPUT, authorization),
            (RATIONALE_OUTPUT, rationale),
            (REGISTRATION_OUTPUT, registration),
            (PROTOCOL_OUTPUT, protocol),
            (METHODS_OUTPUT, methods_payload),
            (ENVIRONMENT_OUTPUT, environment),
        ]
    )
    print(
        json.dumps(
            {
                "authorization_sha256": sha256(AUTHORIZATION_OUTPUT),
                "rationale_sha256": sha256(RATIONALE_OUTPUT),
                "registration_sha256": sha256(REGISTRATION_OUTPUT),
                "protocol_sha256": sha256(PROTOCOL_OUTPUT),
                "methods_sha256": sha256(METHODS_OUTPUT),
                "environment_sha256": sha256(ENVIRONMENT_OUTPUT),
                "physical_methods": len(methods),
                "fresh_environment_cells": len(environment["scenario_descriptors"]),
                "new_remote_sumo_episodes": 135,
                "local_sumo_executed": False,
                "local_training_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
