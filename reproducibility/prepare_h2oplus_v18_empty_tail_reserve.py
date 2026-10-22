#!/usr/bin/env python3
"""Preregister the v18 empty-tail reserve paired development screen."""

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
from reproducibility.run_evaluation_matrix_v18_empty_tail_reserve import (  # noqa: E402
    ACTIVATION_END_FRACTION,
    ACTIVATION_START_FRACTION,
    EMPTY_RESERVE_FLOOR_SECONDS,
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


USER_AUTHORIZATION_TEXT = "那就继续改进"
SEEDS = tuple(range(6201, 6206))
VARIANTS = ("baseline", "empty-forty", "one-seat-forty")
BLOCKS = tuple(
    {
        "name": f"v18-empty-tail-development-{index:02d}",
        "sumo_seed": 7100 + index,
        "demand_realization_seed": 7200 + index,
    }
    for index in range(1, 4)
)
DEMAND_SCALES = (0.85, 1.0, 1.15)
DEPART_JITTER_SECONDS = 120.0
SCENARIO_ROOT = ROOT / "bus_h2o/development_scenarios_v18/empty_tail_reserve_v1"

AUTHORIZATION_OUTPUT = REPRO / "h2oplus_v18_empty_tail_reserve_authorization_v1.json"
RATIONALE_OUTPUT = REPRO / "h2oplus_v18_empty_tail_reserve_rationale_v1.json"
REGISTRATION_OUTPUT = REPRO / "h2oplus_v18_empty_tail_reserve_registration_v1.json"
PROTOCOL_OUTPUT = REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json"
METHODS_OUTPUT = REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json"
ENVIRONMENT_OUTPUT = REPRO / "h2oplus_v18_empty_tail_reserve_environment_v1.json"

INPUTS = {
    "v17_methods": REPRO / "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
    "v17_protocol": REPRO / "h2oplus_v17_tail_service_reserve_protocol_v1.json",
    "v17_analysis": (
        REPRO
        / "h2oplus_v17_tail_service_reserve_results/tail_service_reserve_analysis_v2.json"
    ),
    "v17_audit": (
        REPRO / "h2oplus_v17_tail_service_reserve_results/evaluation_audit_v2.json"
    ),
    "v17_closeout": (
        REPRO / "h2oplus_v17_tail_service_reserve_results/evaluation_closeout_v2.json"
    ),
    "v17_verification": (
        REPRO
        / "h2oplus_v17_tail_service_reserve_results/evaluation_closeout_verification_v2.json"
    ),
    "data_environment_audit": (
        REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json"
    ),
    "data_environment_verification": (
        REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json"
    ),
    "training_environment": REPRO / "training_environment_design_v7_formal_v1.json",
    "formal_environment": REPRO / "evaluation_environment_design_v7_formal_v2.json",
}
EXPECTED = {
    "v17_methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "v17_protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "v17_analysis": "84e80b7f08d16f6e356f09c56ff75d1bbdf322578641df4ede9e73f8d4ff4f78",
    "v17_audit": "c23e5a3f0e7b2a9de889cee024f293f3e07f9087689d16f3c649b23fff7d39ec",
    "v17_closeout": "09351ebb883e5aad18d33353be877ed542f1ed7741e1d03b0c9b4368b39e268b",
    "v17_verification": "d781daea3138595989156935367b15a9e8db543aa464bad0a2c58636db95d596",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "training_environment": "12a26b1edfe9a3435153bf8dd9355e1cd5c39f1ae3653803e59b4f1f03bfb880",
    "formal_environment": "cf403a343db26e09f017d69575dceb0f87e99bfb0d931569ed18500991e04fbd",
}
PRIOR_DEVELOPMENT_ENVIRONMENTS = {
    "v8_stage_a": REPRO / "h2oplus_v8_development_environment_stage_a_v1.json",
    "v8_fewshot": REPRO / "h2oplus_v8_fewshot_environment_v1.json",
    "v9_adaptive": REPRO / "h2oplus_v9_adaptive_environment_v2.json",
    "v10_delayed": REPRO / "h2oplus_v10_delayed_environment_v4.json",
    "v11_actor_hold": REPRO / "h2oplus_v11_actor_hold_environment_v2.json",
    "v12_staged_release": REPRO / "h2oplus_v12_staged_release_environment_v2.json",
}


def encode(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        "sha256": sha256_bytes(value),
    }


def load_frozen(name: str) -> tuple[Path, dict]:
    path = INPUTS[name].resolve(strict=True)
    actual = sha256(path)
    if actual != EXPECTED[name]:
        raise ValueError(f"frozen v18 input drifted: {name}: {actual}")
    payload = json.loads(path.read_text())
    if name not in {"training_environment", "formal_environment"} and payload.get(
        "formal_held_out_outcomes_read"
    ) is not False:
        raise ValueError(f"formal outcome boundary drifted in {name}")
    return path, payload


def atomic_group(payloads: list[tuple[Path, dict]]) -> None:
    existing = [str(path) for path, _ in payloads if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite v18 evidence: {existing}")
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
            if partial.exists():
                partial.unlink()


def validate_parent() -> tuple[dict[str, tuple[Path, dict]], dict[str, Any]]:
    frozen = {name: load_frozen(name) for name in INPUTS}
    analysis = frozen["v17_analysis"][1]
    audit = frozen["v17_audit"][1]
    closeout = frozen["v17_closeout"][1]
    verification = frozen["v17_verification"][1]
    data_audit = frozen["data_environment_audit"][1]
    data_verification = frozen["data_environment_verification"][1]
    result = analysis.get("development_scientific_result", {})
    checks = result.get("checks", {})
    comparisons = result.get("control_comparisons", {})
    completion = result.get("completion_aulc_against_controls", {})
    observed = {
        "v17_twenty_mean_primary_at_25000": result.get("mean_primary_at_25000", {}).get(
            "h2oplus-v17-tail-service-reserve-twenty"
        ),
        "v17_thirty_mean_primary_at_25000": result.get("mean_primary_at_25000", {}).get(
            "h2oplus-v17-tail-service-reserve-thirty"
        ),
        "v17_forty_mean_primary_at_25000": result.get("mean_primary_at_25000", {}).get(
            "h2oplus-v17-tail-service-reserve-forty"
        ),
        "v17_thirty_relative_aulc_vs_v11": comparisons.get(
            "h2oplus-v11-actor-hold", {}
        ).get("relative_improvement"),
        "v17_thirty_completion_aulc_delta_vs_h2oplus": completion.get(
            "h2oplus", {}
        ).get("delta"),
        "v17_thirty_completion_aulc_delta_vs_v10": completion.get(
            "h2oplus-v10-delayed-actor", {}
        ).get("delta"),
        "v17_thirty_completion_aulc_delta_vs_sac": completion.get(
            "sac-online", {}
        ).get("delta"),
    }
    expected_observed = {
        "v17_twenty_mean_primary_at_25000": 2407.029112541763,
        "v17_thirty_mean_primary_at_25000": 2412.2338950552207,
        "v17_forty_mean_primary_at_25000": 2420.231054806202,
        "v17_thirty_relative_aulc_vs_v11": 0.009789298652594222,
        "v17_thirty_completion_aulc_delta_vs_h2oplus": -0.0038785184078677926,
        "v17_thirty_completion_aulc_delta_vs_v10": -0.0030710495473807997,
        "v17_thirty_completion_aulc_delta_vs_sac": -0.003786922323087638,
    }
    if (
        checks.get("completion_rate_noninferiority_against_all_controls") is not False
        or checks.get("relative_aulc_improvement_over_all_controls") is not False
        or checks.get("candidate_mean_lower_than_all_controls_at_25000") is not True
        or checks.get("seed_wins_over_all_controls") is not True
        or audit.get("passed") is not True
        or closeout.get("passed") is not True
        or closeout.get("formal_confirmation_eligible") is not False
        or verification.get("passed") is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
        or observed != expected_observed
    ):
        raise ValueError("v17 closeout, diagnosis, or data audit drifted")
    return frozen, observed


def build_environment_payload(created: str) -> dict:
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
                raise ValueError("v18 fresh environment contains duplicate content")
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
        **PRIOR_DEVELOPMENT_ENVIRONMENTS,
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
        raise ValueError(f"v18 environment disjointness failed: {overlaps}")
    return {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-environment-v1",
        "created_utc": created,
        "status": "frozen-before-any-v18-development-outcome",
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
    frozen, observed = validate_parent()
    created = datetime.now(timezone.utc).isoformat()
    environment = build_environment_payload(created)
    authorization = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-authorization-v1",
        "created_utc": created,
        "status": "authorized-before-any-v18-development-outcome",
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
    frozen_v17 = {
        name: descriptor(path)
        for name, (path, _) in frozen.items()
        if name.startswith("v17_")
    }
    data_audit = {
        name: descriptor(path)
        for name, (path, _) in frozen.items()
        if name.startswith("data_environment_")
    }
    rationale = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-rationale-v1",
        "created_utc": created,
        "status": "frozen-before-any-v18-development-outcome",
        "study_stage": "development",
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
        "v17_observed_design_signal": observed,
        "diagnosis": {
            "v17_failed_boundaries": [
                "completion-rate noninferiority against all controls",
                "relative AULC improvement against every control",
            ],
            "uniform_floor_tradeoff": (
                "raising the broad v17 floor from 20 to 40 seconds increased the "
                "mean primary endpoint by 13.201942264439 seconds, so another broad "
                "floor increase is not an acceptable completion repair"
            ),
            "data_environment_validity_factors": [
                "training objective and completion constraint are not identical",
                "v14-v17 reused the same nine development cells",
                "training and evaluation SUMO runtime versions differ",
            ],
            "mitigation_in_v18": (
                "use three fresh disjoint development blocks and limit the extra hold "
                "to empty tail buses, where direct onboard holding cost is zero"
            ),
        },
        "fixed_design": {
            "parent": "sealed v17 twenty-second 25k behavior",
            "predesignated_candidate": "empty-forty",
            "diagnostics": ["baseline", "one-seat-forty"],
            "activation_fraction": [
                ACTIVATION_START_FRACTION,
                ACTIVATION_END_FRACTION,
            ],
            "empty_reserve_floor_seconds": EMPTY_RESERVE_FLOOR_SECONDS,
            "candidate_max_onboard_capacity_fraction": 0.0,
            "diagnostic_max_onboard_capacity_fraction": 0.02,
            "requires_forward_bus_present": True,
            "requires_backward_bus_absent": True,
            "requires_positive_eligible_arrival_rate": True,
            "new_training_runs": 0,
            "new_remote_development_episodes": 135,
        },
        "claim_boundary": {
            "screen_can_authorize_full_control_matrix_only": True,
            "screen_can_establish_formal_confirmation_eligibility": False,
            "screen_can_establish_formal_held_out_effect": False,
        },
        "authorization": authorization_desc,
        "frozen_v17_evidence": frozen_v17,
        "data_environment_audit": data_audit,
    }
    rationale_desc = future_descriptor(RATIONALE_OUTPUT, rationale)
    registration = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-registration-v1",
        "created_utc": created,
        "status": "registered-before-any-v18-development-outcome",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "candidate": {
            "algorithm": "h2oplus-v18-empty-tail-reserve-empty-forty",
            "empty_reserve_floor_seconds": EMPTY_RESERVE_FLOOR_SECONDS,
            "max_onboard_capacity_fraction": 0.0,
        },
        "diagnostics_not_eligible_for_selection": [
            {"algorithm": "h2oplus-v18-empty-tail-reserve-baseline"},
            {
                "algorithm": "h2oplus-v18-empty-tail-reserve-one-seat-forty",
                "max_onboard_capacity_fraction": 0.02,
            },
        ],
        "selection_rules": {
            "candidate_selection_after_v18_outcomes": False,
            "diagnostics_cannot_replace_candidate": True,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "fresh_environment_required": True,
        },
        "episode_accounting": {
            "physical_methods": 15,
            "development_cells_per_method": 9,
            "new_remote_sumo_episodes": 135,
            "reused_v17_result_episodes": 0,
        },
        "authorization": authorization_desc,
        "rationale": rationale_desc,
        "environment": future_descriptor(ENVIRONMENT_OUTPUT, environment),
        "new_training_runs": 0,
        "parent_training_artifacts": "sealed v14 terminal-quarter 25k actors",
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
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
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-protocol-v1",
        "created_utc": created,
        "status": "frozen-before-any-v18-development-outcome",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
        "registration": registration_desc,
        "authorization": authorization_desc,
        "rationale": rationale_desc,
        "parent_v17_protocol": descriptor(frozen["v17_protocol"][0]),
        "empty_tail_reserve": {
            "parent_algorithm": "h2oplus-v17-tail-service-reserve-twenty",
            "parent_actor_algorithm": "h2oplus-v14-terminal-quarter",
            "parent_tail_service_floor_seconds": PARENT_TAIL_SERVICE_FLOOR_SECONDS,
            "parent_activation_fraction": [
                PARENT_ACTIVATION_START_FRACTION,
                PARENT_ACTIVATION_END_FRACTION,
            ],
            "parent_max_onboard_capacity_fraction": (
                PARENT_MAX_ONBOARD_CAPACITY_FRACTION
            ),
            "predesignated_candidate": "empty-forty",
            "diagnostics": ["baseline", "one-seat-forty"],
            "empty_reserve_floor_seconds": EMPTY_RESERVE_FLOOR_SECONDS,
            "activation_fraction": [
                ACTIVATION_START_FRACTION,
                ACTIVATION_END_FRACTION,
            ],
            "candidate_max_onboard_capacity_fraction": 0.0,
            "diagnostic_max_onboard_capacity_fraction": 0.02,
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
            "candidate": "h2oplus-v18-empty-tail-reserve-empty-forty",
            "paired_baseline": "h2oplus-v18-empty-tail-reserve-baseline",
            "diagnostic": "h2oplus-v18-empty-tail-reserve-one-seat-forty",
            "endpoint_only": True,
            "full_control_matrix_deferred": True,
            "formal_eligibility_assessment_forbidden": True,
        },
        "analysis_gates": gates,
        "environment": future_descriptor(ENVIRONMENT_OUTPUT, environment),
        "frozen_v17_evidence": frozen_v17,
        "data_environment_audit": data_audit,
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
            algorithm = f"h2oplus-v18-empty-tail-reserve-{variant}"
            name = f"{algorithm}-seed{seed}-events000025000"
            key = (
                seed,
                parent["actor_tensor_sha256"],
                variant,
                registered["enabled"],
                registered["max_onboard_fraction"],
            )
            if key in behavior_keys:
                raise ValueError(f"duplicate v18 physical behavior: {key}")
            behavior_keys.add(key)
            row = dict(parent)
            row.update(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "variant": variant,
                    "checkpoint": os.path.relpath(checkpoint, METHODS_OUTPUT.parent),
                    "artifact_lineage": "v18-empty-tail-reserve",
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
                    "parent_activation_end_fraction": (
                        PARENT_ACTIVATION_END_FRACTION
                    ),
                    "parent_max_onboard_capacity_fraction": (
                        PARENT_MAX_ONBOARD_CAPACITY_FRACTION
                    ),
                    "empty_reserve_enabled": registered["enabled"],
                    "empty_reserve_floor_seconds": EMPTY_RESERVE_FLOOR_SECONDS,
                    "activation_start_fraction": ACTIVATION_START_FRACTION,
                    "activation_end_fraction": ACTIVATION_END_FRACTION,
                    "max_onboard_capacity_fraction": registered[
                        "max_onboard_fraction"
                    ],
                    "predesignated_candidate": variant == "empty-forty",
                    "source_protocol_sha256": parent["source_protocol_sha256"],
                    "v17_protocol_sha256": EXPECTED["v17_protocol"],
                    "v18_protocol_sha256": protocol_desc["sha256"],
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
                    "empty_reserve_enabled": registered["enabled"],
                    "empty_reserve_floor_seconds": EMPTY_RESERVE_FLOOR_SECONDS,
                    "activation_start_fraction": ACTIVATION_START_FRACTION,
                    "activation_end_fraction": ACTIVATION_END_FRACTION,
                    "max_onboard_capacity_fraction": registered[
                        "max_onboard_fraction"
                    ],
                    "predesignated_candidate": variant == "empty-forty",
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
                "empty_tail_reserve_physical_behaviors": seed_rows,
            }
        )
    methods_payload = {
        "schema_version": METHODS_SCHEMA,
        "created_utc": created,
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
        "selection_rule": SELECTION_RULE,
        "predesignated_candidate": "h2oplus-v18-empty-tail-reserve-empty-forty",
        "diagnostic_variants": [
            "h2oplus-v18-empty-tail-reserve-baseline",
            "h2oplus-v18-empty-tail-reserve-one-seat-forty",
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
        "v18_protocol_sha256": protocol_desc["sha256"],
        "bindings": {
            "authorization": authorization_desc,
            "rationale": rationale_desc,
            "registration": registration_desc,
            "protocol": protocol_desc,
            "environment": future_descriptor(
                ENVIRONMENT_OUTPUT, environment, METHODS_OUTPUT
            ),
            "v17_methods": descriptor(
                frozen["v17_methods"][0], METHODS_OUTPUT
            ),
            "v17_analysis": descriptor(
                frozen["v17_analysis"][0], METHODS_OUTPUT
            ),
            "v17_closeout": descriptor(
                frozen["v17_closeout"][0], METHODS_OUTPUT
            ),
            "v17_verification": descriptor(
                frozen["v17_verification"][0], METHODS_OUTPUT
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
