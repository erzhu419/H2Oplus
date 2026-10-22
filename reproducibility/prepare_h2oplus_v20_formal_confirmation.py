#!/usr/bin/env python3
"""Prepare and freeze v20 formal confirmation without running or submitting it."""

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
from reproducibility.run_evaluation_matrix_v20_formal_confirmation import (  # noqa: E402
    ALGORITHMS,
    CANDIDATE,
    DIRECT_PARENT,
    EXPECTED_ROLES,
    LINEAGE_CONTROLS,
    METHODS_SCHEMA,
    RAW_PARENT,
    SEEDS,
    SELECTION_RULE,
)


USER_AUTHORIZATION_TEXT = "可以 那就继续"
BLOCKS = tuple(
    {
        "name": f"v20-formal-confirmation-{index:02d}",
        "sumo_seed": 8100 + index,
        "demand_realization_seed": 8200 + index,
    }
    for index in range(1, 11)
)
DEMAND_SCALES = (0.75, 1.0, 1.25)
DEPART_JITTER_SECONDS = 120.0
SCENARIO_ROOT = ROOT / "bus_h2o/formal_confirmation_scenarios_v20/v1"

AUTHORIZATION_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json"
RATIONALE_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_rationale_v1.json"
REGISTRATION_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_registration_v1.json"
PROTOCOL_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_protocol_v1.json"
METHODS_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json"
ENVIRONMENT_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_environment_v1.json"

INPUTS = {
    "v20_analysis": REPRO / "h2oplus_v20_full_control_results/full_control_analysis_v1.json",
    "v20_audit": REPRO / "h2oplus_v20_full_control_results/evaluation_audit_v1.json",
    "v20_closeout": REPRO / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
    "v20_closeout_verification": REPRO
    / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
    "v20_manifest_verification": REPRO
    / "h2oplus_v20_full_control_results/evaluation_file_manifest_verification_v1.json",
    "v20_methods": REPRO / "h2oplus_v20_full_control_evaluation_methods_v1.json",
    "v19_methods": REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json",
    "v19_methods": REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json",
    "v14_methods": REPRO / "h2oplus_v14_terminal_taper_evaluation_methods_v1.json",
    "data_environment_audit": REPRO
    / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPRO
    / "h2oplus_v17_data_environment_convergence_verification_v1.json",
    "training_environment": REPRO / "training_environment_design_v7_formal_v1.json",
    "prior_formal_environment": REPRO / "evaluation_environment_design_v7_formal_v2.json",
    "minimal_sync_policy": REPRO / "h2oplus_minimal_remote_sync_policy_v1.json",
    "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
    "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
}
EXPECTED = {
    "v20_analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "v20_closeout_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "v20_manifest_verification": "6f50f7de8b87519349d48805e2ac7f98a6ac7d9ed2df989378fd2448a95c5cb0",
    "v20_methods": "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9",
    "v19_methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "v14_methods": "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "training_environment": "12a26b1edfe9a3435153bf8dd9355e1cd5c39f1ae3653803e59b4f1f03bfb880",
    "prior_formal_environment": "cf403a343db26e09f017d69575dceb0f87e99bfb0d931569ed18500991e04fbd",
    "minimal_sync_policy": "20b313b7b8cc13a0a8df7c2a137a4b0713c621a33ef8e2e314fb69d03de1a531",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
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
    "v19": (
        REPRO / "h2oplus_v19_completion_safety_environment_v1.json",
        "5cfb1318418b64e619d4be727c41a0ce0eccab8c096020482586d8286cee3a05",
    ),
    "v20": (
        REPRO / "h2oplus_v20_full_control_environment_v1.json",
        "7210826548c94aaa8fbbbb73b62eec5450fec131f2ae3dd1d1dbc93922dff2ba",
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


def future_descriptor(path: Path, payload: dict, owner: Path | None = None) -> dict:
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


def atomic_group(payloads: list[tuple[Path, dict]]) -> None:
    existing = [str(path) for path, _ in payloads if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite v20 formal-confirmation evidence: {existing}"
        )
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


def load_inputs() -> dict[str, tuple[Path, dict]]:
    frozen = {}
    for name, path in INPUTS.items():
        resolved = path.resolve(strict=True)
        actual = sha256(resolved)
        if actual != EXPECTED[name]:
            raise ValueError(f"frozen formal-confirmation input drifted: {name}: {actual}")
        payload = {} if name == "dataset" else json.loads(resolved.read_text())
        frozen[name] = (resolved, payload)
    closeout = frozen["v20_closeout"][1]
    analysis = frozen["v20_analysis"][1]
    if (
        closeout.get("passed") is not True
        or closeout.get("formal_confirmation_eligible") is not True
        or closeout.get("formal_confirmation_authorized_or_submitted") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or analysis.get("development_scientific_result", {}).get(
            "full_development_control_matrix_passed"
        )
        is not True
        or frozen["v20_audit"][1].get("passed") is not True
        or frozen["v20_closeout_verification"][1].get("passed") is not True
        or frozen["v20_manifest_verification"][1].get("passed") is not True
        or frozen["data_environment_audit"][1].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or frozen["data_environment_verification"][1].get("passed") is not True
    ):
        raise ValueError("v20 eligibility or data/environment boundary drifted")
    return frozen


def build_environment(created: str) -> dict:
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
    blocks, scenarios = [], []
    manifest_hashes, runtime_hashes = set(), set()
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
                raise ValueError(
                    "v20 formal-confirmation environment contains duplicate content"
                )
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
        "prior_used_formal_v7": INPUTS["prior_formal_environment"],
        **{
            name: path
            for name, (path, _) in PRIOR_DEVELOPMENT_ENVIRONMENTS.items()
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
        raise ValueError(
            f"v20 formal-confirmation environment disjointness failed: {overlaps}"
        )
    return {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-environment-v1",
        "created_utc": created,
        "status": "frozen-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation",
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "formal_held_out_design_metadata_generated": True,
        "prior_used_formal_v7_environment_reused": False,
        "blocks": blocks,
        "demand_scales": list(DEMAND_SCALES),
        "depart_jitter_seconds": DEPART_JITTER_SECONDS,
        "environment_cells": len(scenarios),
        "independent_realization_blocks": len(blocks),
        "scenario_descriptors": scenarios,
        "disjointness": {
            "verified_disjoint": True,
            "formal_held_out_outcomes_read": False,
            "formal_confirmation_outcomes_read": False,
            **overlaps,
            **{
                f"{name}_design": descriptor(inventory["path"])
                for name, inventory in inventories.items()
            },
        },
        "generator": descriptor(Path(__file__).resolve()),
        "local_sumo_executed": False,
    }


def row_identity(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_methods(frozen: dict, protocol_sha256: str, created: str) -> dict:
    v20_path, v20_payload = frozen["v20_methods"]
    v14_path, v14_payload = frozen["v14_methods"]
    v19_path, v19_payload = frozen["v19_methods"]
    selected = []
    for source, payload in (("v19", v19_payload), ("v14", v14_payload)):
        for row in payload["methods"]:
            algorithm = row.get("algorithm")
            events = row.get("online_events")
            include = (
                source == "v19" and algorithm in {CANDIDATE, DIRECT_PARENT}
            ) or (
                source == "v14"
                and algorithm in {RAW_PARENT, *LINEAGE_CONTROLS}
                and events == 25_000
            )
            if not include:
                continue
            selected.append(
                {
                    "name": row["name"],
                    "algorithm": algorithm,
                    "training_seed": row["training_seed"],
                    "online_events": 25_000,
                    "role": EXPECTED_ROLES[algorithm],
                    "source_manifest": source,
                    "source_row_sha256": row_identity(row),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "actor_tensor_sha256": row["actor_tensor_sha256"],
                    "predesignated_candidate": algorithm == CANDIDATE,
                }
            )
    pairs = {(row["algorithm"], row["training_seed"]) for row in selected}
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    if len(selected) != 45 or pairs != expected:
        raise ValueError("frozen parent manifests do not provide the v20 9-by-5 grid")
    formal_identity = {
        (
            row["name"],
            row["algorithm"],
            row["training_seed"],
            row["checkpoint_sha256"],
            row["actor_tensor_sha256"],
        )
        for row in selected
    }
    v20_identity = {
        (
            row["name"],
            row["algorithm"],
            row["training_seed"],
            row["checkpoint_sha256"],
            row["actor_tensor_sha256"],
        )
        for row in v20_payload["methods"]
    }
    if formal_identity != v20_identity:
        raise ValueError("formal method grid differs from the verified v20 grid")
    actors = {
        (row["training_seed"], row["actor_tensor_sha256"])
        for row in selected
    }
    return {
        "schema_version": METHODS_SCHEMA,
        "created_utc": created,
        "status": "frozen-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "v20_full_control_methods": descriptor(v20_path, METHODS_OUTPUT),
        "predesignated_candidate": CANDIDATE,
        "direct_behavior_parent": DIRECT_PARENT,
        "raw_actor_parent": RAW_PARENT,
        "independent_lineage_controls": list(LINEAGE_CONTROLS),
        "algorithms": list(ALGORITHMS),
        "training_seeds": list(SEEDS),
        "physical_method_count": 45,
        "logical_method_count": 45,
        "unique_actor_tensor_within_seed_count": len(actors),
        "expected_new_remote_sumo_episodes": 1350,
        "parent_method_manifests": {
            "v19": descriptor(v19_path, METHODS_OUTPUT),
            "v14": descriptor(v14_path, METHODS_OUTPUT),
        },
        "v20_protocol_sha256": protocol_sha256,
        "aliases": [
            {
                "name": row["name"],
                "physical_method": row["name"],
                "algorithm": row["algorithm"],
                "training_seed": row["training_seed"],
                "role": row["role"],
            }
            for row in selected
        ],
        "methods": sorted(selected, key=lambda row: row["name"]),
    }


def main() -> int:
    frozen = load_inputs()
    created = datetime.now(timezone.utc).isoformat()
    environment = build_environment(created)
    authorization = {
        "schema_version": (
            "transit-h2oplus-v20-formal-confirmation-preparation-authorization-v1"
        ),
        "created_utc": created,
        "status": "preparation-authorized-before-any-formal-confirmation-outcome",
        "study_stage": "formal-confirmation-preparation",
        "authorization_channel": "explicit user message in current Codex task",
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode()
        ).hexdigest(),
        "authorization_scope": {
            "formal_confirmation_design_preparation_authorized": True,
            "formal_environment_generation_without_sumo_authorized": True,
            "formal_analysis_source_freeze_authorized": True,
            "formal_confirmation_evaluation_authorized": False,
            "formal_confirmation_submission_authorized": False,
            "formal_held_out_outcome_access_authorized": False,
            "remote_sumo_episode_cap": 0,
            "unique_task_count_cap": 0,
            "new_training_authorized": False,
            "local_training_or_sumo_authorized": False,
            "duplicate_submission_authorized": False,
            "force_dispatch_authorized": False,
            "move_host_authorized": False,
            "checkpoint_selection_by_behavior_authorized": False,
        },
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
    }
    authorization_desc = future_descriptor(AUTHORIZATION_OUTPUT, authorization)
    parent_evidence = {
        name: descriptor(path)
        for name, (path, _) in frozen.items()
        if name.startswith("v20_")
    }
    data_evidence = {
        name: descriptor(frozen[name][0])
        for name in ("data_environment_audit", "data_environment_verification")
    }
    rationale = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-rationale-v1",
        "created_utc": created,
        "status": "frozen-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation",
        "prior_v14_through_v20_development_outcomes_used_for_design": True,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "purpose": (
            "perform one independently blocked confirmation of the fixed v19 "
            "completion-safety candidate after the verified v20 development matrix"
        ),
        "design_decisions": {
            "candidate_fixed_from_verified_v20_closeout": CANDIDATE,
            "direct_parent_fixed": DIRECT_PARENT,
            "raw_actor_parent_fixed": RAW_PARENT,
            "lineage_controls_fixed": list(LINEAGE_CONTROLS),
            "endpoint_only": True,
            "confirmatory_comparison": "candidate-vs-direct-behavior-parent",
            "supportive_controls_do_not_expand_confirmatory_claim": True,
            "prior_used_v7_formal_environment_excluded": True,
            "fresh_environment_blocks": 10,
            "fresh_environment_cells": 30,
            "demand_scales": list(DEMAND_SCALES),
            "new_training_runs": 0,
            "planned_remote_formal_episodes": 1350,
            "planned_remote_submission_authorized": False,
        },
        "material_validity_factors": [
            "the optimized reward and completion constraint remain distinct objectives",
            "3,284,718 transitions represent only 30 independent training environment cells",
            "large positive holds remain sparse in the offline dataset",
            "training and evaluation retain a SUMO runtime-version shift",
            "v14-v20 development outcomes adaptively informed candidate selection",
        ],
        "claim_boundary": {
            "this_preparation_creates_formal_effect_evidence": False,
            "this_preparation_authorizes_submission": False,
            "formal_effect_requires_separately_authorized_execution_and_frozen_analysis": True,
            "failed_confirmation_may_not_be_used_for_further_held_out_tuning": True,
        },
        "authorization": authorization_desc,
        "parent_v20_evidence": parent_evidence,
        "data_environment_evidence": data_evidence,
    }
    rationale_desc = future_descriptor(RATIONALE_OUTPUT, rationale)
    registration = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-registration-v1",
        "created_utc": created,
        "status": "registered-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "candidate": CANDIDATE,
        "direct_behavior_parent": DIRECT_PARENT,
        "raw_actor_parent": RAW_PARENT,
        "independent_lineage_controls": list(LINEAGE_CONTROLS),
        "selection_rules": {
            "candidate_replacement_after_registration": False,
            "checkpoint_selection_by_behavior": False,
            "seed_or_scenario_specific_transform_forbidden": True,
            "fresh_environment_required": True,
            "supportive_control_can_replace_candidate": False,
            "formal_outcome_may_inform_future_tuning": False,
        },
        "episode_accounting": {
            "physical_methods": 45,
            "formal_cells_per_method": 30,
            "planned_new_remote_sumo_episodes": 1350,
            "reused_prior_result_episodes": 0,
        },
        "authorization": authorization_desc,
        "rationale": rationale_desc,
        "environment": future_descriptor(ENVIRONMENT_OUTPUT, environment),
        "parent_v20_closeout": descriptor(frozen["v20_closeout"][0]),
        "minimal_remote_sync_policy": descriptor(frozen["minimal_sync_policy"][0]),
        "new_training_runs": 0,
        "formal_confirmation_evaluation_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
    }
    registration_desc = future_descriptor(REGISTRATION_OUTPUT, registration)
    gates = {
        "all_1350_results_valid": True,
        "all_ten_135_job_matrix_manifests_complete": True,
        "parent_artifact_integrity_passed": True,
        "parent_training_stability_passed": True,
        "candidate_mean_primary_change_vs_direct_parent_less_than_seconds": 0.0,
        "candidate_primary_block_wins_vs_direct_parent_at_least": 9,
        "candidate_completion_noninferiority_margin_vs_direct_parent": 0.003,
        "candidate_completion_noninferiority_blocks_vs_direct_parent_at_least": 9,
        "candidate_primary_better_at_each_demand_scale": True,
        "candidate_completion_noninferior_at_each_demand_scale": True,
        "candidate_mean_primary_lower_than_every_comparison_control": True,
        "candidate_completion_noninferiority_margin_per_control": 0.003,
        "candidate_primary_block_wins_vs_sac_at_least": 9,
    }
    protocol = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-protocol-v1",
        "created_utc": created,
        "status": "frozen-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "registration": registration_desc,
        "authorization": authorization_desc,
        "rationale": rationale_desc,
        "parent_v20_closeout": descriptor(frozen["v20_closeout"][0]),
        "evaluation": {
            "training_seeds": list(SEEDS),
            "formal_confirmation_blocks": [block["name"] for block in BLOCKS],
            "sumo_seeds": [block["sumo_seed"] for block in BLOCKS],
            "demand_realization_seeds": [
                block["demand_realization_seed"] for block in BLOCKS
            ],
            "demand_scales": list(DEMAND_SCALES),
            "physical_methods": 45,
            "planned_new_remote_sumo_episodes": 1350,
            "jobs_per_block": 135,
            "workers": 12,
            "planned_host": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
            "submission_authorized": False,
        },
        "analysis": {
            "candidate": CANDIDATE,
            "direct_behavior_parent": DIRECT_PARENT,
            "comparison_controls": [RAW_PARENT, *LINEAGE_CONTROLS],
            "sac_robustness_control": "sac-online",
            "endpoint_only": True,
            "primary_endpoint": (
                "passenger_welfare."
                "accrued_generalized_seconds_per_departed_passenger"
            ),
            "primary_direction": "lower-is-better",
            "completion_endpoint": "passengers.completion_rate",
            "independent_analysis_unit": "formal-environment-block",
            "within_block_average_over_training_seeds_and_demand_scales": True,
            "one_sided_alpha": 0.025,
            "primary_test": "exact-paired-block-sign-test",
            "completion_test": "exact-paired-block-sign-test-with-margin",
            "ties_count_as_nonwins": True,
            "minimum_block_wins": 9,
            "nine_of_ten_one_sided_sign_pvalue": 0.0107421875,
            "eight_of_ten_one_sided_sign_pvalue": 0.0546875,
            "multiplicity": (
                "intersection-union: primary superiority and completion "
                "noninferiority must both pass; supportive controls do not "
                "expand the confirmatory claim"
            ),
            "aulc_claim_forbidden": True,
            "candidate_replacement_forbidden": True,
            "formal_effect_claim_limited_to_direct_parent": True,
            "submission_authorized": False,
        },
        "sample_size_justification": {
            "independent_blocks": 10,
            "paired_observations_per_block": 15,
            "reason": (
                "ten blocks make nine directional wins sufficient at one-sided "
                "alpha 0.025 under the preregistered exact sign test"
            ),
        },
        "analysis_gates": gates,
        "environment": future_descriptor(ENVIRONMENT_OUTPUT, environment),
        "parent_v20_evidence": parent_evidence,
        "data_environment_evidence": data_evidence,
        "dataset": descriptor(frozen["dataset"][0]),
        "sumo_runtime_lock": descriptor(frozen["runtime_lock"][0]),
        "minimal_remote_sync_policy": descriptor(frozen["minimal_sync_policy"][0]),
    }
    protocol_desc = future_descriptor(PROTOCOL_OUTPUT, protocol)
    methods = build_methods(frozen, protocol_desc["sha256"], created)
    atomic_group(
        [
            (AUTHORIZATION_OUTPUT, authorization),
            (RATIONALE_OUTPUT, rationale),
            (REGISTRATION_OUTPUT, registration),
            (PROTOCOL_OUTPUT, protocol),
            (METHODS_OUTPUT, methods),
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
                "physical_methods": 45,
                "fresh_environment_cells": 30,
                "planned_remote_sumo_episodes": 1350,
                "formal_held_out_outcomes_read": False,
                "formal_confirmation_submission_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
