#!/usr/bin/env python3
"""Register, freeze and package the H2O+ v11 actor-hold pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import load_validated_scenario, sha256  # noqa: E402
from reproducibility.prepare_h2oplus_v8_development_stage_a import (  # noqa: E402
    _design_inventory,
    _runtime_key,
)


RATIONALE_SCHEMA = "transit-h2oplus-v11-actor-hold-rationale-v1"
CORRECTION_SCHEMA = "transit-h2oplus-v11-actor-hold-correction-v2"
REGISTRATION_SCHEMA = "transit-h2oplus-v11-actor-hold-registration-v2"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v11-actor-hold-authorization-v2"
ENVIRONMENT_SCHEMA = "transit-h2oplus-v11-actor-hold-environment-v2"
PLAN_SCHEMA = "transit-h2oplus-v11-actor-hold-training-plan-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v11-actor-hold-protocol-v2"
ADMISSION_SCHEMA = "transit-training-admission-v11-actor-hold-development-v2"
FREEZE_SCHEMA = "transit-h2oplus-v11-actor-hold-freeze-v2"
DEPLOYMENT_SCHEMA = "transit-h2oplus-v11-actor-hold-deployment-v2"
USER_AUTHORIZATION_TEXT = "那就继续改善"
AUTHORIZED_PLAN_TEXT = (
    "保持 5k collection warmup；critic、value 和 ratio diagnostics 从 5k 后继续更新，"
    "actor 与 temperature 严格冻结 5000 次 online critic updates，对齐名义 10k milestone；"
    "释放后仅使用 online states 更新 actor。"
    "离线 replay 固定 0.5，保持 v7 actor-before-critic 顺序和 unit ratio weights。"
    "使用五个已注册训练种子与新的开发场景，只做开发筛选，不读取正式 held-out。"
)
TRAINING_SEEDS = (6201, 6202, 6203, 6204, 6205)
VARIANT = "actor-hold-online-only"
VARIANTS = (VARIANT,)
BLOCKS = (
    {
        "name": f"v11-actor-hold-development-{index:02d}",
        "sumo_seed": 4900 + index,
        "demand_realization_seed": 5000 + index,
    }
    for index in range(1, 4)
)
BLOCKS = tuple(BLOCKS)
DEMAND_SCALES = (0.85, 1.0, 1.15)
DEPART_JITTER_SECONDS = 120.0
COMMON_EVALUATION_EVENTS = (0, 5_000, 10_000, 25_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-workspace",
        type=Path,
        default=REPRO / "deployments/v7_formal_v2_training_v7r1/workspace",
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v11_actor_hold_pilot_v2",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o/development_scenarios_v11/actor_hold_v1",
    )
    parser.add_argument(
        "--rationale-record",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_rationale_v1.json",
    )
    parser.add_argument(
        "--correction-record",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_correction_v2.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            os.path.relpath(resolved, base.resolve()) if base else str(resolved)
        ),
        "sha256": sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_stable(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    time_keys = {"created_utc", "registered_utc", "approved_utc"}
    if destination.exists():
        previous = json.loads(destination.read_text())
        old = {key: value for key, value in previous.items() if key not in time_keys}
        new = {key: value for key, value in payload.items() if key not in time_keys}
        if old != new and not force:
            raise FileExistsError(f"refusing to replace different artifact: {destination}")
        for key in time_keys:
            if isinstance(previous.get(key), str):
                payload[key] = previous[key]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _source_map() -> dict[str, str]:
    paths = (
        ROOT / "SimpleSAC/train_transit_v11_actor_hold.py",
        ROOT / "SimpleSAC/transit_v10/__init__.py",
        ROOT / "SimpleSAC/transit_v10/provenance.py",
        ROOT / "SimpleSAC/transit_v10/trainer.py",
        ROOT / "SimpleSAC/transit_v11/__init__.py",
        ROOT / "SimpleSAC/transit_v11/provenance.py",
        ROOT / "SimpleSAC/transit_v11/trainer.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def _build_correction(path: Path, *, force: bool) -> Path:
    v1_root = REPRO / "deployments/h2oplus_v11_actor_hold_pilot_v1"
    v1_manifest = v1_root / "workspace/deployment_manifest_v11.json"
    v1_local = REPRO / "h2oplus_v11_actor_hold_deployment_verification_local_v1.json"
    v1_submission = REPRO / "h2oplus_v11_actor_hold_preflight_submission_v1.json"
    v1_closeout = REPRO / "h2oplus_v11_actor_hold_preflight_closeout_v1.json"
    v1_rationale = REPRO / "h2oplus_v11_actor_hold_rationale_v1.json"
    expected_sha256 = {
        v1_manifest: "9239fa4526866efab89dd03519d646773caacdd54c59cf9e5cd34f4c0da73ae2",
        v1_local: "8596e8aa7b3f5e4dccd2b176b682b52f43ac67c16edc0e1fc75518b3ea289145",
        v1_submission: "c8500cb06b6f160d91f008594f7cf802b84db719cdb1b53e18f4fb8b5b83aa72",
        v1_closeout: "6151f95d07700daca533643a854c54808a906d4bd9da1541d22e0829f4dda01d",
        v1_rationale: "456ddcb73c5d78a8f0bf90f3e8ddfbbfda81c8c45c76d505f4d309eead35605b",
    }
    for artifact, expected in expected_sha256.items():
        if sha256(artifact.resolve(strict=True)) != expected:
            raise ValueError(f"v11 v1 incident evidence drifted: {artifact}")

    manifest_payload = json.loads(v1_manifest.read_text())
    local_payload = json.loads(v1_local.read_text())
    submission_payload = json.loads(v1_submission.read_text())
    closeout_payload = json.loads(v1_closeout.read_text())
    submitted_ids = [row.get("task_id") for row in submission_payload.get("tasks", [])]
    closed_ids = [row.get("task_id") for row in closeout_payload.get("valid_tasks", [])]
    if (
        manifest_payload.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-deployment-v1"
        or manifest_payload.get("overlay_entry_count") != 25
        or local_payload.get("passed") is not True
        or local_payload.get("local_training_or_sumo_executed") is not False
        or submission_payload.get("status") != "submitted"
        or submission_payload.get("remote_training_or_sumo_executed") is not False
        or closeout_payload.get("status") != "passed"
        or closeout_payload.get("local_training_or_sumo_executed") is not False
        or submitted_ids != ["t82894", "t82895"]
        or closed_ids != ["t82894", "t82895"]
        or closeout_payload.get("exact_remote_verifications") != 2
        or (REPRO / "h2oplus_v11_actor_hold_submission_v1.json").exists()
    ):
        raise ValueError("v11 v1 preflight incident contract is incomplete")

    old_sources = {
        entry["path"]: entry["sha256"]
        for entry in manifest_payload["entries"]
        if entry["path"] in _source_map()
    }
    new_sources = _source_map()
    if set(old_sources) != set(new_sources):
        raise ValueError("v11 v1 and corrected runtime source sets differ")
    replacements = [
        {
            "path": source,
            "v1_sha256": old_sources[source],
            "v2_sha256": new_sources[source],
        }
        for source in sorted(new_sources)
        if old_sources[source] != new_sources[source]
    ]
    if {row["path"] for row in replacements} != {
        "SimpleSAC/train_transit_v11_actor_hold.py",
        "SimpleSAC/transit_v11/__init__.py",
        "SimpleSAC/transit_v11/trainer.py",
    }:
        raise ValueError("unexpected v11 v2 runtime source replacement set")

    payload = {
        "schema_version": CORRECTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "corrected-before-any-v11-training-or-sumo-execution",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v1_training_or_sumo_executed": False,
        "v1_training_submission_record_exists": False,
        "defect": {
            "class": "fail-closed-nominal-milestone-boundary-mismatch",
            "description": (
                "The v1 actor release predicate used replay_buffer.online_size. "
                "An atomic simultaneous-decision batch may put actual replay size "
                "above 10000 before the nominal 10k export, permitting an actor and "
                "temperature update before the required 0/5k/10k hash equality check."
            ),
            "observed_scientific_outcome": False,
            "detected_during_no_sumo_preflight_closeout": True,
        },
        "correction": {
            "release_predicate": "online_critic_updates >= 5000 after 5k warmup",
            "nominal_release_milestone": 10_000,
            "atomic_event_overshoot_cannot_release_actor_early": True,
            "actor_and_temperature_state_source_uses_same_release_predicate": True,
            "zero_5k_10k_actor_and_temperature_hash_equality_remains_fail_closed": True,
        },
        "v1_runtime_source_sha256": old_sources,
        "corrected_runtime_source_sha256": new_sources,
        "exact_runtime_source_replacements": replacements,
        "superseded_v1_lineage": {
            "deployment_manifest": descriptor(v1_manifest),
            "local_verification": descriptor(v1_local),
            "preflight_submission": descriptor(v1_submission),
            "preflight_closeout": descriptor(v1_closeout),
            "preflight_task_ids": submitted_ids,
            "training_submission_gate_open_in_v1_closeout": True,
            "eligible_to_authorize_training_after_correction": False,
            "preserved_as_no_training_no_sumo_operational_evidence_only": True,
        },
        "design_rationale": descriptor(v1_rationale),
        "required_next_lineage": {
            "fresh_deployment": "v2",
            "fresh_local_and_two_host_no_sumo_verification": True,
            "fresh_training_submission_record": True,
            "reuse_or_overwrite_v1_result_roots": False,
        },
        "formal_confirmation_stage_authorized_or_submitted": False,
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(path, payload, force=force)
    return path


def _build_rationale(path: Path, *, force: bool) -> Path:
    analysis_path = REPRO / "h2oplus_v10_delayed_results/delayed_analysis_v1.json"
    analysis = json.loads(analysis_path.read_text())
    result = analysis["development_scientific_result"]
    attribution = {
        row["candidate"]: row
        for row in result["single_factor_attribution"]["contrasts"]
    }
    expected_aulc = {
        "h2oplus": 2546.7543397748477,
        "h2oplus-v10-delayed-actor": 2502.5313077748096,
        "h2oplus-v10-delayed-combined": 2528.864053385743,
        "h2oplus-v10-delayed-order": 2528.198083605848,
        "h2oplus-v10-delayed-replay": 2513.3206576040984,
        "h2oplus-v10-unit-control": 2508.8346131194503,
        "sac-online": 2523.0267568393942,
    }
    if (
        analysis["artifact_integrity"].get("passed") is not True
        or analysis["artifact_integrity"].get("physical_evaluation_results")
        != 540
        or analysis["training_scientific_stability"].get("passed") is not False
        or analysis["training_scientific_stability"].get("failure_runs")
        != ["h2oplus_v10c4_delayed_order_seed6205"]
        or result.get("screening_gate_passed") is not False
        or result.get("all_algorithm_mean_aulc") != expected_aulc
        or attribution["h2oplus-v10-delayed-actor"].get(
            "attribution_positive"
        )
        is not True
        or attribution["h2oplus-v10-delayed-replay"].get(
            "attribution_positive"
        )
        is not False
        or attribution["h2oplus-v10-delayed-order"].get(
            "attribution_positive"
        )
        is not False
    ):
        raise ValueError("sealed v10 development evidence drifted")
    payload = {
        "schema_version": RATIONALE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v11-training-or-new-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "v10_artifact_integrity_passed": True,
        "v10_scientific_stability_passed": False,
        "v10_stability_failure_preserved": (
            "delayed-order seed6205 transient hold_probability crossing"
        ),
        "observed_v10_development_evidence": {
            "mean_aulc": expected_aulc,
            "mean_passenger_seconds_per_person_at_10000": result[
                "budget_results"
            ]["10000"]["means"],
            "mean_passenger_seconds_per_person_at_25000": result[
                "budget_results"
            ]["25000"]["means"],
            "delayed_actor_relative_improvement_over_unit_control": (
                attribution["h2oplus-v10-delayed-actor"][
                    "relative_improvement"
                ]
            ),
            "delayed_actor_seed_wins_over_unit_control": attribution[
                "h2oplus-v10-delayed-actor"
            ]["seed_wins"],
            "replay_factor_positive": False,
            "update_order_factor_positive": False,
            "combined_candidate_screening_passed": False,
        },
        "diagnosis": {
            "common_v10_mean_at_5000": 2437.909788934886,
            "common_v10_mean_at_10000": 2593.071494436721,
            "early_5k_to_10k_actor_updates_degraded_mean": True,
            "only_positive_registered_single_factor": "online-only actor states",
            "excluded_factors": [
                "offline replay ramp",
                "critic-before-actor update order",
                "combined v10 intervention",
            ],
        },
        "selected_v11_single_factor": {
            "name": VARIANT,
            "collection_warmup_events": 5_000,
            "critic_value_and_ratio_updates_continue_after_warmup": True,
            "actor_and_temperature_held_through_online_events": 10_000,
            "actor_state_source_strictly_after_10000": "online-only",
            "offline_replay_share": 0.50,
            "online_update_order": "v7-actor-before-critic",
            "unit_ratio_weights": True,
            "checkpoint_selection_by_behavior": False,
        },
        "sealed_inputs": {
            "v10_analysis": descriptor(analysis_path),
            "v10_evaluation_closeout": descriptor(
                REPRO / "h2oplus_v10_delayed_results/evaluation_closeout_v1.json"
            ),
            "v10_evaluation_verification": descriptor(
                REPRO
                / "h2oplus_v10_delayed_results/evaluation_closeout_verification_v1.json"
            ),
            "v10_branch_closeout": descriptor(
                REPRO / "h2oplus_v10_delayed_results/branch_closeout_v4.json"
            ),
            "v10_branch_verification": descriptor(
                REPRO
                / "h2oplus_v10_delayed_results/branch_closeout_verification_v4.json"
            ),
        },
        "claim_boundary": {
            "supports_formal_held_out_effect_claim": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "all_v10_outcomes_preserved": True,
        },
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(path, payload, force=force)
    return path


def _registration(
    base_manifest: Path, rationale: Path, correction: Path
) -> dict[str, Any]:
    v8 = REPRO / "h2oplus_v8_fewshot_results"
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v11 actor-hold single-factor development pilot",
        "status": "registered-before-training-or-new-development-evaluation",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "frozen_v7_v8_v9_v10_trainer_or_evidence_modified": False,
        "design_rationale": descriptor(rationale),
        "v2_boundary_correction": descriptor(correction),
        "authorized_plan_text": AUTHORIZED_PLAN_TEXT,
        "authorized_plan_text_sha256": hashlib.sha256(
            AUTHORIZED_PLAN_TEXT.encode()
        ).hexdigest(),
        "causal_design": {
            "fixed_controls": [
                "sealed v8 h2oplus",
                "sealed v8 sac-online",
                "sealed v10 delayed-actor",
            ],
            "new_variants": list(VARIANTS),
            "isolated_factor": "hold actor and temperature from 5k through 10k",
            "new_training_runs": len(TRAINING_SEEDS),
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_online_events": 25_000,
        },
        "intervention": {
            "warmup_events": 5_000,
            "actor_and_temperature_hold_through_online_events": 10_000,
            "actor_and_temperature_hold_online_critic_updates": 5_000,
            "actor_state_source_after_release": "online-only",
            "offline_replay_share_knots": [[0, 0.50], [25_000, 0.50]],
            "online_update_order": "v7-actor-before-critic",
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "zero_5k_10k_actor_and_temperature_hash_equality_required": True,
        },
        "development_evaluation": {
            "blocks": list(BLOCKS),
            "demand_scales": list(DEMAND_SCALES),
            "common_online_event_budgets": list(COMMON_EVALUATION_EVENTS),
            "logical_policy_aliases": 80,
            "physical_actor_tensors": (
                "determined only by exact actor hashes after all training closes"
            ),
            "maximum_unique_remote_sumo_episodes": 720,
            "submitted_with_training": False,
        },
        "screening_gates": {
            "candidate_relative_aulc_improvement_over_sac_at_least": 0.02,
            "candidate_relative_aulc_improvement_over_fixed_h2o_at_least": 0.03,
            "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least": 0.01,
            "candidate_seed_wins_over_sac_at_least": 4,
            "candidate_seed_wins_over_fixed_h2o_at_least": 4,
            "candidate_seed_wins_over_v10_delayed_actor_at_least": 3,
            "candidate_mean_lower_than_sac_and_fixed_h2o_at_10000_and_25000": True,
            "completion_rate_noninferiority_margin": 0.005,
            "no_demand_scale_relative_aulc_degradation_over_sac_greater_than": 0.01,
            "all_5_training_runs_pass_registered_stability": True,
            "zero_5k_10k_actor_and_temperature_hashes_identical": True,
            "formal_confirmation_eligibility_created": False,
        },
        "interpretation": {
            "five_seed_development_screen_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "formal_stage_authorized_or_submitted": False,
            "passing_all_gates_creates_eligibility_only": True,
            "all_registered_outcomes_retained": True,
        },
        "sources": {
            "v11_source_sha256": _source_map(),
            "deployment_verifier": descriptor(
                REPRO / "verify_h2oplus_v11_actor_hold_deployment.py"
            ),
            "design_rationale": descriptor(rationale),
            "v2_boundary_correction": descriptor(correction),
            "base_v7_training_deployment": descriptor(base_manifest),
            "sealed_v8_training_closeout": descriptor(v8 / "training_closeout.json"),
            "sealed_v8_training_manifest_verification": descriptor(
                v8 / "training_file_manifest_verification.json"
            ),
            "sealed_v8_fewshot_closeout": descriptor(v8 / "fewshot_closeout_v3.json"),
            "sealed_v8_analysis": descriptor(v8 / "fewshot_analysis_v3.json"),
        },
        "stop_rules": [
            "stop on nonfinite state, nonzero exit or source/event/update mismatch",
            "never restart or checkpoint-select because an outcome is poor",
            "never use formal held-out outcomes for development selection",
            "never train or run SUMO locally",
            "do not submit development evaluation until all five training runs close",
            "do not launch a formal confirmation stage from this registration",
        ],
        "generator": descriptor(Path(__file__).resolve()),
    }


def _authorization(
    registration: Path, rationale: Path, correction: Path
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_decision": "authorize-v11-development-training-and-later-development-evaluation",
        "authorization_channel": "explicit user message in current Codex task",
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode()
        ).hexdigest(),
        "authorized_plan_text": AUTHORIZED_PLAN_TEXT,
        "authorized_plan_text_sha256": hashlib.sha256(
            AUTHORIZED_PLAN_TEXT.encode()
        ).hexdigest(),
        "registration": descriptor(registration),
        "design_rationale": descriptor(rationale),
        "v2_boundary_correction": descriptor(correction),
        "remote_training_authorized": True,
        "remote_development_evaluation_after_training_authorized": True,
        "local_training_or_sumo_authorized": False,
        "formal_held_out_access_authorized": False,
        "formal_confirmation_stage_authorized": False,
    }


def _build_environment(
    args: argparse.Namespace,
    registration: Path,
    authorization: Path,
    rationale: Path,
    correction: Path,
    *,
    force: bool,
) -> Path:
    destination = REPRO / "h2oplus_v11_actor_hold_environment_v2.json"
    v1_environment_path = REPRO / "h2oplus_v11_actor_hold_environment_v1.json"
    if (
        sha256(v1_environment_path)
        != "4a43e12bee12dbc43ac7aeba05e4a3610739d07afa405f570fb727c884ba29ca"
    ):
        raise ValueError("frozen v11 v1 environment drifted")
    v1_environment = json.loads(v1_environment_path.read_text())
    block_payloads = list(v1_environment.get("blocks", []))
    scenario_payloads = list(v1_environment.get("scenario_descriptors", []))
    if (
        len(block_payloads) != 3
        or len(scenario_payloads) != 9
        or tuple(
            {
                "name": row["name"],
                "sumo_seed": row["sumo_seed"],
                "demand_realization_seed": row["demand_realization_seed"],
            }
            for row in block_payloads
        )
        != BLOCKS
    ):
        raise ValueError("frozen v11 v1 environment matrix drifted")
    scenario_root = args.scenario_root.expanduser().resolve(strict=True)
    manifest_hashes: set[str] = set()
    runtime_hashes: set[str] = set()
    for row in scenario_payloads:
        manifest = (v1_environment_path.parent / row["manifest"]).resolve(strict=True)
        if scenario_root not in manifest.parents or sha256(manifest) != row["manifest_sha256"]:
            raise ValueError(f"frozen v11 scenario manifest drifted: {manifest}")
        _, evidence = load_validated_scenario(manifest)
        runtime_hash = _runtime_key(evidence)
        if runtime_hash != row["runtime_content_sha256"]:
            raise ValueError(f"frozen v11 scenario runtime drifted: {manifest}")
        if row["manifest_sha256"] in manifest_hashes or runtime_hash in runtime_hashes:
            raise ValueError("v11 development scenario content is duplicated")
        manifest_hashes.add(row["manifest_sha256"])
        runtime_hashes.add(runtime_hash)

    inventories = {
        "training": _design_inventory(REPRO / "training_environment_design_v7_formal_v1.json"),
        "formal_held_out": _design_inventory(REPRO / "evaluation_environment_design_v7_formal_v2.json"),
        "v8_stage_a": _design_inventory(REPRO / "h2oplus_v8_development_environment_stage_a_v1.json"),
        "v8_fewshot": _design_inventory(REPRO / "h2oplus_v8_fewshot_environment_v1.json"),
        "v9_adaptive": _design_inventory(
            REPRO / "h2oplus_v9_adaptive_environment_v2.json"
        ),
        "v10_delayed": _design_inventory(
            REPRO / "h2oplus_v10_delayed_environment_v4.json"
        ),
    }
    demand_seeds = {block["demand_realization_seed"] for block in BLOCKS}
    sumo_seeds = {block["sumo_seed"] for block in BLOCKS}
    overlaps: dict[str, list[Any]] = {}
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
        raise ValueError(f"v11 development disjointness failed: {overlaps}")
    v1_scenarios = {
        (
            row["block"],
            float(row["demand_scale"]),
            row["manifest_sha256"],
            row["runtime_content_sha256"],
        )
        for row in v1_environment.get("scenario_descriptors", [])
    }
    v2_scenarios = {
        (
            row["block"],
            float(row["demand_scale"]),
            row["manifest_sha256"],
            row["runtime_content_sha256"],
        )
        for row in scenario_payloads
    }
    if v1_scenarios != v2_scenarios or len(v2_scenarios) != 9:
        raise ValueError("v11 v2 environment does not exactly reuse frozen v1 cells")
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v11-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "blocks": block_payloads,
        "demand_scales": list(DEMAND_SCALES),
        "depart_jitter_seconds": DEPART_JITTER_SECONDS,
        "environment_cells": len(scenario_payloads),
        "scenario_descriptors": scenario_payloads,
        "v1_environment_reused_before_any_training_or_sumo_outcome": descriptor(
            v1_environment_path
        ),
        "disjointness": {
            "verified_disjoint": True,
            "formal_held_out_outcomes_read": False,
            **overlaps,
            **{
                f"{name}_design": descriptor(inventory["path"])
                for name, inventory in inventories.items()
            },
        },
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "design_rationale": descriptor(rationale),
        "v2_boundary_correction": descriptor(correction),
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(destination, payload, force=force)
    return destination


def _initializer_records(workspace: Path) -> dict[int, dict[str, Any]]:
    records = {}
    source_root = REPRO / "h2oplus_v8_fewshot_results/training"
    target_root = workspace / "reproducibility/h2oplus_v11_initializers"
    for seed in TRAINING_SEEDS:
        source = source_root / f"h2oplus_seed{seed}/checkpoints/offline_initializer.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        extra = dict(payload["extra"])
        records[seed] = {
            "source": descriptor(source),
            "path": str(target_root / f"h2oplus_seed{seed}.pt"),
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
            "extra_sha256": stable_object_sha256(extra),
            "source_run_signature": extra["source_run_signature"],
            "source_training_admission": extra["training_admission"],
        }
    return records


def _variant_settings(variant: str) -> dict[str, Any]:
    if variant != VARIANT:
        raise ValueError(f"unknown v11 actor-hold variant: {variant}")
    warmup = 5_000
    exports = [0, 5_000, 10_000, 25_000]
    epochs = (25_000 - warmup) // 500
    return {
        "stage": "candidate",
        "warmup_events": warmup,
        "epochs": epochs,
        "offline_updates": 50_000,
        "expected_total_updates": 50_000 + epochs * 500,
        "online_event_cap": 25_000,
        "export_online_events": exports,
    }


def _milestones(variant: str) -> list[dict[str, int]]:
    settings = _variant_settings(variant)
    result = []
    for events in settings["export_online_events"]:
        online_steps = max(events - settings["warmup_events"], 0)
        ratio_steps = 0 if not online_steps else 1_000 + (online_steps + 3) // 4
        result.append(
            {
                "online_events": events,
                "learner_update_steps": 50_000 + online_steps,
                "ratio_optimizer_steps": ratio_steps,
            }
        )
    return result


def _command_arguments(
    *,
    variant: str,
    seed: int,
    run_name: str,
    run_signature: str,
    admission: Path,
    initializer: dict[str, Any],
    workspace: Path,
    output_root: Path,
) -> list[str]:
    settings = _variant_settings(variant)
    return [
        "--method", "h2oplus",
        "--run-name", run_name,
        "--run-signature", run_signature,
        "--seed", str(seed),
        "--device", "cuda:0",
        "--max-canonical-processes-per-gpu", "1",
        "--cpu-threads", "6",
        "--dataset", str(workspace / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"),
        "--sim-env", str(workspace / "bus_h2o/calibrated_env"),
        "--training-admission", str(admission),
        "--offline-initializer", initializer["path"],
        "--offline-initializer-source-signature", initializer["source_run_signature"],
        "--output-root", str(output_root),
        "--max-offline-rows", "0",
        "--online-capacity", "30_000".replace("_", ""),
        "--offline-updates", "50000",
        "--warmup-events", str(settings["warmup_events"]),
        "--ratio-pretrain-updates", "1000",
        "--epochs", str(settings["epochs"]),
        "--events-per-epoch", "500",
        "--interaction-chunk-events", "16",
        "--updates-per-epoch", "500",
        "--utd-ratio", "1",
        "--batch-size", "256",
        "--reward-scale", "0.0001",
        "--critic-loss", "huber",
        "--huber-delta", "1.0",
        "--ensemble-size", "2",
        "--hidden-dim", "256",
        "--online-fraction", "0.5",
        "--target-entropy", "0.0",
        "--initial-temperature", "1.0",
        "--iql-advantage-temperature", "3.0",
        "--offline-cql-alpha", "0.0",
        "--cql-n-actions", "10",
        "--ratio-update-interval", "4",
        "--offline-checkpoint-every", "10000",
        "--checkpoint-every", "1",
        "--log-every", "250",
        "--resource-interval", "5.0",
        "--device-cache-observation-dtype", "float32",
        "--compile-networks",
        "--compile-mode", "default",
        "--sim-demand-scales", "0.75", "1.0", "1.25",
    ]


def _build_plan(
    *,
    deployment_root: Path,
    workspace: Path,
    registration: Path,
    authorization: Path,
    rationale: Path,
    correction: Path,
    environment: Path,
    initializers: dict[int, dict[str, Any]],
    force: bool,
) -> Path:
    control = workspace / "reproducibility/h2oplus_v11_actor_hold_control"
    output_root = deployment_root / "remote_results"
    identity = {
        "registration_sha256": sha256(registration),
        "authorization_sha256": sha256(authorization),
        "rationale_sha256": sha256(rationale),
        "correction_sha256": sha256(correction),
        "environment_sha256": sha256(environment),
        "v11_source_sha256": _source_map(),
    }
    jobs = []
    for variant in VARIANTS:
        settings = _variant_settings(variant)
        for seed in TRAINING_SEEDS:
            run_name = f"h2oplus_v11a2_{variant.replace('-', '_')}_seed{seed}"
            signature = stable_object_sha256(
                {
                    **identity,
                    "variant": variant,
                    "seed": seed,
                    "settings": settings,
                    "initializer_sha256": initializers[seed]["sha256"],
                }
            )
            admission = control / "admissions" / f"{run_name}.json"
            command = _command_arguments(
                variant=variant,
                seed=seed,
                run_name=run_name,
                run_signature=signature,
                admission=admission,
                initializer=initializers[seed],
                workspace=workspace,
                output_root=output_root,
            )
            jobs.append(
                {
                    "name": run_name,
                    "method": "h2oplus",
                    "variant": variant,
                    "seed": seed,
                    "run_signature": signature,
                    "training_admission": str(admission),
                    "offline_initializer": {
                        key: value
                        for key, value in initializers[seed].items()
                        if key != "source"
                    },
                    "command_arguments": command,
                    **settings,
                    "milestones": _milestones(variant),
                }
            )
    payload = {
        "schema_version": PLAN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v11-training",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "workspace": str(workspace),
        "remote_output_root": str(output_root),
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "design_rationale": descriptor(rationale),
        "v2_boundary_correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "v11_source_sha256": _source_map(),
        "required_environment": {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
            "CPATH": str(workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
            "CUDA_VISIBLE_DEVICES": "scheduler-assigned exactly one physical GPU",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "jobs": jobs,
    }
    path = REPRO / "h2oplus_v11_actor_hold_training_plan_v2.json"
    write_stable(path, payload, force=force)
    return path


def _build_protocol(
    *,
    registration: Path,
    authorization: Path,
    rationale: Path,
    correction: Path,
    environment: Path,
    plan: Path,
    base_manifest: Path,
    force: bool,
) -> Path:
    jobs = json.loads(plan.read_text())["jobs"]
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v11-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "design_rationale": descriptor(rationale),
        "v2_boundary_correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "base_v7_training_deployment": descriptor(base_manifest),
        "v11_source_sha256": _source_map(),
        "training_jobs": jobs,
        "staged_release": {
            "training_jobs": len(jobs),
            "development_evaluation_release_requires": (
                "all five jobs token-bound backend exit0, exact artifacts, "
                "independent audit and training closeout"
            ),
        },
        "policy_budgets": {
            variant: {
                "export_online_events": _variant_settings(variant)["export_online_events"],
                "common_evaluation_events": list(COMMON_EVALUATION_EVENTS),
                "milestones": _milestones(variant),
            }
            for variant in VARIANTS
        },
        "fixed_control": {
            "training_closeout": descriptor(
                REPRO / "h2oplus_v8_fewshot_results/training_closeout.json"
            ),
            "methods_manifest": descriptor(
                REPRO / "h2oplus_v8_fewshot_results/evaluation_methods_v8_fewshot.json"
            ),
            "seeds": list(TRAINING_SEEDS),
            "methods": ["h2oplus", "sac-online"],
            "common_evaluation_events": list(COMMON_EVALUATION_EVENTS),
        },
        "previous_best_development_candidate": {
            "method": "h2oplus-v10-delayed-actor",
            "methods_manifest": descriptor(
                REPRO / "h2oplus_v10_delayed_evaluation_methods_v1.json"
            ),
            "analysis": descriptor(
                REPRO / "h2oplus_v10_delayed_results/delayed_analysis_v1.json"
            ),
        },
        "analysis_gates": json.loads(registration.read_text())["screening_gates"],
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v11_actor_hold_protocol_v2.json"
    write_stable(path, payload, force=force)
    return path


def _build_admissions(
    *,
    formal_workspace: Path,
    workspace: Path,
    plan: Path,
    protocol: Path,
    rationale: Path,
    correction: Path,
    force: bool,
) -> list[Path]:
    formal_admission = formal_workspace / (
        "reproducibility/event_cap_regate_v7_v1_remote/"
        "training_admission_v7_formal_v2_event_cap_fix_v1.json"
    )
    source = json.loads(formal_admission.read_text())
    output = REPRO / "h2oplus_v11_actor_hold_training_admissions_v2"
    paths = []
    for job in json.loads(plan.read_text())["jobs"]:
        payload = {
            "schema_version": ADMISSION_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_stage": "development",
            "passed": True,
            "formal_held_out_outcomes_read": False,
            "method": job["method"],
            "variant": job["variant"],
            "stage": job["stage"],
            "seed": job["seed"],
            "run_name": job["name"],
            "run_signature": job["run_signature"],
            "dataset": source["dataset"],
            "simulator": source["simulator"],
            "training_source_sha256": source["training_source_sha256"],
            "v11_source_sha256": _source_map(),
            "v11_wrapper": descriptor(ROOT / "SimpleSAC/train_transit_v11_actor_hold.py"),
            "v11_protocol": descriptor(protocol),
            "design_rationale": descriptor(rationale),
            "v2_boundary_correction": descriptor(correction),
            "base_v7_admission_source": descriptor(formal_admission),
            "offline_initializer": job["offline_initializer"],
            "selected_arguments": job,
            "checks": {
                "base_v7_source_map_reused_exactly": True,
                "v11_overlay_bound_separately": True,
                "offline_initializer_exactly_bound": True,
                "formal_held_out_outcomes_read": False,
            },
        }
        path = output / f"{job['name']}.json"
        write_stable(path, payload, force=force)
        paths.append(path)
    return paths


def _build_freeze(
    *,
    registration: Path,
    authorization: Path,
    rationale: Path,
    correction: Path,
    environment: Path,
    plan: Path,
    protocol: Path,
    admissions: list[Path],
    force: bool,
) -> Path:
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v11-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "design_rationale": descriptor(rationale),
        "v2_boundary_correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "protocol": descriptor(protocol),
        "training_admissions": [descriptor(path) for path in admissions],
        "v11_source_sha256": _source_map(),
        "new_training_runs": 5,
        "maximum_future_development_sumo_episodes": 720,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v11_actor_hold_freeze_v2.json"
    write_stable(path, payload, force=force)
    return path


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"deployment copy hash mismatch: {destination}")


def _build_deployment(
    *,
    args: argparse.Namespace,
    formal_workspace: Path,
    registration: Path,
    authorization: Path,
    rationale: Path,
    correction: Path,
    environment: Path,
    plan: Path,
    protocol: Path,
    admissions: list[Path],
    freeze: Path,
    initializers: dict[int, dict[str, Any]],
) -> Path:
    deployment_root = args.deployment_root.expanduser().resolve()
    workspace = deployment_root / "workspace"
    if deployment_root.exists():
        if not args.force:
            raise FileExistsError(f"refusing to replace deployment: {deployment_root}")
        shutil.rmtree(deployment_root)
    deployment_root.mkdir(parents=True)
    shutil.copytree(
        formal_workspace,
        workspace,
        copy_function=os.link,
        symlinks=True,
    )
    control = workspace / "reproducibility/h2oplus_v11_actor_hold_control"
    mappings = {
        ROOT / "SimpleSAC/train_transit_v11_actor_hold.py": workspace / "SimpleSAC/train_transit_v11_actor_hold.py",
        ROOT / "SimpleSAC/transit_v10/__init__.py": workspace / "SimpleSAC/transit_v10/__init__.py",
        ROOT / "SimpleSAC/transit_v10/provenance.py": workspace / "SimpleSAC/transit_v10/provenance.py",
        ROOT / "SimpleSAC/transit_v10/trainer.py": workspace / "SimpleSAC/transit_v10/trainer.py",
        ROOT / "SimpleSAC/transit_v11/__init__.py": workspace / "SimpleSAC/transit_v11/__init__.py",
        ROOT / "SimpleSAC/transit_v11/provenance.py": workspace / "SimpleSAC/transit_v11/provenance.py",
        ROOT / "SimpleSAC/transit_v11/trainer.py": workspace / "SimpleSAC/transit_v11/trainer.py",
        REPRO / "verify_h2oplus_v11_actor_hold_deployment.py": workspace / "reproducibility/verify_h2oplus_v11_actor_hold_deployment.py",
        registration: control / "registration.json",
        authorization: control / "authorization.json",
        rationale: control / "design_rationale.json",
        correction: control / "boundary_correction.json",
        environment: control / "environment_design.json",
        plan: control / "training_plan.json",
        protocol: control / "protocol.json",
        freeze: control / "freeze.json",
    }
    for admission in admissions:
        mappings[admission] = control / "admissions" / admission.name
    for seed, record in initializers.items():
        mappings[Path(record["source"]["path"])] = Path(record["path"])
    for source, destination in mappings.items():
        _copy(source, destination)
    entries = []
    for destination in sorted(mappings.values()):
        entries.append(
            {
                "path": destination.relative_to(workspace).as_posix(),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    base_manifest = workspace / "deployment_manifest.json"
    payload = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "immutable-v11-actor-hold-overlay-on-frozen-v7-workspace",
        "study_stage": "development",
        "formal_held_out_outcomes_included": False,
        "base_v7_training_deployment": descriptor(base_manifest),
        "design_rationale": descriptor(control / "design_rationale.json"),
        "v2_boundary_correction": descriptor(control / "boundary_correction.json"),
        "overlay_entry_count": len(entries),
        "overlay_aggregate_sha256": stable_object_sha256({"entries": entries}),
        "entries": entries,
    }
    manifest = workspace / "deployment_manifest_v11.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    formal_workspace = args.formal_workspace.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve()
    workspace = deployment_root / "workspace"
    base_manifest = formal_workspace / "deployment_manifest.json"
    rationale = args.rationale_record.expanduser().resolve(strict=True)
    rationale_payload = json.loads(rationale.read_text())
    if (
        sha256(rationale)
        != "456ddcb73c5d78a8f0bf90f3e8ddfbbfda81c8c45c76d505f4d309eead35605b"
        or rationale_payload.get("schema_version") != RATIONALE_SCHEMA
        or rationale_payload.get("status")
        != "frozen-before-v11-training-or-new-development-outcomes"
    ):
        raise ValueError("frozen v11 design rationale drifted")
    correction = _build_correction(
        args.correction_record.expanduser().resolve(), force=args.force
    )
    registration = REPRO / "h2oplus_v11_actor_hold_registration_v2.json"
    authorization = REPRO / "h2oplus_v11_actor_hold_authorization_v2.json"
    write_stable(
        registration,
        _registration(base_manifest, rationale, correction),
        force=args.force,
    )
    write_stable(
        authorization,
        _authorization(registration, rationale, correction),
        force=args.force,
    )
    environment = _build_environment(
        args,
        registration,
        authorization,
        rationale,
        correction,
        force=args.force,
    )
    initializers = _initializer_records(workspace)
    plan = _build_plan(
        deployment_root=deployment_root,
        workspace=workspace,
        registration=registration,
        authorization=authorization,
        rationale=rationale,
        correction=correction,
        environment=environment,
        initializers=initializers,
        force=args.force,
    )
    protocol = _build_protocol(
        registration=registration,
        authorization=authorization,
        rationale=rationale,
        correction=correction,
        environment=environment,
        plan=plan,
        base_manifest=base_manifest,
        force=args.force,
    )
    admissions = _build_admissions(
        formal_workspace=formal_workspace,
        workspace=workspace,
        plan=plan,
        protocol=protocol,
        rationale=rationale,
        correction=correction,
        force=args.force,
    )
    freeze = _build_freeze(
        registration=registration,
        authorization=authorization,
        rationale=rationale,
        correction=correction,
        environment=environment,
        plan=plan,
        protocol=protocol,
        admissions=admissions,
        force=args.force,
    )
    deployment = _build_deployment(
        args=args,
        formal_workspace=formal_workspace,
        registration=registration,
        authorization=authorization,
        rationale=rationale,
        correction=correction,
        environment=environment,
        plan=plan,
        protocol=protocol,
        admissions=admissions,
        freeze=freeze,
        initializers=initializers,
    )
    print(
        json.dumps(
            {
                "registration": descriptor(registration),
                "authorization": descriptor(authorization),
                "design_rationale": descriptor(rationale),
                "v2_boundary_correction": descriptor(correction),
                "environment_design": descriptor(environment),
                "training_plan": descriptor(plan),
                "protocol": descriptor(protocol),
                "freeze": descriptor(freeze),
                "deployment_manifest": descriptor(deployment),
                "new_training_runs": 5,
                "maximum_future_development_sumo_episodes": 720,
                "development_evaluation_submitted": False,
                "formal_held_out_outcomes_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
