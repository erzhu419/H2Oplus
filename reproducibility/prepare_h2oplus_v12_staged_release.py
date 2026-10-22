#!/usr/bin/env python3
"""Preregister and package the H2O+ v12 staged-release development pilot."""

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


REGISTRATION_SCHEMA = "transit-h2oplus-v12-staged-release-registration-v2"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v12-staged-release-authorization-v2"
ENVIRONMENT_SCHEMA = "transit-h2oplus-v12-staged-release-environment-v2"
PLAN_SCHEMA = "transit-h2oplus-v12-staged-release-training-plan-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v12-staged-release-protocol-v2"
ADMISSION_SCHEMA = "transit-training-admission-v12-staged-release-development-v2"
FREEZE_SCHEMA = "transit-h2oplus-v12-staged-release-freeze-v2"
DEPLOYMENT_SCHEMA = "transit-h2oplus-v12-staged-release-deployment-v2"
RATIONALE_SCHEMA = "transit-h2oplus-v12-staged-release-rationale-v1"
CORRECTION_SCHEMA = "transit-h2oplus-v12-staged-release-deployment-correction-v2"
FROZEN_RATIONALE_V1_SHA256 = (
    "4e6a66c21c37495247dc635173633a1c4afc158193d64305ded5f8f610c4c79d"
)
USER_AUTHORIZATION_TEXT = "那就继续v12开发"
AUTHORIZED_PLAN_TEXT = (
    "保持 V11 的 5k warmup 和 actor/temperature 严格冻结到 10k，并由每个种子的一个"
    "完整 10k 状态分出五个精确分支。突变对照保持 V11 的 10k 后行为；actor-ramp 与 "
    "temperature-ramp 分别把对应优化器学习率在 10k-15k 从 0.1 线性恢复到 1.0；"
    "actor-anchor 保持 actor batch 大小并使用 20% 离线、80% 在线状态；combined 同时"
    "应用三个因素。critic replay 固定 0.5、更新顺序保持 actor-before-critic、ratio "
    "仅作诊断且 TD 权重恒为 1。使用新的开发场景和五个配对种子，只做开发筛选。"
)
TRAINING_SEEDS = (6201, 6202, 6203, 6204, 6205)
PREFIX_VARIANT = "shared-prefix"
BRANCH_VARIANTS = (
    "abrupt-control",
    "actor-ramp",
    "temperature-ramp",
    "actor-anchor",
    "staged-combined",
)
VARIANTS = (PREFIX_VARIANT, *BRANCH_VARIANTS)
BLOCKS = (
    {
        "name": f"v12-staged-release-development-{index:02d}",
        "sumo_seed": 5100 + index,
        "demand_realization_seed": 5200 + index,
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
        default=REPRO / "deployments/h2oplus_v12_staged_release_pilot_v2",
    )
    parser.add_argument(
        "--rationale-record",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_rationale_v1.json",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o/development_scenarios_v12/staged_release_v1",
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
        ROOT / "SimpleSAC/train_transit_v12_staged_release.py",
        ROOT / "SimpleSAC/transit_v10/__init__.py",
        ROOT / "SimpleSAC/transit_v10/provenance.py",
        ROOT / "SimpleSAC/transit_v10/trainer.py",
        ROOT / "SimpleSAC/transit_v11/__init__.py",
        ROOT / "SimpleSAC/transit_v11/provenance.py",
        ROOT / "SimpleSAC/transit_v11/trainer.py",
        ROOT / "SimpleSAC/transit_v12/__init__.py",
        ROOT / "SimpleSAC/transit_v12/provenance.py",
        ROOT / "SimpleSAC/transit_v12/trainer.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def _build_rationale(path: Path, *, force: bool) -> Path:
    del force
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256(resolved)
    if actual != FROZEN_RATIONALE_V1_SHA256:
        raise ValueError(
            "frozen v12 rationale-v1 drifted: "
            f"expected {FROZEN_RATIONALE_V1_SHA256}, got {actual}"
        )
    return resolved


def _legacy_build_rationale_v1(path: Path, *, force: bool) -> Path:
    """Retained only to document how the already-frozen v1 record was built."""
    v11 = REPRO / "h2oplus_v11_actor_hold_results"
    analysis_path = v11 / "actor_hold_analysis_v1.json"
    expected = {
        analysis_path: "e3f6df3d075774dec3e662cbf2b6ef848e5f361f57be45fff7cf159768fa04e1",
        v11 / "training_closeout_v2.json": "489c7efe5cf91f73bb31a9b6df03441c2fb3bcd4d280c65493c5ae8ec3af01a6",
        v11 / "training_closeout_verification_v2.json": "ff0ef7c69bac850bf9f743275ffd9bbfc87c0b1c9713a83769a052bd936d256e",
        v11 / "evaluation_closeout_v1.json": "27456e2d2a9acd639e8a7e70c69b8003e0167e4dc0289272a57406a56dc60cd2",
        v11 / "evaluation_closeout_verification_v1.json": "45d1e924d2e55d11b1f74a19f628b3ff3d00760bf44488819afeef360f19f8a5",
    }
    drift = {
        str(source): {"expected": digest, "actual": sha256(source)}
        for source, digest in expected.items()
        if sha256(source) != digest
    }
    if drift:
        raise ValueError(f"sealed v11 rationale inputs drifted: {drift}")
    analysis = json.loads(analysis_path.read_text())
    result = analysis["development_scientific_result"]
    payload = {
        "schema_version": RATIONALE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v12-training-or-new-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "diagnosis": {
            "v11_screening_passed": result["screening_gate_passed"],
            "v11_mean_aulc": result["all_algorithm_mean_aulc"][
                "h2oplus-v11-actor-hold"
            ],
            "v11_mean_at_10000": result["budget_results"]["10000"]["means"][
                "h2oplus-v11-actor-hold"
            ],
            "v11_mean_at_25000": result["budget_results"]["25000"]["means"][
                "h2oplus-v11-actor-hold"
            ],
            "v10_delayed_actor_mean_at_25000": result["budget_results"][
                "25000"
            ]["means"]["h2oplus-v10-delayed-actor"],
            "v11_worse_than_v10_delayed_actor_at_25000": True,
            "v11_worst_completion_aulc_margin": min(
                row["delta"] for row in result["completion_aulc"].values()
            ),
            "mechanistic_hypothesis": (
                "simultaneous full-speed actor and temperature release after a "
                "10k frozen prefix causes avoidable late policy drift"
            ),
        },
        "selected_factors": {
            "actor_learning_rate_release": "0.1 to 1.0 from 10k to 15k",
            "temperature_learning_rate_release": "0.1 to 1.0 from 10k to 15k",
            "actor_state_anchor": "20% offline and 80% online at fixed batch size",
            "combined": "all three registered factors",
            "abrupt_control": "exact V11 behavior after the shared 10k prefix",
        },
        "sealed_inputs": {
            "v11_analysis": descriptor(analysis_path),
            "v11_training_closeout": descriptor(v11 / "training_closeout_v2.json"),
            "v11_training_verification": descriptor(
                v11 / "training_closeout_verification_v2.json"
            ),
            "v11_evaluation_closeout": descriptor(
                v11 / "evaluation_closeout_v1.json"
            ),
            "v11_evaluation_verification": descriptor(
                v11 / "evaluation_closeout_verification_v1.json"
            ),
        },
        "claim_boundary": {
            "all_v11_outcomes_preserved": True,
            "supports_formal_held_out_effect_claim": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
        },
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(path, payload, force=force)
    return path


def _build_correction(path: Path, *, force: bool) -> Path:
    v1_root = REPRO / "deployments/h2oplus_v12_staged_release_pilot_v1/workspace"
    sources = {
        "registration_v1": REPRO / "h2oplus_v12_staged_release_registration_v1.json",
        "authorization_v1": REPRO / "h2oplus_v12_staged_release_authorization_v1.json",
        "environment_v1": REPRO / "h2oplus_v12_staged_release_environment_v1.json",
        "training_plan_v1": REPRO / "h2oplus_v12_staged_release_training_plan_v1.json",
        "protocol_v1": REPRO / "h2oplus_v12_staged_release_protocol_v1.json",
        "freeze_v1": REPRO / "h2oplus_v12_staged_release_freeze_v1.json",
        "deployment_manifest_v1": v1_root / "deployment_manifest_v12.json",
        "local_verification_v1": REPRO
        / "h2oplus_v12_staged_release_deployment_verification_v1_local.json",
        "deployment_sync_v1": REPRO
        / "h2oplus_v12_staged_release_deployment_sync_v1.json",
    }
    expected = {
        "registration_v1": "bedf92da7e922b212dcb31771b9ef928b2f9f35ba1d850445b4d8ec95c0749f7",
        "authorization_v1": "1d0586956f90a342c86a766dd9dfe208b67c814530e544cfb91c423f005aceda",
        "environment_v1": "3125394a676f025bc1d3e14644db4bfcd2e261edcf53f00d3e4c3dcedb373d53",
        "training_plan_v1": "8c04325a7d24a53e122d661a8337eea268cc03be9bd3007a0db9ff463bd9c6fd",
        "protocol_v1": "061e1341b152caa5edd91f1e36f4fc4820eb36a5a43465478d29308fb0117483",
        "freeze_v1": "863327c5d28a9eceab5aabb925853a2e1a758d4a2b2cce65581253d08ae4c1db",
        "deployment_manifest_v1": "cbb0e4465dfb87a0248dbc67da2571861ad9aa195b88db78b64d497ad1b7b8b2",
        "local_verification_v1": "c63d8705a135b4ddfe210508cfa8db4da1bd2096552b3144c0f7ae6c3b3288f2",
        "deployment_sync_v1": "2659d7f7ef820346d86a42fabf698f35195b017297dc1d8c61625a534165f069",
    }
    drift = {
        label: {"expected": expected[label], "actual": sha256(source)}
        for label, source in sources.items()
        if sha256(source) != expected[label]
    }
    if drift:
        raise ValueError(f"v12 deployment-v1 incident evidence drifted: {drift}")
    payload = {
        "schema_version": CORRECTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v12-training-or-sumo",
        "invalid_lineage": "deployment-v1-preflight-only",
        "failure_stage": "jtl110gpu no-SUMO deployment verification",
        "failure": (
            "fixed-control descriptors resolved to local-only V10/V11 evidence "
            "paths outside the copied deployment workspace"
        ),
        "observed_error": (
            "FileNotFoundError for reproducibility/h2oplus_v10_delayed_results "
            "while verifying fixed_controls"
        ),
        "v1_remote_sync_completed": ["jtl110gpu", "jtl110gpu2"],
        "v1_remote_verification_attempted": ["jtl110gpu"],
        "v1_remote_verification_passed": [],
        "v1_training_tasks_submitted": [],
        "v1_training_or_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "correction": {
            "fresh_deployment_root_required": True,
            "fresh_protocol_admission_and_run_signatures_required": True,
            "copy_all_fixed_control_descriptors_into_the_workspace": True,
            "verify_all_522_base_and_all_overlay_entries_on_each_host": True,
            "scientific_design_scenarios_seeds_and_gates_changed": False,
        },
        "incident_evidence": {
            label: descriptor(source) for label, source in sources.items()
        },
        "claim_boundary": {
            "deployment_v1_may_be_used_for_training": False,
            "deployment_v1_may_be_used_as_scientific_evidence": False,
            "deployment_v1_preserved_as_incident_evidence": True,
        },
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(path, payload, force=force)
    return path


def _registration(
    base_manifest: Path, rationale: Path, correction: Path
) -> dict[str, Any]:
    v8 = REPRO / "h2oplus_v8_fewshot_results"
    v9 = REPRO / "h2oplus_v9_adaptive_results"
    v10 = REPRO / "h2oplus_v10_delayed_results"
    v11 = REPRO / "h2oplus_v11_actor_hold_results"
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v12 staged-release single-factor development pilot",
        "status": "registered-before-v12-training-or-new-development-evaluation",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "frozen_v7_v8_v9_v10_v11_trainer_or_evidence_modified": False,
        "rationale": descriptor(rationale),
        "deployment_correction": descriptor(correction),
        "authorized_plan_text": AUTHORIZED_PLAN_TEXT,
        "authorized_plan_text_sha256": hashlib.sha256(
            AUTHORIZED_PLAN_TEXT.encode()
        ).hexdigest(),
        "causal_design": {
            "shared_prefix": "one exact 0-10k trajectory per seed, then five branches",
            "fixed_controls": [
                "sealed v8 h2oplus",
                "sealed v8 sac-online",
                "sealed v10 delayed-actor",
                "sealed v11 actor-hold",
            ],
            "new_variants": list(VARIANTS),
            "branch_variants": list(BRANCH_VARIANTS),
            "new_training_runs": len(TRAINING_SEEDS) * (1 + len(BRANCH_VARIANTS)),
            "prefix_runs": len(TRAINING_SEEDS),
            "branch_runs": len(TRAINING_SEEDS) * len(BRANCH_VARIANTS),
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_online_events": 25_000,
        },
        "staged_release": {
            "warmup_events": 5_000,
            "common_prefix_end_online_events": 10_000,
            "actor_and_temperature_held_through_online_events": 10_000,
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "offline_critic_replay_share": 0.50,
            "online_update_order": "actor-before-critic",
            "factors": {
                "actor_ramp": "actor LR 0.1 to 1.0 over 10k-15k",
                "temperature_ramp": "temperature LR 0.1 to 1.0 over 10k-15k",
                "actor_anchor": (
                    "nearest-row 20% nominal anchor: 26 offline and 102 online "
                    "states at the registered actor batch size 128"
                ),
            },
        },
        "development_evaluation": {
            "blocks": list(BLOCKS),
            "demand_scales": list(DEMAND_SCALES),
            "common_online_event_budgets": list(COMMON_EVALUATION_EVENTS),
            "actor_hash_deduplication_required": True,
            "logical_policy_aliases": 180,
            "physical_actor_tensors": (
                "determined only by exact actor hashes after all training closes"
            ),
            "maximum_unique_remote_sumo_episodes": 1620,
            "submitted_with_training": False,
        },
        "screening_gates": {
            "candidate_relative_aulc_improvement_over_v11_at_least": 0.01,
            "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least": 0.02,
            "candidate_relative_aulc_improvement_over_sac_at_least": 0.02,
            "candidate_relative_aulc_improvement_over_fixed_h2o_at_least": 0.03,
            "candidate_seed_wins_over_v11_at_least": 4,
            "candidate_seed_wins_over_v10_delayed_actor_at_least": 4,
            "candidate_seed_wins_over_sac_at_least": 4,
            "candidate_seed_wins_over_fixed_h2o_at_least": 4,
            "candidate_mean_lower_than_v11_and_v10_delayed_actor_at_25000": True,
            "candidate_mean_lower_than_sac_and_fixed_h2o_at_10000_and_25000": True,
            "completion_rate_noninferiority_margin": 0.003,
            "no_demand_scale_relative_aulc_degradation_over_sac_greater_than": 0.01,
            "all_30_training_runs_pass_registered_stability": True,
            "all_branches_share_exact_seed_matched_10k_state": True,
            "single_factors_beating_abrupt_control_in_mean_aulc_at_least": 2,
            "candidate_relative_aulc_improvement_over_abrupt_control_at_least": 0.01,
            "formal_confirmation_eligibility_created": False,
        },
        "interpretation": {
            "five_seed_development_screen_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "formal_stage_authorized_or_submitted": False,
            "all_registered_outcomes_retained": True,
        },
        "sources": {
            "v12_source_sha256": _source_map(),
            "deployment_verifier": descriptor(
                REPRO / "verify_h2oplus_v12_staged_release_deployment_v2.py"
            ),
            "base_v7_training_deployment": descriptor(base_manifest),
            "sealed_v8_training_closeout": descriptor(v8 / "training_closeout.json"),
            "sealed_v8_training_manifest_verification": descriptor(
                v8 / "training_file_manifest_verification.json"
            ),
            "sealed_v8_fewshot_closeout": descriptor(v8 / "fewshot_closeout_v3.json"),
            "sealed_v8_analysis": descriptor(v8 / "fewshot_analysis_v3.json"),
            "sealed_v9_training_closeout": descriptor(v9 / "training_closeout.json"),
            "sealed_v9_closeout": descriptor(v9 / "adaptive_closeout_v3.json"),
            "sealed_v9_closeout_verification": descriptor(
                v9 / "adaptive_closeout_verification_v3.json"
            ),
            "sealed_v9_analysis": descriptor(v9 / "adaptive_analysis_v3.json"),
            "sealed_v10_analysis": descriptor(
                v10 / "delayed_analysis_v1.json"
            ),
            "sealed_v10_branch_closeout": descriptor(v10 / "branch_closeout_v4.json"),
            "sealed_v10_branch_verification": descriptor(
                v10 / "branch_closeout_verification_v4.json"
            ),
            "sealed_v10_evaluation_closeout": descriptor(
                v10 / "evaluation_closeout_v1.json"
            ),
            "sealed_v10_evaluation_verification": descriptor(
                v10 / "evaluation_closeout_verification_v1.json"
            ),
            "sealed_v10_methods_manifest": descriptor(
                REPRO / "h2oplus_v10_delayed_evaluation_methods_v1.json"
            ),
            "sealed_v11_analysis": descriptor(
                v11 / "actor_hold_analysis_v1.json"
            ),
            "sealed_v11_training_closeout": descriptor(
                v11 / "training_closeout_v2.json"
            ),
            "sealed_v11_training_verification": descriptor(
                v11 / "training_closeout_verification_v2.json"
            ),
            "sealed_v11_evaluation_closeout": descriptor(
                v11 / "evaluation_closeout_v1.json"
            ),
            "sealed_v11_evaluation_verification": descriptor(
                v11 / "evaluation_closeout_verification_v1.json"
            ),
            "sealed_v11_methods_manifest": descriptor(
                REPRO / "h2oplus_v11_actor_hold_evaluation_methods_v1.json"
            ),
        },
        "stop_rules": [
            "stop on nonfinite state, nonzero exit or source/event/update mismatch",
            "never restart or checkpoint-select because an outcome is poor",
            "never use formal held-out outcomes for development selection",
            "never train or run SUMO locally",
            "submit only five prefix runs before exact prefix closeout",
            "do not submit branches until all five handoffs are independently audited",
            "do not submit development evaluation until all 30 training runs close",
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
        "authorization_decision": (
            "authorize-v12-prefix-then-audited-branches-and-development-evaluation"
        ),
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
        "rationale": descriptor(rationale),
        "deployment_correction": descriptor(correction),
        "remote_training_authorized": True,
        "prefix_submission_authorized_now": True,
        "branch_submission_authorized_only_after_prefix_closeout": True,
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
    base_config = SUMO_ROOT / "online_control/control_sim_traci_period.sumocfg"
    routes = (
        SUMO_ROOT / "e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml",
        SUMO_ROOT / "e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml",
    )
    passengers = SUMO_ROOT / "online_control/initialize_obj/save_obj_passenger.add.xml"
    destination = REPRO / "h2oplus_v12_staged_release_environment_v2.json"
    scenario_root = args.scenario_root.expanduser().resolve()
    block_payloads = []
    scenario_payloads = []
    manifest_hashes: set[str] = set()
    runtime_hashes: set[str] = set()
    for block in BLOCKS:
        manifests = []
        for scale in DEMAND_SCALES:
            original = build_scenario(
                base_config=base_config.resolve(strict=True),
                passenger_routes=tuple(path.resolve(strict=True) for path in routes),
                passenger_objects=passengers.resolve(strict=True),
                output_root=scenario_root,
                scale=scale,
                seed=block["demand_realization_seed"],
                depart_jitter_seconds=DEPART_JITTER_SECONDS,
            )
            portable, evidence = _portable_scenario(original, force=force)
            runtime_hash = _runtime_key(evidence)
            if evidence["manifest_sha256"] in manifest_hashes or runtime_hash in runtime_hashes:
                raise ValueError("v12 development scenario content is duplicated")
            manifest_hashes.add(evidence["manifest_sha256"])
            runtime_hashes.add(runtime_hash)
            relative = os.path.relpath(portable, destination.parent)
            manifests.append(relative)
            scenario_payloads.append(
                {
                    "block": block["name"],
                    "demand_scale": float(evidence["demand_scale"]),
                    "manifest": relative,
                    "manifest_sha256": evidence["manifest_sha256"],
                    "runtime_content_sha256": runtime_hash,
                }
            )
        block_payloads.append({**block, "scenario_manifests": manifests})

    inventories = {
        "training": _design_inventory(REPRO / "training_environment_design_v7_formal_v1.json"),
        "formal_held_out": _design_inventory(REPRO / "evaluation_environment_design_v7_formal_v2.json"),
        "v8_stage_a": _design_inventory(REPRO / "h2oplus_v8_development_environment_stage_a_v1.json"),
        "v8_fewshot": _design_inventory(REPRO / "h2oplus_v8_fewshot_environment_v1.json"),
        "v9_adaptive": _design_inventory(REPRO / "h2oplus_v9_adaptive_environment_v2.json"),
        "v10_delayed": _design_inventory(
            REPRO / "h2oplus_v10_delayed_environment_v4.json"
        ),
        "v11_actor_hold": _design_inventory(
            REPRO / "h2oplus_v11_actor_hold_environment_v2.json"
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
        raise ValueError(f"v12 development disjointness failed: {overlaps}")
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v12-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "blocks": block_payloads,
        "demand_scales": list(DEMAND_SCALES),
        "depart_jitter_seconds": DEPART_JITTER_SECONDS,
        "environment_cells": len(scenario_payloads),
        "scenario_descriptors": scenario_payloads,
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
        "rationale": descriptor(rationale),
        "deployment_correction": descriptor(correction),
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(destination, payload, force=force)
    return destination


def _initializer_records(workspace: Path) -> dict[int, dict[str, Any]]:
    records = {}
    source_root = REPRO / "h2oplus_v8_fewshot_results/training"
    target_root = workspace / "reproducibility/h2oplus_v12_initializers_v2"
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
    if variant not in VARIANTS:
        raise ValueError(f"unknown v12 variant: {variant}")
    prefix = variant == PREFIX_VARIANT
    return {
        "stage": "prefix" if prefix else "branch",
        "warmup_events": 5_000,
        "epochs": 10 if prefix else 40,
        "offline_updates": 50_000,
        "expected_total_updates": 55_000 if prefix else 70_000,
        "online_event_cap": 10_000 if prefix else 25_000,
        "export_online_events": [0, 5_000, 10_000] if prefix else [25_000],
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
    branch_handoff_manifest: Path | None,
) -> list[str]:
    settings = _variant_settings(variant)
    arguments = [
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
    if settings["stage"] == "branch":
        if branch_handoff_manifest is None:
            raise ValueError("branch command lacks a shared-prefix handoff")
        arguments.extend(
            ["--branch-handoff-manifest", str(branch_handoff_manifest)]
        )
    elif branch_handoff_manifest is not None:
        raise ValueError("prefix command unexpectedly has a branch handoff")
    return arguments


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
    control = workspace / "reproducibility/h2oplus_v12_control_v2"
    output_root = deployment_root / "remote_results"
    identity = {
        "registration_sha256": sha256(registration),
        "authorization_sha256": sha256(authorization),
        "rationale_sha256": sha256(rationale),
        "deployment_correction_sha256": sha256(correction),
        "environment_sha256": sha256(environment),
        "v12_source_sha256": _source_map(),
    }
    jobs = []
    prefix_by_seed = {}
    for seed in TRAINING_SEEDS:
        variant = PREFIX_VARIANT
        settings = _variant_settings(variant)
        run_name = f"h2oplus_v12a2_shared_prefix_seed{seed}"
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
            branch_handoff_manifest=None,
        )
        prefix = {
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
        jobs.append(prefix)
        prefix_by_seed[seed] = prefix

    for variant in BRANCH_VARIANTS:
        settings = _variant_settings(variant)
        for seed in TRAINING_SEEDS:
            prefix = prefix_by_seed[seed]
            handoff = (
                output_root / prefix["name"] / "v12_handoff/manifest.json"
            )
            run_name = f"h2oplus_v12a2_{variant.replace('-', '_')}_seed{seed}"
            signature = stable_object_sha256(
                {
                    **identity,
                    "variant": variant,
                    "seed": seed,
                    "settings": settings,
                    "initializer_sha256": initializers[seed]["sha256"],
                    "prefix_run_signature": prefix["run_signature"],
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
                branch_handoff_manifest=handoff,
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
                    "branch_handoff_manifest": str(handoff),
                    "prefix": {
                        "name": prefix["name"],
                        "run_signature": prefix["run_signature"],
                        "seed": seed,
                    },
                    **settings,
                    "milestones": _milestones(variant),
                }
            )
    payload = {
        "schema_version": PLAN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v12-training",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "workspace": str(workspace),
        "remote_output_root": str(output_root),
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "rationale": descriptor(rationale),
        "deployment_correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "v12_source_sha256": _source_map(),
        "required_environment": {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
            "CPATH": str(workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
            "CUDA_VISIBLE_DEVICES": "scheduler-assigned exactly one physical GPU",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "jobs": jobs,
    }
    path = REPRO / "h2oplus_v12_staged_release_training_plan_v2.json"
    write_stable(path, payload, force=force)
    return path


def _fixed_control_sources() -> dict[str, Path]:
    return {
        "v8_training_closeout": REPRO
        / "h2oplus_v8_fewshot_results/training_closeout.json",
        "v8_methods_manifest": REPRO
        / "h2oplus_v8_fewshot_results/evaluation_methods_v8_fewshot.json",
        "v10_branch_closeout": REPRO
        / "h2oplus_v10_delayed_results/branch_closeout_v4.json",
        "v10_evaluation_closeout": REPRO
        / "h2oplus_v10_delayed_results/evaluation_closeout_v1.json",
        "v10_methods_manifest": REPRO
        / "h2oplus_v10_delayed_evaluation_methods_v1.json",
        "v11_training_closeout": REPRO
        / "h2oplus_v11_actor_hold_results/training_closeout_v2.json",
        "v11_evaluation_closeout": REPRO
        / "h2oplus_v11_actor_hold_results/evaluation_closeout_v1.json",
        "canonical_methods_manifest": REPRO
        / "h2oplus_v11_actor_hold_evaluation_methods_v1.json",
    }


def _deployed_descriptor(source: Path, destination: Path) -> dict[str, Any]:
    value = descriptor(source)
    value["path"] = str(destination)
    return value


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
    plan_payload = json.loads(plan.read_text())
    jobs = plan_payload["jobs"]
    workspace = Path(plan_payload["workspace"])
    fixed_control_root = (
        workspace / "reproducibility/h2oplus_v12_control_v2/fixed_controls"
    )
    fixed_control_sources = _fixed_control_sources()
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v12-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "rationale": descriptor(rationale),
        "deployment_correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "base_v7_training_deployment": descriptor(base_manifest),
        "v12_source_sha256": _source_map(),
        "training_jobs": jobs,
        "staged_release": {
            "prefix_jobs": 5,
            "branch_jobs": 25,
            "branch_release_requires": (
                "all five prefix tasks token-bound backend exit0, complete artifacts, "
                "exact handoff manifests and independent prefix closeout"
            ),
            "development_evaluation_release_requires": (
                "all 30 training jobs exact exit0 and independent training closeout"
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
        "fixed_controls": {
            **{
                label: _deployed_descriptor(
                    source, fixed_control_root / f"{label}.json"
                )
                for label, source in fixed_control_sources.items()
            },
            "seeds": list(TRAINING_SEEDS),
            "methods": [
                "h2oplus",
                "sac-online",
                "h2oplus-v10-delayed-actor",
                "h2oplus-v11-actor-hold",
            ],
            "common_evaluation_events": list(COMMON_EVALUATION_EVENTS),
        },
        "analysis_gates": json.loads(registration.read_text())["screening_gates"],
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v12_staged_release_protocol_v2.json"
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
    output = REPRO / "h2oplus_v12_staged_release_training_admissions_v2"
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
            "v12_source_sha256": _source_map(),
            "v12_wrapper": descriptor(ROOT / "SimpleSAC/train_transit_v12_staged_release.py"),
            "v12_protocol": descriptor(protocol),
            "v12_rationale": descriptor(rationale),
            "v12_deployment_correction": descriptor(correction),
            "base_v7_admission_source": descriptor(formal_admission),
            "offline_initializer": job["offline_initializer"],
            "selected_arguments": job,
            "checks": {
                "base_v7_source_map_reused_exactly": True,
                "v12_overlay_bound_separately": True,
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
        "status": "frozen-before-v12-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "rationale": descriptor(rationale),
        "deployment_correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "protocol": descriptor(protocol),
        "training_admissions": [descriptor(path) for path in admissions],
        "v12_source_sha256": _source_map(),
        "new_training_runs": 30,
        "prefix_runs_released_initially": 5,
        "branch_runs_conditionally_released": 25,
        "maximum_future_development_sumo_episodes": 1620,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v12_staged_release_freeze_v2.json"
    write_stable(path, payload, force=force)
    return path


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
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
    control = workspace / "reproducibility/h2oplus_v12_control_v2"
    mappings = {
        ROOT / "SimpleSAC/train_transit_v12_staged_release.py": workspace / "SimpleSAC/train_transit_v12_staged_release.py",
        ROOT / "SimpleSAC/transit_v10/__init__.py": workspace / "SimpleSAC/transit_v10/__init__.py",
        ROOT / "SimpleSAC/transit_v10/provenance.py": workspace / "SimpleSAC/transit_v10/provenance.py",
        ROOT / "SimpleSAC/transit_v10/trainer.py": workspace / "SimpleSAC/transit_v10/trainer.py",
        ROOT / "SimpleSAC/transit_v11/__init__.py": workspace / "SimpleSAC/transit_v11/__init__.py",
        ROOT / "SimpleSAC/transit_v11/provenance.py": workspace / "SimpleSAC/transit_v11/provenance.py",
        ROOT / "SimpleSAC/transit_v11/trainer.py": workspace / "SimpleSAC/transit_v11/trainer.py",
        ROOT / "SimpleSAC/transit_v12/__init__.py": workspace / "SimpleSAC/transit_v12/__init__.py",
        ROOT / "SimpleSAC/transit_v12/provenance.py": workspace / "SimpleSAC/transit_v12/provenance.py",
        ROOT / "SimpleSAC/transit_v12/trainer.py": workspace / "SimpleSAC/transit_v12/trainer.py",
        REPRO / "verify_h2oplus_v12_staged_release_deployment_v2.py": workspace / "reproducibility/verify_h2oplus_v12_staged_release_deployment_v2.py",
        registration: control / "registration.json",
        authorization: control / "authorization.json",
        rationale: control / "rationale.json",
        correction: control / "deployment_correction.json",
        environment: control / "environment_design.json",
        plan: control / "training_plan.json",
        protocol: control / "protocol.json",
        freeze: control / "freeze.json",
    }
    for admission in admissions:
        mappings[admission] = control / "admissions" / admission.name
    for seed, record in initializers.items():
        mappings[Path(record["source"]["path"])] = Path(record["path"])
    for label, source in _fixed_control_sources().items():
        mappings[source] = control / "fixed_controls" / f"{label}.json"
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
        "status": "immutable-v12-v2-overlay-on-frozen-v7-workspace",
        "study_stage": "development",
        "formal_held_out_outcomes_included": False,
        "base_v7_training_deployment": descriptor(base_manifest),
        "rationale": descriptor(control / "rationale.json"),
        "deployment_correction": descriptor(control / "deployment_correction.json"),
        "overlay_entry_count": len(entries),
        "overlay_aggregate_sha256": stable_object_sha256({"entries": entries}),
        "entries": entries,
    }
    manifest = workspace / "deployment_manifest_v12_v2.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    formal_workspace = args.formal_workspace.expanduser().resolve(strict=True)
    rationale = _build_rationale(
        args.rationale_record.expanduser().resolve(), force=args.force
    )
    correction = _build_correction(
        REPRO / "h2oplus_v12_staged_release_deployment_correction_v2.json",
        force=args.force,
    )
    deployment_root = args.deployment_root.expanduser().resolve()
    workspace = deployment_root / "workspace"
    base_manifest = formal_workspace / "deployment_manifest.json"
    registration = REPRO / "h2oplus_v12_staged_release_registration_v2.json"
    authorization = REPRO / "h2oplus_v12_staged_release_authorization_v2.json"
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
                "rationale": descriptor(rationale),
                "deployment_correction": descriptor(correction),
                "environment_design": descriptor(environment),
                "training_plan": descriptor(plan),
                "protocol": descriptor(protocol),
                "freeze": descriptor(freeze),
                "deployment_manifest": descriptor(deployment),
                "new_training_runs": 30,
                "prefix_runs_released_initially": 5,
                "maximum_future_development_sumo_episodes": 1620,
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
