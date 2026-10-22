#!/usr/bin/env python3
"""Register, freeze and package the H2O+ v10 delayed-transition pilot."""

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


REGISTRATION_SCHEMA = "transit-h2oplus-v10-delayed-registration-v1"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v10-delayed-authorization-v1"
ENVIRONMENT_SCHEMA = "transit-h2oplus-v10-delayed-environment-v1"
PLAN_SCHEMA = "transit-h2oplus-v10-delayed-training-plan-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v10-delayed-protocol-v1"
ADMISSION_SCHEMA = "transit-training-admission-v10-delayed-development-v1"
FREEZE_SCHEMA = "transit-h2oplus-v10-delayed-freeze-v1"
DEPLOYMENT_SCHEMA = "transit-h2oplus-v10-delayed-deployment-v1"
USER_AUTHORIZATION_TEXT = "那就继续"
AUTHORIZED_PLAN_TEXT = (
    "恢复 5k warmup，并保持固定 H2O+ 行为到 10k。10k 后再逐渐把离线 replay "
    "比例从 0.5 降到约 0.25，不立即降到 0.1。暂时使用 unit ratio weights，不依赖从未"
    "打开的 ratio gate。将 replay 比例、online-only actor、更新顺序拆成单因素，不再一次"
    "组合。使用新的开发场景和 5 个种子，主门限仍是 0–25k 同在线事件 AULC、10k/25k "
    "均胜 SAC、至少 4/5 种子获胜且稳定性全通过。"
)
TRAINING_SEEDS = (6201, 6202, 6203, 6204, 6205)
PREFIX_VARIANT = "shared-prefix"
BRANCH_VARIANTS = (
    "unit-control",
    "delayed-replay",
    "delayed-actor",
    "delayed-order",
    "delayed-combined",
)
VARIANTS = (PREFIX_VARIANT, *BRANCH_VARIANTS)
BLOCKS = (
    {
        "name": f"v10-delayed-development-{index:02d}",
        "sumo_seed": 4700 + index,
        "demand_realization_seed": 4800 + index,
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
        default=REPRO / "deployments/h2oplus_v10_delayed_pilot_v1",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o/development_scenarios_v10/delayed_v1",
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
        ROOT / "SimpleSAC/train_transit_v10_delayed.py",
        ROOT / "SimpleSAC/transit_v10/__init__.py",
        ROOT / "SimpleSAC/transit_v10/provenance.py",
        ROOT / "SimpleSAC/transit_v10/trainer.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def _registration(base_manifest: Path) -> dict[str, Any]:
    v8 = REPRO / "h2oplus_v8_fewshot_results"
    v9 = REPRO / "h2oplus_v9_adaptive_results"
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v10 delayed-transition single-factor development pilot",
        "status": "registered-before-training-or-new-development-evaluation",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "frozen_v7_v8_v9_trainer_or_evidence_modified": False,
        "authorized_plan_text": AUTHORIZED_PLAN_TEXT,
        "authorized_plan_text_sha256": hashlib.sha256(
            AUTHORIZED_PLAN_TEXT.encode()
        ).hexdigest(),
        "causal_design": {
            "shared_prefix": "one exact 0-10k trajectory per seed, then five branches",
            "fixed_controls": ["sealed v8 h2oplus", "sealed v8 sac-online"],
            "new_variants": list(VARIANTS),
            "branch_variants": list(BRANCH_VARIANTS),
            "new_training_runs": len(TRAINING_SEEDS) * (1 + len(BRANCH_VARIANTS)),
            "prefix_runs": len(TRAINING_SEEDS),
            "branch_runs": len(TRAINING_SEEDS) * len(BRANCH_VARIANTS),
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_online_events": 25_000,
        },
        "delayed_transition": {
            "warmup_events": 5_000,
            "common_prefix_end_online_events": 10_000,
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "actor_burn_in_critic_updates": 0,
            "offline_replay_share_knots": [[10_000, 0.50], [25_000, 0.25]],
            "factors": {
                "replay": "0.50 through 10k, then linear to 0.25 at 25k",
                "actor_states": "online-only strictly after 10k",
                "update_order": "critic-before-actor strictly after 10k",
            },
        },
        "development_evaluation": {
            "blocks": list(BLOCKS),
            "demand_scales": list(DEMAND_SCALES),
            "common_online_event_budgets": list(COMMON_EVALUATION_EVENTS),
            "actor_hash_deduplication_required": True,
            "expected_unique_actor_tensors": 60,
            "expected_unique_remote_sumo_episodes": 540,
            "submitted_with_training": False,
        },
        "screening_gates": {
            "delayed_combined_relative_aulc_improvement_over_sac_at_least": 0.02,
            "delayed_combined_relative_aulc_improvement_over_fixed_h2o_at_least": 0.03,
            "delayed_combined_seed_wins_over_sac_at_least": 4,
            "delayed_combined_seed_wins_over_fixed_h2o_at_least": 4,
            "delayed_combined_mean_lower_than_sac_at_10000_and_25000": True,
            "completion_rate_noninferiority_margin": 0.005,
            "no_demand_scale_relative_aulc_degradation_over_sac_greater_than": 0.01,
            "all_30_training_runs_pass_registered_stability": True,
            "all_branches_share_exact_seed_matched_10k_state": True,
            "single_factor_attribution_required": True,
            "formal_confirmation_eligibility_created": False,
        },
        "interpretation": {
            "five_seed_development_screen_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "formal_stage_authorized_or_submitted": False,
            "all_registered_outcomes_retained": True,
        },
        "sources": {
            "v10_source_sha256": _source_map(),
            "deployment_verifier": descriptor(
                REPRO / "verify_h2oplus_v10_delayed_deployment.py"
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


def _authorization(registration: Path) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_decision": "authorize-v10-prefix-then-audited-branches-and-development-evaluation",
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
    *,
    force: bool,
) -> Path:
    base_config = SUMO_ROOT / "online_control/control_sim_traci_period.sumocfg"
    routes = (
        SUMO_ROOT / "e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml",
        SUMO_ROOT / "e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml",
    )
    passengers = SUMO_ROOT / "online_control/initialize_obj/save_obj_passenger.add.xml"
    destination = REPRO / "h2oplus_v10_delayed_environment_v1.json"
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
                raise ValueError("v10 development scenario content is duplicated")
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
        raise ValueError(f"v10 development disjointness failed: {overlaps}")
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v10-training-and-development-outcomes",
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
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(destination, payload, force=force)
    return destination


def _initializer_records(workspace: Path) -> dict[int, dict[str, Any]]:
    records = {}
    source_root = REPRO / "h2oplus_v8_fewshot_results/training"
    target_root = workspace / "reproducibility/h2oplus_v10_initializers"
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
        raise ValueError(f"unknown v10 variant: {variant}")
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
    environment: Path,
    initializers: dict[int, dict[str, Any]],
    force: bool,
) -> Path:
    control = workspace / "reproducibility/h2oplus_v10_control"
    output_root = deployment_root / "remote_results"
    identity = {
        "registration_sha256": sha256(registration),
        "authorization_sha256": sha256(authorization),
        "environment_sha256": sha256(environment),
        "v10_source_sha256": _source_map(),
    }
    jobs = []
    prefix_by_seed = {}
    for seed in TRAINING_SEEDS:
        variant = PREFIX_VARIANT
        settings = _variant_settings(variant)
        run_name = f"h2oplus_v10_shared_prefix_seed{seed}"
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
                output_root / prefix["name"] / "v10_handoff/manifest.json"
            )
            run_name = f"h2oplus_v10_{variant.replace('-', '_')}_seed{seed}"
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
        "status": "frozen-before-v10-training",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "workspace": str(workspace),
        "remote_output_root": str(output_root),
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "environment_design": descriptor(environment),
        "v10_source_sha256": _source_map(),
        "required_environment": {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
            "CPATH": str(workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
            "CUDA_VISIBLE_DEVICES": "scheduler-assigned exactly one physical GPU",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "jobs": jobs,
    }
    path = REPRO / "h2oplus_v10_delayed_training_plan_v1.json"
    write_stable(path, payload, force=force)
    return path


def _build_protocol(
    *,
    registration: Path,
    authorization: Path,
    environment: Path,
    plan: Path,
    base_manifest: Path,
    force: bool,
) -> Path:
    jobs = json.loads(plan.read_text())["jobs"]
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v10-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "base_v7_training_deployment": descriptor(base_manifest),
        "v10_source_sha256": _source_map(),
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
        "analysis_gates": json.loads(registration.read_text())["screening_gates"],
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v10_delayed_protocol_v1.json"
    write_stable(path, payload, force=force)
    return path


def _build_admissions(
    *,
    formal_workspace: Path,
    workspace: Path,
    plan: Path,
    protocol: Path,
    force: bool,
) -> list[Path]:
    formal_admission = formal_workspace / (
        "reproducibility/event_cap_regate_v7_v1_remote/"
        "training_admission_v7_formal_v2_event_cap_fix_v1.json"
    )
    source = json.loads(formal_admission.read_text())
    output = REPRO / "h2oplus_v10_delayed_training_admissions_v1"
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
            "v10_source_sha256": _source_map(),
            "v10_wrapper": descriptor(ROOT / "SimpleSAC/train_transit_v10_delayed.py"),
            "v10_protocol": descriptor(protocol),
            "base_v7_admission_source": descriptor(formal_admission),
            "offline_initializer": job["offline_initializer"],
            "selected_arguments": job,
            "checks": {
                "base_v7_source_map_reused_exactly": True,
                "v10_overlay_bound_separately": True,
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
    environment: Path,
    plan: Path,
    protocol: Path,
    admissions: list[Path],
    force: bool,
) -> Path:
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v10-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "protocol": descriptor(protocol),
        "training_admissions": [descriptor(path) for path in admissions],
        "v10_source_sha256": _source_map(),
        "new_training_runs": 30,
        "prefix_runs_released_initially": 5,
        "branch_runs_conditionally_released": 25,
        "future_development_sumo_episodes": 540,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v10_delayed_freeze_v1.json"
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
    control = workspace / "reproducibility/h2oplus_v10_control"
    mappings = {
        ROOT / "SimpleSAC/train_transit_v10_delayed.py": workspace / "SimpleSAC/train_transit_v10_delayed.py",
        ROOT / "SimpleSAC/transit_v10/__init__.py": workspace / "SimpleSAC/transit_v10/__init__.py",
        ROOT / "SimpleSAC/transit_v10/provenance.py": workspace / "SimpleSAC/transit_v10/provenance.py",
        ROOT / "SimpleSAC/transit_v10/trainer.py": workspace / "SimpleSAC/transit_v10/trainer.py",
        REPRO / "verify_h2oplus_v10_delayed_deployment.py": workspace / "reproducibility/verify_h2oplus_v10_delayed_deployment.py",
        registration: control / "registration.json",
        authorization: control / "authorization.json",
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
        "status": "immutable-v10-overlay-on-frozen-v7-workspace",
        "study_stage": "development",
        "formal_held_out_outcomes_included": False,
        "base_v7_training_deployment": descriptor(base_manifest),
        "overlay_entry_count": len(entries),
        "overlay_aggregate_sha256": stable_object_sha256({"entries": entries}),
        "entries": entries,
    }
    manifest = workspace / "deployment_manifest_v10.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    formal_workspace = args.formal_workspace.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve()
    workspace = deployment_root / "workspace"
    base_manifest = formal_workspace / "deployment_manifest.json"
    registration = REPRO / "h2oplus_v10_delayed_registration_v1.json"
    authorization = REPRO / "h2oplus_v10_delayed_authorization_v1.json"
    write_stable(
        registration,
        _registration(base_manifest),
        force=args.force,
    )
    write_stable(
        authorization,
        _authorization(registration),
        force=args.force,
    )
    environment = _build_environment(
        args, registration, authorization, force=args.force
    )
    initializers = _initializer_records(workspace)
    plan = _build_plan(
        deployment_root=deployment_root,
        workspace=workspace,
        registration=registration,
        authorization=authorization,
        environment=environment,
        initializers=initializers,
        force=args.force,
    )
    protocol = _build_protocol(
        registration=registration,
        authorization=authorization,
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
        force=args.force,
    )
    freeze = _build_freeze(
        registration=registration,
        authorization=authorization,
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
                "environment_design": descriptor(environment),
                "training_plan": descriptor(plan),
                "protocol": descriptor(protocol),
                "freeze": descriptor(freeze),
                "deployment_manifest": descriptor(deployment),
                "new_training_runs": 30,
                "prefix_runs_released_initially": 5,
                "future_development_sumo_episodes": 540,
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
