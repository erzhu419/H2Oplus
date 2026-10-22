#!/usr/bin/env python3
"""Register, freeze and package corrected v9 H2O+ adaptation pilot v2."""

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


REGISTRATION_SCHEMA = "transit-h2oplus-v9-adaptive-registration-v2"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v9-adaptive-authorization-v2"
ENVIRONMENT_SCHEMA = "transit-h2oplus-v9-adaptive-environment-v2"
PLAN_SCHEMA = "transit-h2oplus-v9-adaptive-training-plan-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v9-adaptive-protocol-v2"
ADMISSION_SCHEMA = "transit-training-admission-v9-adaptive-development-v2"
FREEZE_SCHEMA = "transit-h2oplus-v9-adaptive-freeze-v2"
DEPLOYMENT_SCHEMA = "transit-h2oplus-v9-adaptive-deployment-v2"
USER_AUTHORIZATION_TEXT = "可以 继续"
TRAINING_SEEDS = (6201, 6202, 6203)
VARIANTS = ("ratio-gated", "adaptation", "combined")
BLOCKS = (
    {
        "name": f"v9-adaptive-development-{index:02d}",
        "sumo_seed": 4500 + index,
        "demand_realization_seed": 4600 + index,
    }
    for index in range(1, 3)
)
BLOCKS = tuple(BLOCKS)
DEMAND_SCALES = (0.85, 1.0, 1.15)
DEPART_JITTER_SECONDS = 90.0
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
        default=REPRO / "deployments/h2oplus_v9_adaptive_pilot_v2",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o/development_scenarios_v9/adaptive_v1",
    )
    parser.add_argument(
        "--correction-record",
        type=Path,
        default=REPRO / "h2oplus_v9_adaptive_correction_v2.json",
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
        ROOT / "SimpleSAC/train_transit_v9_adaptive.py",
        ROOT / "SimpleSAC/transit_v9/__init__.py",
        ROOT / "SimpleSAC/transit_v9/trainer.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def _registration(base_manifest: Path, correction: Path) -> dict[str, Any]:
    v8 = REPRO / "h2oplus_v8_fewshot_results"
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v9 development ratio/replay adaptation 2x2 pilot",
        "status": "registered-before-training-or-new-development-evaluation",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "frozen_v7_trainer_or_evidence_modified": False,
        "correction": descriptor(correction),
        "causal_design": {
            "fixed_fixed_control": "sealed v8 h2oplus seed6201-6203",
            "new_variants": list(VARIANTS),
            "ratio_factor": ["fixed-v7", "reliability-gated-tempered"],
            "adaptation_factor": ["fixed-v7", "adaptive-v9"],
            "new_training_runs": len(VARIANTS) * len(TRAINING_SEEDS),
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_online_events": 25_000,
        },
        "adaptive_bundle": {
            "warmup_events": 1_000,
            "actor_burn_in_critic_updates": 500,
            "actor_state_source": "online-only after online interaction begins",
            "online_update_order": "critic-before-actor",
            "offline_replay_share_knots": [
                [1_000, 0.50],
                [10_000, 0.25],
                [25_000, 0.10],
            ],
        },
        "ratio_bundle": {
            "validation_partition": "trajectory-disjoint ratio_validation",
            "hysteresis": "open after two passes; close after two failures",
            "minimum_joint_auc": 0.60,
            "maximum_joint_brier": 0.24,
            "maximum_state_action_brier": 0.26,
            "minimum_raw_ess_fraction": 0.25,
            "maximum_high_clip_fraction": 0.25,
            "maximum_total_clip_fraction": 0.50,
            "target_applied_ess_fraction": 0.50,
            "closed_gate_behavior": "unit TD weights while classifiers keep training",
        },
        "development_evaluation": {
            "blocks": list(BLOCKS),
            "demand_scales": list(DEMAND_SCALES),
            "common_online_event_budgets": list(COMMON_EVALUATION_EVENTS),
            "unique_policies_after_training": 45,
            "unique_remote_sumo_episodes": 270,
            "submitted_with_training": False,
        },
        "screening_gates": {
            "combined_relative_aulc_improvement_over_sac_at_least": 0.02,
            "combined_relative_aulc_improvement_over_fixed_h2o_at_least": 0.03,
            "combined_seed_wins_over_sac": 3,
            "combined_seed_wins_over_fixed_h2o": 3,
            "combined_mean_lower_at_10000_and_25000": True,
            "completion_rate_noninferiority_margin": 0.005,
            "no_demand_scale_relative_aulc_degradation_over_sac_greater_than": 0.01,
            "intervention_attribution_required": True,
            "formal_confirmation_eligibility_created": False,
        },
        "interpretation": {
            "three_seed_development_screen_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "formal_stage_authorized_or_submitted": False,
            "all_registered_outcomes_retained": True,
        },
        "sources": {
            "v9_source_sha256": _source_map(),
            "deployment_verifier": descriptor(
                REPRO / "verify_h2oplus_v9_adaptive_deployment.py"
            ),
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
            "do not submit development evaluation until all nine training runs close",
            "do not launch a formal confirmation stage from this registration",
        ],
        "generator": descriptor(Path(__file__).resolve()),
    }


def _authorization(registration: Path, correction: Path) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_decision": "authorize-v9-development-training-and-later-development-evaluation",
        "authorization_channel": "explicit user message in current Codex task",
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode()
        ).hexdigest(),
        "registration": descriptor(registration),
        "correction": descriptor(correction),
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
    destination = REPRO / "h2oplus_v9_adaptive_environment_v2.json"
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
                raise ValueError("v9 development scenario content is duplicated")
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
        raise ValueError(f"v9 development disjointness failed: {overlaps}")
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v9-training-and-development-outcomes",
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
        "correction": descriptor(correction),
        "generator": descriptor(Path(__file__).resolve()),
    }
    write_stable(destination, payload, force=force)
    return destination


def _initializer_records(workspace: Path) -> dict[int, dict[str, Any]]:
    records = {}
    source_root = REPRO / "h2oplus_v8_fewshot_results/training"
    target_root = workspace / "reproducibility/h2oplus_v9_initializers"
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
    warmup = 5_000 if variant == "ratio-gated" else 1_000
    exports = (
        [0, 5_000, 10_000, 25_000]
        if variant == "ratio-gated"
        else [0, 1_000, 2_500, 5_000, 10_000, 25_000]
    )
    epochs = (25_000 - warmup) // 500
    return {
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
    correction: Path,
    environment: Path,
    initializers: dict[int, dict[str, Any]],
    force: bool,
) -> Path:
    control = workspace / "reproducibility/h2oplus_v9_control"
    output_root = deployment_root / "remote_results"
    identity = {
        "registration_sha256": sha256(registration),
        "authorization_sha256": sha256(authorization),
        "correction_sha256": sha256(correction),
        "environment_sha256": sha256(environment),
        "v9_source_sha256": _source_map(),
    }
    jobs = []
    for variant in VARIANTS:
        settings = _variant_settings(variant)
        for seed in TRAINING_SEEDS:
            run_name = f"h2oplus_v9_{variant.replace('-', '_')}_seed{seed}"
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
        "status": "frozen-before-v9-training",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "workspace": str(workspace),
        "remote_output_root": str(output_root),
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "v9_source_sha256": _source_map(),
        "required_environment": {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
            "CPATH": str(workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
            "CUDA_VISIBLE_DEVICES": "scheduler-assigned exactly one physical GPU",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "jobs": jobs,
    }
    path = REPRO / "h2oplus_v9_adaptive_training_plan_v2.json"
    write_stable(path, payload, force=force)
    return path


def _build_protocol(
    *,
    registration: Path,
    authorization: Path,
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
        "status": "frozen-before-v9-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "base_v7_training_deployment": descriptor(base_manifest),
        "v9_source_sha256": _source_map(),
        "training_jobs": jobs,
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
    path = REPRO / "h2oplus_v9_adaptive_protocol_v2.json"
    write_stable(path, payload, force=force)
    return path


def _build_admissions(
    *,
    formal_workspace: Path,
    workspace: Path,
    plan: Path,
    protocol: Path,
    correction: Path,
    force: bool,
) -> list[Path]:
    formal_admission = formal_workspace / (
        "reproducibility/event_cap_regate_v7_v1_remote/"
        "training_admission_v7_formal_v2_event_cap_fix_v1.json"
    )
    source = json.loads(formal_admission.read_text())
    output = REPRO / "h2oplus_v9_adaptive_training_admissions_v2"
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
            "seed": job["seed"],
            "run_name": job["name"],
            "run_signature": job["run_signature"],
            "dataset": source["dataset"],
            "simulator": source["simulator"],
            "training_source_sha256": source["training_source_sha256"],
            "v9_source_sha256": _source_map(),
            "v9_wrapper": descriptor(ROOT / "SimpleSAC/train_transit_v9_adaptive.py"),
            "v9_protocol": descriptor(protocol),
            "correction": descriptor(correction),
            "base_v7_admission_source": descriptor(formal_admission),
            "offline_initializer": job["offline_initializer"],
            "selected_arguments": job,
            "checks": {
                "base_v7_source_map_reused_exactly": True,
                "v9_overlay_bound_separately": True,
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
        "status": "frozen-before-v9-training-and-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "correction": descriptor(correction),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "protocol": descriptor(protocol),
        "training_admissions": [descriptor(path) for path in admissions],
        "v9_source_sha256": _source_map(),
        "new_training_runs": 9,
        "future_development_sumo_episodes": 270,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v9_adaptive_freeze_v2.json"
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
    control = workspace / "reproducibility/h2oplus_v9_control"
    mappings = {
        ROOT / "SimpleSAC/train_transit_v9_adaptive.py": workspace / "SimpleSAC/train_transit_v9_adaptive.py",
        ROOT / "SimpleSAC/transit_v9/__init__.py": workspace / "SimpleSAC/transit_v9/__init__.py",
        ROOT / "SimpleSAC/transit_v9/trainer.py": workspace / "SimpleSAC/transit_v9/trainer.py",
        REPRO / "verify_h2oplus_v9_adaptive_deployment.py": workspace / "reproducibility/verify_h2oplus_v9_adaptive_deployment.py",
        registration: control / "registration.json",
        authorization: control / "authorization.json",
        correction: control / "correction.json",
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
        "status": "immutable-v9-overlay-on-frozen-v7-workspace",
        "study_stage": "development",
        "formal_held_out_outcomes_included": False,
        "base_v7_training_deployment": descriptor(base_manifest),
        "correction": descriptor(control / "correction.json"),
        "overlay_entry_count": len(entries),
        "overlay_aggregate_sha256": stable_object_sha256({"entries": entries}),
        "entries": entries,
    }
    manifest = workspace / "deployment_manifest_v9.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    formal_workspace = args.formal_workspace.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve()
    workspace = deployment_root / "workspace"
    base_manifest = formal_workspace / "deployment_manifest.json"
    correction = args.correction_record.expanduser().resolve(strict=True)
    correction_payload = json.loads(correction.read_text())
    if (
        correction_payload.get("status")
        != "v1-invalid-sealed-v2-correction-authorized"
        or correction_payload.get("formal_held_out_outcomes_read") is not False
        or correction_payload.get("local_training_or_sumo_executed") is not False
    ):
        raise ValueError("v9 correction record is stale or invalid")
    registration = REPRO / "h2oplus_v9_adaptive_registration_v2.json"
    authorization = REPRO / "h2oplus_v9_adaptive_authorization_v2.json"
    write_stable(
        registration,
        _registration(base_manifest, correction),
        force=args.force,
    )
    write_stable(
        authorization,
        _authorization(registration, correction),
        force=args.force,
    )
    environment = _build_environment(
        args, registration, authorization, correction, force=args.force
    )
    initializers = _initializer_records(workspace)
    plan = _build_plan(
        deployment_root=deployment_root,
        workspace=workspace,
        registration=registration,
        authorization=authorization,
        correction=correction,
        environment=environment,
        initializers=initializers,
        force=args.force,
    )
    protocol = _build_protocol(
        registration=registration,
        authorization=authorization,
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
        correction=correction,
        force=args.force,
    )
    freeze = _build_freeze(
        registration=registration,
        authorization=authorization,
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
                "correction": descriptor(correction),
                "environment_design": descriptor(environment),
                "training_plan": descriptor(plan),
                "protocol": descriptor(protocol),
                "freeze": descriptor(freeze),
                "deployment_manifest": descriptor(deployment),
                "new_training_runs": 9,
                "future_development_sumo_episodes": 270,
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
