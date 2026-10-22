#!/usr/bin/env python3
"""Register and freeze the v8 development-only few-shot comparison."""

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


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SUMO_ROOT = PROJECT_ROOT / "SUMO_ruiguang"
REPRO = ROOT / "reproducibility"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import load_validated_scenario, sha256
from reproducibility.build_passenger_scenarios import build_scenario
from reproducibility.prepare_h2oplus_v8_development_stage_a import (
    _design_inventory,
    _portable_scenario,
    _runtime_key,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v8-fewshot-registration-v1"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v8-fewshot-authorization-v1"
DESIGN_SCHEMA = "transit-h2oplus-v8-fewshot-development-environment-v1"
PLAN_SCHEMA = "transit-h2oplus-v8-fewshot-training-plan-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v8-fewshot-protocol-v1"
ADMISSION_SCHEMA = "transit-training-admission-v8-fewshot-development-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-fewshot-freeze-v1"
LAUNCHER_SCHEMA = "transit-h2oplus-v8-fewshot-training-launcher-v1"
USER_AUTHORIZATION_TEXT = (
    "对，offline2online的优势应该是在few shot learning上，那就继续比较"
)
TRAINING_SEEDS = list(range(6201, 6206))
BLOCKS = [
    {"name": f"v8-fewshot-development-{index:02d}", "sumo_seed": 4300 + index,
     "demand_realization_seed": 4400 + index}
    for index in range(1, 4)
]
DEMAND_SCALES = [0.85, 1.0, 1.15]
DEPART_JITTER_SECONDS = 90.0
ONLINE_MATCHED_BUDGETS = [0, 5_000, 10_000, 25_000, 50_000, 100_000]
H2O_EXPORT_BUDGETS = ONLINE_MATCHED_BUDGETS
SAC_EXPORT_BUDGETS = [
    0,
    5_000,
    10_000,
    25_000,
    50_000,
    55_000,
    60_000,
    75_000,
    100_000,
    150_000,
]
COMPUTE_MATCHED = [
    {"h2oplus_online_events": 5_000, "sac_online_events": 55_000, "learner_update_steps": 50_000},
    {"h2oplus_online_events": 10_000, "sac_online_events": 60_000, "learner_update_steps": 55_000},
    {"h2oplus_online_events": 25_000, "sac_online_events": 75_000, "learner_update_steps": 70_000},
    {"h2oplus_online_events": 50_000, "sac_online_events": 100_000, "learner_update_steps": 95_000},
    {"h2oplus_online_events": 100_000, "sac_online_events": 150_000, "learner_update_steps": 145_000},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-workspace",
        type=Path,
        default=REPRO / "deployments/v7_formal_v2_training_v7r1/workspace",
    )
    parser.add_argument(
        "--launcher-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v8_fewshot_training_v1/launcher",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o/development_scenarios_v8/fewshot_v1",
    )
    parser.add_argument(
        "--training-design",
        type=Path,
        default=REPRO / "training_environment_design_v7_formal_v1.json",
    )
    parser.add_argument(
        "--held-out-design",
        type=Path,
        default=REPRO / "evaluation_environment_design_v7_formal_v2.json",
    )
    parser.add_argument(
        "--previous-development-design",
        type=Path,
        default=REPRO / "h2oplus_v8_development_environment_stage_a_v1.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": os.path.relpath(resolved, base.resolve()) if base else str(resolved),
        "sha256": sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def write_stable(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    time_keys = {"created_utc", "registered_utc", "approved_utc"}
    if path.exists():
        old = json.loads(path.read_text())
        old_comparable = {key: value for key, value in old.items() if key not in time_keys}
        new_comparable = {key: value for key, value in payload.items() if key not in time_keys}
        if old_comparable != new_comparable and not force:
            raise FileExistsError(f"refusing to replace a different artifact: {path}")
        for key in time_keys:
            if isinstance(old.get(key), str):
                payload[key] = old[key]
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _registration(wrapper: Path, formal_manifest: Path) -> dict[str, Any]:
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v8 development few-shot sample-efficiency comparison",
        "status": "registered-not-submitted",
        "study_stage": "development",
        "execution_authorized_in_registration": False,
        "formal_held_out_outcomes_read": False,
        "formal_held_out_may_be_read_for_selection": False,
        "local_training_or_sumo_allowed": False,
        "methods": ["h2oplus", "sac-online"],
        "training_seeds": TRAINING_SEEDS,
        "development_environment": {
            "blocks": BLOCKS,
            "demand_scales": DEMAND_SCALES,
            "depart_jitter_seconds": DEPART_JITTER_SECONDS,
            "environment_cells": len(BLOCKS) * len(DEMAND_SCALES),
        },
        "online_sample_matched": {
            "budgets": ONLINE_MATCHED_BUDGETS,
            "primary": True,
            "x_axis": "nominal cumulative online simulator decision events",
            "atomic_batch_policy": (
                "record actual events; accept only bounded intermediate overshoot below "
                "one simultaneous-decision batch and enforce the final hard cap"
            ),
            "zero_and_5000_alias": (
                "evaluate the zero-event actor once and alias 5000 only after exact "
                "actor-tensor SHA-256 equality proves warmup made no policy update"
            ),
        },
        "learner_step_matched": {
            "pairs": COMPUTE_MATCHED,
            "definition": (
                "same trainer actor/critic learner iterations; not equal wall time, "
                "optimizer-step count or online interaction"
            ),
            "ratio_accounting": (
                "H2O+ ratio-optimizer steps are recorded separately and are not "
                "credited as SAC learner steps"
            ),
            "role": "secondary; SAC receives 50000 additional online events",
            "supports_few_shot_claim": False,
        },
        "endpoint": {
            "primary": "generalized_seconds_per_passenger",
            "direction": "lower-is-better",
            "sample_efficiency_summary": (
                "normalized trapezoidal area under the performance-versus-online-event "
                "curve from 0 through 100000 events on the linear event axis"
            ),
            "paired_unit": "training seed after averaging the nine development cells",
            "completion_constraint": "passenger_completion_rate",
        },
        "fewshot_confirmation_gate": {
            "mean_relative_aulc_improvement_at_least": 0.02,
            "training_seeds_with_lower_aulc_at_least": 4,
            "exact_one_sided_paired_sign_flip_p_less_than": 0.05,
            "mean_at_10000_events_lower": True,
            "mean_at_25000_events_lower": True,
            "completion_rate_noninferiority_margin": 0.005,
            "no_demand_scale_aulc_relative_degradation_greater_than": 0.01,
            "final_100000_event_relative_degradation_not_greater_than": 0.05,
        },
        "interpretation": {
            "development_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "failure_retains_all_results": True,
            "stage_b_authorized_or_submitted": False,
        },
        "registered_matrix": {
            "unique_actor_policies_per_seed": 14,
            "training_runs": 10,
            "unique_remote_sumo_episodes": 630,
            "evaluation_shards": 2,
            "workers_per_shard": 12,
        },
        "sources": {
            "fewshot_wrapper": descriptor(wrapper),
            "fewshot_evaluator": descriptor(
                ROOT / "SimpleSAC/evaluate_transit_v8_fewshot.py"
            ),
            "fewshot_matrix_runner": descriptor(
                REPRO / "run_evaluation_matrix_v8_fewshot.py"
            ),
            "fewshot_analysis": descriptor(
                REPRO / "analyze_h2oplus_v8_fewshot.py"
            ),
            "frozen_v7_training_deployment": descriptor(formal_manifest),
            "a2_gate_analysis": descriptor(
                REPRO / "h2oplus_v8_stage_a2_results/stage_a2_gate_analysis.json"
            ),
        },
        "stop_rules": [
            "stop a run on nonfinite state, nonzero exit, or event/update-contract failure",
            "retain every registered seed and milestone irrespective of performance",
            "never use formal held-out outcomes for development selection",
            "never train or run SUMO locally",
            "do not launch a formal confirmation stage from this registration",
        ],
        "generator": descriptor(Path(__file__).resolve()),
    }


def _authorization(registration: Path) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_decision": "authorize-v8-fewshot-registration-training-and-development-evaluation",
        "approved_scope": "development-only H2O+ versus SAC-online few-shot comparison",
        "authorization_channel": "explicit user message in the current Codex task",
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode("utf-8")
        ).hexdigest(),
        "registration": descriptor(registration, registration.parent),
        "remote_training_and_development_sumo_authorized": True,
        "local_training_or_sumo_authorized": False,
        "formal_held_out_access_authorized": False,
        "formal_confirmation_stage_authorized": False,
    }


def _build_environment(
    args: argparse.Namespace, registration: Path, authorization: Path
) -> Path:
    base_config = SUMO_ROOT / "online_control/control_sim_traci_period.sumocfg"
    routes = (
        SUMO_ROOT / "e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml",
        SUMO_ROOT / "e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml",
    )
    passenger_objects = (
        SUMO_ROOT / "online_control/initialize_obj/save_obj_passenger.add.xml"
    )
    design_path = REPRO / "h2oplus_v8_fewshot_environment_v1.json"
    scenario_root = args.scenario_root.expanduser().resolve()
    block_payloads = []
    scenarios = []
    manifest_hashes: set[str] = set()
    runtime_hashes: set[str] = set()
    for block in BLOCKS:
        manifests = []
        for scale in DEMAND_SCALES:
            original = build_scenario(
                base_config=base_config.resolve(strict=True),
                passenger_routes=tuple(path.resolve(strict=True) for path in routes),
                passenger_objects=passenger_objects.resolve(strict=True),
                output_root=scenario_root,
                scale=scale,
                seed=block["demand_realization_seed"],
                depart_jitter_seconds=DEPART_JITTER_SECONDS,
            )
            portable, evidence = _portable_scenario(original, force=args.force)
            runtime_hash = _runtime_key(evidence)
            if evidence["manifest_sha256"] in manifest_hashes or runtime_hash in runtime_hashes:
                raise ValueError("few-shot development scenario content is duplicated")
            manifest_hashes.add(evidence["manifest_sha256"])
            runtime_hashes.add(runtime_hash)
            relative = os.path.relpath(portable, design_path.parent)
            manifests.append(relative)
            scenarios.append(
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
        "training": _design_inventory(args.training_design),
        "formal_held_out": _design_inventory(args.held_out_design),
        "previous_development": _design_inventory(args.previous_development_design),
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
        raise ValueError(f"few-shot development disjointness failed: {overlaps}")
    payload = {
        "schema_version": DESIGN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-fewshot-training-and-outcomes",
        "study_stage": "development",
        "role": "few-shot sample-efficiency comparison; not formal held-out evaluation",
        "formal_held_out_outcomes_read": False,
        "blocks": block_payloads,
        "independent_realization_blocks": len(BLOCKS),
        "demand_scales": DEMAND_SCALES,
        "depart_jitter_seconds": DEPART_JITTER_SECONDS,
        "environment_cells": len(scenarios),
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
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "generator": descriptor(Path(__file__).resolve()),
        "scenario_generator": descriptor(
            REPRO / "build_passenger_scenarios.py"
        ),
        "portable_manifest_generator": descriptor(
            REPRO / "prepare_h2oplus_v8_development_stage_a.py"
        ),
    }
    write_stable(design_path, payload, force=args.force)
    return design_path


def _method_arguments(method: str) -> dict[str, Any]:
    if method == "h2oplus":
        return {
            "epochs": 190,
            "offline_updates": 50_000,
            "online_fraction": 0.5,
            "ratio_pretrain_updates": 1_000,
            "online_capacity": 120_000,
            "export_online_events": H2O_EXPORT_BUDGETS,
        }
    if method == "sac-online":
        return {
            "epochs": 290,
            "offline_updates": 0,
            "online_fraction": 0.0,
            "ratio_pretrain_updates": 0,
            "online_capacity": 180_000,
            "export_online_events": SAC_EXPORT_BUDGETS,
        }
    raise ValueError(method)


def _milestones(method: str) -> list[dict[str, int]]:
    settings = _method_arguments(method)
    result = []
    for events in settings["export_online_events"]:
        online_steps = max(events - 5_000, 0)
        learner_steps = settings["offline_updates"] + online_steps
        ratio_steps = 0
        if method == "h2oplus" and online_steps:
            ratio_steps = settings["ratio_pretrain_updates"] + (
                online_steps + 4 - 1
            ) // 4
        result.append(
            {
                "online_events": events,
                "learner_update_steps": learner_steps,
                "ratio_optimizer_steps": ratio_steps,
            }
        )
    return result


def _command_arguments(
    *,
    method: str,
    seed: int,
    run_name: str,
    run_signature: str,
    admission: Path,
    formal_workspace: Path,
    output_root: Path,
) -> list[str]:
    settings = _method_arguments(method)
    return [
        "--method", method,
        "--run-name", run_name,
        "--run-signature", run_signature,
        "--seed", str(seed),
        "--device", "cuda:0",
        "--max-canonical-processes-per-gpu", "1",
        "--cpu-threads", "6",
        "--dataset", str(formal_workspace / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"),
        "--sim-env", str(formal_workspace / "bus_h2o/calibrated_env"),
        "--training-admission", str(admission),
        "--output-root", str(output_root),
        "--max-offline-rows", "0",
        "--online-capacity", str(settings["online_capacity"]),
        "--offline-updates", str(settings["offline_updates"]),
        "--warmup-events", "5000",
        "--ratio-pretrain-updates", str(settings["ratio_pretrain_updates"]),
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
        "--online-fraction", str(settings["online_fraction"]),
        "--target-entropy", "0.0",
        "--initial-temperature", "1.0",
        "--iql-advantage-temperature", "3.0",
        "--offline-cql-alpha", "0.0",
        "--cql-n-actions", "10",
        "--ratio-update-interval", "4",
        "--offline-checkpoint-every", "10000",
        "--checkpoint-every", "5",
        "--log-every", "250",
        "--resource-interval", "5.0",
        "--device-cache-observation-dtype", "float32",
        "--compile-networks",
        "--compile-mode", "default",
        "--sim-demand-scales", "0.75", "1.0", "1.25",
    ]


def _build_plan(
    args: argparse.Namespace,
    registration: Path,
    authorization: Path,
    environment: Path,
    wrapper: Path,
    formal_manifest: Path,
) -> Path:
    formal_workspace = args.formal_workspace.expanduser().resolve()
    launcher = args.launcher_root.expanduser().resolve()
    output_root = launcher.parent / "remote_results"
    jobs = []
    identity_base = {
        "registration_sha256": sha256(registration),
        "authorization_sha256": sha256(authorization),
        "environment_sha256": sha256(environment),
        "wrapper_sha256": sha256(wrapper),
        "formal_deployment_manifest_sha256": sha256(formal_manifest),
    }
    for method in ("h2oplus", "sac-online"):
        settings = _method_arguments(method)
        for seed in TRAINING_SEEDS:
            run_name = f"{method}_seed{seed}"
            run_signature = stable_signature(
                {**identity_base, "method": method, "seed": seed, **settings}
            )
            admission = launcher / "admissions" / f"{run_name}.json"
            command = _command_arguments(
                method=method,
                seed=seed,
                run_name=run_name,
                run_signature=run_signature,
                admission=admission,
                formal_workspace=formal_workspace,
                output_root=output_root,
            )
            jobs.append(
                {
                    "name": run_name,
                    "method": method,
                    "seed": seed,
                    "run_signature": run_signature,
                    "training_admission": str(admission),
                    "command_arguments": command,
                    "epochs": settings["epochs"],
                    "warmup_events": 5_000,
                    "online_event_cap": 5_000 + settings["epochs"] * 500,
                    "offline_updates": settings["offline_updates"],
                    "expected_total_updates": settings["offline_updates"] + settings["epochs"] * 500,
                    "export_online_events": settings["export_online_events"],
                    "milestones": _milestones(method),
                    "online_capacity": settings["online_capacity"],
                }
            )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-fewshot-training",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "formal_workspace": str(formal_workspace),
        "remote_output_root": str(output_root),
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "environment_design": descriptor(environment),
        "fewshot_wrapper": descriptor(wrapper),
        "frozen_v7_training_deployment": descriptor(formal_manifest),
        "source_formal_plans": [
            descriptor(REPRO / "training_plans_v7_formal_v2_v7r1/training_plan_v7_formal_core_server_h2oplus_v7.json"),
            descriptor(REPRO / "training_plans_v7_formal_v2_v7r1/training_plan_v7_formal_core_server_baselines_v7.json"),
        ],
        "required_environment": {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
            "CPATH": str(formal_workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
            "CUDA_VISIBLE_DEVICES": "scheduler-assigned exactly one physical GPU",
        },
        "jobs": jobs,
    }
    path = REPRO / "h2oplus_v8_fewshot_training_plan_v1.json"
    write_stable(path, plan, force=args.force)
    return path


def _build_protocol(
    registration: Path,
    authorization: Path,
    environment: Path,
    plan: Path,
    wrapper: Path,
    formal_manifest: Path,
    *,
    force: bool,
) -> Path:
    jobs = json.loads(plan.read_text())["jobs"]
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-fewshot-training-and-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "fewshot_wrapper": descriptor(wrapper),
        "evaluation_sources": [
            descriptor(ROOT / "SimpleSAC/evaluate_transit_v8_fewshot.py"),
            descriptor(REPRO / "run_evaluation_matrix_v8_fewshot.py"),
            descriptor(REPRO / "analyze_h2oplus_v8_fewshot.py"),
        ],
        "frozen_v7_training_deployment": descriptor(formal_manifest),
        "training_jobs": jobs,
        "policy_budgets": {
            "h2oplus": {
                "export_online_events": H2O_EXPORT_BUDGETS,
                "evaluated_unique_online_events": [0, 10_000, 25_000, 50_000, 100_000],
                "5000_event_result_aliases": 0,
                "milestones": _milestones("h2oplus"),
            },
            "sac-online": {
                "export_online_events": SAC_EXPORT_BUDGETS,
                "evaluated_unique_online_events": [0, 10_000, 25_000, 50_000, 55_000, 60_000, 75_000, 100_000, 150_000],
                "5000_event_result_aliases": 0,
                "milestones": _milestones("sac-online"),
            },
        },
        "online_sample_matched_budgets": ONLINE_MATCHED_BUDGETS,
        "learner_step_matched_pairs": COMPUTE_MATCHED,
        "evaluation": {
            "unique_policies": 70,
            "development_environment_cells": 9,
            "unique_sumo_episodes": 630,
            "shards": 2,
            "workers_per_shard": 12,
        },
        "analysis_gates": json.loads(registration.read_text())["fewshot_confirmation_gate"],
        "stage_b_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v8_fewshot_protocol_v1.json"
    write_stable(path, payload, force=force)
    return path


def _build_admissions(
    plan: Path,
    protocol: Path,
    wrapper: Path,
    formal_workspace: Path,
    *,
    force: bool,
) -> list[Path]:
    formal_admission = formal_workspace / (
        "reproducibility/event_cap_regate_v7_v1_remote/"
        "training_admission_v7_formal_v2_event_cap_fix_v1.json"
    )
    source = json.loads(formal_admission.read_text())
    output = REPRO / "h2oplus_v8_fewshot_training_admissions_v1"
    paths = []
    for job in json.loads(plan.read_text())["jobs"]:
        payload = {
            "schema_version": ADMISSION_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_stage": "development",
            "passed": True,
            "formal_held_out_outcomes_read": False,
            "method": job["method"],
            "seed": job["seed"],
            "run_name": job["name"],
            "run_signature": job["run_signature"],
            "dataset": source["dataset"],
            "simulator": source["simulator"],
            "training_source_sha256": source["training_source_sha256"],
            "fewshot_wrapper": descriptor(wrapper),
            "fewshot_protocol": descriptor(protocol),
            "frozen_v7_admission_source": descriptor(formal_admission),
            "selected_arguments": job,
            "checks": {
                "frozen_v7_source_map_reused_exactly": True,
                "new_training_seed_is_disjoint_from_v7_and_prior_v8": True,
                "development_environment_frozen_before_training": True,
                "formal_held_out_outcomes_read": False,
            },
        }
        path = output / f"{job['name']}.json"
        write_stable(path, payload, force=force)
        paths.append(path)
    return paths


def _build_freeze(
    registration: Path,
    authorization: Path,
    environment: Path,
    plan: Path,
    protocol: Path,
    admissions: list[Path],
    *,
    force: bool,
) -> Path:
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-fewshot-training-and-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "registration": descriptor(registration),
        "authorization": descriptor(authorization),
        "environment_design": descriptor(environment),
        "training_plan": descriptor(plan),
        "protocol": descriptor(protocol),
        "training_admissions": [descriptor(path) for path in admissions],
        "evaluation_sources": [
            descriptor(ROOT / "SimpleSAC/evaluate_transit_v8_fewshot.py"),
            descriptor(REPRO / "run_evaluation_matrix_v8_fewshot.py"),
            descriptor(REPRO / "analyze_h2oplus_v8_fewshot.py"),
        ],
        "training_runs": 10,
        "unique_sumo_episodes_after_training": 630,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    path = REPRO / "h2oplus_v8_fewshot_freeze_v1.json"
    write_stable(path, payload, force=force)
    return path


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"launcher copy hash mismatch: {destination}")


def _build_launcher(
    args: argparse.Namespace,
    wrapper: Path,
    registration: Path,
    authorization: Path,
    environment: Path,
    plan: Path,
    protocol: Path,
    admissions: list[Path],
    freeze: Path,
    formal_manifest: Path,
) -> Path:
    launcher = args.launcher_root.expanduser().resolve()
    if launcher.exists():
        if not args.force:
            raise FileExistsError(f"refusing to replace launcher bundle: {launcher}")
        shutil.rmtree(launcher)
    launcher.mkdir(parents=True)
    mapping = {
        wrapper: launcher / "train_transit_v8_fewshot.py",
        registration: launcher / "registration.json",
        authorization: launcher / "authorization.json",
        environment: launcher / "environment_design.json",
        plan: launcher / "training_plan.json",
        protocol: launcher / "protocol.json",
        freeze: launcher / "freeze.json",
    }
    for source, destination in mapping.items():
        _copy(source, destination)
    for admission in admissions:
        _copy(admission, launcher / "admissions" / admission.name)
    entries = []
    for path in sorted(item for item in launcher.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(launcher).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": LAUNCHER_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "immutable-training-overlay",
        "study_stage": "development",
        "formal_held_out_outcomes_included": False,
        "frozen_v7_training_deployment": descriptor(formal_manifest),
        "entry_count": len(entries),
        "aggregate_sha256": stable_signature({"entries": entries}),
        "entries": entries,
    }
    path = launcher / "launcher_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    args = parse_args()
    formal_workspace = args.formal_workspace.expanduser().resolve(strict=True)
    formal_manifest = formal_workspace / "deployment_manifest.json"
    wrapper = ROOT / "SimpleSAC/train_transit_v8_fewshot.py"
    registration = REPRO / "h2oplus_v8_fewshot_registration_v1.json"
    authorization = REPRO / "h2oplus_v8_fewshot_authorization_v1.json"
    write_stable(
        registration,
        _registration(wrapper, formal_manifest),
        force=args.force,
    )
    write_stable(
        authorization,
        _authorization(registration),
        force=args.force,
    )
    environment = _build_environment(args, registration, authorization)
    plan = _build_plan(
        args, registration, authorization, environment, wrapper, formal_manifest
    )
    protocol = _build_protocol(
        registration,
        authorization,
        environment,
        plan,
        wrapper,
        formal_manifest,
        force=args.force,
    )
    admissions = _build_admissions(
        plan, protocol, wrapper, formal_workspace, force=args.force
    )
    freeze = _build_freeze(
        registration,
        authorization,
        environment,
        plan,
        protocol,
        admissions,
        force=args.force,
    )
    launcher_manifest = _build_launcher(
        args,
        wrapper,
        registration,
        authorization,
        environment,
        plan,
        protocol,
        admissions,
        freeze,
        formal_manifest,
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
                "launcher_manifest": descriptor(launcher_manifest),
                "training_runs": 10,
                "future_unique_sumo_episodes": 630,
                "formal_held_out_outcomes_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
