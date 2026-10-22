#!/usr/bin/env python3
"""Preregister and package the H2O+ v13 proximal-release development pilot."""

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

from SimpleSAC.transit_v13.trainer import (  # noqa: E402
    PREDESIGNATED_CANDIDATE,
    PROXIMAL_RATES,
    REGISTERED_POST_10K_ACTOR_UPDATES,
    proximal_retention,
)
from SimpleSAC.transit_v13.provenance import (  # noqa: E402
    checkpoint_state_sha256,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v13-proximal-release-registration-v1"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v13-proximal-release-authorization-v1"
RATIONALE_SCHEMA = "transit-h2oplus-v13-proximal-release-rationale-v1"
PLAN_SCHEMA = "transit-h2oplus-v13-proximal-release-training-plan-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v13-proximal-release-protocol-v1"
ADMISSION_SCHEMA = "transit-training-admission-v13-proximal-release-development-v1"
FREEZE_SCHEMA = "transit-h2oplus-v13-proximal-release-freeze-v1"
DEPLOYMENT_SCHEMA = "transit-h2oplus-v13-proximal-release-deployment-v1"
SOURCE_HANDOFF_SCHEMA = "transit-h2oplus-v12-prefix-handoff-v2"
TRAINING_SEEDS = (6201, 6202, 6203, 6204, 6205)
VARIANTS = tuple(PROXIMAL_RATES)
USER_AUTHORIZATION_TEXT = "那就继续改进"
BASE_MANIFEST_SHA256 = (
    "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
)
V12_PROTOCOL_SHA256 = (
    "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
)
SEALED_V12 = {
    "prefix_closeout": (
        REPRO / "h2oplus_v12_staged_release_prefix_closeout_v2.json",
        "8bb303dbe038089422ba88f28bf06cd47a672b71f929d1c14306679b943d0ef1",
    ),
    "prefix_verification": (
        REPRO / "h2oplus_v12_staged_release_prefix_closeout_verification_v2.json",
        "b13b7f0827326f52c6c9f512a7385e71e5abf40a8f15df0e70d5ec26126242e7",
    ),
    "branch_closeout": (
        REPRO / "h2oplus_v12_staged_release_branch_closeout_v2.json",
        "998f47488b50b2cd40f9a6ddd5a399712bb9208cb1ef883cad1181f800385b66",
    ),
    "branch_verification": (
        REPRO / "h2oplus_v12_staged_release_branch_closeout_verification_v2.json",
        "6835e3c545be22c25bd9e3c7eae7ef987bb1864e027c4fef9849788f744868c8",
    ),
    "development_analysis": (
        REPRO / "h2oplus_v12_staged_release_results/staged_release_analysis_v1.json",
        "49a172e8ff4340c64c33f683223c4dd9bb7b2a8aa501ff3a3abe13c8d20f4f13",
    ),
    "evaluation_closeout": (
        REPRO / "h2oplus_v12_staged_release_results/evaluation_closeout_v1.json",
        "18379992c2c04700fc6460cffda3b5a4ad23feef73841b8ba00b25d2a9c14969",
    ),
    "evaluation_verification": (
        REPRO / "h2oplus_v12_staged_release_results/evaluation_closeout_verification_v1.json",
        "ed3db861b6a27add86e809ec76ea00485fe3ddd914ed302618e9264fb484f8d1",
    ),
    "v12_protocol": (
        REPRO / "h2oplus_v12_staged_release_protocol_v2.json",
        V12_PROTOCOL_SHA256,
    ),
}


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
        default=REPRO / "deployments/h2oplus_v13_proximal_release_pilot_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def descriptor(path: Path, *, deployed_path: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(deployed_path if deployed_path is not None else resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def write_stable(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    time_keys = {"created_utc", "registered_utc", "authorized_utc"}
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


def source_map() -> dict[str, str]:
    relative_paths = (
        "SimpleSAC/train_transit_v13_proximal_release.py",
        "SimpleSAC/transit_v10/__init__.py",
        "SimpleSAC/transit_v10/provenance.py",
        "SimpleSAC/transit_v10/trainer.py",
        "SimpleSAC/transit_v11/__init__.py",
        "SimpleSAC/transit_v11/provenance.py",
        "SimpleSAC/transit_v11/trainer.py",
        "SimpleSAC/transit_v12/__init__.py",
        "SimpleSAC/transit_v12/provenance.py",
        "SimpleSAC/transit_v12/trainer.py",
        "SimpleSAC/transit_v13/__init__.py",
        "SimpleSAC/transit_v13/provenance.py",
        "SimpleSAC/transit_v13/trainer.py",
    )
    return {relative: sha256(ROOT / relative) for relative in relative_paths}


def _verify_sealed_v12() -> dict[str, Path]:
    paths = {}
    for label, (path, expected) in SEALED_V12.items():
        resolved = path.resolve(strict=True)
        actual = sha256(resolved)
        if actual != expected:
            raise ValueError(f"sealed v12 input drifted for {label}: {actual}")
        paths[label] = resolved
    prefix = json.loads(paths["prefix_closeout"].read_text())
    prefix_verification = json.loads(paths["prefix_verification"].read_text())
    branch = json.loads(paths["branch_closeout"].read_text())
    branch_verification = json.loads(paths["branch_verification"].read_text())
    evaluation = json.loads(paths["evaluation_closeout"].read_text())
    evaluation_verification = json.loads(paths["evaluation_verification"].read_text())
    if (
        prefix.get("status") != "passed"
        or prefix.get("artifact_integrity_passed") is not True
        or prefix.get("exact_seed_matched_10k_handoffs_passed") is not True
        or prefix_verification.get("status") != "passed"
        or prefix_verification.get("verified_run_count") != 5
        or branch.get("status") != "passed"
        or branch.get("artifact_integrity_passed") is not True
        or branch.get("scientific_stability_passed_for_all_30_training_runs") is not True
        or branch_verification.get("status") != "passed"
        or evaluation.get("passed") is not True
        or evaluation.get("formal_confirmation_eligible") is not False
        or evaluation_verification.get("development_screening_gate_passed") is not False
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (
                prefix,
                prefix_verification,
                branch,
                branch_verification,
                evaluation,
                evaluation_verification,
            )
        )
    ):
        raise ValueError("sealed v12 closeout contract is not suitable for v13 development")
    return paths


def _build_rationale(
    sealed: dict[str, Path], control: Path, *, force: bool
) -> Path:
    analysis = json.loads(sealed["development_analysis"].read_text())
    result = analysis["development_scientific_result"]
    means = result["all_algorithm_mean_aulc"]
    failed = sorted(name for name, passed in result["checks"].items() if not passed)
    expected_failed = [
        "candidate_lower_than_sac_at_25000",
        "candidate_lower_than_v10_delayed_actor_at_25000",
        "candidate_lower_than_v11_at_25000",
        "relative_aulc_improvement_over_abrupt_control",
        "relative_aulc_improvement_over_sac",
        "relative_aulc_improvement_over_v10_delayed_actor",
        "relative_aulc_improvement_over_v11",
        "seed_wins_over_v11",
    ]
    if failed != sorted(expected_failed):
        raise ValueError("v12 failed-check diagnosis drifted")
    payload = {
        "schema_version": RATIONALE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v13-training-or-new-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "diagnosis": {
            "v12_artifact_integrity_passed": True,
            "v12_all_30_training_runs_stable": True,
            "v12_development_screening_passed": False,
            "v12_failed_checks": expected_failed,
            "mean_aulc": {
                key: means[key]
                for key in (
                    "h2oplus-v11-actor-hold",
                    "h2oplus-v12-abrupt-control",
                    "h2oplus-v12-actor-ramp",
                    "h2oplus-v12-temperature-ramp",
                    "h2oplus-v12-actor-anchor",
                    "h2oplus-v12-staged-combined",
                )
            },
            "best_v12_new_branch": "h2oplus-v12-actor-ramp",
            "state_anchor_harmed_mean_aulc": True,
            "mechanistic_hypothesis": (
                "the actor-LR ramp is the strongest v12 branch, while offline-state "
                "anchoring is harmful; a direct mild parameter-space pull can bound "
                "late actor drift without changing actor states or temperature updates"
            ),
        },
        "selected_intervention": {
            "common_base": "exact sealed seed-matched v12 10k handoff",
            "preserved_behavior": (
                "v12 actor-ramp with full temperature rate, online-only actor states, "
                "50/50 critic replay and unit ratio weights"
            ),
            "sole_new_factor": (
                "theta <- (1-rate)*theta + rate*theta_anchor after each actor update"
            ),
            "rates": PROXIMAL_RATES,
            "no_gradient_retention_at_25k": {
                name: proximal_retention(rate, REGISTERED_POST_10K_ACTOR_UPDATES)
                for name, rate in PROXIMAL_RATES.items()
            },
            "predesignated_candidate": PREDESIGNATED_CANDIDATE,
            "checkpoint_selection_by_behavior": False,
        },
        "sealed_v12_inputs": {
            label: descriptor(
                path, deployed_path=control / "sealed_v12" / f"{label}.json"
            )
            for label, path in sealed.items()
        },
        "claim_boundary": {
            "all_v7_through_v12_evidence_preserved": True,
            "supports_formal_held_out_effect_claim": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
        },
        "generator": descriptor(Path(__file__).resolve()),
    }
    path = REPRO / "h2oplus_v13_proximal_release_rationale_v1.json"
    write_stable(path, payload, force=force)
    return path


def _source_handoffs(
    workspace: Path, prefix_verification: Path
) -> dict[int, dict[str, Any]]:
    verified = {
        int(row["seed"]): row
        for row in json.loads(prefix_verification.read_text())["runs"]
    }
    records = {}
    source_root = REPRO / "h2oplus_v12_staged_release_results/prefix_v2"
    target_root = workspace / "reproducibility/h2oplus_v13_source_handoffs_v1"
    for seed in TRAINING_SEEDS:
        run_name = f"h2oplus_v12a2_shared_prefix_seed{seed}"
        manifest = source_root / run_name / "v12_handoff/manifest.json"
        payload = json.loads(manifest.read_text())
        row = verified[seed]
        if (
            payload.get("schema_version") != SOURCE_HANDOFF_SCHEMA
            or payload.get("passed") is not True
            or payload.get("formal_held_out_outcomes_read") is not False
            or payload.get("seed") != seed
            or payload.get("prefix_run_name") != run_name
            or payload.get("protocol_sha256") != V12_PROTOCOL_SHA256
            or payload.get("checkpoint_state_sha256") != row["checkpoint_state_sha256"]
            or payload.get("actor_tensor_sha256") != row["actor_tensor_sha256"]
            or payload.get("temperature_tensor_sha256")
            != row["temperature_tensor_sha256"]
            or checkpoint_state_sha256(manifest.parent / "checkpoints/latest.pt")
            != row["checkpoint_state_sha256"]
        ):
            raise ValueError(f"sealed v12 source handoff drifted for seed {seed}")
        for file_record in payload["files"]:
            source = manifest.parent / file_record["path"]
            if (
                not source.is_file()
                or source.stat().st_size != file_record["bytes"]
                or sha256(source) != file_record["sha256"]
            ):
                raise ValueError(f"v12 source handoff file drifted: {source}")
        records[seed] = {
            "source_manifest": manifest,
            "path": str(target_root / f"seed{seed}/v12_handoff/manifest.json"),
            "manifest_sha256": sha256(manifest),
            "prefix_run_name": run_name,
            "prefix_run_signature": payload["prefix_run_signature"],
            "v12_protocol_sha256": payload["protocol_sha256"],
            "v12_source_sha256": payload["v12_source_sha256"],
            "checkpoint_state_sha256": payload["checkpoint_state_sha256"],
            "actor_tensor_sha256": payload["actor_tensor_sha256"],
            "temperature_tensor_sha256": payload["temperature_tensor_sha256"],
            "online_events_actual": row["simulator_decision_events"],
        }
    return records


def _initializers(workspace: Path) -> dict[int, dict[str, Any]]:
    source_root = (
        REPRO
        / "deployments/h2oplus_v12_staged_release_pilot_v2/workspace/"
        "reproducibility/h2oplus_v12_initializers_v2"
    )
    target_root = workspace / "reproducibility/h2oplus_v13_initializers_v1"
    records = {}
    for seed in TRAINING_SEEDS:
        source = source_root / f"h2oplus_seed{seed}.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        extra = dict(payload["extra"])
        if (
            payload.get("artifact_schema") != "transit-offline-initializer-v7-v1"
            or extra.get("source_training_seed") != seed
            or extra.get("offline_updates_completed") != 50_000
        ):
            raise ValueError(f"v13 initializer source drifted for seed {seed}")
        records[seed] = {
            "source": source,
            "path": str(target_root / source.name),
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
            "extra_sha256": stable_object_sha256(extra),
            "source_run_signature": extra["source_run_signature"],
        }
    return records


def _command_arguments(
    *,
    workspace: Path,
    output_root: Path,
    run_name: str,
    run_signature: str,
    seed: int,
    admission: Path,
    initializer: dict[str, Any],
    handoff: dict[str, Any],
) -> list[str]:
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
        "--online-capacity", "30000",
        "--offline-updates", "50000",
        "--warmup-events", "5000",
        "--ratio-pretrain-updates", "1000",
        "--epochs", "40",
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
        "--branch-handoff-manifest", handoff["path"],
    ]


def _build_jobs(
    *,
    workspace: Path,
    deployment_root: Path,
    identity: dict[str, Any],
    initializers: dict[int, dict[str, Any]],
    handoffs: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    output_root = deployment_root / "remote_results"
    admission_root = workspace / "reproducibility/h2oplus_v13_control_v1/admissions"
    jobs = []
    for variant in VARIANTS:
        for seed in TRAINING_SEEDS:
            settings = {
                "stage": "branch",
                "warmup_events": 5_000,
                "epochs": 40,
                "offline_updates": 50_000,
                "expected_total_updates": 70_000,
                "online_event_cap": 25_000,
                "export_online_events": [25_000],
                "proximal_rate": PROXIMAL_RATES[variant],
            }
            run_name = f"h2oplus_v13a1_{variant.replace('-', '_')}_seed{seed}"
            signature = stable_object_sha256(
                {
                    **identity,
                    "variant": variant,
                    "seed": seed,
                    "settings": settings,
                    "initializer_sha256": initializers[seed]["sha256"],
                    "source_handoff_manifest_sha256": handoffs[seed][
                        "manifest_sha256"
                    ],
                    "source_checkpoint_state_sha256": handoffs[seed][
                        "checkpoint_state_sha256"
                    ],
                }
            )
            admission = admission_root / f"{run_name}.json"
            command = _command_arguments(
                workspace=workspace,
                output_root=output_root,
                run_name=run_name,
                run_signature=signature,
                seed=seed,
                admission=admission,
                initializer=initializers[seed],
                handoff=handoffs[seed],
            )
            jobs.append(
                {
                    "name": run_name,
                    "method": "h2oplus",
                    "variant": variant,
                    "seed": seed,
                    "run_signature": signature,
                    "stage": "branch",
                    "training_admission": str(admission),
                    "offline_initializer": {
                        key: value
                        for key, value in initializers[seed].items()
                        if key != "source"
                    },
                    "branch_handoff_manifest": handoffs[seed]["path"],
                    "prefix": {
                        "name": handoffs[seed]["prefix_run_name"],
                        "run_signature": handoffs[seed]["prefix_run_signature"],
                        "seed": seed,
                    },
                    "source_handoff": {
                        key: value
                        for key, value in handoffs[seed].items()
                        if key not in {"source_manifest", "path", "prefix_run_name", "prefix_run_signature"}
                    },
                    "command_arguments": command,
                    "milestones": [
                        {
                            "online_events": 25_000,
                            "learner_update_steps": 70_000,
                            "ratio_optimizer_steps": 6_000,
                        }
                    ],
                    "predesignated_candidate": variant == PREDESIGNATED_CANDIDATE,
                    **settings,
                }
            )
    return jobs


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"deployment copy hash mismatch: {destination}")


def _remove_unregistered_bytecode(workspace: Path, base_manifest: Path) -> None:
    base = json.loads(base_manifest.read_text())
    allowed = {
        row["path"]
        for row in base.get("entries", [])
        if row.get("type") == "file"
        and (
            Path(row["path"]).suffix in {".pyc", ".pyo"}
            or "__pycache__" in Path(row["path"]).parts
        )
    }
    for path in sorted(workspace.rglob("*"), reverse=True):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        is_bytecode = (
            path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.relative_to(workspace).parts
        )
        if is_bytecode and relative not in allowed:
            path.unlink()
    for path in sorted(workspace.rglob("__pycache__"), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> int:
    args = parse_args()
    formal_workspace = args.formal_workspace.expanduser().resolve(strict=True)
    base_manifest = formal_workspace / "deployment_manifest.json"
    if sha256(base_manifest) != BASE_MANIFEST_SHA256:
        raise ValueError("frozen base v7 deployment manifest drifted")
    sealed = _verify_sealed_v12()
    deployment_root = args.deployment_root.expanduser().resolve()
    workspace = deployment_root / "workspace"
    control = workspace / "reproducibility/h2oplus_v13_control_v1"
    source_handoffs = _source_handoffs(workspace, sealed["prefix_verification"])
    initializers = _initializers(workspace)
    rationale = _build_rationale(sealed, control, force=args.force)

    registration = REPRO / "h2oplus_v13_proximal_release_registration_v1.json"
    registration_payload = {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v13 proximal actor-release development pilot",
        "status": "registered-before-v13-training-or-new-development-evaluation",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "frozen_v7_through_v12_evidence_modified": False,
        "rationale": descriptor(rationale, deployed_path=control / "rationale.json"),
        "new_training_runs": 15,
        "reused_prefix_runs": 5,
        "rerun_prefix_runs": 0,
        "training_seeds": list(TRAINING_SEEDS),
        "variants": list(VARIANTS),
        "proximal_rates": PROXIMAL_RATES,
        "predesignated_candidate": PREDESIGNATED_CANDIDATE,
        "maximum_online_events": 25_000,
        "checkpoint_selection_by_behavior": False,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "v13_source_sha256": source_map(),
    }
    write_stable(registration, registration_payload, force=args.force)

    authorization = REPRO / "h2oplus_v13_proximal_release_authorization_v1.json"
    authorization_payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-development-only",
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "user_authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode()
        ).hexdigest(),
        "registration": descriptor(
            registration, deployed_path=control / "registration.json"
        ),
        "rationale": descriptor(rationale, deployed_path=control / "rationale.json"),
        "remote_training_authorized": True,
        "reuse_sealed_v12_10k_handoffs_authorized": True,
        "prefix_training_rerun_authorized": False,
        "remote_development_evaluation_after_training_closeout_authorized": True,
        "local_training_or_sumo_authorized": False,
        "formal_held_out_access_authorized": False,
        "formal_confirmation_stage_authorized": False,
    }
    write_stable(authorization, authorization_payload, force=args.force)

    identity = {
        "registration_sha256": sha256(registration),
        "authorization_sha256": sha256(authorization),
        "rationale_sha256": sha256(rationale),
        "v13_source_sha256": source_map(),
        "v12_evaluation_closeout_sha256": SEALED_V12["evaluation_closeout"][1],
        "v12_evaluation_verification_sha256": SEALED_V12[
            "evaluation_verification"
        ][1],
    }
    jobs = _build_jobs(
        workspace=workspace,
        deployment_root=deployment_root,
        identity=identity,
        initializers=initializers,
        handoffs=source_handoffs,
    )

    plan = REPRO / "h2oplus_v13_proximal_release_training_plan_v1.json"
    plan_payload = {
        "schema_version": PLAN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v13-training",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "workspace": str(workspace),
        "remote_output_root": str(deployment_root / "remote_results"),
        "registration": descriptor(
            registration, deployed_path=control / "registration.json"
        ),
        "authorization": descriptor(
            authorization, deployed_path=control / "authorization.json"
        ),
        "rationale": descriptor(rationale, deployed_path=control / "rationale.json"),
        "base_v7_training_deployment": descriptor(
            base_manifest, deployed_path=workspace / "deployment_manifest.json"
        ),
        "required_environment": {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
            "CPATH": str(workspace / "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"),
            "CUDA_VISIBLE_DEVICES": "scheduler-assigned exactly one physical GPU",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "v13_source_sha256": source_map(),
        "jobs": jobs,
    }
    write_stable(plan, plan_payload, force=args.force)

    protocol = REPRO / "h2oplus_v13_proximal_release_protocol_v1.json"
    protocol_payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v13-training-and-new-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_allowed": False,
        "registration": descriptor(
            registration, deployed_path=control / "registration.json"
        ),
        "authorization": descriptor(
            authorization, deployed_path=control / "authorization.json"
        ),
        "rationale": descriptor(rationale, deployed_path=control / "rationale.json"),
        "training_plan": descriptor(plan, deployed_path=control / "training_plan.json"),
        "base_v7_training_deployment": descriptor(
            base_manifest, deployed_path=workspace / "deployment_manifest.json"
        ),
        "v13_source_sha256": source_map(),
        "training_jobs": jobs,
        "proximal_release": {
            "new_branch_jobs": 15,
            "reused_sealed_v12_prefix_jobs": 5,
            "rerun_prefix_jobs": 0,
            "rates": PROXIMAL_RATES,
            "predesignated_candidate": PREDESIGNATED_CANDIDATE,
            "post_10k_actor_updates": REGISTERED_POST_10K_ACTOR_UPDATES,
            "development_evaluation_release_requires": (
                "all 15 branch tasks exact token-bound backend exit0, exact terminal "
                "artifacts, registered stability and independent training closeout"
            ),
        },
        "policy_budgets": {
            variant: {
                "export_online_events": [25_000],
                "common_evaluation_events": [0, 5_000, 10_000, 25_000],
                "milestones": [
                    {
                        "online_events": 25_000,
                        "learner_update_steps": 70_000,
                        "ratio_optimizer_steps": 6_000,
                    }
                ],
            }
            for variant in VARIANTS
        },
        "analysis_gates": {
            "predesignated_candidate": PREDESIGNATED_CANDIDATE,
            "candidate_relative_aulc_improvement_over_v12_actor_ramp_at_least": 0.01,
            "candidate_relative_aulc_improvement_over_v11_at_least": 0.01,
            "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least": 0.02,
            "candidate_relative_aulc_improvement_over_sac_at_least": 0.02,
            "candidate_relative_aulc_improvement_over_fixed_h2o_at_least": 0.03,
            "candidate_seed_wins_over_v12_actor_ramp_at_least": 4,
            "candidate_seed_wins_over_v11_at_least": 4,
            "candidate_seed_wins_over_v10_delayed_actor_at_least": 4,
            "candidate_seed_wins_over_sac_at_least": 4,
            "candidate_seed_wins_over_fixed_h2o_at_least": 4,
            "candidate_mean_lower_than_all_controls_at_25000": True,
            "completion_rate_noninferiority_margin": 0.003,
            "no_demand_scale_relative_aulc_degradation_over_sac_greater_than": 0.01,
            "all_15_training_runs_pass_registered_stability": True,
            "all_branches_share_exact_seed_matched_v12_10k_state": True,
        },
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    write_stable(protocol, protocol_payload, force=args.force)

    formal_admission = formal_workspace / (
        "reproducibility/event_cap_regate_v7_v1_remote/"
        "training_admission_v7_formal_v2_event_cap_fix_v1.json"
    )
    base_admission = json.loads(formal_admission.read_text())
    admissions_root = REPRO / "h2oplus_v13_proximal_release_training_admissions_v1"
    admissions = []
    for job in jobs:
        admission_payload = {
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
            "dataset": base_admission["dataset"],
            "simulator": base_admission["simulator"],
            "training_source_sha256": base_admission["training_source_sha256"],
            "v13_source_sha256": source_map(),
            "v13_wrapper": descriptor(ROOT / "SimpleSAC/train_transit_v13_proximal_release.py"),
            "v13_protocol": descriptor(protocol, deployed_path=control / "protocol.json"),
            "v13_rationale": descriptor(rationale, deployed_path=control / "rationale.json"),
            "base_v7_admission_source": descriptor(formal_admission),
            "offline_initializer": job["offline_initializer"],
            "source_v12_handoff": job["source_handoff"],
            "selected_arguments": job,
            "checks": {
                "base_v7_source_map_reused_exactly": True,
                "v13_overlay_bound_separately": True,
                "offline_initializer_exactly_bound": True,
                "sealed_v12_10k_handoff_exactly_bound": True,
                "formal_held_out_outcomes_read": False,
            },
        }
        path = admissions_root / f"{job['name']}.json"
        write_stable(path, admission_payload, force=args.force)
        admissions.append(path)

    freeze = REPRO / "h2oplus_v13_proximal_release_freeze_v1.json"
    freeze_payload = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-v13-training-and-new-development-outcomes",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "registration": descriptor(
            registration, deployed_path=control / "registration.json"
        ),
        "authorization": descriptor(
            authorization, deployed_path=control / "authorization.json"
        ),
        "rationale": descriptor(rationale, deployed_path=control / "rationale.json"),
        "training_plan": descriptor(plan, deployed_path=control / "training_plan.json"),
        "protocol": descriptor(protocol, deployed_path=control / "protocol.json"),
        "training_admissions": [
            descriptor(path, deployed_path=control / "admissions" / path.name)
            for path in admissions
        ],
        "v13_source_sha256": source_map(),
        "new_training_runs": 15,
        "prefix_runs_rerun": 0,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    write_stable(freeze, freeze_payload, force=args.force)

    if deployment_root.exists():
        if not args.force:
            raise FileExistsError(f"refusing to replace deployment: {deployment_root}")
        shutil.rmtree(deployment_root)
    deployment_root.mkdir(parents=True)
    shutil.copytree(formal_workspace, workspace, copy_function=os.link, symlinks=True)
    _remove_unregistered_bytecode(workspace, workspace / "deployment_manifest.json")

    mappings: dict[Path, Path] = {}
    for relative in source_map():
        mappings[ROOT / relative] = workspace / relative
    mappings.update(
        {
            ROOT / "reproducibility/verify_h2oplus_v13_proximal_release_deployment.py": workspace / "reproducibility/verify_h2oplus_v13_proximal_release_deployment.py",
            registration: control / "registration.json",
            authorization: control / "authorization.json",
            rationale: control / "rationale.json",
            plan: control / "training_plan.json",
            protocol: control / "protocol.json",
            freeze: control / "freeze.json",
        }
    )
    for admission in admissions:
        mappings[admission] = control / "admissions" / admission.name
    for label, path in sealed.items():
        mappings[path] = control / "sealed_v12" / f"{label}.json"
    for seed, initializer in initializers.items():
        mappings[initializer["source"]] = Path(initializer["path"])
    for seed, handoff in source_handoffs.items():
        source_manifest = handoff["source_manifest"]
        source_root = source_manifest.parent
        target_root = Path(handoff["path"]).parent
        payload = json.loads(source_manifest.read_text())
        mappings[source_manifest] = target_root / "manifest.json"
        for row in payload["files"]:
            mappings[source_root / row["path"]] = target_root / row["path"]
    for source, destination in mappings.items():
        _copy(source, destination)

    entries = [
        {
            "path": destination.relative_to(workspace).as_posix(),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
        for destination in sorted(mappings.values())
    ]
    deployment_payload = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "immutable-v13-v1-overlay-on-frozen-v7-workspace",
        "study_stage": "development",
        "formal_held_out_outcomes_included": False,
        "base_v7_training_deployment": descriptor(
            workspace / "deployment_manifest.json"
        ),
        "overlay_entry_count": len(entries),
        "overlay_aggregate_sha256": stable_object_sha256({"entries": entries}),
        "entries": entries,
    }
    deployment_manifest = workspace / "deployment_manifest_v13_v1.json"
    deployment_manifest.write_text(
        json.dumps(deployment_payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "registration": descriptor(registration),
                "authorization": descriptor(authorization),
                "rationale": descriptor(rationale),
                "training_plan": descriptor(plan),
                "protocol": descriptor(protocol),
                "freeze": descriptor(freeze),
                "deployment_manifest": descriptor(deployment_manifest),
                "overlay_entry_count": len(entries),
                "new_training_runs": 15,
                "prefix_runs_rerun": 0,
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
