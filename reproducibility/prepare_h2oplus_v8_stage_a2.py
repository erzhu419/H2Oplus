#!/usr/bin/env python3
"""Register and freeze the authorized v8 Stage A2 H2O+ versus SAC screen."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    V7_POLICY_SELECTION_RULE,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-registration-v1"
AUTHORIZATION_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-authorization-v1"
EXPORT_SCHEMA = "transit-h2oplus-v8-stage-a2-policy-export-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-freeze-v1"
METHODS_NAME = "evaluation_methods_v8_stage_a2.json"
EXPORT_MANIFEST_NAME = "policy_export_manifest_v8_stage_a2.json"
EXPECTED = {
    "baseline_manifest": "9d35a4397c758b06b03f1c4e399e727d116c6a9198656ba2b13a49f4a5e2826f",
    "baseline_audit": "52c88d9d33a4aae518a8bf3889f7cedd4e3e971e84e18d6b23fcd3b951b53a12",
    "development_design": "d4a40a900cf9f72043c1d7c2067d02d6c79f2081f856b38cb257ec2ad0d18f3c",
    "stage_a_file_manifest": "a424b65d330d8338a2aaaf1f7af3c381ef787a6c85267553ff9e7dec8a6ceb8d",
    "stage_a_verification": "bfc52b45946b66440e27e4248d5a790e9b9cab9e2053cd6b60c0c37bf1dacf3c",
    "stage_a_analysis": "20c8532b264d3f0048bf2b654ea6a1968aa4c272fcc87e48c989e578bf4e95c8",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "runtime_requirements": "bea74923890a4b990418fdad6f93c55a298c7c22e3c2d0a9f3b7ad3c44e21a7c",
}
USER_AUTHORIZATION_TEXT = "那就继续 都批准"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repro = ROOT / "reproducibility"
    parser.add_argument(
        "--registration",
        type=Path,
        default=repro / "h2oplus_v8_stage_a2_registration_v1.json",
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        default=repro / "h2oplus_v8_stage_a2_execution_authorization_v1.json",
    )
    parser.add_argument(
        "--environment-design",
        type=Path,
        default=repro / "h2oplus_v8_development_environment_stage_a_v1.json",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=repro / "formal_results_v7_v7r1" / "baseline_manifest.json",
    )
    parser.add_argument(
        "--baseline-audit",
        type=Path,
        default=repro / "formal_results_v7_v7r1" / "baseline_matrix_audit.json",
    )
    parser.add_argument(
        "--stage-a-file-manifest",
        type=Path,
        default=repro / "h2oplus_v8_stage_a_results" / "stage_a_file_manifest.json",
    )
    parser.add_argument(
        "--stage-a-verification",
        type=Path,
        default=repro
        / "h2oplus_v8_stage_a_results"
        / "stage_a_file_manifest_verification.json",
    )
    parser.add_argument(
        "--stage-a-analysis",
        type=Path,
        default=repro / "h2oplus_v8_stage_a_results" / "stage_a_gate_analysis.json",
    )
    parser.add_argument(
        "--policy-export",
        type=Path,
        default=repro / "h2oplus_v8_stage_a2_policy_export_v1",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=repro / "h2oplus_v8_stage_a2_freeze_v1.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
    )
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=repro / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    )
    parser.add_argument(
        "--runtime-requirements",
        type=Path,
        default=repro / "h2oplus_v8_stage_a_sumo_runtime_requirements_py38.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": os.path.relpath(resolved, base.resolve()) if base else str(resolved),
        "sha256": sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _stable_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text())
        timestamp_keys = ("created_utc", "registered_utc", "approved_utc")
        old = {key: previous.get(key) for key in timestamp_keys}
        previous_without_time = {
            key: value for key, value in previous.items() if key not in timestamp_keys
        }
        payload_without_time = {
            key: value for key, value in payload.items() if key not in timestamp_keys
        }
        if previous_without_time != payload_without_time and not force:
            raise FileExistsError(f"refusing to replace a different file: {path}")
        for key, value in old.items():
            if isinstance(value, str):
                payload[key] = value
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _bound(path: Path, expected: str, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} hash drifted: expected {expected}, got {actual}")
    return resolved


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"different frozen export exists: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"copy hash mismatch: {destination}")


def _manifest_entry(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    entries = manifest.get("merged_results", {}).get("files_manifest", [])
    matches = [entry for entry in entries if entry.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"baseline manifest lacks unique entry {relative_path}")
    return matches[0]


def _validate_inputs(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "baseline_manifest": _bound(
            args.baseline_manifest, EXPECTED["baseline_manifest"], "baseline manifest"
        ),
        "baseline_audit": _bound(
            args.baseline_audit, EXPECTED["baseline_audit"], "baseline audit"
        ),
        "development_design": _bound(
            args.environment_design,
            EXPECTED["development_design"],
            "development design",
        ),
        "stage_a_file_manifest": _bound(
            args.stage_a_file_manifest,
            EXPECTED["stage_a_file_manifest"],
            "Stage A file manifest",
        ),
        "stage_a_verification": _bound(
            args.stage_a_verification,
            EXPECTED["stage_a_verification"],
            "Stage A independent verification",
        ),
        "stage_a_analysis": _bound(
            args.stage_a_analysis, EXPECTED["stage_a_analysis"], "Stage A analysis"
        ),
        "dataset": _bound(args.dataset, EXPECTED["dataset"], "canonical dataset"),
        "runtime_lock": _bound(
            args.runtime_lock, EXPECTED["runtime_lock"], "SUMO runtime lock"
        ),
        "runtime_requirements": _bound(
            args.runtime_requirements,
            EXPECTED["runtime_requirements"],
            "runtime requirements",
        ),
    }
    baseline = json.loads(paths["baseline_manifest"].read_text())
    audit = json.loads(paths["baseline_audit"].read_text())
    stage_a = json.loads(paths["stage_a_file_manifest"].read_text())
    verification = json.loads(paths["stage_a_verification"].read_text())
    design = json.loads(paths["development_design"].read_text())
    if (
        baseline.get("artifact_integrity_verified") is not True
        or baseline.get("status") != "verified"
        or audit.get("passed") is not True
        or audit.get("jobs_audited") != 35
        or stage_a.get("artifact_integrity", {}).get("passed") is not True
        or stage_a.get("artifact_integrity", {}).get("result_count") != 150
        or verification.get("passed") is not True
        or verification.get("formal_held_out_outcomes_read") is not False
        or design.get("study_stage") != "development"
        or design.get("disjointness", {}).get("verified_disjoint") is not True
        or design.get("disjointness", {}).get("formal_held_out_outcomes_read")
        is not False
    ):
        raise ValueError("A2 source evidence is incomplete or outside development scope")
    if any(
        value
        for key, value in design.get("disjointness", {}).items()
        if key.endswith("_overlap")
    ):
        raise ValueError("development design has a frozen-input overlap")
    return paths


def _registration_payload(paths: dict[str, Path]) -> dict[str, Any]:
    design = json.loads(paths["development_design"].read_text())
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "H2O+ v8 Stage A2 development SAC-baseline confirmation",
        "status": "registered-not-submitted",
        "execution_authorized": False,
        "submission_requires_explicit_user_approval": True,
        "local_sumo_or_training_allowed": False,
        "training": False,
        "formal_held_out_outcomes_read": False,
        "formal_held_out_may_be_read_for_selection": False,
        "purpose": (
            "determine whether the already-frozen H2O+ lambda=0.1 policies "
            "outperform SAC-mixed and SAC-online on the disjoint development set"
        ),
        "new_policies": ["sac-mixed", "sac-online"],
        "reference_policy": "h2oplus",
        "training_seeds": list(range(5101, 5106)),
        "development_environment": {
            "design": _descriptor(paths["development_design"]),
            "blocks": [
                {
                    "name": block["name"],
                    "sumo_seed": block["sumo_seed"],
                    "demand_realization_seed": block["demand_realization_seed"],
                }
                for block in design["blocks"]
            ],
            "demand_scales": design["demand_scales"],
            "environment_cells": design["environment_cells"],
            "verified_disjoint": True,
        },
        "matrix": {
            "new_sac_methods": 10,
            "new_sumo_episodes": 150,
            "reused_h2oplus_sumo_episodes": 75,
            "shards": 2,
            "workers_per_shard": 12,
            "assignment": "sorted-portable-job-key-round-robin-v1 within each block",
        },
        "endpoint": {
            "primary": "generalized_seconds_per_passenger",
            "direction": "lower-is-better",
            "paired_unit": "training seed x development environment block",
            "secondary_constraint": "passenger_completion_rate",
        },
        "multiple_testing": {
            "family": ["h2oplus-vs-sac-mixed", "h2oplus-vs-sac-online"],
            "procedure": "Holm step-down",
            "familywise_alpha": 0.05,
            "one_sided_alternative": "H2O+ has lower generalized seconds per passenger",
        },
        "comparison_gate": {
            "directional_mean_improvement_greater_than": 0.0,
            "mean_relative_improvement_at_least": 0.02,
            "training_seeds_with_lower_paired_mean_at_least": 4,
            "holm_adjusted_one_sided_p_less_than": 0.05,
            "completion_rate_noninferiority_margin": 0.005,
            "no_demand_scale_mean_relative_degradation_greater_than": 0.01,
        },
        "interpretation": {
            "directional_superiority": "positive mean effect and Holm-corrected p<0.05",
            "practical_confirmation": (
                "directional superiority plus >=2% mean improvement, 4/5 seed "
                "robustness, completion noninferiority, and no scale degradation >1%"
            ),
            "development_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "stage_b_authorized_or_submitted": False,
        },
        "existing_reference_evidence": {
            "stage_a_file_manifest": _descriptor(paths["stage_a_file_manifest"]),
            "stage_a_independent_verification": _descriptor(
                paths["stage_a_verification"]
            ),
            "stage_a_analysis": _descriptor(paths["stage_a_analysis"]),
            "note": (
                "the failed lambda0.5 candidate gate does not answer H2O+ versus SAC; "
                "only the 75 sealed H2O+ rows are reused"
            ),
        },
        "source_training_evidence": {
            "baseline_manifest": _descriptor(paths["baseline_manifest"]),
            "baseline_matrix_audit": _descriptor(paths["baseline_audit"]),
        },
        "stop_rules": [
            "stop on nonzero exit, nonfinite result, or source/plan/dispatch mismatch",
            "retain every registered seed and final checkpoint",
            "never use formal held-out outcomes for development selection",
            "never train or run SUMO locally",
            "do not launch Stage B from this registration",
        ],
        "scheduler_task_ids": [],
        "generator": _descriptor(Path(__file__).resolve()),
    }


def _authorization_payload(registration: Path) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_decision": "authorize-stage-a2-registration-and-execution",
        "approved_scope": "A2-H2Oplus-vs-SAC-development-baseline-confirmation",
        "authorization_channel": "explicit user message in the current Codex task",
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_text_sha256": hashlib.sha256(
            USER_AUTHORIZATION_TEXT.encode()
        ).hexdigest(),
        "registration": _descriptor(registration, registration.parent),
        "formal_held_out_access_authorized": False,
        "local_sumo_or_training_authorized": False,
        "stage_b_authorized": False,
    }


def _build_policy_export(
    args: argparse.Namespace,
    paths: dict[str, Path],
    registration: Path,
    authorization: Path,
) -> tuple[Path, Path]:
    destination = args.policy_export.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads(paths["baseline_manifest"].read_text())
    plan = Path(baseline_manifest["frozen_inputs"]["plan"]["path"]).resolve(strict=True)
    plan_hash = baseline_manifest["frozen_inputs"]["plan"]["sha256"]
    if sha256(plan) != plan_hash:
        raise ValueError("baseline training plan hash drifted")
    source_root = (
        ROOT / "reproducibility/formal_results_v7_v7r1/server_baselines"
    ).resolve()
    methods: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for algorithm in ("sac-mixed", "sac-online"):
        for seed in range(5101, 5106):
            run_name = f"{algorithm}_seed{seed}"
            run_dir = source_root / run_name
            actor = run_dir / "checkpoints/final_actor.pt"
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "summary.json"
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            actor_hash = sha256(actor)
            manifest_entry = _manifest_entry(
                baseline_manifest, f"{run_name}/checkpoints/final_actor.pt"
            )
            if (
                manifest_entry.get("sha256") != actor_hash
                or manifest_entry.get("bytes") != actor.stat().st_size
                or summary.get("completed_epochs") != 190
                or summary.get("total_updates") != 95000
                or summary.get("final_policy", {}).get("sha256") != actor_hash
                or metadata.get("run_signature") != summary.get("run_signature")
            ):
                raise ValueError(f"incomplete or unbound baseline policy: {run_name}")
            admission = metadata.get("training_admission", {}).get("sha256")
            if not isinstance(admission, str) or len(admission) != 64:
                raise ValueError(f"missing training admission for {run_name}")
            exported_actor = (
                destination / "policies" / algorithm / f"seed{seed}" / "final_actor.pt"
            )
            exported_metadata = (
                destination / "provenance" / algorithm / f"seed{seed}" / "metadata.json"
            )
            exported_summary = (
                destination / "provenance" / algorithm / f"seed{seed}" / "summary.json"
            )
            for source, target in (
                (actor, exported_actor),
                (metadata_path, exported_metadata),
                (summary_path, exported_summary),
            ):
                _copy_exact(source, target)
                files.append(_descriptor(target, destination))
            methods.append(
                {
                    "name": f"{algorithm}-seed{seed}",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "type": "checkpoint",
                    "checkpoint": os.path.relpath(exported_actor, destination),
                    "training_run_signature": summary["run_signature"],
                    "training_admission_sha256": admission,
                    "training_checkpoint_sha256": summary["checkpoint"]["sha256"],
                    "training_plan_sha256": plan_hash,
                }
            )
    methods.sort(key=lambda item: (item["algorithm"], item["training_seed"]))
    methods_payload = {
        "schema_version": METHODS_SCHEMAS["v7"],
        "study_stage": "formal",
        "selection_rule": V7_POLICY_SELECTION_RULE,
        "scope": (
            "all final actors from the independently audited v7 SAC baselines, "
            "evaluated only on the disjoint v8 development screen"
        ),
        "development_registration_sha256": sha256(registration),
        "methods": methods,
    }
    methods_path = destination / METHODS_NAME
    _stable_json(methods_path, methods_payload, force=args.force)
    export_payload = {
        "schema_version": EXPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-stage-A2-SAC-outcomes",
        "selection_rule": V7_POLICY_SELECTION_RULE,
        "registration": _descriptor(registration),
        "execution_authorization": _descriptor(authorization),
        "methods_manifest": _descriptor(methods_path, destination),
        "source_baseline_manifest": _descriptor(paths["baseline_manifest"]),
        "source_baseline_audit": _descriptor(paths["baseline_audit"]),
        "payload_policy": {
            "compact_final_actor_files": 10,
            "metadata_files": 10,
            "summary_files": 10,
            "training_checkpoint_files": 0,
            "optimizer_or_replay_files": 0,
        },
        "methods": methods,
        "files": sorted(files, key=lambda item: item["path"]),
        "generator": _descriptor(Path(__file__).resolve()),
    }
    export_path = destination / EXPORT_MANIFEST_NAME
    _stable_json(export_path, export_payload, force=args.force)
    return methods_path, export_path


def _freeze_payload(
    paths: dict[str, Path],
    registration: Path,
    authorization: Path,
    methods: Path,
    export: Path,
) -> dict[str, Any]:
    registration_payload = json.loads(registration.read_text())
    return {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-stage-A2-SAC-outcomes",
        "stage": "A2-H2Oplus-vs-SAC-development-baseline-confirmation",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "registration": _descriptor(registration),
        "execution_authorization": _descriptor(authorization),
        "environment_design": _descriptor(paths["development_design"]),
        "methods_manifest": _descriptor(methods),
        "policy_export_manifest": _descriptor(export),
        "dataset": _descriptor(paths["dataset"]),
        "sumo_runtime_lock": _descriptor(paths["runtime_lock"]),
        "runtime_requirements": _descriptor(paths["runtime_requirements"]),
        "baseline_manifest": _descriptor(paths["baseline_manifest"]),
        "baseline_audit": _descriptor(paths["baseline_audit"]),
        "reference_stage_a_file_manifest": _descriptor(
            paths["stage_a_file_manifest"]
        ),
        "reference_stage_a_verification": _descriptor(
            paths["stage_a_verification"]
        ),
        "reference_stage_a_analysis": _descriptor(paths["stage_a_analysis"]),
        "analysis_source": _descriptor(
            ROOT / "reproducibility/analyze_h2oplus_v8_stage_a2.py"
        ),
        "matrix": registration_payload["matrix"],
        "endpoint": registration_payload["endpoint"],
        "multiple_testing": registration_payload["multiple_testing"],
        "comparison_gate": registration_payload["comparison_gate"],
        "reference_reuse_policy": (
            "reuse exactly the 75 sealed H2O+ rows from the verified Stage A "
            "manifest; do not rerun H2O+"
        ),
        "stage_b_authorized_or_submitted": False,
        "generator": _descriptor(Path(__file__).resolve()),
    }


def main() -> int:
    args = parse_args()
    paths = _validate_inputs(args)
    registration = args.registration.expanduser().resolve()
    authorization = args.authorization.expanduser().resolve()
    freeze = args.freeze.expanduser().resolve()
    _stable_json(registration, _registration_payload(paths), force=args.force)
    _stable_json(
        authorization, _authorization_payload(registration), force=args.force
    )
    methods, export = _build_policy_export(
        args, paths, registration, authorization
    )
    _stable_json(
        freeze,
        _freeze_payload(paths, registration, authorization, methods, export),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "registration": _descriptor(registration),
                "authorization": _descriptor(authorization),
                "methods_manifest": _descriptor(methods),
                "policy_export_manifest": _descriptor(export),
                "freeze": _descriptor(freeze),
                "new_sumo_episodes": 150,
                "training": False,
                "formal_held_out_outcomes_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
