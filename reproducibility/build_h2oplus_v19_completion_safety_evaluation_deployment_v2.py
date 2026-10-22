#!/usr/bin/env python3
"""Build or verify the protocol-environment-bound v19 deployment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility import (
    build_h2oplus_v19_completion_safety_evaluation_deployment as v1,
)


SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-deployment-manifest-v2"
AUDIT_SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-deployment-audit-v2"
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = v1.EXPECTED_PARENT_MANIFEST
EXPECTED_PARENT_AGGREGATE = v1.EXPECTED_PARENT_AGGREGATE
EXPECTED_V1_MANIFEST = (
    "24b3e9aeba8ace7b10356ff9fe6d1d935282f3d529cef9b527bf1469513b9787"
)
EXPECTED_V1_AUDIT = (
    "4b699882df1e8829bf7d937fe3093a410528a068c38cbcbe41f0518d71f85ed7"
)
EXPECTED = dict(v1.EXPECTED)
EXPECTED.update(
    {
        "source_freeze": "6c77e7117b1c7499635e66e05d500bbc77e01e769530c1fe0d04b514c5dc11d7",
        "correction": "2aec76694036ad038d2d15b4197c1444db8587171898ed0abdb7bdc410b4c282",
        "source_freeze_v1": "9d8619304dd8f433d568012a25777a8f9a774eea0a4ba56cf7c7fed70007be63",
        "deployment_v1_audit": EXPECTED_V1_AUDIT,
    }
)
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v19_completion_safety_reserve.py",
    "reproducibility/run_evaluation_matrix_v19_completion_safety.py",
    "reproducibility/analyze_h2oplus_v19_completion_safety.py",
    "reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py",
    "reproducibility/prepare_h2oplus_v19_completion_safety.py",
    "reproducibility/audit_h2oplus_v19_completion_safety_design.py",
    "reproducibility/verify_h2oplus_v19_completion_safety_design.py",
    "reproducibility/test_h2oplus_v19_completion_safety.py",
    "reproducibility/build_h2oplus_v19_completion_safety_source_freeze.py",
    "reproducibility/build_h2oplus_v19_completion_safety_source_freeze_v2.py",
    "reproducibility/h2oplus_v19_completion_safety_authorization_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_rationale_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_registration_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_protocol_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_environment_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_source_freeze_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_source_freeze_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_environment_binding_correction_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_design_audit_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_design_audit_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_design_verification_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_evaluation_deployment_audit_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_verification_v3.json",
)
PARENT_MANIFEST_COPY = (
    "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_deployment_manifest_v3.json"
)
V1_MANIFEST_COPY = (
    "reproducibility/h2oplus_v19_completion_safety_evaluation_deployment_manifest_v1.json"
)


def input_paths() -> dict[str, Path]:
    paths = v1.input_paths()
    paths["source_freeze"] = ROOT / "reproducibility/h2oplus_v19_completion_safety_source_freeze_v2.json"
    paths["correction"] = ROOT / "reproducibility/h2oplus_v19_completion_safety_environment_binding_correction_v2.json"
    paths["source_freeze_v1"] = ROOT / "reproducibility/h2oplus_v19_completion_safety_source_freeze_v1.json"
    paths["deployment_v1_audit"] = ROOT / "reproducibility/h2oplus_v19_completion_safety_evaluation_deployment_audit_v1.json"
    return paths


def validate_inputs() -> tuple[dict[str, Path], dict]:
    paths = input_paths()
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v19-v2 deployment input drifted: {name}")
    freeze = json.loads(paths["source_freeze"].read_text())
    correction = json.loads(paths["correction"].read_text())
    v1_audit = json.loads(paths["deployment_v1_audit"].read_text())
    if (
        freeze.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-source-freeze-v2"
        or freeze.get("evidence_revision")
        != "v2-exact-protocol-environment-binding"
        or freeze.get("v19_development_evaluation_outcomes_read") is not False
        or freeze.get("expected_matrix_source_file_count") != 55
        or freeze.get("matrix_source_count") != 16
        or freeze.get("operational_source_count") != 4
        or correction.get("execution_boundary", {}).get(
            "completed_scientific_result_count"
        )
        != 0
        or correction.get("execution_boundary", {}).get(
            "scientific_numeric_outcomes_read"
        )
        is not False
        or v1_audit.get("passed") is not True
        or v1_audit.get("local_or_remote_sumo_simulation_executed") is not False
    ):
        raise ValueError("v19-v2 deployment correction boundary drifted")
    for row in freeze["matrix_sources"] + freeze["operational_sources"]:
        if sha256((ROOT / row["path"]).resolve(strict=True)) != row["sha256"]:
            raise ValueError(f"v19-v2 frozen source drifted: {row['path']}")
    return paths, freeze


def clone_parent(parent: Path, output: Path) -> None:
    shutil.copytree(parent, output, copy_function=os.link, symlinks=True)
    manifest = output / MANIFEST_NAME
    if sha256(manifest) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("v18-v3 parent manifest drifted during v19-v2 clone")
    manifest.unlink()


def copy_overlay(parent_manifest: Path, v1_manifest: Path, output: Path) -> None:
    repo = output / "H2Oplus"
    for relative in NEW_REPO_FILES:
        source = (ROOT / relative).resolve(strict=True)
        destination = repo / relative
        if destination.exists():
            raise FileExistsError(f"v19-v2 overlay would replace parent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_scenarios = (ROOT / v1.SCENARIO_ROOT).resolve(strict=True)
    destination_scenarios = repo / v1.SCENARIO_ROOT
    if destination_scenarios.exists():
        raise FileExistsError(f"v19-v2 scenario overlay exists: {destination_scenarios}")
    shutil.copytree(source_scenarios, destination_scenarios)
    for source, relative in (
        (parent_manifest, PARENT_MANIFEST_COPY),
        (v1_manifest, V1_MANIFEST_COPY),
    ):
        destination = repo / relative
        if destination.exists():
            raise FileExistsError(f"v19-v2 manifest copy exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    v1_manifest_path = args.v1_deployment.expanduser().resolve(strict=True) / MANIFEST_NAME
    if sha256(parent_manifest_path) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("sealed v18-v3 parent manifest drifted")
    if sha256(v1_manifest_path) != EXPECTED_V1_MANIFEST:
        raise ValueError("superseded local v19-v1 manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 836
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v18-v3 parent deployment scope drifted")
    paths, freeze = validate_inputs()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v19-v2 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_overlay(parent_manifest_path, v1_manifest_path, output)
        repo = output / "H2Oplus"
        repo_path = lambda name: repo / paths[name].relative_to(ROOT)
        entries = [
            v1.descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        runtime_lock_path = repo_path("runtime_lock")
        runtime = json.loads(runtime_lock_path.read_text())
        scientific_sources = freeze["matrix_sources"] + freeze["operational_sources"]
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "total_file_bytes": sum(row["size_bytes"] for row in entries),
            "aggregate_sha256": v1.aggregate(entries),
            "training": False,
            "study_stage": "development",
            "screen_type": "paired-mechanism-screen",
            "adaptive_development_screen": True,
            "predeployment_environment_binding_correction_only": True,
            "scientific_behavior_unchanged_from_v19_v1": True,
            "v19_v1_completed_scientific_result_count": 0,
            "v19_v1_results_reused": False,
            "formal_held_out_outcomes_included": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "formal_confirmation_eligibility_created": False,
            "passing_only_permits_fresh_full_development_control_matrix": True,
            "physical_behavior_count": 15,
            "logical_method_count": 15,
            "new_actor_file_count": 0,
            "expected_new_remote_sumo_episodes": 135,
            "critical_matrix_source_file_count": 55,
            "scientific_source_file_count": len(scientific_sources),
            "sumo_runtime_signature": runtime["runtime_fingerprint"]["portable_signature"],
            "parent_v18_artifact_integrity_passed": True,
            "parent_training_stability_passed": True,
            "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
            "data_environment_material_factors_disclosed": True,
            "data_policy": v1.data_policy(entries),
            "parent_v18_v3_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
                "results_reused": False,
            },
            "superseded_local_v19_v1_deployment": {
                "manifest_path": str(v1_manifest_path),
                "manifest_sha256": EXPECTED_V1_MANIFEST,
                "audit_sha256": EXPECTED_V1_AUDIT,
                "remote_sync_started": False,
                "scheduler_task_submitted": False,
                "completed_scientific_result_count": 0,
                "results_reused": False,
            },
            "authorization": {"path": str(repo_path("authorization")), "sha256": EXPECTED["authorization"]},
            "rationale": {"path": str(repo_path("rationale")), "sha256": EXPECTED["rationale"]},
            "registration": {"path": str(repo_path("registration")), "sha256": EXPECTED["registration"]},
            "protocol": {"path": str(repo_path("protocol")), "sha256": EXPECTED["protocol"]},
            "correction": {"path": str(repo_path("correction")), "sha256": EXPECTED["correction"]},
            "methods_manifest": {"path": str(repo_path("methods")), "sha256": EXPECTED["methods"]},
            "environment_design": {"path": str(repo_path("environment")), "sha256": EXPECTED["environment"]},
            "source_freeze": {"path": str(repo_path("source_freeze")), "sha256": EXPECTED["source_freeze"]},
            "parent_v18_audit": {"path": str(repo_path("v18_audit")), "sha256": EXPECTED["v18_audit"]},
            "parent_v18_closeout": {"path": str(repo_path("v18_closeout")), "sha256": EXPECTED["v18_closeout"]},
            "parent_v18_verification": {"path": str(repo_path("v18_verification")), "sha256": EXPECTED["v18_verification"]},
            "dataset": {"path": str(repo_path("dataset")), "sha256": EXPECTED["dataset"]},
            "sumo_runtime_lock": {"path": str(runtime_lock_path), "sha256": EXPECTED["runtime_lock"]},
            "evaluation_source_sha256": {row["path"]: row["sha256"] for row in scientific_sources},
            "matrix_source_sha256": {row["path"]: row["sha256"] for row in freeze["matrix_sources"]},
            "entries": entries,
        }
        manifest_path = output / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "manifest_sha256": sha256(manifest_path),
                    "aggregate_sha256": manifest["aggregate_sha256"],
                    "entry_count": manifest["entry_count"],
                    "total_file_bytes": manifest["total_file_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


def verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    errors: list[str] = []
    manifest: dict = {}
    try:
        _, freeze = validate_inputs()
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if manifest.get("schema_version") != SCHEMA or v1.aggregate(entries) != manifest.get("aggregate_sha256"):
            raise ValueError("v19-v2 manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v19-v2 deployment membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v19-v2 file drifted: {row['path']}")
        v1.data_policy(entries)
        expected_sources = {
            row["path"]: row["sha256"]
            for row in freeze["matrix_sources"] + freeze["operational_sources"]
        }
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("predeployment_environment_binding_correction_only") is not True
            or manifest.get("scientific_behavior_unchanged_from_v19_v1") is not True
            or manifest.get("v19_v1_completed_scientific_result_count") != 0
            or manifest.get("v19_v1_results_reused") is not False
            or manifest.get("formal_held_out_outcomes_included") is not False
            or manifest.get("formal_confirmation_eligibility_created") is not False
            or manifest.get("physical_behavior_count") != 15
            or manifest.get("expected_new_remote_sumo_episodes") != 135
            or manifest.get("critical_matrix_source_file_count") != 55
            or manifest.get("scientific_source_file_count") != 20
            or manifest.get("source_freeze", {}).get("sha256") != EXPECTED["source_freeze"]
            or manifest.get("correction", {}).get("sha256") != EXPECTED["correction"]
            or manifest.get("registration", {}).get("sha256") != EXPECTED["registration"]
            or manifest.get("methods_manifest", {}).get("sha256") != EXPECTED["methods"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256") != EXPECTED["runtime_lock"]
            or manifest.get("parent_v18_v3_deployment", {}).get("sha256") != EXPECTED_PARENT_MANIFEST
            or manifest.get("superseded_local_v19_v1_deployment", {}).get("manifest_sha256") != EXPECTED_V1_MANIFEST
            or manifest.get("evaluation_source_sha256") != expected_sources
        ):
            raise ValueError("v19-v2 deployment scope drifted")
    except Exception as error:
        errors.append(str(error))
    report = {
        "schema_version": AUDIT_SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "root": str(root),
        "manifest_sha256": sha256(manifest_path),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "entry_count": manifest.get("entry_count"),
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
    }
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite v19-v2 audit: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument(
        "--parent-workspace",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v18_empty_tail_reserve_evaluation_v3/workspace",
    )
    build_parser.add_argument(
        "--v1-deployment",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v19_completion_safety_evaluation_v1/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v19_completion_safety_evaluation_v2/workspace",
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v19_completion_safety_evaluation_v2/workspace",
    )
    verify_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
