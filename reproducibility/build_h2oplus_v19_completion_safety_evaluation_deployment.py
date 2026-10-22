#!/usr/bin/env python3
"""Build or verify the frozen v19 completion-safety deployment."""

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
from reproducibility.build_h2oplus_v11_actor_hold_evaluation_deployment import (
    aggregate,
    descriptor,
)


SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
SCENARIO_ROOT = "bus_h2o/development_scenarios_v19/completion_safety_v1"
EXPECTED_PARENT_MANIFEST = (
    "51476b297ea807efe2e2cd5073818e34ae1e241237f125b560aa0e45b3820456"
)
EXPECTED_PARENT_AGGREGATE = (
    "55c4dbc4a49ea9489ece540566aef9dc0cace9672bd7d5feeb408b8dce4f1fa3"
)
EXPECTED = {
    "authorization": "d81add0e13aed729412950f79a947aa8dee8e18f1df4eb3fcd5e622992e419c8",
    "rationale": "ee16744487f43645babf9322ea42ffa016ddd3b9fe12a866d5adf11198af9c0b",
    "registration": "536741a185e2d9b10c99ed821292a51c64bf64e8cb35771075ab787565246fd4",
    "protocol": "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184",
    "methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "environment": "5cfb1318418b64e619d4be727c41a0ce0eccab8c096020482586d8286cee3a05",
    "source_freeze": "9d8619304dd8f433d568012a25777a8f9a774eea0a4ba56cf7c7fed70007be63",
    "design_audit_v1": "c9edb3bb264b3d3984693599d3641ef769eb31a9b207a49eecd0264af7fe33f2",
    "design_audit_v2": "2238ceb015a456b80c2be5d4d06ebd96d2293fdbc88cadcfb08f087d2dafe1e8",
    "design_verification_v2": "3f8c0e03514b2fb6e2ed46249cfe8260cfdc3c9a2872316f6575de1504543a78",
    "v18_audit": "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    "v18_closeout": "e2a0285e9d95bc999cbe371ac261afb1f4526ff7a6722ba2d9d114f67c4c401c",
    "v18_verification": "6a08fa047a4616589fba04cd39eec41e9450ef1ce1fd100933039cb7354fa35a",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
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
    "reproducibility/h2oplus_v19_completion_safety_authorization_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_rationale_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_registration_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_protocol_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_environment_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_source_freeze_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_design_audit_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_design_audit_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_design_verification_v2.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_verification_v3.json",
)
PARENT_MANIFEST_COPY = (
    "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_deployment_manifest_v3.json"
)


def input_paths() -> dict[str, Path]:
    return {
        "authorization": ROOT / "reproducibility/h2oplus_v19_completion_safety_authorization_v1.json",
        "rationale": ROOT / "reproducibility/h2oplus_v19_completion_safety_rationale_v1.json",
        "registration": ROOT / "reproducibility/h2oplus_v19_completion_safety_registration_v1.json",
        "protocol": ROOT / "reproducibility/h2oplus_v19_completion_safety_protocol_v1.json",
        "methods": ROOT / "reproducibility/h2oplus_v19_completion_safety_evaluation_methods_v1.json",
        "environment": ROOT / "reproducibility/h2oplus_v19_completion_safety_environment_v1.json",
        "source_freeze": ROOT / "reproducibility/h2oplus_v19_completion_safety_source_freeze_v1.json",
        "design_audit_v1": ROOT / "reproducibility/h2oplus_v19_completion_safety_design_audit_v1.json",
        "design_audit_v2": ROOT / "reproducibility/h2oplus_v19_completion_safety_design_audit_v2.json",
        "design_verification_v2": ROOT / "reproducibility/h2oplus_v19_completion_safety_design_verification_v2.json",
        "v18_audit": ROOT / "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
        "v18_closeout": ROOT / "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
        "v18_verification": ROOT / "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_verification_v3.json",
        "data_environment_audit": ROOT / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": ROOT / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }


def validate_inputs() -> tuple[dict[str, Path], dict]:
    paths = input_paths()
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v19 deployment input drifted: {name}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    freeze = payloads["source_freeze"]
    if (
        freeze.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-source-freeze-v1"
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("v19_development_evaluation_outcomes_read") is not False
        or freeze.get("expected_matrix_source_file_count") != 55
        or freeze.get("matrix_source_count") != 16
        or freeze.get("operational_source_count") != 4
        or any(
            payloads[name].get("passed") is not True
            for name in (
                "v18_audit",
                "v18_closeout",
                "v18_verification",
                "design_verification_v2",
                "data_environment_verification",
            )
        )
        or payloads["data_environment_audit"].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
    ):
        raise ValueError("v19 deployment scientific scope drifted")
    for row in freeze["matrix_sources"] + freeze["operational_sources"]:
        if sha256((ROOT / row["path"]).resolve(strict=True)) != row["sha256"]:
            raise ValueError(f"v19 frozen source drifted: {row['path']}")
    return paths, freeze


def clone_parent(parent: Path, output: Path) -> None:
    shutil.copytree(parent, output, copy_function=os.link, symlinks=True)
    manifest = output / MANIFEST_NAME
    if sha256(manifest) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("v18-v3 parent manifest drifted during clone")
    manifest.unlink()


def copy_overlay(parent_manifest: Path, output: Path) -> None:
    repo = output / "H2Oplus"
    for relative in NEW_REPO_FILES:
        source = (ROOT / relative).resolve(strict=True)
        destination = repo / relative
        if destination.exists():
            raise FileExistsError(f"v19 overlay would replace parent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_scenarios = (ROOT / SCENARIO_ROOT).resolve(strict=True)
    destination_scenarios = repo / SCENARIO_ROOT
    if destination_scenarios.exists():
        raise FileExistsError(f"v19 scenario overlay exists: {destination_scenarios}")
    shutil.copytree(source_scenarios, destination_scenarios)
    destination = repo / PARENT_MANIFEST_COPY
    if destination.exists():
        raise FileExistsError(f"v18-v3 manifest copy already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_manifest, destination)


def data_policy(entries: list[dict]) -> dict:
    hdf5 = [row for row in entries if Path(row["path"]).suffix.lower() in {".h5", ".hdf5"}]
    actors = [row for row in entries if Path(row["path"]).suffix.lower() == ".pt"]
    forbidden = [
        row["path"]
        for row in entries
        if Path(row["path"]).suffix.lower() in {".pth", ".ckpt", ".pkl"}
        or "remote_results" in Path(row["path"]).parts
        or "evaluation_partition" in Path(row["path"]).parts
        or Path(row["path"]).name.startswith("sumo_seed")
    ]
    scenario_files = [row for row in entries if SCENARIO_ROOT in row["path"]]
    if (
        len(hdf5) != 1
        or hdf5[0]["sha256"] != EXPECTED["dataset"]
        or len(actors) != 85
        or len(scenario_files) != 63
        or forbidden
    ):
        raise ValueError(
            "v19 deployment data policy drifted: "
            f"hdf5={len(hdf5)}, pt={len(actors)}, "
            f"scenario_files={len(scenario_files)}, forbidden={forbidden}"
        )
    return {
        "hdf5_files": 1,
        "compact_actor_files": 85,
        "new_actor_files": 0,
        "fresh_v19_scenario_files": 63,
        "training_checkpoint_or_optimizer_files": 0,
        "physical_evaluation_result_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": EXPECTED["dataset"],
    }


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    if sha256(parent_manifest_path) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("sealed v18-v3 parent manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 836
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v18-v3 parent deployment scope drifted")
    paths, source_freeze = validate_inputs()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v19 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_overlay(parent_manifest_path, output)
        repo = output / "H2Oplus"
        repo_path = lambda name: repo / paths[name].relative_to(ROOT)
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        runtime_lock_path = repo_path("runtime_lock")
        runtime = json.loads(runtime_lock_path.read_text())
        scientific_sources = (
            source_freeze["matrix_sources"] + source_freeze["operational_sources"]
        )
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "total_file_bytes": sum(row["size_bytes"] for row in entries),
            "aggregate_sha256": aggregate(entries),
            "training": False,
            "study_stage": "development",
            "screen_type": "paired-mechanism-screen",
            "adaptive_development_screen": True,
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
            "data_policy": data_policy(entries),
            "parent_v18_v3_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
                "results_reused": False,
            },
            "parent_v18_v3_deployment_manifest_copy": {
                "path": str(repo / PARENT_MANIFEST_COPY),
                "sha256": EXPECTED_PARENT_MANIFEST,
            },
            "authorization": {"path": str(repo_path("authorization")), "sha256": EXPECTED["authorization"]},
            "rationale": {"path": str(repo_path("rationale")), "sha256": EXPECTED["rationale"]},
            "registration": {"path": str(repo_path("registration")), "sha256": EXPECTED["registration"]},
            "protocol": {"path": str(repo_path("protocol")), "sha256": EXPECTED["protocol"]},
            "methods_manifest": {"path": str(repo_path("methods")), "sha256": EXPECTED["methods"]},
            "environment_design": {"path": str(repo_path("environment")), "sha256": EXPECTED["environment"]},
            "source_freeze": {"path": str(repo_path("source_freeze")), "sha256": EXPECTED["source_freeze"]},
            "design_audit_v1": {"path": str(repo_path("design_audit_v1")), "sha256": EXPECTED["design_audit_v1"]},
            "design_audit_v2": {"path": str(repo_path("design_audit_v2")), "sha256": EXPECTED["design_audit_v2"]},
            "design_verification_v2": {"path": str(repo_path("design_verification_v2")), "sha256": EXPECTED["design_verification_v2"]},
            "parent_v18_audit": {"path": str(repo_path("v18_audit")), "sha256": EXPECTED["v18_audit"]},
            "parent_v18_closeout": {"path": str(repo_path("v18_closeout")), "sha256": EXPECTED["v18_closeout"]},
            "parent_v18_verification": {"path": str(repo_path("v18_verification")), "sha256": EXPECTED["v18_verification"]},
            "dataset": {"path": str(repo_path("dataset")), "sha256": EXPECTED["dataset"]},
            "sumo_runtime_lock": {"path": str(runtime_lock_path), "sha256": EXPECTED["runtime_lock"]},
            "evaluation_source_sha256": {row["path"]: row["sha256"] for row in scientific_sources},
            "matrix_source_sha256": {row["path"]: row["sha256"] for row in source_freeze["matrix_sources"]},
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
        _, source_freeze = validate_inputs()
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if manifest.get("schema_version") != SCHEMA or aggregate(entries) != manifest.get("aggregate_sha256"):
            raise ValueError("v19 manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v19 deployment membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v19 file drifted: {row['path']}")
        data_policy(entries)
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("adaptive_development_screen") is not True
            or manifest.get("formal_held_out_outcomes_included") is not False
            or manifest.get("formal_confirmation_eligibility_created") is not False
            or manifest.get("passing_only_permits_fresh_full_development_control_matrix") is not True
            or manifest.get("physical_behavior_count") != 15
            or manifest.get("expected_new_remote_sumo_episodes") != 135
            or manifest.get("critical_matrix_source_file_count") != 55
            or manifest.get("scientific_source_file_count") != 20
            or manifest.get("source_freeze", {}).get("sha256") != EXPECTED["source_freeze"]
            or manifest.get("registration", {}).get("sha256") != EXPECTED["registration"]
            or manifest.get("methods_manifest", {}).get("sha256") != EXPECTED["methods"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256") != EXPECTED["runtime_lock"]
            or manifest.get("parent_v18_v3_deployment", {}).get("sha256") != EXPECTED_PARENT_MANIFEST
            or manifest.get("evaluation_source_sha256")
            != {row["path"]: row["sha256"] for row in source_freeze["matrix_sources"] + source_freeze["operational_sources"]}
        ):
            raise ValueError("v19 deployment scope drifted")
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
            raise FileExistsError(f"refusing to overwrite v19 audit: {destination}")
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
        "--output",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v19_completion_safety_evaluation_v1/workspace",
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v19_completion_safety_evaluation_v1/workspace",
    )
    verify_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
