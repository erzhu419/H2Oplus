#!/usr/bin/env python3
"""Build and verify the frozen v20 full-control deployment."""

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

from bus_h2o.sumo_provenance import sha256  # noqa: E402
from reproducibility.build_h2oplus_v11_actor_hold_evaluation_deployment import (  # noqa: E402
    aggregate,
    descriptor,
)


SCHEMA = "transit-h2oplus-v20-full-control-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v20-full-control-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
SCENARIO_ROOT = "bus_h2o/development_scenarios_v20/full_control_v1"
PARENT_MANIFEST_SHA256 = (
    "3f58a40bde629420cfdbb20edef5c7a0bd2dbcb259bdce6baab4455282809864"
)
PARENT_AGGREGATE_SHA256 = (
    "ef668968b10b1bd77348ed20699f6004824e0f88d797e106c2110305891c086e"
)
EXPECTED = {
    "authorization": "ab535766ffad546c07e4291bfd2f0519d7cfa0e83f4cb3f4022c369d4cb19a06",
    "rationale": "e93a4114760e8755d27dcd7042ca666f4aefdf17e03d3bb55990068cf552998c",
    "registration": "f930a3612720097d4f5e5b635501405245f04ffd54445f389a23800d06f7f7d8",
    "protocol": "7211d32aeb1416840de442ca43fa997a918dfa282afcec5009091f78834ffa5d",
    "methods": "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9",
    "environment": "7210826548c94aaa8fbbbb73b62eec5450fec131f2ae3dd1d1dbc93922dff2ba",
    "design_audit": "0c583278b57eee5b5d479db78d294643f11b5d04a51840e35eeaf97e14a0015c",
    "design_verification": "66cc5133fa2a25bc323f880d148dba282db9708161af3df4c5f764541c3d70b9",
    "source_freeze": "b30de4416829a0bfa683184712a842dcff64732d27c3c51a5f8a3b107da22159",
    "v19_audit": "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    "v19_closeout": "3e95e4cf964983257fd3fabc5cab2987e1a249b5b9a25d3ae455f081932b3f74",
    "v19_verification": "ec44263eebc8ba61049973f34aadf5154d0b07f5ccc096d3659662002b95f9ab",
    "v19_analysis": "78a2fc88b81d3f171af5812c36bcf3e7b5cdab5d8660f4a8f1f151d00565565f",
    "v19_execution": "fff91eb84e229426ae62d1d781296b54d3d242541f6e934f239199d8a384a336",
    "v19_file_manifest": "db80e286f39b7d8b1a984e6e1a5ffd6b1b0747e2bd61ac463ab663aa7298b125",
    "v19_file_verification": "867a110f72010f51e35f55d237ecd1b58a35a8ed6bedc1d326960775d3abb70e",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v20_full_control.py",
    "reproducibility/run_evaluation_matrix_v20_full_control.py",
    "reproducibility/analyze_h2oplus_v20_full_control.py",
    "reproducibility/run_h2oplus_v20_full_control_evaluation_dispatch.py",
    "reproducibility/prepare_h2oplus_v20_full_control.py",
    "reproducibility/audit_h2oplus_v20_full_control_design.py",
    "reproducibility/verify_h2oplus_v20_full_control_design.py",
    "reproducibility/build_h2oplus_v20_full_control_source_freeze.py",
    "reproducibility/h2oplus_v20_full_control_authorization_v1.json",
    "reproducibility/h2oplus_v20_full_control_rationale_v1.json",
    "reproducibility/h2oplus_v20_full_control_registration_v1.json",
    "reproducibility/h2oplus_v20_full_control_protocol_v1.json",
    "reproducibility/h2oplus_v20_full_control_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v20_full_control_environment_v1.json",
    "reproducibility/h2oplus_v20_full_control_design_audit_v1.json",
    "reproducibility/h2oplus_v20_full_control_design_verification_v1.json",
    "reproducibility/h2oplus_v20_full_control_source_freeze_v1.json",
    "reproducibility/h2oplus_v19_completion_safety_results/analysis_execution_evidence_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_results/completion_safety_analysis_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_results/evaluation_audit_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_results/evaluation_closeout_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_results/evaluation_closeout_verification_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_results/evaluation_file_manifest_v2.json",
    "reproducibility/h2oplus_v19_completion_safety_results/evaluation_file_manifest_verification_v2.json",
)
PARENT_MANIFEST_COPY = (
    "reproducibility/h2oplus_v19_completion_safety_evaluation_deployment_manifest_v2.json"
)


def input_paths() -> dict[str, Path]:
    return {
        "authorization": ROOT / "reproducibility/h2oplus_v20_full_control_authorization_v1.json",
        "rationale": ROOT / "reproducibility/h2oplus_v20_full_control_rationale_v1.json",
        "registration": ROOT / "reproducibility/h2oplus_v20_full_control_registration_v1.json",
        "protocol": ROOT / "reproducibility/h2oplus_v20_full_control_protocol_v1.json",
        "methods": ROOT / "reproducibility/h2oplus_v20_full_control_evaluation_methods_v1.json",
        "environment": ROOT / "reproducibility/h2oplus_v20_full_control_environment_v1.json",
        "design_audit": ROOT / "reproducibility/h2oplus_v20_full_control_design_audit_v1.json",
        "design_verification": ROOT / "reproducibility/h2oplus_v20_full_control_design_verification_v1.json",
        "source_freeze": ROOT / "reproducibility/h2oplus_v20_full_control_source_freeze_v1.json",
        "v19_audit": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/evaluation_audit_v2.json",
        "v19_closeout": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/evaluation_closeout_v2.json",
        "v19_verification": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/evaluation_closeout_verification_v2.json",
        "v19_analysis": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/completion_safety_analysis_v2.json",
        "v19_execution": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/analysis_execution_evidence_v2.json",
        "v19_file_manifest": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/evaluation_file_manifest_v2.json",
        "v19_file_verification": ROOT / "reproducibility/h2oplus_v19_completion_safety_results/evaluation_file_manifest_verification_v2.json",
        "data_environment_audit": ROOT / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": ROOT / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }


def validate_inputs() -> tuple[dict[str, Path], dict]:
    paths = input_paths()
    for name, path in paths.items():
        observed = sha256(path.resolve(strict=True))
        if observed != EXPECTED[name]:
            raise ValueError(f"v20 deployment input drifted: {name}: {observed}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    freeze = payloads["source_freeze"]
    if (
        freeze.get("schema_version")
        != "transit-h2oplus-v20-full-control-source-freeze-v1"
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("v20_development_evaluation_outcomes_read") is not False
        or freeze.get("expected_matrix_source_file_count") != 57
        or freeze.get("matrix_source_count") != 18
        or freeze.get("operational_source_count") != 8
        or payloads["design_audit"].get("passed") is not True
        or payloads["design_verification"].get("passed") is not True
        or payloads["v19_audit"].get("artifact_integrity", {}).get("passed")
        is not True
        or payloads["v19_closeout"].get("eligible_for_full_development_control_matrix")
        is not True
        or payloads["v19_verification"].get("passed") is not True
        or payloads["data_environment_audit"].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or payloads["data_environment_verification"].get("passed") is not True
    ):
        raise ValueError("v20 deployment scientific scope drifted")
    for row in freeze["matrix_sources"] + freeze["operational_sources"]:
        if sha256((ROOT / row["path"]).resolve(strict=True)) != row["sha256"]:
            raise ValueError(f"v20 frozen source drifted: {row['path']}")
    return paths, freeze


def clone_parent(parent: Path, output: Path) -> None:
    shutil.copytree(parent, output, copy_function=os.link, symlinks=True)
    manifest = output / MANIFEST_NAME
    if sha256(manifest) != PARENT_MANIFEST_SHA256:
        raise ValueError("v19 parent manifest drifted during hardlink clone")
    manifest.unlink()


def copy_overlay(parent_manifest: Path, output: Path) -> None:
    repo = output / "H2Oplus"
    for relative in NEW_REPO_FILES:
        source = (ROOT / relative).resolve(strict=True)
        destination = repo / relative
        if destination.exists():
            raise FileExistsError(f"v20 overlay would replace parent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_scenarios = (ROOT / SCENARIO_ROOT).resolve(strict=True)
    destination_scenarios = repo / SCENARIO_ROOT
    if destination_scenarios.exists():
        raise FileExistsError(f"v20 scenario overlay exists: {destination_scenarios}")
    shutil.copytree(source_scenarios, destination_scenarios)
    destination = repo / PARENT_MANIFEST_COPY
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
    scenarios = [row for row in entries if SCENARIO_ROOT in row["path"]]
    if (
        len(hdf5) != 1
        or hdf5[0]["sha256"] != EXPECTED["dataset"]
        or len(actors) != 85
        or len(scenarios) != 63
        or forbidden
    ):
        raise ValueError(
            "v20 deployment data policy drifted: "
            f"hdf5={len(hdf5)}, actors={len(actors)}, "
            f"scenarios={len(scenarios)}, forbidden={forbidden}"
        )
    return {
        "hdf5_files": 1,
        "compact_actor_files": 85,
        "new_actor_files": 0,
        "fresh_v20_scenario_files": 63,
        "training_checkpoint_or_optimizer_files": 0,
        "physical_evaluation_result_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": EXPECTED["dataset"],
    }


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    if sha256(parent_manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("sealed v19 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != PARENT_AGGREGATE_SHA256
        or parent_manifest.get("entry_count") != 927
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v19 parent deployment scope drifted")
    paths, freeze = validate_inputs()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v20 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_overlay(parent_manifest_path, output)
        repo = output / "H2Oplus"

        def repo_path(name: str) -> Path:
            return repo / paths[name].relative_to(ROOT)

        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        runtime_lock_path = repo_path("runtime_lock")
        runtime = json.loads(runtime_lock_path.read_text())
        sources = freeze["matrix_sources"] + freeze["operational_sources"]
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "total_file_bytes": sum(row["size_bytes"] for row in entries),
            "aggregate_sha256": aggregate(entries),
            "training": False,
            "study_stage": "development",
            "matrix_type": "full-development-control-matrix",
            "adaptive_development_matrix": True,
            "formal_held_out_outcomes_included": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "formal_confirmation_eligibility_created_before_v20": False,
            "passing_matrix_only_creates_separate_formal_confirmation_eligibility": True,
            "physical_behavior_count": 45,
            "logical_method_count": 45,
            "new_actor_file_count": 0,
            "expected_new_remote_sumo_episodes": 405,
            "critical_matrix_source_file_count": 57,
            "scientific_source_file_count": len(sources),
            "sumo_runtime_signature": runtime["runtime_fingerprint"]["portable_signature"],
            "parent_v19_artifact_integrity_passed": True,
            "parent_training_stability_passed": True,
            "prior_v14_through_v19_development_outcomes_used_for_design": True,
            "data_environment_material_factors_disclosed": True,
            "data_policy": data_policy(entries),
            "parent_v19_deployment": {
                "path": str(parent_manifest_path),
                "sha256": PARENT_MANIFEST_SHA256,
                "aggregate_sha256": PARENT_AGGREGATE_SHA256,
                "scientific_result_files_reused": False,
            },
            "parent_v19_deployment_manifest_copy": {
                "path": str(repo / PARENT_MANIFEST_COPY),
                "sha256": PARENT_MANIFEST_SHA256,
            },
            "authorization": {"path": str(repo_path("authorization")), "sha256": EXPECTED["authorization"]},
            "rationale": {"path": str(repo_path("rationale")), "sha256": EXPECTED["rationale"]},
            "registration": {"path": str(repo_path("registration")), "sha256": EXPECTED["registration"]},
            "protocol": {"path": str(repo_path("protocol")), "sha256": EXPECTED["protocol"]},
            "methods_manifest": {"path": str(repo_path("methods")), "sha256": EXPECTED["methods"]},
            "environment_design": {"path": str(repo_path("environment")), "sha256": EXPECTED["environment"]},
            "source_freeze": {"path": str(repo_path("source_freeze")), "sha256": EXPECTED["source_freeze"]},
            "design_audit": {"path": str(repo_path("design_audit")), "sha256": EXPECTED["design_audit"]},
            "design_verification": {"path": str(repo_path("design_verification")), "sha256": EXPECTED["design_verification"]},
            "parent_v19_audit": {"path": str(repo_path("v19_audit")), "sha256": EXPECTED["v19_audit"]},
            "parent_v19_closeout": {"path": str(repo_path("v19_closeout")), "sha256": EXPECTED["v19_closeout"]},
            "parent_v19_verification": {"path": str(repo_path("v19_verification")), "sha256": EXPECTED["v19_verification"]},
            "dataset": {"path": str(repo_path("dataset")), "sha256": EXPECTED["dataset"]},
            "sumo_runtime_lock": {"path": str(runtime_lock_path), "sha256": EXPECTED["runtime_lock"]},
            "evaluation_source_sha256": {row["path"]: row["sha256"] for row in sources},
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
        if manifest.get("schema_version") != SCHEMA:
            raise ValueError("v20 deployment schema drifted")
        if aggregate(entries) != manifest.get("aggregate_sha256"):
            raise ValueError("v20 deployment aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        expected = {row["path"] for row in entries}
        if actual != expected:
            raise ValueError("v20 deployment membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v20 deployment file drifted: {row['path']}")
        repo = root / "H2Oplus"
        for row in freeze["matrix_sources"] + freeze["operational_sources"]:
            if sha256(repo / row["path"]) != row["sha256"]:
                raise ValueError(f"v20 deployed source drifted: {row['path']}")
        if (
            manifest.get("entry_count") != len(entries)
            or manifest.get("total_file_bytes")
            != sum(row["size_bytes"] for row in entries)
            or manifest.get("physical_behavior_count") != 45
            or manifest.get("expected_new_remote_sumo_episodes") != 405
            or manifest.get("critical_matrix_source_file_count") != 57
            or manifest.get("formal_held_out_outcomes_included") is not False
            or manifest.get("formal_confirmation_stage_authorized_or_submitted")
            is not False
            or manifest.get("parent_v19_artifact_integrity_passed") is not True
            or manifest.get("parent_training_stability_passed") is not True
            or manifest.get("source_freeze", {}).get("sha256")
            != EXPECTED["source_freeze"]
            or manifest.get("parent_v19_deployment", {}).get("sha256")
            != PARENT_MANIFEST_SHA256
            or manifest.get("data_policy", {}).get("physical_evaluation_result_files")
            != 0
        ):
            raise ValueError("v20 deployment scope drifted")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report = {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_root": str(root),
        "deployment_manifest": str(manifest_path),
        "deployment_manifest_sha256": sha256(manifest_path),
        "entry_count": manifest.get("entry_count"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "expected_matrix_source_file_count": 57,
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
        "errors": errors,
        "passed": not errors,
    }
    if args.output:
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite v20 deployment audit: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(partial, destination)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument(
        "--parent-workspace",
        type=Path,
        default=ROOT / (
            "reproducibility/deployments/"
            "h2oplus_v19_completion_safety_evaluation_v2/workspace"
        ),
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / (
            "reproducibility/deployments/"
            "h2oplus_v20_full_control_evaluation_v1/workspace"
        ),
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / (
            "reproducibility/deployments/"
            "h2oplus_v20_full_control_evaluation_v1/workspace"
        ),
    )
    verify_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
