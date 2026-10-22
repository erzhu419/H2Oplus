#!/usr/bin/env python3
"""Build or verify the isolated v15 development-evaluation deployment."""

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


SCHEMA = "transit-h2oplus-v15-hold-calibration-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v15-hold-calibration-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = "fe227722da303dd38be684222845bbbd84cb80f14e0cf2fb175b333c2495ee04"
EXPECTED_PARENT_AGGREGATE = "45d16d2cd610bd7861b36e7f41b56ce63b4dafc7589e437f890ed3d66a9d9fbc"
EXPECTED = {
    "authorization": "5b3eaefa9a163ebfef2596691a54e61fbe2685cda943f76fa1e7790c7e7b971c",
    "rationale": "cd9c6fb56b2a34771675624799d9a28d2ad988349568f8b96a31257a94278de4",
    "registration": "b37d595056923f9bd746226b6c4863c2d7374750e8d45291de564bc9247428b9",
    "protocol": "6cd856504438ad0e9614b49156100cef12f24201b41562bf223e898f15ef0a8c",
    "methods": "7712a35cbb1adbc55acf50bd8843bb3f4f7dc17090d0e65deeeb10ffe5177e5a",
    "source_freeze": "af025dbd817a4d7c4661227fefae7653ecc5ca9bf666e94a1c0a8ca7e05f642d",
    "environment_design": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v14_analysis": "937a5300988311f80a84dc1220dfee3f0f96f2b6f83691d4e309c38d6bbe6856",
    "v14_closeout": "95fef5b3a5206ce14a3da164d9f2948b0dc3b5d61ee1d1b34c52c431d1cd1d32",
    "v14_verification": "0e20ccc1e7855e4feb5860e5cb334cd60237f28032a948b7325037ad0aa61dc0",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v15_hold_calibration.py",
    "reproducibility/analyze_h2oplus_v15_hold_calibration.py",
    "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
    "reproducibility/run_h2oplus_v15_hold_calibration_evaluation_dispatch.py",
    "reproducibility/h2oplus_v15_hold_calibration_authorization_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_rationale_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_registration_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_protocol_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json",
    "reproducibility/h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
    "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_v1.json",
    "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_verification_v1.json",
)


def clone_parent(parent: Path, output: Path) -> None:
    shutil.copytree(parent, output, copy_function=os.link, symlinks=True)
    manifest = output / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest.unlink()


def copy_new_files(output: Path) -> None:
    repo = output / "H2Oplus"
    for relative in NEW_REPO_FILES:
        source = (ROOT / relative).resolve(strict=True)
        destination = repo / relative
        if destination.exists():
            raise FileExistsError(f"v15 overlay would replace parent deployment content: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def data_policy(entries: list[dict], dataset_hash: str) -> dict:
    hdf5 = [row for row in entries if Path(row["path"]).suffix.lower() in {".h5", ".hdf5"}]
    pt_files = [row for row in entries if Path(row["path"]).suffix.lower() == ".pt"]
    forbidden = [
        row["path"]
        for row in entries
        if Path(row["path"]).suffix.lower() in {".pth", ".ckpt", ".pkl"}
    ]
    if len(hdf5) != 1 or hdf5[0]["sha256"] != dataset_hash or len(pt_files) != 85 or forbidden:
        raise ValueError(
            f"v15 deployment data policy drifted: hdf5={len(hdf5)}, pt={len(pt_files)}, forbidden={forbidden}"
        )
    return {
        "hdf5_files": 1,
        "compact_actor_files": 85,
        "new_actor_files": 0,
        "training_checkpoint_or_optimizer_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": dataset_hash,
    }


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    if sha256(parent_manifest_path) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("sealed v14 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 685
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v14 parent deployment scope drifted")
    for name, path in {
        "authorization": ROOT / "reproducibility/h2oplus_v15_hold_calibration_authorization_v1.json",
        "rationale": ROOT / "reproducibility/h2oplus_v15_hold_calibration_rationale_v1.json",
        "registration": ROOT / "reproducibility/h2oplus_v15_hold_calibration_registration_v1.json",
        "protocol": ROOT / "reproducibility/h2oplus_v15_hold_calibration_protocol_v1.json",
        "methods": ROOT / "reproducibility/h2oplus_v15_hold_calibration_evaluation_methods_v1.json",
        "source_freeze": ROOT / "reproducibility/h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json",
        "environment_design": ROOT / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "v14_analysis": ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
        "v14_closeout": ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_v1.json",
        "v14_verification": ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v15 deployment input drifted: {name}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v15 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_new_files(output)
        repo = output / "H2Oplus"
        source_freeze_path = repo / "reproducibility/h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json"
        source_freeze = json.loads(source_freeze_path.read_text())
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        dataset_path = repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
        runtime_lock_path = repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
        methods_path = repo / "reproducibility/h2oplus_v15_hold_calibration_evaluation_methods_v1.json"
        protocol_path = repo / "reproducibility/h2oplus_v15_hold_calibration_protocol_v1.json"
        registration_path = repo / "reproducibility/h2oplus_v15_hold_calibration_registration_v1.json"
        authorization_path = repo / "reproducibility/h2oplus_v15_hold_calibration_authorization_v1.json"
        runtime_payload = json.loads(runtime_lock_path.read_text())
        runtime_signature = runtime_payload["runtime_fingerprint"]["portable_signature"]
        scientific_sources = source_freeze["matrix_sources"] + source_freeze["operational_sources"]
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "total_file_bytes": sum(row["size_bytes"] for row in entries),
            "aggregate_sha256": aggregate(entries),
            "training": False,
            "study_stage": "development",
            "formal_held_out_outcomes_included": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "physical_behavior_count": 15,
            "logical_method_count": 15,
            "new_actor_file_count": 0,
            "expected_new_remote_sumo_episodes": 135,
            "critical_matrix_source_file_count": 49,
            "scientific_source_file_count": len(scientific_sources),
            "sumo_runtime_signature": runtime_signature,
            "parent_v14_artifact_integrity_passed": True,
            "parent_v14_training_stability_passed": True,
            "prior_v14_development_outcomes_used_for_design": True,
            "data_policy": data_policy(entries, EXPECTED["dataset"]),
            "parent_v14_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
            },
            "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
            "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "methods_manifest": {"path": str(methods_path), "sha256": sha256(methods_path)},
            "source_freeze": {"path": str(source_freeze_path), "sha256": sha256(source_freeze_path)},
            "environment_design": {
                "path": str(repo / "reproducibility/h2oplus_v12_staged_release_environment_v2.json"),
                "sha256": EXPECTED["environment_design"],
            },
            "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path)},
            "sumo_runtime_lock": {"path": str(runtime_lock_path), "sha256": sha256(runtime_lock_path)},
            "evaluation_source_sha256": {
                row["path"]: row["sha256"] for row in scientific_sources
            },
            "matrix_source_sha256": {
                row["path"]: row["sha256"] for row in source_freeze["matrix_sources"]
            },
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
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if manifest.get("schema_version") != SCHEMA or aggregate(entries) != manifest.get("aggregate_sha256"):
            raise ValueError("v15 deployment manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v15 deployment file membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v15 deployment file drifted: {row['path']}")
        data_policy(entries, EXPECTED["dataset"])
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("formal_held_out_outcomes_included") is not False
            or manifest.get("formal_confirmation_stage_authorized_or_submitted") is not False
            or manifest.get("physical_behavior_count") != 15
            or manifest.get("logical_method_count") != 15
            or manifest.get("new_actor_file_count") != 0
            or manifest.get("expected_new_remote_sumo_episodes") != 135
            or manifest.get("critical_matrix_source_file_count") != 49
            or manifest.get("scientific_source_file_count") != 12
            or manifest.get("parent_v14_artifact_integrity_passed") is not True
            or manifest.get("parent_v14_training_stability_passed") is not True
            or manifest.get("prior_v14_development_outcomes_used_for_design") is not True
            or manifest.get("methods_manifest", {}).get("sha256") != EXPECTED["methods"]
            or manifest.get("source_freeze", {}).get("sha256") != EXPECTED["source_freeze"]
            or manifest.get("dataset", {}).get("sha256") != EXPECTED["dataset"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256") != EXPECTED["runtime_lock"]
        ):
            raise ValueError("v15 deployment scope drifted")
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
            raise FileExistsError(f"refusing to overwrite v15 deployment audit: {destination}")
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
        default=ROOT / "reproducibility/deployments/h2oplus_v14_terminal_taper_evaluation_v1/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v15_hold_calibration_evaluation_v1/workspace",
    )
    build_parser.set_defaults(function=build)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path)
    verify_parser.set_defaults(function=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
