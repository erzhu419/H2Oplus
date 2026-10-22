#!/usr/bin/env python3
"""Build or independently verify the isolated v17-v2 evaluation deployment."""

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


SCHEMA = (
    "transit-h2oplus-v17-tail-service-reserve-evaluation-deployment-manifest-v2"
)
AUDIT_SCHEMA = (
    "transit-h2oplus-v17-tail-service-reserve-evaluation-deployment-audit-v2"
)
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = (
    "a4bb3706a8f401d347c0e403f7b37ce47f142401f3c4aac3a4507d6e1a64ef0d"
)
EXPECTED_PARENT_AGGREGATE = (
    "9264b79fdcebf90dbfd7b2822a45e568de90c3466b3181161d81b5ea743d3fd4"
)
EXPECTED = {
    "authorization": "4c61471e71c3190c3b450b2805eba0a48712da8c25a9840806b3bd849029ebd4",
    "correction": "1a521809e08618001b19e5d2d7d1077472d819aa648fcaa46e87423e833268ed",
    "registration": "41d8399dcf07aedea421fed516407fbeff7fa4dda296652b8c70aa24582c90cf",
    "source_freeze": "5d0e4496183cbd90a2a82191b10255b4e27c70d2c56b41cb2fc651d344a1724e",
    "incident_closeout": "0d189243e83ee014c414fd154f26ebc52107c934dee51131a5aecaf9cd3e19ed",
    "incident_verification": "922b514bbab7228da049eed036cb01e1f85b93c1e3de24bc79ad2a97b23150e0",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "environment_design": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "reproducibility/run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2.py",
    "reproducibility/h2oplus_v17_tail_service_reserve_successor_authorization_v2.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_python38_correction_v2.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_successor_registration_v2.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v2.json",
    "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
    "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
    (
        "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "evaluation_incidents/v1_python38_removeprefix/incident_closeout_v1.json"
    ),
    (
        "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "evaluation_incidents/v1_python38_removeprefix/incident_verification_v1.json"
    ),
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
            raise FileExistsError(
                f"v17-v2 overlay would replace parent content: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def data_policy(entries: list[dict], dataset_hash: str) -> dict:
    hdf5 = [
        row
        for row in entries
        if Path(row["path"]).suffix.lower() in {".h5", ".hdf5"}
    ]
    pt_files = [
        row for row in entries if Path(row["path"]).suffix.lower() == ".pt"
    ]
    forbidden = [
        row["path"]
        for row in entries
        if Path(row["path"]).suffix.lower() in {".pth", ".ckpt", ".pkl"}
        or "remote_results" in Path(row["path"]).parts
        or Path(row["path"]).name.startswith("sumo_seed")
    ]
    if (
        len(hdf5) != 1
        or hdf5[0]["sha256"] != dataset_hash
        or len(pt_files) != 85
        or forbidden
    ):
        raise ValueError(
            "v17-v2 deployment data policy drifted: "
            f"hdf5={len(hdf5)}, pt={len(pt_files)}, forbidden={forbidden}"
        )
    return {
        "hdf5_files": 1,
        "compact_actor_files": 85,
        "new_actor_files": 0,
        "training_checkpoint_or_optimizer_files": 0,
        "v17_v1_incident_result_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": dataset_hash,
    }


def input_paths() -> dict[str, Path]:
    incident = (
        ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "evaluation_incidents/v1_python38_removeprefix"
    )
    return {
        "authorization": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_successor_authorization_v2.json",
        "correction": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_python38_correction_v2.json",
        "registration": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_successor_registration_v2.json",
        "source_freeze": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v2.json",
        "incident_closeout": incident / "incident_closeout_v1.json",
        "incident_verification": incident / "incident_verification_v1.json",
        "data_environment_audit": ROOT
        / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": ROOT
        / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
        "methods": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
        "protocol": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json",
        "environment_design": ROOT
        / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "dataset": ROOT
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT
        / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }


def validate_inputs() -> dict[str, Path]:
    paths = input_paths()
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v17-v2 deployment input drifted: {name}")
    return paths


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    if sha256(parent_manifest_path) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("sealed v17-v1 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 727
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v17-v1 parent deployment scope drifted")
    paths = validate_inputs()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v17-v2 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_new_files(output)
        repo = output / "H2Oplus"
        source_freeze_path = repo / paths["source_freeze"].relative_to(ROOT)
        source_freeze = json.loads(source_freeze_path.read_text())
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        repo_path = lambda name: repo / paths[name].relative_to(ROOT)
        runtime_lock_path = repo_path("runtime_lock")
        runtime_payload = json.loads(runtime_lock_path.read_text())
        runtime_signature = runtime_payload["runtime_fingerprint"]["portable_signature"]
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
            "formal_held_out_outcomes_included": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "physical_behavior_count": 15,
            "logical_method_count": 15,
            "new_actor_file_count": 0,
            "expected_new_remote_sumo_episodes": 135,
            "critical_matrix_source_file_count": 53,
            "scientific_source_file_count": len(scientific_sources),
            "sumo_runtime_signature": runtime_signature,
            "parent_v16_artifact_integrity_passed": True,
            "parent_v14_training_stability_passed": True,
            "prior_v14_v15_v16_development_outcomes_used_for_design": True,
            "v17_v1_incident_results_included_or_reused": False,
            "data_environment_audit_used_to_change_v17_v2_design": False,
            "data_policy": data_policy(entries, EXPECTED["dataset"]),
            "parent_v17_v1_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
            },
            "authorization": {
                "path": str(repo_path("authorization")),
                "sha256": EXPECTED["authorization"],
            },
            "correction": {
                "path": str(repo_path("correction")),
                "sha256": EXPECTED["correction"],
            },
            "registration": {
                "path": str(repo_path("registration")),
                "sha256": EXPECTED["registration"],
            },
            "protocol": {
                "path": str(repo_path("protocol")),
                "sha256": EXPECTED["protocol"],
            },
            "methods_manifest": {
                "path": str(repo_path("methods")),
                "sha256": EXPECTED["methods"],
            },
            "source_freeze": {
                "path": str(source_freeze_path),
                "sha256": EXPECTED["source_freeze"],
            },
            "incident_evidence": {
                "closeout": {
                    "path": str(repo_path("incident_closeout")),
                    "sha256": EXPECTED["incident_closeout"],
                },
                "verification": {
                    "path": str(repo_path("incident_verification")),
                    "sha256": EXPECTED["incident_verification"],
                },
            },
            "data_environment_evidence": {
                "audit": {
                    "path": str(repo_path("data_environment_audit")),
                    "sha256": EXPECTED["data_environment_audit"],
                },
                "verification": {
                    "path": str(repo_path("data_environment_verification")),
                    "sha256": EXPECTED["data_environment_verification"],
                },
            },
            "environment_design": {
                "path": str(repo_path("environment_design")),
                "sha256": EXPECTED["environment_design"],
            },
            "dataset": {
                "path": str(repo_path("dataset")),
                "sha256": EXPECTED["dataset"],
            },
            "sumo_runtime_lock": {
                "path": str(runtime_lock_path),
                "sha256": EXPECTED["runtime_lock"],
            },
            "evaluation_source_sha256": {
                row["path"]: row["sha256"] for row in scientific_sources
            },
            "matrix_source_sha256": {
                row["path"]: row["sha256"]
                for row in source_freeze["matrix_sources"]
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
        validate_inputs()
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if (
            manifest.get("schema_version") != SCHEMA
            or aggregate(entries) != manifest.get("aggregate_sha256")
        ):
            raise ValueError("v17-v2 deployment manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v17-v2 deployment file membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v17-v2 deployment file drifted: {row['path']}")
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
            or manifest.get("critical_matrix_source_file_count") != 53
            or manifest.get("scientific_source_file_count") != 16
            or manifest.get("parent_v16_artifact_integrity_passed") is not True
            or manifest.get("parent_v14_training_stability_passed") is not True
            or manifest.get("prior_v14_v15_v16_development_outcomes_used_for_design") is not True
            or manifest.get("v17_v1_incident_results_included_or_reused") is not False
            or manifest.get("data_environment_audit_used_to_change_v17_v2_design") is not False
            or manifest.get("authorization", {}).get("sha256") != EXPECTED["authorization"]
            or manifest.get("correction", {}).get("sha256") != EXPECTED["correction"]
            or manifest.get("registration", {}).get("sha256") != EXPECTED["registration"]
            or manifest.get("methods_manifest", {}).get("sha256") != EXPECTED["methods"]
            or manifest.get("source_freeze", {}).get("sha256") != EXPECTED["source_freeze"]
            or manifest.get("dataset", {}).get("sha256") != EXPECTED["dataset"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256") != EXPECTED["runtime_lock"]
            or manifest.get("incident_evidence", {}).get("closeout", {}).get("sha256")
            != EXPECTED["incident_closeout"]
            or manifest.get("incident_evidence", {}).get("verification", {}).get("sha256")
            != EXPECTED["incident_verification"]
            or manifest.get("data_environment_evidence", {}).get("audit", {}).get("sha256")
            != EXPECTED["data_environment_audit"]
            or manifest.get("data_environment_evidence", {}).get("verification", {}).get("sha256")
            != EXPECTED["data_environment_verification"]
        ):
            raise ValueError("v17-v2 deployment scope drifted")
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
            raise FileExistsError(
                f"refusing to overwrite v17-v2 deployment audit: {destination}"
            )
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
        default=ROOT
        / "reproducibility/deployments/h2oplus_v17_tail_service_reserve_evaluation_v1/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility/deployments/h2oplus_v17_tail_service_reserve_evaluation_v2/workspace",
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
