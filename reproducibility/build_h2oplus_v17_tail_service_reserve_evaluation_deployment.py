#!/usr/bin/env python3
"""Build or verify the isolated v17 development-evaluation deployment."""

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


SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = "4637a2dd2b935c00070c4ca514abf2ab24213c9ef80943dcdb2ae781d6d7653c"
EXPECTED_PARENT_AGGREGATE = "1e361c6dce106e9d841d7fabc6cbf37414591c4afba2ee075bafbe28159f2805"
EXPECTED = {
    "authorization": "6871d458b23bbc796aa8f0a981eb21f865f6a5a0d61c3c427173573e44261e4a",
    "offline_design": "5bc0e0366fa19ef9597db20de702d554bb1dc05ddc6ec7ddb1fdedc72818b24e",
    "rationale": "0b41d827e79bdbea83b689caf0afd144d13f83780e74ec44e716e65ca7099e5a",
    "registration": "d5465e677e29971eb1c6e88a926387bacd172de737f669f6fd84426bb7c68364",
    "protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "source_freeze": "30e4b23219dc01c96b7eeaf89e4a8564716a887b9e525e8bcf65164ca8026f95",
    "environment_design": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v16_analysis": "4a88f48946c2ec478ca47274df39e0ccdf3535c1c21ad41141142d529617f75e",
    "v16_audit": "ed5e58e52f1ea50d7195a10b313bd9c36e88b618fa802de2693ffdfe133112c9",
    "v16_closeout": "ef7d456e6c71bb2fbcbd176642c01acf123f8e709cefffd2b8d4ae26a850ddf5",
    "v16_verification": "54b25d3c9307e9f926319b7f9ae543e6747d4639ce9b8af762e123bade9f4c85",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v17_tail_service_reserve.py",
    "reproducibility/analyze_h2oplus_v17_tail_service_reserve.py",
    "reproducibility/run_evaluation_matrix_v17_tail_service_reserve.py",
    "reproducibility/run_h2oplus_v17_tail_service_reserve_evaluation_dispatch.py",
    "reproducibility/h2oplus_v17_tail_service_reserve_authorization_v1.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_offline_design_evidence_v1.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_rationale_v1.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_registration_v1.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_audit_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_closeout_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_closeout_verification_v1.json",
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
            raise FileExistsError(f"v17 overlay would replace parent deployment content: {destination}")
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
            f"v17 deployment data policy drifted: hdf5={len(hdf5)}, pt={len(pt_files)}, forbidden={forbidden}"
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
        raise ValueError("sealed v16 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 712
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v16 parent deployment scope drifted")
    for name, path in {
        "authorization": ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_authorization_v1.json",
        "offline_design": ROOT
        / "reproducibility/h2oplus_v17_tail_service_reserve_offline_design_evidence_v1.json",
        "rationale": ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_rationale_v1.json",
        "registration": ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_registration_v1.json",
        "protocol": ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json",
        "methods": ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
        "source_freeze": ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json",
        "environment_design": ROOT / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "v16_analysis": ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json",
        "v16_audit": ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_audit_v1.json",
        "v16_closeout": ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_closeout_v1.json",
        "v16_verification": ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_closeout_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v17 deployment input drifted: {name}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v17 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_new_files(output)
        repo = output / "H2Oplus"
        source_freeze_path = repo / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json"
        source_freeze = json.loads(source_freeze_path.read_text())
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        dataset_path = repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
        runtime_lock_path = repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
        methods_path = repo / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json"
        protocol_path = repo / "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json"
        registration_path = repo / "reproducibility/h2oplus_v17_tail_service_reserve_registration_v1.json"
        authorization_path = repo / "reproducibility/h2oplus_v17_tail_service_reserve_authorization_v1.json"
        offline_design_path = (
            repo
            / "reproducibility/h2oplus_v17_tail_service_reserve_offline_design_evidence_v1.json"
        )
        v16_results = repo / "reproducibility/h2oplus_v16_headway_reserve_results"
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
            "critical_matrix_source_file_count": 53,
            "scientific_source_file_count": len(scientific_sources),
            "sumo_runtime_signature": runtime_signature,
            "parent_v16_artifact_integrity_passed": True,
            "parent_v14_training_stability_passed": True,
            "prior_v14_v15_v16_development_outcomes_used_for_design": True,
            "data_policy": data_policy(entries, EXPECTED["dataset"]),
            "parent_v16_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
            },
            "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
            "offline_design_evidence": {
                "path": str(offline_design_path),
                "sha256": sha256(offline_design_path),
            },
            "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "methods_manifest": {"path": str(methods_path), "sha256": sha256(methods_path)},
            "source_freeze": {"path": str(source_freeze_path), "sha256": sha256(source_freeze_path)},
            "parent_v16_evidence": {
                "analysis": {
                    "path": str(v16_results / "headway_reserve_analysis_v1.json"),
                    "sha256": EXPECTED["v16_analysis"],
                },
                "audit": {
                    "path": str(v16_results / "evaluation_audit_v1.json"),
                    "sha256": EXPECTED["v16_audit"],
                },
                "closeout": {
                    "path": str(v16_results / "evaluation_closeout_v1.json"),
                    "sha256": EXPECTED["v16_closeout"],
                },
                "verification": {
                    "path": str(v16_results / "evaluation_closeout_verification_v1.json"),
                    "sha256": EXPECTED["v16_verification"],
                },
            },
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
            raise ValueError("v17 deployment manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v17 deployment file membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v17 deployment file drifted: {row['path']}")
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
            or manifest.get(
                "prior_v14_v15_v16_development_outcomes_used_for_design"
            )
            is not True
            or manifest.get("methods_manifest", {}).get("sha256") != EXPECTED["methods"]
            or manifest.get("offline_design_evidence", {}).get("sha256")
            != EXPECTED["offline_design"]
            or manifest.get("source_freeze", {}).get("sha256") != EXPECTED["source_freeze"]
            or manifest.get("dataset", {}).get("sha256") != EXPECTED["dataset"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256") != EXPECTED["runtime_lock"]
            or manifest.get("parent_v16_evidence", {}).get("analysis", {}).get("sha256")
            != EXPECTED["v16_analysis"]
            or manifest.get("parent_v16_evidence", {}).get("audit", {}).get("sha256")
            != EXPECTED["v16_audit"]
            or manifest.get("parent_v16_evidence", {}).get("closeout", {}).get("sha256")
            != EXPECTED["v16_closeout"]
            or manifest.get("parent_v16_evidence", {}).get("verification", {}).get("sha256")
            != EXPECTED["v16_verification"]
        ):
            raise ValueError("v17 deployment scope drifted")
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
            raise FileExistsError(f"refusing to overwrite v17 deployment audit: {destination}")
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
        default=ROOT / "reproducibility/deployments/h2oplus_v16_headway_reserve_evaluation_v1/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v17_tail_service_reserve_evaluation_v1/workspace",
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
