#!/usr/bin/env python3
"""Build or verify the isolated v16 development-evaluation deployment."""

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


SCHEMA = "transit-h2oplus-v16-headway-reserve-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v16-headway-reserve-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = "fe510525503e3e44cc83b33916be308d6d1fea2c39c299c7fc4e399a33891d6c"
EXPECTED_PARENT_AGGREGATE = "02c3fac83f0daf6a1e200f2aac38a81def3fdc0a0338a74ddaab4d2ae55fd2ec"
EXPECTED = {
    "authorization": "77f2234b90ba510fea089800494c637c12d8a3f24b2f2fdd236238ae04fb19a1",
    "rationale": "48e0445e1d82b4f08fc2068e57d5ec1931c26c5d31db9761f51d5082eadadf25",
    "registration": "6858bea266a4f46ec919804e643a5b8c6de4c189dd0517ee5bf123f08b64950a",
    "protocol": "978faf1b8ddf51128b5fac570552e8388c3d882e39244c4b6f0ec7615e79a43b",
    "methods": "35047fd2ce616acbccd4592ae333a7ba84a1e4231c1a2505c7768d96b91de06a",
    "source_freeze": "606a15bf7db8b9aa1c4dbefa7b9bcc12bc94ca60e605de5a3e185eefec679577",
    "environment_design": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v15_analysis": "d122aa79f6ee2cd5010e241607347921ed9bdc80bbed6ba41c3105e601ac9d9c",
    "v15_audit": "a060b49e3155092542681c7d6afef66bcb2b31ab04ffe523273754ae2dcfa7e8",
    "v15_closeout": "2e4598c433eb6ffaf18fb9919b608b00d5cdd17f5a53e466ca587d6c8213feab",
    "v15_verification": "f5edd5c2ef651172068c9cd595e6cb7710a3ffdd66e545671f5b4a014885e0b6",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v16_headway_reserve.py",
    "reproducibility/analyze_h2oplus_v16_headway_reserve.py",
    "reproducibility/run_evaluation_matrix_v16_headway_reserve.py",
    "reproducibility/run_h2oplus_v16_headway_reserve_evaluation_dispatch.py",
    "reproducibility/h2oplus_v16_headway_reserve_authorization_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_rationale_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_registration_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_protocol_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_results/hold_calibration_analysis_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_audit_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_closeout_v1.json",
    "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_closeout_verification_v1.json",
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
            raise FileExistsError(f"v16 overlay would replace parent deployment content: {destination}")
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
            f"v16 deployment data policy drifted: hdf5={len(hdf5)}, pt={len(pt_files)}, forbidden={forbidden}"
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
        raise ValueError("sealed v15 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 698
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v15 parent deployment scope drifted")
    for name, path in {
        "authorization": ROOT / "reproducibility/h2oplus_v16_headway_reserve_authorization_v1.json",
        "rationale": ROOT / "reproducibility/h2oplus_v16_headway_reserve_rationale_v1.json",
        "registration": ROOT / "reproducibility/h2oplus_v16_headway_reserve_registration_v1.json",
        "protocol": ROOT / "reproducibility/h2oplus_v16_headway_reserve_protocol_v1.json",
        "methods": ROOT / "reproducibility/h2oplus_v16_headway_reserve_evaluation_methods_v1.json",
        "source_freeze": ROOT / "reproducibility/h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
        "environment_design": ROOT / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
        "v15_analysis": ROOT / "reproducibility/h2oplus_v15_hold_calibration_results/hold_calibration_analysis_v1.json",
        "v15_audit": ROOT / "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_audit_v1.json",
        "v15_closeout": ROOT / "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_closeout_v1.json",
        "v15_verification": ROOT / "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_closeout_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v16 deployment input drifted: {name}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v16 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_new_files(output)
        repo = output / "H2Oplus"
        source_freeze_path = repo / "reproducibility/h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json"
        source_freeze = json.loads(source_freeze_path.read_text())
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        dataset_path = repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
        runtime_lock_path = repo / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
        methods_path = repo / "reproducibility/h2oplus_v16_headway_reserve_evaluation_methods_v1.json"
        protocol_path = repo / "reproducibility/h2oplus_v16_headway_reserve_protocol_v1.json"
        registration_path = repo / "reproducibility/h2oplus_v16_headway_reserve_registration_v1.json"
        authorization_path = repo / "reproducibility/h2oplus_v16_headway_reserve_authorization_v1.json"
        v15_results = repo / "reproducibility/h2oplus_v15_hold_calibration_results"
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
            "critical_matrix_source_file_count": 51,
            "scientific_source_file_count": len(scientific_sources),
            "sumo_runtime_signature": runtime_signature,
            "parent_v15_artifact_integrity_passed": True,
            "parent_v14_training_stability_passed": True,
            "prior_v14_and_v15_development_outcomes_used_for_design": True,
            "data_policy": data_policy(entries, EXPECTED["dataset"]),
            "parent_v15_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
            },
            "authorization": {"path": str(authorization_path), "sha256": sha256(authorization_path)},
            "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "methods_manifest": {"path": str(methods_path), "sha256": sha256(methods_path)},
            "source_freeze": {"path": str(source_freeze_path), "sha256": sha256(source_freeze_path)},
            "parent_v15_evidence": {
                "analysis": {
                    "path": str(v15_results / "hold_calibration_analysis_v1.json"),
                    "sha256": EXPECTED["v15_analysis"],
                },
                "audit": {
                    "path": str(v15_results / "evaluation_audit_v1.json"),
                    "sha256": EXPECTED["v15_audit"],
                },
                "closeout": {
                    "path": str(v15_results / "evaluation_closeout_v1.json"),
                    "sha256": EXPECTED["v15_closeout"],
                },
                "verification": {
                    "path": str(v15_results / "evaluation_closeout_verification_v1.json"),
                    "sha256": EXPECTED["v15_verification"],
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
            raise ValueError("v16 deployment manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v16 deployment file membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v16 deployment file drifted: {row['path']}")
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
            or manifest.get("critical_matrix_source_file_count") != 51
            or manifest.get("scientific_source_file_count") != 14
            or manifest.get("parent_v15_artifact_integrity_passed") is not True
            or manifest.get("parent_v14_training_stability_passed") is not True
            or manifest.get("prior_v14_and_v15_development_outcomes_used_for_design")
            is not True
            or manifest.get("methods_manifest", {}).get("sha256") != EXPECTED["methods"]
            or manifest.get("source_freeze", {}).get("sha256") != EXPECTED["source_freeze"]
            or manifest.get("dataset", {}).get("sha256") != EXPECTED["dataset"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256") != EXPECTED["runtime_lock"]
            or manifest.get("parent_v15_evidence", {}).get("analysis", {}).get("sha256")
            != EXPECTED["v15_analysis"]
            or manifest.get("parent_v15_evidence", {}).get("audit", {}).get("sha256")
            != EXPECTED["v15_audit"]
            or manifest.get("parent_v15_evidence", {}).get("closeout", {}).get("sha256")
            != EXPECTED["v15_closeout"]
            or manifest.get("parent_v15_evidence", {}).get("verification", {}).get("sha256")
            != EXPECTED["v15_verification"]
        ):
            raise ValueError("v16 deployment scope drifted")
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
            raise FileExistsError(f"refusing to overwrite v16 deployment audit: {destination}")
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
        default=ROOT / "reproducibility/deployments/h2oplus_v15_hold_calibration_evaluation_v1/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/deployments/h2oplus_v16_headway_reserve_evaluation_v1/workspace",
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
