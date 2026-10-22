#!/usr/bin/env python3
"""Build or independently verify the isolated v18 evaluation deployment."""

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


SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = (
    "8555e2c2d727eaf17a81ce17b07bbcb07b0b1e3f9dfd334babb96afd98a3ad10"
)
EXPECTED_PARENT_AGGREGATE = (
    "1cea33aa52cf6ad65dbd9d6272a237e7bf58bc80bba593cc93e7bd89ac129a06"
)
EXPECTED = {
    "authorization": "8354467e41971cc39dc99382a2327a75c358f8baa06df22e215854a5c8036f92",
    "rationale": "d19a3fdf2b5daa6407f1cdfe905df89bad43dfa7fb5941109026ea43d6c432b0",
    "registration": "45e160f7104a8a494d3231576c15c09ba9f28f184e3a6da4ac6637a34e3135dc",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "source_freeze": "1769740ba02cb14a9ca483460c44f4651109fe9e297dec9d42409bc802df49a7",
    "v17_analysis": "84e80b7f08d16f6e356f09c56ff75d1bbdf322578641df4ede9e73f8d4ff4f78",
    "v17_audit": "c23e5a3f0e7b2a9de889cee024f293f3e07f9087689d16f3c649b23fff7d39ec",
    "v17_closeout": "09351ebb883e5aad18d33353be877ed542f1ed7741e1d03b0c9b4368b39e268b",
    "v17_verification": "d781daea3138595989156935367b15a9e8db543aa464bad0a2c58636db95d596",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v18_empty_tail_reserve.py",
    "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve.py",
    "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
    "reproducibility/h2oplus_v18_empty_tail_reserve_authorization_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_rationale_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_registration_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_protocol_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_environment_v1.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v1.json",
    (
        "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "tail_service_reserve_analysis_v2.json"
    ),
    (
        "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "evaluation_audit_v2.json"
    ),
    (
        "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "evaluation_closeout_v2.json"
    ),
    (
        "reproducibility/h2oplus_v17_tail_service_reserve_results/"
        "evaluation_closeout_verification_v2.json"
    ),
)
SCENARIO_ROOT = "bus_h2o/development_scenarios_v18/empty_tail_reserve_v1"


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
            raise FileExistsError(f"v18 overlay would replace parent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_scenarios = (ROOT / SCENARIO_ROOT).resolve(strict=True)
    destination_scenarios = repo / SCENARIO_ROOT
    if destination_scenarios.exists():
        raise FileExistsError(f"v18 scenario overlay exists: {destination_scenarios}")
    shutil.copytree(source_scenarios, destination_scenarios)


def data_policy(entries: list[dict], dataset_hash: str) -> dict:
    hdf5 = [
        row
        for row in entries
        if Path(row["path"]).suffix.lower() in {".h5", ".hdf5"}
    ]
    actors = [
        row for row in entries if Path(row["path"]).suffix.lower() == ".pt"
    ]
    forbidden = [
        row["path"]
        for row in entries
        if Path(row["path"]).suffix.lower() in {".pth", ".ckpt", ".pkl"}
        or "remote_results" in Path(row["path"]).parts
        or "evaluation_partition" in Path(row["path"]).parts
        or Path(row["path"]).name.startswith("sumo_seed")
    ]
    scenario_files = [
        row for row in entries if SCENARIO_ROOT in row["path"]
    ]
    if (
        len(hdf5) != 1
        or hdf5[0]["sha256"] != dataset_hash
        or len(actors) != 85
        or len(scenario_files) != 63
        or forbidden
    ):
        raise ValueError(
            "v18 deployment data policy drifted: "
            f"hdf5={len(hdf5)}, pt={len(actors)}, "
            f"scenario_files={len(scenario_files)}, forbidden={forbidden}"
        )
    return {
        "hdf5_files": 1,
        "compact_actor_files": 85,
        "new_actor_files": 0,
        "fresh_v18_scenario_files": 63,
        "training_checkpoint_or_optimizer_files": 0,
        "physical_evaluation_result_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": dataset_hash,
    }


def input_paths() -> dict[str, Path]:
    result_root = (
        ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_results"
    )
    return {
        "authorization": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_authorization_v1.json",
        "rationale": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_rationale_v1.json",
        "registration": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_registration_v1.json",
        "protocol": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "methods": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "environment": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "source_freeze": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v1.json",
        "v17_analysis": result_root / "tail_service_reserve_analysis_v2.json",
        "v17_audit": result_root / "evaluation_audit_v2.json",
        "v17_closeout": result_root / "evaluation_closeout_v2.json",
        "v17_verification": result_root
        / "evaluation_closeout_verification_v2.json",
        "data_environment_audit": ROOT
        / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": ROOT
        / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT
        / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }


def validate_inputs() -> dict[str, Path]:
    paths = input_paths()
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v18 deployment input drifted: {name}")
    return paths


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    if sha256(parent_manifest_path) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("sealed v17-v2 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 736
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v17-v2 parent deployment scope drifted")
    paths = validate_inputs()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v18 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_new_files(output)
        repo = output / "H2Oplus"
        repo_path = lambda name: repo / paths[name].relative_to(ROOT)
        source_freeze_path = repo_path("source_freeze")
        source_freeze = json.loads(source_freeze_path.read_text())
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        runtime_lock_path = repo_path("runtime_lock")
        runtime_payload = json.loads(runtime_lock_path.read_text())
        runtime_signature = runtime_payload["runtime_fingerprint"][
            "portable_signature"
        ]
        scientific_sources = (
            source_freeze["matrix_sources"]
            + source_freeze["operational_sources"]
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
            "formal_held_out_outcomes_included": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "physical_behavior_count": 15,
            "logical_method_count": 15,
            "new_actor_file_count": 0,
            "expected_new_remote_sumo_episodes": 135,
            "critical_matrix_source_file_count": 55,
            "scientific_source_file_count": len(scientific_sources),
            "fresh_v18_scenario_file_count": 63,
            "sumo_runtime_signature": runtime_signature,
            "parent_v17_artifact_integrity_passed": True,
            "parent_training_stability_passed": True,
            "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
            "data_environment_material_factors_disclosed": True,
            "data_policy": data_policy(entries, EXPECTED["dataset"]),
            "parent_v17_v2_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
            },
            "authorization": {
                "path": str(repo_path("authorization")),
                "sha256": EXPECTED["authorization"],
            },
            "rationale": {
                "path": str(repo_path("rationale")),
                "sha256": EXPECTED["rationale"],
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
            "environment_design": {
                "path": str(repo_path("environment")),
                "sha256": EXPECTED["environment"],
            },
            "source_freeze": {
                "path": str(source_freeze_path),
                "sha256": EXPECTED["source_freeze"],
            },
            "parent_v17_evidence": {
                name: {"path": str(repo_path(name)), "sha256": EXPECTED[name]}
                for name in (
                    "v17_analysis",
                    "v17_audit",
                    "v17_closeout",
                    "v17_verification",
                )
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
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
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
    errors = []
    manifest = {}
    try:
        validate_inputs()
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if (
            manifest.get("schema_version") != SCHEMA
            or aggregate(entries) != manifest.get("aggregate_sha256")
        ):
            raise ValueError("v18 deployment manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v18 deployment file membership drifted")
        for row in entries:
            path = root / row["path"]
            if (
                path.stat().st_size != row["size_bytes"]
                or sha256(path) != row["sha256"]
            ):
                raise ValueError(f"v18 deployment file drifted: {row['path']}")
        data_policy(entries, EXPECTED["dataset"])
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("screen_type") != "paired-mechanism-screen"
            or manifest.get("formal_held_out_outcomes_included") is not False
            or manifest.get("formal_confirmation_stage_authorized_or_submitted")
            is not False
            or manifest.get("physical_behavior_count") != 15
            or manifest.get("logical_method_count") != 15
            or manifest.get("new_actor_file_count") != 0
            or manifest.get("expected_new_remote_sumo_episodes") != 135
            or manifest.get("critical_matrix_source_file_count") != 55
            or manifest.get("scientific_source_file_count") != 18
            or manifest.get("fresh_v18_scenario_file_count") != 63
            or manifest.get("parent_v17_artifact_integrity_passed") is not True
            or manifest.get("parent_training_stability_passed") is not True
            or manifest.get("data_environment_material_factors_disclosed") is not True
            or manifest.get("authorization", {}).get("sha256")
            != EXPECTED["authorization"]
            or manifest.get("rationale", {}).get("sha256")
            != EXPECTED["rationale"]
            or manifest.get("registration", {}).get("sha256")
            != EXPECTED["registration"]
            or manifest.get("protocol", {}).get("sha256") != EXPECTED["protocol"]
            or manifest.get("methods_manifest", {}).get("sha256")
            != EXPECTED["methods"]
            or manifest.get("environment_design", {}).get("sha256")
            != EXPECTED["environment"]
            or manifest.get("source_freeze", {}).get("sha256")
            != EXPECTED["source_freeze"]
            or manifest.get("dataset", {}).get("sha256") != EXPECTED["dataset"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256")
            != EXPECTED["runtime_lock"]
        ):
            raise ValueError("v18 deployment scope drifted")
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
                f"refusing to overwrite v18 deployment audit: {destination}"
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
        / "reproducibility/deployments/"
        "h2oplus_v17_tail_service_reserve_evaluation_v2/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility/deployments/"
        "h2oplus_v18_empty_tail_reserve_evaluation_v1/workspace",
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT
        / "reproducibility/deployments/"
        "h2oplus_v18_empty_tail_reserve_evaluation_v1/workspace",
    )
    verify_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build(args)
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
