#!/usr/bin/env python3
"""Build or verify the isolated parser-corrected v18-v2 deployment."""

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
from reproducibility.build_h2oplus_v18_empty_tail_reserve_evaluation_deployment import (
    data_policy,
)


SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v2"
AUDIT_SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-audit-v2"
MANIFEST_NAME = "deployment_manifest.json"
EXPECTED_PARENT_MANIFEST = (
    "b837b687951c8dc7f84b04defea90158b67fe9f5ec34ace96c204b698852d9b7"
)
EXPECTED_PARENT_AGGREGATE = (
    "8e8df7df5517c14564a39cb3f4e938678e9b5d9705978ba03ccd66768679e538"
)
EXPECTED = {
    "authorization": "e1f9a4b3237468e984d463ed42315e53906f95ee004f91d07086fa428713c085",
    "correction": "6fee854c13083c00c05a9b60ab9340191effd827ac460d1bf7953cb4839e3822",
    "registration": "0a36553e568664594a7c25139f10a62f481be16b98af767b27add4bd05a0f6d7",
    "rationale": "d19a3fdf2b5daa6407f1cdfe905df89bad43dfa7fb5941109026ea43d6c432b0",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "source_freeze": "7be31254cf1433642fd6078781dc483101ca3c2baf9c53be9dd0605ff69819ee",
    "v1_submission": "6ddda52a1619288451a0c5d0dbb8ab4052a526db6e0f17d0d37b814ca4d613b5",
    "incident_closeout": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
NEW_REPO_FILES = (
    "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py",
    "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v2.py",
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v2.py",
    "reproducibility/h2oplus_v18_empty_tail_reserve_successor_authorization_v2.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_parser_correction_v2.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_successor_registration_v2.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v2.json",
    "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_submission_v1.json",
    (
        "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_incidents/v1_parser_contract/incident_closeout_v1.json"
    ),
    (
        "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
        "evaluation_incidents/v1_parser_contract/incident_verification_v1.json"
    ),
)
PARENT_MANIFEST_COPY = (
    "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_deployment_manifest_v1.json"
)


def clone_parent(parent: Path, output: Path) -> None:
    shutil.copytree(parent, output, copy_function=os.link, symlinks=True)
    manifest = output / MANIFEST_NAME
    if sha256(manifest) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("v18-v1 parent manifest drifted during clone")
    manifest.unlink()


def copy_overlay(parent_manifest: Path, output: Path) -> None:
    repo = output / "H2Oplus"
    for relative in NEW_REPO_FILES:
        source = (ROOT / relative).resolve(strict=True)
        destination = repo / relative
        if destination.exists():
            raise FileExistsError(f"v18-v2 overlay would replace parent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    destination = repo / PARENT_MANIFEST_COPY
    if destination.exists():
        raise FileExistsError(f"v18-v1 manifest copy already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_manifest, destination)


def input_paths() -> dict[str, Path]:
    return {
        "authorization": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_successor_authorization_v2.json",
        "correction": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_parser_correction_v2.json",
        "registration": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_successor_registration_v2.json",
        "rationale": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_rationale_v1.json",
        "protocol": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "methods": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "environment": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "source_freeze": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v2.json",
        "v1_submission": ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_submission_v1.json",
        "incident_closeout": ROOT
        / (
            "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
            "evaluation_incidents/v1_parser_contract/incident_closeout_v1.json"
        ),
        "incident_verification": ROOT
        / (
            "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
            "evaluation_incidents/v1_parser_contract/incident_verification_v1.json"
        ),
        "dataset": ROOT
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": ROOT
        / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }


def validate_inputs() -> dict[str, Path]:
    paths = input_paths()
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"v18-v2 deployment input drifted: {name}")
    return paths


def build(args: argparse.Namespace) -> int:
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / MANIFEST_NAME
    if sha256(parent_manifest_path) != EXPECTED_PARENT_MANIFEST:
        raise ValueError("sealed v18-v1 parent manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != EXPECTED_PARENT_AGGREGATE
        or parent_manifest.get("entry_count") != 814
        or parent_manifest.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("sealed v18-v1 parent deployment scope drifted")
    paths = validate_inputs()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v18-v2 deployment: {output}")
    try:
        clone_parent(parent, output)
        copy_overlay(parent_manifest_path, output)
        repo = output / "H2Oplus"
        repo_path = lambda name: repo / paths[name].relative_to(ROOT)
        source_freeze_path = repo_path("source_freeze")
        source_freeze = json.loads(source_freeze_path.read_text())
        entries = [
            descriptor(path, output)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        ]
        runtime_lock_path = repo_path("runtime_lock")
        runtime = json.loads(runtime_lock_path.read_text())
        runtime_signature = runtime["runtime_fingerprint"]["portable_signature"]
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
            "parser_contract_correction_only": True,
            "scientific_behavior_unchanged_from_v18_v1": True,
            "v18_v1_result_count": 0,
            "v18_v1_results_reused": False,
            "formal_held_out_outcomes_included": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "physical_behavior_count": 15,
            "logical_method_count": 15,
            "new_actor_file_count": 0,
            "expected_new_remote_sumo_episodes": 135,
            "critical_matrix_source_file_count": 57,
            "scientific_source_file_count": len(scientific_sources),
            "sumo_runtime_signature": runtime_signature,
            "parent_v17_artifact_integrity_passed": True,
            "parent_training_stability_passed": True,
            "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
            "data_environment_material_factors_disclosed": True,
            "data_policy": data_policy(entries, EXPECTED["dataset"]),
            "parent_v18_v1_deployment": {
                "path": str(parent_manifest_path),
                "sha256": EXPECTED_PARENT_MANIFEST,
                "aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
                "completed_result_count": 0,
                "results_reused": False,
            },
            "parent_v18_v1_deployment_manifest_copy": {
                "path": str(repo / PARENT_MANIFEST_COPY),
                "sha256": EXPECTED_PARENT_MANIFEST,
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
            "rationale": {
                "path": str(repo_path("rationale")),
                "sha256": EXPECTED["rationale"],
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
            "v18_v1_submission": {
                "path": str(repo_path("v1_submission")),
                "sha256": EXPECTED["v1_submission"],
            },
            "v18_v1_incident": {
                "closeout": {
                    "path": str(repo_path("incident_closeout")),
                    "sha256": EXPECTED["incident_closeout"],
                },
                "verification": {
                    "path": str(repo_path("incident_verification")),
                    "sha256": EXPECTED["incident_verification"],
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
            raise ValueError("v18-v2 manifest aggregate drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != {row["path"] for row in entries}:
            raise ValueError("v18-v2 deployment membership drifted")
        for row in entries:
            path = root / row["path"]
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"v18-v2 file drifted: {row['path']}")
        data_policy(entries, EXPECTED["dataset"])
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("parser_contract_correction_only") is not True
            or manifest.get("scientific_behavior_unchanged_from_v18_v1") is not True
            or manifest.get("v18_v1_result_count") != 0
            or manifest.get("v18_v1_results_reused") is not False
            or manifest.get("formal_held_out_outcomes_included") is not False
            or manifest.get("physical_behavior_count") != 15
            or manifest.get("expected_new_remote_sumo_episodes") != 135
            or manifest.get("critical_matrix_source_file_count") != 57
            or manifest.get("scientific_source_file_count") != 21
            or manifest.get("authorization", {}).get("sha256")
            != EXPECTED["authorization"]
            or manifest.get("correction", {}).get("sha256") != EXPECTED["correction"]
            or manifest.get("registration", {}).get("sha256")
            != EXPECTED["registration"]
            or manifest.get("source_freeze", {}).get("sha256")
            != EXPECTED["source_freeze"]
            or manifest.get("methods_manifest", {}).get("sha256")
            != EXPECTED["methods"]
            or manifest.get("sumo_runtime_lock", {}).get("sha256")
            != EXPECTED["runtime_lock"]
        ):
            raise ValueError("v18-v2 deployment scope drifted")
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
        "parser_contract_correction_only": True,
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
    }
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite v18-v2 audit: {destination}")
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
        / "reproducibility/deployments/h2oplus_v18_empty_tail_reserve_evaluation_v1/workspace",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility/deployments/h2oplus_v18_empty_tail_reserve_evaluation_v2/workspace",
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument(
        "--root",
        type=Path,
        default=ROOT
        / "reproducibility/deployments/h2oplus_v18_empty_tail_reserve_evaluation_v2/workspace",
    )
    verify_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
