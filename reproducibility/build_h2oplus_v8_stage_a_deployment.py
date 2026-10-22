#!/usr/bin/env python3
"""Build or verify the self-contained v8 Stage A remote evaluation workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
)
from reproducibility.build_evaluation_deployment_v7 import (
    _python_sources,
    _source_destination,
)
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    evaluation_source_paths_v7,
    load_methods,
)


SCHEMA_VERSION = "transit-h2oplus-v8-stage-a-deployment-v1"
AUDIT_SCHEMA = "transit-h2oplus-v8-stage-a-deployment-audit-v1"
DESIGN_SCHEMA = "transit-h2oplus-v8-development-environment-v1"
MANIFEST_NAME = "deployment_manifest.json"


def aggregate(entries: list[dict]) -> str:
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def descriptor(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def resolve_relative(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def copy_source(source: Path, destination: Path, link_large_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if link_large_files and source.stat().st_size >= 100 * 1024**2:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def bound(path: Path, expected_sha256: str) -> Path:
    resolved = path.expanduser().resolve()
    actual = sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"hash mismatch for {resolved}: expected {expected_sha256}, got {actual}"
        )
    return resolved


def collect_sources(args: argparse.Namespace) -> tuple[set[Path], dict]:
    registration = bound(args.registration, args.expected_registration_sha256)
    authorization = bound(args.authorization, args.expected_authorization_sha256)
    freeze = bound(args.freeze, args.expected_freeze_sha256)
    design = bound(args.environment_design, args.expected_design_sha256)
    methods_manifest = bound(args.methods_manifest, args.expected_methods_sha256)
    export_manifest = bound(
        args.policy_export_manifest, args.expected_policy_export_sha256
    )
    dataset = bound(args.dataset, args.expected_dataset_sha256)
    runtime_lock = bound(args.sumo_runtime_lock, args.expected_runtime_lock_sha256)
    runtime_requirements = bound(
        args.runtime_requirements, args.expected_runtime_requirements_sha256
    )

    authorization_payload = json.loads(authorization.read_text())
    freeze_payload = json.loads(freeze.read_text())
    design_payload = json.loads(design.read_text())
    registration_payload = json.loads(registration.read_text())
    export_payload = json.loads(export_manifest.read_text())
    if (
        registration_payload.get("execution_authorized") is not False
        or registration_payload.get("local_sumo_or_training_allowed") is not False
        or authorization_payload.get("authorization_decision")
        != "authorize-stage-a-execution"
        or authorization_payload.get("formal_held_out_access_authorized") is not False
        or freeze_payload.get("training") is not False
        or freeze_payload.get("formal_held_out_outcomes_read") is not False
        or design_payload.get("schema_version") != DESIGN_SCHEMA
        or design_payload.get("study_stage") != "development"
        or design_payload.get("disjointness", {}).get("verified_disjoint") is not True
        or design_payload.get("disjointness", {}).get(
            "formal_held_out_outcomes_read"
        )
        is not False
    ):
        raise ValueError("v8 Stage A scope or authorization drifted")
    overlap_keys = [
        key
        for key, value in design_payload["disjointness"].items()
        if key.endswith("_overlap") and value != []
    ]
    if overlap_keys:
        raise ValueError(f"development disjointness overlap: {overlap_keys}")

    methods = load_methods(methods_manifest, METHODS_SCHEMAS["v7"])
    if len(methods) != 10:
        raise ValueError("Stage A deployment requires ten frozen policies")
    sources = _python_sources()
    sources.update(
        {
            registration,
            authorization,
            freeze,
            design,
            methods_manifest,
            export_manifest,
            dataset,
            runtime_lock,
            runtime_requirements,
            args.edge_xml.expanduser().resolve(),
        }
    )
    for method in methods:
        sources.add(Path(method["checkpoint"]).resolve())

    export_root = export_manifest.parent
    export_files = export_payload.get("files")
    if not isinstance(export_files, list) or len(export_files) != 30:
        raise ValueError("Stage A policy export manifest drifted")
    for entry in export_files:
        source = (export_root / entry["path"]).resolve()
        if export_root not in source.parents:
            raise ValueError(f"policy export path escapes its root: {source}")
        if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
            raise ValueError(f"policy export source drifted: {source}")
        sources.add(source)

    wheel_dir = args.wheel_dir.expanduser().resolve()
    requirements_payload = json.loads(runtime_requirements.read_text())
    wheel_descriptors = []
    for wheel in requirements_payload.get("wheels", []):
        source = (wheel_dir / wheel["filename"]).resolve()
        if source.parent != wheel_dir or sha256(source) != wheel["sha256"]:
            raise ValueError(f"runtime wheel drifted: {source}")
        sources.add(source)
        wheel_descriptors.append(
            {
                "path": str(source),
                "sha256": wheel["sha256"],
                "size_bytes": int(source.stat().st_size),
            }
        )

    scenario_descriptors = []
    blocks = design_payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise ValueError("Stage A environment block count drifted")
    for block in blocks:
        for raw_manifest in block.get("scenario_manifests", []):
            manifest = resolve_relative(design, raw_manifest)
            runtime, evidence = load_validated_scenario(manifest)
            scenario_paths = {
                manifest,
                Path(runtime["sumo_cfg"]).resolve(),
                Path(runtime["passenger_obj_xml"]).resolve(),
                *(Path(path).resolve() for path in evidence["runtime_input_sha256"]),
            }
            sources.update(scenario_paths)
            scenario_descriptors.append(
                {
                    "block": block["name"],
                    "manifest": str(manifest),
                    "manifest_sha256": evidence["manifest_sha256"],
                    "runtime_content_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                "digests": sorted(
                                    evidence["runtime_input_sha256"].values()
                                )
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
            )
    if len(scenario_descriptors) != 15:
        raise ValueError("Stage A deployment requires 15 scenarios")

    sources.update(
        evaluation_source_paths_v7(
            args.sumo_dir.expanduser().resolve(),
            args.edge_xml.expanduser().resolve(),
        )
    )
    for generator in (design_payload.get("generator"), design_payload.get("scenario_generator")):
        if isinstance(generator, dict):
            source = Path(generator["path"]).expanduser().resolve()
            if sha256(source) != generator["sha256"]:
                raise ValueError(f"development generator drifted: {source}")
            sources.add(source)
    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"deployment sources are missing: {missing[:20]}")

    lock_payload, lock_identity = load_sumo_runtime_lock(
        runtime_lock, expected_sha256=args.expected_runtime_lock_sha256
    )
    metadata = {
        "registration": {"path": str(registration), "sha256": sha256(registration)},
        "authorization": {"path": str(authorization), "sha256": sha256(authorization)},
        "freeze": {"path": str(freeze), "sha256": sha256(freeze)},
        "environment_design": {
            "path": str(design),
            "sha256": sha256(design),
            "cells": 15,
        },
        "methods_manifest": {
            "path": str(methods_manifest),
            "sha256": sha256(methods_manifest),
            "methods": [method["name"] for method in methods],
        },
        "policy_export_manifest": {
            "path": str(export_manifest),
            "sha256": sha256(export_manifest),
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256(dataset),
            "size_bytes": int(dataset.stat().st_size),
        },
        "sumo_runtime_lock": lock_identity,
        "sumo_runtime_signature": lock_payload["runtime_signature"],
        "runtime_requirements": {
            "path": str(runtime_requirements),
            "sha256": sha256(runtime_requirements),
        },
        "runtime_wheels": wheel_descriptors,
        "scenario_descriptors": scenario_descriptors,
    }
    return sources, metadata


def data_policy(entries: list[dict], dataset_sha256: str) -> dict:
    hdf5 = [entry for entry in entries if Path(entry["path"]).suffix.lower() in {".h5", ".hdf5"}]
    actor_files = [entry for entry in entries if Path(entry["path"]).name == "final_actor.pt"]
    forbidden = [
        entry["path"]
        for entry in entries
        if Path(entry["path"]).suffix.lower() in {".pth", ".ckpt", ".pkl"}
        or (
            Path(entry["path"]).suffix.lower() == ".pt"
            and Path(entry["path"]).name != "final_actor.pt"
        )
    ]
    if len(hdf5) != 1 or hdf5[0]["sha256"] != dataset_sha256:
        raise ValueError("deployment must contain exactly the canonical Stage A HDF5")
    if len(actor_files) != 10 or forbidden:
        raise ValueError(
            f"deployment policy payload drifted: actors={len(actor_files)}, forbidden={forbidden}"
        )
    return {
        "hdf5_files": 1,
        "compact_final_actor_files": 10,
        "training_checkpoint_or_optimizer_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": dataset_sha256,
    }


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace deployment {output}")
    sources, metadata = collect_sources(args)
    output.mkdir(parents=True)
    try:
        for source in sorted(sources):
            copy_source(
                source,
                _source_destination(output, source),
                args.link_large_files,
            )
        files = sorted(path for path in output.rglob("*") if path.is_file())
        entries = [descriptor(path, output) for path in files]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(ROOT),
            "entry_count": len(entries),
            "total_file_bytes": sum(entry["size_bytes"] for entry in entries),
            "aggregate_sha256": aggregate(entries),
            "training": False,
            "study_stage": "development",
            "formal_held_out_outcomes_included": False,
            "data_policy": data_policy(entries, metadata["dataset"]["sha256"]),
            **metadata,
            "entries": entries,
        }
        destination = output / MANIFEST_NAME
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "manifest_sha256": sha256(destination),
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
    root = args.root.expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    errors = []
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("deployment manifest schema mismatch")
        entries = manifest.get("entries", [])
        if aggregate(entries) != manifest.get("aggregate_sha256"):
            raise ValueError("deployment aggregate mismatch")
        for entry in entries:
            path = root / entry["path"]
            if (
                not path.is_file()
                or path.stat().st_size != entry["size_bytes"]
                or sha256(path) != entry["sha256"]
            ):
                raise ValueError(f"missing or stale deployment file: {entry['path']}")
        data_policy(entries, manifest["dataset"]["sha256"])
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("formal_held_out_outcomes_included") is not False
        ):
            raise ValueError("deployment scientific scope drifted")
        lock_path = root / "H2Oplus" / "reproducibility" / Path(
            manifest["sumo_runtime_lock"]["path"]
        ).name
        lock, identity = load_sumo_runtime_lock(
            lock_path,
            expected_sha256=manifest["sumo_runtime_lock"]["sha256"],
        )
        if identity["runtime_signature"] != manifest["sumo_runtime_signature"]:
            raise ValueError("deployment runtime lock signature drifted")
        if args.verify_runtime:
            active = sumo_runtime_fingerprint(
                FORMAL_V7_SUMO_BACKEND,
                expected_version=FORMAL_V7_SUMO_VERSION,
            )
            if (
                active["portable_signature"] != identity["runtime_signature"]
                or active["portable_identity"]
                != lock["runtime_fingerprint"]["portable_identity"]
            ):
                raise RuntimeError("active server runtime differs from deployment lock")
    except Exception as error:
        errors.append(str(error))
        manifest = {}
    report = {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "runtime_verified": bool(args.verify_runtime and not errors),
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    for name in (
        "registration",
        "authorization",
        "freeze",
        "environment_design",
        "methods_manifest",
        "policy_export_manifest",
        "dataset",
        "sumo_runtime_lock",
        "runtime_requirements",
        "wheel_dir",
        "sumo_dir",
        "edge_xml",
        "output",
    ):
        build_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in (
        "registration",
        "authorization",
        "freeze",
        "design",
        "methods",
        "policy_export",
        "dataset",
        "runtime_lock",
        "runtime_requirements",
    ):
        build_parser.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256", required=True
        )
    build_parser.add_argument("--link-large-files", action="store_true")
    build_parser.set_defaults(function=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--verify-runtime", action="store_true")
    verify_parser.set_defaults(function=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
