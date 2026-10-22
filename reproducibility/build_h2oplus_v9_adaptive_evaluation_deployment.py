#!/usr/bin/env python3
"""Build or verify the development-only v9 adaptive SUMO deployment."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import load_sumo_runtime_lock, load_validated_scenario, sha256
from reproducibility.build_evaluation_deployment_v7 import (
    _python_sources,
    _source_destination,
)
from reproducibility.run_evaluation_matrix_v9_adaptive import (
    METHODS_SCHEMA,
    evaluation_sources,
    load_development_methods,
)


LEGACY_SCHEMAS = {"transit-h2oplus-v9-adaptive-evaluation-deployment-v1"}
SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-deployment-v2"
AUDIT_SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-deployment-audit-v2"
MANIFEST_NAME = "deployment_manifest.json"


def aggregate(entries: list[dict]) -> str:
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def descriptor(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def bound(path: Path, expected: str, schema: str | None = None) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected:
        raise ValueError(f"identity drift for {resolved}: {actual}")
    payload = json.loads(resolved.read_text())
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"schema drift for {resolved}")
    return resolved, payload


def resolve_relative(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def add_source(sources: set[Path], path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    sources.add(resolved)
    return resolved


def collect(args: argparse.Namespace) -> tuple[set[Path], dict]:
    registration, registration_payload = bound(
        args.registration,
        args.expected_registration_sha256,
        "transit-h2oplus-v9-adaptive-registration-v2",
    )
    authorization, authorization_payload = bound(
        args.authorization,
        args.expected_authorization_sha256,
        "transit-h2oplus-v9-adaptive-authorization-v2",
    )
    freeze, freeze_payload = bound(
        args.freeze,
        args.expected_freeze_sha256,
        "transit-h2oplus-v9-adaptive-freeze-v2",
    )
    design, design_payload = bound(
        args.environment_design,
        args.expected_design_sha256,
        "transit-h2oplus-v9-adaptive-environment-v2",
    )
    protocol, protocol_payload = bound(
        args.protocol,
        args.expected_protocol_sha256,
        "transit-h2oplus-v9-adaptive-protocol-v2",
    )
    count_correction, count_payload = bound(
        args.count_correction,
        args.expected_count_correction_sha256,
        "transit-h2oplus-v9-adaptive-evaluation-count-correction-v1",
    )
    source_freeze, source_payload = bound(
        args.source_freeze,
        args.expected_source_freeze_sha256,
        "transit-h2oplus-v9-adaptive-evaluation-source-freeze-v1",
    )
    source_correction_descriptor = source_payload.get("source_correction", {})
    source_correction = resolve_relative(
        source_freeze, source_correction_descriptor.get("path", "")
    )
    if (
        source_correction == source_freeze.parent
        or not source_correction.is_file()
        or sha256(source_correction) != source_correction_descriptor.get("sha256")
    ):
        raise ValueError("v9 evaluation source correction drifted")
    source_correction_payload = json.loads(source_correction.read_text())
    if (
        source_correction_payload.get("schema_version")
        != "transit-h2oplus-v9-adaptive-evaluation-source-correction-v2"
        or source_correction_payload.get("formal_held_out_outcomes_read") is not False
        or source_correction_payload.get("scientific_evaluation_outcomes_read") is not False
    ):
        raise ValueError("v9 evaluation source correction scope drifted")
    closeout, closeout_payload = bound(
        args.training_closeout,
        args.expected_training_closeout_sha256,
        "transit-h2oplus-v9-adaptive-training-closeout-v1",
    )
    verification, verification_payload = bound(
        args.training_verification,
        args.expected_training_verification_sha256,
        "transit-h2oplus-v9-adaptive-training-verification-v1",
    )
    training_audit, training_audit_payload = bound(
        args.training_audit,
        args.expected_training_audit_sha256,
        "transit-h2oplus-v9-adaptive-training-audit-v1",
    )
    training_manifest, training_manifest_payload = bound(
        args.training_file_manifest,
        args.expected_training_file_manifest_sha256,
        "transit-h2oplus-v9-adaptive-training-file-manifest-v1",
    )
    methods = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods) != args.expected_methods_sha256:
        raise ValueError("v9 adaptive methods identity drifted")
    methods_payload = json.loads(methods.read_text())
    method_rows = load_development_methods(methods, METHODS_SCHEMA)
    portability = methods_payload.get("portability_correction", {})
    portability_path = resolve_relative(methods, portability.get("path", ""))
    if (
        portability_path == methods.parent
        or not portability_path.is_file()
        or sha256(portability_path) != portability.get("sha256")
    ):
        raise ValueError("v9 evaluation portability correction drifted")
    portability_payload = json.loads(portability_path.read_text())
    if (
        portability_payload.get("schema_version")
        != "transit-h2oplus-v9-adaptive-evaluation-portability-correction-v1"
        or portability_payload.get("scientific_evaluation_outcomes_read") is not False
        or portability_payload.get("formal_held_out_outcomes_read") is not False
        or portability_payload.get("corrected_methods_manifest", {}).get("path")
        != methods.name
        or portability_payload.get("corrected_methods_manifest", {}).get("schema_version")
        != METHODS_SCHEMA
    ):
        raise ValueError("v9 evaluation portability correction scope drifted")
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("v9 adaptive dataset identity drifted")
    runtime_lock = args.sumo_runtime_lock.expanduser().resolve(strict=True)
    if sha256(runtime_lock) != args.expected_runtime_lock_sha256:
        raise ValueError("SUMO runtime lock identity drifted")
    runtime_payload, runtime_identity = load_sumo_runtime_lock(
        runtime_lock, expected_sha256=args.expected_runtime_lock_sha256
    )
    runtime_requirements = args.runtime_requirements.expanduser().resolve(strict=True)
    if sha256(runtime_requirements) != args.expected_runtime_requirements_sha256:
        raise ValueError("runtime requirements identity drifted")
    formal_deployment = args.formal_deployment_manifest.expanduser().resolve(strict=True)
    if sha256(formal_deployment) != args.expected_formal_deployment_sha256:
        raise ValueError("frozen v7 deployment identity drifted")

    v8_protocol, v8_protocol_payload = bound(
        args.v8_protocol,
        args.expected_v8_protocol_sha256,
        "transit-h2oplus-v8-fewshot-protocol-v1",
    )
    v8_closeout, v8_closeout_payload = bound(
        args.v8_training_closeout,
        args.expected_v8_training_closeout_sha256,
        "transit-h2oplus-v8-fewshot-training-closeout-v1",
    )
    v8_verification, v8_verification_payload = bound(
        args.v8_training_verification,
        args.expected_v8_training_verification_sha256,
        "transit-h2oplus-v8-fewshot-training-verification-v1",
    )

    if (
        registration_payload.get("formal_held_out_outcomes_read") is not False
        or authorization_payload.get("formal_held_out_access_authorized") is not False
        or authorization_payload.get("local_training_or_sumo_authorized") is not False
        or freeze_payload.get("formal_held_out_outcomes_read") is not False
        or protocol_payload.get("formal_held_out_outcomes_read") is not False
        or design_payload.get("disjointness", {}).get("verified_disjoint") is not True
        or verification_payload.get("passed") is not True
        or training_audit_payload.get("artifact_integrity", {}).get("passed") is not True
        or closeout_payload.get("artifact_integrity_passed") is not True
        or authorization_payload.get("remote_development_evaluation_after_training_authorized") is not True
        or closeout_payload.get("development_evaluation_may_proceed") is not True
        or closeout_payload.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or training_manifest_payload.get("file_count") != verification_payload.get("checked_files")
        or count_payload.get("performance_or_evaluation_outcomes_read") is not False
        or count_payload.get("corrected_unique_actor_tensors") != 42
        or count_payload.get("corrected_remote_sumo_episodes") != 252
        or source_payload.get("evaluation_outcomes_read") is not False
        or source_payload.get("formal_held_out_outcomes_read") is not False
        or source_payload.get("physical_actor_count") != 42
        or v8_protocol_payload.get("formal_held_out_outcomes_read") is not False
        or v8_closeout_payload.get("artifact_integrity_passed") is not True
        or v8_verification_payload.get("passed") is not True
        or len(method_rows) != 42
    ):
        raise ValueError("v9 deployment scope or sealed training evidence drifted")
    if any(
        value
        for key, value in design_payload.get("disjointness", {}).items()
        if key.endswith("_overlap")
    ):
        raise ValueError("v9 development environment overlaps a frozen design")

    source_bindings = {item["path"]: item["sha256"] for item in source_payload.get("sources", [])}
    expected_source_names = {
        "SimpleSAC/evaluate_transit_v9_adaptive.py",
        "reproducibility/run_evaluation_matrix_v9_adaptive.py",
        "reproducibility/run_h2oplus_v9_adaptive_evaluation_dispatch.py",
        "reproducibility/analyze_h2oplus_v9_adaptive.py",
    }
    if set(source_bindings) != expected_source_names:
        raise ValueError("v9 source freeze membership drifted")
    for relative, expected in source_bindings.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"v9 evaluation source drifted: {relative}")

    sources = _python_sources()
    sources.update(
        {
            registration,
            authorization,
            freeze,
            design,
            protocol,
            count_correction,
            source_freeze,
            source_correction,
            closeout,
            verification,
            training_audit,
            training_manifest,
            methods,
            portability_path,
            dataset,
            runtime_lock,
            runtime_requirements,
            formal_deployment,
            v8_protocol,
            v8_closeout,
            v8_verification,
            (ROOT / "reproducibility/run_h2oplus_v9_adaptive_evaluation_dispatch.py").resolve(strict=True),
            (ROOT / "reproducibility/analyze_h2oplus_v9_adaptive.py").resolve(strict=True),
        }
    )
    sources.update(Path(item["checkpoint"]).resolve(strict=True) for item in method_rows)

    requirements_payload = json.loads(runtime_requirements.read_text())
    wheel_dir = args.wheel_dir.expanduser().resolve(strict=True)
    wheel_evidence = []
    for item in requirements_payload.get("wheels", []):
        wheel = (wheel_dir / item["filename"]).resolve(strict=True)
        if wheel.parent != wheel_dir or sha256(wheel) != item["sha256"]:
            raise ValueError(f"runtime wheel drifted: {wheel}")
        sources.add(wheel)
        wheel_evidence.append({"filename": wheel.name, "sha256": sha256(wheel)})

    blocks = design_payload.get("blocks", [])
    if len(blocks) != 2 or design_payload.get("environment_cells") != 6:
        raise ValueError("v9 environment must contain two blocks and six cells")
    scenarios = []
    for block in blocks:
        if len(block.get("scenario_manifests", [])) != 3:
            raise ValueError("v9 block does not contain three demand scales")
        for raw in block["scenario_manifests"]:
            manifest = resolve_relative(design, raw)
            runtime, evidence = load_validated_scenario(manifest)
            paths = {
                manifest,
                Path(runtime["sumo_cfg"]).resolve(strict=True),
                Path(runtime["passenger_obj_xml"]).resolve(strict=True),
                *(Path(path).resolve(strict=True) for path in evidence["runtime_input_sha256"]),
            }
            sources.update(paths)
            scenarios.append(
                {
                    "block": block["name"],
                    "manifest_sha256": evidence["manifest_sha256"],
                    "demand_scale": float(evidence["demand_scale"]),
                }
            )
    if len(scenarios) != 6:
        raise ValueError("v9 scenario inventory drifted")

    critical_sources = set(
        evaluation_sources(
            args.sumo_dir.expanduser().resolve(strict=True),
            args.edge_xml.expanduser().resolve(strict=True),
        )
    )
    formal_entries = {
        item["path"]: item["sha256"]
        for item in json.loads(formal_deployment.read_text()).get("entries", [])
        if item.get("type") == "file" and "sha256" in item
    }
    v9_specific = {
        "SimpleSAC/evaluate_transit_v9_adaptive.py",
        "reproducibility/run_evaluation_matrix_v9_adaptive.py",
    }
    for source in critical_sources:
        resolved = source.resolve(strict=True)
        if ROOT in resolved.parents:
            relative = resolved.relative_to(ROOT).as_posix()
            expected = formal_entries.get(relative)
            if relative not in v9_specific and expected is not None and sha256(resolved) != expected:
                raise ValueError(f"frozen v7 evaluation source drifted: {relative}")
    sources.update(critical_sources)
    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"deployment sources are missing: {missing[:20]}")
    metadata = {
        "registration": {"path": str(registration), "sha256": sha256(registration)},
        "authorization": {"path": str(authorization), "sha256": sha256(authorization)},
        "freeze": {"path": str(freeze), "sha256": sha256(freeze)},
        "environment_design": {"path": str(design), "sha256": sha256(design)},
        "protocol": {"path": str(protocol), "sha256": sha256(protocol)},
        "count_correction": {"path": str(count_correction), "sha256": sha256(count_correction)},
        "source_freeze": {"path": str(source_freeze), "sha256": sha256(source_freeze)},
        "source_correction": {
            "path": str(source_correction),
            "sha256": sha256(source_correction),
        },
        "training_closeout": {"path": str(closeout), "sha256": sha256(closeout)},
        "training_verification": {"path": str(verification), "sha256": sha256(verification)},
        "training_audit": {"path": str(training_audit), "sha256": sha256(training_audit)},
        "training_file_manifest": {"path": str(training_manifest), "sha256": sha256(training_manifest)},
        "methods_manifest": {"path": str(methods), "sha256": sha256(methods), "methods": 42, "aliases": 60},
        "portability_correction": {
            "path": str(portability_path),
            "sha256": sha256(portability_path),
        },
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "sumo_runtime_lock": runtime_identity,
        "sumo_runtime_signature": runtime_payload["runtime_signature"],
        "runtime_requirements": {"path": str(runtime_requirements), "sha256": sha256(runtime_requirements)},
        "runtime_wheels": wheel_evidence,
        "frozen_v7_deployment": {"path": str(formal_deployment), "sha256": sha256(formal_deployment)},
        "v8_protocol": {"path": str(v8_protocol), "sha256": sha256(v8_protocol)},
        "v8_training_closeout": {"path": str(v8_closeout), "sha256": sha256(v8_closeout)},
        "v8_training_verification": {"path": str(v8_verification), "sha256": sha256(v8_verification)},
        "scenarios": scenarios,
    }
    return sources, metadata


def data_policy(entries: list[dict], dataset_hash: str) -> dict:
    hdf5 = [item for item in entries if Path(item["path"]).suffix.lower() in {".h5", ".hdf5"}]
    pt_files = [item for item in entries if Path(item["path"]).suffix.lower() == ".pt"]
    forbidden = [
        item["path"]
        for item in entries
        if Path(item["path"]).suffix.lower() in {".pth", ".ckpt", ".pkl"}
    ]
    if len(hdf5) != 1 or hdf5[0]["sha256"] != dataset_hash:
        raise ValueError("deployment must contain exactly the frozen HDF5 dataset")
    if len(pt_files) != 42 or forbidden:
        raise ValueError(f"deployment payload drifted: actors={len(pt_files)}, forbidden={forbidden}")
    return {
        "hdf5_files": 1,
        "compact_actor_files": 42,
        "training_checkpoint_or_optimizer_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": dataset_hash,
    }


def copy(source: Path, destination: Path, link_large_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if link_large_files and source.stat().st_size >= 100 * 1024**2:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace deployment {output}")
    sources, metadata = collect(args)
    output.mkdir(parents=True)
    try:
        for source in sorted(sources):
            copy(source, _source_destination(output, source), args.link_large_files)
        files = sorted(path for path in output.rglob("*") if path.is_file())
        entries = [descriptor(path, output) for path in files]
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "total_file_bytes": sum(item["size_bytes"] for item in entries),
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
        print(json.dumps({
            "output": str(output),
            "manifest_sha256": sha256(destination),
            "aggregate_sha256": manifest["aggregate_sha256"],
            "entry_count": manifest["entry_count"],
            "total_file_bytes": manifest["total_file_bytes"],
        }, indent=2, sort_keys=True))
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
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if manifest.get("schema_version") not in LEGACY_SCHEMAS | {SCHEMA} or aggregate(entries) != manifest.get("aggregate_sha256"):
            raise ValueError("deployment manifest identity drifted")
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != manifest_path}
        if actual != {item["path"] for item in entries}:
            raise ValueError("deployment file membership drifted")
        for item in entries:
            path = root / item["path"]
            if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"deployment file drifted: {item['path']}")
        data_policy(entries, manifest["dataset"]["sha256"])
        if (
            manifest.get("training") is not False
            or manifest.get("study_stage") != "development"
            or manifest.get("formal_held_out_outcomes_included") is not False
            or not isinstance(manifest.get("count_correction", {}).get("sha256"), str)
            or not isinstance(manifest.get("source_freeze", {}).get("sha256"), str)
            or not isinstance(manifest.get("source_correction", {}).get("sha256"), str)
            or not isinstance(manifest.get("portability_correction", {}).get("sha256"), str)
        ):
            raise ValueError("deployment scope drifted")
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
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    paths = (
        "registration", "authorization", "freeze", "environment_design", "protocol",
        "count_correction", "source_freeze",
        "training_closeout", "training_verification", "training_audit",
        "training_file_manifest", "methods_manifest", "dataset", "sumo_runtime_lock",
        "runtime_requirements", "wheel_dir", "formal_deployment_manifest", "sumo_dir",
        "edge_xml", "v8_protocol", "v8_training_closeout", "v8_training_verification",
        "output",
    )
    for name in paths:
        build_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    expected = (
        "registration", "authorization", "freeze", "design", "protocol",
        "count_correction", "source_freeze",
        "training_closeout", "training_verification", "training_audit",
        "training_file_manifest", "methods", "dataset", "runtime_lock",
        "runtime_requirements", "formal_deployment", "v8_protocol",
        "v8_training_closeout", "v8_training_verification",
    )
    for name in expected:
        build_parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    build_parser.add_argument("--link-large-files", action="store_true")
    build_parser.set_defaults(function=build)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.set_defaults(function=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
