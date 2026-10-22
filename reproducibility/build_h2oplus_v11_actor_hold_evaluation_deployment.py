#!/usr/bin/env python3
"""Build or verify the development-only v11 actor-hold SUMO deployment."""

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
from reproducibility.run_evaluation_matrix_v11_actor_hold import (
    METHODS_SCHEMA,
    evaluation_sources,
    load_development_methods,
)


SCHEMA = "transit-h2oplus-v11-actor-hold-evaluation-deployment-v1"
AUDIT_SCHEMA = "transit-h2oplus-v11-actor-hold-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
SOURCE_NAMES = {
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_h2oplus_v11_actor_hold_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v11_actor_hold.py",
}


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


def collect(args: argparse.Namespace) -> tuple[set[Path], dict]:
    schemas = {
        "registration": "transit-h2oplus-v11-actor-hold-registration-v2",
        "authorization": "transit-h2oplus-v11-actor-hold-authorization-v2",
        "freeze": "transit-h2oplus-v11-actor-hold-freeze-v2",
        "environment_design": "transit-h2oplus-v11-actor-hold-environment-v2",
        "protocol": "transit-h2oplus-v11-actor-hold-protocol-v2",
        "correction": "transit-h2oplus-v11-actor-hold-correction-v2",
        "source_freeze": "transit-h2oplus-v11-actor-hold-evaluation-source-freeze-v1",
        "training_closeout": "transit-h2oplus-v11-actor-hold-training-closeout-v2",
        "training_verification": (
            "transit-h2oplus-v11-actor-hold-training-closeout-verification-v2"
        ),
        "training_audit": "transit-h2oplus-v11-actor-hold-training-audit-v2",
        "training_file_manifest": (
            "transit-h2oplus-v11-actor-hold-training-file-manifest-v2"
        ),
        "v8_protocol": "transit-h2oplus-v8-fewshot-protocol-v1",
        "v8_training_closeout": "transit-h2oplus-v8-fewshot-training-closeout-v1",
        "v8_training_verification": "transit-h2oplus-v8-fewshot-training-verification-v1",
        "v10_protocol": "transit-h2oplus-v10-delayed-protocol-v4",
        "v10_methods": "transit-evaluation-methods-v10-delayed-v1",
        "v10_evaluation_closeout": "transit-h2oplus-v10-delayed-evaluation-closeout-v1",
        "v10_evaluation_verification": (
            "transit-h2oplus-v10-delayed-evaluation-closeout-verification-v1"
        ),
    }
    controls = {}
    for name, schema in schemas.items():
        controls[name] = bound(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), schema
        )

    methods = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods) != args.expected_methods_sha256:
        raise ValueError("v11 methods identity drifted")
    methods_payload = json.loads(methods.read_text())
    method_rows = load_development_methods(methods, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("v11 dataset identity drifted")
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

    registration_payload = controls["registration"][1]
    authorization_payload = controls["authorization"][1]
    freeze_payload = controls["freeze"][1]
    design_payload = controls["environment_design"][1]
    protocol_payload = controls["protocol"][1]
    correction_payload = controls["correction"][1]
    source_payload = controls["source_freeze"][1]
    closeout_payload = controls["training_closeout"][1]
    verification_payload = controls["training_verification"][1]
    audit_payload = controls["training_audit"][1]
    file_manifest_payload = controls["training_file_manifest"][1]
    development = registration_payload.get("development_evaluation", {})
    if (
        registration_payload.get("formal_held_out_outcomes_read") is not False
        or development.get("logical_policy_aliases") != 80
        or development.get("maximum_unique_remote_sumo_episodes") != 720
        or authorization_payload.get("formal_held_out_access_authorized") is not False
        or authorization_payload.get("local_training_or_sumo_authorized") is not False
        or authorization_payload.get("remote_development_evaluation_after_training_authorized")
        is not True
        or freeze_payload.get("formal_held_out_outcomes_read") is not False
        or freeze_payload.get("development_evaluation_submitted") is not False
        or protocol_payload.get("formal_held_out_outcomes_read") is not False
        or correction_payload.get("formal_held_out_outcomes_read") is not False
        or design_payload.get("disjointness", {}).get("verified_disjoint") is not True
        or closeout_payload.get("artifact_integrity_passed") is not True
        or closeout_payload.get("all_5_training_runs_stability_gate_passed") is not True
        or closeout_payload.get("development_evaluation_release_gate_open") is not True
        or verification_payload.get("passed") is not True
        or verification_payload.get("artifact_integrity_verified") is not True
        or audit_payload.get("artifact_integrity", {}).get("passed") is not True
        or file_manifest_payload.get("file_count") != 100
        or source_payload.get("evaluation_outcomes_read") is not False
        or source_payload.get("formal_held_out_outcomes_read") is not False
        or source_payload.get("physical_actor_count") != 45
        or source_payload.get("logical_alias_count") != 80
        or methods_payload.get("physical_method_count") != 45
        or methods_payload.get("logical_method_budget_alias_count") != 80
        or methods_payload.get("expected_remote_sumo_episodes") != 405
        or controls["v8_protocol"][1].get("formal_held_out_outcomes_read") is not False
        or controls["v8_training_closeout"][1].get("artifact_integrity_passed") is not True
        or controls["v8_training_verification"][1].get("passed") is not True
        or controls["v10_methods"][1].get("physical_method_count") != 60
        or controls["v10_evaluation_closeout"][1].get("artifact_integrity", {}).get("passed")
        is not True
        or controls["v10_evaluation_verification"][1].get("passed") is not True
        or len(method_rows) != 45
    ):
        raise ValueError("v11 deployment scope or sealed evidence drifted")
    if any(
        value
        for key, value in design_payload.get("disjointness", {}).items()
        if key.endswith("_overlap")
    ):
        raise ValueError("v11 development environment overlaps a frozen design")

    source_bindings = {
        item["path"]: item["sha256"] for item in source_payload.get("sources", [])
    }
    if set(source_bindings) != SOURCE_NAMES:
        raise ValueError("v11 source freeze membership drifted")
    for relative, expected in source_bindings.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"v11 evaluation source drifted: {relative}")

    sources = _python_sources()
    sources.update(path for path, _ in controls.values())
    sources.update(
        {
            methods,
            dataset,
            runtime_lock,
            runtime_requirements,
            formal_deployment,
        }
    )
    sources.update(Path(item["checkpoint"]).resolve(strict=True) for item in method_rows)

    witness_count = 0
    for identity in methods_payload.get("identity_evidence", []):
        for witness in identity.get("v11_source_policies", []):
            policy = resolve_relative(methods, witness["policy"]["path"])
            if (
                sha256(policy) != witness["policy"]["sha256"]
                or policy.stat().st_size != witness["policy"]["bytes"]
            ):
                raise ValueError(f"v11 hold witness drifted: {policy}")
            sources.add(policy)
            witness_count += 1
    if witness_count != 20:
        raise ValueError(f"v11 hold witness inventory drifted: {witness_count}")

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
    if len(blocks) != 3 or design_payload.get("environment_cells") != 9:
        raise ValueError("v11 environment must contain three blocks and nine cells")
    scenarios = []
    for block in blocks:
        if len(block.get("scenario_manifests", [])) != 3:
            raise ValueError("v11 block does not contain three demand scales")
        for raw in block["scenario_manifests"]:
            manifest = resolve_relative(controls["environment_design"][0], raw)
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
    if len(scenarios) != 9:
        raise ValueError("v11 scenario inventory drifted")

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
    v11_specific = {
        "SimpleSAC/evaluate_transit_v11_actor_hold.py",
        "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    }
    for source in critical_sources:
        resolved = source.resolve(strict=True)
        if ROOT in resolved.parents:
            relative = resolved.relative_to(ROOT).as_posix()
            expected = formal_entries.get(relative)
            if relative not in v11_specific and expected is not None and sha256(resolved) != expected:
                raise ValueError(f"frozen v7 evaluation source drifted: {relative}")
    sources.update(critical_sources)
    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"deployment sources are missing: {missing[:20]}")

    control_metadata = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, (path, _) in controls.items()
    }
    metadata = {
        **control_metadata,
        "evaluation_source_sha256": source_bindings,
        "methods_manifest": {
            "path": str(methods),
            "sha256": sha256(methods),
            "methods": 45,
            "aliases": 80,
        },
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "sumo_runtime_lock": runtime_identity,
        "sumo_runtime_signature": runtime_payload["runtime_signature"],
        "runtime_requirements": {
            "path": str(runtime_requirements),
            "sha256": sha256(runtime_requirements),
        },
        "runtime_wheels": wheel_evidence,
        "frozen_v7_deployment": {
            "path": str(formal_deployment),
            "sha256": sha256(formal_deployment),
        },
        "scenarios": scenarios,
        "training_scientific_stability_passed": True,
        "stability_failure_runs": [],
        "hold_source_policy_witness_count": witness_count,
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
    if len(pt_files) != 60 or forbidden:
        raise ValueError(f"deployment payload drifted: actors={len(pt_files)}, forbidden={forbidden}")
    return {
        "hdf5_files": 1,
        "compact_actor_files": 60,
        "physical_actor_files": 45,
        "additional_hold_identity_witness_files": 15,
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
            "expected_physical_actor_count": 45,
            "expected_logical_alias_count": 80,
            "expected_remote_sumo_episodes": 405,
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
    errors, manifest = [], {}
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("entries", [])
        if manifest.get("schema_version") != SCHEMA or aggregate(entries) != manifest.get(
            "aggregate_sha256"
        ):
            raise ValueError("deployment manifest identity drifted")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
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
            or manifest.get("expected_physical_actor_count") != 45
            or manifest.get("expected_logical_alias_count") != 80
            or manifest.get("expected_remote_sumo_episodes") != 405
            or manifest.get("hold_source_policy_witness_count") != 20
            or set(manifest.get("evaluation_source_sha256", {})) != SOURCE_NAMES
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
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
    }
    if args.output is not None:
        destination = args.output.expanduser().resolve()
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
    paths = (
        "registration",
        "authorization",
        "freeze",
        "environment_design",
        "protocol",
        "correction",
        "source_freeze",
        "training_closeout",
        "training_verification",
        "training_audit",
        "training_file_manifest",
        "methods_manifest",
        "dataset",
        "sumo_runtime_lock",
        "runtime_requirements",
        "wheel_dir",
        "formal_deployment_manifest",
        "sumo_dir",
        "edge_xml",
        "v8_protocol",
        "v8_training_closeout",
        "v8_training_verification",
        "v10_protocol",
        "v10_methods",
        "v10_evaluation_closeout",
        "v10_evaluation_verification",
        "output",
    )
    for name in paths:
        build_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    expected = (
        "registration",
        "authorization",
        "freeze",
        "environment_design",
        "protocol",
        "correction",
        "source_freeze",
        "training_closeout",
        "training_verification",
        "training_audit",
        "training_file_manifest",
        "methods",
        "dataset",
        "runtime_lock",
        "runtime_requirements",
        "formal_deployment",
        "v8_protocol",
        "v8_training_closeout",
        "v8_training_verification",
        "v10_protocol",
        "v10_methods",
        "v10_evaluation_closeout",
        "v10_evaluation_verification",
    )
    for name in expected:
        build_parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    build_parser.add_argument("--link-large-files", action="store_true")
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
