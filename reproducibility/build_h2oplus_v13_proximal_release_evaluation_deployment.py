#!/usr/bin/env python3
"""Build or verify the development-only v13 SUMO evaluation deployment."""

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

from bus_h2o.sumo_provenance import load_sumo_runtime_lock, load_validated_scenario, sha256
from reproducibility.build_evaluation_deployment_v7 import (
    _python_sources,
    _source_destination,
)
from reproducibility.build_h2oplus_v11_actor_hold_evaluation_deployment import (
    aggregate,
    bound,
    copy as copy_file,
    descriptor,
    resolve_relative,
)
from reproducibility.run_evaluation_matrix_v13_proximal_release import (
    METHODS_SCHEMA,
    evaluation_sources,
    load_development_methods,
)


SCHEMA = "transit-h2oplus-v13-proximal-release-evaluation-deployment-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v13-proximal-release-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
SOURCE_NAMES = {
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "SimpleSAC/evaluate_transit_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_h2oplus_v13_proximal_release_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v13_proximal_release.py",
}


def collect(args: argparse.Namespace) -> tuple[set[Path], dict]:
    schemas = {
        "registration": "transit-h2oplus-v13-proximal-release-registration-v2",
        "authorization": "transit-h2oplus-v13-proximal-release-authorization-v2",
        "freeze": "transit-h2oplus-v13-proximal-release-freeze-v2",
        "environment_design": "transit-h2oplus-v12-staged-release-environment-v2",
        "protocol": "transit-h2oplus-v13-proximal-release-protocol-v2",
        "correction": "transit-h2oplus-v13-pretraining-admission-correction-v2",
        "source_freeze": (
            "transit-h2oplus-v13-proximal-release-evaluation-source-freeze-v1"
        ),
        "training_closeout": (
            "transit-h2oplus-v13-proximal-release-branch-closeout-v2"
        ),
        "training_verification": (
            "transit-h2oplus-v13-proximal-release-branch-closeout-verification-v2"
        ),
        "training_audit": "transit-h2oplus-v13-proximal-release-branch-audit-v2",
        "training_file_manifest": (
            "transit-h2oplus-v13-proximal-release-branch-file-manifest-v2"
        ),
    }
    controls = {
        name: bound(getattr(args, name), getattr(args, f"expected_{name}_sha256"), schema)
        for name, schema in schemas.items()
    }
    methods = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods) != args.expected_methods_sha256:
        raise ValueError("v13 methods identity drifted")
    methods_payload = json.loads(methods.read_text())
    method_rows = load_development_methods(methods, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("dataset identity drifted")
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

    registration = controls["registration"][1]
    authorization = controls["authorization"][1]
    freeze = controls["freeze"][1]
    design = controls["environment_design"][1]
    protocol = controls["protocol"][1]
    correction = controls["correction"][1]
    source_freeze = controls["source_freeze"][1]
    closeout = controls["training_closeout"][1]
    verification = controls["training_verification"][1]
    audit = controls["training_audit"][1]
    file_manifest = controls["training_file_manifest"][1]
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("development_evaluation_submitted") is not False
        or registration.get("new_training_runs") != 15
        or registration.get("predesignated_candidate") != "proximal-medium"
        or authorization.get("formal_held_out_access_authorized") is not False
        or authorization.get("local_training_or_sumo_authorized") is not False
        or authorization.get(
            "remote_development_evaluation_after_training_closeout_authorized"
        )
        is not True
        or authorization.get("formal_confirmation_stage_authorized") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("development_evaluation_submitted") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("scientific_design_changed") is not False
        or design.get("formal_held_out_outcomes_read") is not False
        or design.get("disjointness", {}).get("verified_disjoint") is not True
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_15_branches") is not True
        or closeout.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or closeout.get("development_evaluation_gate_open") is not True
        or verification.get("passed") is not True
        or verification.get("all_15_scientific_stability_passed") is not True
        or verification.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or verification.get("development_evaluation_gate_open") is not True
        or audit.get("passed") is not True
        or audit.get("artifact_integrity_passed") is not True
        or audit.get("all_15_stability_passed") is not True
        or audit.get("all_seed_matched_v12_10k_handoffs_exact") is not True
        or file_manifest.get("file_count") != 255
        or source_freeze.get("evaluation_outcomes_read") is not False
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("physical_actor_count") != 65
        or source_freeze.get("logical_alias_count") != 160
        or source_freeze.get("expected_remote_sumo_episodes") != 585
        or source_freeze.get("registered_maximum_remote_sumo_episodes") != 1440
        or source_freeze.get("source_file_count") != 8
        or source_freeze.get("environment_design", {}).get("sha256")
        != sha256(controls["environment_design"][0])
        or methods_payload.get("physical_method_count") != 65
        or methods_payload.get("logical_method_budget_alias_count") != 160
        or methods_payload.get("expected_remote_sumo_episodes") != 585
        or methods_payload.get("registered_maximum_remote_sumo_episodes") != 1440
        or methods_payload.get("v13_protocol_sha256") != sha256(controls["protocol"][0])
        or len(method_rows) != 65
    ):
        raise ValueError("v13 deployment scope or sealed evidence drifted")
    if any(
        value
        for key, value in design.get("disjointness", {}).items()
        if key.endswith("_overlap")
    ):
        raise ValueError("development environment overlaps a frozen design")

    source_bindings = {
        item["path"]: item["sha256"] for item in source_freeze.get("sources", [])
    }
    if set(source_bindings) != SOURCE_NAMES:
        raise ValueError("v13 source freeze membership drifted")
    for relative, expected in source_bindings.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"v13 evaluation source drifted: {relative}")

    sources = _python_sources()
    sources.update(path for path, _ in controls.values())
    sources.update({methods, dataset, runtime_lock, runtime_requirements, formal_deployment})
    sources.update(Path(item["checkpoint"]).resolve(strict=True) for item in method_rows)

    binding_paths = {}
    for name, item in methods_payload.get("bindings", {}).items():
        path = resolve_relative(methods, item["path"])
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"methods binding drifted: {name}")
        sources.add(path)
        binding_paths[name] = {"path": str(path), "sha256": sha256(path)}
    required_bindings = {
        "v12_methods",
        "v13_protocol",
        "v13_training_plan",
        "v13_branch_closeout",
        "v13_branch_verification",
        "v13_branch_audit",
        "v13_branch_file_manifest",
    }
    if set(binding_paths) != required_bindings:
        raise ValueError("methods control bindings drifted")

    prefix_witnesses = 0
    branch_sources = 0
    handoff_manifests = set()
    for identity in methods_payload.get("identity_evidence", []):
        prefix = identity.get("prefix_source_policies", [])
        branches = identity.get("branch_source_policies", [])
        if len(prefix) != 3 or len(branches) != 3:
            raise ValueError("v13 identity witness inventory drifted")
        for witness in prefix:
            policy = resolve_relative(methods, witness["policy"]["path"])
            if (
                sha256(policy) != witness["policy"]["sha256"]
                or policy.stat().st_size != witness["policy"]["bytes"]
            ):
                raise ValueError(f"v13 prefix identity witness drifted: {policy}")
            sources.add(policy)
            prefix_witnesses += 1
        for witness in branches:
            policy = resolve_relative(methods, witness["policy"]["path"])
            handoff = resolve_relative(methods, witness["source_handoff_manifest"]["path"])
            for path, evidence in (
                (policy, witness["policy"]),
                (handoff, witness["source_handoff_manifest"]),
            ):
                if sha256(path) != evidence["sha256"] or path.stat().st_size != evidence["bytes"]:
                    raise ValueError(f"v13 branch identity witness drifted: {path}")
                sources.add(path)
            handoff_manifests.add(handoff)
            branch_sources += 1
    if prefix_witnesses != 15 or branch_sources != 15 or len(handoff_manifests) != 5:
        raise ValueError("v13 identity witness totals drifted")

    requirements_payload = json.loads(runtime_requirements.read_text())
    wheel_dir = args.wheel_dir.expanduser().resolve(strict=True)
    wheel_evidence = []
    for item in requirements_payload.get("wheels", []):
        wheel = (wheel_dir / item["filename"]).resolve(strict=True)
        if wheel.parent != wheel_dir or sha256(wheel) != item["sha256"]:
            raise ValueError(f"runtime wheel drifted: {wheel}")
        sources.add(wheel)
        wheel_evidence.append({"filename": wheel.name, "sha256": sha256(wheel)})

    blocks = design.get("blocks", [])
    if len(blocks) != 3 or design.get("environment_cells") != 9:
        raise ValueError("environment must contain three blocks and nine cells")
    scenarios = []
    for block in blocks:
        if len(block.get("scenario_manifests", [])) != 3:
            raise ValueError("development block does not contain three demand scales")
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
        raise ValueError("v13 scenario inventory drifted")

    critical_sources = set(
        evaluation_sources(
            args.sumo_dir.expanduser().resolve(strict=True),
            args.edge_xml.expanduser().resolve(strict=True),
        )
    )
    if len(critical_sources) != 45:
        raise ValueError(f"critical evaluation source count drifted: {len(critical_sources)}")
    formal_entries = {
        item["path"]: item["sha256"]
        for item in json.loads(formal_deployment.read_text()).get("entries", [])
        if item.get("type") == "file" and "sha256" in item
    }
    lineage_specific = SOURCE_NAMES - {
        "reproducibility/run_h2oplus_v13_proximal_release_evaluation_dispatch.py",
        "reproducibility/analyze_h2oplus_v13_proximal_release.py",
    }
    for source in critical_sources:
        resolved = source.resolve(strict=True)
        if ROOT in resolved.parents:
            relative = resolved.relative_to(ROOT).as_posix()
            expected = formal_entries.get(relative)
            if relative not in lineage_specific and expected is not None and sha256(resolved) != expected:
                raise ValueError(f"frozen v7 evaluation source drifted: {relative}")
    sources.update(critical_sources)
    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"deployment sources are missing: {missing[:20]}")

    metadata = {
        **{
            name: {"path": str(path), "sha256": sha256(path)}
            for name, (path, _) in controls.items()
        },
        "evaluation_source_sha256": source_bindings,
        "critical_evaluation_source_file_count": len(critical_sources),
        "methods_manifest": {
            "path": str(methods),
            "sha256": sha256(methods),
            "methods": 65,
            "aliases": 160,
        },
        "methods_control_bindings": binding_paths,
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
        "exact_seed_matched_v12_10k_handoffs_passed": True,
        "prefix_identity_witness_count": prefix_witnesses,
        "branch_source_policy_count": branch_sources,
        "source_handoff_manifest_count": len(handoff_manifests),
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
    if len(pt_files) != 80 or forbidden:
        raise ValueError(f"deployment payload drifted: actors={len(pt_files)}, forbidden={forbidden}")
    return {
        "hdf5_files": 1,
        "compact_actor_files": 80,
        "physical_actor_files": 65,
        "additional_prefix_identity_witness_files": 15,
        "training_checkpoint_or_optimizer_files": 0,
        "formal_held_out_outcome_files": 0,
        "canonical_dataset_sha256": dataset_hash,
    }


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace deployment {output}")
    sources, metadata = collect(args)
    output.mkdir(parents=True)
    try:
        for source in sorted(sources):
            copy_file(source, _source_destination(output, source), args.link_large_files)
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
            "physical_actor_count": 65,
            "logical_alias_count": 160,
            "expected_remote_sumo_episodes": 585,
            "registered_maximum_remote_sumo_episodes": 1440,
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
            or manifest.get("physical_actor_count") != 65
            or manifest.get("logical_alias_count") != 160
            or manifest.get("expected_remote_sumo_episodes") != 585
            or manifest.get("registered_maximum_remote_sumo_episodes") != 1440
            or manifest.get("critical_evaluation_source_file_count") != 45
            or manifest.get("prefix_identity_witness_count") != 15
            or manifest.get("branch_source_policy_count") != 15
            or manifest.get("source_handoff_manifest_count") != 5
            or manifest.get("training_scientific_stability_passed") is not True
            or manifest.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
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
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite deployment audit: {destination}")
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
