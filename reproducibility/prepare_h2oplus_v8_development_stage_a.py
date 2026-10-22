#!/usr/bin/env python3
"""Freeze disjoint v8 development scenarios and the Stage A policy screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SUMO_ROOT = PROJECT_ROOT / "SUMO_ruiguang"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import load_validated_scenario, sha256
from reproducibility.build_passenger_scenarios import build_scenario
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    V7_POLICY_SELECTION_RULE,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v8-development-pilot-registration-v1"
AUTHORIZATION_SCHEMA = (
    "transit-h2oplus-v8-development-execution-authorization-v1"
)
DESIGN_SCHEMA = "transit-h2oplus-v8-development-environment-v1"
EXPORT_SCHEMA = "transit-h2oplus-v8-stage-a-policy-export-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-stage-a-freeze-v1"
PORTABLE_CONFIG_NAME = "scenario_portable_v8.sumocfg"
PORTABLE_MANIFEST_NAME = "manifest_portable_v8.json"
METHODS_NAME = "evaluation_methods_v8_stage_a.json"
EXPORT_MANIFEST_NAME = "policy_export_manifest_v8_stage_a.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "h2oplus_v8_development_pilot_registration_v1.json",
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "h2oplus_v8_development_stage_a_execution_authorization_v1.json",
    )
    parser.add_argument(
        "--training-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "training_environment_design_v7_formal_v1.json",
    )
    parser.add_argument(
        "--held-out-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "evaluation_environment_design_v7_formal_v2.json",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=ROOT / "bus_h2o" / "development_scenarios_v8" / "stage_a_v1",
    )
    parser.add_argument(
        "--environment-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "h2oplus_v8_development_environment_stage_a_v1.json",
    )
    parser.add_argument(
        "--policy-export",
        type=Path,
        default=ROOT / "reproducibility" / "h2oplus_v8_stage_a_policy_export_v1",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "reproducibility" / "h2oplus_v8_stage_a_freeze_v1.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    return resolved, json.loads(resolved.read_text())


def _descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            os.path.relpath(resolved, base.resolve()) if base is not None else str(resolved)
        ),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _stable_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        previous = json.loads(path.read_text())
        previous_created = previous.pop("created_utc", None)
        comparable = dict(payload)
        comparable.pop("created_utc", None)
        if previous == comparable and isinstance(previous_created, str):
            created_utc = previous_created
        elif not force:
            raise FileExistsError(f"refusing to replace a different file: {path}")
    payload = {**payload, "created_utc": created_utc}
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verify_authorization(
    registration_path: Path,
    registration: dict[str, Any],
    authorization_path: Path,
    authorization: dict[str, Any],
) -> None:
    if registration.get("schema_version") != REGISTRATION_SCHEMA:
        raise ValueError("unexpected v8 registration schema")
    if registration.get("status") != "registered-not-submitted":
        raise ValueError("v8 registration status drifted")
    if registration.get("execution_authorized") is not False:
        raise ValueError("immutable registration was altered post hoc")
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unexpected execution authorization schema")
    if authorization.get("authorization_decision") != "authorize-stage-a-execution":
        raise ValueError("Stage A execution is not authorized")
    if authorization.get("approved_scope") != "A-existing-policy-development-screen":
        raise ValueError("authorization scope does not match Stage A")
    if authorization.get("formal_held_out_access_authorized") is not False:
        raise ValueError("authorization must not unlock formal held-out outcomes")
    if authorization.get("stage_b_training_authorized_before_stage_a_gate") is not False:
        raise ValueError("authorization must keep Stage B behind the Stage A gate")
    registration_ref = authorization.get("registration", {})
    if registration_ref.get("sha256") != sha256(registration_path):
        raise ValueError("authorization does not bind the current registration")
    referenced = (authorization_path.parent / registration_ref.get("path", "")).resolve()
    if referenced != registration_path:
        raise ValueError("authorization registration path mismatch")


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _logical_source_name(path: Path) -> str:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        return "H2Oplus/" + resolved.relative_to(ROOT).as_posix()
    if resolved == SUMO_ROOT or SUMO_ROOT in resolved.parents:
        return "SUMO_ruiguang/" + resolved.relative_to(SUMO_ROOT).as_posix()
    return "external/" + resolved.name


def _portable_config(source: Path, destination: Path, *, force: bool) -> None:
    tree = ET.parse(source)
    input_block = tree.getroot().find("input")
    if input_block is None:
        raise ValueError(f"SUMO configuration has no input block: {source}")
    for element in input_block:
        values = []
        for raw in element.get("value", "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = source.parent / path
            resolved = path.resolve(strict=True)
            values.append(Path(os.path.relpath(resolved, destination.parent)).as_posix())
        element.set("value", ",".join(values))
    temporary = destination.with_suffix(destination.suffix + ".partial")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    if destination.is_file() and destination.read_bytes() == temporary.read_bytes():
        temporary.unlink()
        return
    if destination.exists() and not force:
        temporary.unlink()
        raise FileExistsError(f"refusing to replace {destination}")
    os.replace(temporary, destination)


def _portable_scenario(manifest_path: Path, *, force: bool) -> tuple[Path, dict]:
    original_path, original = _load(manifest_path)
    load_validated_scenario(original_path)
    scenario_dir = original_path.parent
    source_config = _resolve(scenario_dir, original["sumo_config"])
    portable_config = scenario_dir / PORTABLE_CONFIG_NAME
    _portable_config(source_config, portable_config, force=force)
    passenger_objects = _resolve(scenario_dir, original["passenger_objects"])
    passenger_routes = [
        _resolve(scenario_dir, value) for value in original["passenger_routes"]
    ]
    declared = [portable_config, passenger_objects, *passenger_routes]
    source_content = {
        _logical_source_name(Path(raw)): digest
        for raw, digest in original.get("source_sha256", {}).items()
    }
    payload = {
        **{
            key: value
            for key, value in original.items()
            if key
            not in {
                "generator_sha256",
                "passenger_objects",
                "passenger_routes",
                "scenario_sha256",
                "source_sha256",
                "sumo_config",
            }
        },
        "sumo_config": portable_config.name,
        "passenger_objects": passenger_objects.name,
        "passenger_routes": [path.name for path in passenger_routes],
        "scenario_sha256": {
            path.relative_to(scenario_dir).as_posix(): sha256(path)
            for path in declared
        },
        "source_content_sha256": source_content,
        "demand_generator_sha256": original.get("generator_sha256"),
        "portable_manifest_generator": Path(
            os.path.relpath(Path(__file__).resolve(), scenario_dir)
        ).as_posix(),
        "portable_manifest_generator_sha256": sha256(Path(__file__).resolve()),
        "path_semantics": (
            "all runtime paths are relative to this manifest/config; content "
            "identity is independent of host mount point"
        ),
    }
    destination = scenario_dir / PORTABLE_MANIFEST_NAME
    _stable_json(destination, payload, force=force)
    _, evidence = load_validated_scenario(destination)
    return destination, evidence


def _runtime_key(evidence: dict[str, Any]) -> str:
    encoded = json.dumps(
        sorted(evidence["runtime_input_sha256"].values()), separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _design_inventory(path: Path) -> dict[str, Any]:
    design_path, payload = _load(path)
    demand_seeds: set[int] = set()
    sumo_seeds: set[int] = set()
    manifest_hashes: set[str] = set()
    runtime_hashes: set[str] = set()
    for block in payload.get("blocks", []):
        demand_seeds.add(int(block["demand_realization_seed"]))
        sumo_seeds.add(int(block["sumo_seed"]))
        for raw in block["scenario_manifests"]:
            manifest = _resolve(design_path.parent, raw)
            _, evidence = load_validated_scenario(manifest)
            manifest_hashes.add(evidence["manifest_sha256"])
            runtime_hashes.add(_runtime_key(evidence))
    return {
        "path": design_path,
        "payload": payload,
        "demand_seeds": demand_seeds,
        "sumo_seeds": sumo_seeds,
        "manifest_hashes": manifest_hashes,
        "runtime_hashes": runtime_hashes,
    }


def _build_environment(
    args: argparse.Namespace,
    registration_path: Path,
    registration: dict[str, Any],
    authorization_path: Path,
) -> Path:
    specification = registration["development_environment"]
    scales = [float(value) for value in specification["demand_scales"]]
    jitter = float(specification["depart_jitter_seconds"])
    scenario_root = args.scenario_root.expanduser().resolve()
    base_config = SUMO_ROOT / "online_control" / "control_sim_traci_period.sumocfg"
    direct_route = (
        SUMO_ROOT
        / "e_passenger_rou"
        / "3_modified_period_workday_passenger_VC_0805.rou.xml"
    )
    transfer_route = (
        SUMO_ROOT
        / "e_passenger_rou"
        / "3_modified_transfer_passenger_0805.rou.xml"
    )
    passenger_objects = (
        SUMO_ROOT
        / "online_control"
        / "initialize_obj"
        / "save_obj_passenger.add.xml"
    )
    blocks = []
    descriptors = []
    development_manifest_hashes: set[str] = set()
    development_runtime_hashes: set[str] = set()
    for block in specification["blocks"]:
        paths = []
        for scale in scales:
            original = build_scenario(
                base_config=base_config.resolve(strict=True),
                passenger_routes=(
                    direct_route.resolve(strict=True),
                    transfer_route.resolve(strict=True),
                ),
                passenger_objects=passenger_objects.resolve(strict=True),
                output_root=scenario_root,
                scale=scale,
                seed=int(block["demand_realization_seed"]),
                depart_jitter_seconds=jitter,
            )
            portable, evidence = _portable_scenario(original, force=args.force)
            manifest_hash = evidence["manifest_sha256"]
            runtime_hash = _runtime_key(evidence)
            if (
                manifest_hash in development_manifest_hashes
                or runtime_hash in development_runtime_hashes
            ):
                raise ValueError("development scenario content is duplicated")
            development_manifest_hashes.add(manifest_hash)
            development_runtime_hashes.add(runtime_hash)
            relative = os.path.relpath(portable, args.environment_design.parent.resolve())
            paths.append(relative)
            descriptors.append(
                {
                    "block": block["name"],
                    "demand_scale": float(evidence["demand_scale"]),
                    "manifest": relative,
                    "manifest_sha256": manifest_hash,
                    "runtime_content_sha256": runtime_hash,
                }
            )
        blocks.append(
            {
                "name": block["name"],
                "demand_realization_seed": int(block["demand_realization_seed"]),
                "sumo_seed": int(block["sumo_seed"]),
                "scenario_manifests": paths,
            }
        )

    training = _design_inventory(args.training_design)
    held_out = _design_inventory(args.held_out_design)
    development_demand = {int(block["demand_realization_seed"]) for block in blocks}
    development_sumo = {int(block["sumo_seed"]) for block in blocks}
    overlaps = {
        "training_demand_seed_overlap": sorted(
            development_demand & training["demand_seeds"]
        ),
        "training_sumo_seed_overlap": sorted(development_sumo & training["sumo_seeds"]),
        "held_out_demand_seed_overlap": sorted(
            development_demand & held_out["demand_seeds"]
        ),
        "held_out_sumo_seed_overlap": sorted(development_sumo & held_out["sumo_seeds"]),
        "training_manifest_hash_overlap": sorted(
            development_manifest_hashes & training["manifest_hashes"]
        ),
        "training_runtime_hash_overlap": sorted(
            development_runtime_hashes & training["runtime_hashes"]
        ),
        "held_out_manifest_hash_overlap": sorted(
            development_manifest_hashes & held_out["manifest_hashes"]
        ),
        "held_out_runtime_hash_overlap": sorted(
            development_runtime_hashes & held_out["runtime_hashes"]
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"development disjointness failed: {overlaps}")
    if len(blocks) != 5 or len(descriptors) != specification["environment_cells"]:
        raise ValueError("development environment coverage drifted")
    payload = {
        "schema_version": DESIGN_SCHEMA,
        "mdp_version": "v7",
        "study_stage": "development",
        "role": "candidate screen; not formal held-out evaluation",
        "status": "frozen-before-stage-A-outcomes",
        "design": (
            "five independent paired blocks; three registered demand scales per "
            "block; no formal held-out outcome access"
        ),
        "independent_realization_blocks": len(blocks),
        "environment_cells": len(descriptors),
        "demand_scales": scales,
        "depart_jitter_seconds": jitter,
        "blocks": blocks,
        "scenario_descriptors": descriptors,
        "disjointness": {
            "verified_disjoint": True,
            **overlaps,
            "training_design": _descriptor(training["path"]),
            "formal_held_out_design": _descriptor(held_out["path"]),
            "formal_held_out_outcomes_read": False,
            "runtime_identity": "SHA-256 over sorted runtime input content hashes",
        },
        "registration": _descriptor(registration_path),
        "execution_authorization": _descriptor(authorization_path),
        "generator": _descriptor(Path(__file__).resolve()),
        "scenario_generator": _descriptor(
            ROOT / "reproducibility" / "build_passenger_scenarios.py"
        ),
        "path_semantics": "relative to the file declaring each path",
    }
    output = args.environment_design.expanduser().resolve()
    _stable_json(output, payload, force=args.force)
    return output


def _manifest_entry(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    entries = manifest.get("merged_results", {}).get("files_manifest", [])
    matches = [entry for entry in entries if entry.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"file-level manifest lacks unique entry {relative_path}")
    return matches[0]


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"different frozen export already exists: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"policy export hash mismatch: {destination}")


def _build_policy_export(
    args: argparse.Namespace,
    registration_path: Path,
    registration: dict[str, Any],
    authorization_path: Path,
) -> tuple[Path, Path]:
    destination = args.policy_export.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    roots = {
        "h2oplus": (
            ROOT / "reproducibility" / "formal_results_v7_v7r1" / "h2oplus_core",
            ROOT / "reproducibility" / "formal_results_v7_v7r1" / "manifest.json",
            ROOT
            / "reproducibility"
            / "training_plans_v7_formal_v2_v7r1"
            / "training_plan_v7_formal_core_server_h2oplus_v7.json",
        ),
        "h2oplus-lambda0p5": (
            ROOT
            / "reproducibility"
            / "formal_results_v7_v7r1"
            / "h2oplus_sensitivity",
            ROOT
            / "reproducibility"
            / "formal_results_v7_v7r1"
            / "sensitivity_manifest.json",
            ROOT
            / "reproducibility"
            / "training_plans_v7_formal_v2_v7r1"
            / "training_plan_v7_formal_sensitivity_server_h2oplus_v7.json",
        ),
    }
    methods = []
    files = []
    for algorithm in registration["stages"][0]["policies"]:
        source_root, manifest_path, plan_path = roots[algorithm]
        merged_manifest = json.loads(manifest_path.read_text())
        plan_hash = sha256(plan_path)
        for seed in registration["stages"][0]["training_seeds"]:
            run_name = f"{algorithm}_seed{seed}"
            run_dir = source_root / run_name
            summary_path = run_dir / "summary.json"
            metadata_path = run_dir / "metadata.json"
            actor_path = run_dir / "checkpoints" / "final_actor.pt"
            summary = json.loads(summary_path.read_text())
            metadata = json.loads(metadata_path.read_text())
            actor_hash = sha256(actor_path)
            relative_actor = f"{run_name}/checkpoints/final_actor.pt"
            entry = _manifest_entry(merged_manifest, relative_actor)
            if entry.get("sha256") != actor_hash or entry.get("bytes") != actor_path.stat().st_size:
                raise ValueError(f"file-level manifest mismatch for {relative_actor}")
            if summary.get("final_policy", {}).get("sha256") != actor_hash:
                raise ValueError(f"summary policy hash mismatch for {run_name}")
            if summary.get("completed_epochs") != 190 or summary.get("total_updates") != 145000:
                raise ValueError(f"incomplete Stage A source policy: {run_name}")
            if metadata.get("run_signature") != summary.get("run_signature"):
                raise ValueError(f"run signature mismatch for {run_name}")
            admission_hash = metadata.get("training_admission", {}).get("sha256")
            if not isinstance(admission_hash, str) or len(admission_hash) != 64:
                raise ValueError(f"missing training admission for {run_name}")
            exported_actor = (
                destination / "policies" / algorithm / f"seed{seed}" / "final_actor.pt"
            )
            exported_metadata = (
                destination / "provenance" / algorithm / f"seed{seed}" / "metadata.json"
            )
            exported_summary = (
                destination / "provenance" / algorithm / f"seed{seed}" / "summary.json"
            )
            _copy_exact(actor_path, exported_actor)
            _copy_exact(metadata_path, exported_metadata)
            _copy_exact(summary_path, exported_summary)
            methods.append(
                {
                    "name": f"{algorithm}-seed{seed}",
                    "algorithm": algorithm,
                    "training_seed": int(seed),
                    "type": "checkpoint",
                    "checkpoint": os.path.relpath(exported_actor, destination),
                    "training_run_signature": summary["run_signature"],
                    "training_admission_sha256": admission_hash,
                    "training_checkpoint_sha256": summary["checkpoint"]["sha256"],
                    "training_plan_sha256": plan_hash,
                }
            )
            files.extend(
                _descriptor(path, destination)
                for path in (exported_actor, exported_metadata, exported_summary)
            )
    methods.sort(key=lambda item: (item["algorithm"], item["training_seed"]))
    if len(methods) != 10:
        raise ValueError("Stage A must export exactly ten policies")
    methods_payload = {
        "schema_version": METHODS_SCHEMAS["v7"],
        "study_stage": "formal",
        "selection_rule": V7_POLICY_SELECTION_RULE,
        "scope": (
            "preselected completed v7 actors evaluated only on the disjoint v8 "
            "development screen"
        ),
        "development_registration_sha256": sha256(registration_path),
        "methods": methods,
    }
    methods_path = destination / METHODS_NAME
    _stable_json(methods_path, methods_payload, force=args.force)
    export_payload = {
        "schema_version": EXPORT_SCHEMA,
        "status": "frozen-before-stage-A-outcomes",
        "selection_rule": V7_POLICY_SELECTION_RULE,
        "registration": _descriptor(registration_path),
        "execution_authorization": _descriptor(authorization_path),
        "methods_manifest": _descriptor(methods_path, destination),
        "methods": methods,
        "source_file_level_manifests": [
            _descriptor(roots[name][1]) for name in sorted(roots)
        ],
        "payload_policy": {
            "compact_final_actor_files": 10,
            "metadata_files": 10,
            "summary_files": 10,
            "training_checkpoint_files": 0,
            "optimizer_or_replay_files": 0,
        },
        "files": sorted(files, key=lambda item: item["path"]),
        "generator": _descriptor(Path(__file__).resolve()),
    }
    export_path = destination / EXPORT_MANIFEST_NAME
    _stable_json(export_path, export_payload, force=args.force)
    return methods_path, export_path


def _build_freeze(
    args: argparse.Namespace,
    registration_path: Path,
    registration: dict[str, Any],
    authorization_path: Path,
    design_path: Path,
    methods_path: Path,
    export_path: Path,
) -> Path:
    dataset = (
        ROOT
        / "bus_h2o"
        / "datasets_v7"
        / "formal_v2_full"
        / "transit_smdp_v7_formal_v2.h5"
    )
    methods = json.loads(methods_path.read_text())["methods"]
    design = json.loads(design_path.read_text())
    stage = registration["stages"][0]
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen-before-stage-A-outcomes",
        "stage": stage["name"],
        "training": False,
        "formal_held_out_outcomes_read": False,
        "registration": _descriptor(registration_path),
        "execution_authorization": _descriptor(authorization_path),
        "environment_design": _descriptor(design_path),
        "methods_manifest": _descriptor(methods_path),
        "policy_export_manifest": _descriptor(export_path),
        "dataset": _descriptor(dataset),
        "matrix": {
            "methods": len(methods),
            "environment_cells": design["environment_cells"],
            "jobs": len(methods) * design["environment_cells"],
            "shard_count": 2,
            "workers_per_host": 12,
            "assignment": "sorted-portable-job-key-round-robin-v1 within each block",
        },
        "endpoint": {
            "name": stage["primary_endpoint"],
            "direction": stage["primary_direction"],
            "paired_unit": stage["paired_unit"],
        },
        "advance_gate": stage["advance_gate"],
        "stage_b_policy": "blocked unless every registered Stage A gate passes",
        "generator": _descriptor(Path(__file__).resolve()),
    }
    output = args.freeze.expanduser().resolve()
    _stable_json(output, payload, force=args.force)
    return output


def main() -> None:
    args = parse_args()
    registration_path, registration = _load(args.registration)
    authorization_path, authorization = _load(args.authorization)
    _verify_authorization(
        registration_path, registration, authorization_path, authorization
    )
    design_path = _build_environment(
        args, registration_path, registration, authorization_path
    )
    methods_path, export_path = _build_policy_export(
        args, registration_path, registration, authorization_path
    )
    freeze_path = _build_freeze(
        args,
        registration_path,
        registration,
        authorization_path,
        design_path,
        methods_path,
        export_path,
    )
    print(
        json.dumps(
            {
                "environment_design": _descriptor(design_path),
                "methods_manifest": _descriptor(methods_path),
                "policy_export_manifest": _descriptor(export_path),
                "freeze": _descriptor(freeze_path),
                "jobs": 150,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
