#!/usr/bin/env python3
"""Independently verify a v11 actor-hold deployment without running SUMO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEPLOYMENT_SCHEMA = "transit-h2oplus-v11-actor-hold-deployment-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v11-actor-hold-protocol-v2"
PLAN_SCHEMA = "transit-h2oplus-v11-actor-hold-training-plan-v2"
ADMISSION_SCHEMA = "transit-training-admission-v11-actor-hold-development-v2"
RATIONALE_SCHEMA = "transit-h2oplus-v11-actor-hold-rationale-v1"
CORRECTION_SCHEMA = "transit-h2oplus-v11-actor-hold-correction-v2"
EXPECTED_VARIANTS = {"actor-hold-online-only"}
EXPECTED_SEEDS = {6201, 6202, 6203, 6204, 6205}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify(workspace: Path) -> dict[str, Any]:
    root = workspace.expanduser().resolve(strict=True)
    deployment_path = root / "deployment_manifest_v11.json"
    deployment = json.loads(deployment_path.read_text())
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("unexpected v11 deployment schema")
    entries = deployment.get("entries", [])
    if (
        len(entries) != 26
        or len(entries) != deployment.get("overlay_entry_count")
        or stable_object_sha256({"entries": entries})
        != deployment.get("overlay_aggregate_sha256")
    ):
        raise ValueError("v11 overlay manifest aggregate mismatch")
    paths = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in paths:
            raise ValueError("v11 overlay contains an unsafe or duplicate path")
        paths.add(entry["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"v11 overlay entry mismatch: {relative}")

    base = deployment["base_v7_training_deployment"]
    base_path = root / "deployment_manifest.json"
    if sha256(base_path) != base["sha256"]:
        raise ValueError("base v7 deployment manifest drifted")
    control = root / "reproducibility/h2oplus_v11_actor_hold_control"
    rationale_path = control / "design_rationale.json"
    correction_path = control / "boundary_correction.json"
    protocol_path = control / "protocol.json"
    plan_path = control / "training_plan.json"
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    rationale = json.loads(rationale_path.read_text())
    correction = json.loads(correction_path.read_text())
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or rationale.get("schema_version") != RATIONALE_SCHEMA
        or rationale.get("status")
        != "frozen-before-v11-training-or-new-development-outcomes"
        or rationale.get("formal_held_out_outcomes_read") is not False
        or rationale.get("local_training_or_sumo_executed") is not False
        or rationale.get("v10_artifact_integrity_passed") is not True
        or rationale.get("v10_scientific_stability_passed") is not False
        or correction.get("schema_version") != CORRECTION_SCHEMA
        or correction.get("status")
        != "corrected-before-any-v11-training-or-sumo-execution"
        or correction.get("v1_training_or_sumo_executed") is not False
        or correction.get("v1_training_submission_record_exists") is not False
        or correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("correction", {}).get(
            "atomic_event_overshoot_cannot_release_actor_early"
        )
        is not True
        or protocol.get("design_rationale", {}).get("sha256")
        != sha256(rationale_path)
        or plan.get("design_rationale", {}).get("sha256")
        != sha256(rationale_path)
        or deployment.get("design_rationale", {}).get("sha256")
        != sha256(rationale_path)
        or protocol.get("v2_boundary_correction", {}).get("sha256")
        != sha256(correction_path)
        or plan.get("v2_boundary_correction", {}).get("sha256")
        != sha256(correction_path)
        or deployment.get("v2_boundary_correction", {}).get("sha256")
        != sha256(correction_path)
    ):
        raise ValueError("v11 protocol or plan scope drifted")
    jobs = plan.get("jobs", [])
    if jobs != protocol.get("training_jobs") or len(jobs) != 5:
        raise ValueError("v11 protocol and plan job matrices disagree")
    if (
        {job["variant"] for job in jobs} != EXPECTED_VARIANTS
        or {job["seed"] for job in jobs} != EXPECTED_SEEDS
        or len({job["run_signature"] for job in jobs}) != 5
    ):
        raise ValueError("v11 job identity matrix is incomplete")

    source_paths = {
        "SimpleSAC/train_transit_v11_actor_hold.py": root
        / "SimpleSAC/train_transit_v11_actor_hold.py",
        "SimpleSAC/transit_v10/__init__.py": root
        / "SimpleSAC/transit_v10/__init__.py",
        "SimpleSAC/transit_v10/provenance.py": root
        / "SimpleSAC/transit_v10/provenance.py",
        "SimpleSAC/transit_v10/trainer.py": root
        / "SimpleSAC/transit_v10/trainer.py",
        "SimpleSAC/transit_v11/__init__.py": root
        / "SimpleSAC/transit_v11/__init__.py",
        "SimpleSAC/transit_v11/provenance.py": root
        / "SimpleSAC/transit_v11/provenance.py",
        "SimpleSAC/transit_v11/trainer.py": root
        / "SimpleSAC/transit_v11/trainer.py",
    }
    source = {name: sha256(path) for name, path in source_paths.items()}
    if (
        source != protocol.get("v11_source_sha256")
        or source != correction.get("corrected_runtime_source_sha256")
    ):
        raise ValueError("v11 deployed source map drifted")

    initializer_hashes = set()
    for job in jobs:
        admission_path = Path(job["training_admission"])
        admission = json.loads(admission_path.read_text())
        if (
            admission.get("schema_version") != ADMISSION_SCHEMA
            or admission.get("passed") is not True
            or admission.get("formal_held_out_outcomes_read") is not False
            or admission.get("run_signature") != job["run_signature"]
            or admission.get("variant") != job["variant"]
            or admission.get("stage") != "candidate"
            or admission.get("v11_source_sha256") != source
            or admission.get("v11_protocol", {}).get("sha256")
            != sha256(protocol_path)
            or admission.get("design_rationale", {}).get("sha256")
            != sha256(rationale_path)
            or admission.get("v2_boundary_correction", {}).get("sha256")
            != sha256(correction_path)
            or not job.get("name", "").startswith("h2oplus_v11a2_")
        ):
            raise ValueError(f"v11 admission mismatch: {admission_path}")
        initializer = job["offline_initializer"]
        initializer_path = Path(initializer["path"])
        if (
            not initializer_path.is_file()
            or sha256(initializer_path) != initializer["sha256"]
            or initializer_path.stat().st_size != initializer["bytes"]
        ):
            raise ValueError(f"v11 initializer mismatch: {initializer_path}")
        payload = torch.load(initializer_path, map_location="cpu", weights_only=False)
        if (
            payload.get("artifact_schema") != "transit-offline-initializer-v7-v1"
            or stable_object_sha256(payload.get("extra"))
            != initializer["extra_sha256"]
            or payload.get("extra", {}).get("source_run_signature")
            != initializer["source_run_signature"]
        ):
            raise ValueError(f"v11 initializer provenance mismatch: {initializer_path}")
        initializer_hashes.add(initializer["sha256"])
        arguments = job["command_arguments"]
        if (
            arguments.count("--offline-initializer") != 1
            or arguments.count("--training-admission") != 1
            or "--resume" in arguments
            or "--allow-no-resume" in arguments
            or "--branch-handoff-manifest" in arguments
            or job.get("stage") != "candidate"
            or job.get("warmup_events") != 5_000
            or job.get("epochs") != 40
            or job["online_event_cap"] != 25_000
            or job["expected_total_updates"] != 70_000
            or job.get("export_online_events") != [0, 5_000, 10_000, 25_000]
            or job.get("milestones")
            != [
                {
                    "online_events": 0,
                    "learner_update_steps": 50_000,
                    "ratio_optimizer_steps": 0,
                },
                {
                    "online_events": 5_000,
                    "learner_update_steps": 50_000,
                    "ratio_optimizer_steps": 0,
                },
                {
                    "online_events": 10_000,
                    "learner_update_steps": 55_000,
                    "ratio_optimizer_steps": 2_250,
                },
                {
                    "online_events": 25_000,
                    "learner_update_steps": 70_000,
                    "ratio_optimizer_steps": 6_000,
                },
            ]
        ):
            raise ValueError(f"v11 command contract mismatch: {job['name']}")
    if len(initializer_hashes) != 5:
        raise ValueError("v11 deployment does not contain five paired initializers")

    return {
        "schema_version": "transit-h2oplus-v11-actor-hold-deployment-verification-v2",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "verification_host": os.uname().nodename,
        "workspace": str(root),
        "deployment_manifest_sha256": sha256(deployment_path),
        "base_v7_deployment_manifest_sha256": sha256(base_path),
        "design_rationale_sha256": sha256(rationale_path),
        "v2_boundary_correction_sha256": sha256(correction_path),
        "overlay_entries": len(entries),
        "training_jobs": len(jobs),
        "variants": sorted(EXPECTED_VARIANTS),
        "training_seeds": sorted(EXPECTED_SEEDS),
        "offline_initializers": len(initializer_hashes),
        "development_evaluation_submitted": False,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
    }


def main() -> int:
    args = parse_args()
    payload = verify(args.workspace)
    if args.output:
        _write(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
