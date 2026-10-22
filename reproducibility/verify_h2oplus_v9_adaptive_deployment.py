#!/usr/bin/env python3
"""Independently verify a v9 adaptation deployment without running SUMO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEPLOYMENT_SCHEMA = "transit-h2oplus-v9-adaptive-deployment-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v9-adaptive-protocol-v2"
PLAN_SCHEMA = "transit-h2oplus-v9-adaptive-training-plan-v2"
ADMISSION_SCHEMA = "transit-training-admission-v9-adaptive-development-v2"
EXPECTED_VARIANTS = {"ratio-gated", "adaptation", "combined"}
EXPECTED_SEEDS = {6201, 6202, 6203}


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
    deployment_path = root / "deployment_manifest_v9.json"
    deployment = json.loads(deployment_path.read_text())
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("unexpected v9 deployment schema")
    entries = deployment.get("entries", [])
    if (
        len(entries) != deployment.get("overlay_entry_count")
        or stable_object_sha256({"entries": entries})
        != deployment.get("overlay_aggregate_sha256")
    ):
        raise ValueError("v9 overlay manifest aggregate mismatch")
    paths = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in paths:
            raise ValueError("v9 overlay contains an unsafe or duplicate path")
        paths.add(entry["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"v9 overlay entry mismatch: {relative}")

    base = deployment["base_v7_training_deployment"]
    base_path = root / "deployment_manifest.json"
    if sha256(base_path) != base["sha256"]:
        raise ValueError("base v7 deployment manifest drifted")
    control = root / "reproducibility/h2oplus_v9_control"
    correction_path = control / "correction.json"
    protocol_path = control / "protocol.json"
    plan_path = control / "training_plan.json"
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    correction = json.loads(correction_path.read_text())
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or correction.get("status")
        != "v1-invalid-sealed-v2-correction-authorized"
        or correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("local_training_or_sumo_executed") is not False
        or correction.get("invalid_v1_lineage", {}).get(
            "eligible_as_scientific_evidence"
        )
        is not False
        or protocol.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or plan.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or deployment.get("correction", {}).get("sha256")
        != sha256(correction_path)
    ):
        raise ValueError("v9 protocol or plan scope drifted")
    jobs = plan.get("jobs", [])
    if jobs != protocol.get("training_jobs") or len(jobs) != 9:
        raise ValueError("v9 protocol and plan job matrices disagree")
    if (
        {job["variant"] for job in jobs} != EXPECTED_VARIANTS
        or {job["seed"] for job in jobs} != EXPECTED_SEEDS
        or len({job["run_signature"] for job in jobs}) != 9
    ):
        raise ValueError("v9 job identity matrix is incomplete")

    source_paths = {
        "SimpleSAC/train_transit_v9_adaptive.py": root
        / "SimpleSAC/train_transit_v9_adaptive.py",
        "SimpleSAC/transit_v9/__init__.py": root
        / "SimpleSAC/transit_v9/__init__.py",
        "SimpleSAC/transit_v9/trainer.py": root
        / "SimpleSAC/transit_v9/trainer.py",
    }
    source = {name: sha256(path) for name, path in source_paths.items()}
    if source != protocol.get("v9_source_sha256"):
        raise ValueError("v9 deployed source map drifted")

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
            or admission.get("v9_source_sha256") != source
            or admission.get("v9_protocol", {}).get("sha256")
            != sha256(protocol_path)
            or admission.get("correction", {}).get("sha256")
            != sha256(correction_path)
        ):
            raise ValueError(f"v9 admission mismatch: {admission_path}")
        initializer = job["offline_initializer"]
        initializer_path = Path(initializer["path"])
        if (
            not initializer_path.is_file()
            or sha256(initializer_path) != initializer["sha256"]
            or initializer_path.stat().st_size != initializer["bytes"]
        ):
            raise ValueError(f"v9 initializer mismatch: {initializer_path}")
        payload = torch.load(initializer_path, map_location="cpu", weights_only=False)
        if (
            payload.get("artifact_schema") != "transit-offline-initializer-v7-v1"
            or stable_object_sha256(payload.get("extra"))
            != initializer["extra_sha256"]
            or payload.get("extra", {}).get("source_run_signature")
            != initializer["source_run_signature"]
        ):
            raise ValueError(f"v9 initializer provenance mismatch: {initializer_path}")
        initializer_hashes.add(initializer["sha256"])
        arguments = job["command_arguments"]
        if (
            arguments.count("--offline-initializer") != 1
            or arguments.count("--training-admission") != 1
            or "--resume" in arguments
            or job["online_event_cap"] != 25_000
            or job["expected_total_updates"] not in {70_000, 74_000}
        ):
            raise ValueError(f"v9 command contract mismatch: {job['name']}")
    if len(initializer_hashes) != 3:
        raise ValueError("v9 deployment does not contain three paired initializers")

    return {
        "schema_version": "transit-h2oplus-v9-adaptive-deployment-verification-v2",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "workspace": str(root),
        "deployment_manifest_sha256": sha256(deployment_path),
        "base_v7_deployment_manifest_sha256": sha256(base_path),
        "correction_sha256": sha256(correction_path),
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
