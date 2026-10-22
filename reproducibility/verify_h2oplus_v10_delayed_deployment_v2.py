#!/usr/bin/env python3
"""Independently verify corrected v10 deployment v2 without running SUMO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEPLOYMENT_SCHEMA = "transit-h2oplus-v10-delayed-deployment-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v10-delayed-protocol-v2"
PLAN_SCHEMA = "transit-h2oplus-v10-delayed-training-plan-v2"
ADMISSION_SCHEMA = "transit-training-admission-v10-delayed-development-v2"
CORRECTION_SCHEMA = "transit-h2oplus-v10-delayed-correction-v2"
EXPECTED_VARIANTS = {
    "shared-prefix",
    "unit-control",
    "delayed-replay",
    "delayed-actor",
    "delayed-order",
    "delayed-combined",
}
EXPECTED_BRANCH_VARIANTS = EXPECTED_VARIANTS - {"shared-prefix"}
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
    deployment_path = root / "deployment_manifest_v10.json"
    deployment = json.loads(deployment_path.read_text())
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("unexpected v10 deployment schema")
    entries = deployment.get("entries", [])
    if (
        len(entries) != deployment.get("overlay_entry_count")
        or stable_object_sha256({"entries": entries})
        != deployment.get("overlay_aggregate_sha256")
    ):
        raise ValueError("v10 overlay manifest aggregate mismatch")
    paths = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in paths:
            raise ValueError("v10 overlay contains an unsafe or duplicate path")
        paths.add(entry["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"v10 overlay entry mismatch: {relative}")

    base = deployment["base_v7_training_deployment"]
    base_path = root / "deployment_manifest.json"
    if sha256(base_path) != base["sha256"]:
        raise ValueError("base v7 deployment manifest drifted")
    control = root / "reproducibility/h2oplus_v10_control"
    registration_path = control / "registration.json"
    authorization_path = control / "authorization.json"
    correction_path = control / "correction.json"
    environment_path = control / "environment_design.json"
    freeze_path = control / "freeze.json"
    protocol_path = control / "protocol.json"
    plan_path = control / "training_plan.json"
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    registration = json.loads(registration_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    correction = json.loads(correction_path.read_text())
    environment = json.loads(environment_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
        or correction.get("schema_version") != CORRECTION_SCHEMA
        or correction.get("status")
        != "prefix-v1-invalid-sealed-v2-correction-authorized"
        or correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("local_training_or_sumo_executed") is not False
        or correction.get("invalid_v1_lineage", {}).get(
            "eligible_as_scientific_evidence"
        )
        is not False
        or set(correction.get("invalid_v1_lineage", {}).get("failed_task_ids", []))
        != {"t82617", "t82618", "t82619", "t82620"}
        or set(
            correction.get("invalid_v1_lineage", {}).get(
                "cancelled_before_launch_task_ids", []
            )
        )
        != {"t82621", "t82623", "t82624", "t82625", "t82626"}
        or registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("local_training_or_sumo_allowed") is not False
        or authorization.get("prefix_submission_authorized_now") is not True
        or authorization.get("branch_submission_authorized_only_after_prefix_closeout")
        is not True
        or authorization.get("formal_confirmation_stage_authorized") is not False
        or environment.get("disjointness", {}).get("verified_disjoint") is not True
        or environment.get("formal_held_out_outcomes_read") is not False
        or freeze.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or protocol.get("registration", {}).get("sha256")
        != sha256(registration_path)
        or protocol.get("authorization", {}).get("sha256")
        != sha256(authorization_path)
        or protocol.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or protocol.get("environment_design", {}).get("sha256")
        != sha256(environment_path)
        or protocol.get("training_plan", {}).get("sha256") != sha256(plan_path)
        or plan.get("registration", {}).get("sha256")
        != sha256(registration_path)
        or plan.get("authorization", {}).get("sha256")
        != sha256(authorization_path)
        or plan.get("correction", {}).get("sha256") != sha256(correction_path)
        or plan.get("environment_design", {}).get("sha256")
        != sha256(environment_path)
        or registration.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or authorization.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or environment.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or freeze.get("correction", {}).get("sha256")
        != sha256(correction_path)
        or deployment.get("correction", {}).get("sha256")
        != sha256(correction_path)
    ):
        raise ValueError("v10 protocol or plan scope drifted")
    jobs = plan.get("jobs", [])
    if jobs != protocol.get("training_jobs") or len(jobs) != 30:
        raise ValueError("v10 protocol and plan job matrices disagree")
    if (
        Path(plan.get("workspace", "")).resolve() != root
        or Path(plan.get("remote_output_root", "")).parent != root.parent
        or not all(str(job.get("name", "")).startswith("h2oplus_v10c2_") for job in jobs)
    ):
        raise ValueError("v10 corrected workspace or result identity drifted")
    prefix_jobs = [job for job in jobs if job.get("stage") == "prefix"]
    branch_jobs = [job for job in jobs if job.get("stage") == "branch"]
    if (
        {job["variant"] for job in jobs} != EXPECTED_VARIANTS
        or {job["seed"] for job in jobs} != EXPECTED_SEEDS
        or len({job["run_signature"] for job in jobs}) != 30
        or len(prefix_jobs) != 5
        or len(branch_jobs) != 25
        or {job["seed"] for job in prefix_jobs} != EXPECTED_SEEDS
        or {job["variant"] for job in prefix_jobs} != {"shared-prefix"}
        or any(
            sum(
                job["variant"] == variant and job["seed"] == seed
                for job in branch_jobs
            )
            != 1
            for variant in EXPECTED_BRANCH_VARIANTS
            for seed in EXPECTED_SEEDS
        )
    ):
        raise ValueError("v10 job identity matrix is incomplete")

    source_paths = {
        "SimpleSAC/train_transit_v10_delayed.py": root
        / "SimpleSAC/train_transit_v10_delayed.py",
        "SimpleSAC/transit_v10/__init__.py": root
        / "SimpleSAC/transit_v10/__init__.py",
        "SimpleSAC/transit_v10/provenance.py": root
        / "SimpleSAC/transit_v10/provenance.py",
        "SimpleSAC/transit_v10/trainer.py": root
        / "SimpleSAC/transit_v10/trainer.py",
    }
    source = {name: sha256(path) for name, path in source_paths.items()}
    if source != protocol.get("v10_source_sha256"):
        raise ValueError("v10 deployed source map drifted")
    if (
        correction.get("correction", {}).get("corrected_source_sha256") != source
        or correction.get("correction", {}).get("invalid_source_sha256") == source
    ):
        raise ValueError("v10 correction does not bind the deployed source")

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
            or admission.get("stage") != job["stage"]
            or admission.get("v10_source_sha256") != source
            or admission.get("v10_protocol", {}).get("sha256")
            != sha256(protocol_path)
            or admission.get("v10_correction", {}).get("sha256")
            != sha256(correction_path)
        ):
            raise ValueError(f"v10 admission mismatch: {admission_path}")
        initializer = job["offline_initializer"]
        initializer_path = Path(initializer["path"])
        if (
            not initializer_path.is_file()
            or sha256(initializer_path) != initializer["sha256"]
            or initializer_path.stat().st_size != initializer["bytes"]
        ):
            raise ValueError(f"v10 initializer mismatch: {initializer_path}")
        payload = torch.load(initializer_path, map_location="cpu", weights_only=False)
        if (
            payload.get("artifact_schema") != "transit-offline-initializer-v7-v1"
            or stable_object_sha256(payload.get("extra"))
            != initializer["extra_sha256"]
            or payload.get("extra", {}).get("source_run_signature")
            != initializer["source_run_signature"]
        ):
            raise ValueError(f"v10 initializer provenance mismatch: {initializer_path}")
        initializer_hashes.add(initializer["sha256"])
        arguments = job["command_arguments"]
        common_invalid = (
            arguments.count("--offline-initializer") != 1
            or arguments.count("--training-admission") != 1
            or "--resume" in arguments
            or arguments.count("--run-signature") != 1
            or arguments.count("--run-name") != 1
            or arguments.count("--seed") != 1
        )
        if job["stage"] == "prefix":
            stage_invalid = (
                job["variant"] != "shared-prefix"
                or arguments.count("--branch-handoff-manifest") != 0
                or job["online_event_cap"] != 10_000
                or job["epochs"] != 10
                or job["expected_total_updates"] != 55_000
                or job["export_online_events"] != [0, 5_000, 10_000]
                or "prefix" in job
            )
        else:
            prefix = job.get("prefix", {})
            matching = next(
                candidate
                for candidate in prefix_jobs
                if candidate["seed"] == job["seed"]
            )
            stage_invalid = (
                job["variant"] not in EXPECTED_BRANCH_VARIANTS
                or arguments.count("--branch-handoff-manifest") != 1
                or job["online_event_cap"] != 25_000
                or job["epochs"] != 40
                or job["expected_total_updates"] != 70_000
                or job["export_online_events"] != [25_000]
                or prefix.get("run_signature") != matching["run_signature"]
                or prefix.get("seed") != job["seed"]
                or job.get("branch_handoff_manifest")
                != str(
                    Path(plan["remote_output_root"])
                    / matching["name"]
                    / "v10_handoff/manifest.json"
                )
            )
        if common_invalid or stage_invalid:
            raise ValueError(f"v10 command contract mismatch: {job['name']}")
    if len(initializer_hashes) != 5:
        raise ValueError("v10 deployment does not contain five paired initializers")

    return {
        "schema_version": "transit-h2oplus-v10-delayed-deployment-verification-v2",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "verification_host": socket.gethostname(),
        "python_version": platform.python_version(),
        "workspace": str(root),
        "deployment_manifest_sha256": sha256(deployment_path),
        "base_v7_deployment_manifest_sha256": sha256(base_path),
        "registration_sha256": sha256(registration_path),
        "authorization_sha256": sha256(authorization_path),
        "correction_sha256": sha256(correction_path),
        "environment_sha256": sha256(environment_path),
        "freeze_sha256": sha256(freeze_path),
        "overlay_entries": len(entries),
        "training_jobs": len(jobs),
        "prefix_jobs": len(prefix_jobs),
        "branch_jobs": len(branch_jobs),
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
