#!/usr/bin/env python3
"""Independently verify the v12 deployment without running SUMO."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEPLOYMENT_SCHEMA = "transit-h2oplus-v12-staged-release-deployment-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v12-staged-release-protocol-v2"
PLAN_SCHEMA = "transit-h2oplus-v12-staged-release-training-plan-v2"
ADMISSION_SCHEMA = "transit-training-admission-v12-staged-release-development-v2"
RATIONALE_SCHEMA = "transit-h2oplus-v12-staged-release-rationale-v1"
CORRECTION_SCHEMA = "transit-h2oplus-v12-staged-release-deployment-correction-v2"
EXPECTED_VARIANTS = {
    "shared-prefix",
    "abrupt-control",
    "actor-ramp",
    "temperature-ramp",
    "actor-anchor",
    "staged-combined",
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


def module_string_constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f"missing unique string constant {name} in {path}")
    return values[0]


def verify_descriptor(value: dict[str, Any], *, label: str) -> Path:
    path = Path(value.get("path", "")).expanduser().resolve(strict=True)
    if path.stat().st_size != value.get("bytes") or sha256(path) != value.get(
        "sha256"
    ):
        raise ValueError(f"v12 descriptor drifted: {label}")
    return path


def verify(workspace: Path) -> dict[str, Any]:
    root = workspace.expanduser().resolve(strict=True)
    deployment_path = root / "deployment_manifest_v12_v2.json"
    deployment = json.loads(deployment_path.read_text())
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("unexpected v12 deployment schema")
    entries = deployment.get("entries", [])
    if (
        len(entries) != deployment.get("overlay_entry_count")
        or stable_object_sha256({"entries": entries})
        != deployment.get("overlay_aggregate_sha256")
    ):
        raise ValueError("v12 overlay manifest aggregate mismatch")
    paths = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in paths:
            raise ValueError("v12 overlay contains an unsafe or duplicate path")
        paths.add(entry["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"v12 overlay entry mismatch: {relative}")

    base = deployment["base_v7_training_deployment"]
    base_path = root / "deployment_manifest.json"
    if sha256(base_path) != base["sha256"]:
        raise ValueError("base v7 deployment manifest drifted")
    base_payload = json.loads(base_path.read_text())
    base_entries = base_payload.get("entries", [])
    if (
        len(base_entries) != base_payload.get("entry_count")
        or stable_object_sha256(base_entries) != base_payload.get("aggregate_sha256")
    ):
        raise ValueError("base v7 deployment manifest aggregate mismatch")
    base_paths = set()
    for entry in base_entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in base_paths:
            raise ValueError("base v7 deployment contains an unsafe or duplicate path")
        base_paths.add(entry["path"])
        path = root / relative
        if entry.get("type") == "file":
            if (
                not path.is_file()
                or path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]
            ):
                raise ValueError(f"base v7 deployment entry mismatch: {relative}")
        elif entry.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != entry["target"]:
                raise ValueError(f"base v7 deployment symlink mismatch: {relative}")
        else:
            raise ValueError(f"base v7 deployment entry has unknown type: {relative}")
    if paths & base_paths:
        raise ValueError("v12 overlay unexpectedly replaces a frozen base v7 path")
    control = root / "reproducibility/h2oplus_v12_control_v2"
    registration_path = control / "registration.json"
    authorization_path = control / "authorization.json"
    rationale_path = control / "rationale.json"
    correction_path = control / "deployment_correction.json"
    environment_path = control / "environment_design.json"
    freeze_path = control / "freeze.json"
    protocol_path = control / "protocol.json"
    plan_path = control / "training_plan.json"
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    registration = json.loads(registration_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    rationale = json.loads(rationale_path.read_text())
    correction = json.loads(correction_path.read_text())
    environment = json.loads(environment_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
        or rationale.get("schema_version") != RATIONALE_SCHEMA
        or rationale.get("status")
        != "frozen-before-v12-training-or-new-development-outcomes"
        or rationale.get("formal_held_out_outcomes_read") is not False
        or rationale.get("local_training_or_sumo_executed") is not False
        or correction.get("schema_version") != CORRECTION_SCHEMA
        or correction.get("status") != "frozen-before-any-v12-training-or-sumo"
        or correction.get("v1_training_or_sumo_executed") is not False
        or correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("claim_boundary", {}).get(
            "deployment_v1_may_be_used_for_training"
        )
        is not False
        or rationale.get("claim_boundary", {}).get(
            "formal_confirmation_stage_authorized_or_submitted"
        )
        is not False
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
        or protocol.get("rationale", {}).get("sha256")
        != sha256(rationale_path)
        or protocol.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
        or protocol.get("environment_design", {}).get("sha256")
        != sha256(environment_path)
        or protocol.get("training_plan", {}).get("sha256") != sha256(plan_path)
        or plan.get("registration", {}).get("sha256")
        != sha256(registration_path)
        or plan.get("authorization", {}).get("sha256")
        != sha256(authorization_path)
        or plan.get("rationale", {}).get("sha256") != sha256(rationale_path)
        or plan.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
        or plan.get("environment_design", {}).get("sha256")
        != sha256(environment_path)
        or registration.get("rationale", {}).get("sha256")
        != sha256(rationale_path)
        or registration.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
        or authorization.get("rationale", {}).get("sha256")
        != sha256(rationale_path)
        or authorization.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
        or environment.get("rationale", {}).get("sha256")
        != sha256(rationale_path)
        or environment.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
        or freeze.get("rationale", {}).get("sha256")
        != sha256(rationale_path)
        or freeze.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
        or deployment.get("rationale", {}).get("sha256")
        != sha256(rationale_path)
        or deployment.get("deployment_correction", {}).get("sha256")
        != sha256(correction_path)
    ):
        raise ValueError("v12 protocol or plan scope drifted")
    jobs = plan.get("jobs", [])
    if jobs != protocol.get("training_jobs") or len(jobs) != 30:
        raise ValueError("v12 protocol and plan job matrices disagree")
    if (
        Path(plan.get("workspace", "")).resolve() != root
        or Path(plan.get("remote_output_root", "")).parent != root.parent
        or not all(str(job.get("name", "")).startswith("h2oplus_v12a2_") for job in jobs)
    ):
        raise ValueError("v12 workspace or result identity drifted")
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
        raise ValueError("v12 job identity matrix is incomplete")

    source_paths = {
        "SimpleSAC/train_transit_v12_staged_release.py": root
        / "SimpleSAC/train_transit_v12_staged_release.py",
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
        "SimpleSAC/transit_v12/__init__.py": root
        / "SimpleSAC/transit_v12/__init__.py",
        "SimpleSAC/transit_v12/provenance.py": root
        / "SimpleSAC/transit_v12/provenance.py",
        "SimpleSAC/transit_v12/trainer.py": root
        / "SimpleSAC/transit_v12/trainer.py",
    }
    source = {name: sha256(path) for name, path in source_paths.items()}
    if source != protocol.get("v12_source_sha256"):
        raise ValueError("v12 deployed source map drifted")
    fixed_controls = protocol.get("fixed_controls", {})
    expected_control_descriptors = {
        "v8_training_closeout",
        "v8_methods_manifest",
        "v10_branch_closeout",
        "v10_evaluation_closeout",
        "v10_methods_manifest",
        "v11_training_closeout",
        "v11_evaluation_closeout",
        "canonical_methods_manifest",
    }
    if (
        set(fixed_controls) - {"seeds", "methods", "common_evaluation_events"}
        != expected_control_descriptors
        or fixed_controls.get("seeds") != sorted(EXPECTED_SEEDS)
        or fixed_controls.get("methods")
        != [
            "h2oplus",
            "sac-online",
            "h2oplus-v10-delayed-actor",
            "h2oplus-v11-actor-hold",
        ]
        or fixed_controls.get("common_evaluation_events") != [0, 5_000, 10_000, 25_000]
    ):
        raise ValueError("v12 fixed-control contract drifted")
    for label in expected_control_descriptors:
        path = verify_descriptor(fixed_controls[label], label=label)
        if path.parent != control / "fixed_controls" or path.name != f"{label}.json":
            raise ValueError(f"v12 fixed control is outside the deployment: {label}")
    launcher_protocol_schema = module_string_constant(
        source_paths["SimpleSAC/train_transit_v12_staged_release.py"],
        "PROTOCOL_SCHEMA",
    )
    if launcher_protocol_schema != protocol.get("schema_version"):
        raise ValueError("v12 launcher and frozen protocol schema disagree")
    launcher_admission_schema = module_string_constant(
        source_paths["SimpleSAC/train_transit_v12_staged_release.py"],
        "ADMISSION_SCHEMA",
    )
    if launcher_admission_schema != ADMISSION_SCHEMA:
        raise ValueError("v12 launcher and frozen admission schema disagree")
    sealed_inputs = rationale.get("sealed_inputs", {})
    expected_sealed_hashes = {
        "v11_analysis": "e3f6df3d075774dec3e662cbf2b6ef848e5f361f57be45fff7cf159768fa04e1",
        "v11_training_closeout": "489c7efe5cf91f73bb31a9b6df03441c2fb3bcd4d280c65493c5ae8ec3af01a6",
        "v11_training_verification": "ff0ef7c69bac850bf9f743275ffd9bbfc87c0b1c9713a83769a052bd936d256e",
        "v11_evaluation_closeout": "27456e2d2a9acd639e8a7e70c69b8003e0167e4dc0289272a57406a56dc60cd2",
        "v11_evaluation_verification": "45d1e924d2e55d11b1f74a19f628b3ff3d00760bf44488819afeef360f19f8a5",
    }
    if (
        rationale.get("diagnosis", {}).get(
            "v11_worse_than_v10_delayed_actor_at_25000"
        )
        is not True
        or {key: value.get("sha256") for key, value in sealed_inputs.items()}
        != expected_sealed_hashes
    ):
        raise ValueError("v12 rationale inputs drifted")
    expected_v1_incident_hashes = {
        "registration_v1": "bedf92da7e922b212dcb31771b9ef928b2f9f35ba1d850445b4d8ec95c0749f7",
        "authorization_v1": "1d0586956f90a342c86a766dd9dfe208b67c814530e544cfb91c423f005aceda",
        "environment_v1": "3125394a676f025bc1d3e14644db4bfcd2e261edcf53f00d3e4c3dcedb373d53",
        "training_plan_v1": "8c04325a7d24a53e122d661a8337eea268cc03be9bd3007a0db9ff463bd9c6fd",
        "protocol_v1": "061e1341b152caa5edd91f1e36f4fc4820eb36a5a43465478d29308fb0117483",
        "freeze_v1": "863327c5d28a9eceab5aabb925853a2e1a758d4a2b2cce65581253d08ae4c1db",
        "deployment_manifest_v1": "cbb0e4465dfb87a0248dbc67da2571861ad9aa195b88db78b64d497ad1b7b8b2",
        "local_verification_v1": "c63d8705a135b4ddfe210508cfa8db4da1bd2096552b3144c0f7ae6c3b3288f2",
        "deployment_sync_v1": "2659d7f7ef820346d86a42fabf698f35195b017297dc1d8c61625a534165f069",
    }
    if {
        label: value.get("sha256")
        for label, value in correction.get("incident_evidence", {}).items()
    } != expected_v1_incident_hashes:
        raise ValueError("v12 deployment-v1 incident binding drifted")

    initializer_hashes = set()
    for job in jobs:
        admission_path = Path(job["training_admission"])
        admission = json.loads(admission_path.read_text())
        base_admission_path = Path(admission["base_v7_admission_source"]["path"])
        base_admission = json.loads(base_admission_path.read_text())
        if (
            admission.get("schema_version") != ADMISSION_SCHEMA
            or admission.get("passed") is not True
            or admission.get("formal_held_out_outcomes_read") is not False
            or admission.get("run_signature") != job["run_signature"]
            or admission.get("variant") != job["variant"]
            or admission.get("stage") != job["stage"]
            or admission.get("method") != job["method"]
            or admission.get("seed") != job["seed"]
            or admission.get("run_name") != job["name"]
            or admission.get("base_v7_admission_source", {}).get("sha256")
            != sha256(base_admission_path)
            or admission.get("base_v7_admission_source", {}).get("bytes")
            != base_admission_path.stat().st_size
            or admission.get("dataset") != base_admission.get("dataset")
            or admission.get("simulator") != base_admission.get("simulator")
            or admission.get("training_source_sha256")
            != base_admission.get("training_source_sha256")
            or admission.get("v12_source_sha256") != source
            or admission.get("v12_wrapper", {}).get("sha256")
            != source["SimpleSAC/train_transit_v12_staged_release.py"]
            or admission.get("v12_protocol", {}).get("sha256")
            != sha256(protocol_path)
            or admission.get("v12_rationale", {}).get("sha256")
            != sha256(rationale_path)
            or admission.get("v12_deployment_correction", {}).get("sha256")
            != sha256(correction_path)
            or admission.get("selected_arguments") != job
            or admission_path.resolve() != Path(job["training_admission"]).resolve()
        ):
            raise ValueError(f"v12 admission mismatch: {admission_path}")
        initializer = job["offline_initializer"]
        initializer_path = Path(initializer["path"])
        if (
            not initializer_path.is_file()
            or sha256(initializer_path) != initializer["sha256"]
            or initializer_path.stat().st_size != initializer["bytes"]
        ):
            raise ValueError(f"v12 initializer mismatch: {initializer_path}")
        payload = torch.load(initializer_path, map_location="cpu", weights_only=False)
        if (
            payload.get("artifact_schema") != "transit-offline-initializer-v7-v1"
            or stable_object_sha256(payload.get("extra"))
            != initializer["extra_sha256"]
            or payload.get("extra", {}).get("source_run_signature")
            != initializer["source_run_signature"]
        ):
            raise ValueError(f"v12 initializer provenance mismatch: {initializer_path}")
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
                    / "v12_handoff/manifest.json"
                )
            )
        if common_invalid or stage_invalid:
            raise ValueError(f"v12 command contract mismatch: {job['name']}")
    if len(initializer_hashes) != 5:
        raise ValueError("v12 deployment does not contain five paired initializers")

    return {
        "schema_version": "transit-h2oplus-v12-staged-release-deployment-verification-v2",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "verification_host": socket.gethostname(),
        "python_version": platform.python_version(),
        "workspace": str(root),
        "deployment_manifest_sha256": sha256(deployment_path),
        "base_v7_deployment_manifest_sha256": sha256(base_path),
        "registration_sha256": sha256(registration_path),
        "authorization_sha256": sha256(authorization_path),
        "rationale_sha256": sha256(rationale_path),
        "deployment_correction_sha256": sha256(correction_path),
        "environment_sha256": sha256(environment_path),
        "freeze_sha256": sha256(freeze_path),
        "overlay_entries": len(entries),
        "base_entries": len(base_entries),
        "training_jobs": len(jobs),
        "prefix_jobs": len(prefix_jobs),
        "branch_jobs": len(branch_jobs),
        "variants": sorted(EXPECTED_VARIANTS),
        "training_seeds": sorted(EXPECTED_SEEDS),
        "offline_initializers": len(initializer_hashes),
        "launcher_protocol_schema": launcher_protocol_schema,
        "launcher_admission_schema": launcher_admission_schema,
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
