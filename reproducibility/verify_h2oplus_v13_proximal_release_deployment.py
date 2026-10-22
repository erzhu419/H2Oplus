#!/usr/bin/env python3
"""Independently verify the v13 deployment without training or running SUMO."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEPLOYMENT_SCHEMA = "transit-h2oplus-v13-proximal-release-deployment-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v13-proximal-release-protocol-v1"
PLAN_SCHEMA = "transit-h2oplus-v13-proximal-release-training-plan-v1"
ADMISSION_SCHEMA = "transit-training-admission-v13-proximal-release-development-v1"
SOURCE_HANDOFF_SCHEMA = "transit-h2oplus-v12-prefix-handoff-v2"
EXPECTED_SEEDS = {6201, 6202, 6203, 6204, 6205}
EXPECTED_RATES = {
    "proximal-weak": 1.0e-5,
    "proximal-medium": 2.5e-5,
    "proximal-strong": 5.0e-5,
}
BASE_MANIFEST_SHA256 = (
    "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def actor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_descriptor(value: dict[str, Any]) -> Path:
    path = Path(value.get("path", "")).expanduser().resolve(strict=True)
    if path.stat().st_size != value.get("bytes") or sha256(path) != value.get(
        "sha256"
    ):
        raise ValueError(f"descriptor drifted: {path}")
    return path


def verify(workspace: Path) -> dict[str, Any]:
    root = workspace.expanduser().resolve(strict=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from SimpleSAC.transit_v13.provenance import checkpoint_state_sha256
    from SimpleSAC.transit_v13.trainer import (
        PREDESIGNATED_CANDIDATE,
        PROXIMAL_RATES,
        REGISTERED_POST_10K_ACTOR_UPDATES,
        V13_TRAINER_VARIANTS,
        proximal_retention,
    )

    deployment_path = root / "deployment_manifest_v13_v1.json"
    deployment = json.loads(deployment_path.read_text())
    entries = deployment.get("entries", [])
    if (
        deployment.get("schema_version") != DEPLOYMENT_SCHEMA
        or deployment.get("study_stage") != "development"
        or deployment.get("formal_held_out_outcomes_included") is not False
        or len(entries) != deployment.get("overlay_entry_count")
        or stable_object_sha256({"entries": entries})
        != deployment.get("overlay_aggregate_sha256")
    ):
        raise ValueError("v13 deployment manifest scope or aggregate drifted")
    overlay_paths = set()
    for row in entries:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in overlay_paths:
            raise ValueError("v13 overlay path is unsafe or duplicated")
        overlay_paths.add(row["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"v13 overlay entry drifted: {relative}")

    base_path = root / "deployment_manifest.json"
    if sha256(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError("base v7 deployment manifest drifted")
    base = json.loads(base_path.read_text())
    base_entries = base.get("entries", [])
    if (
        len(base_entries) != base.get("entry_count")
        or stable_object_sha256(base_entries) != base.get("aggregate_sha256")
    ):
        raise ValueError("base v7 deployment aggregate drifted")
    base_paths = set()
    for row in base_entries:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in base_paths:
            raise ValueError("base v7 deployment path is unsafe or duplicated")
        base_paths.add(row["path"])
        path = root / relative
        if row.get("type") == "file":
            if (
                not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256(path) != row["sha256"]
            ):
                raise ValueError(f"base v7 file drifted: {relative}")
        elif row.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != row["target"]:
                raise ValueError(f"base v7 symlink drifted: {relative}")
        else:
            raise ValueError(f"unknown base deployment entry type: {relative}")
    if overlay_paths & base_paths:
        raise ValueError("v13 overlay replaces a frozen base v7 entry")

    control = root / "reproducibility/h2oplus_v13_control_v1"
    registration = json.loads((control / "registration.json").read_text())
    authorization = json.loads((control / "authorization.json").read_text())
    rationale = json.loads((control / "rationale.json").read_text())
    plan = json.loads((control / "training_plan.json").read_text())
    protocol = json.loads((control / "protocol.json").read_text())
    freeze = json.loads((control / "freeze.json").read_text())
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or protocol.get("schema_version") != PROTOCOL_SCHEMA
        or plan.get("jobs") != protocol.get("training_jobs")
        or registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("local_training_or_sumo_allowed") is not False
        or authorization.get("remote_training_authorized") is not True
        or authorization.get("prefix_training_rerun_authorized") is not False
        or authorization.get("local_training_or_sumo_authorized") is not False
        or authorization.get("formal_confirmation_stage_authorized") is not False
        or rationale.get("status")
        != "frozen-before-v13-training-or-new-development-outcomes"
        or rationale.get("formal_held_out_outcomes_read") is not False
        or rationale.get("local_training_or_sumo_executed") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or freeze.get("development_evaluation_submitted") is not False
        or freeze.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("v13 control scope drifted")

    source_paths = {
        relative: root / relative for relative in protocol["v13_source_sha256"]
    }
    observed_source = {name: sha256(path) for name, path in source_paths.items()}
    if (
        observed_source != protocol.get("v13_source_sha256")
        or observed_source != plan.get("v13_source_sha256")
        or observed_source != registration.get("v13_source_sha256")
        or observed_source != freeze.get("v13_source_sha256")
    ):
        raise ValueError("v13 source map drifted")
    wrapper = root / "SimpleSAC/train_transit_v13_proximal_release.py"
    if (
        module_string_constant(wrapper, "PROTOCOL_SCHEMA") != PROTOCOL_SCHEMA
        or module_string_constant(wrapper, "ADMISSION_SCHEMA") != ADMISSION_SCHEMA
    ):
        raise ValueError("v13 wrapper schema drifted")
    if (
        PROXIMAL_RATES != EXPECTED_RATES
        or set(V13_TRAINER_VARIANTS) != set(EXPECTED_RATES)
        or PREDESIGNATED_CANDIDATE != "proximal-medium"
        or REGISTERED_POST_10K_ACTOR_UPDATES != 15_000
        or not 0.0
        < proximal_retention(EXPECTED_RATES["proximal-strong"], 15_000)
        < proximal_retention(EXPECTED_RATES["proximal-medium"], 15_000)
        < proximal_retention(EXPECTED_RATES["proximal-weak"], 15_000)
        < 1.0
    ):
        raise ValueError("v13 proximal intervention registry drifted")

    sealed = rationale.get("sealed_v12_inputs", {})
    if set(sealed) != {
        "prefix_closeout",
        "prefix_verification",
        "branch_closeout",
        "branch_verification",
        "development_analysis",
        "evaluation_closeout",
        "evaluation_verification",
        "v12_protocol",
    }:
        raise ValueError("v13 rationale sealed-input set drifted")
    for value in sealed.values():
        path = verify_descriptor(value)
        if path.parent != control / "sealed_v12":
            raise ValueError("sealed v12 evidence is outside v13 deployment")

    jobs = plan.get("jobs", [])
    if (
        len(jobs) != 15
        or len({job["run_signature"] for job in jobs}) != 15
        or {job["seed"] for job in jobs} != EXPECTED_SEEDS
        or {job["variant"] for job in jobs} != set(EXPECTED_RATES)
        or sum(bool(job.get("predesignated_candidate")) for job in jobs) != 5
        or Path(plan.get("workspace", "")).resolve() != root
        or (Path(plan.get("remote_output_root", "")).exists())
    ):
        raise ValueError("v13 job matrix or pristine output-root contract drifted")

    handoff_hashes = set()
    initializer_hashes = set()
    actor_hashes = set()
    for job in jobs:
        if (
            job.get("stage") != "branch"
            or job.get("proximal_rate") != EXPECTED_RATES[job["variant"]]
            or job.get("online_event_cap") != 25_000
            or job.get("epochs") != 40
            or job.get("expected_total_updates") != 70_000
            or job.get("export_online_events") != [25_000]
            or job.get("milestones")
            != [
                {
                    "online_events": 25_000,
                    "learner_update_steps": 70_000,
                    "ratio_optimizer_steps": 6_000,
                }
            ]
            or "--resume" in job["command_arguments"]
            or "--allow-no-resume" in job["command_arguments"]
            or job["command_arguments"].count("--branch-handoff-manifest") != 1
        ):
            raise ValueError(f"v13 job contract drifted: {job.get('name')}")

        admission_path = Path(job["training_admission"]).resolve(strict=True)
        admission = json.loads(admission_path.read_text())
        if (
            admission_path.parent != control / "admissions"
            or admission.get("schema_version") != ADMISSION_SCHEMA
            or admission.get("passed") is not True
            or admission.get("formal_held_out_outcomes_read") is not False
            or admission.get("selected_arguments") != job
            or admission.get("run_signature") != job["run_signature"]
            or admission.get("v13_source_sha256") != observed_source
            or admission.get("v13_wrapper", {}).get("sha256") != sha256(wrapper)
            or admission.get("v13_protocol", {}).get("sha256")
            != sha256(control / "protocol.json")
        ):
            raise ValueError(f"v13 admission drifted: {admission_path}")

        initializer = job["offline_initializer"]
        initializer_path = Path(initializer["path"]).resolve(strict=True)
        initializer_payload = torch.load(
            initializer_path, map_location="cpu", weights_only=False
        )
        if (
            sha256(initializer_path) != initializer["sha256"]
            or initializer_path.stat().st_size != initializer["bytes"]
            or stable_object_sha256(initializer_payload.get("extra"))
            != initializer["extra_sha256"]
            or initializer_payload.get("extra", {}).get("source_training_seed")
            != job["seed"]
        ):
            raise ValueError(f"v13 initializer drifted: {initializer_path}")
        initializer_hashes.add(initializer["sha256"])

        manifest_path = Path(job["branch_handoff_manifest"]).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        source = job["source_handoff"]
        if (
            manifest_path.parent.parent.parent.name
            != "h2oplus_v13_source_handoffs_v1"
            or manifest.get("schema_version") != SOURCE_HANDOFF_SCHEMA
            or manifest.get("passed") is not True
            or manifest.get("formal_held_out_outcomes_read") is not False
            or manifest.get("seed") != job["seed"]
            or sha256(manifest_path) != source["manifest_sha256"]
            or manifest.get("protocol_sha256") != source["v12_protocol_sha256"]
            or manifest.get("v12_source_sha256") != source["v12_source_sha256"]
            or manifest.get("checkpoint_state_sha256")
            != source["checkpoint_state_sha256"]
            or manifest.get("actor_tensor_sha256") != source["actor_tensor_sha256"]
            or manifest.get("temperature_tensor_sha256")
            != source["temperature_tensor_sha256"]
        ):
            raise ValueError(f"v13 source handoff drifted: {manifest_path}")
        for row in manifest["files"]:
            path = manifest_path.parent / row["path"]
            if (
                not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256(path) != row["sha256"]
            ):
                raise ValueError(f"v13 source handoff file drifted: {path}")
        checkpoint = manifest_path.parent / "checkpoints/latest.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            checkpoint_state_sha256(checkpoint)
            != source["checkpoint_state_sha256"]
            or int(checkpoint_payload.get("total_updates", -1)) != 55_000
            or int(checkpoint_payload.get("online_critic_updates", -1)) != 5_000
            or actor_digest(checkpoint_payload["actor"])
            != source["actor_tensor_sha256"]
            or actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
            != source["temperature_tensor_sha256"]
        ):
            raise ValueError(f"v13 source checkpoint drifted: {checkpoint}")
        handoff_hashes.add(source["checkpoint_state_sha256"])
        actor_hashes.add(source["actor_tensor_sha256"])

    if len(initializer_hashes) != 5 or len(handoff_hashes) != 5 or len(actor_hashes) != 5:
        raise ValueError("v13 seed-specific source identities are not one-to-one")
    frozen_base_bytecode = {
        row["path"]
        for row in base_entries
        if row.get("type") == "file"
        and (
            Path(row["path"]).suffix in {".pyc", ".pyo"}
            or "__pycache__" in Path(row["path"]).parts
        )
    }
    actual_bytecode = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".pyc", ".pyo"}
            or "__pycache__" in path.relative_to(root).parts
        )
    }
    if actual_bytecode != frozen_base_bytecode:
        raise ValueError("v13 deployment contains unregistered Python bytecode")

    return {
        "schema_version": "transit-h2oplus-v13-proximal-release-deployment-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "deployment_root": str(root),
        "deployment_manifest_sha256": sha256(deployment_path),
        "base_deployment_manifest_sha256": sha256(base_path),
        "base_entries": len(base_entries),
        "overlay_entries": len(entries),
        "verified_jobs": len(jobs),
        "verified_source_handoffs": len(handoff_hashes),
        "verified_initializers": len(initializer_hashes),
        "frozen_base_bytecode_files": len(frozen_base_bytecode),
        "predesignated_candidate": PREDESIGNATED_CANDIDATE,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
    }


def main() -> int:
    args = parse_args()
    payload = verify(args.workspace)
    if args.output is not None:
        atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
