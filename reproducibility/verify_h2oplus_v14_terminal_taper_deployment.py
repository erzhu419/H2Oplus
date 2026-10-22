#!/usr/bin/env python3
"""Independently verify the v14 deployment without training or running SUMO."""

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


DEPLOYMENT_SCHEMA = "transit-h2oplus-v14-terminal-taper-deployment-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v14-terminal-taper-protocol-v1"
PLAN_SCHEMA = "transit-h2oplus-v14-terminal-taper-training-plan-v1"
ADMISSION_SCHEMA = "transit-training-admission-v14-terminal-taper-development-v1"
SOURCE_HANDOFF_SCHEMA = "transit-h2oplus-v12-prefix-handoff-v2"
EXPECTED_SEEDS = {6201, 6202, 6203, 6204, 6205}
EXPECTED_TERMINAL_MULTIPLIERS = {
    "terminal-half": 0.50,
    "terminal-quarter": 0.25,
    "terminal-zero": 0.0,
}
EXPECTED_BASE_PROXIMAL_RATE = 5.0e-5
BASE_MANIFEST_SHA256 = (
    "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
)
SEALED_V12_HASHES = {
    "prefix_closeout": "8bb303dbe038089422ba88f28bf06cd47a672b71f929d1c14306679b943d0ef1",
    "prefix_verification": "b13b7f0827326f52c6c9f512a7385e71e5abf40a8f15df0e70d5ec26126242e7",
    "branch_closeout": "998f47488b50b2cd40f9a6ddd5a399712bb9208cb1ef883cad1181f800385b66",
    "branch_verification": "6835e3c545be22c25bd9e3c7eae7ef987bb1864e027c4fef9849788f744868c8",
    "development_analysis": "49a172e8ff4340c64c33f683223c4dd9bb7b2a8aa501ff3a3abe13c8d20f4f13",
    "evaluation_closeout": "18379992c2c04700fc6460cffda3b5a4ad23feef73841b8ba00b25d2a9c14969",
    "evaluation_verification": "ed3db861b6a27add86e809ec76ea00485fe3ddd914ed302618e9264fb484f8d1",
    "v12_protocol": "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
}
SEALED_V13_HASHES = {
    "branch_closeout": "94c848856082de9f5cda3f794489b5744059fbe9508967e272bf942f689579ec",
    "branch_verification": "c8486651fe1f676128ce7f472463f3ba3a0d82321f8b3aa7855ec3a577262b9b",
    "development_analysis": "9365833e0756f1a2876d5abfa7bc90e96edd27b2c66cfedf59ed0241dd7c5c57",
    "evaluation_closeout": "229bfa6e3825082969362a9602faba9c02a6784d0ecbe74f5e41c626f4a7e0aa",
    "evaluation_verification": "632fb716c9842e6d40bcb037b0b8b84a854ebe46891920309e079438273f7bce",
    "evaluation_file_manifest": "c4ab0b2f8f07f0e5ffcc7105a759931aee063c11cae817ac85e7c6a65d130a21",
    "evaluation_file_manifest_verification": "90fe7bfb78629d9c617723e09ad40ad7c966bd628821594c03365a3accd66113",
    "protocol": "8f330bc7a55b35c9c10ae03434ddbecd30528e4529b01e734dbbd6c44bdd5f12",
    "evaluation_source_freeze": "6bf24fe1bbadd1fa55332b7f90c0127ca25188f590a008d9e61d2951930a1519",
    "evaluation_methods": "712cea480fe1c35bfee4186fbbea137682b8515888d40a05870a55f695caa104",
}


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


def runtime_source_paths(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_v14_source_sha256"
    ]
    if len(functions) != 1:
        raise ValueError("v14 wrapper runtime source-map function drifted")
    assignment = next(
        (
            node
            for node in functions[0].body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "paths"
                for target in node.targets
            )
        ),
        None,
    )
    if assignment is None or not isinstance(assignment.value, ast.Tuple):
        raise ValueError("v14 wrapper runtime source-map tuple drifted")
    result = {"SimpleSAC/train_transit_v14_terminal_taper.py"}
    for value in assignment.value.elts[1:]:
        if (
            not isinstance(value, ast.BinOp)
            or not isinstance(value.op, ast.Div)
            or not isinstance(value.right, ast.Constant)
            or not isinstance(value.right.value, str)
        ):
            raise ValueError("v14 runtime source-map path is not statically auditable")
        result.add(value.right.value)
    return result


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
    from SimpleSAC.transit_v14.provenance import checkpoint_state_sha256
    from SimpleSAC.transit_v14.trainer import (
        BASE_PROXIMAL_RATE,
        PREDESIGNATED_CANDIDATE,
        TERMINAL_ACTOR_LR_MULTIPLIERS,
        V14_TRAINER_VARIANTS,
        integrated_actor_lr_exposure,
        terminal_actor_lr_multiplier,
    )

    deployment_path = root / "deployment_manifest_v14_v1.json"
    deployment = json.loads(deployment_path.read_text())
    entries = deployment.get("entries", [])
    if (
        deployment.get("schema_version") != DEPLOYMENT_SCHEMA
        or deployment.get("status")
        != "immutable-v14-v1-overlay-on-frozen-v7-workspace"
        or deployment.get("study_stage") != "development"
        or deployment.get("formal_held_out_outcomes_included") is not False
        or deployment.get("frozen_v7_through_v13_evidence_modified") is not False
        or deployment.get("checkpoint_selection_by_behavior") is not False
        or len(entries) != deployment.get("overlay_entry_count")
        or stable_object_sha256({"entries": entries})
        != deployment.get("overlay_aggregate_sha256")
    ):
        raise ValueError("v14 deployment manifest scope or aggregate drifted")
    overlay_paths = set()
    for row in entries:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in overlay_paths:
            raise ValueError("v14 overlay path is unsafe or duplicated")
        overlay_paths.add(row["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"v14 overlay entry drifted: {relative}")

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
        raise ValueError("v14 overlay replaces a frozen base v7 entry")

    control = root / "reproducibility/h2oplus_v14_control_v1"
    registration = json.loads((control / "registration.json").read_text())
    authorization = json.loads((control / "authorization.json").read_text())
    diagnosis_path = control / "diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text())
    rationale_path = control / "rationale.json"
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
        or diagnosis.get("schema_version")
        != "transit-h2oplus-v14-terminal-taper-diagnosis-v1"
        or diagnosis.get("status") != "frozen-before-v14-registration-and-training"
        or diagnosis.get("formal_held_out_outcomes_read") is not False
        or diagnosis.get("local_training_or_sumo_executed") is not False
        or diagnosis.get("v13_artifact_integrity_passed") is not True
        or diagnosis.get("v13_training_stability_passed_15_of_15") is not True
        or diagnosis.get("v13_development_screening_passed") is not False
        or diagnosis.get("strong_lower_aulc_than_medium_in_all_five_seeds")
        is not True
        or diagnosis.get("late_relative_l2_acceleration_observed") is not True
        or diagnosis.get("checkpoint_selection_by_behavior") is not False
        or rationale.get("status")
        != "frozen-before-v14-training-or-new-development-outcomes"
        or rationale.get("formal_held_out_outcomes_read") is not False
        or rationale.get("local_training_or_sumo_executed") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or protocol.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or freeze.get("development_evaluation_submitted") is not False
        or freeze.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("v14 control scope drifted")
    for payload in (registration, authorization, plan, protocol, freeze):
        diagnosis_descriptor = payload.get("diagnosis")
        rationale_descriptor = payload.get("rationale")
        if (
            not isinstance(diagnosis_descriptor, dict)
            or verify_descriptor(diagnosis_descriptor) != diagnosis_path
            or not isinstance(rationale_descriptor, dict)
            or verify_descriptor(rationale_descriptor) != rationale_path
        ):
            raise ValueError("v14 control does not consistently bind its diagnosis")

    sealed_v12 = rationale.get("sealed_v12_handoff_inputs", {})
    sealed_v13 = rationale.get("sealed_v13_inputs", {})
    if set(sealed_v12) != set(SEALED_V12_HASHES):
        raise ValueError("v14 rationale sealed-v12 input set drifted")
    if set(sealed_v13) != set(SEALED_V13_HASHES):
        raise ValueError("v14 rationale sealed-v13 input set drifted")
    for label, value in sealed_v12.items():
        path = verify_descriptor(value)
        if (
            path.parent != control / "sealed_v12"
            or sha256(path) != SEALED_V12_HASHES[label]
        ):
            raise ValueError(f"sealed v12 evidence drifted: {label}")
    for label, value in sealed_v13.items():
        path = verify_descriptor(value)
        if (
            path.parent != control / "sealed_v13"
            or sha256(path) != SEALED_V13_HASHES[label]
        ):
            raise ValueError(f"sealed v13 evidence drifted: {label}")
    if diagnosis.get("sealed_v13_inputs") != sealed_v13:
        raise ValueError("v14 diagnosis and rationale bind different v13 evidence")

    source_paths = {
        relative: root / relative for relative in protocol["v14_source_sha256"]
    }
    observed_source = {name: sha256(path) for name, path in source_paths.items()}
    if (
        observed_source != protocol.get("v14_source_sha256")
        or observed_source != plan.get("v14_source_sha256")
        or observed_source != registration.get("v14_source_sha256")
        or observed_source != freeze.get("v14_source_sha256")
    ):
        raise ValueError("v14 source map drifted")
    wrapper = root / "SimpleSAC/train_transit_v14_terminal_taper.py"
    if (
        runtime_source_paths(wrapper) != set(observed_source)
        or len(observed_source) != 16
        or module_string_constant(wrapper, "PROTOCOL_SCHEMA") != PROTOCOL_SCHEMA
        or module_string_constant(wrapper, "ADMISSION_SCHEMA") != ADMISSION_SCHEMA
    ):
        raise ValueError("v14 wrapper schema drifted")
    if (
        BASE_PROXIMAL_RATE != EXPECTED_BASE_PROXIMAL_RATE
        or TERMINAL_ACTOR_LR_MULTIPLIERS != EXPECTED_TERMINAL_MULTIPLIERS
        or set(V14_TRAINER_VARIANTS) != set(EXPECTED_TERMINAL_MULTIPLIERS)
        or PREDESIGNATED_CANDIDATE != "terminal-quarter"
        or any(
            terminal_actor_lr_multiplier(10_000, terminal) != 1.0
            or terminal_actor_lr_multiplier(19_999, terminal) != terminal
            for terminal in EXPECTED_TERMINAL_MULTIPLIERS.values()
        )
        or not (
            integrated_actor_lr_exposure(0.0)
            < integrated_actor_lr_exposure(0.25)
            < integrated_actor_lr_exposure(0.50)
        )
    ):
        raise ValueError("v14 terminal-taper intervention registry drifted")
    selected = rationale.get("selected_intervention", {})
    if (
        selected.get("base_proximal_rate") != EXPECTED_BASE_PROXIMAL_RATE
        or selected.get("terminal_actor_lr_multipliers")
        != EXPECTED_TERMINAL_MULTIPLIERS
        or selected.get("predesignated_candidate") != PREDESIGNATED_CANDIDATE
        or selected.get("checkpoint_selection_by_behavior") is not False
    ):
        raise ValueError("v14 registered intervention drifted")
    taper = protocol.get("terminal_taper", {})
    gates = protocol.get("analysis_gates", {})
    if (
        taper.get("base_proximal_rate") != EXPECTED_BASE_PROXIMAL_RATE
        or taper.get("terminal_actor_lr_multipliers")
        != EXPECTED_TERMINAL_MULTIPLIERS
        or taper.get("terminal_taper_online_event_interval") != [15_000, 25_000]
        or taper.get("predesignated_candidate") != PREDESIGNATED_CANDIDATE
        or gates.get("predesignated_candidate") != PREDESIGNATED_CANDIDATE
        or gates.get(
            "candidate_relative_aulc_improvement_over_v13_strong_at_least"
        )
        != 0.005
        or gates.get("candidate_seed_wins_over_v13_strong_at_least") != 4
        or gates.get("candidate_mean_lower_than_all_controls_at_25000") is not True
        or gates.get("all_15_training_runs_pass_registered_stability") is not True
        or gates.get("all_branches_share_exact_seed_matched_v12_10k_state")
        is not True
    ):
        raise ValueError("v14 preregistered scientific gates drifted")

    jobs = plan.get("jobs", [])
    if (
        len(jobs) != 15
        or len({job["run_signature"] for job in jobs}) != 15
        or {job["seed"] for job in jobs} != EXPECTED_SEEDS
        or {job["variant"] for job in jobs}
        != set(EXPECTED_TERMINAL_MULTIPLIERS)
        or sum(bool(job.get("predesignated_candidate")) for job in jobs) != 5
        or any(not job["name"].startswith("h2oplus_v14_terminal_") for job in jobs)
        or Path(plan.get("workspace", "")).resolve() != root
        or Path(plan.get("remote_output_root", "")).name != "remote_results_v1"
        or (Path(plan.get("remote_output_root", "")).exists())
        or protocol.get("terminal_taper", {}).get(
            "runtime_admission_preflight_required_before_submission"
        )
        is not True
    ):
        raise ValueError("v14 job matrix or pristine output-root contract drifted")

    handoff_hashes = set()
    initializer_hashes = set()
    actor_hashes = set()
    for job in jobs:
        if (
            job.get("stage") != "branch"
            or job.get("base_proximal_rate") != EXPECTED_BASE_PROXIMAL_RATE
            or job.get("terminal_actor_lr_multiplier")
            != EXPECTED_TERMINAL_MULTIPLIERS[job["variant"]]
            or job.get("full_rate_equivalent_actor_updates")
            != integrated_actor_lr_exposure(
                EXPECTED_TERMINAL_MULTIPLIERS[job["variant"]]
            )
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
            raise ValueError(f"v14 job contract drifted: {job.get('name')}")

        admission_path = Path(job["training_admission"]).resolve(strict=True)
        admission = json.loads(admission_path.read_text())
        if (
            admission_path.parent != control / "admissions"
            or admission.get("schema_version") != ADMISSION_SCHEMA
            or admission.get("passed") is not True
            or admission.get("formal_held_out_outcomes_read") is not False
            or admission.get("selected_arguments") != job
            or admission.get("run_signature") != job["run_signature"]
            or admission.get("v14_source_sha256") != observed_source
            or admission.get("v14_wrapper", {}).get("sha256") != sha256(wrapper)
            or admission.get("v14_protocol", {}).get("sha256")
            != sha256(control / "protocol.json")
            or admission.get("v14_diagnosis", {}).get("sha256")
            != sha256(diagnosis_path)
            or admission.get("v14_rationale", {}).get("sha256")
            != sha256(rationale_path)
            or admission.get("sealed_v13_branch_closeout", {}).get("sha256")
            != SEALED_V13_HASHES["branch_closeout"]
            or admission.get("checks", {}).get(
                "runtime_admission_preflight_required_before_submission"
            )
            is not True
        ):
            raise ValueError(f"v14 admission drifted: {admission_path}")

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
            raise ValueError(f"v14 initializer drifted: {initializer_path}")
        initializer_hashes.add(initializer["sha256"])

        manifest_path = Path(job["branch_handoff_manifest"]).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        source = job["source_handoff"]
        if (
            manifest_path.parent.parent.parent.name
            != "h2oplus_v14_source_handoffs_v1"
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
            raise ValueError(f"v14 source handoff drifted: {manifest_path}")
        for row in manifest["files"]:
            path = manifest_path.parent / row["path"]
            if (
                not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256(path) != row["sha256"]
            ):
                raise ValueError(f"v14 source handoff file drifted: {path}")
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
            raise ValueError(f"v14 source checkpoint drifted: {checkpoint}")
        handoff_hashes.add(source["checkpoint_state_sha256"])
        actor_hashes.add(source["actor_tensor_sha256"])

    if len(initializer_hashes) != 5 or len(handoff_hashes) != 5 or len(actor_hashes) != 5:
        raise ValueError("v14 seed-specific source identities are not one-to-one")
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
        raise ValueError("v14 deployment contains unregistered Python bytecode")

    return {
        "schema_version": "transit-h2oplus-v14-terminal-taper-deployment-verification-v1",
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
        "runtime_source_path_count": len(runtime_source_paths(wrapper)),
        "runtime_source_map_matches_frozen_admissions": True,
        "diagnosis_sha256": sha256(diagnosis_path),
        "sealed_v12_files": len(sealed_v12),
        "sealed_v13_files": len(sealed_v13),
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
