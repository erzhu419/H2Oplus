#!/usr/bin/env python3
"""Audit and seal the 15 v14 terminal-taper development branches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v12.provenance import checkpoint_state_sha256


REPRO = ROOT / "reproducibility"
STUDY = REPRO / "h2oplus_v14_terminal_taper_results"
BRANCH_ROOT = STUDY / "branches_v1"
EVIDENCE_ROOT = STUDY / "evidence_branches_v1"
DEPLOYMENT = REPRO / "deployments/h2oplus_v14_terminal_taper_pilot_v1"
WORKSPACE = DEPLOYMENT / "workspace"
CONTROL_ROOT = WORKSPACE / "reproducibility/h2oplus_v14_control_v1"
SOURCE_HANDOFF_ROOT = WORKSPACE / "reproducibility/h2oplus_v14_source_handoffs_v1"
PROTOCOL_SHA256 = "ba275fc4c134feadb564b49548699571ca5bf6bfd1aee529759d6ce791df12c5"
SUBMISSION_SHA256 = "d1c2f4211f479edca9dba4e7c31a7d366f8fd4a0ec9ef10c3c054e35aaa7c5b4"
V12_PROTOCOL_SHA256 = "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
STABILITY = {
    "maximum_critic_grad_norm": 1_000_000.0,
    "minimum_hold_probability": 0.01,
    "maximum_hold_probability": 0.99,
    "maximum_absolute_positive_action_mean": 0.98,
}
BASE_PROXIMAL_RATE = 5e-5
TERMINAL_MULTIPLIERS = {
    "terminal-half": 0.5,
    "terminal-quarter": 0.25,
    "terminal-zero": 0.0,
}
EXPECTED_TASK_IDS = {f"t{task_id}" for task_id in range(83541, 83556)}
REQUIRED_FIXED_FILES = {
    "checkpoints/final_actor.pt",
    "checkpoints/latest.pt",
    "checkpoints/runtime_epoch000040.pkl",
    "metadata.json",
    "metrics.jsonl",
    "phase_metrics.json",
    "run.log",
    "summary.json",
    "timings.json",
    "v14_branch_materialization.json",
    "v14_branch_source/latest.pt",
    "v14_branch_source/runtime_epoch000010.pkl",
    "v14_checkpoints/online_events_000025000.pt",
    "v14_export_summary.json",
    "v14_policies/online_events_000025000.pt",
    "v14_policies/policy_manifest.json",
}
FATAL_TEXT = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "Segmentation fault",
    "AssertionError",
    "RuntimeError:",
    "ValueError:",
)
CONTROLS = {
    "training_plan": (CONTROL_ROOT / "training_plan.json", "71e43c0a6ec063f2b96459388835e0b29277695d1714f375d5115921cbd8c831"),
    "protocol": (CONTROL_ROOT / "protocol.json", PROTOCOL_SHA256),
    "freeze": (CONTROL_ROOT / "freeze.json", "3a1eb6a31f6cc4708714541917ef101ce68a3eddae4b64228d27d8ce529b5b5d"),
    "authorization": (CONTROL_ROOT / "authorization.json", "fc3abf9d22a99a8b6fe48e2a74e778f4515d77f9b49943c0f690809d17126f8b"),
    "diagnosis": (CONTROL_ROOT / "diagnosis.json", "1f00a72052df2c12bbbbf5f20da67a5242897eea310e463afc4858447d1e8710"),
    "registration": (CONTROL_ROOT / "registration.json", "6be53642facea16013dc90f84f60a1d861c112e08a9b748a61011eb985dd2a41"),
    "rationale": (CONTROL_ROOT / "rationale.json", "4884febef94455b47cead54dd7e141641dd90160b34dd0245c0eb63108312c80"),
    "deployment_manifest": (WORKSPACE / "deployment_manifest_v14_v1.json", "1df1873929f9d6273201d2266810ff5ab3d4fcb14f8db5fe55eaa91818ae569b"),
    "submission": (REPRO / "h2oplus_v14_terminal_taper_submission_v1.json", SUBMISSION_SHA256),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return False


def actor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"closeout output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_controls() -> dict[str, dict[str, Any]]:
    result = {}
    for name, (path, expected) in CONTROLS.items():
        current = descriptor(path)
        if current["sha256"] != expected:
            raise ValueError(f"frozen control drifted: {name}")
        result[name] = current
    return result


def exact_file_contract(run: Path) -> list[Path]:
    paths = sorted(path for path in run.rglob("*") if path.is_file())
    relative = {path.relative_to(run).as_posix() for path in paths}
    resources = {name for name in relative if name.startswith("resources_resume_") and name.endswith(".csv")}
    if len(resources) != 1 or relative - resources != REQUIRED_FIXED_FILES:
        raise ValueError(f"v14 branch file contract drifted: {run}")
    return paths


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 40 or not finite(rows):
        raise ValueError(f"metrics integrity failed: {path}")
    return rows


def expected_terminal_taper(online_critic_updates: int, terminal: float) -> float:
    update = online_critic_updates
    if update <= 10_000:
        return 1.0
    if update >= 19_999:
        return terminal
    progress = (update - 10_000) / 9_999
    return 1.0 + progress * (terminal - 1.0)


def stability_and_mechanism(
    rows: list[dict[str, Any]], variant: str, expected_exposure: float
) -> dict[str, Any]:
    violations = []
    critics, holds, actions, actor_gradients, proximal_distances = [], [], [], [], []
    actor_multipliers, terminal_tapers = [], []
    terminal = TERMINAL_MULTIPLIERS[variant]
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        critic = float(train["train/critic_grad_norm"])
        critics.append(critic)
        if critic > STABILITY["maximum_critic_grad_norm"]:
            violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": critic})
        if "train/hold_probability" in train:
            hold = float(train["train/hold_probability"])
            holds.append(hold)
            if not STABILITY["minimum_hold_probability"] <= hold <= STABILITY["maximum_hold_probability"]:
                violations.append({"epoch": epoch, "metric": "hold_probability", "value": hold})
        if "train/positive_action_mean" in train:
            action = float(train["train/positive_action_mean"])
            actions.append(action)
            if abs(action) >= STABILITY["maximum_absolute_positive_action_mean"]:
                violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": action})
        actor_gradients.append(float(train["train/actor_grad_norm"]))
        fixed = {
            "ratio/v10_unit_weights": 1.0,
            "v10/unit_ratio_weights": 1.0,
            "v10/offline_replay_fraction": 0.5,
            "v10/online_replay_fraction": 0.5,
            "v11/offline_replay_fraction": 0.5,
            "v11/unit_ratio_weights": 1.0,
        }
        if any(float(train.get(key, math.nan)) != value for key, value in fixed.items()):
            raise ValueError(f"fixed replay/ratio mechanism drifted: {variant} epoch {epoch}")
        if epoch <= 10:
            expected = {
                "v11/actor_hold_active": 1.0,
                "v11/online_only_actor_active": 0.0,
                "train/actor_updated": 0.0,
                "v12/actor_lr_multiplier": 0.0,
                "v12/temperature_lr_multiplier": 0.0,
                "v12/actor_anchor_offline_fraction": 0.0,
                "v12/staged_release_active": 0.0,
            }
        else:
            expected = {
                "v11/actor_hold_active": 0.0,
                "v11/online_only_actor_active": 1.0,
                "train/actor_updated": 1.0,
                "v12/temperature_lr_multiplier": 1.0,
                "v12/actor_anchor_offline_fraction": 0.0,
                "v12/staged_release_active": 1.0,
                "v13/proximal_rate": BASE_PROXIMAL_RATE,
                "v13/proximal_update_executed": 1.0,
                "v13/actor_anchor_restored": 1.0,
                "v14/base_proximal_rate": BASE_PROXIMAL_RATE,
                "v14/terminal_actor_lr_multiplier": terminal,
            }
            online_critic_updates = int(train["train/online_critic_update"])
            expected_release = (
                1.0
                if online_critic_updates >= 10_000
                else 0.1 + (online_critic_updates - 5_000) / 5_000 * 0.9
            )
            actor_multiplier = float(train["v12/actor_lr_multiplier"])
            taper = float(train["v14/current_terminal_taper_multiplier"])
            if not math.isclose(actor_multiplier, expected_release, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"actor release ramp formula drifted: {variant} epoch {epoch}")
            if not math.isclose(
                taper,
                expected_terminal_taper(online_critic_updates, terminal),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"terminal taper formula drifted: {variant} epoch {epoch}")
            if not math.isclose(
                float(train["v14/full_rate_equivalent_actor_updates"]),
                expected_exposure,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"integrated actor exposure drifted: {variant} epoch {epoch}")
            actor_multipliers.append(actor_multiplier)
            terminal_tapers.append(taper)
            proximal_distances.append(float(train["v13/actor_relative_l2_after_pull"]))
        if any(float(train.get(key, math.nan)) != value for key, value in expected.items()):
            raise ValueError(f"terminal-taper mechanism drifted: {variant} epoch {epoch}")
    if (
        len(actor_multipliers) != 30
        or actor_multipliers[-1] != 1.0
        or any(after + 1e-12 < before for before, after in zip(actor_multipliers, actor_multipliers[1:]))
        or terminal_tapers[0] != 1.0
        or terminal_tapers[-1] != terminal
        or any(after > before + 1e-12 for before, after in zip(terminal_tapers, terminal_tapers[1:]))
    ):
        raise ValueError(f"actor release/taper sequence drifted: {variant}")
    return {
        "passed": not violations,
        "thresholds": STABILITY,
        "violations": violations,
        "maximum_critic_grad_norm": max(critics),
        "minimum_hold_probability": min(holds),
        "maximum_hold_probability": max(holds),
        "maximum_absolute_positive_action_mean": max(abs(value) for value in actions),
        "maximum_actor_grad_norm": max(actor_gradients),
        "actor_lr_multiplier_first_after_release": actor_multipliers[0],
        "actor_lr_multiplier_final": actor_multipliers[-1],
        "terminal_taper_multiplier_first_after_release": terminal_tapers[0],
        "terminal_taper_multiplier_final": terminal_tapers[-1],
        "full_rate_equivalent_actor_updates": expected_exposure,
        "final_actor_relative_l2_after_pull": proximal_distances[-1],
    }


def verify_policy(run: Path, entry: dict[str, Any], job: dict[str, Any], admission_sha: str, source_map: dict[str, str]) -> dict[str, str]:
    actor = run / entry["path"]
    checkpoint = run / entry["model_checkpoint"]["path"]
    if (
        actor.stat().st_size != entry["bytes"]
        or sha256(actor) != entry["sha256"]
        or checkpoint.stat().st_size != entry["model_checkpoint"]["bytes"]
        or sha256(checkpoint) != entry["model_checkpoint"]["sha256"]
        or entry["training_checkpoint_sha256"] != entry["model_checkpoint"]["sha256"]
    ):
        raise ValueError(f"policy/checkpoint descriptor drifted: {run.name}")
    policy_payload = torch.load(actor, map_location="cpu", weights_only=False)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_sha = actor_digest(policy_payload["actor"])
    checkpoint_actor_sha = actor_digest(checkpoint_payload["actor"])
    temperature_sha = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
    extra = policy_payload["extra"]
    intervention = extra.get("v14_intervention", {})
    expected = {
        "run_signature": job["run_signature"],
        "method": "h2oplus",
        "variant": job["variant"],
        "stage": "branch",
        "training_seed": job["seed"],
        "total_updates": 70000,
        "v14_online_events_nominal": 25000,
        "v14_online_events_actual": entry["online_events_actual"],
        "v14_learner_update_steps": 70000,
        "v14_ratio_optimizer_steps": 6000,
        "v14_actor_tensor_sha256": entry["actor_tensor_sha256"],
        "v14_temperature_tensor_sha256": entry["temperature_tensor_sha256"],
        "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if any(extra.get(key) != value for key, value in expected.items()):
        raise ValueError(f"v14 policy extra drifted: {run.name}")
    if (
        actor_sha != entry["actor_tensor_sha256"]
        or checkpoint_actor_sha != actor_sha
        or temperature_sha != entry["temperature_tensor_sha256"]
        or extra.get("v14_source_sha256") != source_map
        or extra.get("v14_protocol", {}).get("sha256") != PROTOCOL_SHA256
        or extra.get("training_admission", {}).get("sha256") != admission_sha
        or extra.get("selection_rule") != "all preregistered v14 online-event milestones; no evaluation outcome selects a checkpoint"
        or intervention.get("variant") != job["variant"]
        or intervention.get("schema_version") != "transit-h2oplus-v14-terminal-taper-variant-v1"
        or intervention.get("proximal_rate_per_actor_update") != BASE_PROXIMAL_RATE
        or intervention.get("registered_post_10k_actor_updates") != 15000
        or intervention.get("terminal_actor_lr_taper_events") != [15000, 25000]
        or intervention.get("terminal_actor_lr_taper_critic_updates") != [10000, 19999]
        or intervention.get("terminal_actor_lr_multiplier") != job["terminal_actor_lr_multiplier"]
        or not math.isclose(
            intervention.get("full_rate_equivalent_actor_updates", math.nan),
            job["full_rate_equivalent_actor_updates"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or intervention.get("predesignated_candidate") is not job["predesignated_candidate"]
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
        or intervention.get("actor_and_temperature_hold_through_online_events") != 10000
    ):
        raise ValueError(f"v14 policy tensor/provenance drifted: {run.name}")
    return {"actor": actor_sha, "temperature": temperature_sha}


def audit_run(job: dict[str, Any], submission_row: dict[str, Any], sync_row: dict[str, Any], scheduler_row: dict[str, Any], source_map: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = BRANCH_ROOT / job["name"]
    paths = exact_file_contract(run)
    inventory = [{"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths]
    if inventory != sync_row["files"]:
        raise ValueError(f"post-sync inventory drifted: {run.name}")
    for path in paths:
        if path.suffix == ".json" and not finite(load_json(path)):
            raise ValueError(f"nonfinite JSON: {path}")
        if path.suffix == ".jsonl" and not finite([json.loads(line) for line in path.read_text().splitlines() if line.strip()]):
            raise ValueError(f"nonfinite JSONL: {path}")
    metadata = load_json(run / "metadata.json")
    summary = load_json(run / "summary.json")
    phase = load_json(run / "phase_metrics.json")
    materialization = load_json(run / "v14_branch_materialization.json")
    policies = load_json(run / "v14_policies/policy_manifest.json")
    export = load_json(run / "v14_export_summary.json")
    rows = read_metrics(run / "metrics.jsonl")
    admission_path = Path(job["training_admission"])
    admission = load_json(admission_path)
    admission_sha = sha256(admission_path)
    args = metadata["arguments"]
    runtime = metadata["hardware"]["runtime_fingerprint"]["host"]
    if (
        metadata.get("run_signature") != job["run_signature"]
        or args.get("run_name") != job["name"]
        or args.get("seed") != job["seed"]
        or args.get("method") != "h2oplus"
        or args.get("epochs") != 40
        or args.get("cpu_threads") != 6
        or args.get("warmup_events") != 5000
        or args.get("offline_updates") != 50000
        or metadata.get("mdp_version") != "v7"
        or metadata.get("training_admission", {}).get("sha256") != admission_sha
        or metadata.get("training_admission", {}).get("selected_arguments") != job
        or metadata.get("training_admission", {}).get("v14_source_sha256") != source_map
        or admission.get("selected_arguments") != job
        or admission.get("passed") is not True
        or admission.get("formal_held_out_outcomes_read") is not False
        or runtime.get("cc_environment") != "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
        or runtime.get("device") != "cuda:0"
        or runtime.get("cuda_visible_devices") not in {"0", "1"}
    ):
        raise ValueError(f"metadata/admission/runtime drifted: {run.name}")
    events = int(summary["simulator_decision_events"])
    cap = int(summary["simulator_decision_event_cap"])
    underfill = int(summary["simulator_decision_event_underfill"])
    blocked = int(summary["final_blocked_simultaneous_batch_size"])
    maximum_batch = int(summary["maximum_simultaneous_decisions"])
    if (
        summary.get("schema_version") != "transit-training-summary-v8"
        or summary.get("method") != "h2oplus"
        or summary.get("seed") != job["seed"]
        or summary.get("run_signature") != job["run_signature"]
        or summary.get("completed_epochs") != 40
        or summary.get("start_epoch") != 10
        or summary.get("total_updates") != 70000
        or cap != 25000
        or not 0 <= events <= cap
        or underfill != cap - events
        or maximum_batch <= 0
        or (underfill == 0 and blocked != 0)
        or (underfill > 0 and not underfill < blocked <= maximum_batch)
        or sha256(run / "checkpoints/latest.pt") != summary["checkpoint"]["sha256"]
        or sha256(run / "checkpoints/final_actor.pt") != summary["final_policy"]["sha256"]
        or summary["final_policy"]["source_training_checkpoint_sha256"] != summary["checkpoint"]["sha256"]
        or [int(row["epoch"]) for row in rows] != list(range(1, 41))
        or int(rows[-1]["actual_cumulative_simulator_decision_events"]) != events
        or int(rows[-1]["total_updates"]) != 70000
        or phase.get("offline_updates_completed") != 50000
        or phase.get("ratio_pretraining_complete") is not True
        or phase.get("warmup_complete") is not True
    ):
        raise ValueError(f"summary/event sequence drifted: {run.name}")
    branch_initial = run / "v14_branch_source/latest.pt"
    branch_runtime = run / "v14_branch_source/runtime_epoch000010.pkl"
    handoff_manifest = SOURCE_HANDOFF_ROOT / f"seed{job['seed']}/v12_handoff/manifest.json"
    source_checkpoint = SOURCE_HANDOFF_ROOT / f"seed{job['seed']}/v12_handoff/checkpoints/latest.pt"
    handoff = submission_row["source_handoff"]
    initial_payload = torch.load(branch_initial, map_location="cpu", weights_only=False)
    initial_actor = actor_digest(initial_payload["actor"])
    initial_temperature = actor_digest({"log_alpha": initial_payload["log_alpha"]})
    if (
        materialization.get("schema_version") != "transit-h2oplus-v14-branch-materialization-v1"
        or materialization.get("passed") is not True
        or materialization.get("formal_held_out_outcomes_read") is not False
        or materialization.get("run_name") != job["name"]
        or materialization.get("run_signature") != job["run_signature"]
        or materialization.get("seed") != job["seed"]
        or materialization.get("variant") != job["variant"]
        or materialization.get("protocol_sha256") != PROTOCOL_SHA256
        or materialization.get("handoff_manifest_sha256") != sha256(handoff_manifest)
        or materialization.get("source_checkpoint_sha256") != sha256(source_checkpoint)
        or materialization.get("branch_initial_checkpoint_sha256") != sha256(branch_initial)
        or materialization.get("branch_initial_runtime_sha256") != sha256(branch_runtime)
        or checkpoint_state_sha256(branch_initial) != handoff["checkpoint_state_sha256"]
        or initial_actor != handoff["actor_tensor_sha256"]
        or initial_temperature != handoff["temperature_tensor_sha256"]
        or materialization.get("source_v12_protocol_sha256") != V12_PROTOCOL_SHA256
    ):
        raise ValueError(f"seed-matched v12 handoff drifted: {run.name}")
    entries = policies.get("entries", [])
    if (
        policies.get("schema_version") != "transit-h2oplus-v14-terminal-taper-policy-manifest-v1"
        or policies.get("run_signature") != job["run_signature"]
        or policies.get("seed") != job["seed"]
        or policies.get("variant") != job["variant"]
        or policies.get("stage") != "branch"
        or policies.get("protocol_sha256") != PROTOCOL_SHA256
        or len(entries) != 1
    ):
        raise ValueError(f"policy manifest drifted: {run.name}")
    entry = entries[0]
    if (
        entry.get("online_events_nominal") != 25000
        or entry.get("online_events_actual") != events
        or entry.get("event_budget_deviation") != events - 25000
        or entry.get("learner_update_steps") != 70000
        or entry.get("ratio_optimizer_steps") != 6000
        or entry.get("epoch") != 40
        or entry.get("phase") != "complete"
    ):
        raise ValueError(f"25k milestone accounting drifted: {run.name}")
    tensors = verify_policy(run, entry, job, admission_sha, source_map)
    if (
        export.get("schema_version") != "transit-h2oplus-v14-terminal-taper-export-summary-v1"
        or export.get("passed") is not True
        or export.get("formal_held_out_outcomes_read") is not False
        or export.get("run_signature") != job["run_signature"]
        or export.get("summary_sha256") != sha256(run / "summary.json")
        or export.get("policy_manifest", {}).get("sha256") != sha256(run / "v14_policies/policy_manifest.json")
        or export.get("branch_materialization", {}).get("sha256") != sha256(run / "v14_branch_materialization.json")
    ):
        raise ValueError(f"export closeout drifted: {run.name}")
    log_text = (run / "run.log").read_text(errors="replace")
    scheduler_log = Path(scheduler_row["scheduler_log"]["path"]).read_text(errors="replace")
    if any(marker in log_text or marker in scheduler_log for marker in FATAL_TEXT):
        raise ValueError(f"fatal text in logs: {run.name}")
    stability = stability_and_mechanism(
        rows, job["variant"], job["full_rate_equivalent_actor_updates"]
    )
    record = {
        "task_id": submission_row["task_id"],
        "run_name": job["name"],
        "variant": job["variant"],
        "seed": job["seed"],
        "base_proximal_rate": job["base_proximal_rate"],
        "terminal_actor_lr_multiplier": job["terminal_actor_lr_multiplier"],
        "full_rate_equivalent_actor_updates": job["full_rate_equivalent_actor_updates"],
        "predesignated_candidate": job["predesignated_candidate"],
        "online_events_actual": events,
        "online_event_underfill": underfill,
        "final_blocked_simultaneous_batch_size": blocked,
        "learner_update_steps": 70000,
        "ratio_optimizer_steps": 6000,
        "exact_seed_matched_v12_10k_handoff": True,
        "initial_actor_tensor_sha256": initial_actor,
        "initial_temperature_tensor_sha256": initial_temperature,
        "final_actor_tensor_sha256": tensors["actor"],
        "final_temperature_tensor_sha256": tensors["temperature"],
        "scientific_stability": stability,
        "artifact_integrity_passed": True,
        "runtime_fingerprint_cpath_is_inherited_resume_evidence": runtime.get("cpath_environment"),
        "scheduler_environment_cpath": scheduler_row["required_environment"]["CPATH"],
        "formal_held_out_outcomes_read": False,
    }
    manifest = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths]
    return record, manifest


def main() -> int:
    controls = verify_controls()
    plan = load_json(CONTROLS["training_plan"][0])
    protocol = load_json(CONTROLS["protocol"][0])
    submission = load_json(CONTROLS["submission"][0])
    sync_path = EVIDENCE_ROOT / "branch_sync_v1.json"
    scheduler_path = EVIDENCE_ROOT / "scheduler_evidence_branches_v1.json"
    sync = load_json(sync_path)
    scheduler = load_json(scheduler_path)
    jobs = plan.get("jobs", [])
    if (
        len(jobs) != 15
        or len(submission.get("tasks", [])) != 15
        or sync.get("passed") is not True
        or sync.get("total_runs") != 15
        or scheduler.get("passed") is not True
        or scheduler.get("duplicate_or_retry_count") != 0
        or {row["task_id"] for row in scheduler.get("tasks", [])} != EXPECTED_TASK_IDS
        or protocol.get("analysis_gates", {}).get("all_15_training_runs_pass_registered_stability") is not True
        or protocol.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v14 closeout inputs failed")
    source_map = protocol["v14_source_sha256"]
    for relative, digest in source_map.items():
        if sha256(WORKSPACE / relative) != digest:
            raise ValueError(f"frozen v14 source drifted: {relative}")
    sync_by_name = {row["run_name"]: row for row in sync["runs"]}
    scheduler_by_name = {row["run_name"]: row for row in scheduler["tasks"]}
    submission_by_name = {row["name"]: row for row in submission["tasks"]}
    audit_rows, manifest_records = [], []
    for job in sorted(jobs, key=lambda row: (row["variant"], row["seed"])):
        row, records = audit_run(job, submission_by_name[job["name"]], sync_by_name[job["name"]], scheduler_by_name[job["name"]], source_map)
        audit_rows.append(row)
        manifest_records.extend(records)
    all_stable = all(row["scientific_stability"]["passed"] for row in audit_rows)
    handoff_groups = {}
    for row in audit_rows:
        handoff_groups.setdefault(row["seed"], set()).add((row["initial_actor_tensor_sha256"], row["initial_temperature_tensor_sha256"]))
    if any(len(values) != 1 for values in handoff_groups.values()):
        raise ValueError("variant initial tensors differ within a seed")
    manifest_records.sort(key=lambda row: row["path"])
    audit_path = STUDY / "branch_audit_v1.json"
    manifest_path = STUDY / "branch_file_manifest_v1.json"
    closeout_path = REPRO / "h2oplus_v14_terminal_taper_branch_closeout_v1.json"
    atomic_json(audit_path, {
        "schema_version": "transit-h2oplus-v14-terminal-taper-branch-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "artifact_integrity_passed": True,
        "exact_run_count": len(audit_rows),
        "variant_counts": {
            variant: sum(row["variant"] == variant for row in audit_rows)
            for variant in TERMINAL_MULTIPLIERS
        },
        "runs": audit_rows,
        "all_seed_matched_v12_10k_handoffs_exact": True,
        "all_15_stability_passed": all_stable,
        "stability_violation_count": sum(len(row["scientific_stability"]["violations"]) for row in audit_rows),
        "formal_held_out_outcomes_read": False,
    })
    atomic_json(manifest_path, {
        "schema_version": "transit-h2oplus-v14-terminal-taper-branch-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "root": str(ROOT),
        "file_count": len(manifest_records),
        "total_bytes": sum(row["bytes"] for row in manifest_records),
        "aggregate_sha256": aggregate(manifest_records),
        "files": manifest_records,
        "formal_held_out_outcomes_read": False,
    })
    atomic_json(closeout_path, {
        "schema_version": "transit-h2oplus-v14-terminal-taper-branch-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "artifact_integrity_passed": True,
        "exact_seed_matched_v12_10k_handoffs_passed": True,
        "scientific_stability_passed_for_all_15_branches": all_stable,
        "controls": controls,
        "sync": descriptor(sync_path),
        "scheduler_evidence": descriptor(scheduler_path),
        "audit": descriptor(audit_path),
        "file_manifest": descriptor(manifest_path),
        "branch_tasks": [row["task_id"] for row in audit_rows],
        "training_run_count": 15,
        "development_evaluation_gate_open": all_stable,
        "development_evaluation_submitted": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
    })
    print(json.dumps({"audit": descriptor(audit_path), "manifest": descriptor(manifest_path), "closeout": descriptor(closeout_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
