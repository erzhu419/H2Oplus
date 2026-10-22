#!/usr/bin/env python3
"""Audit and seal all 25 v12-v2 branches and the 30-run training gate."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from build_h2oplus_v12_staged_release_prefix_closeout_v2 import (
    CONTROL as PREFIX_CONTROL,
    FATAL_TEXT,
    REPRO,
    ROOT,
    STABILITY,
    STUDY_ROOT,
    actor_digest,
    aggregate,
    atomic_json,
    checkpoint_state_sha256,
    descriptor,
    finite,
    load_json,
    sha256,
    verify_descriptor,
)


BRANCH_ROOT = STUDY_ROOT / "branches_v2"
PREFIX_ROOT = STUDY_ROOT / "prefix_v2"
EVIDENCE_ROOT = STUDY_ROOT / "evidence_branches_v2"
CONTROL = {
    **PREFIX_CONTROL,
    "prefix_closeout": (
        REPRO / "h2oplus_v12_staged_release_prefix_closeout_v2.json",
        "8bb303dbe038089422ba88f28bf06cd47a672b71f929d1c14306679b943d0ef1",
    ),
    "prefix_verification": (
        REPRO / "h2oplus_v12_staged_release_prefix_closeout_verification_v2.json",
        "b13b7f0827326f52c6c9f512a7385e71e5abf40a8f15df0e70d5ec26126242e7",
    ),
    "branch_submission": (
        REPRO / "h2oplus_v12_staged_release_branch_submission_v2.json",
        "e5c2b52d8529f24f28ad670d673645884c271f3b670da6c3dfb003087efaf7ca",
    ),
}
VARIANT_FLAGS = {
    "abrupt-control": (False, False, False),
    "actor-ramp": (True, False, False),
    "temperature-ramp": (False, True, False),
    "actor-anchor": (False, False, True),
    "staged-combined": (True, True, True),
}
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
    "v12_branch_materialization.json",
    "v12_branch_source/latest.pt",
    "v12_branch_source/runtime_epoch000010.pkl",
    "v12_checkpoints/online_events_000025000.pt",
    "v12_export_summary.json",
    "v12_policies/online_events_000025000.pt",
    "v12_policies/policy_manifest.json",
}


def read_metrics(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != count or not finite(rows):
        raise ValueError(f"metrics integrity failed: {path}")
    return rows


def exact_file_contract(run: Path) -> list[Path]:
    paths = sorted(path for path in run.rglob("*") if path.is_file())
    relative = {path.relative_to(run).as_posix() for path in paths}
    resources = {name for name in relative if name.startswith("resources_resume_") and name.endswith(".csv")}
    if len(resources) != 1 or relative - resources != REQUIRED_FIXED_FILES:
        raise ValueError(f"v12 branch file contract drifted: {run}")
    return paths


def stability_and_mechanism(
    rows: list[dict[str, Any]], variant: str
) -> dict[str, Any]:
    actor_ramp, temperature_ramp, actor_anchor = VARIANT_FLAGS[variant]
    violations = []
    critics, holds, actions, actor_gradients = [], [], [], []
    actor_multipliers, temperature_multipliers, anchor_fractions = [], [], []
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

        common = {
            "ratio/v10_unit_weights": 1.0,
            "v10/unit_ratio_weights": 1.0,
            "v10/offline_replay_fraction": 0.5,
            "v10/online_replay_fraction": 0.5,
            "v11/offline_replay_fraction": 0.5,
            "v11/unit_ratio_weights": 1.0,
        }
        if any(float(train.get(key, math.nan)) != value for key, value in common.items()):
            raise ValueError(f"fixed replay/ratio mechanism drifted for {variant} epoch {epoch}")
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
                "v12/staged_release_active": 1.0,
            }
        if any(float(train.get(key, math.nan)) != value for key, value in expected.items()):
            raise ValueError(f"staged-release mechanism drifted for {variant} epoch {epoch}")
        if epoch > 10:
            actor_value = float(train["v12/actor_lr_multiplier"])
            temperature_value = float(train["v12/temperature_lr_multiplier"])
            anchor_value = float(train["v12/actor_anchor_offline_fraction"])
            actor_multipliers.append(actor_value)
            temperature_multipliers.append(temperature_value)
            anchor_fractions.append(anchor_value)
            if actor_ramp:
                if not 0.1 <= actor_value <= 1.0:
                    raise ValueError(f"actor release ramp out of range: {variant} epoch {epoch}")
            elif actor_value != 1.0:
                raise ValueError(f"unexpected actor ramp: {variant} epoch {epoch}")
            if temperature_ramp:
                if not 0.1 <= temperature_value <= 1.0:
                    raise ValueError(f"temperature release ramp out of range: {variant} epoch {epoch}")
            elif temperature_value != 1.0:
                raise ValueError(f"unexpected temperature ramp: {variant} epoch {epoch}")
            expected_anchor = 26.0 / 128.0 if actor_anchor else 0.0
            if anchor_value != expected_anchor:
                raise ValueError(f"actor anchor drifted: {variant} epoch {epoch}")

    for values, active, name in (
        (actor_multipliers, actor_ramp, "actor"),
        (temperature_multipliers, temperature_ramp, "temperature"),
    ):
        if active and (
            any(after + 1e-12 < before for before, after in zip(values, values[1:]))
            or values[-1] != 1.0
            or values[0] >= 1.0
        ):
            raise ValueError(f"{name} release ramp sequence drifted for {variant}")
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
        "temperature_lr_multiplier_first_after_release": temperature_multipliers[0],
        "temperature_lr_multiplier_final": temperature_multipliers[-1],
        "actor_anchor_offline_fraction_final": anchor_fractions[-1],
    }


def verify_policy(
    run: Path,
    entry: dict[str, Any],
    job: dict[str, Any],
    admission_sha: str,
    source_map: dict[str, str],
) -> dict[str, str]:
    actor = verify_descriptor(run, {key: entry[key] for key in ("path", "bytes", "sha256")})
    checkpoint = verify_descriptor(run, entry["model_checkpoint"])
    if entry["training_checkpoint_sha256"] != entry["model_checkpoint"]["sha256"]:
        raise ValueError("policy/checkpoint binding drifted")
    policy_payload = torch.load(actor, map_location="cpu", weights_only=False)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = policy_payload["extra"]
    actor_sha = actor_digest(policy_payload["actor"])
    checkpoint_actor_sha = actor_digest(checkpoint_payload["actor"])
    temperature_sha = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
    actor_ramp, temperature_ramp, actor_anchor = VARIANT_FLAGS[job["variant"]]
    intervention = extra.get("v12_intervention", {})
    expected = {
        "run_signature": job["run_signature"],
        "method": "h2oplus",
        "variant": job["variant"],
        "stage": "branch",
        "training_seed": job["seed"],
        "total_updates": 70000,
        "v12_online_events_nominal": 25000,
        "v12_online_events_actual": entry["online_events_actual"],
        "v12_learner_update_steps": 70000,
        "v12_ratio_optimizer_steps": 6000,
        "v12_actor_tensor_sha256": entry["actor_tensor_sha256"],
        "v12_temperature_tensor_sha256": entry["temperature_tensor_sha256"],
        "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if any(extra.get(key) != value for key, value in expected.items()):
        raise ValueError(f"v12 policy extra drifted: {actor}")
    if (
        actor_sha != entry["actor_tensor_sha256"]
        or checkpoint_actor_sha != actor_sha
        or temperature_sha != entry["temperature_tensor_sha256"]
        or extra.get("v12_source_sha256") != source_map
        or extra.get("v12_protocol", {}).get("sha256") != CONTROL["protocol"][1]
        or extra.get("training_admission", {}).get("sha256") != admission_sha
        or extra.get("selection_rule")
        != "all preregistered v12 online-event milestones; no evaluation outcome selects a checkpoint"
        or intervention.get("variant") != job["variant"]
        or intervention.get("actor_learning_rate_ramp") is not actor_ramp
        or intervention.get("temperature_learning_rate_ramp") is not temperature_ramp
        or intervention.get("actor_state_anchor") is not actor_anchor
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
        or intervention.get("actor_and_temperature_hold_through_online_events") != 10000
        or intervention.get("release_ramp_online_events") != [10000, 15000]
    ):
        raise ValueError(f"v12 policy tensor/provenance drifted: {actor}")
    return {"actor": actor_sha, "temperature": temperature_sha}


def audit_run(
    job: dict[str, Any],
    sync_row: dict[str, Any],
    scheduler_row: dict[str, Any],
    submission_row: dict[str, Any],
    prefix_audit: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = BRANCH_ROOT / job["name"]
    paths = exact_file_contract(run)
    current_inventory = [
        {
            "path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    if current_inventory != sync_row["files"]:
        raise ValueError(f"post-sync file drift detected: {job['name']}")
    for path in paths:
        if path.suffix == ".json" and not finite(load_json(path)):
            raise ValueError(f"nonfinite JSON: {path}")
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not finite(values):
                raise ValueError(f"nonfinite JSONL: {path}")

    metadata = load_json(run / "metadata.json")
    summary = load_json(run / "summary.json")
    policies = load_json(run / "v12_policies/policy_manifest.json")
    export = load_json(run / "v12_export_summary.json")
    materialization = load_json(run / "v12_branch_materialization.json")
    rows = read_metrics(run / "metrics.jsonl", 40)
    prefix_run = PREFIX_ROOT / job["prefix"]["name"]
    prefix_rows = read_metrics(prefix_run / "metrics.jsonl", 10)
    prefix_handoff_path = prefix_run / "v12_handoff/manifest.json"
    prefix_handoff = load_json(prefix_handoff_path)
    if rows[:10] != prefix_rows:
        raise ValueError(f"shared-prefix history drifted: {job['name']}")

    admission_path = REPRO / "h2oplus_v12_staged_release_training_admissions_v2" / f"{job['name']}.json"
    admission = load_json(admission_path)
    admission_sha = sha256(admission_path)
    source_map = admission["v12_source_sha256"]
    if any(sha256(ROOT / relative) != digest for relative, digest in source_map.items()):
        raise ValueError("local v12 source differs from frozen admission")
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
        or not str(args.get("resume", "")).endswith(f"/{job['name']}/checkpoints/latest.pt")
        or metadata.get("mdp_version") != "v7"
        or metadata.get("training_admission", {}).get("sha256") != admission_sha
        or metadata.get("training_admission", {}).get("selected_arguments") != job
        or metadata.get("training_admission", {}).get("v12_source_sha256") != source_map
        or metadata.get("source_sha256") != admission["training_source_sha256"]
        or admission.get("selected_arguments") != job
        or admission.get("passed") is not True
        or admission.get("formal_held_out_outcomes_read") is not False
        or runtime.get("cc_environment")
        != "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
        or not runtime.get("cpath_environment", "").endswith(
            "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
        )
        or runtime.get("device") != "cuda:0"
        or runtime.get("cuda_visible_devices") not in {"0", "1"}
    ):
        raise ValueError(f"metadata/admission/runtime drifted: {job['name']}")

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
        or summary["final_policy"]["source_training_checkpoint_sha256"]
        != summary["checkpoint"]["sha256"]
        or [int(row["epoch"]) for row in rows] != list(range(1, 41))
        or int(rows[-1]["actual_cumulative_simulator_decision_events"]) != events
        or int(rows[-1]["total_updates"]) != 70000
    ):
        raise ValueError(f"summary/event sequence drifted: {job['name']}")

    prefix = prefix_audit[job["seed"]]
    source_checkpoint = prefix_run / "v12_handoff/checkpoints/latest.pt"
    branch_initial = run / "v12_branch_source/latest.pt"
    branch_runtime = run / "v12_branch_source/runtime_epoch000010.pkl"
    if (
        materialization.get("schema_version") != "transit-h2oplus-v12-branch-materialization-v2"
        or materialization.get("passed") is not True
        or materialization.get("formal_held_out_outcomes_read") is not False
        or materialization.get("run_name") != job["name"]
        or materialization.get("run_signature") != job["run_signature"]
        or materialization.get("seed") != job["seed"]
        or materialization.get("variant") != job["variant"]
        or materialization.get("protocol_sha256") != CONTROL["protocol"][1]
        or materialization.get("handoff_manifest_sha256") != sha256(prefix_handoff_path)
        or materialization.get("source_checkpoint_sha256") != sha256(source_checkpoint)
        or materialization.get("branch_initial_checkpoint_sha256") != sha256(branch_initial)
        or materialization.get("branch_initial_runtime_sha256") != sha256(branch_runtime)
        or materialization.get("checkpoint_state_sha256") != checkpoint_state_sha256(branch_initial)
        or materialization.get("checkpoint_state_sha256") != prefix["checkpoint_state_sha256"]
        or materialization.get("actor_tensor_sha256") != prefix["actor_tensor_sha256"]
        or materialization.get("temperature_tensor_sha256") != prefix["temperature_tensor_sha256"]
        or prefix_handoff.get("prefix_run_signature") != job["prefix"]["run_signature"]
        or prefix_handoff.get("seed") != job["seed"]
    ):
        raise ValueError(f"seed-matched handoff drifted: {job['name']}")

    entries = policies.get("entries", [])
    if (
        policies.get("schema_version") != "transit-h2oplus-v12-staged-release-policy-manifest-v2"
        or policies.get("run_signature") != job["run_signature"]
        or policies.get("seed") != job["seed"]
        or policies.get("variant") != job["variant"]
        or policies.get("stage") != "branch"
        or policies.get("protocol_sha256") != CONTROL["protocol"][1]
        or len(entries) != 1
    ):
        raise ValueError(f"policy manifest drifted: {job['name']}")
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
        raise ValueError(f"25k milestone accounting drifted: {job['name']}")
    final_tensors = verify_policy(run, entry, job, admission_sha, source_map)
    if (
        export.get("schema_version") != "transit-h2oplus-v12-staged-release-export-summary-v2"
        or export.get("passed") is not True
        or export.get("formal_held_out_outcomes_read") is not False
        or export.get("stage") != "branch"
        or export.get("variant") != job["variant"]
        or export.get("summary_sha256") != sha256(run / "summary.json")
        or export.get("policy_manifest", {}).get("sha256")
        != sha256(run / "v12_policies/policy_manifest.json")
        or export.get("policy_manifest", {}).get("entries") != 1
        or export.get("branch_materialization", {}).get("sha256")
        != sha256(run / "v12_branch_materialization.json")
        or export.get("shared_prefix_handoff") is not None
    ):
        raise ValueError(f"export summary drifted: {job['name']}")

    stability = stability_and_mechanism(rows, job["variant"])
    texts = (run / "run.log").read_text(errors="replace") + "\n" + Path(
        scheduler_row["scheduler_log"]["path"]
    ).read_text(errors="replace")
    if FATAL_TEXT.search(texts):
        raise ValueError(f"fatal text found: {job['name']}")
    if (
        submission_row["run_signature"] != job["run_signature"]
        or scheduler_row["signature"] != submission_row["scheduler_signature"]
        or scheduler_row["node"] != submission_row["require_node"]
        or scheduler_row["backend_returncode"] != 0
        or scheduler_row["token_matches"] is not True
        or scheduler_row["residual_canonical_process_count"] != 0
    ):
        raise ValueError(f"scheduler lineage drifted: {job['name']}")

    record = {
        "run_name": job["name"],
        "variant": job["variant"],
        "seed": job["seed"],
        "run_signature": job["run_signature"],
        "task_id": submission_row["task_id"],
        "node": submission_row["require_node"],
        "file_count": len(paths),
        "simulator_decision_events": events,
        "event_underfill": underfill,
        "final_blocked_simultaneous_batch_size": blocked,
        "learner_update_steps": 70000,
        "ratio_optimizer_steps": 6000,
        "actor_tensor_sha256": final_tensors["actor"],
        "temperature_tensor_sha256": final_tensors["temperature"],
        "checkpoint_state_sha256_at_handoff": prefix["checkpoint_state_sha256"],
        "exact_seed_matched_10k_handoff": True,
        "scientific_stability": stability,
        "artifact_integrity_passed": True,
        "formal_held_out_outcomes_read": False,
    }
    return record, [descriptor(path, ROOT) for path in paths]


def main() -> int:
    controls = {}
    payloads = {}
    for name, (path, expected) in CONTROL.items():
        if sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"frozen v12 control drifted: {name}")
        controls[name] = descriptor(path)
        payloads[name] = load_json(path)
    if (
        payloads["prefix_closeout"].get("status") != "passed"
        or payloads["prefix_closeout"].get("artifact_integrity_passed") is not True
        or payloads["prefix_closeout"].get("scientific_stability_passed_for_all_five_prefix_runs") is not True
        or payloads["prefix_verification"].get("status") != "passed"
        or payloads["prefix_verification"].get("artifact_integrity_passed") is not True
    ):
        raise ValueError("sealed v12 prefix dependency is invalid")

    plan = payloads["plan"]
    protocol = payloads["protocol"]
    jobs = [job for job in plan["jobs"] if job.get("stage") == "branch"]
    matrix = {(job["variant"], int(job["seed"])) for job in jobs}
    expected_matrix = {(variant, seed) for variant in VARIANT_FLAGS for seed in range(6201, 6206)}
    if (
        plan.get("jobs") != protocol.get("training_jobs")
        or len(jobs) != 25
        or matrix != expected_matrix
        or payloads["branch_submission"].get("development_evaluation_submitted") is not False
        or payloads["branch_submission"].get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("frozen v12 branch matrix drifted")

    sync_path = EVIDENCE_ROOT / "branch_sync_v2.json"
    scheduler_path = EVIDENCE_ROOT / "scheduler_evidence_branches_v2.json"
    sync = load_json(sync_path)
    scheduler = load_json(scheduler_path)
    if (
        sync.get("schema_version") != "transit-h2oplus-v12-staged-release-branch-sync-v2"
        or sync.get("passed") is not True
        or sync.get("total_runs") != 25
        or scheduler.get("schema_version")
        != "transit-h2oplus-v12-staged-release-branch-scheduler-evidence-v2"
        or scheduler.get("passed") is not True
        or scheduler.get("duplicate_or_retry_count") != 0
        or len(scheduler.get("tasks", [])) != 25
    ):
        raise ValueError("v12 branch scheduler/sync evidence is incomplete")
    sync_by_run = {row["run_name"]: row for row in sync["runs"]}
    scheduler_by_run = {row["run_name"]: row for row in scheduler["tasks"]}
    submission_by_run = {row["name"]: row for row in payloads["branch_submission"]["tasks"]}
    expected_names = {job["name"] for job in jobs}
    if set(sync_by_run) != expected_names or set(scheduler_by_run) != expected_names or set(submission_by_run) != expected_names:
        raise ValueError("synchronized branch set differs from frozen matrix")

    prefix_audit_payload = load_json(STUDY_ROOT / "prefix_audit_v2.json")
    prefix_audit = {int(row["seed"]): row for row in prefix_audit_payload["runs"]}
    audit_rows = []
    manifest_records = []
    for job in sorted(jobs, key=lambda item: (item["variant"], int(item["seed"]))):
        row, records = audit_run(
            job,
            sync_by_run[job["name"]],
            scheduler_by_run[job["name"]],
            submission_by_run[job["name"]],
            prefix_audit,
        )
        audit_rows.append(row)
        manifest_records.extend(records)

    all_branch_stable = all(row["scientific_stability"]["passed"] for row in audit_rows)
    all_prefix_stable = prefix_audit_payload.get("all_prefix_stability_passed") is True
    all_30_stable = all_branch_stable and all_prefix_stable
    manifest_records.sort(key=lambda record: record["path"])
    audit_path = STUDY_ROOT / "branch_audit_v2.json"
    manifest_path = STUDY_ROOT / "branch_file_manifest_v2.json"
    closeout_path = REPRO / "h2oplus_v12_staged_release_branch_closeout_v2.json"
    atomic_json(
        audit_path,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-branch-audit-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "artifact_integrity_passed": True,
            "runs": audit_rows,
            "exact_run_count": len(audit_rows),
            "variant_counts": {variant: sum(row["variant"] == variant for row in audit_rows) for variant in VARIANT_FLAGS},
            "all_seed_matched_10k_handoffs_exact": all(row["exact_seed_matched_10k_handoff"] for row in audit_rows),
            "all_branch_stability_passed": all_branch_stable,
            "stability_violation_count": sum(len(row["scientific_stability"]["violations"]) for row in audit_rows),
            "formal_held_out_outcomes_read": False,
        },
    )
    atomic_json(
        manifest_path,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-branch-file-manifest-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "root": str(ROOT),
            "file_count": len(manifest_records),
            "total_bytes": sum(record["bytes"] for record in manifest_records),
            "aggregate_sha256": aggregate(manifest_records),
            "files": manifest_records,
            "formal_held_out_outcomes_read": False,
        },
    )
    atomic_json(
        closeout_path,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-branch-closeout-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "artifact_integrity_passed": True,
            "exact_seed_matched_10k_handoffs_passed": True,
            "scientific_stability_passed_for_all_25_branches": all_branch_stable,
            "scientific_stability_passed_for_all_30_training_runs": all_30_stable,
            "controls": controls,
            "sync": descriptor(sync_path),
            "scheduler_evidence": descriptor(scheduler_path),
            "audit": descriptor(audit_path),
            "file_manifest": descriptor(manifest_path),
            "prefix_tasks": payloads["prefix_closeout"]["prefix_tasks"],
            "branch_tasks": [row["task_id"] for row in audit_rows],
            "training_run_count": 30,
            "development_evaluation_gate_open": all_30_stable,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "formal_held_out_outcomes_read": False,
        },
    )
    print(json.dumps({"audit": descriptor(audit_path), "manifest": descriptor(manifest_path), "closeout": descriptor(closeout_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
