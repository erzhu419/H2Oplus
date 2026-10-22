#!/usr/bin/env python3
"""Independently audit and seal all 25 v10-v4 delayed-transition branches."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from build_h2oplus_v10_delayed_prefix_closeout_v4 import (
    CC,
    CPATH_SUFFIX,
    FATAL_TEXT,
    REPRO,
    ROOT,
    STABILITY,
    actor_digest,
    atomic_json,
    checkpoint_state_sha256,
    descriptor,
    finite,
    load_json,
    read_metrics,
    records_aggregate,
    sha256,
)


BRANCH_ROOT = REPRO / "h2oplus_v10_delayed_results/branches_v4"
PREFIX_ROOT = REPRO / "h2oplus_v10_delayed_results/prefix_v4"
EVIDENCE_ROOT = REPRO / "h2oplus_v10_delayed_results/evidence_branches_v4"
OUTPUT_ROOT = REPRO / "h2oplus_v10_delayed_results"
CONTROL = {
    "plan": (
        REPRO / "h2oplus_v10_delayed_training_plan_v4.json",
        "20d87242072b4aab8332d25e2d2d27af986818339b6c63672d781607e7a0b829",
    ),
    "protocol": (
        REPRO / "h2oplus_v10_delayed_protocol_v4.json",
        "83a45e7d06ec30bdaba3631d7d8c6854e7609b8dcff9c46aa735d9a7944daaf3",
    ),
    "registration": (
        REPRO / "h2oplus_v10_delayed_registration_v4.json",
        "59fbc0c7f1148fe74c4c8a0c8af0879c7900753520b5b112e0526eb3c05acf56",
    ),
    "authorization": (
        REPRO / "h2oplus_v10_delayed_authorization_v4.json",
        "7aa6e68db01503dc9e1c3bbfbce1639dd08890e5fb46fec2cda347ade6942741",
    ),
    "environment": (
        REPRO / "h2oplus_v10_delayed_environment_v4.json",
        "ee3de3b3990e35c579f5fcfdbafe16168a83f5011f9e5b5ce44b59cea5365bb4",
    ),
    "freeze": (
        REPRO / "h2oplus_v10_delayed_freeze_v4.json",
        "3776bf088ba7c7cf3c2e65fb8407afb2adb9c8fbcc2266b7781bfd8d1b04cb64",
    ),
    "correction": (
        REPRO / "h2oplus_v10_delayed_correction_v4.json",
        "2bf2a6d707283ae694e18ce796a209810415a2e079ca6b023c88a7b2e6857163",
    ),
    "prefix_closeout": (
        OUTPUT_ROOT / "prefix_closeout_v4.json",
        "6a05ddc9ef1a3a07605851360d3733d6442f6aa27e4fb2b65d69f4b2a25951ad",
    ),
    "prefix_verification": (
        OUTPUT_ROOT / "prefix_closeout_verification_v4.json",
        "1d1d8b592485d24038cf363b233a4228c96d4a156c00804c6eadfbe3b0906c65",
    ),
    "branch_submission": (
        REPRO / "h2oplus_v10_delayed_branch_submission_v4.json",
        "9a55e9c7ad208cf1d4f69f569c61734ff355bd7b9cfadeb81d864d977e371648",
    ),
}
VARIANT_FLAGS = {
    "unit-control": (0.0, 0.0, 0.0),
    "delayed-replay": (1.0, 0.0, 0.0),
    "delayed-actor": (0.0, 1.0, 0.0),
    "delayed-order": (0.0, 0.0, 1.0),
    "delayed-combined": (1.0, 1.0, 1.0),
}


def verify_actor(
    run: Path,
    entry: dict[str, Any],
    job: dict[str, Any],
    admission_sha: str,
    source_sha: dict[str, str],
    summary: dict[str, Any],
) -> str:
    actor = run / entry["path"]
    checkpoint = run / entry["model_checkpoint"]["path"]
    if (
        actor.stat().st_size != entry["bytes"]
        or sha256(actor) != entry["sha256"]
        or checkpoint.stat().st_size != entry["model_checkpoint"]["bytes"]
        or sha256(checkpoint) != entry["model_checkpoint"]["sha256"]
        or entry["training_checkpoint_sha256"] != entry["model_checkpoint"]["sha256"]
    ):
        raise ValueError(f"policy/checkpoint descriptor drifted: {actor}")
    payload = torch.load(actor, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    digest = actor_digest(payload["actor"])
    expected = {
        "run_signature": job["run_signature"],
        "method": "h2oplus",
        "variant": job["variant"],
        "stage": "branch",
        "training_seed": job["seed"],
        "total_updates": 70000,
        "v10_online_events_nominal": 25000,
        "v10_online_events_actual": summary["simulator_decision_events"],
        "v10_learner_update_steps": 70000,
        "v10_ratio_optimizer_steps": 6000,
        "v10_actor_tensor_sha256": entry["actor_tensor_sha256"],
        "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {key: (extra.get(key), value) for key, value in expected.items() if extra.get(key) != value}
    intervention = extra.get("v10_intervention", {})
    replay, actor_states, order = VARIANT_FLAGS[job["variant"]]
    if (
        digest != entry["actor_tensor_sha256"]
        or mismatches
        or extra.get("training_admission", {}).get("sha256") != admission_sha
        or extra.get("v10_protocol", {}).get("sha256") != CONTROL["protocol"][1]
        or extra.get("v10_source_sha256") != source_sha
        or intervention.get("variant") != job["variant"]
        or intervention.get("unit_ratio_weights") is not True
        or bool(intervention.get("delayed_replay")) != bool(replay)
        or bool(intervention.get("delayed_actor_states")) != bool(actor_states)
        or bool(intervention.get("delayed_update_order")) != bool(order)
        or extra.get("selection_rule")
        != "all preregistered v10 online-event milestones; no evaluation outcome selects a checkpoint"
    ):
        raise ValueError(f"actor provenance drifted for {actor}: {mismatches}")
    return digest


def audit_history(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    violations = []
    critic, hold, action, actor_grad, replay_values = [], [], [], [], []
    replay_flag, actor_flag, order_flag = VARIANT_FLAGS[variant]
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        values = {
            "critic_grad_norm": float(train["train/critic_grad_norm"]),
            "hold_probability": float(train["train/hold_probability"]),
            "positive_action_mean": float(train["train/positive_action_mean"]),
            "actor_grad_norm": float(train["train/actor_grad_norm"]),
        }
        critic.append(values["critic_grad_norm"])
        hold.append(values["hold_probability"])
        action.append(values["positive_action_mean"])
        actor_grad.append(values["actor_grad_norm"])
        if values["critic_grad_norm"] > STABILITY["maximum_critic_grad_norm"]:
            violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": values["critic_grad_norm"]})
        if not STABILITY["minimum_hold_probability"] <= values["hold_probability"] <= STABILITY["maximum_hold_probability"]:
            violations.append({"epoch": epoch, "metric": "hold_probability", "value": values["hold_probability"]})
        if abs(values["positive_action_mean"]) >= STABILITY["maximum_absolute_positive_action_mean"]:
            violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": values["positive_action_mean"]})

        expected_transition = 0.0 if epoch <= 10 else 1.0
        expected_flags = (0.0, 0.0, 0.0) if epoch <= 10 else (replay_flag, actor_flag, order_flag)
        observed_flags = (
            train.get("v10/delayed_replay_active"),
            train.get("v10/delayed_actor_states_active"),
            train.get("v10/delayed_update_order_active"),
        )
        offline = float(train["v10/offline_replay_fraction"])
        online = float(train["v10/online_replay_fraction"])
        if (
            train.get("ratio/v10_unit_weights") != 1.0
            or train.get("v10/unit_ratio_weights") != 1.0
            or train.get("v10/transition_active") != expected_transition
            or observed_flags != expected_flags
            or train.get("train/actor_updated") != 1.0
            or not math.isclose(offline + online, 1.0, abs_tol=1e-12)
        ):
            raise ValueError(f"v10 mechanism drifted for {variant} at epoch {epoch}")
        if epoch <= 10 or replay_flag == 0.0:
            if offline != 0.5 or online != 0.5:
                raise ValueError(f"fixed replay fraction drifted for {variant} at epoch {epoch}")
        else:
            if not 0.25 <= offline <= 0.5:
                raise ValueError(f"delayed replay fraction drifted for {variant} at epoch {epoch}")
            replay_values.append(offline)
    if replay_flag:
        if (
            any(after > before + 1e-12 for before, after in zip(replay_values, replay_values[1:]))
            or not 0.25 <= replay_values[-1] <= 0.251
        ):
            raise ValueError(f"delayed replay ramp drifted for {variant}")
    return {
        "passed": not violations,
        "thresholds": STABILITY,
        "violations": violations,
        "maximum_critic_grad_norm": max(critic),
        "minimum_hold_probability": min(hold),
        "maximum_hold_probability": max(hold),
        "maximum_absolute_positive_action_mean": max(abs(value) for value in action),
        "maximum_actor_grad_norm": max(actor_grad),
        "final_offline_replay_fraction": replay_values[-1] if replay_values else 0.5,
    }


def audit_run(
    job: dict[str, Any],
    source_sha: dict[str, str],
    task_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run = BRANCH_ROOT / job["name"]
    summary_path = run / "summary.json"
    metadata_path = run / "metadata.json"
    policies_path = run / "v10_policies/policy_manifest.json"
    export_path = run / "v10_export_summary.json"
    materialization_path = run / "v10_branch_materialization.json"
    summary = load_json(summary_path)
    metadata = load_json(metadata_path)
    policies = load_json(policies_path)
    export = load_json(export_path)
    materialization = load_json(materialization_path)
    if not finite([summary, metadata, policies, export, materialization]):
        raise ValueError(f"nonfinite JSON artifact in {run}")

    cap = 25000
    events = int(summary.get("simulator_decision_events", -1))
    underfill = int(summary.get("simulator_decision_event_underfill", -1))
    blocked = int(summary.get("final_blocked_simultaneous_batch_size", -1))
    maximum_batch = int(summary.get("maximum_simultaneous_decisions", -1))
    latest = run / "checkpoints/latest.pt"
    final_actor = run / "checkpoints/final_actor.pt"
    if (
        summary.get("schema_version") != "transit-training-summary-v8"
        or summary.get("method") != "h2oplus"
        or summary.get("seed") != job["seed"]
        or summary.get("run_signature") != job["run_signature"]
        or summary.get("completed_epochs") != 40
        or summary.get("start_epoch") != 10
        or summary.get("total_updates") != 70000
        or summary.get("simulator_decision_event_cap") != cap
        or not 0 <= events <= cap
        or underfill != cap - events
        or maximum_batch <= 0
        or (underfill == 0 and blocked != 0)
        or (underfill > 0 and not underfill < blocked <= maximum_batch)
        or summary.get("checkpoint", {}).get("sha256") != sha256(latest)
        or summary.get("final_policy", {}).get("sha256") != sha256(final_actor)
        or summary.get("final_policy", {}).get("source_training_checkpoint_sha256") != sha256(latest)
    ):
        raise ValueError(f"summary/event-cap contract drifted for {run}")

    admission_path = REPRO / "h2oplus_v10_delayed_training_admissions_v4" / f"{job['name']}.json"
    admission_sha = sha256(admission_path)
    admission_payload = load_json(admission_path)
    admission = metadata.get("training_admission", {})
    runtime = metadata.get("hardware", {}).get("runtime_fingerprint", {}).get("host", {})
    arguments = metadata.get("arguments", {})
    initializer = metadata.get("offline_initializer_consumed", {})
    if (
        admission_payload.get("schema_version") != "transit-training-admission-v10-delayed-development-v4"
        or admission_payload.get("passed") is not True
        or admission_payload.get("selected_arguments") != job
        or admission_payload.get("v10_protocol", {}).get("sha256") != CONTROL["protocol"][1]
        or admission_payload.get("v10_source_sha256") != source_sha
        or admission.get("sha256") != admission_sha
        or admission.get("selected_arguments") != job
        or admission.get("v10_protocol_sha256") != CONTROL["protocol"][1]
        or admission.get("v10_source_sha256") != source_sha
        or arguments.get("run_name") != job["name"]
        or arguments.get("run_signature") != job["run_signature"]
        or arguments.get("seed") != job["seed"]
        or arguments.get("epochs") != 40
        or arguments.get("warmup_events") != 5000
        or arguments.get("ratio_pretrain_updates") != 1000
        or arguments.get("online_fraction") != 0.5
        or runtime.get("cc_environment") != CC
        or not str(runtime.get("cpath_environment", "")).endswith(CPATH_SUFFIX)
        or runtime.get("cuda_visible_devices") not in {"0", "1"}
        or initializer.get("sha256") != job["offline_initializer"]["sha256"]
        or initializer.get("source_run_signature") != job["offline_initializer"]["source_run_signature"]
        or initializer.get("v10_reuse_authorized") is not True
        or metadata.get("source_sha256") != admission_payload.get("training_source_sha256")
    ):
        raise ValueError(f"metadata/admission/environment drifted for {run}")

    prefix_run = PREFIX_ROOT / job["prefix"]["name"]
    prefix_handoff_path = prefix_run / "v10_handoff/manifest.json"
    prefix_handoff = load_json(prefix_handoff_path)
    prefix_rows = read_metrics(prefix_run / "metrics.jsonl", 10)
    source_checkpoint = prefix_run / "v10_handoff/checkpoints/latest.pt"
    branch_initial = run / "v10_branch_source/latest.pt"
    branch_runtime = run / "v10_branch_source/runtime_epoch000010.pkl"
    if (
        materialization.get("schema_version") != "transit-h2oplus-v10-branch-materialization-v1"
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
        or materialization.get("checkpoint_state_sha256") != prefix_handoff["checkpoint_state_sha256"]
        or materialization.get("actor_tensor_sha256") != prefix_handoff["actor_tensor_sha256"]
        or prefix_handoff.get("prefix_run_name") != job["prefix"]["name"]
        or prefix_handoff.get("prefix_run_signature") != job["prefix"]["run_signature"]
        or prefix_handoff.get("seed") != job["seed"]
    ):
        raise ValueError(f"seed-matched branch materialization drifted for {run}")

    entries = policies.get("entries", [])
    if (
        policies.get("schema_version") != "transit-h2oplus-v10-delayed-policy-manifest-v1"
        or policies.get("run_signature") != job["run_signature"]
        or policies.get("seed") != job["seed"]
        or policies.get("variant") != job["variant"]
        or policies.get("stage") != "branch"
        or policies.get("protocol_sha256") != CONTROL["protocol"][1]
        or len(entries) != 1
    ):
        raise ValueError(f"policy manifest drifted for {run}")
    entry = entries[0]
    if (
        entry.get("online_events_nominal") != 25000
        or entry.get("online_events_actual") != events
        or entry.get("event_budget_deviation") != events - 25000
        or entry.get("learner_update_steps") != 70000
        or entry.get("ratio_optimizer_steps") != 6000
        or entry.get("epoch") != 40
    ):
        raise ValueError(f"25k milestone accounting drifted for {run}")
    verify_actor(run, entry, job, admission_sha, source_sha, summary)
    if (
        export.get("schema_version") != "transit-h2oplus-v10-delayed-export-summary-v1"
        or export.get("passed") is not True
        or export.get("formal_held_out_outcomes_read") is not False
        or export.get("stage") != "branch"
        or export.get("variant") != job["variant"]
        or export.get("summary_sha256") != sha256(summary_path)
        or export.get("policy_manifest", {}).get("sha256") != sha256(policies_path)
        or export.get("policy_manifest", {}).get("entries") != 1
        or export.get("branch_materialization", {}).get("sha256") != sha256(materialization_path)
        or export.get("shared_prefix_handoff") is not None
    ):
        raise ValueError(f"export summary drifted for {run}")

    rows = read_metrics(run / "metrics.jsonl", 40)
    if [int(row["epoch"]) for row in rows] != list(range(1, 41)) or rows[:10] != prefix_rows:
        raise ValueError(f"epoch sequence or shared-prefix history drifted for {run}")
    stability = audit_history(rows, job["variant"])
    task = task_by_name[job["name"]]
    texts = "\n".join(
        path.read_text(errors="replace")
        for path in (run / "run.log", EVIDENCE_ROOT / f"sched_{task['task_id']}.log")
    )
    if FATAL_TEXT.search(texts):
        raise ValueError(f"fatal text marker found for {run}")

    return {
        "run": job["name"],
        "variant": job["variant"],
        "seed": job["seed"],
        "task_id": task["task_id"],
        "events": events,
        "event_cap": cap,
        "underfill": underfill,
        "blocked_atomic_batch": blocked,
        "maximum_simultaneous_decisions": maximum_batch,
        "completed_epochs": 40,
        "total_updates": 70000,
        "ratio_optimizer_steps": 6000,
        "unit_ratio_weights": True,
        "exact_seed_matched_10k_handoff": True,
        "scientific_stability": stability,
        "summary": descriptor(summary_path, BRANCH_ROOT),
        "metadata": descriptor(metadata_path, BRANCH_ROOT),
        "policy_manifest": descriptor(policies_path, BRANCH_ROOT),
        "export_summary": descriptor(export_path, BRANCH_ROOT),
        "branch_materialization": descriptor(materialization_path, BRANCH_ROOT),
        "prefix_handoff_manifest": descriptor(prefix_handoff_path),
        "training_admission": descriptor(admission_path),
    }


def main() -> int:
    controls: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in CONTROL.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen control drifted: {path}")
        controls[name] = descriptor(path)
        payloads[name] = load_json(path)
    if any(
        payloads[name].get("formal_held_out_outcomes_read") is not False
        for name in ("plan", "protocol", "registration", "freeze", "correction", "branch_submission")
    ) or payloads["authorization"].get("formal_held_out_access_authorized") is not False:
        raise ValueError("v10 development scope drifted")
    if (
        payloads["prefix_closeout"].get("artifact_integrity_passed") is not True
        or payloads["prefix_closeout"].get("scientific_stability_passed") is not True
        or payloads["prefix_verification"].get("passed") is not True
    ):
        raise ValueError("sealed prefix dependency is not valid")

    plan = payloads["plan"]
    protocol = payloads["protocol"]
    branch_jobs = [job for job in plan["jobs"] if job["stage"] == "branch"]
    if (
        plan.get("schema_version") != "transit-h2oplus-v10-delayed-training-plan-v4"
        or protocol.get("schema_version") != "transit-h2oplus-v10-delayed-protocol-v4"
        or plan["jobs"] != protocol["training_jobs"]
        or len(branch_jobs) != 25
        or {job["variant"] for job in branch_jobs} != set(VARIANT_FLAGS)
        or any(sum(job["seed"] == seed for job in branch_jobs) != 5 for seed in range(6201, 6206))
    ):
        raise ValueError("v10 plan/protocol branch matrix drifted")

    scheduler_path = EVIDENCE_ROOT / "scheduler_evidence_branches_v4.json"
    sync_path = EVIDENCE_ROOT / "source_sync_branches_v4.json"
    scheduler = load_json(scheduler_path)
    sync = load_json(sync_path)
    expected_tasks = {item["task_id"]: item for item in payloads["branch_submission"]["tasks"]}
    task_by_name = {item["name"]: item for item in expected_tasks.values()}
    observed = {item["task_id"]: item for item in scheduler.get("tasks", [])}
    if (
        scheduler.get("schema_version") != "transit-h2oplus-v10-delayed-branch-scheduler-evidence-v4"
        or set(observed) != set(expected_tasks)
        or any(
            row.get("status") != "done"
            or row.get("exit_code") != 0
            or row.get("backend_returncode") != 0
            or row.get("token_matches") is not True
            or row.get("residual_canonical_process_count") != 0
            or row.get("signature") != expected_tasks[task_id]["scheduler_signature"]
            for task_id, row in observed.items()
        )
    ):
        raise ValueError("scheduler/backend evidence drifted")
    if (
        sync.get("schema_version") != "transit-h2oplus-v10-delayed-branch-sync-v4"
        or sync.get("passed") is not True
        or sync.get("formal_held_out_outcomes_read") is not False
        or len(sync.get("runs", [])) != 25
        or {row["run_name"] for row in sync["runs"]} != {job["name"] for job in branch_jobs}
    ):
        raise ValueError("branch synchronization evidence drifted")

    runs = [audit_run(job, plan["v10_source_sha256"], task_by_name) for job in branch_jobs]
    failures = [row for row in runs if not row["scientific_stability"]["passed"]]
    files = sorted(path for path in BRANCH_ROOT.rglob("*") if path.is_file())
    records = [descriptor(path, BRANCH_ROOT) for path in files]
    manifest_path = OUTPUT_ROOT / "branch_file_manifest_v4.json"
    manifest = {
        "schema_version": "transit-h2oplus-v10-delayed-branch-file-manifest-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(BRANCH_ROOT.resolve()),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "aggregate_sha256": records_aggregate(records),
        "files": records,
    }
    atomic_json(manifest_path, manifest)

    audit_path = OUTPUT_ROOT / "branch_audit_v4.json"
    audit = {
        "schema_version": "transit-h2oplus-v10-delayed-branch-audit-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "artifact_integrity": {
            "passed": True,
            "branch_runs": 25,
            "variants": 5,
            "seeds": 5,
            "milestones": 25,
            "exact_seed_matched_handoffs": 25,
        },
        "scientific_stability": {
            "passed": not failures,
            "all_30_training_runs_gate_passed": not failures,
            "failure_runs": [row["run"] for row in failures],
            "violations": [
                {"run": row["run"], **violation}
                for row in failures
                for violation in row["scientific_stability"]["violations"]
            ],
            "artifact_integrity_implication": "none",
            "training_rewards_support_held_out_effect_claim": False,
        },
        "development_evaluation_release_gate_open": True,
        "formal_confirmation_eligibility_created": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "controls": controls,
        "scheduler_evidence": descriptor(scheduler_path),
        "sync_evidence": descriptor(sync_path),
        "runs": runs,
        "file_manifest": descriptor(manifest_path),
    }
    atomic_json(audit_path, audit)

    closeout_path = OUTPUT_ROOT / "branch_closeout_v4.json"
    closeout = {
        "schema_version": "transit-h2oplus-v10-delayed-branch-closeout-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity_passed": True,
        "scientific_stability_passed": not failures,
        "all_30_training_runs_stability_gate_passed": not failures,
        "stability_failure_runs": [row["run"] for row in failures],
        "branch_runs": 25,
        "branch_milestones": 25,
        "exact_seed_matched_handoffs": 25,
        "development_evaluation_release_gate_open": True,
        "formal_confirmation_eligibility_created": False,
        "development_evaluation_submitted": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "branch_audit": descriptor(audit_path),
        "branch_file_manifest": descriptor(manifest_path),
    }
    atomic_json(closeout_path, closeout)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "files": len(records),
                "aggregate_sha256": manifest["aggregate_sha256"],
                "stability_failure_runs": [row["run"] for row in failures],
                "branch_audit_sha256": sha256(audit_path),
                "branch_manifest_sha256": sha256(manifest_path),
                "branch_closeout_sha256": sha256(closeout_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
