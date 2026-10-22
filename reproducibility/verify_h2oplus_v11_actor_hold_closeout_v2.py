#!/usr/bin/env python3
"""Producer-independent verification of the v11-v2 training closeout."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v11_actor_hold_results"
TRAINING_ROOT = RESULTS / "training_v2"
EVIDENCE_ROOT = RESULTS / "evidence_training_v2"
PLAN_PATH = REPRO / "h2oplus_v11_actor_hold_training_plan_v2.json"
SUBMISSION_PATH = REPRO / "h2oplus_v11_actor_hold_submission_v2.json"
MANIFEST_PATH = RESULTS / "training_file_manifest_v2.json"
AUDIT_PATH = RESULTS / "training_audit_v2.json"
CLOSEOUT_PATH = RESULTS / "training_closeout_v2.json"
OUTPUT_PATH = RESULTS / "training_closeout_verification_v2.json"
EXPECTED_CONTROLS = {
    REPRO / "h2oplus_v11_actor_hold_rationale_v1.json": (
        "456ddcb73c5d78a8f0bf90f3e8ddfbbfda81c8c45c76d505f4d309eead35605b"
    ),
    REPRO / "h2oplus_v11_actor_hold_registration_v2.json": (
        "caab9c2c3aa7ee2ce977c13403b415be9b0debc2d7c88fc6b7a12c51b3e9efb5"
    ),
    REPRO / "h2oplus_v11_actor_hold_authorization_v2.json": (
        "8dc0ca71f8c45abc3e5e86af789281e4e43a882533ba4202ca567f00d87281aa"
    ),
    REPRO / "h2oplus_v11_actor_hold_environment_v2.json": (
        "0f005b94d5ecd361d7e10bac363be8dac5e6c9cdad7cbaf3f063744d280871de"
    ),
    PLAN_PATH: "61aa351f7e36c9abd3f4822c20be17c7e5b5343cd822e031d0ede001889e7234",
    REPRO / "h2oplus_v11_actor_hold_protocol_v2.json": (
        "c5b162541c6ce2476bb0f57f6dc936de13b85b5e1e6ede77b533a5cf9a4dfced"
    ),
    REPRO / "h2oplus_v11_actor_hold_freeze_v2.json": (
        "095754d08f14b6e2eb84baadfa6ce1a8d4100ebac5b92f34891b2d1e01afed04"
    ),
    REPRO / "h2oplus_v11_actor_hold_correction_v2.json": (
        "8f1defa7d756f6336d26af0b1fe8ebf87db39f81554c89891fc2fbaaf17d1258"
    ),
    REPRO
    / "deployments/h2oplus_v11_actor_hold_pilot_v2/workspace/deployment_manifest_v11.json": (
        "f408050ed84575728ea6f77aad7837856d932d39afcc612635a143c5b3a8b801"
    ),
    REPRO / "h2oplus_v11_actor_hold_deployment_verification_local_v2.json": (
        "673fb49a3620d456b76a69aa50fe255847e198a8d5d02b5ec979d16e6aeed598"
    ),
    REPRO / "h2oplus_v11_actor_hold_preflight_closeout_v2.json": (
        "60ae6a967780bc8f3fcc49c64693d9a9aeef7536786459e8241099698fd842f1"
    ),
    SUBMISSION_PATH: (
        "d6aafb8afa387392285fcd6f8ef5dc030012d8bc04e6040934248225c3d6dc39"
    ),
}
THRESHOLDS = {
    "maximum_critic_grad_norm": 1_000_000.0,
    "minimum_hold_probability": 0.01,
    "maximum_hold_probability": 0.99,
    "maximum_absolute_positive_action_mean": 0.98,
}
PROTOCOL_SHA256 = (
    "c5b162541c6ce2476bb0f57f6dc936de13b85b5e1e6ede77b533a5cf9a4dfced"
)
EXPECTED_INTERVENTION = {
    "schema_version": "transit-h2oplus-v11-actor-hold-variant-v2",
    "variant": "actor-hold-online-only",
    "warmup_events": 5_000,
    "actor_and_temperature_hold_through_online_events": 10_000,
    "actor_and_temperature_hold_online_critic_updates": 5_000,
    "actor_state_source_after_release": "online-only",
    "offline_replay_schedule": [[0, 0.5], [25_000, 0.5]],
    "online_update_order": "v7-actor-before-critic",
    "unit_ratio_weights": True,
    "ratio_classifiers_continue_for_diagnostics": True,
    "checkpoint_selection_by_behavior": False,
}
MILESTONES = {
    0: (0, 50_000, 0),
    5_000: (0, 50_000, 0),
    10_000: (10, 55_000, 2_250),
    25_000: (40, 70_000, 6_000),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


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


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
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


def event_evidence(
    rows: list[dict[str, Any]], phase_metrics: dict[str, Any], nominal: int
) -> dict[str, Any]:
    if nominal == 0:
        return {
            "actual": 0,
            "deviation": 0,
            "maximum_simultaneous_decisions": 0,
            "hard_event_cap_reached": False,
            "blocked_simultaneous_batch_size": 0,
        }
    epoch = MILESTONES[nominal][0]
    warmup = phase_metrics.get("warmup", {})
    committed = rows[:epoch]
    if nominal == 5_000:
        actual = int(warmup["events"])
        latest = warmup
    else:
        if len(committed) != epoch:
            raise ValueError(f"missing metrics for milestone {nominal}")
        actual = int(committed[-1]["actual_cumulative_simulator_decision_events"])
        latest = committed[-1]["rollout"]
    maximum_batch = max(
        [int(warmup.get("maximum_simultaneous_decisions", 0))]
        + [
            int(row.get("rollout", {}).get("maximum_simultaneous_decisions", 0))
            for row in committed
        ]
    )
    deviation = actual - nominal
    if deviation > 0 and (maximum_batch <= 0 or deviation >= maximum_batch):
        raise ValueError(f"unbounded overshoot at milestone {nominal}")
    blocked = int(latest.get("blocked_simultaneous_batch_size", 0))
    hard_cap = bool(latest.get("hard_event_cap_reached", False))
    if nominal == 25_000:
        underfill = nominal - actual
        if actual > nominal:
            raise ValueError("final hard cap exceeded")
        if underfill and (not hard_cap or blocked <= underfill):
            raise ValueError("final underfill lacks atomic-batch evidence")
    return {
        "actual": actual,
        "deviation": deviation,
        "maximum_simultaneous_decisions": maximum_batch,
        "hard_event_cap_reached": hard_cap,
        "blocked_simultaneous_batch_size": blocked,
    }


def recompute_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 40 or [int(row["epoch"]) for row in rows] != list(range(1, 41)):
        raise ValueError("metrics must contain the exact epoch sequence 1..40")
    violations: list[dict[str, Any]] = []
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        held = epoch <= 10
        expected_updates = epoch * 500
        if (
            float(train["train/online_critic_update"]) != expected_updates
            or float(train["train/update"]) != 50_000 + expected_updates
            or train.get("v11/actor_hold_active") != float(held)
            or train.get("v11/online_only_actor_active") != float(not held)
            or train.get("v11/actor_hold_critic_updates") != 5_000.0
            or train.get("v11/actor_release_events") != 10_000.0
            or train.get("v11/unit_ratio_weights") != 1.0
            or train.get("v11/offline_replay_fraction") != 0.5
            or train.get("v10/offline_replay_fraction") != 0.5
            or train.get("v10/online_replay_fraction") != 0.5
            or train.get("v10/delayed_replay_active") != 0.0
            or train.get("v10/delayed_update_order_active") != 0.0
            or train.get("v10/delayed_actor_states_active") != float(not held)
            or train.get("ratio/v10_unit_weights") != 1.0
        ):
            raise ValueError(f"mechanism drift at epoch {epoch}")
        if held:
            if (
                train.get("train/actor_updated") != 0.0
                or float(train["train/actor_grad_norm"]) != 0.0
                or float(train["train/actor_loss"]) != 0.0
                or float(train["train/alpha"]) != 1.0
                or "train/hold_probability" in train
                or "train/positive_action_mean" in train
            ):
                raise ValueError(f"actor hold drift at epoch {epoch}")
        else:
            if train.get("train/actor_updated") != 1.0:
                raise ValueError(f"released actor did not update at epoch {epoch}")
            hold = float(train["train/hold_probability"])
            action = float(train["train/positive_action_mean"])
            if not THRESHOLDS["minimum_hold_probability"] <= hold <= THRESHOLDS[
                "maximum_hold_probability"
            ]:
                violations.append(
                    {"epoch": epoch, "metric": "hold_probability", "value": hold}
                )
            if abs(action) >= THRESHOLDS["maximum_absolute_positive_action_mean"]:
                violations.append(
                    {
                        "epoch": epoch,
                        "metric": "positive_action_mean",
                        "value": action,
                    }
                )
        critic = float(train["train/critic_grad_norm"])
        if critic > THRESHOLDS["maximum_critic_grad_norm"]:
            violations.append(
                {"epoch": epoch, "metric": "critic_grad_norm", "value": critic}
            )

        cap = 5_000 + expected_updates
        actual = int(row["actual_cumulative_simulator_decision_events"])
        maximum_batch = int(row["rollout"]["maximum_simultaneous_decisions"])
        blocked = int(row["rollout"]["blocked_simultaneous_batch_size"])
        if int(row["cumulative_simulator_decision_event_cap"]) != cap:
            raise ValueError(f"event cap drift at epoch {epoch}")
        if actual > cap and actual - cap >= maximum_batch:
            raise ValueError(f"atomic overshoot drift at epoch {epoch}")
        if actual < cap and not cap - actual < blocked <= maximum_batch:
            raise ValueError(f"atomic underfill drift at epoch {epoch}")
    return violations


def main() -> int:
    errors: list[str] = []
    for path, expected in EXPECTED_CONTROLS.items():
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen control drifted: {path}")

    plan = load(PLAN_PATH)
    submission = load(SUBMISSION_PATH)
    manifest = load(MANIFEST_PATH)
    audit = load(AUDIT_PATH)
    closeout = load(CLOSEOUT_PATH)
    jobs = plan.get("jobs", [])
    expected_names = {job["name"] for job in jobs}
    task_by_name = {task["name"]: task for task in submission.get("tasks", [])}
    actual_names = {
        path.name for path in TRAINING_ROOT.iterdir() if path.is_dir()
    }
    if (
        len(jobs) != 5
        or {job.get("seed") for job in jobs} != set(range(6201, 6206))
        or {job.get("variant") for job in jobs} != {"actor-hold-online-only"}
        or set(task_by_name) != expected_names
        or actual_names != expected_names
    ):
        errors.append("training matrix or canonical directory set drifted")

    records = inventory(TRAINING_ROOT)
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-training-file-manifest-v2"
        or manifest.get("root") != str(TRAINING_ROOT.resolve())
        or manifest.get("file_count") != len(records)
        or manifest.get("total_bytes") != sum(row["bytes"] for row in records)
        or manifest.get("aggregate_sha256") != aggregate(records)
        or manifest.get("files") != records
    ):
        errors.append("training file manifest differs from independent inventory")

    scheduler = load(EVIDENCE_ROOT / "scheduler_evidence_training_v2.json")
    sync = load(EVIDENCE_ROOT / "source_sync_training_v2.json")
    scheduler_rows = {row["run_name"]: row for row in scheduler.get("tasks", [])}
    sync_rows = {row["run_name"]: row for row in sync.get("runs", [])}
    if (
        scheduler.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-scheduler-evidence-v2"
        or sync.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-training-sync-v2"
        or sync.get("passed") is not True
        or sync.get("formal_held_out_outcomes_read") is not False
        or set(scheduler_rows) != expected_names
        or set(sync_rows) != expected_names
    ):
        errors.append("scheduler or synchronization coverage drifted")

    recomputed: dict[str, dict[str, Any]] = {}
    failure_runs: list[str] = []
    all_violations: list[dict[str, Any]] = []
    for job in jobs:
        name = job["name"]
        run = TRAINING_ROOT / name
        try:
            summary = load(run / "summary.json")
            metadata = load(run / "metadata.json")
            policies = load(run / "v11_policies/policy_manifest.json")
            export = load(run / "v11_export_summary.json")
            phase_metrics = load(run / "phase_metrics.json")
            admission_path = (
                REPRO
                / "h2oplus_v11_actor_hold_training_admissions_v2"
                / f"{name}.json"
            )
            admission = load(admission_path)
            rows = [
                json.loads(line)
                for line in (run / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if not finite([summary, metadata, policies, export, phase_metrics, rows]):
                raise ValueError("nonfinite artifact")
            milestone_evidence = {
                nominal: event_evidence(rows, phase_metrics, nominal)
                for nominal in MILESTONES
            }
            final_events = milestone_evidence[25_000]["actual"]
            final_underfill = -milestone_evidence[25_000]["deviation"]
            if (
                summary.get("completed_epochs") != 40
                or summary.get("total_updates") != 70_000
                or summary.get("run_signature") != job["run_signature"]
                or summary.get("simulator_decision_event_cap") != 25_000
                or summary.get("simulator_decision_events") != final_events
                or summary.get("simulator_decision_event_underfill")
                != final_underfill
                or metadata.get("arguments", {}).get("run_signature")
                != job["run_signature"]
                or metadata.get("training_admission", {}).get("selected_arguments")
                != job
                or metadata.get("training_admission", {}).get("sha256")
                != sha256(admission_path)
                or metadata.get("training_admission", {}).get(
                    "v11_protocol_sha256"
                )
                != PROTOCOL_SHA256
                or metadata.get("training_admission", {}).get("v11_source_sha256")
                != plan["v11_source_sha256"]
                or admission.get("selected_arguments") != job
                or admission.get("v11_protocol", {}).get("sha256")
                != PROTOCOL_SHA256
                or admission.get("v11_source_sha256") != plan["v11_source_sha256"]
                or export.get("passed") is not True
                or export.get("formal_held_out_outcomes_read") is not False
                or export.get("checkpoint_selection_by_behavior") is not False
                or export.get("zero_5k_10k_actor_and_temperature_identical")
                is not True
                or export.get("summary_sha256") != sha256(run / "summary.json")
                or export.get("policy_manifest", {}).get("sha256")
                != sha256(run / "v11_policies/policy_manifest.json")
            ):
                raise ValueError("summary, metadata, or export contract mismatch")

            entries = policies.get("entries", [])
            if (
                policies.get("schema_version")
                != "transit-h2oplus-v11-actor-hold-policy-manifest-v2"
                or policies.get("run_signature") != job["run_signature"]
                or policies.get("seed") != job["seed"]
                or policies.get("variant") != "actor-hold-online-only"
                or policies.get("protocol_sha256") != PROTOCOL_SHA256
                or [entry.get("online_events_nominal") for entry in entries]
                != list(MILESTONES)
            ):
                raise ValueError("policy milestone set drifted")
            actor_hashes: dict[int, str] = {}
            temperature_hashes: dict[int, str] = {}
            for entry in entries:
                nominal = int(entry["online_events_nominal"])
                epoch, updates, ratio_updates = MILESTONES[nominal]
                policy_path = (run / entry["path"]).resolve(strict=True)
                checkpoint_entry = entry["model_checkpoint"]
                checkpoint_path = (run / checkpoint_entry["path"]).resolve(
                    strict=True
                )
                if (
                    not policy_path.is_relative_to(run.resolve())
                    or not checkpoint_path.is_relative_to(run.resolve())
                    or sha256(policy_path) != entry["sha256"]
                    or policy_path.stat().st_size != entry["bytes"]
                    or sha256(checkpoint_path) != checkpoint_entry["sha256"]
                    or checkpoint_path.stat().st_size != checkpoint_entry["bytes"]
                    or entry.get("online_events_actual")
                    != milestone_evidence[nominal]["actual"]
                    or entry.get("event_budget_deviation")
                    != milestone_evidence[nominal]["deviation"]
                    or entry.get("epoch") != epoch
                    or entry.get("learner_update_steps") != updates
                    or entry.get("ratio_optimizer_steps") != ratio_updates
                ):
                    raise ValueError(f"milestone descriptor drifted at {nominal}")
                policy_payload = torch.load(
                    policy_path, map_location="cpu", weights_only=False
                )
                checkpoint_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                policy_extra = policy_payload.get("extra", {})
                checkpoint_extra = checkpoint_payload.get("extra", {})
                actor_hash = tensor_digest(policy_payload["actor"])
                checkpoint_actor_hash = tensor_digest(checkpoint_payload["actor"])
                temperature_hash = tensor_digest(
                    {"log_alpha": checkpoint_payload["log_alpha"]}
                )
                if (
                    actor_hash != entry["actor_tensor_sha256"]
                    or checkpoint_actor_hash != actor_hash
                    or temperature_hash != entry["temperature_tensor_sha256"]
                    or int(checkpoint_payload["total_updates"]) != updates
                    or policy_extra.get("v11_event_cap_evidence")
                    != milestone_evidence[nominal]
                    or policy_extra.get("v11_intervention")
                    != EXPECTED_INTERVENTION
                    or policy_extra.get("v11_source_sha256")
                    != plan["v11_source_sha256"]
                    or policy_extra.get("v11_protocol", {}).get("sha256")
                    != PROTOCOL_SHA256
                    or policy_extra.get("training_admission", {}).get("sha256")
                    != sha256(admission_path)
                    or policy_extra.get("formal_held_out_outcomes_read") is not False
                    or policy_extra.get("selection_rule")
                    != "all preregistered v11 online-event milestones; no evaluation outcome selects a checkpoint"
                    or checkpoint_extra.get("v11_intervention")
                    != EXPECTED_INTERVENTION
                    or checkpoint_extra.get("v11_online_events_nominal") != nominal
                    or checkpoint_extra.get("v11_online_events_actual")
                    != milestone_evidence[nominal]["actual"]
                    or checkpoint_extra.get("v11_learner_update_steps") != updates
                ):
                    raise ValueError(f"tensor hash drifted at {nominal}")
                actor_hashes[nominal] = actor_hash
                temperature_hashes[nominal] = temperature_hash
            held = {0, 5_000, 10_000}
            if (
                len({actor_hashes[event] for event in held}) != 1
                or len({temperature_hashes[event] for event in held}) != 1
            ):
                raise ValueError("held actor or temperature hashes diverged")
            latest_path = run / "checkpoints/latest.pt"
            final_actor_path = run / "checkpoints/final_actor.pt"
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            final_actor = torch.load(
                final_actor_path, map_location="cpu", weights_only=False
            )
            if (
                summary.get("checkpoint", {}).get("sha256")
                != sha256(latest_path)
                or summary.get("final_policy", {}).get("sha256")
                != sha256(final_actor_path)
                or summary.get("final_policy", {}).get(
                    "source_training_checkpoint_sha256"
                )
                != sha256(latest_path)
                or int(latest["total_updates"]) != 70_000
                or tensor_digest(latest["actor"]) != actor_hashes[25_000]
                or tensor_digest(final_actor["actor"]) != actor_hashes[25_000]
            ):
                raise ValueError("final actor or runtime checkpoint drifted")

            local_records = inventory(run)
            scheduler_row = scheduler_rows[name]
            sync_row = sync_rows[name]
            task = task_by_name[name]
            backend_fields = (
                EVIDENCE_ROOT / f"{task['task_id']}.backend"
            ).read_text().split()
            if (
                sync_row.get("files") != local_records
                or sync_row.get("file_count") != len(local_records)
                or sync_row.get("total_bytes")
                != sum(record["bytes"] for record in local_records)
                or scheduler_row.get("task_id") != task["task_id"]
                or scheduler_row.get("signature") != task["scheduler_signature"]
                or scheduler_row.get("status") != "done"
                or scheduler_row.get("exit_code") != 0
                or scheduler_row.get("backend_returncode") != 0
                or scheduler_row.get("token_matches") is not True
                or scheduler_row.get("residual_canonical_process_count") != 0
                or scheduler_row.get("declared_cpu_cores") != 6
                or scheduler_row.get("declared_ram_mb") != 32_768
                or scheduler_row.get("declared_vram_mb") != 8_192
                or len(backend_fields) != 3
                or backend_fields[0] != "0"
                or backend_fields[2] != scheduler_row.get("backend_token")
            ):
                raise ValueError("sync or scheduler evidence mismatch")

            violations = recompute_history(rows)
            if violations:
                failure_runs.append(name)
                all_violations.extend({"run": name, **item} for item in violations)
            recomputed[name] = {
                "events": final_events,
                "underfill": final_underfill,
                "violations": violations,
            }
        except Exception as error:
            errors.append(f"{name}: {error}")

    audit_runs = {row["run"]: row for row in audit.get("runs", [])}
    if set(audit_runs) != expected_names:
        errors.append("audit run set differs from registered matrix")
    for name, result in recomputed.items():
        recorded = audit_runs.get(name, {}).get("scientific_stability", {})
        if (
            audit_runs.get(name, {}).get("events") != result["events"]
            or audit_runs.get(name, {}).get("underfill") != result["underfill"]
            or recorded.get("violations") != result["violations"]
            or recorded.get("passed") != (not result["violations"])
        ):
            errors.append(f"audit differs from independent recomputation: {name}")

    recorded_stability = audit.get("scientific_stability", {})
    if (
        audit.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-training-audit-v2"
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("development_evaluation_release_gate_open") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or recorded_stability.get("passed") != (not failure_runs)
        or recorded_stability.get("all_5_training_runs_gate_passed")
        != (not failure_runs)
        or recorded_stability.get("failure_runs") != failure_runs
        or recorded_stability.get("violations") != all_violations
        or recorded_stability.get("artifact_integrity_implication") != "none"
    ):
        errors.append("scientific stability disclosure differs from recomputation")

    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-training-closeout-v2"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed") != (not failure_runs)
        or closeout.get("all_5_training_runs_stability_gate_passed")
        != (not failure_runs)
        or closeout.get("stability_failure_runs") != failure_runs
        or closeout.get("training_runs") != 5
        or closeout.get("policy_milestones") != 20
        or closeout.get("held_actor_and_temperature_hash_contracts") != 5
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("formal_confirmation_eligibility_created") is not False
        or closeout.get("development_evaluation_submitted") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or closeout.get("training_audit", {}).get("sha256") != sha256(AUDIT_PATH)
        or closeout.get("training_file_manifest", {}).get("sha256")
        != sha256(MANIFEST_PATH)
    ):
        errors.append("training closeout contract drifted")

    report = {
        "schema_version": (
            "transit-h2oplus-v11-actor-hold-training-closeout-verification-v2"
        ),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "artifact_integrity_verified": not any(
            "artifact" in error or "manifest" in error or "sync" in error
            for error in errors
        ),
        "scientific_stability_verified_separately": not any(
            "stability" in error for error in errors
        ),
        "verified_training_runs": len(recomputed),
        "verified_stability_failure_runs": failure_runs,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_file_manifest": descriptor(MANIFEST_PATH),
        "training_audit": descriptor(AUDIT_PATH),
        "training_closeout": descriptor(CLOSEOUT_PATH),
    }
    atomic_json(OUTPUT_PATH, report)
    print(
        json.dumps(
            {**report, "verification": descriptor(OUTPUT_PATH)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
