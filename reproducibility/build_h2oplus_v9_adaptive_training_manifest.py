#!/usr/bin/env python3
"""Audit and seal the corrected H2O+ v9 adaptive development training stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "transit-h2oplus-v9-adaptive-training-plan-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v9-adaptive-protocol-v2"
POLICY_SCHEMA = "transit-h2oplus-v9-adaptive-policy-manifest-v1"
EXPORT_SCHEMA = "transit-h2oplus-v9-adaptive-export-summary-v1"
AUDIT_SCHEMA = "transit-h2oplus-v9-adaptive-training-audit-v1"
MANIFEST_SCHEMA = "transit-h2oplus-v9-adaptive-training-file-manifest-v1"
CLOSEOUT_SCHEMA = "transit-h2oplus-v9-adaptive-training-closeout-v1"
SYNC_SCHEMA = "transit-h2oplus-v9-adaptive-source-sync-v1"
EVIDENCE_SCHEMA = "transit-h2oplus-v9-adaptive-scheduler-evidence-v1"
SELECTION_RULE = (
    "all preregistered v9 online-event milestones; no evaluation outcome "
    "selects a checkpoint"
)
CC = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
CPATH_SUFFIX = "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
STABILITY_THRESHOLDS = {
    "maximum_critic_grad_norm": 1_000_000.0,
    "minimum_hold_probability": 0.01,
    "maximum_hold_probability": 0.99,
    "maximum_absolute_positive_action_mean": 0.98,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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


def descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            resolved.relative_to(base.resolve()).as_posix()
            if base is not None
            else str(resolved)
        ),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def records_aggregate(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_bound(path: Path, expected: str, schema: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected:
        raise ValueError(f"identity drift for {resolved}: {actual}")
    payload = load_json(resolved)
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"schema drift for {resolved}")
    return payload


def resolve_run_path(run: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        matches = list(run.rglob(candidate.name))
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot localize {value!r} under {run}")
        return matches[0].resolve()
    return (run / candidate).resolve(strict=True)


def read_jsonl(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not finite(row):
                raise ValueError(f"nonfinite metrics row {number}: {path}")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(f"metrics row count drifted for {path}: {len(rows)}")
    return rows


def verify_policy_artifact(
    path: Path,
    *,
    job: dict[str, Any],
    entry: dict[str, Any],
    admission_sha256: str,
    protocol_sha256: str,
) -> None:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    extra = artifact.get("extra", {})
    expected = {
        "method": "h2oplus",
        "variant": job["variant"],
        "training_seed": int(job["seed"]),
        "run_signature": job["run_signature"],
        "v9_online_events_nominal": int(entry["online_events_nominal"]),
        "v9_online_events_actual": int(entry["online_events_actual"]),
        "v9_learner_update_steps": int(entry["learner_update_steps"]),
        "v9_ratio_optimizer_steps": int(entry["ratio_optimizer_steps"]),
        "v9_actor_tensor_sha256": entry["actor_tensor_sha256"],
        "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    if (
        mismatches
        or extra.get("training_admission", {}).get("sha256") != admission_sha256
        or extra.get("v9_protocol", {}).get("sha256") != protocol_sha256
        or extra.get("selection_rule") != SELECTION_RULE
    ):
        raise ValueError(f"policy provenance drifted for {path}: {mismatches}")


def verify_scheduler_evidence(
    submission: dict[str, Any], evidence_path: Path
) -> list[dict[str, Any]]:
    payload = load_json(evidence_path)
    if payload.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("scheduler evidence schema drifted")
    observed = {item["task_id"]: item for item in payload.get("tasks", [])}
    verified = []
    for expected in submission.get("tasks", []):
        task_id = expected["task_id"]
        task = observed.get(task_id)
        if (
            task is None
            or task.get("status") != "done"
            or task.get("exit_code") != 0
            or task.get("node") != expected["require_node"]
            or task.get("signature") != expected["scheduler_signature"]
            or task.get("token_matches") is not True
            or task.get("backend_returncode") != 0
            or task.get("residual_process_count") != 0
        ):
            raise ValueError(f"scheduler completion evidence drifted for {task_id}")
        exit_path = Path(task["backend_exit"]["path"]).resolve(strict=True)
        if (
            exit_path.stat().st_size != task["backend_exit"]["bytes"]
            or sha256(exit_path) != task["backend_exit"]["sha256"]
        ):
            raise ValueError(f"backend exit descriptor drifted for {task_id}")
        verified.append(task)
    if len(verified) != 9 or set(observed) != {item["task_id"] for item in submission["tasks"]}:
        raise ValueError("scheduler evidence does not cover exactly nine runs")
    return verified


def verify_sync_evidence(
    paths: list[Path], submission: dict[str, Any]
) -> list[dict[str, Any]]:
    expected: dict[str, set[str]] = {}
    for task in submission["tasks"]:
        expected.setdefault(task["require_node"], set()).add(task["name"])
    verified = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        payload = load_json(resolved)
        host = payload.get("source_host")
        runs = payload.get("run_names", [])
        if (
            payload.get("schema_version") != SYNC_SCHEMA
            or payload.get("passed") is not True
            or payload.get("checksum_mode") != "rsync-content-checksum-dry-run"
            or payload.get("differences") != []
            or host not in expected
            or host in seen
            or set(runs) != expected[host]
            or len(runs) != len(set(runs))
            or payload.get("source_file_count") != payload.get("local_file_count")
            or payload.get("source_total_bytes") != payload.get("local_total_bytes")
        ):
            raise ValueError(f"source-sync evidence drifted: {resolved}")
        seen.add(host)
        verified.append({
            "source_host": host,
            "run_names": sorted(runs),
            "source_file_count": payload["source_file_count"],
            "source_total_bytes": payload["source_total_bytes"],
            "evidence": descriptor(resolved),
        })
    if seen != set(expected):
        raise ValueError("source-sync evidence host coverage drifted")
    return sorted(verified, key=lambda item: item["source_host"])


def stability_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    critic, hold, action, actor = [], [], [], []
    for row in rows:
        epoch = int(row["epoch"])
        train = row.get("train", {})
        if "train/critic_grad_norm" in train:
            value = float(train["train/critic_grad_norm"])
            critic.append(value)
            if value > STABILITY_THRESHOLDS["maximum_critic_grad_norm"]:
                violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": value})
        if "train/hold_probability" in train:
            value = float(train["train/hold_probability"])
            hold.append(value)
            if not (
                STABILITY_THRESHOLDS["minimum_hold_probability"]
                <= value
                <= STABILITY_THRESHOLDS["maximum_hold_probability"]
            ):
                violations.append({"epoch": epoch, "metric": "hold_probability", "value": value})
        if "train/positive_action_mean" in train:
            value = float(train["train/positive_action_mean"])
            action.append(value)
            if abs(value) >= STABILITY_THRESHOLDS["maximum_absolute_positive_action_mean"]:
                violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": value})
        if "train/actor_grad_norm" in train:
            actor.append(float(train["train/actor_grad_norm"]))
    return {
        "passed": not violations,
        "thresholds": STABILITY_THRESHOLDS,
        "violations": violations,
        "maximum_critic_grad_norm": max(critic),
        "minimum_hold_probability": min(hold),
        "maximum_hold_probability": max(hold),
        "maximum_absolute_positive_action_mean": max(abs(value) for value in action),
        "maximum_actor_grad_norm": max(actor),
    }


def ratio_gate_audit(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    validation = [
        (int(row["epoch"]), row.get("ratio_validation", {}))
        for row in rows
        if row.get("ratio_validation", {}).get("ratio/validation_available") == 1.0
    ]
    gate_rows = [
        int(row["epoch"])
        for row in rows
        if row.get("train", {}).get("ratio/v9_gate_open") == 1.0
    ]
    pass_rows = [
        epoch
        for epoch, metrics in validation
        if metrics.get("ratio/validation_v9_gate_pass") == 1.0
    ]
    relevant = variant in {"ratio-gated", "combined"}
    result = {
        "reliability_gate_configured": relevant,
        "validation_rows": len(validation),
        "first_validation_epoch": validation[0][0] if validation else None,
        "validation_pass_epochs": pass_rows,
        "gate_open_epochs": gate_rows,
        "gate_ever_opened": bool(gate_rows),
        "ratio_intervention_applied": bool(gate_rows) if relevant else None,
        "last_validation": validation[-1][1] if validation else None,
    }
    if relevant and not validation:
        raise ValueError(f"ratio validation never became available for {variant}")
    return result


def audit_run(
    training_root: Path,
    job: dict[str, Any],
    protocol_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = training_root / job["name"]
    if not run.is_dir():
        raise FileNotFoundError(run)
    summary_path = run / "summary.json"
    metadata_path = run / "metadata.json"
    export_path = run / "v9_export_summary.json"
    policy_manifest_path = run / "v9_policies/policy_manifest.json"
    summary = load_json(summary_path)
    metadata = load_json(metadata_path)
    export = load_json(export_path)
    policies = load_json(policy_manifest_path)
    if not all(finite(item) for item in (summary, metadata, export, policies)):
        raise ValueError(f"nonfinite JSON artifact in {run}")

    seed = int(job["seed"])
    variant = job["variant"]
    epochs = int(job["epochs"])
    cap = int(job["online_event_cap"])
    expected_updates = int(job["expected_total_updates"])
    events = int(summary.get("simulator_decision_events", -1))
    underfill = int(summary.get("simulator_decision_event_underfill", -1))
    blocked = int(summary.get("final_blocked_simultaneous_batch_size", -1))
    maximum_batch = int(summary.get("maximum_simultaneous_decisions", -1))
    if (
        summary.get("schema_version") != "transit-training-summary-v8"
        or summary.get("method") != "h2oplus"
        or summary.get("seed") != seed
        or summary.get("run_signature") != job["run_signature"]
        or summary.get("completed_epochs") != epochs
        or summary.get("total_updates") != expected_updates
        or summary.get("simulator_decision_event_cap") != cap
        or not (0 <= events <= cap)
        or underfill != cap - events
        or maximum_batch <= 0
        or (underfill == 0 and blocked != 0)
        or (underfill > 0 and not (underfill < blocked <= maximum_batch))
    ):
        raise ValueError(f"summary or final event-cap contract drifted for {run}")

    admission_path = (
        ROOT
        / "reproducibility/h2oplus_v9_adaptive_training_admissions_v2"
        / f"{job['name']}.json"
    )
    admission_sha256 = sha256(admission_path)
    admission_payload = load_json(admission_path)
    admission = metadata.get("training_admission", {})
    runtime = metadata.get("hardware", {}).get("runtime_fingerprint", {}).get("host", {})
    arguments = metadata.get("arguments", {})
    initializer = metadata.get("offline_initializer_consumed", {})
    expected_initializer = job["offline_initializer"]
    expected_sources = admission_payload.get("v9_source_sha256", {})
    if (
        admission_payload.get("schema_version")
        != "transit-training-admission-v9-adaptive-development-v2"
        or admission_payload.get("passed") is not True
        or admission_payload.get("run_name") != job["name"]
        or admission_payload.get("run_signature") != job["run_signature"]
        or admission_payload.get("variant") != variant
        or arguments.get("method") != "h2oplus"
        or arguments.get("seed") != seed
        or arguments.get("run_signature") != job["run_signature"]
        or admission.get("sha256") != admission_sha256
        or admission.get("v9_protocol_sha256") != protocol_sha256
        or admission.get("variant") != variant
        or runtime.get("cc_environment") != CC
        or not str(runtime.get("cpath_environment", "")).endswith(CPATH_SUFFIX)
        or runtime.get("cuda_visible_devices") not in {"0", "1"}
        or initializer.get("sha256") != expected_initializer["sha256"]
        or initializer.get("bytes") != expected_initializer["bytes"]
        or initializer.get("source_run_signature") != expected_initializer["source_run_signature"]
        or initializer.get("v9_reuse_authorized") is not True
        or admission.get("v9_source_sha256") != expected_sources
    ):
        raise ValueError(f"metadata/admission/environment contract drifted for {run}")

    expected_events = list(job["export_online_events"])
    entries = policies.get("entries", [])
    by_nominal = {int(item["online_events_nominal"]): item for item in entries}
    warmup = int(job["warmup_events"])
    if (
        policies.get("schema_version") != POLICY_SCHEMA
        or policies.get("method") != "h2oplus"
        or policies.get("seed") != seed
        or policies.get("variant") != variant
        or policies.get("run_signature") != job["run_signature"]
        or policies.get("protocol_sha256") != protocol_sha256
        or len(entries) != len(by_nominal)
        or sorted(by_nominal) != sorted(expected_events)
        or export.get("schema_version") != EXPORT_SCHEMA
        or export.get("passed") is not True
        or export.get("formal_held_out_outcomes_read") is not False
        or export.get("variant") != variant
        or export.get("seed") != seed
        or export.get("run_signature") != job["run_signature"]
        or export.get("protocol_sha256") != protocol_sha256
        or export.get("summary_sha256") != sha256(summary_path)
        or export.get("policy_manifest", {}).get("sha256") != sha256(policy_manifest_path)
        or export.get("zero_and_warmup_actor_identical") is not True
        or by_nominal[0]["actor_tensor_sha256"] != by_nominal[warmup]["actor_tensor_sha256"]
    ):
        raise ValueError(f"v9 export contract drifted for {run}")

    registered = {int(item["online_events"]): item for item in job["milestones"]}
    method_rows = []
    for nominal in expected_events:
        entry = by_nominal[nominal]
        actor = resolve_run_path(run, entry["path"])
        checkpoint = resolve_run_path(run, entry["model_checkpoint"]["path"])
        actual = int(entry["online_events_actual"])
        milestone = registered[nominal]
        if (
            actor.stat().st_size != int(entry["bytes"])
            or sha256(actor) != entry["sha256"]
            or checkpoint.stat().st_size != int(entry["model_checkpoint"]["bytes"])
            or sha256(checkpoint) != entry["model_checkpoint"]["sha256"]
            or entry["training_checkpoint_sha256"] != entry["model_checkpoint"]["sha256"]
            or int(entry["learner_update_steps"]) != int(milestone["learner_update_steps"])
            or int(entry["ratio_optimizer_steps"]) != int(milestone["ratio_optimizer_steps"])
            or int(entry["event_budget_deviation"]) != actual - nominal
            or (nominal not in (0, cap) and abs(actual - nominal) >= maximum_batch)
            or (nominal == cap and actual != events)
        ):
            raise ValueError(f"milestone artifact contract drifted: {actor}")
        verify_policy_artifact(
            actor,
            job=job,
            entry=entry,
            admission_sha256=admission_sha256,
            protocol_sha256=protocol_sha256,
        )
        method_rows.append({
            "name": f"h2oplus-v9-{variant}-seed{seed}-events{nominal:09d}",
            "algorithm": f"h2oplus-v9-{variant}",
            "variant": variant,
            "type": "checkpoint",
            "checkpoint": actor,
            "training_seed": seed,
            "online_events": nominal,
            "actual_online_events": actual,
            "learner_update_steps": int(entry["learner_update_steps"]),
            "ratio_optimizer_steps": int(entry["ratio_optimizer_steps"]),
            "actor_tensor_sha256": entry["actor_tensor_sha256"],
            "training_run_signature": job["run_signature"],
            "training_admission_sha256": admission_sha256,
            "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        })

    rows = read_jsonl(run / "metrics.jsonl", epochs)
    stability = stability_audit(rows)
    ratio = ratio_gate_audit(rows, variant)
    return (
        {
            "run": job["name"],
            "variant": variant,
            "seed": seed,
            "events": events,
            "event_cap": cap,
            "underfill": underfill,
            "blocked_atomic_batch": blocked,
            "maximum_simultaneous_decisions": maximum_batch,
            "completed_epochs": epochs,
            "total_updates": expected_updates,
            "summary": descriptor(summary_path, training_root),
            "metadata": descriptor(metadata_path, training_root),
            "export_summary": descriptor(export_path, training_root),
            "policy_manifest": descriptor(policy_manifest_path, training_root),
            "training_admission": descriptor(admission_path),
            "zero_and_warmup_actor_identical": True,
            "ratio_gate": ratio,
            "scientific_stability": stability,
        },
        method_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--sync-evidence", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    for name in (
        "plan", "protocol", "registration", "authorization", "freeze", "correction", "submission"
    ):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training_root = args.training_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    plan = read_bound(args.plan, args.expected_plan_sha256, PLAN_SCHEMA)
    protocol = read_bound(args.protocol, args.expected_protocol_sha256, PROTOCOL_SCHEMA)
    registration = read_bound(args.registration, args.expected_registration_sha256)
    authorization = read_bound(args.authorization, args.expected_authorization_sha256)
    freeze = read_bound(args.freeze, args.expected_freeze_sha256)
    correction = read_bound(args.correction, args.expected_correction_sha256)
    submission = read_bound(args.submission, args.expected_submission_sha256)
    if (
        protocol.get("formal_held_out_outcomes_read") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or authorization.get("formal_held_out_access_authorized") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or correction.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v9 development scope drifted")

    scheduler = verify_scheduler_evidence(
        submission, args.scheduler_evidence.expanduser().resolve(strict=True)
    )
    source_sync = verify_sync_evidence(args.sync_evidence, submission)
    jobs = plan.get("jobs", plan.get("training_jobs", []))
    if len(jobs) != 9:
        raise ValueError("v9 plan does not contain nine jobs")
    runs, policy_rows = [], []
    for job in jobs:
        run, policies = audit_run(training_root, job, args.expected_protocol_sha256)
        runs.append(run)
        policy_rows.extend(policies)
    if len(policy_rows) != 48 or len({item["name"] for item in policy_rows}) != 48:
        raise ValueError("v9 exports do not contain the registered 48 milestone aliases")

    output_root.mkdir(parents=True, exist_ok=True)
    policies_path = output_root / "training_policies_v9_adaptive.json"
    policies_payload = {
        "schema_version": "transit-h2oplus-v9-adaptive-training-policies-v1",
        "study_stage": "development",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "protocol_sha256": args.expected_protocol_sha256,
        "policies": [],
    }
    for item in sorted(policy_rows, key=lambda row: row["name"]):
        row = dict(item)
        actor = row.pop("checkpoint")
        row["checkpoint"] = os.path.relpath(actor, policies_path.parent)
        policies_payload["policies"].append(row)
    atomic_json(policies_path, policies_payload)

    files = sorted(path for path in training_root.rglob("*") if path.is_file())
    records = [descriptor(path, training_root) for path in files]
    manifest_path = output_root / "training_file_manifest.json"
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_root": str(training_root),
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "aggregate_sha256": records_aggregate(records),
        "files": records,
    }
    atomic_json(manifest_path, manifest_payload)

    stability_failures = [
        {"run": item["run"], "violations": item["scientific_stability"]["violations"]}
        for item in runs
        if not item["scientific_stability"]["passed"]
    ]
    ratio_relevant = [item for item in runs if item["variant"] in {"ratio-gated", "combined"}]
    ratio_open = [item["run"] for item in ratio_relevant if item["ratio_gate"]["gate_ever_opened"]]
    audit_path = output_root / "training_audit.json"
    audit_payload = {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {"passed": True, "training_runs": 9, "milestone_aliases": 48},
        "development_scientific_diagnostics": {
            "stability_passed": not stability_failures,
            "stability_failures": stability_failures,
            "ratio_gate_relevant_runs": len(ratio_relevant),
            "ratio_gate_open_runs": ratio_open,
            "ratio_intervention_ever_applied": bool(ratio_open),
            "intervention_attribution_available": bool(ratio_open),
            "training_rewards_support_held_out_effect_claim": False,
        },
        "formal_held_out_outcomes_read": False,
        "controls": {
            "plan_sha256": args.expected_plan_sha256,
            "protocol_sha256": args.expected_protocol_sha256,
            "registration_sha256": args.expected_registration_sha256,
            "authorization_sha256": args.expected_authorization_sha256,
            "freeze_sha256": args.expected_freeze_sha256,
            "correction_sha256": args.expected_correction_sha256,
            "submission_sha256": args.expected_submission_sha256,
        },
        "scheduler_tasks": scheduler,
        "source_partition_sync": source_sync,
        "runs": runs,
        "training_policies": descriptor(policies_path),
        "file_manifest": descriptor(manifest_path),
    }
    atomic_json(audit_path, audit_payload)

    closeout_path = output_root / "training_closeout.json"
    closeout = {
        "schema_version": CLOSEOUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity_passed": True,
        "training_runs": 9,
        "milestone_aliases": 48,
        "scientific_stability_passed": not stability_failures,
        "ratio_intervention_ever_applied": bool(ratio_open),
        "intervention_attribution_available": bool(ratio_open),
        "development_evaluation_may_proceed": True,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_audit": descriptor(audit_path),
        "training_file_manifest": descriptor(manifest_path),
        "training_policies": descriptor(policies_path),
    }
    atomic_json(closeout_path, closeout)
    print(json.dumps({
        "training_runs": 9,
        "milestone_aliases": 48,
        "file_count": len(records),
        "total_bytes": manifest_payload["total_bytes"],
        "aggregate_sha256": manifest_payload["aggregate_sha256"],
        "stability_failure_runs": [item["run"] for item in stability_failures],
        "ratio_gate_open_runs": ratio_open,
        "training_audit_sha256": sha256(audit_path),
        "training_policies_sha256": sha256(policies_path),
        "closeout_sha256": sha256(closeout_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
