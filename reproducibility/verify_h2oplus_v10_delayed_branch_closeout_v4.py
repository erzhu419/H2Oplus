#!/usr/bin/env python3
"""Producer-independent verification of the v10-v4 branch closeout."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v10_delayed_results"
BRANCH_ROOT = RESULTS / "branches_v4"
PREFIX_ROOT = RESULTS / "prefix_v4"
EVIDENCE_ROOT = RESULTS / "evidence_branches_v4"
PLAN_PATH = REPRO / "h2oplus_v10_delayed_training_plan_v4.json"
SUBMISSION_PATH = REPRO / "h2oplus_v10_delayed_branch_submission_v4.json"
MANIFEST_PATH = RESULTS / "branch_file_manifest_v4.json"
AUDIT_PATH = RESULTS / "branch_audit_v4.json"
CLOSEOUT_PATH = RESULTS / "branch_closeout_v4.json"
OUTPUT_PATH = RESULTS / "branch_closeout_verification_v4.json"
EXPECTED = {
    PLAN_PATH: "20d87242072b4aab8332d25e2d2d27af986818339b6c63672d781607e7a0b829",
    SUBMISSION_PATH: "9a55e9c7ad208cf1d4f69f569c61734ff355bd7b9cfadeb81d864d977e371648",
}
VARIANTS = {"unit-control", "delayed-replay", "delayed-actor", "delayed-order", "delayed-combined"}
THRESHOLDS = {
    "maximum_critic_grad_norm": 1_000_000.0,
    "minimum_hold_probability": 0.01,
    "maximum_hold_probability": 0.99,
    "maximum_absolute_positive_action_mean": 0.98,
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
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


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


def stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        critic = float(train["train/critic_grad_norm"])
        hold = float(train["train/hold_probability"])
        action = float(train["train/positive_action_mean"])
        if critic > THRESHOLDS["maximum_critic_grad_norm"]:
            violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": critic})
        if not THRESHOLDS["minimum_hold_probability"] <= hold <= THRESHOLDS["maximum_hold_probability"]:
            violations.append({"epoch": epoch, "metric": "hold_probability", "value": hold})
        if abs(action) >= THRESHOLDS["maximum_absolute_positive_action_mean"]:
            violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": action})
    return violations


def main() -> int:
    errors = []
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            errors.append(f"frozen control drifted: {path}")
    plan = load(PLAN_PATH)
    submission = load(SUBMISSION_PATH)
    manifest = load(MANIFEST_PATH)
    audit = load(AUDIT_PATH)
    closeout = load(CLOSEOUT_PATH)
    branch_jobs = [job for job in plan.get("jobs", []) if job.get("stage") == "branch"]
    task_by_name = {task["name"]: task for task in submission.get("tasks", [])}
    expected_names = {job["name"] for job in branch_jobs}
    actual_names = {path.name for path in BRANCH_ROOT.iterdir() if path.is_dir()}
    if (
        len(branch_jobs) != 25
        or {job["variant"] for job in branch_jobs} != VARIANTS
        or actual_names != expected_names
        or set(task_by_name) != expected_names
    ):
        errors.append("branch matrix or canonical directory set drifted")

    records = inventory(BRANCH_ROOT)
    if (
        manifest.get("schema_version") != "transit-h2oplus-v10-delayed-branch-file-manifest-v4"
        or manifest.get("root") != str(BRANCH_ROOT.resolve())
        or manifest.get("file_count") != len(records)
        or manifest.get("total_bytes") != sum(row["bytes"] for row in records)
        or manifest.get("aggregate_sha256") != aggregate(records)
        or manifest.get("files") != records
    ):
        errors.append("branch file manifest differs from independent inventory")

    scheduler = load(EVIDENCE_ROOT / "scheduler_evidence_branches_v4.json")
    sync = load(EVIDENCE_ROOT / "source_sync_branches_v4.json")
    scheduler_rows = {row["run_name"]: row for row in scheduler.get("tasks", [])}
    sync_rows = {row["run_name"]: row for row in sync.get("runs", [])}
    if set(scheduler_rows) != expected_names or set(sync_rows) != expected_names:
        errors.append("scheduler/sync evidence does not cover the exact branch matrix")

    recomputed_runs = {}
    recomputed_failures = []
    recomputed_violations = []
    for job in branch_jobs:
        name = job["name"]
        run = BRANCH_ROOT / name
        try:
            summary = load(run / "summary.json")
            metadata = load(run / "metadata.json")
            policies = load(run / "v10_policies/policy_manifest.json")
            export = load(run / "v10_export_summary.json")
            materialization = load(run / "v10_branch_materialization.json")
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
            if not finite([summary, metadata, policies, export, materialization, rows]):
                raise ValueError("nonfinite artifact")
            events = int(summary["simulator_decision_events"])
            underfill = int(summary["simulator_decision_event_underfill"])
            blocked = int(summary["final_blocked_simultaneous_batch_size"])
            maximum_batch = int(summary["maximum_simultaneous_decisions"])
            entry = policies["entries"][0]
            prefix = PREFIX_ROOT / job["prefix"]["name"]
            prefix_rows = [json.loads(line) for line in (prefix / "metrics.jsonl").read_text().splitlines() if line.strip()]
            handoff = prefix / "v10_handoff/manifest.json"
            if (
                len(rows) != 40
                or [int(row["epoch"]) for row in rows] != list(range(1, 41))
                or rows[:10] != prefix_rows
                or summary["completed_epochs"] != 40
                or summary["total_updates"] != 70000
                or summary["simulator_decision_event_cap"] != 25000
                or not 0 <= events <= 25000
                or underfill != 25000 - events
                or (underfill == 0 and blocked != 0)
                or (underfill > 0 and not underfill < blocked <= maximum_batch)
                or len(policies["entries"]) != 1
                or entry["online_events_nominal"] != 25000
                or entry["online_events_actual"] != events
                or entry["learner_update_steps"] != 70000
                or entry["ratio_optimizer_steps"] != 6000
                or entry["event_budget_deviation"] != events - 25000
                or export.get("passed") is not True
                or export.get("summary_sha256") != sha256(run / "summary.json")
                or export.get("policy_manifest", {}).get("sha256") != sha256(run / "v10_policies/policy_manifest.json")
                or export.get("branch_materialization", {}).get("sha256") != sha256(run / "v10_branch_materialization.json")
                or materialization.get("passed") is not True
                or materialization.get("seed") != job["seed"]
                or materialization.get("variant") != job["variant"]
                or materialization.get("handoff_manifest_sha256") != sha256(handoff)
                or materialization.get("branch_initial_checkpoint_sha256") != sha256(run / "v10_branch_source/latest.pt")
                or materialization.get("branch_initial_runtime_sha256") != sha256(run / "v10_branch_source/runtime_epoch000010.pkl")
                or metadata.get("arguments", {}).get("run_signature") != job["run_signature"]
                or metadata.get("training_admission", {}).get("selected_arguments") != job
            ):
                raise ValueError("artifact contract mismatch")
            local_records = inventory(run)
            if sync_rows.get(name, {}).get("files") != local_records:
                raise ValueError("source-sync inventory mismatch")
            scheduler_row = scheduler_rows.get(name, {})
            task = task_by_name[name]
            if (
                scheduler_row.get("task_id") != task["task_id"]
                or scheduler_row.get("status") != "done"
                or scheduler_row.get("exit_code") != 0
                or scheduler_row.get("backend_returncode") != 0
                or scheduler_row.get("token_matches") is not True
                or scheduler_row.get("signature") != task["scheduler_signature"]
            ):
                raise ValueError("scheduler/backend contract mismatch")
            run_violations = stability(rows)
            if run_violations:
                recomputed_failures.append(name)
                recomputed_violations.extend({"run": name, **item} for item in run_violations)
            recomputed_runs[name] = {
                "events": events,
                "underfill": underfill,
                "violations": run_violations,
            }
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    audit_runs = {row["run"]: row for row in audit.get("runs", [])}
    if set(audit_runs) != expected_names:
        errors.append("audit run set differs from the registered matrix")
    for name, observed in recomputed_runs.items():
        recorded = audit_runs.get(name, {})
        if (
            recorded.get("events") != observed["events"]
            or recorded.get("underfill") != observed["underfill"]
            or recorded.get("scientific_stability", {}).get("violations") != observed["violations"]
            or recorded.get("scientific_stability", {}).get("passed") != (not observed["violations"])
        ):
            errors.append(f"audit result differs from independent recomputation: {name}")

    recorded_stability = audit.get("scientific_stability", {})
    expected_failure = ["h2oplus_v10c4_delayed_order_seed6205"]
    expected_violation = [
        {
            "run": "h2oplus_v10c4_delayed_order_seed6205",
            "epoch": 35,
            "metric": "hold_probability",
            "value": 0.0007603305275551975,
        }
    ]
    if (
        recomputed_failures != expected_failure
        or recomputed_violations != expected_violation
        or recorded_stability.get("passed") is not False
        or recorded_stability.get("all_30_training_runs_gate_passed") is not False
        or recorded_stability.get("failure_runs") != expected_failure
        or recorded_stability.get("violations") != expected_violation
        or recorded_stability.get("artifact_integrity_implication") != "none"
        or audit.get("artifact_integrity", {}).get("passed") is not True
    ):
        errors.append("scientific stability disclosure differs from independent recomputation")

    if (
        closeout.get("schema_version") != "transit-h2oplus-v10-delayed-branch-closeout-v4"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed") is not False
        or closeout.get("all_30_training_runs_stability_gate_passed") is not False
        or closeout.get("stability_failure_runs") != expected_failure
        or closeout.get("branch_runs") != 25
        or closeout.get("exact_seed_matched_handoffs") != 25
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("formal_confirmation_eligibility_created") is not False
        or closeout.get("branch_audit", {}).get("sha256") != sha256(AUDIT_PATH)
        or closeout.get("branch_file_manifest", {}).get("sha256") != sha256(MANIFEST_PATH)
    ):
        errors.append("branch closeout contract drifted")

    report = {
        "schema_version": "transit-h2oplus-v10-delayed-branch-closeout-verification-v4",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "artifact_integrity_verified": not any("artifact" in error or "manifest" in error for error in errors),
        "scientific_stability_verified_separately": not any("stability" in error for error in errors),
        "verified_branch_runs": len(recomputed_runs),
        "verified_stability_failure_runs": recomputed_failures,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "branch_file_manifest": descriptor(MANIFEST_PATH),
        "branch_audit": descriptor(AUDIT_PATH),
        "branch_closeout": descriptor(CLOSEOUT_PATH),
    }
    atomic_json(OUTPUT_PATH, report)
    print(json.dumps({**report, "verification": descriptor(OUTPUT_PATH)}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
