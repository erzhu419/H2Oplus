#!/usr/bin/env python3
"""Audit, seal, and export the v8 few-shot training milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "transit-h2oplus-v8-fewshot-protocol-v1"
PLAN_SCHEMA = "transit-h2oplus-v8-fewshot-training-plan-v1"
POLICY_SCHEMA = "transit-h2oplus-v8-fewshot-policy-manifest-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v8-fewshot-v1"
AUDIT_SCHEMA = "transit-h2oplus-v8-fewshot-training-audit-v1"
MANIFEST_SCHEMA = "transit-h2oplus-v8-fewshot-training-file-manifest-v1"
CLOSEOUT_SCHEMA = "transit-h2oplus-v8-fewshot-training-closeout-v1"
SYNC_SCHEMA = "transit-h2oplus-v8-fewshot-source-sync-v1"
SELECTION_RULE = (
    "all preregistered online-event milestones from every registered training "
    "seed; no evaluation outcome selects a checkpoint"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    resolved = path.resolve()
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
        candidate = run / candidate.name
        matches = list(run.rglob(candidate.name))
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot localize {value!r} under {run}")
        return matches[0].resolve()
    return (run / candidate).resolve(strict=True)


def verify_jsonl(path: Path, expected_rows: int) -> None:
    rows = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            payload = json.loads(line)
            if not finite(payload):
                raise ValueError(f"nonfinite metrics row {number}: {path}")
            rows.append(payload)
    if len(rows) != expected_rows:
        raise ValueError(f"metrics row count drifted for {path}: {len(rows)}")


def verify_policy_artifact(
    path: Path,
    *,
    method: str,
    seed: int,
    nominal: int,
    actual: int,
    learner_steps: int,
    ratio_steps: int,
    run_signature: str,
    admission_sha256: str,
    protocol_sha256: str,
    training_checkpoint_sha256: str,
) -> None:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    extra = artifact.get("extra", {})
    expected = {
        "method": method,
        "training_seed": seed,
        "fewshot_online_events_nominal": nominal,
        "fewshot_online_events_actual": actual,
        "fewshot_learner_update_steps": learner_steps,
        "fewshot_ratio_optimizer_steps": ratio_steps,
        "run_signature": run_signature,
        "training_checkpoint_sha256": training_checkpoint_sha256,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    if (
        mismatches
        or extra.get("training_admission", {}).get("sha256") != admission_sha256
        or extra.get("fewshot_protocol", {}).get("sha256") != protocol_sha256
    ):
        raise ValueError(f"policy provenance drifted for {path}: {mismatches}")


def verify_scheduler_evidence(
    submission: dict[str, Any], queue_snapshot: dict[str, Any], exit_root: Path
) -> list[dict[str, Any]]:
    tasks = {item["id"]: item for item in queue_snapshot.get("tasks", [])}
    evidence = []
    for expected in submission.get("tasks", []):
        task_id = expected["task_id"]
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"scheduler snapshot lacks {task_id}")
        if (
            task.get("status") != "done"
            or task.get("exit_code") != 0
            or task.get("signature")
            != f"H2Oplus/v8/development/fewshot/26d4f5c4ca9f/{expected['run_name']}"
        ):
            raise ValueError(f"scheduler completion drifted for {task_id}")
        exit_path = exit_root / f"backend_exit_{task_id}.status"
        fields = exit_path.read_text().strip().split("\t")
        if len(fields) != 3 or fields[0] != "0" or fields[2] != task.get("exit_status_token"):
            raise ValueError(f"token-bound backend exit drifted for {task_id}")
        evidence.append(
            {
                "task_id": task_id,
                "node": task.get("node"),
                "status": task.get("status"),
                "exit_code": task.get("exit_code"),
                "finished_at": task.get("finished_at"),
                "exit_status_token": task.get("exit_status_token"),
                "backend_exit": descriptor(exit_path),
            }
        )
    if len(evidence) != 10:
        raise ValueError("scheduler evidence does not cover ten runs")
    return evidence


def verify_sync_evidence(
    paths: list[Path], submission: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_by_host: dict[str, set[str]] = {}
    for task in submission.get("tasks", []):
        expected_by_host.setdefault(task["require_node"], set()).add(task["run_name"])
    if set(expected_by_host) != {"jtl110gpu", "jtl110gpu2"} or len(paths) != 2:
        raise ValueError("source-sync evidence must cover the two registered hosts")

    verified = []
    seen_hosts = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        payload = load_json(path)
        host = payload.get("source_host")
        runs = payload.get("run_names", [])
        if (
            payload.get("schema_version") != SYNC_SCHEMA
            or payload.get("passed") is not True
            or payload.get("checksum_mode") != "rsync-content-checksum-dry-run"
            or payload.get("differences") != []
            or host not in expected_by_host
            or host in seen_hosts
            or set(runs) != expected_by_host[host]
            or len(runs) != len(set(runs))
            or int(payload.get("source_file_count", -1)) <= 0
            or payload.get("source_file_count") != payload.get("local_file_count")
            or int(payload.get("source_total_bytes", -1)) <= 0
            or payload.get("source_total_bytes") != payload.get("local_total_bytes")
        ):
            raise ValueError(f"source-sync evidence drifted: {path}")
        seen_hosts.add(host)
        verified.append({
            "source_host": host,
            "run_names": sorted(runs),
            "source_file_count": payload["source_file_count"],
            "source_total_bytes": payload["source_total_bytes"],
            "evidence": descriptor(path),
        })
    if seen_hosts != set(expected_by_host):
        raise ValueError("source-sync evidence host coverage drifted")
    return sorted(verified, key=lambda item: item["source_host"])


def audit_run(
    root: Path,
    job: dict[str, Any],
    protocol: dict[str, Any],
    protocol_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = root / job["name"]
    if not run.is_dir():
        raise FileNotFoundError(run)
    summary_path = run / "summary.json"
    metadata_path = run / "metadata.json"
    export_path = run / "fewshot_export_summary.json"
    policy_manifest_path = run / "fewshot_policies/policy_manifest.json"
    summary = load_json(summary_path)
    metadata = load_json(metadata_path)
    export = load_json(export_path)
    policies = load_json(policy_manifest_path)
    if not all(finite(payload) for payload in (summary, metadata, export, policies)):
        raise ValueError(f"nonfinite JSON artifact in {run}")

    method = job["method"]
    seed = int(job["seed"])
    epochs = int(job["epochs"])
    cap = int(job["online_event_cap"])
    expected_updates = int(job["expected_total_updates"])
    run_signature = job["run_signature"]
    events = int(summary.get("simulator_decision_events", -1))
    underfill = int(summary.get("simulator_decision_event_underfill", -1))
    blocked = int(summary.get("final_blocked_simultaneous_batch_size", -1))
    maximum_batch = int(summary.get("maximum_simultaneous_decisions", -1))
    if (
        summary.get("schema_version") != "transit-training-summary-v8"
        or summary.get("method") != method
        or summary.get("seed") != seed
        or summary.get("run_signature") != run_signature
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

    admission = metadata.get("training_admission", {})
    admission_path = ROOT / "reproducibility/h2oplus_v8_fewshot_training_admissions_v1" / f"{job['name']}.json"
    admission_hash = sha256(admission_path)
    runtime_host = metadata.get("hardware", {}).get("runtime_fingerprint", {}).get("host", {})
    arguments = metadata.get("arguments", {})
    if (
        arguments.get("method") != method
        or arguments.get("seed") != seed
        or arguments.get("run_signature") != run_signature
        or admission.get("sha256") != admission_hash
        or admission.get("fewshot_protocol_sha256") != protocol_hash
        or runtime_host.get("cc_environment")
        != "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
        or runtime_host.get("cpath_environment")
        != "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
        or runtime_host.get("cuda_visible_devices") not in {"0", "1"}
    ):
        raise ValueError(f"metadata/admission/environment contract drifted for {run}")

    expected_export = list(job["export_online_events"])
    entries = policies.get("entries", [])
    by_nominal = {int(item["online_events_nominal"]): item for item in entries}
    if (
        policies.get("schema_version") != POLICY_SCHEMA
        or policies.get("method") != method
        or policies.get("seed") != seed
        or policies.get("run_signature") != run_signature
        or policies.get("protocol_sha256") != protocol_hash
        or len(entries) != len(by_nominal)
        or sorted(by_nominal) != sorted(expected_export)
        or export.get("passed") is not True
        or export.get("method") != method
        or export.get("seed") != seed
        or export.get("run_signature") != run_signature
        or export.get("protocol_sha256") != protocol_hash
        or export.get("summary_sha256") != sha256(summary_path)
        or export.get("policy_manifest", {}).get("sha256") != sha256(policy_manifest_path)
        or export.get("zero_and_warmup_actor_identical") is not True
        or by_nominal[0]["actor_tensor_sha256"] != by_nominal[5000]["actor_tensor_sha256"]
    ):
        raise ValueError(f"few-shot export contract drifted for {run}")

    registered = {
        int(item["online_events"]): item for item in protocol["policy_budgets"][method]["milestones"]
    }
    method_rows = []
    for nominal in expected_export:
        entry = by_nominal[nominal]
        actor = resolve_run_path(run, entry["path"])
        checkpoint = resolve_run_path(run, entry["model_checkpoint"]["path"])
        actual = int(entry["online_events_actual"])
        learner_steps = int(entry["learner_update_steps"])
        ratio_steps = int(entry["ratio_optimizer_steps"])
        milestone = registered[nominal]
        deviation = abs(actual - nominal)
        if (
            actor.stat().st_size != int(entry["bytes"])
            or sha256(actor) != entry["sha256"]
            or checkpoint.stat().st_size != int(entry["model_checkpoint"]["bytes"])
            or sha256(checkpoint) != entry["model_checkpoint"]["sha256"]
            or entry["training_checkpoint_sha256"] != entry["model_checkpoint"]["sha256"]
            or learner_steps != int(milestone["learner_update_steps"])
            or ratio_steps != int(milestone["ratio_optimizer_steps"])
            or int(entry["event_budget_deviation"]) != actual - nominal
            or (nominal not in (0, cap) and deviation >= maximum_batch)
            or (nominal == cap and actual != events)
        ):
            raise ValueError(f"milestone artifact contract drifted: {actor}")
        verify_policy_artifact(
            actor,
            method=method,
            seed=seed,
            nominal=nominal,
            actual=actual,
            learner_steps=learner_steps,
            ratio_steps=ratio_steps,
            run_signature=run_signature,
            admission_sha256=admission_hash,
            protocol_sha256=protocol_hash,
            training_checkpoint_sha256=entry["training_checkpoint_sha256"],
        )
        if nominal in protocol["policy_budgets"][method]["evaluated_unique_online_events"]:
            method_rows.append(
                {
                    "name": f"{method}-seed{seed}-events{nominal:09d}",
                    "algorithm": method,
                    "type": "checkpoint",
                    "checkpoint": actor,
                    "training_seed": seed,
                    "online_events": nominal,
                    "actual_online_events": actual,
                    "learner_update_steps": learner_steps,
                    "ratio_optimizer_steps": ratio_steps,
                    "training_run_signature": run_signature,
                    "training_admission_sha256": admission_hash,
                    "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
                }
            )

    verify_jsonl(run / "metrics.jsonl", epochs)
    return (
        {
            "run": job["name"],
            "method": method,
            "seed": seed,
            "events": events,
            "event_cap": cap,
            "underfill": underfill,
            "blocked_atomic_batch": blocked,
            "maximum_simultaneous_decisions": maximum_batch,
            "completed_epochs": epochs,
            "total_updates": expected_updates,
            "summary": descriptor(summary_path, root),
            "metadata": descriptor(metadata_path, root),
            "export_summary": descriptor(export_path, root),
            "policy_manifest": descriptor(policy_manifest_path, root),
            "training_admission": descriptor(admission_path),
            "zero_and_5000_actor_identical": True,
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
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--queue-snapshot", type=Path, required=True)
    parser.add_argument("--backend-exit-root", type=Path, required=True)
    parser.add_argument("--sync-evidence", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    for name in ("plan", "protocol", "registration", "authorization", "freeze", "submission"):
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
    submission = read_bound(args.submission, args.expected_submission_sha256)
    if (
        protocol.get("formal_held_out_outcomes_read") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or authorization.get("formal_held_out_access_authorized") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("few-shot development scope drifted")

    queue_snapshot = load_json(args.queue_snapshot.expanduser().resolve(strict=True))
    scheduler = verify_scheduler_evidence(
        submission, queue_snapshot, args.backend_exit_root.expanduser().resolve(strict=True)
    )
    source_sync = verify_sync_evidence(args.sync_evidence, submission)
    jobs = plan.get("jobs", plan.get("training_jobs", []))
    if len(jobs) != 10:
        raise ValueError("few-shot plan does not contain ten jobs")
    runs, method_rows = [], []
    for job in jobs:
        run, methods = audit_run(training_root, job, protocol, args.expected_protocol_sha256)
        runs.append(run)
        method_rows.extend(methods)
    names = {item["name"] for item in method_rows}
    if len(method_rows) != 70 or len(names) != 70:
        raise ValueError("few-shot export did not produce 70 unique evaluation policies")

    output_root.mkdir(parents=True, exist_ok=True)
    methods_path = output_root / "evaluation_methods_v8_fewshot.json"
    methods_payload = {
        "schema_version": METHODS_SCHEMA,
        "study_stage": "development",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "fewshot_protocol_sha256": args.expected_protocol_sha256,
        "methods": [],
    }
    for item in sorted(method_rows, key=lambda row: row["name"]):
        row = dict(item)
        actor = row.pop("checkpoint")
        row["checkpoint"] = os.path.relpath(actor, methods_path.parent)
        methods_payload["methods"].append(row)
    atomic_json(methods_path, methods_payload)

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

    audit_path = output_root / "training_audit.json"
    audit_payload = {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {"passed": True, "training_runs": 10, "evaluation_policies": 70},
        "scientific_results_analyzed": False,
        "formal_held_out_outcomes_read": False,
        "controls": {
            "plan_sha256": args.expected_plan_sha256,
            "protocol_sha256": args.expected_protocol_sha256,
            "registration_sha256": args.expected_registration_sha256,
            "authorization_sha256": args.expected_authorization_sha256,
            "freeze_sha256": args.expected_freeze_sha256,
            "submission_sha256": args.expected_submission_sha256,
        },
        "scheduler_tasks": scheduler,
        "source_partition_sync": source_sync,
        "runs": runs,
        "methods_manifest": descriptor(methods_path),
        "file_manifest": descriptor(manifest_path),
    }
    atomic_json(audit_path, audit_payload)

    closeout_path = output_root / "training_closeout.json"
    closeout = {
        "schema_version": CLOSEOUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity_passed": True,
        "training_runs": 10,
        "evaluation_policies": 70,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_audit": descriptor(audit_path),
        "training_file_manifest": descriptor(manifest_path),
        "evaluation_methods": descriptor(methods_path),
    }
    atomic_json(closeout_path, closeout)
    print(json.dumps({
        "training_runs": 10,
        "evaluation_policies": 70,
        "file_count": len(records),
        "total_bytes": manifest_payload["total_bytes"],
        "aggregate_sha256": manifest_payload["aggregate_sha256"],
        "training_audit_sha256": sha256(audit_path),
        "methods_sha256": sha256(methods_path),
        "closeout_sha256": sha256(closeout_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
