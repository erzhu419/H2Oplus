#!/usr/bin/env python3
"""Independently audit and seal the five v10-v4 shared-prefix runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v10.provenance import checkpoint_state_sha256


REPRO = ROOT / "reproducibility"
PREFIX_ROOT = REPRO / "h2oplus_v10_delayed_results/prefix_v4"
EVIDENCE_ROOT = REPRO / "h2oplus_v10_delayed_results/evidence_prefix_v4"
OUTPUT_ROOT = REPRO / "h2oplus_v10_delayed_results"
CONTROL = {
    "plan": (REPRO / "h2oplus_v10_delayed_training_plan_v4.json", "20d87242072b4aab8332d25e2d2d27af986818339b6c63672d781607e7a0b829"),
    "protocol": (REPRO / "h2oplus_v10_delayed_protocol_v4.json", "83a45e7d06ec30bdaba3631d7d8c6854e7609b8dcff9c46aa735d9a7944daaf3"),
    "registration": (REPRO / "h2oplus_v10_delayed_registration_v4.json", "59fbc0c7f1148fe74c4c8a0c8af0879c7900753520b5b112e0526eb3c05acf56"),
    "authorization": (REPRO / "h2oplus_v10_delayed_authorization_v4.json", "7aa6e68db01503dc9e1c3bbfbce1639dd08890e5fb46fec2cda347ade6942741"),
    "environment": (REPRO / "h2oplus_v10_delayed_environment_v4.json", "ee3de3b3990e35c579f5fcfdbafe16168a83f5011f9e5b5ce44b59cea5365bb4"),
    "freeze": (REPRO / "h2oplus_v10_delayed_freeze_v4.json", "3776bf088ba7c7cf3c2e65fb8407afb2adb9c8fbcc2266b7781bfd8d1b04cb64"),
    "correction": (REPRO / "h2oplus_v10_delayed_correction_v4.json", "2bf2a6d707283ae694e18ce796a209810415a2e079ca6b023c88a7b2e6857163"),
    "preflight_closeout": (REPRO / "h2oplus_v10_delayed_preflight_closeout_v4.json", "6871dd014e3cfd0428828ee5af00a33f5574ecb5fa5c1e33eedab42ceacbdbe9"),
    "prefix_submission": (REPRO / "h2oplus_v10_delayed_prefix_submission_v4.json", "da9f5148b721f533e7291ebdfe7adc34127f5f613c76fc1126c627c1019e2bdd"),
}
CC = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
CPATH_SUFFIX = "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libcurl.so.4"
STABILITY = {
    "maximum_critic_grad_norm": 1_000_000.0,
    "minimum_hold_probability": 0.01,
    "maximum_hold_probability": 0.99,
    "maximum_absolute_positive_action_mean": 0.98,
}
FATAL_TEXT = re.compile(
    r"Traceback|CUDA out of memory|\bOOM\b|nonfinite|ValueError|TypeError|RuntimeError|event-count",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def descriptor(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": resolved.relative_to(base.resolve()).as_posix() if base else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


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


def localize(run: Path, value: str, expected: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        candidate = expected
    else:
        candidate = run / candidate
    resolved = candidate.resolve(strict=True)
    if resolved != expected.resolve(strict=True):
        raise ValueError(f"artifact path drifted: {value}")
    return resolved


def read_metrics(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows or not finite(rows):
        raise ValueError(f"metrics integrity failed: {path}")
    return rows


def stability_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    critic, hold, action, actor = [], [], [], []
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
        actor.append(values["actor_grad_norm"])
        if values["critic_grad_norm"] > STABILITY["maximum_critic_grad_norm"]:
            violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": values["critic_grad_norm"]})
        if not STABILITY["minimum_hold_probability"] <= values["hold_probability"] <= STABILITY["maximum_hold_probability"]:
            violations.append({"epoch": epoch, "metric": "hold_probability", "value": values["hold_probability"]})
        if abs(values["positive_action_mean"]) >= STABILITY["maximum_absolute_positive_action_mean"]:
            violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": values["positive_action_mean"]})
        if (
            train.get("ratio/v10_unit_weights") != 1.0
            or train.get("v10/transition_active") != 0.0
            or train.get("v10/delayed_replay_active") != 0.0
            or train.get("v10/delayed_actor_states_active") != 0.0
            or train.get("v10/delayed_update_order_active") != 0.0
            or train.get("v10/offline_replay_fraction") != 0.5
            or train.get("v10/online_replay_fraction") != 0.5
            or train.get("train/actor_updated") != 1.0
        ):
            raise ValueError(f"shared-prefix mechanism drifted at epoch {epoch}")
    return {
        "passed": not violations,
        "thresholds": STABILITY,
        "violations": violations,
        "maximum_critic_grad_norm": max(critic),
        "minimum_hold_probability": min(hold),
        "maximum_hold_probability": max(hold),
        "maximum_absolute_positive_action_mean": max(abs(value) for value in action),
        "maximum_actor_grad_norm": max(actor),
    }


def verify_policy(
    run: Path,
    entry: dict[str, Any],
    job: dict[str, Any],
    admission_sha: str,
    source_sha: dict[str, str],
) -> None:
    actor = localize(run, entry["path"], run / entry["path"])
    checkpoint = localize(
        run,
        entry["model_checkpoint"]["path"],
        run / entry["model_checkpoint"]["path"],
    )
    if (
        descriptor(actor)["bytes"] != entry["bytes"]
        or sha256(actor) != entry["sha256"]
        or descriptor(checkpoint)["bytes"] != entry["model_checkpoint"]["bytes"]
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
        "variant": "shared-prefix",
        "stage": "prefix",
        "training_seed": int(job["seed"]),
        "total_updates": int(entry["learner_update_steps"]),
        "v10_online_events_nominal": int(entry["online_events_nominal"]),
        "v10_online_events_actual": int(entry["online_events_actual"]),
        "v10_learner_update_steps": int(entry["learner_update_steps"]),
        "v10_ratio_optimizer_steps": int(entry["ratio_optimizer_steps"]),
        "v10_actor_tensor_sha256": entry["actor_tensor_sha256"],
        "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {key: (extra.get(key), value) for key, value in expected.items() if extra.get(key) != value}
    if (
        digest != entry["actor_tensor_sha256"]
        or mismatches
        or extra.get("training_admission", {}).get("sha256") != admission_sha
        or extra.get("v10_protocol", {}).get("sha256") != CONTROL["protocol"][1]
        or extra.get("v10_source_sha256") != source_sha
        or extra.get("v10_intervention", {}).get("unit_ratio_weights") is not True
        or extra.get("selection_rule")
        != "all preregistered v10 online-event milestones; no evaluation outcome selects a checkpoint"
    ):
        raise ValueError(f"policy provenance drifted for {actor}: {mismatches}")


def audit_run(job: dict[str, Any], source_sha: dict[str, str]) -> dict[str, Any]:
    run = PREFIX_ROOT / job["name"]
    summary_path = run / "summary.json"
    metadata_path = run / "metadata.json"
    policies_path = run / "v10_policies/policy_manifest.json"
    export_path = run / "v10_export_summary.json"
    handoff_path = run / "v10_handoff/manifest.json"
    summary = load_json(summary_path)
    metadata = load_json(metadata_path)
    policies = load_json(policies_path)
    export = load_json(export_path)
    handoff = load_json(handoff_path)
    if not finite([summary, metadata, policies, export, handoff]):
        raise ValueError(f"nonfinite JSON artifact in {run}")

    cap = int(job["online_event_cap"])
    events = int(summary.get("simulator_decision_events", -1))
    underfill = int(summary.get("simulator_decision_event_underfill", -1))
    blocked = int(summary.get("final_blocked_simultaneous_batch_size", -1))
    maximum_batch = int(summary.get("maximum_simultaneous_decisions", -1))
    if (
        summary.get("schema_version") != "transit-training-summary-v8"
        or summary.get("method") != "h2oplus"
        or summary.get("seed") != job["seed"]
        or summary.get("run_signature") != job["run_signature"]
        or summary.get("completed_epochs") != job["epochs"]
        or summary.get("total_updates") != job["expected_total_updates"]
        or summary.get("simulator_decision_event_cap") != cap
        or not 0 <= events <= cap
        or underfill != cap - events
        or maximum_batch <= 0
        or (underfill == 0 and blocked != 0)
        or (underfill > 0 and not underfill < blocked <= maximum_batch)
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
        or admission.get("v10_protocol_sha256") != CONTROL["protocol"][1]
        or admission.get("v10_source_sha256") != source_sha
        or arguments.get("run_name") != job["name"]
        or arguments.get("run_signature") != job["run_signature"]
        or arguments.get("seed") != job["seed"]
        or arguments.get("epochs") != job["epochs"]
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

    entries = policies.get("entries", [])
    by_nominal = {int(entry["online_events_nominal"]): entry for entry in entries}
    registered = {int(row["online_events"]): row for row in job["milestones"]}
    if (
        policies.get("schema_version") != "transit-h2oplus-v10-delayed-policy-manifest-v1"
        or policies.get("run_signature") != job["run_signature"]
        or policies.get("seed") != job["seed"]
        or policies.get("variant") != "shared-prefix"
        or policies.get("stage") != "prefix"
        or policies.get("protocol_sha256") != CONTROL["protocol"][1]
        or sorted(by_nominal) != [0, 5000, 10000]
        or len(entries) != 3
    ):
        raise ValueError(f"policy manifest drifted for {run}")
    for nominal, entry in by_nominal.items():
        milestone = registered[nominal]
        actual = int(entry["online_events_actual"])
        if (
            int(entry["learner_update_steps"]) != int(milestone["learner_update_steps"])
            or int(entry["ratio_optimizer_steps"]) != int(milestone["ratio_optimizer_steps"])
            or int(entry["event_budget_deviation"]) != actual - nominal
            or (nominal == 0 and actual != 0)
            or (nominal == 5000 and abs(actual - nominal) >= maximum_batch)
            or (nominal == 10000 and actual != events)
        ):
            raise ValueError(f"milestone accounting drifted for {run} at {nominal}")
        verify_policy(run, entry, job, admission_sha, source_sha)
    if (
        by_nominal[0]["actor_tensor_sha256"] != by_nominal[5000]["actor_tensor_sha256"]
        or by_nominal[10000]["actor_tensor_sha256"] == by_nominal[5000]["actor_tensor_sha256"]
    ):
        raise ValueError(f"warmup/update actor identity contract failed for {run}")

    if (
        export.get("schema_version") != "transit-h2oplus-v10-delayed-export-summary-v1"
        or export.get("passed") is not True
        or export.get("formal_held_out_outcomes_read") is not False
        or export.get("summary_sha256") != sha256(summary_path)
        or export.get("policy_manifest", {}).get("sha256") != sha256(policies_path)
        or export.get("shared_prefix_handoff", {}).get("sha256") != sha256(handoff_path)
        or export.get("zero_and_warmup_actor_identical") is not True
        or export.get("branch_materialization") is not None
    ):
        raise ValueError(f"export summary drifted for {run}")

    if (
        handoff.get("schema_version") != "transit-h2oplus-v10-prefix-handoff-v1"
        or handoff.get("passed") is not True
        or handoff.get("formal_held_out_outcomes_read") is not False
        or handoff.get("prefix_run_name") != job["name"]
        or handoff.get("prefix_run_signature") != job["run_signature"]
        or handoff.get("seed") != job["seed"]
        or handoff.get("online_events_nominal") != 10000
        or handoff.get("learner_update_steps") != 55000
        or handoff.get("actor_tensor_sha256") != by_nominal[10000]["actor_tensor_sha256"]
        or handoff.get("protocol_sha256") != CONTROL["protocol"][1]
        or handoff.get("v10_source_sha256") != source_sha
        or len(handoff.get("files", [])) != 5
    ):
        raise ValueError(f"handoff manifest drifted for {run}")
    for record in handoff["files"]:
        path = run / "v10_handoff" / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"handoff file drifted: {path}")
    if checkpoint_state_sha256(run / "v10_handoff/checkpoints/latest.pt") != handoff["checkpoint_state_sha256"]:
        raise ValueError(f"handoff checkpoint state drifted for {run}")

    rows = read_metrics(run / "metrics.jsonl", int(job["epochs"]))
    stability = stability_audit(rows)
    if [int(row["epoch"]) for row in rows] != list(range(1, 11)):
        raise ValueError(f"epoch sequence drifted for {run}")
    texts = "\n".join(
        path.read_text(errors="replace")
        for path in (run / "run.log", EVIDENCE_ROOT / f"sched_{next(item['task_id'] for item in load_json(CONTROL['prefix_submission'][0])['tasks'] if item['name'] == job['name'])}.log")
    )
    if FATAL_TEXT.search(texts):
        raise ValueError(f"fatal text marker found for {run}")

    return {
        "run": job["name"],
        "seed": job["seed"],
        "events": events,
        "event_cap": cap,
        "underfill": underfill,
        "blocked_atomic_batch": blocked,
        "maximum_simultaneous_decisions": maximum_batch,
        "completed_epochs": job["epochs"],
        "total_updates": job["expected_total_updates"],
        "warmup_actual_events": int(by_nominal[5000]["online_events_actual"]),
        "zero_and_warmup_actor_identical": True,
        "tenk_actor_updated": True,
        "unit_ratio_weights_through_prefix": True,
        "transition_inactive_through_10k": True,
        "scientific_stability": stability,
        "summary": descriptor(summary_path, PREFIX_ROOT),
        "metadata": descriptor(metadata_path, PREFIX_ROOT),
        "policy_manifest": descriptor(policies_path, PREFIX_ROOT),
        "export_summary": descriptor(export_path, PREFIX_ROOT),
        "handoff_manifest": descriptor(handoff_path, PREFIX_ROOT),
        "training_admission": descriptor(admission_path),
    }


def main() -> int:
    controls = {}
    payloads = {}
    for name, (path, expected) in CONTROL.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen control drifted: {path}")
        controls[name] = descriptor(path)
        payloads[name] = load_json(path)
    if any(
        payloads[name].get("formal_held_out_outcomes_read") is not False
        for name in ("plan", "protocol", "registration", "freeze", "correction", "prefix_submission")
    ) or payloads["authorization"].get("formal_held_out_access_authorized") is not False:
        raise ValueError("v10 development scope drifted")
    plan = payloads["plan"]
    protocol = payloads["protocol"]
    prefix_jobs = [job for job in plan["jobs"] if job["stage"] == "prefix"]
    branch_jobs = [job for job in plan["jobs"] if job["stage"] == "branch"]
    if (
        plan.get("schema_version") != "transit-h2oplus-v10-delayed-training-plan-v4"
        or protocol.get("schema_version") != "transit-h2oplus-v10-delayed-protocol-v4"
        or plan["jobs"] != protocol["training_jobs"]
        or len(prefix_jobs) != 5
        or len(branch_jobs) != 25
        or {job["seed"] for job in prefix_jobs} != {6201, 6202, 6203, 6204, 6205}
        or {job["variant"] for job in branch_jobs}
        != {"unit-control", "delayed-replay", "delayed-actor", "delayed-order", "delayed-combined"}
    ):
        raise ValueError("v10 plan/protocol matrix drifted")

    scheduler_path = EVIDENCE_ROOT / "scheduler_evidence_prefix_v4.json"
    sync_path = EVIDENCE_ROOT / "source_sync_prefix_v4.json"
    scheduler = load_json(scheduler_path)
    sync = load_json(sync_path)
    expected_tasks = {item["task_id"]: item for item in payloads["prefix_submission"]["tasks"]}
    observed = {item["task_id"]: item for item in scheduler.get("tasks", [])}
    if (
        scheduler.get("schema_version") != "transit-h2oplus-v10-delayed-prefix-scheduler-evidence-v4"
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
        sync.get("schema_version") != "transit-h2oplus-v10-delayed-prefix-sync-v4"
        or sync.get("passed") is not True
        or sync.get("formal_held_out_outcomes_read") is not False
        or len(sync.get("runs", [])) != 5
        or {row["run_name"] for row in sync["runs"]} != {job["name"] for job in prefix_jobs}
    ):
        raise ValueError("prefix synchronization evidence drifted")

    source_sha = plan["v10_source_sha256"]
    runs = [audit_run(job, source_sha) for job in prefix_jobs]
    stability_failures = [row for row in runs if not row["scientific_stability"]["passed"]]
    files = sorted(path for path in PREFIX_ROOT.rglob("*") if path.is_file())
    records = [descriptor(path, PREFIX_ROOT) for path in files]
    manifest_path = OUTPUT_ROOT / "prefix_file_manifest_v4.json"
    manifest = {
        "schema_version": "transit-h2oplus-v10-delayed-prefix-file-manifest-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(PREFIX_ROOT.resolve()),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "aggregate_sha256": records_aggregate(records),
        "files": records,
    }
    atomic_json(manifest_path, manifest)
    audit_path = OUTPUT_ROOT / "prefix_audit_v4.json"
    audit = {
        "schema_version": "transit-h2oplus-v10-delayed-prefix-audit-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "artifact_integrity": {"passed": True, "runs": 5, "milestones": 15, "handoffs": 5},
        "scientific_stability": {
            "passed": not stability_failures,
            "failure_runs": [row["run"] for row in stability_failures],
            "training_rewards_support_held_out_effect_claim": False,
        },
        "branch_submission_gate_open": not stability_failures,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "controls": controls,
        "scheduler_evidence": descriptor(scheduler_path),
        "sync_evidence": descriptor(sync_path),
        "runs": runs,
        "file_manifest": descriptor(manifest_path),
    }
    atomic_json(audit_path, audit)
    closeout_path = OUTPUT_ROOT / "prefix_closeout_v4.json"
    closeout = {
        "schema_version": "transit-h2oplus-v10-delayed-prefix-closeout-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity_passed": True,
        "scientific_stability_passed": not stability_failures,
        "prefix_runs": 5,
        "prefix_milestones": 15,
        "exact_handoffs": 5,
        "branch_jobs_registered": 25,
        "branch_submission_gate_open": not stability_failures,
        "development_evaluation_submitted": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "prefix_audit": descriptor(audit_path),
        "prefix_file_manifest": descriptor(manifest_path),
    }
    atomic_json(closeout_path, closeout)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "files": len(records),
                "aggregate_sha256": manifest["aggregate_sha256"],
                "stability_failure_runs": [row["run"] for row in stability_failures],
                "prefix_audit_sha256": sha256(audit_path),
                "prefix_manifest_sha256": sha256(manifest_path),
                "prefix_closeout_sha256": sha256(closeout_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
