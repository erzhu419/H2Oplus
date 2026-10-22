#!/usr/bin/env python3
"""Audit and seal the five synchronized v12-v2 shared-prefix runs."""

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

from SimpleSAC.transit_v12.provenance import checkpoint_state_sha256


REPRO = ROOT / "reproducibility"
STUDY_ROOT = REPRO / "h2oplus_v12_staged_release_results"
PREFIX_ROOT = STUDY_ROOT / "prefix_v2"
EVIDENCE_ROOT = STUDY_ROOT / "evidence_prefix_v2"
CONTROL = {
    "plan": (
        REPRO / "h2oplus_v12_staged_release_training_plan_v2.json",
        "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa",
    ),
    "protocol": (
        REPRO / "h2oplus_v12_staged_release_protocol_v2.json",
        "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
    ),
    "registration": (
        REPRO / "h2oplus_v12_staged_release_registration_v2.json",
        "59cb1e14ad0bbc8944472f5bbf3e849a62b65c4517766e3fd3845b76a5ec3ba3",
    ),
    "authorization": (
        REPRO / "h2oplus_v12_staged_release_authorization_v2.json",
        "bb1eeda1c451916aba7576f0592418471f42afaefe499386482be540def7fa23",
    ),
    "environment": (
        REPRO / "h2oplus_v12_staged_release_environment_v2.json",
        "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    ),
    "freeze": (
        REPRO / "h2oplus_v12_staged_release_freeze_v2.json",
        "0d8ab1e7852c679ae6e444d1c767799635c8790e4bc787587210a63b0e92dfe1",
    ),
    "deployment_correction": (
        REPRO / "h2oplus_v12_staged_release_deployment_correction_v2.json",
        "40dabd0dddab6ce7fd8f08ac025f8cc96a5545d1c36ff24730ce885b5eb67b25",
    ),
    "preflight_correction": (
        REPRO / "h2oplus_v12_staged_release_preflight_operational_correction_v4.json",
        "e3bfe30fbd19fb81e22ff92483f6f422adc666ddf7fd907f1c131f6eacea39db",
    ),
    "preflight_closeout": (
        REPRO / "h2oplus_v12_staged_release_preflight_closeout_v4.json",
        "24aa2a30ef34c6e79cc567995b427ec8db94c5eaa303bb72a595caf6891b8e80",
    ),
    "preflight_verification": (
        REPRO / "h2oplus_v12_staged_release_preflight_closeout_verification_v4.json",
        "75fc5357ba271d8cb25affc6b1b2a0d35f8600a81940ba5562d409e5c4c2ed03",
    ),
    "prefix_submission": (
        REPRO / "h2oplus_v12_staged_release_prefix_submission_v2.json",
        "3a5dfe377818cfc33fded151a173c74d7feeb2b1a012fe9dec1e98f71842813d",
    ),
    "deployment_manifest": (
        REPRO
        / "deployments/h2oplus_v12_staged_release_pilot_v2/workspace/deployment_manifest_v12_v2.json",
        "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9",
    ),
}
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
REQUIRED_FILES = {
    "checkpoints/final_actor.pt",
    "checkpoints/latest.pt",
    "checkpoints/runtime_epoch000010.pkl",
    "metadata.json",
    "metrics.jsonl",
    "phase_metrics.json",
    "resources.csv",
    "run.log",
    "summary.json",
    "timings.json",
    "v12_checkpoints/online_events_000000000.pt",
    "v12_checkpoints/online_events_000005000.pt",
    "v12_checkpoints/online_events_000010000.pt",
    "v12_export_summary.json",
    "v12_handoff/checkpoints/latest.pt",
    "v12_handoff/checkpoints/runtime_epoch000010.pkl",
    "v12_handoff/manifest.json",
    "v12_handoff/metadata.json",
    "v12_handoff/metrics.jsonl",
    "v12_handoff/phase_metrics.json",
    "v12_policies/online_events_000000000.pt",
    "v12_policies/online_events_000005000.pt",
    "v12_policies/online_events_000010000.pt",
    "v12_policies/policy_manifest.json",
}


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
        "path": (
            resolved.relative_to(base.resolve()).as_posix()
            if base is not None
            else str(resolved)
        ),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite closeout evidence: {path}")
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


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 10 or not finite(rows):
        raise ValueError(f"metrics integrity failed: {path}")
    return rows


def stability_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    critics = []
    hold_values = []
    action_values = []
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        critic = float(train["train/critic_grad_norm"])
        critics.append(critic)
        if critic > STABILITY["maximum_critic_grad_norm"]:
            violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": critic})
        if "train/hold_probability" in train:
            hold = float(train["train/hold_probability"])
            hold_values.append(hold)
            if not STABILITY["minimum_hold_probability"] <= hold <= STABILITY["maximum_hold_probability"]:
                violations.append({"epoch": epoch, "metric": "hold_probability", "value": hold})
        if "train/positive_action_mean" in train:
            action = float(train["train/positive_action_mean"])
            action_values.append(action)
            if abs(action) >= STABILITY["maximum_absolute_positive_action_mean"]:
                violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": action})
        expected = {
            "ratio/v10_unit_weights": 1.0,
            "v10/offline_replay_fraction": 0.5,
            "v10/online_replay_fraction": 0.5,
            "v11/actor_hold_active": 1.0,
            "v11/online_only_actor_active": 0.0,
            "train/actor_updated": 0.0,
            "v12/actor_lr_multiplier": 0.0,
            "v12/temperature_lr_multiplier": 0.0,
            "v12/actor_anchor_offline_fraction": 0.0,
            "v12/staged_release_active": 0.0,
        }
        if any(float(train.get(key, math.nan)) != value for key, value in expected.items()):
            raise ValueError(f"v12 shared-prefix mechanism drifted at epoch {epoch}")
    return {
        "passed": not violations,
        "thresholds": STABILITY,
        "violations": violations,
        "maximum_critic_grad_norm": max(critics),
        "actor_behavior_metrics_expected_during_hold": False,
        "observed_hold_probability_count": len(hold_values),
        "observed_positive_action_mean_count": len(action_values),
        "minimum_hold_probability": min(hold_values) if hold_values else None,
        "maximum_hold_probability": max(hold_values) if hold_values else None,
        "maximum_absolute_positive_action_mean": (
            max(abs(value) for value in action_values) if action_values else None
        ),
    }


def verify_descriptor(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe descriptor path")
    path = (root / relative).resolve(strict=True)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"descriptor drifted: {path}")
    return path


def verify_policy(
    run: Path,
    entry: dict[str, Any],
    job: dict[str, Any],
    admission_sha: str,
    source_map: dict[str, str],
) -> dict[str, str]:
    actor = verify_descriptor(
        run,
        {key: entry[key] for key in ("path", "bytes", "sha256")},
    )
    checkpoint = verify_descriptor(run, entry["model_checkpoint"])
    if entry["training_checkpoint_sha256"] != entry["model_checkpoint"]["sha256"]:
        raise ValueError("policy/checkpoint binding drifted")
    policy_payload = torch.load(actor, map_location="cpu", weights_only=False)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = policy_payload["extra"]
    actor_sha = actor_digest(policy_payload["actor"])
    checkpoint_actor_sha = actor_digest(checkpoint_payload["actor"])
    temperature_sha = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
    expected = {
        "run_signature": job["run_signature"],
        "method": "h2oplus",
        "variant": "shared-prefix",
        "stage": "prefix",
        "training_seed": int(job["seed"]),
        "total_updates": int(entry["learner_update_steps"]),
        "v12_online_events_nominal": int(entry["online_events_nominal"]),
        "v12_online_events_actual": int(entry["online_events_actual"]),
        "v12_learner_update_steps": int(entry["learner_update_steps"]),
        "v12_ratio_optimizer_steps": int(entry["ratio_optimizer_steps"]),
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
        or extra.get("v12_intervention", {}).get("variant") != "shared-prefix"
        or extra.get("v12_intervention", {}).get(
            "actor_and_temperature_hold_through_online_events"
        )
        != 10_000
    ):
        raise ValueError(f"v12 policy tensor/provenance drifted: {actor}")
    return {"actor": actor_sha, "temperature": temperature_sha}


def main() -> int:
    for name, (path, expected) in CONTROL.items():
        if sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"frozen v12 control drifted: {name}")
    plan = load_json(CONTROL["plan"][0])
    protocol = load_json(CONTROL["protocol"][0])
    submission = load_json(CONTROL["prefix_submission"][0])
    jobs = [
        job
        for job in plan["jobs"]
        if job.get("stage") == "prefix" and job.get("variant") == "shared-prefix"
    ]
    if (
        plan.get("jobs") != protocol.get("training_jobs")
        or len(jobs) != 5
        or {int(job["seed"]) for job in jobs} != set(range(6201, 6206))
        or submission.get("submitted_branch_tasks") != 0
        or submission.get("development_evaluation_submitted") is not False
    ):
        raise ValueError("frozen v12 prefix matrix drifted")

    sync_path = EVIDENCE_ROOT / "prefix_sync_v2.json"
    scheduler_path = EVIDENCE_ROOT / "scheduler_evidence_prefix_v2.json"
    sync = load_json(sync_path)
    scheduler = load_json(scheduler_path)
    if (
        sync.get("schema_version")
        != "transit-h2oplus-v12-staged-release-prefix-sync-v2"
        or sync.get("passed") is not True
        or sync.get("total_runs") != 5
        or scheduler.get("schema_version")
        != "transit-h2oplus-v12-staged-release-prefix-scheduler-evidence-v2"
        or scheduler.get("passed") is not True
        or scheduler.get("duplicate_or_retry_count") != 0
        or len(scheduler.get("tasks", [])) != 5
        or any(
            row.get("backend_returncode") != 0
            or row.get("token_matches") is not True
            or row.get("residual_canonical_process_count") != 0
            for row in scheduler["tasks"]
        )
    ):
        raise ValueError("v12 scheduler/sync evidence is incomplete")

    sync_by_run = {row["run_name"]: row for row in sync["runs"]}
    task_by_run = {row["run_name"]: row for row in scheduler["tasks"]}
    submission_by_run = {row["name"]: row for row in submission["tasks"]}
    if set(sync_by_run) != {job["name"] for job in jobs}:
        raise ValueError("synchronized run set differs from frozen prefix matrix")

    audit_rows = []
    manifest_records = []
    for job in sorted(jobs, key=lambda item: int(item["seed"])):
        run = PREFIX_ROOT / job["name"]
        paths = sorted(path for path in run.rglob("*") if path.is_file())
        relative_files = {path.relative_to(run).as_posix() for path in paths}
        if relative_files != REQUIRED_FILES:
            raise ValueError(f"v12 prefix file contract drifted: {job['name']}")
        current_inventory = [
            {
                "path": path.relative_to(run).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
        if current_inventory != sync_by_run[job["name"]]["files"]:
            raise ValueError(f"post-sync file drift detected: {job['name']}")

        for path in paths:
            relative = path.relative_to(run).as_posix()
            if path.suffix == ".json":
                if not finite(load_json(path)):
                    raise ValueError(f"nonfinite JSON: {path}")
            elif path.suffix == ".jsonl":
                values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if not finite(values):
                    raise ValueError(f"nonfinite JSONL: {path}")
            manifest_records.append(
                descriptor(path, ROOT)
            )

        metadata = load_json(run / "metadata.json")
        summary = load_json(run / "summary.json")
        policy_manifest = load_json(run / "v12_policies/policy_manifest.json")
        export_summary = load_json(run / "v12_export_summary.json")
        handoff = load_json(run / "v12_handoff/manifest.json")
        rows = read_metrics(run / "metrics.jsonl")
        admission_path = (
            REPRO
            / "h2oplus_v12_staged_release_training_admissions_v2"
            / f"{job['name']}.json"
        )
        admission = load_json(admission_path)
        admission_sha = sha256(admission_path)
        source_map = admission["v12_source_sha256"]
        if any(sha256(ROOT / relative) != digest for relative, digest in source_map.items()):
            raise ValueError("local v12 source differs from frozen admission")

        args = metadata["arguments"]
        if (
            metadata.get("run_signature") != job["run_signature"]
            or args.get("run_name") != job["name"]
            or args.get("seed") != job["seed"]
            or args.get("method") != "h2oplus"
            or args.get("resume") is not None
            or args.get("cpu_threads") != 6
            or args.get("warmup_events") != 5000
            or args.get("epochs") != 10
            or args.get("offline_updates") != 50000
            or metadata.get("mdp_version") != "v7"
            or metadata.get("training_admission", {}).get("sha256") != admission_sha
            or metadata.get("training_admission", {}).get("selected_arguments") != job
            or metadata.get("training_admission", {}).get("v12_source_sha256") != source_map
            or metadata.get("source_sha256") != admission["training_source_sha256"]
            or admission.get("selected_arguments") != job
            or admission.get("formal_held_out_outcomes_read") is not False
            or admission.get("passed") is not True
        ):
            raise ValueError(f"metadata/admission drifted: {job['name']}")
        runtime_host = metadata["hardware"]["runtime_fingerprint"]["host"]
        if (
            runtime_host.get("cc_environment")
            != "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
            or not runtime_host.get("cpath_environment", "").endswith(
                "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
            )
            or runtime_host.get("device") != "cuda:0"
        ):
            raise ValueError(f"frozen runtime drifted: {job['name']}")

        actual = int(summary["simulator_decision_events"])
        cap = int(summary["simulator_decision_event_cap"])
        underfill = int(summary["simulator_decision_event_underfill"])
        blocked = int(summary["final_blocked_simultaneous_batch_size"])
        maximum_batch = int(summary["maximum_simultaneous_decisions"])
        if (
            summary.get("schema_version") != "transit-training-summary-v8"
            or summary.get("method") != "h2oplus"
            or summary.get("seed") != job["seed"]
            or summary.get("run_signature") != job["run_signature"]
            or summary.get("completed_epochs") != 10
            or summary.get("total_updates") != 55000
            or cap != 10000
            or actual > cap
            or underfill != cap - actual
            or underfill >= maximum_batch
            or (underfill == 0 and blocked != 0)
            or (underfill > 0 and blocked <= underfill)
            or sha256(run / "checkpoints/latest.pt") != summary["checkpoint"]["sha256"]
            or sha256(run / "checkpoints/final_actor.pt") != summary["final_policy"]["sha256"]
            or summary["final_policy"]["source_training_checkpoint_sha256"]
            != summary["checkpoint"]["sha256"]
        ):
            raise ValueError(f"summary/event cap drifted: {job['name']}")
        cumulative = [int(row["actual_cumulative_simulator_decision_events"]) for row in rows]
        if (
            [int(row["epoch"]) for row in rows] != list(range(1, 11))
            or cumulative != sorted(cumulative)
            or cumulative[-1] != actual
            or any(value > cap for value in cumulative)
            or int(rows[-1]["total_updates"]) != 55000
        ):
            raise ValueError(f"metrics/event sequence drifted: {job['name']}")
        stability = stability_audit(rows)
        if not stability["passed"]:
            raise ValueError(f"prefix stability failed: {job['name']}")

        entries = policy_manifest.get("entries", [])
        milestone_by_event = {
            int(item["online_events"]): item for item in job["milestones"]
        }
        if (
            policy_manifest.get("schema_version")
            != "transit-h2oplus-v12-staged-release-policy-manifest-v2"
            or policy_manifest.get("variant") != "shared-prefix"
            or policy_manifest.get("stage") != "prefix"
            or policy_manifest.get("seed") != job["seed"]
            or policy_manifest.get("run_signature") != job["run_signature"]
            or policy_manifest.get("protocol_sha256") != CONTROL["protocol"][1]
            or [entry["online_events_nominal"] for entry in entries]
            != [0, 5000, 10000]
        ):
            raise ValueError(f"policy manifest drifted: {job['name']}")
        tensor_rows = []
        for entry in entries:
            milestone = milestone_by_event[int(entry["online_events_nominal"])]
            if (
                entry["learner_update_steps"] != milestone["learner_update_steps"]
                or entry["ratio_optimizer_steps"] != milestone["ratio_optimizer_steps"]
                or abs(int(entry["event_budget_deviation"])) >= maximum_batch
                or (
                    int(entry["online_events_nominal"]) == 10000
                    and int(entry["online_events_actual"]) != actual
                )
            ):
                raise ValueError(f"policy milestone drifted: {job['name']}")
            tensor_rows.append(
                verify_policy(run, entry, job, admission_sha, source_map)
            )
        if (
            len({row["actor"] for row in tensor_rows}) != 1
            or len({row["temperature"] for row in tensor_rows}) != 1
        ):
            raise ValueError(f"actor/temperature changed before release: {job['name']}")

        handoff_root = run / "v12_handoff"
        handoff_files = {record["path"]: verify_descriptor(handoff_root, record) for record in handoff["files"]}
        original_by_handoff = {
            "metadata.json": run / "metadata.json",
            "metrics.jsonl": run / "metrics.jsonl",
            "phase_metrics.json": run / "phase_metrics.json",
            "checkpoints/latest.pt": run / "checkpoints/latest.pt",
            "checkpoints/runtime_epoch000010.pkl": run / "checkpoints/runtime_epoch000010.pkl",
        }
        if (
            handoff.get("schema_version") != "transit-h2oplus-v12-prefix-handoff-v2"
            or handoff.get("passed") is not True
            or handoff.get("formal_held_out_outcomes_read") is not False
            or handoff.get("prefix_run_name") != job["name"]
            or handoff.get("prefix_run_signature") != job["run_signature"]
            or handoff.get("seed") != job["seed"]
            or handoff.get("online_events_nominal") != 10000
            or handoff.get("learner_update_steps") != 55000
            or handoff.get("protocol_sha256") != CONTROL["protocol"][1]
            or handoff.get("v12_source_sha256") != source_map
            or handoff.get("actor_tensor_sha256") != tensor_rows[-1]["actor"]
            or handoff.get("temperature_tensor_sha256") != tensor_rows[-1]["temperature"]
            or set(handoff_files) != set(original_by_handoff)
            or any(sha256(handoff_files[name]) != sha256(source) for name, source in original_by_handoff.items())
            or handoff.get("checkpoint_state_sha256")
            != checkpoint_state_sha256(handoff_root / "checkpoints/latest.pt")
        ):
            raise ValueError(f"v12 handoff drifted: {job['name']}")

        if (
            export_summary.get("schema_version")
            != "transit-h2oplus-v12-staged-release-export-summary-v2"
            or export_summary.get("passed") is not True
            or export_summary.get("formal_held_out_outcomes_read") is not False
            or export_summary.get("zero_5k_10k_actor_and_temperature_identical") is not True
            or export_summary.get("run_signature") != job["run_signature"]
            or export_summary.get("protocol_sha256") != CONTROL["protocol"][1]
            or export_summary.get("policy_manifest", {}).get("sha256")
            != sha256(run / "v12_policies/policy_manifest.json")
            or export_summary.get("shared_prefix_handoff", {}).get("sha256")
            != sha256(run / "v12_handoff/manifest.json")
            or export_summary.get("summary_sha256") != sha256(run / "summary.json")
        ):
            raise ValueError(f"v12 export closeout drifted: {job['name']}")

        texts = (run / "run.log").read_text(errors="replace") + "\n" + (
            Path(task_by_run[job["name"]]["scheduler_log"]["path"])
        ).read_text(errors="replace")
        if FATAL_TEXT.search(texts):
            raise ValueError(f"fatal text found: {job['name']}")
        submission_row = submission_by_run[job["name"]]
        scheduler_row = task_by_run[job["name"]]
        if (
            submission_row["run_signature"] != job["run_signature"]
            or scheduler_row["signature"] != submission_row["scheduler_signature"]
            or scheduler_row["node"] != submission_row["require_node"]
        ):
            raise ValueError(f"scheduler submission binding drifted: {job['name']}")

        audit_rows.append(
            {
                "run_name": job["name"],
                "seed": job["seed"],
                "run_signature": job["run_signature"],
                "task_id": submission_row["task_id"],
                "node": submission_row["require_node"],
                "file_count": len(paths),
                "simulator_decision_events": actual,
                "event_underfill": underfill,
                "final_blocked_simultaneous_batch_size": blocked,
                "learner_update_steps": 55000,
                "ratio_optimizer_steps": 2250,
                "actor_tensor_sha256": tensor_rows[-1]["actor"],
                "temperature_tensor_sha256": tensor_rows[-1]["temperature"],
                "checkpoint_state_sha256": handoff["checkpoint_state_sha256"],
                "stability": stability,
                "artifact_integrity_passed": True,
                "prefix_hold_passed": True,
                "formal_held_out_outcomes_read": False,
            }
        )

    manifest_records.sort(key=lambda record: record["path"])
    audit_path = STUDY_ROOT / "prefix_audit_v2.json"
    manifest_path = STUDY_ROOT / "prefix_file_manifest_v2.json"
    closeout_path = REPRO / "h2oplus_v12_staged_release_prefix_closeout_v2.json"
    atomic_json(
        audit_path,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-prefix-audit-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "runs": audit_rows,
            "exact_run_count": len(audit_rows),
            "all_prefix_stability_passed": all(row["stability"]["passed"] for row in audit_rows),
            "all_zero_5k_10k_actor_and_temperature_identical": True,
            "all_handoffs_exact": True,
            "formal_held_out_outcomes_read": False,
        },
    )
    atomic_json(
        manifest_path,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-prefix-file-manifest-v2",
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
            "schema_version": "transit-h2oplus-v12-staged-release-prefix-closeout-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "artifact_integrity_passed": True,
            "scientific_stability_passed_for_all_five_prefix_runs": True,
            "exact_seed_matched_10k_handoffs_passed": True,
            "controls": {name: descriptor(path) for name, (path, _) in CONTROL.items()},
            "sync": descriptor(sync_path),
            "scheduler_evidence": descriptor(scheduler_path),
            "audit": descriptor(audit_path),
            "file_manifest": descriptor(manifest_path),
            "prefix_tasks": [row["task_id"] for row in audit_rows],
            "branch_training_submission_gate_open": True,
            "development_evaluation_gate_open": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "formal_held_out_outcomes_read": False,
        },
    )
    print(
        json.dumps(
            {
                "audit": descriptor(audit_path),
                "manifest": descriptor(manifest_path),
                "closeout": descriptor(closeout_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
