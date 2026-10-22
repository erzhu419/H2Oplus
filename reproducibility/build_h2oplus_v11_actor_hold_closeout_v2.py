#!/usr/bin/env python3
"""Audit and seal the five v11-v2 actor-hold training runs."""

from __future__ import annotations

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
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v11_actor_hold_results"
TRAINING_ROOT = RESULTS / "training_v2"
EVIDENCE_ROOT = RESULTS / "evidence_training_v2"
CONTROL = {
    "design_rationale": (
        REPRO / "h2oplus_v11_actor_hold_rationale_v1.json",
        "456ddcb73c5d78a8f0bf90f3e8ddfbbfda81c8c45c76d505f4d309eead35605b",
    ),
    "registration": (
        REPRO / "h2oplus_v11_actor_hold_registration_v2.json",
        "caab9c2c3aa7ee2ce977c13403b415be9b0debc2d7c88fc6b7a12c51b3e9efb5",
    ),
    "authorization": (
        REPRO / "h2oplus_v11_actor_hold_authorization_v2.json",
        "8dc0ca71f8c45abc3e5e86af789281e4e43a882533ba4202ca567f00d87281aa",
    ),
    "environment": (
        REPRO / "h2oplus_v11_actor_hold_environment_v2.json",
        "0f005b94d5ecd361d7e10bac363be8dac5e6c9cdad7cbaf3f063744d280871de",
    ),
    "plan": (
        REPRO / "h2oplus_v11_actor_hold_training_plan_v2.json",
        "61aa351f7e36c9abd3f4822c20be17c7e5b5343cd822e031d0ede001889e7234",
    ),
    "protocol": (
        REPRO / "h2oplus_v11_actor_hold_protocol_v2.json",
        "c5b162541c6ce2476bb0f57f6dc936de13b85b5e1e6ede77b533a5cf9a4dfced",
    ),
    "freeze": (
        REPRO / "h2oplus_v11_actor_hold_freeze_v2.json",
        "095754d08f14b6e2eb84baadfa6ce1a8d4100ebac5b92f34891b2d1e01afed04",
    ),
    "correction": (
        REPRO / "h2oplus_v11_actor_hold_correction_v2.json",
        "8f1defa7d756f6336d26af0b1fe8ebf87db39f81554c89891fc2fbaaf17d1258",
    ),
    "deployment_manifest": (
        REPRO
        / "deployments/h2oplus_v11_actor_hold_pilot_v2/workspace/deployment_manifest_v11.json",
        "f408050ed84575728ea6f77aad7837856d932d39afcc612635a143c5b3a8b801",
    ),
    "deployment_verification": (
        REPRO / "h2oplus_v11_actor_hold_deployment_verification_local_v2.json",
        "673fb49a3620d456b76a69aa50fe255847e198a8d5d02b5ec979d16e6aeed598",
    ),
    "preflight_closeout": (
        REPRO / "h2oplus_v11_actor_hold_preflight_closeout_v2.json",
        "60ae6a967780bc8f3fcc49c64693d9a9aeef7536786459e8241099698fd842f1",
    ),
    "submission": (
        REPRO / "h2oplus_v11_actor_hold_submission_v2.json",
        "d6aafb8afa387392285fcd6f8ef5dc030012d8bc04e6040934248225c3d6dc39",
    ),
}
CC = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
CPATH_SUFFIX = "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libcurl.so.4"
PROTOCOL_SHA256 = CONTROL["protocol"][1]
STABILITY = {
    "maximum_critic_grad_norm": 1_000_000.0,
    "minimum_hold_probability": 0.01,
    "maximum_hold_probability": 0.99,
    "maximum_absolute_positive_action_mean": 0.98,
}
EXPECTED_MILESTONES = {
    0: (0, 50_000, 0, "offline_initializer_loaded"),
    5_000: (0, 50_000, 0, "online_warmup"),
    10_000: (10, 55_000, 2_250, "online_training"),
    25_000: (40, 70_000, 6_000, "complete"),
}
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
FATAL_TEXT = re.compile(
    r"Traceback|CUDA out of memory|\bOOM\b|nonfinite|AssertionError|"
    r"ValueError|TypeError|RuntimeError|event-count",
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


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 40 or not finite(rows):
        raise ValueError(f"metrics integrity failed: {path}")
    return rows


def artifact(run: Path, relative: str) -> Path:
    path = (run / relative).resolve(strict=True)
    if not path.is_relative_to(run.resolve()):
        raise ValueError(f"artifact escaped run root: {relative}")
    return path


def held_state_hashes_identical(entries: list[dict[str, Any]]) -> bool:
    held = [entry for entry in entries if entry["online_events_nominal"] <= 10_000]
    return (
        {entry["online_events_nominal"] for entry in held} == {0, 5_000, 10_000}
        and len({entry["actor_tensor_sha256"] for entry in held}) == 1
        and len({entry["temperature_tensor_sha256"] for entry in held}) == 1
    )


def event_cap_evidence(
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
    epoch = EXPECTED_MILESTONES[nominal][0]
    warmup = phase_metrics.get("warmup", {})
    selected_rows = rows[:epoch]
    if nominal == 5_000:
        actual = int(warmup["events"])
        latest_rollout = warmup
    else:
        if len(selected_rows) != epoch:
            raise ValueError(f"missing committed metrics for milestone {nominal}")
        actual = int(selected_rows[-1]["actual_cumulative_simulator_decision_events"])
        latest_rollout = selected_rows[-1]["rollout"]
    maximum_batch = max(
        [int(warmup.get("maximum_simultaneous_decisions", 0))]
        + [
            int(row.get("rollout", {}).get("maximum_simultaneous_decisions", 0))
            for row in selected_rows
        ]
    )
    deviation = actual - nominal
    if deviation > 0 and (maximum_batch <= 0 or deviation >= maximum_batch):
        raise ValueError(f"unbounded atomic overshoot at milestone {nominal}")
    blocked = int(latest_rollout.get("blocked_simultaneous_batch_size", 0))
    hard_cap = bool(latest_rollout.get("hard_event_cap_reached", False))
    if nominal == 25_000:
        underfill = nominal - actual
        if actual > nominal:
            raise ValueError("completed milestone exceeded the hard event cap")
        if underfill and (not hard_cap or blocked <= underfill):
            raise ValueError("final underfill lacks atomic-batch evidence")
    return {
        "actual": actual,
        "deviation": deviation,
        "maximum_simultaneous_decisions": maximum_batch,
        "hard_event_cap_reached": hard_cap,
        "blocked_simultaneous_batch_size": blocked,
    }


def audit_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    critics: list[float] = []
    holds: list[float] = []
    actions: list[float] = []
    actor_grads: list[float] = []
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        if epoch < 1 or epoch > 40:
            raise ValueError(f"invalid epoch: {epoch}")
        expected_online_updates = epoch * 500
        held = epoch <= 10
        critic = float(train["train/critic_grad_norm"])
        critics.append(critic)
        if critic > STABILITY["maximum_critic_grad_norm"]:
            violations.append(
                {"epoch": epoch, "metric": "critic_grad_norm", "value": critic}
            )
        if (
            float(train["train/online_critic_update"]) != expected_online_updates
            or float(train["train/update"]) != 50_000 + expected_online_updates
            or train.get("v11/actor_hold_active") != float(held)
            or train.get("v11/online_only_actor_active") != float(not held)
            or train.get("v11/actor_hold_critic_updates") != 5_000.0
            or train.get("v11/actor_release_events") != 10_000.0
            or train.get("v11/unit_ratio_weights") != 1.0
            or train.get("ratio/v10_unit_weights") != 1.0
            or train.get("v10/unit_ratio_weights") != 1.0
            or train.get("v11/offline_replay_fraction") != 0.5
            or train.get("v10/offline_replay_fraction") != 0.5
            or train.get("v10/online_replay_fraction") != 0.5
            or train.get("v10/delayed_replay_active") != 0.0
            or train.get("v10/delayed_update_order_active") != 0.0
            or train.get("v10/delayed_actor_states_active") != float(not held)
        ):
            raise ValueError(f"v11 mechanism drifted at epoch {epoch}")
        if held:
            if (
                train.get("train/actor_updated") != 0.0
                or float(train["train/actor_grad_norm"]) != 0.0
                or float(train["train/actor_loss"]) != 0.0
                or float(train["train/alpha"]) != 1.0
                or "train/hold_probability" in train
                or "train/positive_action_mean" in train
            ):
                raise ValueError(f"actor/temperature hold drifted at epoch {epoch}")
        else:
            if train.get("train/actor_updated") != 1.0:
                raise ValueError(f"released actor did not update at epoch {epoch}")
            hold = float(train["train/hold_probability"])
            action = float(train["train/positive_action_mean"])
            actor_grad = float(train["train/actor_grad_norm"])
            holds.append(hold)
            actions.append(action)
            actor_grads.append(actor_grad)
            if not STABILITY["minimum_hold_probability"] <= hold <= STABILITY[
                "maximum_hold_probability"
            ]:
                violations.append(
                    {"epoch": epoch, "metric": "hold_probability", "value": hold}
                )
            if abs(action) >= STABILITY["maximum_absolute_positive_action_mean"]:
                violations.append(
                    {
                        "epoch": epoch,
                        "metric": "positive_action_mean",
                        "value": action,
                    }
                )

        cap = 5_000 + expected_online_updates
        actual = int(row["actual_cumulative_simulator_decision_events"])
        if int(row["cumulative_simulator_decision_event_cap"]) != cap:
            raise ValueError(f"cumulative event cap drifted at epoch {epoch}")
        maximum_batch = int(row["rollout"]["maximum_simultaneous_decisions"])
        blocked = int(row["rollout"]["blocked_simultaneous_batch_size"])
        if actual > cap and not actual - cap < maximum_batch:
            raise ValueError(f"atomic overshoot contract drifted at epoch {epoch}")
        if actual < cap and not cap - actual < blocked <= maximum_batch:
            raise ValueError(f"atomic underfill contract drifted at epoch {epoch}")

    if len(holds) != 30:
        raise ValueError("released actor metrics must cover epochs 11 through 40")
    return {
        "passed": not violations,
        "thresholds": STABILITY,
        "violations": violations,
        "critic_metric_rows": len(critics),
        "released_actor_metric_rows": len(holds),
        "maximum_critic_grad_norm": max(critics),
        "minimum_hold_probability": min(holds),
        "maximum_hold_probability": max(holds),
        "maximum_absolute_positive_action_mean": max(abs(value) for value in actions),
        "maximum_actor_grad_norm": max(actor_grads),
    }


def verify_policy_entry(
    run: Path,
    entry: dict[str, Any],
    job: dict[str, Any],
    admission_sha: str,
    source_sha: dict[str, str],
    expected_event_evidence: dict[str, Any],
) -> str:
    nominal = int(entry["online_events_nominal"])
    epoch, learner_steps, ratio_steps, phase = EXPECTED_MILESTONES[nominal]
    policy = artifact(run, entry["path"])
    checkpoint_descriptor = entry["model_checkpoint"]
    checkpoint = artifact(run, checkpoint_descriptor["path"])
    if (
        policy.stat().st_size != entry["bytes"]
        or sha256(policy) != entry["sha256"]
        or checkpoint.stat().st_size != checkpoint_descriptor["bytes"]
        or sha256(checkpoint) != checkpoint_descriptor["sha256"]
        or entry["training_checkpoint_sha256"] != checkpoint_descriptor["sha256"]
        or entry["online_events_actual"] != expected_event_evidence["actual"]
        or entry["event_budget_deviation"] != expected_event_evidence["deviation"]
        or entry["learner_update_steps"] != learner_steps
        or entry["ratio_optimizer_steps"] != ratio_steps
        or entry["epoch"] != epoch
        or entry["phase"] != phase
        or entry["stage"] != "candidate"
    ):
        raise ValueError(f"policy milestone descriptor drifted: {policy}")

    policy_payload = torch.load(policy, map_location="cpu", weights_only=False)
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    policy_actor = actor_digest(policy_payload["actor"])
    checkpoint_actor = actor_digest(checkpoint_payload["actor"])
    temperature = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
    extra = policy_payload.get("extra", {})
    checkpoint_extra = checkpoint_payload.get("extra", {})
    expected_extra = {
        "run_signature": job["run_signature"],
        "method": "h2oplus",
        "variant": "actor-hold-online-only",
        "stage": "candidate",
        "training_seed": job["seed"],
        "total_updates": learner_steps,
        "v11_online_events_nominal": nominal,
        "v11_online_events_actual": expected_event_evidence["actual"],
        "v11_event_budget_deviation": expected_event_evidence["deviation"],
        "v11_learner_update_steps": learner_steps,
        "v11_ratio_optimizer_steps": ratio_steps,
        "v11_actor_tensor_sha256": entry["actor_tensor_sha256"],
        "v11_temperature_tensor_sha256": entry["temperature_tensor_sha256"],
        "training_checkpoint_sha256": entry["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if (
        policy_actor != entry["actor_tensor_sha256"]
        or checkpoint_actor != policy_actor
        or temperature != entry["temperature_tensor_sha256"]
        or int(checkpoint_payload["total_updates"]) != learner_steps
        or any(extra.get(key) != value for key, value in expected_extra.items())
        or extra.get("training_admission", {}).get("sha256") != admission_sha
        or extra.get("v11_protocol", {}).get("sha256") != PROTOCOL_SHA256
        or extra.get("v11_source_sha256") != source_sha
        or extra.get("v11_intervention") != EXPECTED_INTERVENTION
        or extra.get("v11_event_cap_evidence") != expected_event_evidence
        or checkpoint_extra.get("v11_intervention") != EXPECTED_INTERVENTION
        or checkpoint_extra.get("v11_online_events_nominal") != nominal
        or checkpoint_extra.get("v11_online_events_actual")
        != expected_event_evidence["actual"]
        or checkpoint_extra.get("v11_learner_update_steps") != learner_steps
        or extra.get("selection_rule")
        != "all preregistered v11 online-event milestones; no evaluation outcome selects a checkpoint"
    ):
        raise ValueError(f"policy/checkpoint provenance drifted: {policy}")
    return policy_actor


def audit_run(
    job: dict[str, Any],
    source_sha: dict[str, str],
    task_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run = TRAINING_ROOT / job["name"]
    summary_path = run / "summary.json"
    metadata_path = run / "metadata.json"
    policies_path = run / "v11_policies/policy_manifest.json"
    export_path = run / "v11_export_summary.json"
    summary = load_json(summary_path)
    metadata = load_json(metadata_path)
    policies = load_json(policies_path)
    export = load_json(export_path)
    phase_metrics = load_json(run / "phase_metrics.json")
    if not finite([summary, metadata, policies, export, phase_metrics]):
        raise ValueError(f"nonfinite JSON artifact in {run}")

    latest = run / "checkpoints/latest.pt"
    final_actor = run / "checkpoints/final_actor.pt"
    if (
        summary.get("schema_version") != "transit-training-summary-v8"
        or summary.get("method") != "h2oplus"
        or summary.get("seed") != job["seed"]
        or summary.get("run_signature") != job["run_signature"]
        or summary.get("completed_epochs") != 40
        or summary.get("start_epoch") != 0
        or summary.get("total_updates") != 70_000
        or summary.get("simulator_decision_event_cap") != 25_000
        or summary.get("checkpoint", {}).get("sha256") != sha256(latest)
        or summary.get("final_policy", {}).get("sha256") != sha256(final_actor)
        or summary.get("final_policy", {}).get("source_training_checkpoint_sha256")
        != sha256(latest)
    ):
        raise ValueError(f"summary/event-cap contract drifted for {run}")

    admission_path = (
        REPRO / "h2oplus_v11_actor_hold_training_admissions_v2" / f"{job['name']}.json"
    )
    admission_sha = sha256(admission_path)
    admission_payload = load_json(admission_path)
    admission = metadata.get("training_admission", {})
    runtime = metadata.get("hardware", {}).get("runtime_fingerprint", {}).get(
        "host", {}
    )
    arguments = metadata.get("arguments", {})
    initializer = metadata.get("offline_initializer_consumed", {})
    if (
        metadata.get("schema_version") != "transit-training-run-v8"
        or admission_payload.get("schema_version")
        != "transit-training-admission-v11-actor-hold-development-v2"
        or admission_payload.get("passed") is not True
        or admission_payload.get("study_stage") != "development"
        or admission_payload.get("formal_held_out_outcomes_read") is not False
        or admission_payload.get("selected_arguments") != job
        or admission_payload.get("v11_protocol", {}).get("sha256")
        != PROTOCOL_SHA256
        or admission_payload.get("v11_source_sha256") != source_sha
        or admission.get("sha256") != admission_sha
        or admission.get("selected_arguments") != job
        or admission.get("v11_protocol_sha256") != PROTOCOL_SHA256
        or admission.get("v11_source_sha256") != source_sha
        or metadata.get("source_sha256")
        != admission_payload.get("training_source_sha256")
        or arguments.get("run_name") != job["name"]
        or arguments.get("run_signature") != job["run_signature"]
        or arguments.get("seed") != job["seed"]
        or arguments.get("epochs") != 40
        or arguments.get("warmup_events") != 5_000
        or arguments.get("ratio_pretrain_updates") != 1_000
        or arguments.get("online_fraction") != 0.5
        or runtime.get("cc_environment") != CC
        or not str(runtime.get("cpath_environment", "")).endswith(CPATH_SUFFIX)
        or runtime.get("cuda_visible_devices") not in {"0", "1"}
        or initializer.get("sha256") != job["offline_initializer"]["sha256"]
        or initializer.get("source_run_signature")
        != job["offline_initializer"]["source_run_signature"]
        or initializer.get("source_training_seed") != job["seed"]
        or initializer.get("v11_reuse_authorized") is not True
    ):
        raise ValueError(f"metadata/admission/environment drifted for {run}")

    entries = policies.get("entries", [])
    if (
        policies.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-policy-manifest-v2"
        or policies.get("run_signature") != job["run_signature"]
        or policies.get("seed") != job["seed"]
        or policies.get("variant") != "actor-hold-online-only"
        or policies.get("stage") != "candidate"
        or policies.get("protocol_sha256") != PROTOCOL_SHA256
        or [entry.get("online_events_nominal") for entry in entries]
        != list(EXPECTED_MILESTONES)
        or not held_state_hashes_identical(entries)
    ):
        raise ValueError(f"policy manifest drifted for {run}")
    rows = read_metrics(run / "metrics.jsonl")
    if [int(row["epoch"]) for row in rows] != list(range(1, 41)):
        raise ValueError(f"epoch sequence drifted for {run}")
    milestone_evidence = {
        nominal: event_cap_evidence(rows, phase_metrics, nominal)
        for nominal in EXPECTED_MILESTONES
    }
    final_events = milestone_evidence[25_000]["actual"]
    final_underfill = -milestone_evidence[25_000]["deviation"]
    if (
        summary.get("simulator_decision_events") != final_events
        or summary.get("simulator_decision_event_underfill") != final_underfill
        or summary.get("maximum_simultaneous_decisions")
        != milestone_evidence[25_000]["maximum_simultaneous_decisions"]
        or summary.get("final_blocked_simultaneous_batch_size")
        != milestone_evidence[25_000]["blocked_simultaneous_batch_size"]
    ):
        raise ValueError(f"summary event evidence drifted for {run}")
    actor_digests = [
        verify_policy_entry(
            run,
            entry,
            job,
            admission_sha,
            source_sha,
            milestone_evidence[int(entry["online_events_nominal"])],
        )
        for entry in entries
    ]

    final_payload = torch.load(final_actor, map_location="cpu", weights_only=False)
    latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
    if (
        actor_digest(final_payload["actor"]) != actor_digests[-1]
        or actor_digest(latest_payload["actor"]) != actor_digests[-1]
        or int(latest_payload["total_updates"]) != 70_000
    ):
        raise ValueError(f"final actor/checkpoint drifted for {run}")

    if (
        export.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-export-summary-v2"
        or export.get("passed") is not True
        or export.get("formal_held_out_outcomes_read") is not False
        or export.get("checkpoint_selection_by_behavior") is not False
        or export.get("stage") != "candidate"
        or export.get("variant") != "actor-hold-online-only"
        or export.get("seed") != job["seed"]
        or export.get("run_signature") != job["run_signature"]
        or export.get("protocol_sha256") != PROTOCOL_SHA256
        or export.get("actor_hold_critic_updates") != 5_000
        or export.get("actor_release_events") != 10_000
        or export.get("zero_5k_10k_actor_and_temperature_identical") is not True
        or export.get("summary_sha256") != sha256(summary_path)
        or export.get("policy_manifest", {}).get("sha256") != sha256(policies_path)
        or export.get("policy_manifest", {}).get("entries") != 4
    ):
        raise ValueError(f"export summary drifted for {run}")

    stability = audit_history(rows)
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
        "events": final_events,
        "underfill": final_underfill,
        "completed_epochs": 40,
        "total_updates": 70_000,
        "ratio_optimizer_steps": 6_000,
        "policy_milestones": 4,
        "zero_5k_10k_actor_and_temperature_hashes_identical": True,
        "scientific_stability": stability,
        "summary": descriptor(summary_path, TRAINING_ROOT),
        "metadata": descriptor(metadata_path, TRAINING_ROOT),
        "policy_manifest": descriptor(policies_path, TRAINING_ROOT),
        "export_summary": descriptor(export_path, TRAINING_ROOT),
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
    if (
        payloads["registration"].get("formal_held_out_outcomes_read") is not False
        or payloads["plan"].get("formal_held_out_outcomes_read") is not False
        or payloads["protocol"].get("formal_held_out_outcomes_read") is not False
        or payloads["protocol"].get("local_training_or_sumo_allowed") is not False
        or payloads["submission"].get("development_evaluation_submitted")
        is not False
        or payloads["authorization"].get("formal_held_out_access_authorized")
        is not False
    ):
        raise ValueError("v11 development scope drifted")

    plan = payloads["plan"]
    protocol = payloads["protocol"]
    jobs = plan.get("jobs", [])
    if (
        plan.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-training-plan-v2"
        or protocol.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-protocol-v2"
        or jobs != protocol.get("training_jobs")
        or len(jobs) != 5
        or {job.get("seed") for job in jobs} != set(range(6201, 6206))
        or {job.get("variant") for job in jobs} != {"actor-hold-online-only"}
        or {job.get("stage") for job in jobs} != {"candidate"}
    ):
        raise ValueError("v11 training matrix drifted")

    scheduler_path = EVIDENCE_ROOT / "scheduler_evidence_training_v2.json"
    sync_path = EVIDENCE_ROOT / "source_sync_training_v2.json"
    scheduler = load_json(scheduler_path)
    sync = load_json(sync_path)
    expected_tasks = {
        item["task_id"]: item for item in payloads["submission"].get("tasks", [])
    }
    task_by_name = {item["name"]: item for item in expected_tasks.values()}
    observed = {item["task_id"]: item for item in scheduler.get("tasks", [])}
    required_env = payloads["submission"]["required_environment"]
    if (
        scheduler.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-scheduler-evidence-v2"
        or set(observed) != set(expected_tasks)
        or any(
            row.get("status") != "done"
            or row.get("exit_code") != 0
            or row.get("backend_returncode") != 0
            or row.get("token_matches") is not True
            or row.get("residual_canonical_process_count") != 0
            or row.get("signature")
            != expected_tasks[task_id]["scheduler_signature"]
            or row.get("declared_cpu_cores") != 6
            or row.get("declared_ram_mb") != 32_768
            or row.get("declared_vram_mb") != 8_192
            or row.get("required_environment", {}).get("LD_PRELOAD")
            != LD_PRELOAD
            or row.get("required_environment", {}).get("CC") != CC
            or row.get("required_environment", {}).get("CPATH")
            != required_env["CPATH"]
            or row.get("required_environment", {}).get("PYTHONDONTWRITEBYTECODE")
            != "1"
            for task_id, row in observed.items()
        )
    ):
        raise ValueError("scheduler/backend/environment evidence drifted")
    if (
        sync.get("schema_version")
        != "transit-h2oplus-v11-actor-hold-training-sync-v2"
        or sync.get("passed") is not True
        or sync.get("formal_held_out_outcomes_read") is not False
        or len(sync.get("runs", [])) != 5
        or {row["run_name"] for row in sync["runs"]}
        != {job["name"] for job in jobs}
    ):
        raise ValueError("v11 training synchronization evidence drifted")

    runs = [
        audit_run(job, plan["v11_source_sha256"], task_by_name) for job in jobs
    ]
    failures = [row for row in runs if not row["scientific_stability"]["passed"]]
    records = [
        descriptor(path, TRAINING_ROOT)
        for path in sorted(item for item in TRAINING_ROOT.rglob("*") if item.is_file())
    ]
    manifest_path = RESULTS / "training_file_manifest_v2.json"
    manifest = {
        "schema_version": "transit-h2oplus-v11-actor-hold-training-file-manifest-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(TRAINING_ROOT.resolve()),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "aggregate_sha256": records_aggregate(records),
        "files": records,
    }
    atomic_json(manifest_path, manifest)

    audit_path = RESULTS / "training_audit_v2.json"
    audit = {
        "schema_version": "transit-h2oplus-v11-actor-hold-training-audit-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "artifact_integrity": {
            "passed": True,
            "training_runs": 5,
            "policy_milestones": 20,
            "registered_25k_exports": 5,
            "held_actor_and_temperature_hash_contracts": 5,
        },
        "scientific_stability": {
            "passed": not failures,
            "all_5_training_runs_gate_passed": not failures,
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

    closeout_path = RESULTS / "training_closeout_v2.json"
    closeout = {
        "schema_version": "transit-h2oplus-v11-actor-hold-training-closeout-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity_passed": True,
        "scientific_stability_passed": not failures,
        "all_5_training_runs_stability_gate_passed": not failures,
        "stability_failure_runs": [row["run"] for row in failures],
        "training_runs": 5,
        "policy_milestones": 20,
        "held_actor_and_temperature_hash_contracts": 5,
        "development_evaluation_release_gate_open": True,
        "formal_confirmation_eligibility_created": False,
        "development_evaluation_submitted": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_audit": descriptor(audit_path),
        "training_file_manifest": descriptor(manifest_path),
    }
    atomic_json(closeout_path, closeout)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "files": len(records),
                "aggregate_sha256": manifest["aggregate_sha256"],
                "stability_failure_runs": [row["run"] for row in failures],
                "training_audit_sha256": sha256(audit_path),
                "training_manifest_sha256": sha256(manifest_path),
                "training_closeout_sha256": sha256(closeout_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
