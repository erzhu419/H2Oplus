#!/usr/bin/env python3
"""Independently verify the v13-v2 branch closeout and file manifest."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from build_h2oplus_v13_proximal_release_branch_closeout_v2 import (
    BRANCH_ROOT,
    CONTROLS,
    EXPECTED_TASK_IDS,
    PROTOCOL_SHA256,
    RATES,
    REPRO,
    ROOT,
    SOURCE_HANDOFF_ROOT,
    STABILITY,
    STUDY,
    actor_digest,
    aggregate,
    checkpoint_state_sha256,
    descriptor,
    finite,
    load_json,
    sha256,
)


CLOSEOUT = REPRO / "h2oplus_v13_proximal_release_branch_closeout_v2.json"
OUTPUT = REPRO / "h2oplus_v13_proximal_release_branch_closeout_verification_v2.json"
EXPECTED_ENVIRONMENT = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
        "deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/"
        "toolchain/libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "H2OPLUS_V13_PROTOCOL_SHA256": PROTOCOL_SHA256,
    "H2OPLUS_V13_BASE_DEPLOYMENT_MANIFEST_SHA256": "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd",
}


def verify_descriptor(record: dict[str, Any]) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"descriptor drifted: {path}")
    return path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"verification already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def independent_stability(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    violations = []
    critics, holds, actions, rates = [], [], [], []
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
        if epoch <= 10:
            if (
                train.get("v11/actor_hold_active") != 1.0
                or train.get("train/actor_updated") != 0.0
                or train.get("v12/actor_lr_multiplier") != 0.0
            ):
                raise ValueError(f"independent prefix-hold check failed: {variant} epoch {epoch}")
        else:
            online_updates = int(train["train/online_critic_update"])
            expected_rate = (
                1.0
                if online_updates >= 10_000
                else 0.1 + (online_updates - 5_000) / 5_000 * 0.9
            )
            observed_rate = float(train["v12/actor_lr_multiplier"])
            if not math.isclose(
                observed_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"independent actor-ramp formula failed: {variant} epoch {epoch}"
                )
            rates.append(observed_rate)
            if (
                train.get("v11/actor_hold_active") != 0.0
                or train.get("v11/online_only_actor_active") != 1.0
                or train.get("train/actor_updated") != 1.0
                or train.get("v12/temperature_lr_multiplier") != 1.0
                or train.get("v13/proximal_rate") != RATES[variant]
                or train.get("v13/proximal_update_executed") != 1.0
                or train.get("v13/actor_anchor_restored") != 1.0
            ):
                raise ValueError(f"independent proximal mechanism check failed: {variant} epoch {epoch}")
    if (
        len(rates) != 30
        or rates[-1] != 1.0
        or any(after + 1e-12 < before for before, after in zip(rates, rates[1:]))
    ):
        raise ValueError(f"independent actor-ramp check failed: {variant}")
    return {
        "passed": not violations,
        "violations": violations,
        "maximum_critic_grad_norm": max(critics),
        "minimum_hold_probability": min(holds),
        "maximum_hold_probability": max(holds),
        "maximum_absolute_positive_action_mean": max(abs(value) for value in actions),
    }


def main() -> int:
    closeout = load_json(CLOSEOUT)
    if (
        closeout.get("schema_version") != "transit-h2oplus-v13-proximal-release-branch-closeout-v2"
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_15_branches") is not True
        or closeout.get("training_run_count") != 15
        or closeout.get("development_evaluation_gate_open") is not True
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v13 closeout contract failed")
    for name, record in closeout["controls"].items():
        path = verify_descriptor(record)
        if name not in CONTROLS or record["sha256"] != CONTROLS[name][1] or path != CONTROLS[name][0].resolve():
            raise ValueError(f"unexpected frozen control: {name}")
    audit_path = verify_descriptor(closeout["audit"])
    manifest_path = verify_descriptor(closeout["file_manifest"])
    sync_path = verify_descriptor(closeout["sync"])
    scheduler_path = verify_descriptor(closeout["scheduler_evidence"])
    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    sync = load_json(sync_path)
    scheduler = load_json(scheduler_path)
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity_passed") is not True
        or audit.get("exact_run_count") != 15
        or audit.get("all_15_stability_passed") is not True
        or audit.get("stability_violation_count") != 0
        or sync.get("passed") is not True
        or sync.get("total_runs") != 15
        or scheduler.get("passed") is not True
        or scheduler.get("duplicate_or_retry_count") != 0
        or {row["task_id"] for row in scheduler.get("tasks", [])} != EXPECTED_TASK_IDS
    ):
        raise ValueError("v13 audit/sync/scheduler contract failed")
    current_records = []
    for record in manifest["files"]:
        path = (ROOT / record["path"]).resolve(strict=True)
        path.relative_to(BRANCH_ROOT.resolve())
        current = {"path": record["path"], "bytes": path.stat().st_size, "sha256": sha256(path)}
        if current != record:
            raise ValueError(f"manifest file drifted: {path}")
        if path.suffix == ".json" and not finite(load_json(path)):
            raise ValueError(f"nonfinite JSON: {path}")
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not finite(values):
                raise ValueError(f"nonfinite JSONL: {path}")
        current_records.append(current)
    if len(current_records) != manifest["file_count"] or aggregate(current_records) != manifest["aggregate_sha256"]:
        raise ValueError("manifest aggregate drifted")
    sync_inventory = {
        f"reproducibility/h2oplus_v13_proximal_release_results/branches_v2/{run['run_name']}/{record['path']}": record
        for run in sync["runs"]
        for record in run["files"]
    }
    if set(sync_inventory) != {record["path"] for record in current_records}:
        raise ValueError("sync inventory and file manifest differ")
    for record in current_records:
        source = sync_inventory[record["path"]]
        if source["bytes"] != record["bytes"] or source["sha256"] != record["sha256"]:
            raise ValueError("remote/local file binding drifted")
    for row in scheduler["tasks"]:
        backend = verify_descriptor(row["backend_exit"])
        verify_descriptor(row["scheduler_log"])
        fields = backend.read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != row["backend_token"]
            or row["backend_returncode"] != 0
            or row["token_matches"] is not True
            or row["residual_canonical_process_count"] != 0
            or row["declared_cpu_cores"] != 6
            or row["submitted_declared_ram_mb"] != 32768
            or row["declared_vram_mb"] != 8192
            or any(row["required_environment"].get(key) != value for key, value in EXPECTED_ENVIRONMENT.items())
            or row.get("scheduler_assigned_cuda_visible_devices") is not True
        ):
            raise ValueError(f"token-bound scheduler evidence failed: {row['task_id']}")
    submission = load_json(CONTROLS["submission"][0])
    submission_by_name = {row["name"]: row for row in submission["tasks"]}
    audit_by_name = {row["run_name"]: row for row in audit["runs"]}
    independent_rows = []
    handoff_groups: dict[int, set[tuple[str, str]]] = {}
    for run_name in sorted(audit_by_name):
        audit_row = audit_by_name[run_name]
        source = submission_by_name[run_name]
        run = BRANCH_ROOT / run_name
        summary = load_json(run / "summary.json")
        materialization = load_json(run / "v13_branch_materialization.json")
        policy = load_json(run / "v13_policies/policy_manifest.json")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        stability = independent_stability(metrics, audit_row["variant"])
        entry = policy["entries"][0]
        actor_payload = torch.load(run / entry["path"], map_location="cpu", weights_only=False)
        checkpoint_payload = torch.load(run / entry["model_checkpoint"]["path"], map_location="cpu", weights_only=False)
        actor_sha = actor_digest(actor_payload["actor"])
        checkpoint_actor_sha = actor_digest(checkpoint_payload["actor"])
        temperature_sha = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
        initial_path = run / "v13_branch_source/latest.pt"
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        initial_actor = actor_digest(initial_payload["actor"])
        initial_temperature = actor_digest({"log_alpha": initial_payload["log_alpha"]})
        handoff = source["source_handoff"]
        handoff_manifest = SOURCE_HANDOFF_ROOT / f"seed{source['seed']}/v12_handoff/manifest.json"
        if (
            summary.get("simulator_decision_event_cap") != 25000
            or summary.get("simulator_decision_events") != entry.get("online_events_actual")
            or summary.get("simulator_decision_event_underfill") != 25000 - summary.get("simulator_decision_events")
            or summary.get("total_updates") != 70000
            or entry.get("learner_update_steps") != 70000
            or entry.get("ratio_optimizer_steps") != 6000
            or actor_sha != entry.get("actor_tensor_sha256")
            or checkpoint_actor_sha != actor_sha
            or temperature_sha != entry.get("temperature_tensor_sha256")
            or actor_sha != audit_row["final_actor_tensor_sha256"]
            or temperature_sha != audit_row["final_temperature_tensor_sha256"]
            or checkpoint_state_sha256(initial_path) != handoff["checkpoint_state_sha256"]
            or initial_actor != handoff["actor_tensor_sha256"]
            or initial_temperature != handoff["temperature_tensor_sha256"]
            or materialization.get("handoff_manifest_sha256") != sha256(handoff_manifest)
            or materialization.get("protocol_sha256") != PROTOCOL_SHA256
            or stability["passed"] is not True
            or stability["violations"]
        ):
            raise ValueError(f"independent run verification failed: {run_name}")
        handoff_groups.setdefault(source["seed"], set()).add((initial_actor, initial_temperature))
        independent_rows.append({
            "run_name": run_name,
            "task_id": source["task_id"],
            "seed": source["seed"],
            "variant": source["variant"],
            "online_events_actual": entry["online_events_actual"],
            "learner_update_steps": entry["learner_update_steps"],
            "ratio_optimizer_steps": entry["ratio_optimizer_steps"],
            "scientific_stability": stability,
            "actor_tensor_sha256": actor_sha,
            "temperature_tensor_sha256": temperature_sha,
        })
    if len(independent_rows) != 15 or any(len(values) != 1 for values in handoff_groups.values()):
        raise ValueError("independent run matrix/handoff equality failed")
    payload = {
        "schema_version": "transit-h2oplus-v13-proximal-release-branch-closeout-verification-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "closeout": descriptor(CLOSEOUT),
        "verified_file_count": len(current_records),
        "verified_aggregate_sha256": aggregate(current_records),
        "verified_run_count": len(independent_rows),
        "all_15_scientific_stability_passed": True,
        "exact_seed_matched_v12_10k_handoffs_passed": True,
        "runs": independent_rows,
        "development_evaluation_gate_open": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
