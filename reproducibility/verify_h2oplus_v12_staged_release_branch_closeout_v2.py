#!/usr/bin/env python3
"""Independently verify the v12-v2 25-branch and 30-run closeout."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
BRANCH_ROOT = STUDY_ROOT / "branches_v2"
PREFIX_ROOT = STUDY_ROOT / "prefix_v2"
CLOSEOUT = REPRO / "h2oplus_v12_staged_release_branch_closeout_v2.json"
OUTPUT = REPRO / "h2oplus_v12_staged_release_branch_closeout_verification_v2.json"
PLAN_SHA256 = "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa"
PROTOCOL_SHA256 = "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
SUBMISSION_SHA256 = "e5c2b52d8529f24f28ad670d673645884c271f3b670da6c3dfb003087efaf7ca"
VARIANT_FLAGS = {
    "abrupt-control": (False, False, False),
    "actor-ramp": (True, False, False),
    "temperature-ramp": (False, True, False),
    "actor-anchor": (False, False, True),
    "staged-combined": (True, True, True),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"independent verification already exists: {path}")
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
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
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


def verify_bound(record: dict[str, Any]) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"bound descriptor drifted: {path}")
    return path


def verify_mechanism(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    actor_ramp, temperature_ramp, actor_anchor = VARIANT_FLAGS[variant]
    violations = []
    actor_values, temperature_values = [], []
    for row in rows:
        epoch = int(row["epoch"])
        train = row["train"]
        critic = float(train["train/critic_grad_norm"])
        if critic > 1_000_000.0:
            violations.append({"epoch": epoch, "metric": "critic_grad_norm", "value": critic})
        hold = train.get("train/hold_probability")
        action = train.get("train/positive_action_mean")
        if hold is not None and not 0.01 <= float(hold) <= 0.99:
            violations.append({"epoch": epoch, "metric": "hold_probability", "value": float(hold)})
        if action is not None and abs(float(action)) >= 0.98:
            violations.append({"epoch": epoch, "metric": "positive_action_mean", "value": float(action)})
        if (
            float(train.get("ratio/v10_unit_weights", math.nan)) != 1.0
            or float(train.get("v10/unit_ratio_weights", math.nan)) != 1.0
            or float(train.get("v10/offline_replay_fraction", math.nan)) != 0.5
            or float(train.get("v10/online_replay_fraction", math.nan)) != 0.5
        ):
            raise ValueError(f"fixed mechanism drifted for {variant} epoch {epoch}")
        if epoch <= 10:
            observed = (
                train.get("v11/actor_hold_active"),
                train.get("train/actor_updated"),
                train.get("v12/actor_lr_multiplier"),
                train.get("v12/temperature_lr_multiplier"),
                train.get("v12/actor_anchor_offline_fraction"),
                train.get("v12/staged_release_active"),
            )
            if observed != (1.0, 0.0, 0.0, 0.0, 0.0, 0.0):
                raise ValueError(f"prefix hold drifted for {variant} epoch {epoch}")
        else:
            actor = float(train["v12/actor_lr_multiplier"])
            temperature = float(train["v12/temperature_lr_multiplier"])
            anchor = float(train["v12/actor_anchor_offline_fraction"])
            actor_values.append(actor)
            temperature_values.append(temperature)
            if (
                train.get("v11/actor_hold_active") != 0.0
                or train.get("v11/online_only_actor_active") != 1.0
                or train.get("train/actor_updated") != 1.0
                or train.get("v12/staged_release_active") != 1.0
                or anchor != (26.0 / 128.0 if actor_anchor else 0.0)
                or (actor_ramp and not 0.1 <= actor <= 1.0)
                or (not actor_ramp and actor != 1.0)
                or (temperature_ramp and not 0.1 <= temperature <= 1.0)
                or (not temperature_ramp and temperature != 1.0)
            ):
                raise ValueError(f"release mechanism drifted for {variant} epoch {epoch}")
    for values, active in ((actor_values, actor_ramp), (temperature_values, temperature_ramp)):
        if active and (
            values[0] >= 1.0
            or values[-1] != 1.0
            or any(after + 1e-12 < before for before, after in zip(values, values[1:]))
        ):
            raise ValueError(f"release ramp sequence drifted for {variant}")
    return {"passed": not violations, "violations": violations}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"independent verification already exists: {OUTPUT}")
    closeout = load_json(CLOSEOUT)
    if (
        closeout.get("schema_version") != "transit-h2oplus-v12-staged-release-branch-closeout-v2"
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("exact_seed_matched_10k_handoffs_passed") is not True
        or closeout.get("training_run_count") != 30
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v12 branch closeout contract failed")
    controls = closeout["controls"]
    if (
        controls["plan"]["sha256"] != PLAN_SHA256
        or controls["protocol"]["sha256"] != PROTOCOL_SHA256
        or controls["branch_submission"]["sha256"] != SUBMISSION_SHA256
    ):
        raise ValueError("v12 frozen control hashes drifted")
    for record in controls.values():
        verify_bound(record)

    audit_path = verify_bound(closeout["audit"])
    manifest_path = verify_bound(closeout["file_manifest"])
    sync_path = verify_bound(closeout["sync"])
    scheduler_path = verify_bound(closeout["scheduler_evidence"])
    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    sync = load_json(sync_path)
    scheduler = load_json(scheduler_path)
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity_passed") is not True
        or audit.get("exact_run_count") != 25
        or sync.get("passed") is not True
        or sync.get("total_runs") != 25
        or scheduler.get("passed") is not True
        or scheduler.get("duplicate_or_retry_count") != 0
        or len(scheduler.get("tasks", [])) != 25
    ):
        raise ValueError("v12 branch audit/sync/scheduler contract failed")

    current_records = []
    for record in manifest["files"]:
        path = (ROOT / record["path"]).resolve(strict=True)
        try:
            path.relative_to(BRANCH_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"manifest path escapes branch root: {path}") from error
        current = {"path": record["path"], "bytes": path.stat().st_size, "sha256": sha256(path)}
        if current != record:
            raise ValueError(f"manifest file drifted: {path}")
        current_records.append(current)
        if path.suffix == ".json" and not finite(load_json(path)):
            raise ValueError(f"nonfinite JSON: {path}")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not finite(rows):
                raise ValueError(f"nonfinite JSONL: {path}")
    if (
        len(current_records) != manifest["file_count"]
        or aggregate(current_records) != manifest["aggregate_sha256"]
    ):
        raise ValueError("v12 branch manifest aggregate drifted")

    sync_inventory = {
        f"reproducibility/h2oplus_v12_staged_release_results/branches_v2/{run['run_name']}/{record['path']}": record
        for run in sync["runs"]
        for record in run["files"]
    }
    if set(sync_inventory) != {record["path"] for record in current_records}:
        raise ValueError("sync inventory and file manifest differ")
    for record in current_records:
        source = sync_inventory[record["path"]]
        if source["bytes"] != record["bytes"] or source["sha256"] != record["sha256"]:
            raise ValueError("remote/local sync binding drifted")

    expected_tasks = {f"t{task_id}" for task_id in range(83052, 83077)}
    if {row["task_id"] for row in scheduler["tasks"]} != expected_tasks:
        raise ValueError("unexpected scheduler task lineage")
    for row in scheduler["tasks"]:
        backend = verify_bound(row["backend_exit"])
        verify_bound(row["scheduler_log"])
        fields = backend.read_text().strip().split("\t")
        environment = row.get("required_environment", {})
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != row["backend_token"]
            or row["backend_returncode"] != 0
            or row["token_matches"] is not True
            or row["residual_canonical_process_count"] != 0
            or row["declared_cpu_cores"] != 6
            or row["declared_vram_mb"] != 8192
            or environment.get("LD_PRELOAD")
            != "/usr/lib/x86_64-linux-gnu/libcurl.so.4"
            or environment.get("CC")
            != "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc"
            or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
            or environment.get("H2OPLUS_V12_PROTOCOL_SHA256") != PROTOCOL_SHA256
            or not str(environment.get("CPATH", "")).endswith(
                "toolchain/libxcrypt-4.4.36-hd590300_1/include"
            )
            or row.get("scheduler_assigned_cuda_visible_devices") is not True
        ):
            raise ValueError(f"token-bound scheduler evidence failed: {row['task_id']}")

    plan = load_json(Path(controls["plan"]["path"]))
    jobs = [job for job in plan["jobs"] if job.get("stage") == "branch"]
    if len(jobs) != 25 or {(job["variant"], job["seed"]) for job in jobs} != {
        (variant, seed) for variant in VARIANT_FLAGS for seed in range(6201, 6206)
    }:
        raise ValueError("independent branch matrix check failed")
    audit_by_name = {row["run_name"]: row for row in audit["runs"]}
    independent_rows = []
    all_stable = True
    for job in sorted(jobs, key=lambda item: (item["variant"], item["seed"])):
        run = BRANCH_ROOT / job["name"]
        summary = load_json(run / "summary.json")
        metadata = load_json(run / "metadata.json")
        materialization = load_json(run / "v12_branch_materialization.json")
        policy = load_json(run / "v12_policies/policy_manifest.json")
        export = load_json(run / "v12_export_summary.json")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        prefix_run = PREFIX_ROOT / job["prefix"]["name"]
        prefix_metrics = [json.loads(line) for line in (prefix_run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        prefix_handoff = load_json(prefix_run / "v12_handoff/manifest.json")
        admission_path = REPRO / "h2oplus_v12_staged_release_training_admissions_v2" / f"{job['name']}.json"
        admission = load_json(admission_path)
        events = int(summary.get("simulator_decision_events", -1))
        underfill = 25000 - events
        blocked = int(summary.get("final_blocked_simultaneous_batch_size", -1))
        maximum_batch = int(summary.get("maximum_simultaneous_decisions", -1))
        if (
            len(metrics) != 40
            or metrics[:10] != prefix_metrics
            or [int(row["epoch"]) for row in metrics] != list(range(1, 41))
            or summary.get("run_signature") != job["run_signature"]
            or summary.get("seed") != job["seed"]
            or summary.get("completed_epochs") != 40
            or summary.get("total_updates") != 70000
            or events > 25000
            or summary.get("simulator_decision_event_underfill") != underfill
            or (underfill == 0 and blocked != 0)
            or (underfill > 0 and not underfill < blocked <= maximum_batch)
            or metadata.get("training_admission", {}).get("sha256") != sha256(admission_path)
            or metadata.get("training_admission", {}).get("selected_arguments") != job
            or admission.get("selected_arguments") != job
            or materialization.get("handoff_manifest_sha256") != sha256(prefix_run / "v12_handoff/manifest.json")
            or materialization.get("checkpoint_state_sha256") != checkpoint_state_sha256(run / "v12_branch_source/latest.pt")
            or materialization.get("checkpoint_state_sha256") != prefix_handoff["checkpoint_state_sha256"]
            or materialization.get("actor_tensor_sha256") != prefix_handoff["actor_tensor_sha256"]
            or materialization.get("temperature_tensor_sha256") != prefix_handoff["temperature_tensor_sha256"]
            or export.get("passed") is not True
            or export.get("summary_sha256") != sha256(run / "summary.json")
            or export.get("branch_materialization", {}).get("sha256") != sha256(run / "v12_branch_materialization.json")
            or policy.get("protocol_sha256") != PROTOCOL_SHA256
            or len(policy.get("entries", [])) != 1
        ):
            raise ValueError(f"independent run contract failed: {job['name']}")
        entry = policy["entries"][0]
        actor_path = run / entry["path"]
        checkpoint_path = run / entry["model_checkpoint"]["path"]
        actor_payload = torch.load(actor_path, map_location="cpu", weights_only=False)
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_sha = actor_digest(actor_payload["actor"])
        temperature_sha = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
        flags = VARIANT_FLAGS[job["variant"]]
        intervention = actor_payload["extra"]["v12_intervention"]
        if (
            sha256(actor_path) != entry["sha256"]
            or sha256(checkpoint_path) != entry["model_checkpoint"]["sha256"]
            or actor_sha != entry["actor_tensor_sha256"]
            or actor_digest(checkpoint_payload["actor"]) != actor_sha
            or temperature_sha != entry["temperature_tensor_sha256"]
            or (intervention["actor_learning_rate_ramp"], intervention["temperature_learning_rate_ramp"], intervention["actor_state_anchor"]) != flags
            or intervention.get("unit_ratio_weights") is not True
            or actor_payload["extra"].get("formal_held_out_outcomes_read") is not False
        ):
            raise ValueError(f"independent actor audit failed: {job['name']}")
        stability = verify_mechanism(metrics, job["variant"])
        all_stable = all_stable and stability["passed"]
        recorded = audit_by_name[job["name"]]
        if (
            recorded["actor_tensor_sha256"] != actor_sha
            or recorded["temperature_tensor_sha256"] != temperature_sha
            or recorded["simulator_decision_events"] != events
            or recorded["scientific_stability"]["passed"] != stability["passed"]
        ):
            raise ValueError(f"audit record disagrees with independent result: {job['name']}")
        independent_rows.append({
            "run_name": job["name"],
            "variant": job["variant"],
            "seed": job["seed"],
            "simulator_decision_events": events,
            "actor_tensor_sha256": actor_sha,
            "temperature_tensor_sha256": temperature_sha,
            "exact_seed_matched_10k_handoff": True,
            "scientific_stability_passed": stability["passed"],
            "passed": True,
        })

    prefix_verification = load_json(Path(controls["prefix_verification"]["path"]))
    all_30_stable = all_stable and prefix_verification.get("scientific_stability_passed_for_all_five_prefix_runs") is True
    if (
        audit.get("all_branch_stability_passed") != all_stable
        or closeout.get("scientific_stability_passed_for_all_25_branches") != all_stable
        or closeout.get("scientific_stability_passed_for_all_30_training_runs") != all_30_stable
        or closeout.get("development_evaluation_gate_open") != all_30_stable
    ):
        raise ValueError("stability/development gate accounting drifted")

    atomic_json(
        OUTPUT,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-branch-closeout-verification-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "closeout": descriptor(CLOSEOUT),
            "audit": descriptor(audit_path),
            "file_manifest": descriptor(manifest_path),
            "sync": descriptor(sync_path),
            "scheduler_evidence": descriptor(scheduler_path),
            "runs": independent_rows,
            "verified_branch_run_count": len(independent_rows),
            "verified_training_run_count": 30,
            "verified_file_count": len(current_records),
            "artifact_integrity_passed": True,
            "exact_seed_matched_10k_handoffs_passed": True,
            "scientific_stability_passed_for_all_25_branches": all_stable,
            "scientific_stability_passed_for_all_30_training_runs": all_30_stable,
            "development_evaluation_gate_open": all_30_stable,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "formal_held_out_outcomes_read": False,
        },
    )
    print(json.dumps({"verification": descriptor(OUTPUT), "runs": len(independent_rows), "files": len(current_records)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
