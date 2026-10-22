#!/usr/bin/env python3
"""Independently verify the v12-v2 shared-prefix closeout and branch gate."""

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
PREFIX_ROOT = STUDY_ROOT / "prefix_v2"
EVIDENCE_ROOT = STUDY_ROOT / "evidence_prefix_v2"
CLOSEOUT = REPRO / "h2oplus_v12_staged_release_prefix_closeout_v2.json"
OUTPUT = REPRO / "h2oplus_v12_staged_release_prefix_closeout_verification_v2.json"
PLAN_SHA256 = "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa"
PROTOCOL_SHA256 = "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
SUBMISSION_SHA256 = "3a5dfe377818cfc33fded151a173c74d7feeb2b1a012fe9dec1e98f71842813d"


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
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite independent evidence: {path}")
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


def verify_bound_descriptor(record: dict[str, Any]) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"bound descriptor drifted: {path}")
    return path


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"independent verification already exists: {OUTPUT}")
    closeout = load_json(CLOSEOUT)
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v12-staged-release-prefix-closeout-v2"
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_five_prefix_runs") is not True
        or closeout.get("exact_seed_matched_10k_handoffs_passed") is not True
        or closeout.get("branch_training_submission_gate_open") is not True
        or closeout.get("development_evaluation_gate_open") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v12 prefix closeout does not open the branch gate")
    controls = closeout["controls"]
    if (
        controls["plan"]["sha256"] != PLAN_SHA256
        or controls["protocol"]["sha256"] != PROTOCOL_SHA256
        or controls["prefix_submission"]["sha256"] != SUBMISSION_SHA256
    ):
        raise ValueError("v12 closeout controls drifted")
    for record in controls.values():
        verify_bound_descriptor(record)
    audit_path = verify_bound_descriptor(closeout["audit"])
    manifest_path = verify_bound_descriptor(closeout["file_manifest"])
    sync_path = verify_bound_descriptor(closeout["sync"])
    scheduler_path = verify_bound_descriptor(closeout["scheduler_evidence"])
    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    sync = load_json(sync_path)
    scheduler = load_json(scheduler_path)
    if (
        audit.get("passed") is not True
        or audit.get("exact_run_count") != 5
        or audit.get("all_prefix_stability_passed") is not True
        or audit.get("all_zero_5k_10k_actor_and_temperature_identical") is not True
        or audit.get("all_handoffs_exact") is not True
        or manifest.get("passed") is not True
        or manifest.get("file_count") != 120
        or sync.get("passed") is not True
        or sync.get("total_runs") != 5
        or sync.get("total_files") != 120
        or scheduler.get("passed") is not True
        or scheduler.get("duplicate_or_retry_count") != 0
    ):
        raise ValueError("v12 prefix audit/manifest/sync contract failed")

    current_records = []
    for record in manifest["files"]:
        path = (ROOT / record["path"]).resolve(strict=True)
        try:
            path.relative_to(PREFIX_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"manifest path escapes prefix root: {path}") from error
        current = {
            "path": record["path"],
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if current != record:
            raise ValueError(f"manifest file drifted: {path}")
        current_records.append(current)
        if path.suffix == ".json" and not finite(load_json(path)):
            raise ValueError(f"nonfinite JSON: {path}")
    if aggregate(current_records) != manifest["aggregate_sha256"]:
        raise ValueError("v12 prefix manifest aggregate drifted")

    sync_inventory = {
        f"reproducibility/h2oplus_v12_staged_release_results/prefix_v2/{run['run_name']}/{record['path']}": record
        for run in sync["runs"]
        for record in run["files"]
    }
    if set(sync_inventory) != {record["path"] for record in current_records}:
        raise ValueError("sync inventory and file manifest differ")
    for record in current_records:
        source = sync_inventory[record["path"]]
        if source["bytes"] != record["bytes"] or source["sha256"] != record["sha256"]:
            raise ValueError("remote/local sync binding drifted")

    expected_tasks = {"t83029", "t83030", "t83031", "t83032", "t83033"}
    if {row["task_id"] for row in scheduler["tasks"]} != expected_tasks:
        raise ValueError("unexpected scheduler task lineage")
    for row in scheduler["tasks"]:
        backend = verify_bound_descriptor(row["backend_exit"])
        verify_bound_descriptor(row["scheduler_log"])
        fields = backend.read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != row["backend_token"]
            or row["backend_returncode"] != 0
            or row["token_matches"] is not True
            or row["residual_canonical_process_count"] != 0
        ):
            raise ValueError(f"token-bound scheduler evidence failed: {row['task_id']}")

    plan = load_json(Path(controls["plan"]["path"]))
    jobs = {
        job["name"]: job
        for job in plan["jobs"]
        if job.get("stage") == "prefix" and job.get("variant") == "shared-prefix"
    }
    independent_rows = []
    for run_name, job in sorted(jobs.items(), key=lambda item: int(item[1]["seed"])):
        run = PREFIX_ROOT / run_name
        summary = load_json(run / "summary.json")
        policy_manifest = load_json(run / "v12_policies/policy_manifest.json")
        export_summary = load_json(run / "v12_export_summary.json")
        handoff = load_json(run / "v12_handoff/manifest.json")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if (
            len(metrics) != 10
            or not finite(metrics)
            or summary.get("run_signature") != job["run_signature"]
            or summary.get("seed") != job["seed"]
            or summary.get("completed_epochs") != 10
            or summary.get("total_updates") != 55000
            or summary.get("simulator_decision_events") > 10000
            or summary.get("simulator_decision_event_underfill")
            != 10000 - summary.get("simulator_decision_events")
            or max(float(row["train"]["train/critic_grad_norm"]) for row in metrics)
            > 1_000_000.0
            or any(float(row["train"]["train/actor_updated"]) != 0.0 for row in metrics)
            or any(float(row["train"]["v11/actor_hold_active"]) != 1.0 for row in metrics)
            or any(float(row["train"]["v12/staged_release_active"]) != 0.0 for row in metrics)
        ):
            raise ValueError(f"independent run audit failed: {run_name}")

        entries = policy_manifest["entries"]
        if [entry["online_events_nominal"] for entry in entries] != [0, 5000, 10000]:
            raise ValueError(f"policy milestones drifted: {run_name}")
        actor_shas = []
        temperature_shas = []
        for entry in entries:
            actor_path = run / entry["path"]
            checkpoint_path = run / entry["model_checkpoint"]["path"]
            if (
                sha256(actor_path) != entry["sha256"]
                or sha256(checkpoint_path) != entry["model_checkpoint"]["sha256"]
            ):
                raise ValueError(f"policy descriptor drifted: {run_name}")
            actor_payload = torch.load(actor_path, map_location="cpu", weights_only=False)
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            actor_sha = actor_digest(actor_payload["actor"])
            checkpoint_actor_sha = actor_digest(checkpoint_payload["actor"])
            temperature_sha = actor_digest({"log_alpha": checkpoint_payload["log_alpha"]})
            if (
                actor_sha != checkpoint_actor_sha
                or actor_sha != entry["actor_tensor_sha256"]
                or temperature_sha != entry["temperature_tensor_sha256"]
                or actor_payload["extra"].get("formal_held_out_outcomes_read") is not False
                or actor_payload["extra"].get("v12_protocol", {}).get("sha256")
                != PROTOCOL_SHA256
            ):
                raise ValueError(f"policy tensor/provenance drifted: {run_name}")
            actor_shas.append(actor_sha)
            temperature_shas.append(temperature_sha)
        if len(set(actor_shas)) != 1 or len(set(temperature_shas)) != 1:
            raise ValueError(f"prefix actor/temperature hold failed: {run_name}")

        handoff_root = run / "v12_handoff"
        handoff_files = {record["path"]: record for record in handoff["files"]}
        for relative, record in handoff_files.items():
            path = handoff_root / relative
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"handoff file drifted: {path}")
        originals = {
            "metadata.json": run / "metadata.json",
            "metrics.jsonl": run / "metrics.jsonl",
            "phase_metrics.json": run / "phase_metrics.json",
            "checkpoints/latest.pt": run / "checkpoints/latest.pt",
            "checkpoints/runtime_epoch000010.pkl": run / "checkpoints/runtime_epoch000010.pkl",
        }
        if (
            handoff.get("passed") is not True
            or handoff.get("formal_held_out_outcomes_read") is not False
            or handoff.get("prefix_run_signature") != job["run_signature"]
            or handoff.get("protocol_sha256") != PROTOCOL_SHA256
            or handoff.get("actor_tensor_sha256") != actor_shas[-1]
            or handoff.get("temperature_tensor_sha256") != temperature_shas[-1]
            or set(handoff_files) != set(originals)
            or any(sha256(handoff_root / name) != sha256(path) for name, path in originals.items())
            or handoff.get("checkpoint_state_sha256")
            != checkpoint_state_sha256(handoff_root / "checkpoints/latest.pt")
            or export_summary.get("zero_5k_10k_actor_and_temperature_identical") is not True
            or export_summary.get("shared_prefix_handoff", {}).get("sha256")
            != sha256(run / "v12_handoff/manifest.json")
        ):
            raise ValueError(f"independent handoff audit failed: {run_name}")
        independent_rows.append(
            {
                "run_name": run_name,
                "seed": job["seed"],
                "actor_tensor_sha256": actor_shas[-1],
                "temperature_tensor_sha256": temperature_shas[-1],
                "checkpoint_state_sha256": handoff["checkpoint_state_sha256"],
                "simulator_decision_events": summary["simulator_decision_events"],
                "passed": True,
            }
        )

    audit_by_run = {row["run_name"]: row for row in audit["runs"]}
    for row in independent_rows:
        primary = audit_by_run[row["run_name"]]
        for key in ("actor_tensor_sha256", "temperature_tensor_sha256", "checkpoint_state_sha256"):
            if row[key] != primary[key]:
                raise ValueError(f"independent and primary audit disagree: {row['run_name']}")

    atomic_json(
        OUTPUT,
        {
            "schema_version": "transit-h2oplus-v12-staged-release-prefix-closeout-independent-verification-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "closeout": descriptor(CLOSEOUT),
            "audit": descriptor(audit_path),
            "file_manifest": descriptor(manifest_path),
            "sync": descriptor(sync_path),
            "scheduler_evidence": descriptor(scheduler_path),
            "verified_file_count": len(current_records),
            "verified_run_count": len(independent_rows),
            "runs": independent_rows,
            "artifact_integrity_passed": True,
            "scientific_stability_passed_for_all_five_prefix_runs": True,
            "exact_seed_matched_10k_handoffs_passed": True,
            "branch_training_submission_gate_open": True,
            "development_evaluation_gate_open": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "formal_held_out_outcomes_read": False,
        },
    )
    print(json.dumps({"verification": descriptor(OUTPUT), "runs": 5, "files": 120}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
