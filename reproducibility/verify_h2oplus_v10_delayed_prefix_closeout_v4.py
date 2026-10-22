#!/usr/bin/env python3
"""Producer-independent verification of the v10-v4 prefix closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/h2oplus_v10_delayed_results"
PREFIX = RESULTS / "prefix_v4"
AUDIT = RESULTS / "prefix_audit_v4.json"
MANIFEST = RESULTS / "prefix_file_manifest_v4.json"
CLOSEOUT = RESULTS / "prefix_closeout_v4.json"
SYNC = RESULTS / "evidence_prefix_v4/source_sync_prefix_v4.json"
OUTPUT = RESULTS / "prefix_closeout_verification_v4.json"
PROTOCOL_SHA256 = "83a45e7d06ec30bdaba3631d7d8c6854e7609b8dcff9c46aa735d9a7944daaf3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    closeout = load(CLOSEOUT)
    audit = load(AUDIT)
    manifest = load(MANIFEST)
    sync = load(SYNC)
    if (
        closeout.get("schema_version") != "transit-h2oplus-v10-delayed-prefix-closeout-v4"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed") is not True
        or closeout.get("branch_submission_gate_open") is not True
        or closeout.get("prefix_runs") != 5
        or closeout.get("prefix_milestones") != 15
        or closeout.get("exact_handoffs") != 5
        or closeout.get("branch_jobs_registered") != 25
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
        or closeout.get("prefix_audit", {}).get("sha256") != sha256(AUDIT)
        or closeout.get("prefix_file_manifest", {}).get("sha256") != sha256(MANIFEST)
    ):
        raise ValueError("prefix closeout contract failed independent verification")
    if (
        audit.get("schema_version") != "transit-h2oplus-v10-delayed-prefix-audit-v4"
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_stability", {}).get("passed") is not True
        or audit.get("branch_submission_gate_open") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or len(audit.get("runs", [])) != 5
        or audit.get("file_manifest", {}).get("sha256") != sha256(MANIFEST)
    ):
        raise ValueError("prefix audit contract failed independent verification")

    records = manifest.get("files", [])
    observed = []
    for record in records:
        path = (PREFIX / record["path"]).resolve(strict=True)
        if not path.is_relative_to(PREFIX.resolve()):
            raise ValueError(f"manifest path escapes prefix root: {path}")
        row = {"path": record["path"], "bytes": path.stat().st_size, "sha256": sha256(path)}
        if row != record:
            raise ValueError(f"prefix file changed after manifest creation: {path}")
        observed.append(row)
    actual_paths = sorted(path.relative_to(PREFIX).as_posix() for path in PREFIX.rglob("*") if path.is_file())
    if (
        manifest.get("schema_version") != "transit-h2oplus-v10-delayed-prefix-file-manifest-v4"
        or manifest.get("file_count") != len(records)
        or manifest.get("total_bytes") != sum(row["bytes"] for row in records)
        or manifest.get("aggregate_sha256") != aggregate(records)
        or actual_paths != sorted(row["path"] for row in records)
    ):
        raise ValueError("prefix file-level manifest failed independent verification")

    sync_records = {
        f"{run['run_name']}/{row['path']}": row
        for run in sync.get("runs", [])
        for row in run.get("files", [])
    }
    if set(sync_records) != set(actual_paths):
        raise ValueError("sync evidence and file manifest cover different files")
    for record in records:
        source = sync_records[record["path"]]
        if source["bytes"] != record["bytes"] or source["sha256"] != record["sha256"]:
            raise ValueError(f"sync evidence disagrees for {record['path']}")

    seeds = set()
    milestone_count = 0
    for run_audit in audit["runs"]:
        run = PREFIX / run_audit["run"]
        summary = load(run / "summary.json")
        policies = load(run / "v10_policies/policy_manifest.json")
        handoff = load(run / "v10_handoff/manifest.json")
        seed = int(run_audit["seed"])
        seeds.add(seed)
        entries = {int(row["online_events_nominal"]): row for row in policies["entries"]}
        events = int(summary.get("simulator_decision_events", -1))
        underfill = int(summary.get("simulator_decision_event_underfill", -1))
        blocked = int(summary.get("final_blocked_simultaneous_batch_size", -1))
        maximum_batch = int(summary.get("maximum_simultaneous_decisions", -1))
        if (
            summary.get("seed") != seed
            or summary.get("completed_epochs") != 10
            or summary.get("total_updates") != 55000
            or summary.get("simulator_decision_event_cap") != 10000
            or not 0 <= events <= 10000
            or underfill != 10000 - events
            or (underfill == 0 and blocked != 0)
            or (underfill > 0 and not underfill < blocked <= maximum_batch)
            or policies.get("protocol_sha256") != PROTOCOL_SHA256
            or sorted(entries) != [0, 5000, 10000]
            or int(entries[10000]["online_events_actual"]) != events
            or entries[0]["actor_tensor_sha256"] != entries[5000]["actor_tensor_sha256"]
            or entries[10000]["actor_tensor_sha256"] == entries[5000]["actor_tensor_sha256"]
            or handoff.get("passed") is not True
            or handoff.get("online_events_nominal") != 10000
            or handoff.get("learner_update_steps") != 55000
            or handoff.get("actor_tensor_sha256") != entries[10000]["actor_tensor_sha256"]
            or handoff.get("formal_held_out_outcomes_read") is not False
        ):
            raise ValueError(f"independent run contract failed for {run}")
        for nominal, entry in entries.items():
            artifact = torch.load(run / entry["path"], map_location="cpu", weights_only=False)
            if tensor_digest(artifact["actor"]) != entry["actor_tensor_sha256"]:
                raise ValueError(f"actor tensor digest failed for {run} at {nominal}")
            milestone_count += 1
    if seeds != {6201, 6202, 6203, 6204, 6205} or milestone_count != 15:
        raise ValueError("prefix seed/milestone coverage drifted")

    payload = {
        "schema_version": "transit-h2oplus-v10-delayed-prefix-closeout-verification-v4",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "producer_independent": True,
        "prefix_runs": 5,
        "prefix_milestones": milestone_count,
        "file_count": len(records),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "formal_held_out_outcomes_read": False,
        "prefix_audit": descriptor(AUDIT),
        "prefix_file_manifest": descriptor(MANIFEST),
        "prefix_closeout": descriptor(CLOSEOUT),
        "sync_evidence": descriptor(SYNC),
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({**payload, "verification": descriptor(OUTPUT)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
