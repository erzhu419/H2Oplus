#!/usr/bin/env python3
"""Independently verify the sealed H2O+ v9 adaptive training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_SCHEMA = "transit-h2oplus-v9-adaptive-training-file-manifest-v1"
AUDIT_SCHEMA = "transit-h2oplus-v9-adaptive-training-audit-v1"
CLOSEOUT_SCHEMA = "transit-h2oplus-v9-adaptive-training-closeout-v1"
POLICIES_SCHEMA = "transit-h2oplus-v9-adaptive-training-policies-v1"
REPORT_SCHEMA = "transit-h2oplus-v9-adaptive-training-verification-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--closeout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.training_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    audit_path = args.audit.expanduser().resolve(strict=True)
    policies_path = args.policies.expanduser().resolve(strict=True)
    closeout_path = args.closeout.expanduser().resolve(strict=True)
    errors: list[str] = []
    checked_files = 0
    checked_bytes = 0
    try:
        manifest = load(manifest_path)
        audit = load(audit_path)
        policies = load(policies_path)
        closeout = load(closeout_path)
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("training file-manifest schema drifted")
        if audit.get("schema_version") != AUDIT_SCHEMA:
            raise ValueError("training audit schema drifted")
        if policies.get("schema_version") != POLICIES_SCHEMA:
            raise ValueError("training policies schema drifted")
        if closeout.get("schema_version") != CLOSEOUT_SCHEMA:
            raise ValueError("training closeout schema drifted")

        records = manifest.get("files", [])
        actual_paths = sorted(path for path in root.rglob("*") if path.is_file())
        if len(records) != len(actual_paths) or len(records) != manifest.get("file_count"):
            raise ValueError("training file inventory count drifted")
        expected_names = {record["path"] for record in records}
        actual_names = {path.relative_to(root).as_posix() for path in actual_paths}
        if expected_names != actual_names:
            raise ValueError("training file inventory membership drifted")
        for record in records:
            path = root / record["path"]
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"training file identity drifted: {record['path']}")
            checked_files += 1
            checked_bytes += path.stat().st_size
        if aggregate(records) != manifest.get("aggregate_sha256"):
            raise ValueError("training file aggregate drifted")
        if checked_bytes != manifest.get("total_bytes"):
            raise ValueError("training byte total drifted")

        rows = policies.get("policies", [])
        if len(rows) != 48 or len({item["name"] for item in rows}) != 48:
            raise ValueError("training policies do not contain 48 milestone aliases")
        counts = {"ratio-gated": 0, "adaptation": 0, "combined": 0}
        for item in rows:
            variant = item.get("variant")
            if variant not in counts:
                raise ValueError("unexpected v9 training variant")
            checkpoint = Path(item["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = policies_path.parent / checkpoint
            checkpoint = checkpoint.resolve(strict=True)
            if root not in checkpoint.parents:
                raise ValueError(f"policy checkpoint escapes training root: {checkpoint}")
            counts[variant] += 1
        if counts != {"ratio-gated": 12, "adaptation": 18, "combined": 18}:
            raise ValueError(f"v9 milestone coverage drifted: {counts}")

        diagnostics = audit.get("development_scientific_diagnostics", {})
        if (
            audit.get("artifact_integrity", {}).get("passed") is not True
            or audit.get("artifact_integrity", {}).get("training_runs") != 9
            or len(audit.get("runs", [])) != 9
            or len(audit.get("scheduler_tasks", [])) != 9
            or len(audit.get("source_partition_sync", [])) != 2
            or any(item.get("exit_code") != 0 for item in audit["scheduler_tasks"])
            or diagnostics.get("stability_passed") is not False
            or [item.get("run") for item in diagnostics.get("stability_failures", [])]
            != ["h2oplus_v9_adaptation_seed6202"]
            or diagnostics.get("ratio_gate_relevant_runs") != 6
            or diagnostics.get("ratio_gate_open_runs") != []
            or diagnostics.get("ratio_intervention_ever_applied") is not False
            or diagnostics.get("intervention_attribution_available") is not False
            or closeout.get("artifact_integrity_passed") is not True
            or closeout.get("training_runs") != 9
            or closeout.get("milestone_aliases") != 48
            or closeout.get("formal_held_out_outcomes_read") is not False
        ):
            raise ValueError("audit/closeout assertions drifted")

        sync_hosts = set()
        for item in audit["source_partition_sync"]:
            evidence = item.get("evidence", {})
            path = Path(evidence.get("path", "")).resolve(strict=True)
            payload = load(path)
            if (
                path.stat().st_size != evidence.get("bytes")
                or sha256(path) != evidence.get("sha256")
                or payload.get("passed") is not True
                or payload.get("differences") != []
                or payload.get("source_host") != item.get("source_host")
                or payload.get("source_file_count") != item.get("source_file_count")
                or payload.get("source_total_bytes") != item.get("source_total_bytes")
            ):
                raise ValueError(f"source-sync evidence binding drifted: {path}")
            sync_hosts.add(item["source_host"])
        if sync_hosts != {"jtl110gpu", "jtl110gpu2"}:
            raise ValueError("source-sync evidence host coverage drifted")

        bindings = {
            "training_audit": (audit_path, sha256(audit_path)),
            "training_file_manifest": (manifest_path, sha256(manifest_path)),
            "training_policies": (policies_path, sha256(policies_path)),
        }
        for key, (path, digest) in bindings.items():
            value = closeout.get(key, {})
            if value.get("sha256") != digest or Path(value.get("path", "")).resolve() != path:
                raise ValueError(f"closeout binding drifted for {key}")
    except Exception as error:
        errors.append(str(error))
        manifest = {}
        counts = {}

    report = {
        "schema_version": REPORT_SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "training_root": str(root),
        "checked_files": checked_files,
        "checked_bytes": checked_bytes,
        "training_file_manifest_sha256": sha256(manifest_path),
        "training_aggregate_sha256": manifest.get("aggregate_sha256"),
        "training_audit_sha256": sha256(audit_path),
        "training_policies_sha256": sha256(policies_path),
        "training_closeout_sha256": sha256(closeout_path),
        "policy_counts": counts,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
