#!/usr/bin/env python3
"""Build a file-level integrity manifest for merged formal v7 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.train_transit_v3 import sha256


SCHEMA_VERSION = "transit-formal-baseline-merge-manifest-v7-v1"
CHUNK_BYTES = 8 * 1024 * 1024
TASK_IDS = ("t82016", "t82017")


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _records(root: Path, names: list[str]) -> list[dict[str, Any]]:
    return [
        _record(path, root)
        for name in names
        for path in sorted((root / name).rglob("*"))
        if path.is_file()
    ]


def _records_sha256(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    matrix_root = args.matrix_root.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    dispatch_path = args.dispatch.expanduser().resolve()
    matrix_audit_path = args.matrix_audit.expanduser().resolve()
    stability_path = args.stability_audit.expanduser().resolve()
    scheduler_path = args.scheduler_evidence.expanduser().resolve()
    matrix_manifest_path = matrix_root / "matrix_manifest.json"
    required = (
        plan_path,
        dispatch_path,
        matrix_audit_path,
        stability_path,
        scheduler_path,
        matrix_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing baseline evidence: " + ", ".join(missing))

    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_manifest_path.read_text())
    audit = json.loads(matrix_audit_path.read_text())
    stability = json.loads(stability_path.read_text())
    scheduler = json.loads(scheduler_path.read_text())
    expected = sorted(job["name"] for job in plan.get("jobs", []))
    errors: list[str] = []

    if len(expected) != 35 or len(set(expected)) != 35:
        errors.append("baseline plan does not contain exactly 35 unique jobs")
    if matrix.get("status") != "completed":
        errors.append("baseline matrix manifest is not completed")
    if sorted(matrix.get("results", {})) != expected:
        errors.append("baseline matrix result set differs from the plan")
    if matrix.get("plan_sha256") != sha256(plan_path):
        errors.append("baseline matrix plan hash mismatch")
    merge = matrix.get("partition_merge", {})
    if merge.get("dispatch", {}).get("sha256") != sha256(dispatch_path):
        errors.append("baseline matrix dispatch hash mismatch")
    if not merge.get("partition_runs_disjoint"):
        errors.append("baseline source partitions are not disjoint")
    partition_jobs = [
        name for partition in merge.get("partitions", []) for name in partition["jobs"]
    ]
    if sorted(partition_jobs) != expected or len(partition_jobs) != len(set(partition_jobs)):
        errors.append("baseline source partition union is not exact")
    if not audit.get("passed") or audit.get("jobs_audited") != 35:
        errors.append("independent baseline matrix audit did not pass all 35 jobs")
    if stability.get("seeds_audited") != list(range(5101, 5106)):
        errors.append("RLPD long-horizon stability audit is incomplete")
    if stability.get("training_plan", {}).get("sha256") != sha256(plan_path):
        errors.append("RLPD stability audit plan hash mismatch")

    task_by_id = {task.get("id"): task for task in scheduler.get("tasks", [])}
    task_evidence = {}
    for task_id in TASK_IDS:
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"scheduler evidence lacks {task_id}")
            continue
        if task.get("status") != "done" or task.get("exit_code") != 0:
            errors.append(
                f"{task_id} status={task.get('status')!r} exit={task.get('exit_code')!r}"
            )
        task_evidence[task_id] = {
            key: task.get(key)
            for key in (
                "status",
                "exit_code",
                "node",
                "signature",
                "started_at",
                "finished_at",
                "exit_status_path",
                "exit_status_token",
            )
        }

    records = _records(matrix_root, expected)
    if not records:
        errors.append("merged baseline tree contains no run artifacts")
    source_partitions = {}
    merged_by_path = {record["path"]: record for record in records}
    seen_paths: set[str] = set()
    for partition in merge.get("partitions", []):
        host = partition["host"]
        root = Path(partition["root"]).resolve()
        names = sorted(partition["jobs"])
        source_records = _records(root, names)
        for record in source_records:
            if record["path"] in seen_paths:
                errors.append(f"source partitions overlap on {record['path']}")
            seen_paths.add(record["path"])
            if merged_by_path.get(record["path"]) != record:
                errors.append(f"merged baseline file differs from {host}: {record['path']}")
        source_partitions[host] = {
            "root": str(root),
            "jobs": names,
            "files": len(source_records),
            "bytes": sum(record["bytes"] for record in source_records),
            "records_sha256": _records_sha256(source_records),
            "host_execution": partition["host_execution"],
        }
    if seen_paths != set(merged_by_path):
        errors.append("source file union differs from merged baseline files")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not errors else "failed",
        "artifact_integrity_verified": not errors,
        "errors": errors,
        "campaign": "H2O+ v7 formal-v2 training-v7r1 corrected baselines",
        "frozen_inputs": {
            "plan": _descriptor(plan_path),
            "dispatch": _descriptor(dispatch_path),
        },
        "source_partitions": source_partitions,
        "partition_runs_disjoint": len(partition_jobs) == len(set(partition_jobs)),
        "merged_results": {
            "root": str(matrix_root),
            "runs": expected,
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records),
            "records_sha256": _records_sha256(records),
            "files_manifest": records,
        },
        "matrix_manifest": _descriptor(matrix_manifest_path),
        "independent_matrix_audit": {
            **_descriptor(matrix_audit_path),
            "passed": audit.get("passed"),
            "jobs_audited": audit.get("jobs_audited"),
        },
        "scientific_stability": {
            **_descriptor(stability_path),
            "passed": stability.get("scientific_stability_passed"),
            "failed_seeds": stability.get("failed_seeds"),
            "passed_seeds": stability.get("passed_seeds"),
            "interpretation": (
                "artifact integrity is independent of RLPD scientific stability; "
                "all preregistered outcomes remain in the baseline matrix"
            ),
        },
        "scheduler_evidence": {
            **_descriptor(scheduler_path),
            "tasks": task_evidence,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--matrix-audit", type=Path, required=True)
    parser.add_argument("--stability-audit", type=Path, required=True)
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    _write_json(args.output.expanduser().resolve(), manifest)
    print(args.output.expanduser().resolve())
    if manifest["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
