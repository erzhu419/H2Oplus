#!/usr/bin/env python3
"""Build the fail-closed file manifest for formal v7 H2O+ ablations."""

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


SCHEMA_VERSION = "transit-formal-ablation-merge-manifest-v7-v1"
METHODS = ("h2oplus-no-ratio", "h2oplus-no-warmstart")
ANALYSIS_METHODS = ("h2oplus",) + METHODS
SEEDS = tuple(range(5101, 5106))
EXPECTED_JOBS = {
    f"{method}_seed{seed}" for method in METHODS for seed in SEEDS
}
TASK_IDS = ("t82053", "t82054")


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


def _signed_job_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = matrix.get("jobs", [])
    by_name = {job["name"]: job for job in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("ablation matrix manifest repeats a signed job")
    return by_name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_analysis(
    analysis: dict[str, Any], matrix_root: Path, plan_by_name: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if analysis.get("scope") != "exploratory training-internal mechanism diagnosis":
        errors.append("ablation mechanism analysis has an unexpected scope")
    interpretation = analysis.get("scientific_interpretation", {})
    if interpretation.get("supports_held_out_model_effect_claim") is not False:
        errors.append("ablation analysis does not disclaim held-out effect claims")
    if interpretation.get("supports_formal_method_selection") is not False:
        errors.append("ablation analysis does not disclaim formal method selection")
    if tuple(analysis.get("methods", [])) != ANALYSIS_METHODS:
        errors.append("ablation analysis method set or order is incorrect")
    if analysis.get("seeds") != list(SEEDS):
        errors.append("ablation analysis seed set is incomplete")
    runs = analysis.get("runs", [])
    identities = [(run.get("method"), run.get("seed")) for run in runs]
    expected_identities = {
        (method, seed) for method in ANALYSIS_METHODS for seed in SEEDS
    }
    if len(identities) != len(expected_identities) or set(identities) != expected_identities:
        errors.append("ablation analysis does not contain the exact 15 paired runs")
        return errors

    for run in runs:
        method = run["method"]
        if method == "h2oplus":
            continue
        name = f"{method}_seed{run['seed']}"
        if run.get("run_signature") != plan_by_name[name].get("signature"):
            errors.append(f"ablation analysis run signature mismatch: {name}")
        artifacts = run.get("artifacts", {})
        for key, filename in (
            ("metrics", "metrics.jsonl"),
            ("summary", "summary.json"),
            ("metadata", "metadata.json"),
        ):
            local_path = matrix_root / name / filename
            if not local_path.is_file() or artifacts.get(key, {}).get("sha256") != sha256(
                local_path
            ):
                errors.append(f"ablation analysis artifact mismatch: {name}/{filename}")
    return errors


def build(args: argparse.Namespace) -> dict[str, Any]:
    matrix_root = args.matrix_root.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    dispatch_path = args.dispatch.expanduser().resolve()
    matrix_audit_path = args.matrix_audit.expanduser().resolve()
    analysis_path = args.mechanism_analysis.expanduser().resolve()
    scheduler_path = args.scheduler_evidence.expanduser().resolve()
    matrix_manifest_path = matrix_root / "matrix_manifest.json"
    required = (
        plan_path,
        dispatch_path,
        matrix_audit_path,
        analysis_path,
        scheduler_path,
        matrix_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing ablation evidence: " + ", ".join(missing))

    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_manifest_path.read_text())
    audit = json.loads(matrix_audit_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    scheduler = json.loads(scheduler_path.read_text())
    plan_by_name = {job["name"]: job for job in plan.get("jobs", [])}
    signed_by_name = _signed_job_map(matrix)
    expected = sorted(plan_by_name)
    errors: list[str] = []

    if set(expected) != EXPECTED_JOBS or len(expected) != len(EXPECTED_JOBS):
        errors.append("ablation plan is not the exact registered 10-job matrix")
    if matrix.get("status") != "completed":
        errors.append("ablation matrix manifest is not completed")
    if sorted(matrix.get("results", {})) != expected:
        errors.append("ablation matrix result set differs from the plan")
    if set(signed_by_name) != set(expected):
        errors.append("ablation matrix signed job set differs from the plan")
    if matrix.get("plan_sha256") != sha256(plan_path):
        errors.append("ablation matrix plan hash mismatch")
    merge = matrix.get("partition_merge", {})
    if merge.get("dispatch", {}).get("sha256") != sha256(dispatch_path):
        errors.append("ablation matrix dispatch hash mismatch")
    if not merge.get("partition_runs_disjoint"):
        errors.append("ablation source partitions are not disjoint")
    partition_jobs = [
        name for partition in merge.get("partitions", []) for name in partition["jobs"]
    ]
    if sorted(partition_jobs) != expected or len(partition_jobs) != len(set(partition_jobs)):
        errors.append("ablation source partition union is not exact")
    if not audit.get("passed") or audit.get("jobs_audited") != len(EXPECTED_JOBS):
        errors.append("independent ablation matrix audit did not pass all 10 jobs")
    if audit.get("plan", {}).get("sha256") != sha256(plan_path):
        errors.append("independent ablation audit plan hash mismatch")
    errors.extend(_validate_analysis(analysis, matrix_root, signed_by_name))

    task_by_id = {task.get("id"): task for task in scheduler.get("tasks", [])}
    task_evidence = {}
    for task_id in TASK_IDS:
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"scheduler evidence lacks {task_id}")
            continue
        completion = task.get("completion_evidence", {})
        queue_success = task.get("status") == "done" and task.get("exit_code") == 0
        direct_success = (
            completion.get("verified") is True
            and completion.get("status") == "verified"
            and completion.get("backend_exit", {}).get("exit_code") == 0
            and completion.get("host_execution", {}).get("completed") is True
        )
        if not (queue_success or direct_success):
            errors.append(
                f"{task_id} lacks queue or direct backend exit-0 completion evidence"
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
        task_evidence[task_id]["completion_evidence"] = completion or None

    records = _records(matrix_root, expected)
    if not records:
        errors.append("merged ablation tree contains no run artifacts")
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
                errors.append(f"merged ablation file differs from {host}: {record['path']}")
        source_partitions[host] = {
            "root": str(root),
            "jobs": names,
            "files": len(source_records),
            "bytes": sum(record["bytes"] for record in source_records),
            "records_sha256": _records_sha256(source_records),
            "host_execution": partition["host_execution"],
        }
    if seen_paths != set(merged_by_path):
        errors.append("source file union differs from merged ablation files")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not errors else "failed",
        "artifact_integrity_verified": not errors,
        "errors": errors,
        "campaign": "H2O+ v7 formal-v2 training-v7r1 registered ablations",
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
        "mechanism_analysis": {
            **_descriptor(analysis_path),
            "methods": analysis.get("methods"),
            "seeds": analysis.get("seeds"),
            "supports_held_out_model_effect_claim": analysis.get(
                "scientific_interpretation", {}
            ).get("supports_held_out_model_effect_claim"),
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
    parser.add_argument("--mechanism-analysis", type=Path, required=True)
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
