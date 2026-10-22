#!/usr/bin/env python3
"""Build and verify a file-level manifest for merged formal v7 core results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "transit-formal-core-merge-manifest-v7-v1"
RUN_NAME = re.compile(r"^h2oplus(?:-source)?_seed\d+$")
CHUNK_BYTES = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run_names(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and RUN_NAME.fullmatch(path.name)
    )


def run_file_records(root: Path, names: list[str]) -> list[dict[str, Any]]:
    return [
        file_record(path, root)
        for name in names
        for path in sorted((root / name).rglob("*"))
        if path.is_file()
    ]


def records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def host_execution_jobs(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    jobs: list[str] = []
    if payload.get("status") != "completed":
        errors.append(f"host status is {payload.get('status')!r}")
    physical_gpus: set[str] = set()
    for gpu in payload.get("gpus", []):
        physical = str(gpu.get("physical_gpu"))
        if physical in physical_gpus:
            errors.append(f"duplicate physical GPU {physical}")
        physical_gpus.add(physical)
        if gpu.get("status") != "completed":
            errors.append(f"GPU {physical} status is {gpu.get('status')!r}")
        for stage in gpu.get("stages", []):
            if stage.get("status") != "completed":
                errors.append(
                    f"GPU {physical} stage {stage.get('name')} is {stage.get('status')!r}"
                )
            for worker in stage.get("workers", []):
                if worker.get("status") != "completed" or worker.get("returncode") != 0:
                    errors.append(
                        f"worker {worker.get('name')} status={worker.get('status')!r} "
                        f"returncode={worker.get('returncode')!r}"
                    )
                jobs.extend(str(name) for name in worker.get("jobs", []))
    if len(jobs) != len(set(jobs)):
        errors.append("host execution assigns a job more than once")
    return sorted(jobs), errors


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.expanduser().resolve()
    matrix_root = formal_root / "h2oplus_core"
    matrix_manifest_path = matrix_root / "matrix_manifest.json"
    matrix_audit_path = formal_root / "core_matrix_audit.json"
    initializer_audit_path = formal_root / "core_initializer_audit.json"
    scheduler_evidence_path = formal_root / "scheduler_evidence" / "core_and_baseline_tasks.json"
    remote_audit_paths = {
        host: formal_root / "remote_initializer_audits" / f"{host}.json"
        for host in ("jtl110gpu", "jtl110gpu2")
    }
    required = [
        matrix_manifest_path,
        matrix_audit_path,
        initializer_audit_path,
        scheduler_evidence_path,
        *remote_audit_paths.values(),
        args.plan,
        args.dispatch,
        args.admission,
        args.deployment_manifest,
        args.training_source,
    ]
    missing = [str(path) for path in required if not path.expanduser().resolve().is_file()]
    if missing:
        raise FileNotFoundError("missing required evidence: " + ", ".join(missing))

    plan_path = args.plan.expanduser().resolve()
    dispatch_path = args.dispatch.expanduser().resolve()
    admission_path = args.admission.expanduser().resolve()
    deployment_path = args.deployment_manifest.expanduser().resolve()
    source_path = args.training_source.expanduser().resolve()
    matrix_manifest = json.loads(matrix_manifest_path.read_text())
    matrix_audit = json.loads(matrix_audit_path.read_text())
    initializer_audit = json.loads(initializer_audit_path.read_text())
    deployment = json.loads(deployment_path.read_text())
    scheduler_evidence = json.loads(scheduler_evidence_path.read_text())
    expected_jobs = sorted(job["name"] for job in matrix_manifest.get("jobs", []))
    errors: list[str] = []

    if matrix_manifest.get("status") != "completed":
        errors.append("matrix manifest is not completed")
    if sorted(matrix_manifest.get("results", {})) != expected_jobs:
        errors.append("matrix result set differs from matrix job set")
    if not matrix_audit.get("passed") or matrix_audit.get("jobs_audited") != len(expected_jobs):
        errors.append("independent matrix audit did not pass every expected job")
    if not initializer_audit.get("passed") or initializer_audit.get(
        "initializers_audited"
    ) != 5:
        errors.append("canonical initializer audit did not pass all five seeds")
    canonical_initializer_hashes = {
        name: record.get("sha256")
        for name, record in initializer_audit.get("initializers", {}).items()
    }
    remote_initializer_audits: dict[str, Any] = {}
    for host, audit_path in remote_audit_paths.items():
        report = json.loads(audit_path.read_text())
        remote_hashes = {
            name: record.get("sha256")
            for name, record in report.get("initializers", {}).items()
        }
        if (
            not report.get("passed")
            or report.get("initializers_audited") != 5
            or report.get("source_sha256_count") != 37
            or remote_hashes != canonical_initializer_hashes
        ):
            errors.append(f"{host}: remote initializer audit differs from canonical set")
        remote_initializer_audits[host] = {
            **descriptor(audit_path),
            "passed": report.get("passed"),
            "initializers_audited": report.get("initializers_audited"),
            "source_sha256_count": report.get("source_sha256_count"),
            "initializer_sha256": remote_hashes,
        }

    expected_hashes = {
        "plan": matrix_manifest.get("plan_sha256"),
        "dispatch": "e1666e91f907737bbdfd7106088d07b2a8eaa46847ec93d5f082edd89ba96537",
        "admission": matrix_manifest.get("training_admission", {}).get("sha256"),
        "training_source": matrix_manifest.get("source_sha256", {}).get(
            "SimpleSAC/train_transit_v3.py"
        ),
        "deployment": "0f82663f8248e343bbcffacb990341042102dbc46c748fc9ea769ceb126c1f78",
    }
    observed_hashes = {
        "plan": sha256(plan_path),
        "dispatch": sha256(dispatch_path),
        "admission": sha256(admission_path),
        "training_source": sha256(source_path),
        "deployment": deployment.get("aggregate_sha256"),
    }
    for key, expected in expected_hashes.items():
        if observed_hashes.get(key) != expected:
            errors.append(
                f"{key} hash mismatch: observed={observed_hashes.get(key)} expected={expected}"
            )

    merged_names = run_names(matrix_root)
    if merged_names != expected_jobs:
        errors.append(
            f"merged run set differs from plan: merged={merged_names} expected={expected_jobs}"
        )
    merged_records = run_file_records(matrix_root, merged_names)
    merged_by_path = {record["path"]: record for record in merged_records}
    if len(merged_by_path) != len(merged_records):
        errors.append("merged result tree contains duplicate relative paths")

    host_reports: dict[str, Any] = {}
    source_partitions: dict[str, Any] = {}
    partition_names: set[str] = set()
    partition_paths: set[str] = set()
    for host in ("jtl110gpu", "jtl110gpu2"):
        host_path = formal_root / "host_execution" / f"{host}.json"
        if not host_path.is_file():
            errors.append(f"missing host execution for {host}")
            continue
        host_payload = json.loads(host_path.read_text())
        assigned_jobs, host_errors = host_execution_jobs(host_payload)
        errors.extend(f"{host}: {error}" for error in host_errors)
        if host_payload.get("hostname") != host:
            errors.append(f"{host}: hostname field is {host_payload.get('hostname')!r}")
        if host_payload.get("plan_sha256") != observed_hashes["plan"]:
            errors.append(f"{host}: plan hash differs from merged plan")
        if host_payload.get("dispatch_sha256") != observed_hashes["dispatch"]:
            errors.append(f"{host}: dispatch hash differs from merged dispatch")

        incoming_root = formal_root / "incoming" / host / "h2oplus_core"
        incoming_names = run_names(incoming_root)
        if incoming_names != assigned_jobs:
            errors.append(
                f"{host}: incoming runs {incoming_names} differ from host jobs {assigned_jobs}"
            )
        overlap = partition_names.intersection(incoming_names)
        if overlap:
            errors.append(f"source partitions overlap on runs: {sorted(overlap)}")
        partition_names.update(incoming_names)
        records = run_file_records(incoming_root, incoming_names)
        for record in records:
            relative = record["path"]
            if relative in partition_paths:
                errors.append(f"source partitions overlap on file {relative}")
            partition_paths.add(relative)
            if merged_by_path.get(relative) != record:
                errors.append(f"merged file differs from source partition: {relative}")
        source_partitions[host] = {
            "root": str(incoming_root),
            "runs": incoming_names,
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records),
            "records_sha256": records_sha256(records),
        }
        host_reports[host] = {
            **descriptor(host_path),
            "status": host_payload.get("status"),
            "observed_hostname": host_payload.get("observed_hostname"),
            "physical_gpus": sorted(
                str(gpu.get("physical_gpu")) for gpu in host_payload.get("gpus", [])
            ),
            "jobs": assigned_jobs,
        }

    if sorted(partition_names) != expected_jobs:
        errors.append("source partition union differs from the signed plan")
    if partition_paths != set(merged_by_path):
        errors.append("source file union differs from the merged canonical result tree")

    task_status = {
        task.get("id"): task.get("status") for task in scheduler_evidence.get("tasks", [])
    }
    for task_id in ("t82043", "t82044"):
        if task_status.get(task_id) != "cancelled":
            errors.append(f"duplicate retry {task_id} was not cancelled")
    for task_id in ("t82016", "t82017"):
        if task_status.get(task_id) != "queued":
            errors.append(f"baseline gate task {task_id} was not queued during packaging")

    exit_markers: dict[str, Any] = {}
    for marker in sorted((formal_root / "scheduler_evidence").glob("*/*.status")):
        fields = marker.read_text().strip().split("\t")
        exit_code = int(fields[0]) if fields and fields[0].lstrip("-").isdigit() else None
        exit_markers[marker.name] = {
            **descriptor(marker),
            "exit_code": exit_code,
            "raw": marker.read_text().strip(),
        }
    for task_id in ("t82014", "t82015", "t82040", "t82041"):
        matches = [value for name, value in exit_markers.items() if task_id in name]
        if len(matches) != 1 or matches[0]["exit_code"] != 0:
            errors.append(f"{task_id} lacks exactly one zero exit marker")

    return {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not errors else "failed",
        "errors": errors,
        "campaign": "H2O+ v7 formal-v2 training-v7r1 corrected core",
        "frozen_inputs": {
            "plan": descriptor(plan_path),
            "dispatch": descriptor(dispatch_path),
            "training_admission": descriptor(admission_path),
            "training_source": descriptor(source_path),
            "deployment_manifest": {
                **descriptor(deployment_path),
                "aggregate_sha256": deployment.get("aggregate_sha256"),
                "entry_count": deployment.get("entry_count"),
            },
        },
        "source_partitions": source_partitions,
        "partition_runs_disjoint": len(partition_names) == sum(
            len(item["runs"]) for item in source_partitions.values()
        ),
        "host_execution": host_reports,
        "merged_results": {
            "root": str(matrix_root),
            "runs": merged_names,
            "files": len(merged_records),
            "bytes": sum(record["bytes"] for record in merged_records),
            "records_sha256": records_sha256(merged_records),
            "files_manifest": merged_records,
        },
        "matrix_manifest": descriptor(matrix_manifest_path),
        "independent_matrix_audit": {
            **descriptor(matrix_audit_path),
            "passed": matrix_audit.get("passed"),
            "jobs_audited": matrix_audit.get("jobs_audited"),
        },
        "initializer_audit": {
            **descriptor(initializer_audit_path),
            "passed": initializer_audit.get("passed"),
            "initializers": {
                name: {
                    "seed": record.get("seed"),
                    "bytes": record.get("bytes"),
                    "sha256": record.get("sha256"),
                    "source_run_signature": record.get("source_run_signature"),
                    "source_sha256_count": record.get("source_sha256_count"),
                    "training_admission_sha256": (
                        record.get("training_admission") or {}
                    ).get("sha256"),
                }
                for name, record in initializer_audit.get("initializers", {}).items()
            },
        },
        "remote_initializer_audits": remote_initializer_audits,
        "scheduler_discrepancy": {
            "evidence": descriptor(scheduler_evidence_path),
            "task_status_at_packaging": task_status,
            "exit_markers": exit_markers,
            "interpretation": (
                "t82014/t82015 and no-op retries t82040/t82041 have backend exit code 0; "
                "the scheduler failed labels came from missing top-level success-marker "
                "recognition and are not used as scientific completion evidence"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    _write_json(args.output, manifest)
    print(args.output.expanduser().resolve())
    if manifest["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
