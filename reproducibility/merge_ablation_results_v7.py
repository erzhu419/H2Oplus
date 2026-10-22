#!/usr/bin/env python3
"""Merge disjoint verified-host v7 H2O+ ablation partitions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_training_matrix_v4 import (
    _is_complete,
    _serializable_job,
    load_plan,
)
from SimpleSAC.train_transit_v3 import sha256


SCHEMA_VERSION = "transit-v7-ablation-partition-merge-v1"
METHODS = ("h2oplus-no-ratio", "h2oplus-no-warmstart")
SEEDS = tuple(range(5101, 5106))
EXPECTED_JOBS = {
    f"{method}_seed{seed}" for method in METHODS for seed in SEEDS
}


def host_gpu_jobs(payload: dict[str, Any]) -> dict[str, list[str]]:
    if payload.get("status") != "completed":
        raise ValueError("host execution is not completed")
    assignments: dict[str, list[str]] = {}
    for gpu in payload.get("gpus", []):
        physical = str(gpu.get("physical_gpu"))
        if physical in assignments or gpu.get("status") != "completed":
            raise ValueError(f"invalid host GPU execution record: {physical}")
        gpu_jobs: list[str] = []
        for stage in gpu.get("stages", []):
            if stage.get("status") != "completed":
                raise ValueError(f"incomplete host execution stage: {stage.get('name')}")
            for worker in stage.get("workers", []):
                if worker.get("status") != "completed" or worker.get("returncode") != 0:
                    raise ValueError(f"failed host worker: {worker.get('name')}")
                gpu_jobs.extend(str(name) for name in worker.get("jobs", []))
        assignments[physical] = gpu_jobs
    jobs = [name for gpu_jobs in assignments.values() for name in gpu_jobs]
    if len(jobs) != len(set(jobs)):
        raise ValueError("host execution assigns an ablation job more than once")
    return assignments


def host_jobs(payload: dict[str, Any]) -> list[str]:
    assignments = host_gpu_jobs(payload)
    jobs = [name for gpu_jobs in assignments.values() for name in gpu_jobs]
    return sorted(jobs)


def dispatch_assignments(payload: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    if payload.get("status") != "frozen-before-formal-outcomes":
        raise ValueError("ablation dispatch is not frozen")
    assignments: dict[str, dict[str, list[str]]] = {}
    all_jobs: list[str] = []
    for host in payload.get("hosts", []):
        hostname = str(host.get("hostname"))
        if hostname in assignments:
            raise ValueError(f"duplicate ablation dispatch host: {hostname}")
        gpu_assignments: dict[str, list[str]] = {}
        for gpu in host.get("gpus", []):
            physical = str(gpu.get("physical_gpu"))
            if physical in gpu_assignments:
                raise ValueError(
                    f"duplicate ablation dispatch physical GPU: {hostname}/{physical}"
                )
            jobs = [
                str(name)
                for stage in gpu.get("stages", [])
                for worker in stage.get("workers", [])
                for name in worker.get("jobs", [])
            ]
            gpu_assignments[physical] = jobs
            all_jobs.extend(jobs)
        assignments[hostname] = gpu_assignments
    if len(all_jobs) != len(set(all_jobs)):
        raise ValueError("ablation dispatch assigns a job more than once")
    if set(all_jobs) != EXPECTED_JOBS:
        raise ValueError("ablation dispatch job union is not the exact registered matrix")
    return assignments


def relocated_job(
    job: dict[str, Any], run_dir: Path, initializer_root: Path
) -> dict[str, Any]:
    candidate = {**job, "run_dir": run_dir}
    if job.get("offline_initializer") is None:
        return candidate
    dependency = job.get("offline_initializer_dependency", {})
    expected_source = f"h2oplus_seed{job['seed']}"
    if dependency.get("source_job_name") != expected_source:
        raise ValueError(f"unexpected initializer source for {job['name']}")
    initializer = initializer_root / expected_source / "checkpoints" / "offline_initializer.pt"
    if not initializer.is_file():
        raise FileNotFoundError(
            f"canonical initializer is missing for {job['name']}: {initializer}"
        )
    candidate["offline_initializer"] = initializer
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def merge(
    plan_path: Path,
    dispatch_path: Path,
    partitions: list[tuple[str, Path, Path]],
    initializer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    dispatch_path = dispatch_path.expanduser().resolve()
    initializer_root = initializer_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace ablation matrix: {output_root}")
    jobs, evidence = load_plan(plan_path)
    by_name = {job["name"]: job for job in jobs}
    expected = set(by_name)
    if expected != EXPECTED_JOBS or len(jobs) != len(EXPECTED_JOBS):
        raise ValueError("ablation plan is not the exact registered 2-method x 5-seed matrix")
    dispatch_hash = sha256(dispatch_path)
    dispatch = json.loads(dispatch_path.read_text())
    if dispatch.get("plan", {}).get("sha256") != evidence["plan_sha256"]:
        raise ValueError("ablation dispatch does not bind the registered plan")
    expected_assignments = dispatch_assignments(dispatch)
    partition_hosts = [host for host, _, _ in partitions]
    if len(partition_hosts) != len(set(partition_hosts)):
        raise ValueError("ablation source partitions repeat a host")
    if set(partition_hosts) != set(expected_assignments):
        raise ValueError("ablation source hosts differ from the registered dispatch")
    assigned: set[str] = set()
    partition_records = []

    for host, source_root, execution_path in partitions:
        source_root = source_root.expanduser().resolve()
        execution_path = execution_path.expanduser().resolve()
        execution = json.loads(execution_path.read_text())
        if execution.get("hostname") != host:
            raise ValueError(f"host execution identity mismatch for {host}")
        if execution.get("plan_sha256") != evidence["plan_sha256"]:
            raise ValueError(f"host execution plan hash mismatch for {host}")
        if execution.get("dispatch_sha256") != dispatch_hash:
            raise ValueError(f"host execution dispatch hash mismatch for {host}")
        gpu_jobs = host_gpu_jobs(execution)
        if gpu_jobs != expected_assignments[host]:
            raise ValueError(f"host execution differs from registered lanes for {host}")
        names = sorted(name for jobs_for_gpu in gpu_jobs.values() for name in jobs_for_gpu)
        overlap = assigned.intersection(names)
        if overlap:
            raise ValueError(f"host ablation partitions overlap: {sorted(overlap)}")
        if not set(names) <= expected:
            raise ValueError(f"host partition contains unregistered jobs: {host}")
        for name in names:
            candidate = relocated_job(
                by_name[name], source_root / name, initializer_root
            )
            if not _is_complete(candidate, evidence):
                raise ValueError(f"source partition run is not formally complete: {name}")
        assigned.update(names)
        partition_records.append(
            {
                "host": host,
                "root": str(source_root),
                "host_execution": {
                    "path": str(execution_path),
                    "sha256": sha256(execution_path),
                },
                "jobs": names,
            }
        )
    if assigned != expected:
        raise ValueError(
            "ablation partition union is not exact: "
            f"missing={sorted(expected - assigned)}"
        )

    temporary = output_root.with_name(output_root.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial ablation matrix exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        source_for_job = {
            name: Path(partition["root"])
            for partition in partition_records
            for name in partition["jobs"]
        }
        for name in sorted(expected):
            shutil.copytree(
                source_for_job[name] / name,
                temporary / name,
                copy_function=os.link,
            )
        manifest = {
            "schema_version": evidence["matrix_schema"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **evidence,
            "output_root": str(output_root),
            "jobs": [_serializable_job(job) for job in jobs],
            "results": {
                name: {
                    "status": "completed",
                    "run_dir": str(output_root / name),
                    "source_host": next(
                        record["host"]
                        for record in partition_records
                        if name in record["jobs"]
                    ),
                }
                for name in sorted(expected)
            },
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "partition_merge": {
                "schema_version": SCHEMA_VERSION,
                "dispatch": {
                    "path": str(dispatch_path),
                    "sha256": dispatch_hash,
                },
                "initializer_root": {
                    "path": str(initializer_root),
                    "initializers": {
                        str(seed): {
                            "path": str(
                                initializer_root
                                / f"h2oplus_seed{seed}"
                                / "checkpoints"
                                / "offline_initializer.pt"
                            ),
                            "sha256": sha256(
                                initializer_root
                                / f"h2oplus_seed{seed}"
                                / "checkpoints"
                                / "offline_initializer.pt"
                            ),
                        }
                        for seed in SEEDS
                    },
                },
                "partitions": partition_records,
                "partition_runs_disjoint": sum(
                    len(record["jobs"]) for record in partition_records
                )
                == len(expected),
            },
        }
        _write_json(temporary / "matrix_manifest.json", manifest)
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument(
        "--partition",
        nargs=3,
        metavar=("HOST", "SOURCE_ROOT", "HOST_EXECUTION"),
        action="append",
        required=True,
    )
    parser.add_argument("--initializer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = merge(
        args.plan,
        args.dispatch,
        [(host, Path(root), Path(execution)) for host, root, execution in args.partition],
        args.initializer_root,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.expanduser().resolve()),
                "jobs": len(manifest["jobs"]),
                "manifest_sha256": sha256(
                    args.output_root.expanduser().resolve() / "matrix_manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
