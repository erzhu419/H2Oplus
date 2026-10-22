#!/usr/bin/env python3
"""Merge disjoint verified-host v7 baseline partitions into a local matrix."""

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


SCHEMA_VERSION = "transit-v7-baseline-partition-merge-v1"


def host_jobs(payload: dict[str, Any]) -> list[str]:
    if payload.get("status") != "completed":
        raise ValueError("host execution is not completed")
    jobs: list[str] = []
    physical_gpus: set[str] = set()
    for gpu in payload.get("gpus", []):
        physical = str(gpu.get("physical_gpu"))
        if physical in physical_gpus or gpu.get("status") != "completed":
            raise ValueError(f"invalid host GPU execution record: {physical}")
        physical_gpus.add(physical)
        for stage in gpu.get("stages", []):
            if stage.get("status") != "completed":
                raise ValueError(f"incomplete host execution stage: {stage.get('name')}")
            for worker in stage.get("workers", []):
                if worker.get("status") != "completed" or worker.get("returncode") != 0:
                    raise ValueError(f"failed host worker: {worker.get('name')}")
                jobs.extend(str(name) for name in worker.get("jobs", []))
    if len(jobs) != len(set(jobs)):
        raise ValueError("host execution assigns a baseline job more than once")
    return sorted(jobs)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def merge(
    plan_path: Path,
    dispatch_path: Path,
    partitions: list[tuple[str, Path, Path]],
    output_root: Path,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    dispatch_path = dispatch_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace baseline matrix: {output_root}")
    jobs, evidence = load_plan(plan_path)
    by_name = {job["name"]: job for job in jobs}
    expected = set(by_name)
    dispatch_hash = sha256(dispatch_path)
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
        names = host_jobs(execution)
        overlap = assigned.intersection(names)
        if overlap:
            raise ValueError(f"host baseline partitions overlap: {sorted(overlap)}")
        if not set(names) <= expected:
            raise ValueError(f"host partition contains unregistered jobs: {host}")
        for name in names:
            candidate = {**by_name[name], "run_dir": source_root / name}
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
            f"baseline partition union is not exact: missing={sorted(expected - assigned)}"
        )

    temporary = output_root.with_name(output_root.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial baseline matrix exists: {temporary}")
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
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = merge(
        args.plan,
        args.dispatch,
        [(host, Path(root), Path(execution)) for host, root, execution in args.partition],
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
