#!/usr/bin/env python3
"""Run the immutable formal v7 full-service calibration trace matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "reproducibility" / "collect_full_service_trace_v7.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from collect_policy.build_transit_v7_dataset import load_shard
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
    _load_frozen_zero_audit,
    _zero_jobs,
)
from reproducibility.collect_full_service_trace_v7 import (
    TRACE_MATRIX_SCHEMA,
    TRACE_SCHEMA,
    canonical_signature,
    load_full_service_trace,
    trace_job_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--block-ids",
        type=int,
        nargs="+",
        help="execution-only subset for a pilot or resume; registration remains complete",
    )
    return parser.parse_args()


def _serializable_source(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": int(job["block_id"]),
        "realization_block_id": int(job["realization_block_id"]),
        "realization_block_name": job.get("realization_block_name"),
        "scenario": job["scenario"],
        "sumo_seed": int(job["sumo_seed"]),
        "signature": job["signature"],
        "output": job["output"],
    }


def build_registration(
    matrix_manifest: Path, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix_manifest = matrix_manifest.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    freeze_audit_path, _ = _load_frozen_zero_audit(matrix_manifest)
    matrix, source_jobs, partition = _zero_jobs(
        matrix_manifest, allow_partial=False
    )
    expected_blocks = set(FORMAL_CALIBRATION_REALIZATION_BLOCKS)
    if {int(job["realization_block_id"]) for job in source_jobs} != expected_blocks:
        raise ValueError("trace matrix does not contain the registered fit partition")

    runner_path = Path(__file__).resolve()
    collector_hash = sha256(COLLECTOR)
    runner_hash = sha256(runner_path)
    matrix_hash = sha256(matrix_manifest)
    jobs = []
    for source_job in sorted(source_jobs, key=lambda value: int(value["block_id"])):
        source_shard = Path(source_job["output"]).expanduser().resolve()
        source_summary, _ = load_shard(source_shard, allow_partial=False)
        if (
            source_summary.get("job_signature") != source_job.get("signature")
            or source_summary.get("shard_sha256") != sha256(source_shard)
        ):
            raise ValueError(f"frozen source shard changed: {source_shard}")
        realization = int(source_job["realization_block_id"])
        scale_tag = source_job["scenario"]["tag"]
        output = (
            output_root
            / f"realization_{realization:02d}"
            / scale_tag
            / f"sumo_seed{int(source_job['sumo_seed'])}.full_service.npz"
        )
        trace_signature = trace_job_signature(
            trace_collector_sha256=collector_hash,
            trace_runner_sha256=runner_hash,
            matrix_sha256=matrix_hash,
            source_job=source_job,
            source_shard_sha256=source_summary["shard_sha256"],
        )
        jobs.append(
            {
                "block_id": int(source_job["block_id"]),
                "realization_block_id": realization,
                "demand_scale": float(source_job["scenario"]["demand_scale"]),
                "sumo_seed": int(source_job["sumo_seed"]),
                "source_job_signature": source_job["signature"],
                "source_shard": str(source_shard),
                "source_shard_sha256": source_summary["shard_sha256"],
                "trace_signature": trace_signature,
                "output": str(output),
            }
        )
    expected_cells = int(partition["calibration_environment_cells"])
    if len(jobs) != expected_cells or len({job["trace_signature"] for job in jobs}) != len(jobs):
        raise ValueError("trace matrix is missing or duplicates a fit cell")
    registration = {
        "study_stage": "formal",
        "trace_schema": TRACE_SCHEMA,
        "frozen_zero_matrix": {
            "path": str(matrix_manifest),
            "sha256": matrix_hash,
            "sumo_version": matrix["sumo_version"],
        },
        "frozen_zero_matrix_audit": {
            "path": str(freeze_audit_path),
            "sha256": sha256(freeze_audit_path),
        },
        "trace_collector": {
            "path": str(COLLECTOR),
            "sha256": collector_hash,
        },
        "trace_runner": {
            "path": str(runner_path),
            "sha256": runner_hash,
        },
        "data_partition": partition,
        "source_jobs": [_serializable_source(job) for job in source_jobs],
        "jobs": jobs,
    }
    return registration, jobs


def register_matrix(
    matrix_manifest: Path, output_root: Path
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registration, jobs = build_registration(matrix_manifest, output_root)
    registration_hash = canonical_signature(registration)
    manifest_path = output_root / "trace_matrix_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != TRACE_MATRIX_SCHEMA
            or manifest.get("registration_sha256") != registration_hash
            or manifest.get("registration") != registration
        ):
            raise ValueError(
                "existing full-service trace registration differs from this run"
            )
        return manifest_path, manifest, jobs
    manifest = {
        "schema_version": TRACE_MATRIX_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": registration_hash,
        "registration": registration,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest_path, manifest, jobs


def _job_command(
    job: dict[str, Any], matrix_manifest: Path, trace_manifest: Path
) -> list[str]:
    runner_hash = sha256(Path(__file__).resolve())
    return [
        sys.executable,
        str(COLLECTOR),
        "--matrix-manifest",
        str(matrix_manifest),
        "--trace-matrix-manifest",
        str(trace_manifest),
        "--block-id",
        str(job["block_id"]),
        "--output",
        str(job["output"]),
        "--trace-signature",
        job["trace_signature"],
        "--trace-runner-sha256",
        runner_hash,
    ]


def _log_is_clean(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    text = path.read_text(errors="replace")
    fatal_markers = ("Traceback (most recent call last)", "RuntimeError:", "ValueError:")
    return not any(marker in text for marker in fatal_markers)


def _is_complete(
    job: dict[str, Any], matrix_manifest: Path, trace_manifest: Path
) -> bool:
    output = Path(job["output"])
    summary_path = output.with_suffix(output.suffix + ".json")
    log_path = output.with_suffix(output.suffix + ".log")
    if not output.is_file() or not summary_path.is_file() or not _log_is_clean(log_path):
        return False
    try:
        summary, arrays = load_full_service_trace(output)
        provenance = summary.get("provenance", {})
        return (
            summary.get("trace_signature") == job["trace_signature"]
            and int(summary.get("block_id", -1)) == int(job["block_id"])
            and int(summary.get("realization_block_id", -1))
            == int(job["realization_block_id"])
            and float(summary.get("demand_scale", -1.0)) == float(job["demand_scale"])
            and int(summary.get("sumo_seed", -1)) == int(job["sumo_seed"])
            and provenance.get("frozen_collection_matrix", {}).get("sha256")
            == sha256(matrix_manifest)
            and provenance.get("trace_matrix", {}).get("sha256")
            == sha256(trace_manifest)
            and provenance.get("source_shard", {}).get("sha256")
            == job["source_shard_sha256"]
            and int(summary.get("full_service_trace", {}).get("stop_events", -1))
            == len(arrays["stop_bus_code"])
            and int(summary.get("full_service_trace", {}).get("segments", -1))
            == len(arrays["segment_bus_code"])
        )
    except Exception:
        return False


def _quarantine(path: Path, output_root: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    relative = path.relative_to(output_root)
    destination = output_root / "quarantine" / timestamp / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)


def _run_job(
    job: dict[str, Any],
    matrix_manifest: Path,
    trace_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    if _is_complete(job, matrix_manifest, trace_manifest):
        return {"status": "skipped", "job": job}
    output = Path(job["output"])
    summary_path = output.with_suffix(output.suffix + ".json")
    log_path = output.with_suffix(output.suffix + ".log")
    for path in (output, summary_path, log_path):
        _quarantine(path, output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            _job_command(job, matrix_manifest, trace_manifest),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode or not _is_complete(job, matrix_manifest, trace_manifest):
        return {
            "status": "failed",
            "returncode": result.returncode,
            "log": str(log_path),
            "job": job,
        }
    return {"status": "completed", "job": job}


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("full-service trace worker count must be positive")
    matrix_manifest = args.matrix_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".trace_collection.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"full-service trace collection already running: {output_root}") from error

    try:
        trace_manifest, _, jobs = register_matrix(matrix_manifest, output_root)
        if args.block_ids is not None:
            requested = set(args.block_ids)
            registered = {int(job["block_id"]) for job in jobs}
            if not requested or not requested <= registered:
                raise ValueError(
                    f"requested trace blocks are not registered: {sorted(requested - registered)}"
                )
            jobs = [job for job in jobs if int(job["block_id"]) in requested]
        failures = []
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(jobs))
        ) as executor:
            futures = [
                executor.submit(
                    _run_job,
                    job,
                    matrix_manifest,
                    trace_manifest,
                    output_root,
                )
                for job in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                job = result["job"]
                print(
                    f"[{completed}/{len(jobs)}] {result['status']} "
                    f"block={job['block_id']} realization={job['realization_block_id']} "
                    f"demand={job['demand_scale']} sumo_seed={job['sumo_seed']}",
                    flush=True,
                )
                if result["status"] == "failed":
                    failures.append(result)
        if failures:
            details = "\n".join(
                f"{failure['job']['output']}: {failure['log']}"
                for failure in failures
            )
            raise RuntimeError(
                f"{len(failures)} full-service trace jobs failed:\n{details}"
            )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
