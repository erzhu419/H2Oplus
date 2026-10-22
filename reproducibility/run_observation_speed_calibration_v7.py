#!/usr/bin/env python3
"""Run the hash-locked 5-by-3 SUMO observation-speed calibration matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
)
from reproducibility.collect_observation_speed_trace_v7 import TRACE_SCHEMA
from reproducibility.freeze_zero_collection_matrix_v7 import (
    AUDIT_SCHEMA as ZERO_AUDIT_SCHEMA,
    validate_formal_zero_manifest,
)


MATRIX_SCHEMA = "transit-simulator-observation-speed-matrix-v7-v1"
COLLECTOR = ROOT / "reproducibility" / "collect_observation_speed_trace_v7.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-manifest", type=Path, required=True)
    parser.add_argument("--zero-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    return parser.parse_args()


def _signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _command(job: dict) -> list[str]:
    return [
        sys.executable,
        str(COLLECTOR),
        "--scenario-manifest",
        job["scenario_manifest"],
        "--sumo-seed",
        str(job["sumo_seed"]),
        "--realization-block-id",
        str(job["realization_block_id"]),
        "--source-shard-summary",
        job["source_shard_summary"],
        "--source-job-signature",
        job["source_job_signature"],
        "--job-signature",
        job["signature"],
        "--output",
        job["output"],
        "--sumo-dir",
        job["sumo_dir"],
        "--edge-xml",
        job["edge_xml"],
    ]


def _is_complete(job: dict) -> bool:
    trace = Path(job["output"])
    summary_path = trace.with_suffix(trace.suffix + ".json")
    if not trace.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        with __import__("numpy").load(trace, allow_pickle=False) as arrays:
            count = len(arrays["segment_mean_speed_mps"])
            aligned = all(len(arrays[name]) == count for name in arrays.files)
        return bool(
            summary.get("schema_version") == TRACE_SCHEMA
            and summary.get("job_signature") == job["signature"]
            and summary.get("trace_sha256") == sha256(trace)
            and summary.get("coverage_gate", {}).get("passed")
            and summary.get("events", {}).get("control_decisions") == count
            and aligned
        )
    except Exception:
        return False


def _run(job: dict) -> dict:
    if _is_complete(job):
        return {"status": "skipped", "job": job}
    output = Path(job["output"])
    summary = output.with_suffix(output.suffix + ".json")
    log = output.with_suffix(output.suffix + ".log")
    if output.exists() or summary.exists():
        return {"status": "invalid-existing", "job": job}
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(
            _command(job),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "status": "completed" if result.returncode == 0 and _is_complete(job) else "failed",
        "returncode": result.returncode,
        "log": str(log),
        "job": job,
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    zero_manifest_path = args.zero_manifest.expanduser().resolve()
    zero_audit_path = args.zero_audit.expanduser().resolve()
    zero_manifest = json.loads(zero_manifest_path.read_text())
    topology = validate_formal_zero_manifest(zero_manifest)
    zero_audit = json.loads(zero_audit_path.read_text())
    if (
        zero_audit.get("schema_version") != ZERO_AUDIT_SCHEMA
        or zero_audit.get("passed") is not True
        or zero_audit.get("frozen_manifest", {}).get("sha256")
        != sha256(zero_manifest_path)
    ):
        raise ValueError("frozen zero-hold audit is stale or failed")
    expected_blocks = set(FORMAL_CALIBRATION_REALIZATION_BLOCKS)
    if set(topology["calibration_realization_blocks"]) != expected_blocks:
        raise ValueError("zero-hold calibration partition changed")

    audit_rows = {int(row["block_id"]): row for row in zero_audit["shards"]}
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".collection.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"speed calibration already running: {output_root}") from error

    collector_hash = sha256(COLLECTOR)
    runner_hash = sha256(Path(__file__).resolve())
    jobs = []
    for source_job in zero_manifest["jobs"]:
        realization = int(source_job["realization_block_id"])
        if realization not in expected_blocks:
            continue
        block_id = int(source_job["block_id"])
        audit_row = audit_rows[block_id]
        source_output = Path(source_job["output"])
        source_summary = source_output.with_suffix(source_output.suffix + ".json")
        if (
            audit_row["summary"]["sha256"] != sha256(source_summary)
            or audit_row["shard"]["sha256"] != sha256(source_output)
            or audit_row["job_signature"] != source_job["signature"]
        ):
            raise ValueError(f"frozen zero-hold source changed for block {block_id}")
        scenario = source_job["scenario"]
        output = (
            output_root
            / f"realization_{realization}"
            / scenario["tag"]
            / f"sumo_seed{source_job['sumo_seed']}.npz"
        )
        payload = {
            "collector_sha256": collector_hash,
            "runner_sha256": runner_hash,
            "zero_manifest_sha256": sha256(zero_manifest_path),
            "zero_audit_sha256": sha256(zero_audit_path),
            "source_job_signature": source_job["signature"],
            "source_shard_sha256": audit_row["shard"]["sha256"],
            "source_summary_sha256": audit_row["summary"]["sha256"],
            "realization_block_id": realization,
            "demand_scale": float(scenario["demand_scale"]),
            "scenario_manifest_sha256": scenario["sha256"],
            "sumo_seed": int(source_job["sumo_seed"]),
        }
        jobs.append(
            {
                **payload,
                "block_id": block_id,
                "signature": _signature(payload),
                "scenario_manifest": scenario["path"],
                "source_shard_summary": str(source_summary),
                "source_job_signature": source_job["signature"],
                "sumo_dir": source_job["sumo_dir"],
                "edge_xml": source_job["edge_xml"],
                "output": str(output),
            }
        )
    if len(jobs) != 15:
        raise ValueError(f"expected 15 calibration cells, found {len(jobs)}")

    matrix_path = output_root / "matrix_manifest.json"
    base_manifest = {
        "schema_version": MATRIX_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal-calibration",
        "collector": str(COLLECTOR),
        "collector_sha256": collector_hash,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": runner_hash,
        "zero_manifest": {
            "path": str(zero_manifest_path),
            "sha256": sha256(zero_manifest_path),
        },
        "zero_audit": {
            "path": str(zero_audit_path),
            "sha256": sha256(zero_audit_path),
        },
        "calibration_realization_blocks": sorted(expected_blocks),
        "reserved_fidelity_realization_blocks": topology[
            "reserved_fidelity_realization_blocks"
        ],
        "demand_scales": topology["demand_scales"],
        "environment_cells": len(jobs),
        "workers": args.workers,
        "jobs": jobs,
    }
    if matrix_path.exists():
        existing = json.loads(matrix_path.read_text())
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        current = dict(base_manifest)
        current.pop("created_utc", None)
        if comparable != current:
            raise ValueError("existing speed-calibration matrix manifest changed")
    else:
        _atomic_write(matrix_path, base_manifest)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "block_id": result["job"]["block_id"],
                        "output": result["job"]["output"],
                    }
                ),
                flush=True,
            )
    failures = [result for result in results if result["status"] not in {"completed", "skipped"}]
    if failures:
        raise RuntimeError(
            "observation-speed collection failed: "
            + json.dumps(
                [
                    {
                        "block_id": result["job"]["block_id"],
                        "status": result["status"],
                        "log": result.get("log"),
                    }
                    for result in failures
                ]
            )
        )
    if not all(_is_complete(job) for job in jobs):
        raise RuntimeError("observation-speed matrix did not close cleanly")
    print(json.dumps({"passed": True, "jobs": len(jobs), "manifest": str(matrix_path)}))


if __name__ == "__main__":
    main()
