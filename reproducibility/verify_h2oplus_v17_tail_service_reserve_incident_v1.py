#!/usr/bin/env python3
"""Independently verify the sealed v17 Python 3.8 compatibility incident."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INCIDENT = (
    ROOT
    / "reproducibility/h2oplus_v17_tail_service_reserve_results/"
    "evaluation_incidents/v1_python38_removeprefix"
)
TASK_IDS = {"t83728", "t83729", "t83730", "t83731"}
EXPECTED_ERROR = "AttributeError: 'str' object has no attribute 'removeprefix'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_relative(value: dict[str, Any]) -> Path:
    path = (INCIDENT / value["path"]).resolve(strict=True)
    if path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]:
        raise ValueError(f"incident descriptor drifted: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-closeout-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    closeout_path = INCIDENT / "incident_closeout_v1.json"
    manifest_path = INCIDENT / "incident_file_manifest_v1.json"
    if sha256(closeout_path) != args.expected_closeout_sha256:
        raise ValueError("v17 incident closeout identity drifted")
    if sha256(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("v17 incident file-manifest identity drifted")
    closeout = json.loads(closeout_path.read_text())
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v17-python38-incident-closeout-v1"
        or closeout.get("status") != "sealed-invalid-runtime-lineage"
        or set(closeout.get("invalid_task_ids", [])) != TASK_IDS
        or closeout.get("original_task_id") != "t83728"
        or set(closeout.get("automatic_retry_task_ids", []))
        != {"t83729", "t83730", "t83731"}
        or closeout.get("manual_duplicate_submission_count") != 0
        or closeout.get("backend_exit_code") != 1
        or closeout.get("failure") != EXPECTED_ERROR
        or closeout.get("remote_python_version") != "3.8.20"
        or closeout.get("preserved_incident_result_count") != 45
        or closeout.get("preserved_matrix_manifest_count") != 1
        or closeout.get("incident_results_reusable") is not False
        or closeout.get("incident_results_mergeable") is not False
        or closeout.get("incident_results_analyzable_for_selection") is not False
        or closeout.get("scientific_numeric_outcomes_inspected") is not False
        or closeout.get("scientific_gate_result_registered") is not False
        or closeout.get("algorithm_convergence_failure") is not False
        or closeout.get("frozen_v17_source_modified") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("local_training_or_sumo_executed") is not False
    ):
        raise ValueError("v17 incident closeout contract drifted")

    paths = {
        key: verify_relative(closeout[key])
        for key in (
            "backend_evidence",
            "scheduler_snapshot",
            "runtime_probe",
            "compatibility_diagnosis",
            "file_manifest",
        )
    }
    partition = closeout["partition_metadata_audit"]
    verify_relative(partition["host_execution"])
    verify_relative(partition["matrix_manifest"])
    if (
        partition.get("physical_result_file_count") != 45
        or partition.get("unique_job_signature_count") != 45
        or partition.get("source_file_count") != 53
        or partition.get("scientific_numeric_outcomes_inspected") is not False
    ):
        raise ValueError("v17 incident partition metadata drifted")
    scheduler = json.loads(paths["scheduler_snapshot"].read_text())
    runtime = json.loads(paths["runtime_probe"].read_text())
    diagnosis = json.loads(paths["compatibility_diagnosis"].read_text())
    backend = json.loads(paths["backend_evidence"].read_text())
    if (
        [task["id"] for task in scheduler.get("tasks", [])]
        != ["t83728", "t83729", "t83730", "t83731"]
        or scheduler.get("original_submission_count") != 1
        or scheduler.get("automatic_retry_count") != 3
        or scheduler.get("manual_duplicate_submission_count") != 0
        or any(task.get("status") != "failed" for task in scheduler["tasks"])
        or any(task.get("exit_code") != 1 for task in scheduler["tasks"])
        or runtime.get("python", {}).get("version") != "3.8.20"
        or runtime.get("active_v17_processes") != []
        or diagnosis.get("registered_error") != EXPECTED_ERROR
        or diagnosis.get("unsupported_api")
        != "str.removeprefix (introduced in Python 3.9)"
        or diagnosis.get("scientific_design_failure") is not False
        or diagnosis.get("algorithm_convergence_failure") is not False
        or diagnosis.get("frozen_v17_source_modified") is not False
        or {row["task_id"] for row in backend.get("tasks", [])} != TASK_IDS
        or any(row.get("exit_code") != 1 for row in backend["tasks"])
    ):
        raise ValueError("v17 incident evidence interpretation drifted")
    for row in backend["tasks"]:
        verify_relative(row["status"])
        verify_relative(row["log"])

    manifest = json.loads(manifest_path.read_text())
    excluded = {
        "incident_file_manifest_v1.json",
        "incident_closeout_v1.json",
        "incident_verification_v1.json",
    }
    observed = {}
    for path in sorted(value for value in INCIDENT.rglob("*") if value.is_file()):
        relative = path.relative_to(INCIDENT).as_posix()
        if relative in excluded:
            continue
        observed[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    registered = {
        row["path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
        for row in manifest.get("entries", [])
    }
    if (
        observed != registered
        or manifest.get("entry_count") != len(registered)
        or manifest.get("total_bytes")
        != sum(value["bytes"] for value in registered.values())
        or manifest.get("scientific_numeric_outcomes_inspected") is not False
        or manifest.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v17 incident file manifest drifted")
    result_count = len(list((INCIDENT / "remote_partition").rglob("sumo_seed*.json")))
    matrix_count = len(
        list((INCIDENT / "remote_partition").glob("*/matrix_manifest.json"))
    )
    if result_count != 45 or matrix_count != 1:
        raise ValueError("v17 preserved incident partition drifted")

    payload = {
        "schema_version": "transit-h2oplus-v17-python38-incident-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "closeout_sha256": sha256(closeout_path),
        "file_manifest_sha256": sha256(manifest_path),
        "verified_task_count": 4,
        "verified_backend_nonzero_count": 4,
        "verified_incident_result_count": 45,
        "verified_matrix_manifest_count": 1,
        "verified_file_count": len(registered),
        "scientific_numeric_outcomes_inspected": False,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
    }
    atomic_json(INCIDENT / "incident_verification_v1.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
