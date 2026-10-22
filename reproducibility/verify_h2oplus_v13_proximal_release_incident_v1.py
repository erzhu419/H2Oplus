#!/usr/bin/env python3
"""Independently verify the sealed v13 v1 pre-training admission incident."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT = (
    REPRO
    / "h2oplus_v13_proximal_release_incidents"
    / "v1_pretraining_admission_misbind"
)
RESULTS = (
    REPRO
    / "deployments/h2oplus_v13_proximal_release_pilot_v1/remote_results"
)
CLOSEOUT_SHA256 = (
    "bf1a22cfe221b9f774c39a53811fbe65e37aeb7a8310223acd81a0cb0057bfb3"
)
MANIFEST_SHA256 = (
    "8de9cf7486430cc36a4b14dda748a43723a16d95e9257cb8085f910594a2faac"
)
DIAGNOSIS_SHA256 = (
    "29fc70b5c97013ff84dd1df30762ffeace02a65ed6121c4f6ca1a6f05bb286ce"
)
SCHEDULER_SNAPSHOT_SHA256 = (
    "ac01b55e2593ec7aa5903130ea25a30489fa92c513c82d78964fd1219a04b6fe"
)
EXPECTED_BACKEND_FAILED = {
    "t83252",
    "t83253",
    "t83254",
    "t83255",
    "t83256",
    "t83257",
}
EXPECTED_STARTED_CANCELLED = {"t83258", "t83260"}
EXPECTED_QUEUED_CANCELLED = {
    "t83259",
    "t83261",
    "t83262",
    "t83263",
    "t83264",
    "t83265",
    "t83266",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_descriptor(value: dict[str, Any]) -> Path:
    path = Path(value["path"]).resolve(strict=True)
    if path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]:
        raise ValueError(f"incident descriptor drifted: {path}")
    return path


def main() -> int:
    closeout_path = INCIDENT / "incident_closeout_v1.json"
    manifest_path = INCIDENT / "incident_file_manifest_v1.json"
    diagnosis_path = INCIDENT / "admission_predicate_diagnosis_v1.json"
    scheduler_path = INCIDENT / "scheduler_terminal_snapshot_v1.json"
    expected_hashes = {
        closeout_path: CLOSEOUT_SHA256,
        manifest_path: MANIFEST_SHA256,
        diagnosis_path: DIAGNOSIS_SHA256,
        scheduler_path: SCHEDULER_SNAPSHOT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"sealed v13 v1 incident artifact drifted: {path}")

    closeout = json.loads(closeout_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    scheduler = json.loads(scheduler_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    backend_failed = set(closeout["backend_failed_before_training_task_ids"])
    started_cancelled = set(
        closeout["started_then_cancelled_without_backend_task_ids"]
    )
    queued_cancelled = set(closeout["cancelled_before_start_task_ids"])
    all_ids = {f"t{task}" for task in range(83252, 83267)}
    if (
        closeout.get("status") != "sealed-invalid-pretraining-lineage"
        or closeout.get("study_stage") != "development"
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("local_training_or_sumo_executed") is not False
        or closeout.get("remote_sumo_executed") is not False
        or closeout.get("v13_scientific_training_executed") is not False
        or closeout.get("scientific_result_registered") is not False
        or closeout.get("invalid_lineage_reusable") is not False
        or set(closeout.get("invalid_task_ids", [])) != all_ids
        or backend_failed != EXPECTED_BACKEND_FAILED
        or started_cancelled != EXPECTED_STARTED_CANCELLED
        or queued_cancelled != EXPECTED_QUEUED_CANCELLED
        or backend_failed & started_cancelled
        or backend_failed & queued_cancelled
        or started_cancelled & queued_cancelled
        or backend_failed | started_cancelled | queued_cancelled != all_ids
        or closeout.get("sole_failed_admission_predicate")
        != "v13_source_sha256"
        or closeout.get("materialized_run_count") != 8
    ):
        raise ValueError("v13 v1 incident closeout contract drifted")

    for key in (
        "submission",
        "submission_verification",
        "scheduler_snapshot",
        "admission_diagnosis",
        "file_manifest",
    ):
        verify_descriptor(closeout[key])
    if (
        diagnosis.get("failed_checks") != ["v13_source_sha256"]
        or diagnosis.get("runtime_only_source_paths") != []
        or diagnosis.get("shared_source_hash_mismatches") != []
        or set(diagnosis.get("admission_only_source_paths", []))
        != set(closeout["missing_runtime_bindings"])
        or diagnosis.get("scientific_training_executed") is not False
        or diagnosis.get("sumo_executed_by_diagnosis") is not False
        or diagnosis.get("formal_held_out_outcomes_read") is not False
        or any(
            value is not True
            for key, value in diagnosis["checks"].items()
            if key != "v13_source_sha256"
        )
        or diagnosis["checks"].get("v13_source_sha256") is not False
    ):
        raise ValueError("v13 v1 sole-predicate diagnosis drifted")

    queue_statuses = {row["id"]: row["status"] for row in scheduler["tasks"]}
    if (
        set(queue_statuses) != all_ids
        or {key for key, value in queue_statuses.items() if value == "failed"}
        != {"t83252", "t83253", "t83254", "t83255"}
        or {key for key, value in queue_statuses.items() if value == "cancelled"}
        != {f"t{task}" for task in range(83256, 83267)}
        or set(scheduler["backend_nonzero_overrides_queue_cancelled_for"])
        != {"t83256", "t83257"}
    ):
        raise ValueError("v13 v1 scheduler snapshot drifted")

    backend_rows = closeout["backend_evidence"]
    if (
        {row["task_id"] for row in backend_rows} != EXPECTED_BACKEND_FAILED
        or any(row["exit_code"] != 1 for row in backend_rows)
        or any(not row["launch_token"] for row in backend_rows)
    ):
        raise ValueError("v13 v1 backend evidence drifted")
    for row in backend_rows:
        verify_descriptor(row)

    log_rows = closeout["log_evidence"]
    if {row["task_id"] for row in log_rows} != (
        EXPECTED_BACKEND_FAILED | EXPECTED_STARTED_CANCELLED
    ):
        raise ValueError("v13 v1 log evidence set drifted")
    for row in log_rows:
        verify_descriptor(row)
        if row["contains_registered_error"] != (
            row["task_id"] in EXPECTED_BACKEND_FAILED
        ):
            raise ValueError("v13 v1 log classification drifted")

    materializations = closeout["materialized_v12_handoffs"]
    if (
        {row["task_id"] for row in materializations}
        != EXPECTED_BACKEND_FAILED | EXPECTED_STARTED_CANCELLED
        or any(row["total_updates"] != 55_000 for row in materializations)
        or any(row["online_critic_updates"] != 5_000 for row in materializations)
        or any(row["optimizer_updates_after_v12_handoff"] != 0 for row in materializations)
        or any(row["terminal_summary_present"] for row in materializations)
        or any(row["v13_policy_export_present"] for row in materializations)
        or any(row["host_execution_present"] for row in materializations)
    ):
        raise ValueError("v13 v1 materialization interpretation drifted")
    for row in materializations:
        verify_descriptor(row["materialization"])
        verify_descriptor(row["checkpoint"])

    roots = {
        "remote_results": RESULTS,
        "remote_evidence": INCIDENT / "remote_evidence",
    }
    observed = {}
    for scope, root in roots.items():
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            observed[(scope, path.relative_to(root).as_posix())] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    registered = {
        (row["scope"], row["path"]): {
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in manifest["entries"]
    }
    if (
        observed != registered
        or manifest.get("entry_count") != len(registered)
        or manifest.get("total_bytes")
        != sum(value["bytes"] for value in registered.values())
        or manifest.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v13 v1 incident file manifest drifted")

    payload = {
        "schema_version": "transit-h2oplus-v13-pretraining-admission-incident-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "closeout": descriptor(closeout_path),
        "file_manifest": descriptor(manifest_path),
        "admission_diagnosis": descriptor(diagnosis_path),
        "scheduler_snapshot": descriptor(scheduler_path),
        "verified_task_count": len(all_ids),
        "verified_backend_nonzero_count": len(EXPECTED_BACKEND_FAILED),
        "verified_materialized_v12_handoff_count": len(materializations),
        "verified_file_count": len(registered),
        "verified_file_bytes": sum(value["bytes"] for value in registered.values()),
        "v13_scientific_training_executed": False,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
    }
    output = INCIDENT / "incident_verification_v1.json"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
