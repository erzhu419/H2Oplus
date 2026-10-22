#!/usr/bin/env python3
"""Independently verify the corrected v12 CUDA preflight closeout."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEDULER_ROOT = ROOT.parent.parent / "scheduleurm"
QUEUE = Path.home() / ".claude/scheduler/queue.json"
CORRECTION = REPRO / "h2oplus_v12_staged_release_preflight_operational_correction_v4.json"
CLOSEOUT = REPRO / "h2oplus_v12_staged_release_preflight_closeout_v4.json"
OUTPUT = (
    REPRO / "h2oplus_v12_staged_release_preflight_closeout_verification_v4.json"
)
ORIGINALS = {
    "t83022": ("jtl110gpu", "t83024", 29011, 1066),
    "t83023": ("jtl110gpu2", "t83025", 1699174, 1701956),
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


def check_descriptor(value: dict[str, Any]) -> Path:
    path = Path(value["path"]).expanduser().resolve(strict=True)
    if value.get("sha256") != sha256(path) or value.get("bytes") != path.stat().st_size:
        raise ValueError(f"descriptor drifted: {path}")
    return path


def queue_rows() -> dict[str, dict[str, Any]]:
    payload = json.loads(QUEUE.read_text())
    raw = payload if isinstance(payload, list) else payload.get("tasks", [])
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    return {str(row["id"]): row for row in rows}


def fetch_remote(scheduler: Any, node: str, path: str) -> bytes:
    rc, stdout, stderr = scheduler.run_on(
        node, f"base64 -w0 {path}", timeout=30, check=False
    )
    if rc or not stdout.strip():
        raise RuntimeError(f"remote evidence fetch failed: {node}:{path}: {stderr}")
    return base64.b64decode(stdout.strip(), validate=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace independent verification: {OUTPUT}")
    correction = json.loads(CORRECTION.read_text())
    closeout = json.loads(CLOSEOUT.read_text())
    if (
        correction.get("schema_version")
        != "transit-h2oplus-v12-staged-release-preflight-operational-correction-v4"
        or correction.get("status")
        != "frozen-after-terminal-classification-false-positive"
        or correction.get("training_executed") is not False
        or correction.get("sumo_simulation_executed") is not False
        or correction.get("formal_held_out_outcomes_read") is not False
        or closeout.get("schema_version")
        != "transit-h2oplus-v12-staged-release-preflight-closeout-v4"
        or closeout.get("status")
        != "passed-with-preserved-scheduler-terminal-correction"
        or closeout.get("artifact_runtime_preflight_passed") is not True
        or closeout.get("scheduler_terminal_classification_false_positive")
        is not True
        or closeout.get("exact_original_backend_exit0") != 2
        or closeout.get("cancelled_automatic_retries") != 2
        or closeout.get("prefix_training_submission_gate_open") is not True
        or closeout.get("branch_training_submission_gate_open") is not False
        or closeout.get("development_evaluation_gate_open") is not False
        or closeout.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or closeout.get("training_executed_by_preflight") is not False
        or closeout.get("sumo_simulation_executed_by_preflight") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("corrected v12 preflight closeout gate failed")
    if closeout.get("operational_correction", {}).get("sha256") != sha256(CORRECTION):
        raise ValueError("closeout does not bind the correction")

    for key in (
        "submission",
        "protocol",
        "training_plan",
        "deployment_manifest",
        "remote_deployment_verification",
        "preflight_v3_correction",
        "queue_snapshot",
        "generator",
    ):
        check_descriptor(correction[key])
    for key in (
        "submission",
        "operational_correction",
        "deployment_manifest",
        "protocol",
        "training_plan",
        "generator",
    ):
        check_descriptor(closeout[key])

    originals = {row["task_id"]: row for row in correction["original_tasks"]}
    retries = {row["task_id"]: row for row in correction["cancelled_retries"]}
    if set(originals) != set(ORIGINALS) or set(retries) != {"t83024", "t83025"}:
        raise ValueError("closeout task lineage is incomplete")
    rows = queue_rows()
    sys.path.insert(0, str(SCHEDULER_ROOT / "skill"))
    import scheduler  # type: ignore  # noqa: PLC0415

    remote_checks = []
    for task_id, (node, retry_id, original_pid, retry_pid) in ORIGINALS.items():
        original = originals[task_id]
        retry = retries[retry_id]
        task = rows[task_id]
        retry_task = rows[retry_id]
        if (
            task.get("status") != "failed"
            or task.get("exit_code") != 0
            or task.get("last_block_reason") != original["scheduler_last_block_reason"]
            or retry_task.get("status") != "cancelled"
            or retry_task.get("parent_id") != task_id
            or retry_task.get("last_block_reason") != "user batch force-cancel"
        ):
            raise ValueError(f"terminal scheduler state drifted: {task_id}/{retry_id}")
        archived_result = check_descriptor(original["result"])
        archived_backend = check_descriptor(original["backend"])
        archived_log = check_descriptor(original["scheduler_log"])
        archived_retry_backend = check_descriptor(retry["backend"])
        archived_retry_log = check_descriptor(retry["scheduler_log"])
        comparisons = {
            "result": (task["result_dir"] + "/cuda_preflight.json", archived_result),
            "backend": (task["exit_status_path"], archived_backend),
            "log": (task["log_path"], archived_log),
            "retry_backend": (retry_task["exit_status_path"], archived_retry_backend),
            "retry_log": (retry_task["log_path"], archived_retry_log),
        }
        for kind, (remote_path, local_path) in comparisons.items():
            if fetch_remote(scheduler, node, remote_path) != local_path.read_bytes():
                raise ValueError(f"remote/local {kind} evidence differs for {task_id}")
        result = json.loads(archived_result.read_text())
        if (
            result.get("passed") is not True
            or result.get("training_executed") is not False
            or result.get("sumo_simulation_executed") is not False
            or result.get("formal_held_out_outcomes_read") is not False
        ):
            raise ValueError(f"scientific boundary failed in {task_id} result")
        original_fields = archived_backend.read_text().strip().split("\t")
        retry_fields = archived_retry_backend.read_text().strip().split("\t")
        if (
            original_fields != ["0", str(original["backend_exit_epoch"]), original["backend_token"]]
            or retry_fields
            != ["1", str(retry["backend_exit_epoch"]), retry["backend_token"]]
        ):
            raise ValueError(f"token-bound exit evidence drifted for {task_id}")
        command = "; ".join(
            f"if kill -0 {pid} 2>/dev/null; then echo alive:{pid}; "
            f"else echo dead:{pid}; fi"
            for pid in (original_pid, retry_pid)
        )
        rc, stdout, stderr = scheduler.run_on(node, command, timeout=20, check=False)
        expected_lines = {f"dead:{original_pid}", f"dead:{retry_pid}"}
        if rc or set(stdout.splitlines()) != expected_lines:
            raise RuntimeError(f"preflight process reappeared on {node}: {stdout or stderr}")
        remote_checks.append(
            {
                "task_id": task_id,
                "retry_id": retry_id,
                "node": node,
                "remote_bytes_match": True,
                "original_backend_returncode": 0,
                "retry_backend_returncode": 1,
                "all_recorded_processes_absent": True,
            }
        )

    payload = {
        "schema_version": (
            "transit-h2oplus-v12-staged-release-preflight-closeout-"
            "independent-verification-v4"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "study_stage": "development",
        "closeout": descriptor(CLOSEOUT),
        "operational_correction": descriptor(CORRECTION),
        "remote_checks": remote_checks,
        "exact_remote_verifications": 2,
        "prefix_training_submission_gate_open": True,
        "branch_training_submission_gate_open": False,
        "development_evaluation_gate_open": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_executed_by_verification": False,
        "sumo_simulation_executed_by_verification": False,
        "formal_held_out_outcomes_read": False,
        "verifier": descriptor(Path(__file__).resolve()),
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
