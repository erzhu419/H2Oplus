#!/usr/bin/env python3
"""Seal the v12 CUDA preflight after a scheduler terminal-classification error."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEDULER_ROOT = ROOT.parent.parent / "scheduleurm"
QUEUE = Path.home() / ".claude/scheduler/queue.json"
SUBMISSION = REPRO / "h2oplus_v12_staged_release_preflight_submission_v3.json"
PROTOCOL = REPRO / "h2oplus_v12_staged_release_protocol_v2.json"
PLAN = REPRO / "h2oplus_v12_staged_release_training_plan_v2.json"
REMOTE_DEPLOYMENT_VERIFICATION = (
    REPRO / "h2oplus_v12_staged_release_deployment_verification_v2_remote.json"
)
MANIFEST = (
    REPRO
    / "deployments/h2oplus_v12_staged_release_pilot_v2/workspace/"
    "deployment_manifest_v12_v2.json"
)
PREFLIGHT_CORRECTION = REPRO / "h2oplus_v12_staged_release_preflight_correction_v3.json"
OPERATIONAL_CORRECTION = (
    REPRO / "h2oplus_v12_staged_release_preflight_operational_correction_v4.json"
)
CLOSEOUT = REPRO / "h2oplus_v12_staged_release_preflight_closeout_v4.json"
EVIDENCE = (
    REPRO
    / "h2oplus_v12_staged_release_results/preflight_v3/backend_evidence_v4"
)

EXPECTED = {
    "submission": "60cc4038e8aa24cfb0cb696997d7caf9e1c018252d0fb0d3f188abf7d14d7684",
    "protocol": "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
    "plan": "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa",
    "manifest": "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9",
    "remote_deployment_verification": (
        "ddccb0b092515edb6774544c74a8156234f2f11e287c2d4809fb76aa931db6c9"
    ),
    "preflight_correction": (
        "3d8350feca804a4c54366cb2b9c71334400688b9567023d1c454472003e5a98e"
    ),
}
ORIGINALS = {
    "t83022": {
        "node": "jtl110gpu",
        "retry": "t83024",
        "seed": 6201,
        "pid": 29011,
        "retry_pid": 1066,
    },
    "t83023": {
        "node": "jtl110gpu2",
        "retry": "t83025",
        "seed": 6202,
        "pid": 1699174,
        "retry_pid": 1701956,
    },
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


def queue_rows() -> dict[str, dict[str, Any]]:
    payload = json.loads(QUEUE.read_text())
    raw = payload if isinstance(payload, list) else payload.get("tasks", [])
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    return {str(row["id"]): row for row in rows}


def fetch_remote_bytes(scheduler: Any, node: str, path: str) -> bytes:
    rc, stdout, stderr = scheduler.run_on(
        node, f"base64 -w0 {path}", timeout=30, check=False
    )
    if rc or not stdout.strip():
        raise RuntimeError(f"remote fetch failed on {node}: {path}: {stderr or stdout}")
    return base64.b64decode(stdout.strip(), validate=True)


def parse_backend(data: bytes, token: str, expected_code: int) -> tuple[int, str]:
    fields = data.decode().strip().split("\t")
    if len(fields) != 3 or fields[0] != str(expected_code) or fields[2] != token:
        raise ValueError("backend status is not bound to the expected token and exit code")
    return int(fields[1]), fields[2]


def main() -> int:
    for path in (OPERATIONAL_CORRECTION, CLOSEOUT, EVIDENCE):
        if path.exists():
            raise FileExistsError(f"refusing to replace frozen v12 preflight evidence: {path}")
    bound_paths = {
        "submission": SUBMISSION,
        "protocol": PROTOCOL,
        "plan": PLAN,
        "manifest": MANIFEST,
        "remote_deployment_verification": REMOTE_DEPLOYMENT_VERIFICATION,
        "preflight_correction": PREFLIGHT_CORRECTION,
    }
    for name, path in bound_paths.items():
        if sha256(path) != EXPECTED[name]:
            raise ValueError(f"frozen v12 {name} drifted")

    submission = json.loads(SUBMISSION.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v12-staged-release-preflight-submission-v3"
        or submission.get("status") != "submitted"
        or submission.get("training_executed") is not False
        or submission.get("sumo_simulation_executed") is not False
        or submission.get("formal_held_out_outcomes_read") is not False
        or {row.get("task_id") for row in submission.get("tasks", [])}
        != set(ORIGINALS)
    ):
        raise ValueError("v12 preflight submission is stale or incomplete")
    submitted = {row["task_id"]: row for row in submission["tasks"]}
    rows = queue_rows()
    selected_ids = set(ORIGINALS) | {row["retry"] for row in ORIGINALS.values()}
    if not selected_ids <= set(rows):
        raise ValueError("scheduler queue lacks the exact original/retry lineage")

    sys.path.insert(0, str(SCHEDULER_ROOT / "skill"))
    import scheduler  # type: ignore  # noqa: PLC0415
    from scheduler_failure.analysis import (  # type: ignore  # noqa: PLC0415
        success_patterns_for_task,
    )

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".v12-preflight-v4-", dir=str(EVIDENCE.parent))
    )
    try:
        snapshots = {task_id: rows[task_id] for task_id in sorted(selected_ids)}
        atomic_json(staging / "queue_tasks.json", snapshots)
        original_records = []
        retry_records = []
        for task_id, expected in ORIGINALS.items():
            task = rows[task_id]
            retry_id = expected["retry"]
            retry = rows[retry_id]
            submitted_row = submitted[task_id]
            if (
                task.get("status") != "failed"
                or task.get("node") != expected["node"]
                or task.get("signature") != submitted_row.get("signature")
                or task.get("exit_code") != 0
                or not str(task.get("last_block_reason", "")).startswith(
                    "died after only "
                )
                or not str(task.get("last_block_reason", "")).endswith(
                    "s with no success marker"
                )
                or retry.get("status") != "cancelled"
                or retry.get("parent_id") != task_id
                or retry.get("retry_count") != 1
                or retry.get("node") != expected["node"]
                or retry.get("signature") != task.get("signature")
                or retry.get("last_block_reason") != "user batch force-cancel"
            ):
                raise ValueError(f"scheduler lineage drifted for {task_id}/{retry_id}")

            paths = {
                "result": submitted_row["verification_output"],
                "backend": task["exit_status_path"],
                "log": task["log_path"],
                "retry_backend": retry["exit_status_path"],
                "retry_log": retry["log_path"],
            }
            archived: dict[str, Path] = {}
            for kind, remote_path in paths.items():
                target = staging / f"{task_id}.{kind}"
                target.write_bytes(
                    fetch_remote_bytes(scheduler, expected["node"], remote_path)
                )
                archived[kind] = target

            exit_epoch, backend_token = parse_backend(
                archived["backend"].read_bytes(), task["exit_status_token"], 0
            )
            retry_exit_epoch, retry_backend_token = parse_backend(
                archived["retry_backend"].read_bytes(),
                retry["exit_status_token"],
                1,
            )
            result = json.loads(archived["result"].read_text())
            if (
                result.get("schema_version")
                != "transit-h2oplus-v12-staged-release-cuda-preflight-v3"
                or result.get("passed") is not True
                or result.get("seed") != expected["seed"]
                or result.get("run_name") != submitted_row["run_name"]
                or result.get("run_signature") != submitted_row["run_signature"]
                or result.get("protocol_sha256") != EXPECTED["protocol"]
                or result.get("source_sha256") != protocol.get("v12_source_sha256")
                or result.get("cuda_visible_devices") != str(task.get("gpu_idx"))
                or result.get("torch_cuda_available") is not True
                or result.get("torch_cuda_device_count") != 1
                or result.get("stage") != "prefix"
                or result.get("variant") != "shared-prefix"
                or result.get("training_executed") is not False
                or result.get("sumo_simulation_executed") is not False
                or result.get("formal_held_out_outcomes_read") is not False
            ):
                raise ValueError(f"CUDA preflight payload failed for {task_id}")
            retry_log = archived["retry_log"].read_text()
            if "FileExistsError: v12-v3 CUDA preflight output already exists" not in retry_log:
                raise ValueError(f"retry did not fail closed on existing output: {retry_id}")
            log = archived["log"].read_text()
            matched_markers = [
                marker for marker in success_patterns_for_task(task) if marker in log
            ]
            if matched_markers:
                raise ValueError(
                    f"scheduler reason conflicts with log markers for {task_id}: "
                    f"{matched_markers}"
                )

            pids = [expected["pid"], expected["retry_pid"]]
            process_command = "; ".join(
                f"if kill -0 {pid} 2>/dev/null; then echo alive:{pid}; "
                f"else echo dead:{pid}; fi"
                for pid in pids
            )
            rc, stdout, stderr = scheduler.run_on(
                expected["node"], process_command, timeout=20, check=False
            )
            expected_lines = {f"dead:{pid}" for pid in pids}
            if rc or set(stdout.splitlines()) != expected_lines:
                raise RuntimeError(
                    f"preflight process lineage is still alive on {expected['node']}: "
                    f"{stdout or stderr}"
                )

            original_records.append(
                {
                    "task_id": task_id,
                    "node": expected["node"],
                    "scheduler_status": task["status"],
                    "scheduler_last_block_reason": task["last_block_reason"],
                    "scheduler_queue_exit_code": task.get("exit_code"),
                    "scheduler_success_markers_found": matched_markers,
                    "backend_returncode": 0,
                    "backend_exit_epoch": exit_epoch,
                    "backend_token": backend_token,
                    "remote_pid": expected["pid"],
                    "remote_pid_alive": False,
                    "result": descriptor(archived["result"]),
                    "backend": descriptor(archived["backend"]),
                    "scheduler_log": descriptor(archived["log"]),
                }
            )
            retry_records.append(
                {
                    "task_id": retry_id,
                    "parent_id": task_id,
                    "node": expected["node"],
                    "scheduler_status": retry["status"],
                    "scheduler_last_block_reason": retry["last_block_reason"],
                    "backend_returncode": 1,
                    "backend_exit_epoch": retry_exit_epoch,
                    "backend_token": retry_backend_token,
                    "remote_pid": expected["retry_pid"],
                    "remote_pid_alive": False,
                    "result_overwrite_prevented": True,
                    "backend": descriptor(archived["retry_backend"]),
                    "scheduler_log": descriptor(archived["retry_log"]),
                }
            )

        os.replace(staging, EVIDENCE)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    def relocated(item: dict[str, Any]) -> dict[str, Any]:
        updated = dict(item)
        for key in ("result", "backend", "scheduler_log"):
            if key in updated:
                updated[key] = descriptor(EVIDENCE / Path(updated[key]["path"]).name)
        return updated

    original_records = [relocated(item) for item in original_records]
    retry_records = [relocated(item) for item in retry_records]
    queue_snapshot = descriptor(EVIDENCE / "queue_tasks.json")
    correction = {
        "schema_version": (
            "transit-h2oplus-v12-staged-release-preflight-operational-correction-v4"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-after-terminal-classification-false-positive",
        "study_stage": "development",
        "incident": {
            "scheduler_classification": "failed",
            "root_cause": (
                "the 92/93-second preflight logs contained a valid passed JSON payload "
                "but none of the scheduler's training success substrings"
            ),
            "backend_precedence": (
                "exact token-bound backend exit0 plus immutable result evidence proves "
                "the original preflights completed successfully"
            ),
            "scheduler_code_changed": False,
            "preflight_rerun_authorized": False,
            "automatic_retries": ["t83024", "t83025"],
            "automatic_retries_cancelled": True,
            "automatic_retry_result_overwrite_prevented": True,
        },
        "submission": descriptor(SUBMISSION),
        "protocol": descriptor(PROTOCOL),
        "training_plan": descriptor(PLAN),
        "deployment_manifest": descriptor(MANIFEST),
        "remote_deployment_verification": descriptor(REMOTE_DEPLOYMENT_VERIFICATION),
        "preflight_v3_correction": descriptor(PREFLIGHT_CORRECTION),
        "queue_snapshot": queue_snapshot,
        "original_tasks": original_records,
        "cancelled_retries": retry_records,
        "training_executed": False,
        "sumo_simulation_executed": False,
        "formal_held_out_outcomes_read": False,
        "generator": descriptor(Path(__file__).resolve()),
    }
    atomic_json(OPERATIONAL_CORRECTION, correction)
    closeout = {
        "schema_version": "transit-h2oplus-v12-staged-release-preflight-closeout-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed-with-preserved-scheduler-terminal-correction",
        "study_stage": "development",
        "artifact_runtime_preflight_passed": True,
        "scheduler_terminal_classification_false_positive": True,
        "exact_original_backend_exit0": 2,
        "cancelled_automatic_retries": 2,
        "all_recorded_remote_processes_absent": True,
        "submission": descriptor(SUBMISSION),
        "operational_correction": descriptor(OPERATIONAL_CORRECTION),
        "deployment_manifest": descriptor(MANIFEST),
        "protocol": descriptor(PROTOCOL),
        "training_plan": descriptor(PLAN),
        "evidence_root": {
            "path": str(EVIDENCE.resolve()),
            "files": len(list(EVIDENCE.iterdir())),
        },
        "prefix_training_submission_gate_open": True,
        "branch_training_submission_gate_open": False,
        "development_evaluation_gate_open": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "training_executed_by_preflight": False,
        "sumo_simulation_executed_by_preflight": False,
        "formal_held_out_outcomes_read": False,
        "generator": descriptor(Path(__file__).resolve()),
    }
    atomic_json(CLOSEOUT, closeout)
    print(
        json.dumps(
            {
                "operational_correction": descriptor(OPERATIONAL_CORRECTION),
                "closeout": descriptor(CLOSEOUT),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
