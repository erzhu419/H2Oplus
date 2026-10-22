#!/usr/bin/env python3
"""Build the token-bound scheduler closeout for formal task t84212."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
PARTITION = RESULTS / "evaluation_v4/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v4"
SUBMISSION = REPRO / "h2oplus_v20_formal_confirmation_submission_v4.json"
SYNC = REPRO / "h2oplus_v20_formal_confirmation_minimal_sync_v4.json"
QUEUE = Path.home() / ".claude/scheduler/queue.json"
BACKEND = EVIDENCE / "backend_exit_t84212.status"
LOG = EVIDENCE / "scheduler_t84212.log"
OUTPUT = EVIDENCE / "scheduler_evidence_v4.json"
TASK_ID = "t84212"
SIGNATURE = (
    "H2Oplus/v20/formal-confirmation-v4/6e7c16e278fc/"
    "single/jtl110gpu/glibc227"
)
TOKEN = "1787886815570439473:507459:136033519304704"
SUBMISSION_SHA256 = "6dea4c147d9ee987ef3f9bb970c0e9f5b79be6a9e85a8f8a92fc8e0365da35dc"
SYNC_SHA256 = "c593de405c43cf2c28085072fff593a5cd913526f89b60e8ee61d9b8aa1eed88"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite scheduler evidence: {OUTPUT}")
    if sha256(SUBMISSION) != SUBMISSION_SHA256 or sha256(SYNC) != SYNC_SHA256:
        raise ValueError("formal submission or minimal-sync identity drifted")

    submission = json.loads(SUBMISSION.read_text())
    sync = json.loads(SYNC.read_text())
    queue = json.loads(QUEUE.resolve(strict=True).read_text())
    matches = [row for row in queue.get("tasks", []) if row.get("id") == TASK_ID]
    if len(matches) != 1:
        raise ValueError("formal scheduler record is absent or duplicated")
    task = matches[0]
    backend_parts = BACKEND.resolve(strict=True).read_text().strip().split("\t")
    log_lines = LOG.resolve(strict=True).read_text(errors="replace").splitlines()
    text_errors = [
        line
        for line in log_lines
        if re.search(r"traceback|exception|\berror\b|failed|non-zero", line, re.I)
    ]
    host = json.loads((PARTITION / "host_execution.json").read_text())
    summary = json.loads((PARTITION / "host_summary.json").read_text())
    result_count = len(list(PARTITION.rglob("sumo_seed*.json")))
    manifest_count = len(list(PARTITION.rglob("matrix_manifest.json")))
    if (
        len(backend_parts) != 3
        or backend_parts[0] != "0"
        or backend_parts[2] != TOKEN
        or task.get("status") != "done"
        or task.get("signature") != SIGNATURE
        or task.get("node") != "jtl110gpu"
        or task.get("cpu_cores") != 12
        or task.get("ram_mb") != 32768
        or task.get("est_vram_mb") != 0
        or task.get("peak_vram_mb") != 0
        or task.get("exit_status_token") != TOKEN
        or task.get("exit_status_path")
        != "/tmp/scheduleurm_exit_t84212_40919f54117409638ce0697c.status"
        or task.get("result_sync_attempts") != 0
        or task.get("result_synced_at") is not None
        or submission.get("task", {}).get("task_id") != TASK_ID
        or submission.get("task_count") != 1
        or submission.get("total_remote_sumo_episodes") != 1350
        or submission.get("formal_held_out_outcome_access_authorized") is not True
        or submission.get("formal_held_out_outcomes_read") is not False
        or sync.get("passed") is not True
        or sync.get("task_id") != TASK_ID
        or sync.get("scientific_result_json_count") != 1350
        or sync.get("matrix_manifest_count") != 10
        or sync.get("checkpoint_optimizer_replay_or_workspace_files_synchronized")
        is not False
        or host.get("status") != "completed"
        or host.get("expected_results") != 1350
        or host.get("completed_results") != 1350
        or len(host.get("completed_blocks", [])) != 10
        or host.get("workers") != 12
        or host.get("training") is not False
        or host.get("formal_held_out_outcomes_read") is not False
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 1350
        or summary.get("formal_held_out_outcomes_read") is not False
        or result_count != 1350
        or manifest_count != 10
        or not any(line.strip() == "DONE" for line in log_lines)
        or text_errors
    ):
        raise ValueError("formal scheduler, backend, sync, or host closeout failed")

    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-scheduler-evidence-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "authorized_scientific_task": TASK_ID,
        "task": {
            "task_id": TASK_ID,
            "status": "done",
            "signature": SIGNATURE,
            "node": "jtl110gpu",
            "cpu_cores": 12,
            "ram_mb": 32768,
            "gpu_count": 0,
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "peak_ram_mb": task.get("peak_ram_mb"),
            "peak_vram_mb": task.get("peak_vram_mb"),
        },
        "backend_returncode": 0,
        "backend_finished_epoch": int(backend_parts[1]),
        "backend_token": TOKEN,
        "token_bound_backend_exit0": True,
        "host_execution_completed": True,
        "physical_result_count": 1350,
        "matrix_manifest_count": 10,
        "scheduler_done_marker_present": True,
        "text_error_count": 0,
        "scheduler_managed_result_sync_attempted": False,
        "minimal_allowlist_sync_passed": True,
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_outcomes_read_before_analysis": False,
        "local_sumo_executed": False,
        "bindings": {
            "submission": descriptor(SUBMISSION),
            "minimal_sync": descriptor(SYNC),
            "backend_exit": descriptor(BACKEND),
            "scheduler_log": descriptor(LOG),
            "host_execution": descriptor(PARTITION / "host_execution.json"),
            "host_summary": descriptor(PARTITION / "host_summary.json"),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
