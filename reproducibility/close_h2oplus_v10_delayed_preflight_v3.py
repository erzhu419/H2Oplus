#!/usr/bin/env python3
"""Close corrected v10 deployment preflight from token-bound remote evidence."""

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
SUBMISSION = REPRO / "h2oplus_v10_delayed_preflight_submission_v5.json"
LOCAL_VERIFICATION = REPRO / "h2oplus_v10_delayed_deployment_verification_v3_local.json"
OUTPUT = REPRO / "h2oplus_v10_delayed_preflight_closeout_v3.json"
EVIDENCE = REPRO / "h2oplus_v10_delayed_preflight_backend_evidence_v3"
HOSTNAME_BY_NODE = {
    "jtl110gpu": "huiwei-Super-Server",
    "jtl110gpu2": "erzhu419-Super-Server",
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


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace preflight closeout: {OUTPUT}")
    submission = json.loads(SUBMISSION.read_text())
    if (
        submission.get("status") != "submitted"
        or submission.get("schema_version")
        != "transit-h2oplus-v10-delayed-preflight-submission-v5"
        or submission.get("remote_training_or_sumo_executed") is not False
        or submission.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v10 corrected preflight submission is stale")
    manifest = Path(submission["deployment_manifest"]["path"])
    manifest_sha = sha256(manifest)
    if manifest_sha != submission["deployment_manifest"]["sha256"]:
        raise ValueError("v10 corrected deployment manifest drifted")

    queue = json.loads(QUEUE.read_text())
    rows = queue if isinstance(queue, list) else queue.get("tasks", [])
    selected_ids = {item["task_id"] for item in submission["tasks"]}
    tasks = {row["id"]: row for row in rows if row.get("id") in selected_ids}
    if set(tasks) != selected_ids:
        raise ValueError("preflight queue evidence is incomplete")

    sys.path.insert(0, str(SCHEDULER_ROOT / "skill"))
    import scheduler  # type: ignore  # noqa: PLC0415

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    closed = []
    for submitted in submission["tasks"]:
        task = tasks[submitted["task_id"]]
        if task.get("status") != "done":
            raise RuntimeError(f"{task['id']} is not done: {task.get('status')}")
        node = submitted["node"]
        result_root = Path(submitted["result_root"])
        ok, message = scheduler._sync_one_result(
            {
                "node": node,
                "result_dir": str(result_root),
                "local_result_dir": str(result_root),
            }
        )
        if not ok:
            raise RuntimeError(f"{task['id']} result sync failed: {message}")

        archived = {}
        for kind, source in (
            ("backend", task["exit_status_path"]),
            ("log", task["log_path"]),
        ):
            rc, stdout, stderr = scheduler.run_on(
                node, f"base64 -w0 {source}", timeout=20, check=False
            )
            if rc:
                raise RuntimeError(
                    f"{task['id']} {kind} fetch failed: {stderr or stdout}"
                )
            target = EVIDENCE / f"{task['id']}.{kind}"
            target.write_bytes(base64.b64decode(stdout.strip()))
            archived[kind] = target

        fields = archived["backend"].read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != task.get("exit_status_token")
        ):
            raise ValueError(f"{task['id']} backend exit is not token-bound exit0")
        verification_path = Path(submitted["verification_output"])
        verification = json.loads(verification_path.read_text())
        if (
            verification.get("passed") is not True
            or verification.get("schema_version")
            != "transit-h2oplus-v10-delayed-deployment-verification-v3"
            or verification.get("verification_host") != HOSTNAME_BY_NODE[node]
            or verification.get("deployment_manifest_sha256") != manifest_sha
            or verification.get("overlay_entries") != 47
            or verification.get("prefix_jobs") != 5
            or verification.get("branch_jobs") != 25
            or verification.get("formal_held_out_outcomes_read") is not False
            or verification.get("local_training_or_sumo_executed") is not False
        ):
            raise ValueError(f"{task['id']} remote deployment verification failed")
        closed.append(
            {
                "task_id": task["id"],
                "node": node,
                "status": task["status"],
                "signature": task.get("signature"),
                "submitted_at": task.get("submitted_at"),
                "started_at": task.get("started_at"),
                "finished_at": task.get("finished_at"),
                "backend_returncode": 0,
                "backend_exit_epoch": int(fields[1]),
                "backend_token": fields[2],
                "backend_exit": descriptor(archived["backend"]),
                "scheduler_log": descriptor(archived["log"]),
                "verification": descriptor(verification_path),
                "result_sync": message,
            }
        )

    local = json.loads(LOCAL_VERIFICATION.read_text())
    if (
        local.get("passed") is not True
        or local.get("deployment_manifest_sha256") != manifest_sha
        or local.get("formal_held_out_outcomes_read") is not False
        or local.get("local_training_or_sumo_executed") is not False
    ):
        raise ValueError("local corrected deployment verification drifted")
    payload = {
        "schema_version": "transit-h2oplus-v10-delayed-preflight-closeout-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "deployment_manifest": descriptor(manifest),
        "submission": descriptor(SUBMISSION),
        "local_verification": descriptor(LOCAL_VERIFICATION),
        "valid_tasks": closed,
        "exact_remote_verifications": 2,
        "prefix_submission_gate_open": True,
        "branch_submission_gate_open": False,
        "development_evaluation_gate_open": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "generator": descriptor(Path(__file__).resolve()),
    }
    write(OUTPUT, payload)
    print(json.dumps(descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
