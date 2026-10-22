#!/usr/bin/env python3
"""Capture a bounded, read-only scheduler evidence snapshot for formal v7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "transit-scheduler-evidence-v7-v1"
DEFAULT_TASK_IDS = ("t82016", "t82017", "t82053", "t82054", "t82055", "t82056")
FIELDS = (
    "id",
    "status",
    "exit_code",
    "node",
    "signature",
    "submitted_at",
    "started_at",
    "backend_finished_at",
    "finished_at",
    "exit_status_path",
    "exit_status_token",
    "wait_for_files",
    "cmd",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_completed(payload: dict[str, Any]) -> bool:
    workers = [
        worker
        for gpu in payload.get("gpus", [])
        for stage in gpu.get("stages", [])
        for worker in stage.get("workers", [])
    ]
    return bool(workers) and payload.get("status") == "completed" and all(
        gpu.get("status") == "completed"
        and all(
            stage.get("status") == "completed"
            and all(
                worker.get("status") == "completed"
                and worker.get("returncode") == 0
                for worker in stage.get("workers", [])
            )
            for stage in gpu.get("stages", [])
        )
        for gpu in payload.get("gpus", [])
    )


def _completion_evidence(
    task: dict[str, Any], backend_path: Path, execution_path: Path
) -> dict[str, Any]:
    backend_path = backend_path.expanduser().resolve()
    execution_path = execution_path.expanduser().resolve()
    fields = backend_path.read_text().strip().split()
    if len(fields) != 3:
        raise ValueError(f"invalid token-bound backend exit file: {backend_path}")
    exit_code = int(fields[0])
    finished_at = float(fields[1])
    token = fields[2]
    execution = json.loads(execution_path.read_text())
    token_matches = token == task.get("exit_status_token")
    command_class_matches = "run_training_dispatch_v7.py" in str(task.get("cmd", ""))
    execution_completed = _execution_completed(execution)
    verified = (
        exit_code == 0
        and token_matches
        and command_class_matches
        and execution_completed
    )
    status = "verified" if verified else "failed"
    if exit_code != 0:
        status = "backend-nonzero"
    return {
        "status": status,
        "verified": verified,
        "command_class_matches": command_class_matches,
        "token_matches_queue": token_matches,
        "backend_exit": {
            "path": str(backend_path),
            "bytes": backend_path.stat().st_size,
            "sha256": _sha256(backend_path),
            "exit_code": exit_code,
            "finished_at": finished_at,
            "token": token,
        },
        "host_execution": {
            "path": str(execution_path),
            "bytes": execution_path.stat().st_size,
            "sha256": _sha256(execution_path),
            "status": execution.get("status"),
            "completed": execution_completed,
        },
    }


def capture(
    queue_path: Path,
    task_ids: tuple[str, ...],
    completion_files: dict[str, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    queue_path = queue_path.expanduser().resolve()
    raw_bytes = queue_path.read_bytes()
    raw = json.loads(raw_bytes)
    tasks = raw.get("tasks", {})
    if isinstance(tasks, list):
        task_map = {task["id"]: task for task in tasks}
        if len(task_map) != len(tasks):
            raise ValueError("scheduler queue contains duplicate task ids")
    elif isinstance(tasks, dict):
        task_map = tasks
    else:
        raise TypeError("scheduler queue tasks must be a list or mapping")
    missing = [task_id for task_id in task_ids if task_id not in task_map]
    if missing:
        raise KeyError("scheduler queue lacks tasks: " + ", ".join(missing))
    completion_files = completion_files or {}
    tasks = []
    for task_id in task_ids:
        task = {field: task_map[task_id].get(field) for field in FIELDS}
        if task_id in completion_files:
            task["completion_evidence"] = _completion_evidence(
                task, *completion_files[task_id]
            )
        tasks.append(task)
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(queue_path),
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
        "task_ids": list(task_ids),
        "tasks": tasks,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument(
        "--completion",
        action="append",
        metavar="TASK_ID=BACKEND_EXIT,HOST_EXECUTION",
        help="attach direct token-bound backend and host-execution evidence",
    )
    return parser.parse_args()


def _parse_completions(values: list[str] | None) -> dict[str, tuple[Path, Path]]:
    result: dict[str, tuple[Path, Path]] = {}
    for value in values or []:
        task_id, separator, paths = value.partition("=")
        backend, comma, execution = paths.partition(",")
        if not separator or not comma or not task_id or not backend or not execution:
            raise ValueError(
                "completion must be TASK_ID=BACKEND_EXIT,HOST_EXECUTION"
            )
        if task_id in result:
            raise ValueError(f"duplicate completion evidence for {task_id}")
        result[task_id] = (Path(backend), Path(execution))
    return result


def main() -> None:
    args = parse_args()
    task_ids = tuple(args.task_ids or DEFAULT_TASK_IDS)
    payload = capture(args.queue, task_ids, _parse_completions(args.completion))
    _write_json(args.output, payload)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
