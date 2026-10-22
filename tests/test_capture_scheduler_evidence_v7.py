import json
import subprocess
import sys
from pathlib import Path

from reproducibility.capture_scheduler_evidence_v7 import capture


def test_capture_selects_only_requested_tasks(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": {
                    "t1": {"id": "t1", "status": "done", "exit_code": 0},
                    "t2": {"id": "t2", "status": "queued"},
                    "unrelated": {"id": "unrelated", "status": "failed"},
                }
            }
        )
    )

    evidence = capture(queue, ("t1", "t2"))

    assert evidence["task_ids"] == ["t1", "t2"]
    assert [task["id"] for task in evidence["tasks"]] == ["t1", "t2"]
    assert evidence["tasks"][0]["exit_code"] == 0
    assert evidence["source"]["sha256"]


def test_capture_supports_scheduler_list_format(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "t1", "status": "done", "exit_code": 0},
                    {"id": "t2", "status": "queued"},
                ]
            }
        )
    )

    evidence = capture(queue, ("t2", "t1"))

    assert [task["id"] for task in evidence["tasks"]] == ["t2", "t1"]


def test_capture_attaches_verified_direct_completion(tmp_path):
    queue = tmp_path / "queue.json"
    token = "launch-token"
    queue.write_text(
        json.dumps(
            {
                "tasks": {
                    "t1": {
                        "id": "t1",
                        "status": "running",
                        "exit_status_token": token,
                        "cmd": "python reproducibility/run_training_dispatch_v7.py",
                    }
                }
            }
        )
    )
    backend = tmp_path / "backend.status"
    backend.write_text(f"0\t123.0\t{token}\n")
    execution = tmp_path / "host_execution.json"
    execution.write_text(
        json.dumps(
            {
                "status": "completed",
                "gpus": [
                    {
                        "status": "completed",
                        "stages": [
                            {
                                "status": "completed",
                                "workers": [
                                    {
                                        "status": "completed",
                                        "returncode": 0,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )

    evidence = capture(queue, ("t1",), {"t1": (backend, execution)})

    completion = evidence["tasks"][0]["completion_evidence"]
    assert completion["verified"] is True
    assert completion["backend_exit"]["exit_code"] == 0
    assert completion["host_execution"]["completed"] is True


def test_capture_backend_nonzero_overrides_completed_host_execution(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": {
                    "t1": {
                        "id": "t1",
                        "status": "running",
                        "exit_status_token": "token",
                        "cmd": "python run_training_dispatch_v7.py",
                    }
                }
            }
        )
    )
    backend = tmp_path / "backend.status"
    backend.write_text("1 123 token\n")
    execution = tmp_path / "host_execution.json"
    execution.write_text(
        json.dumps(
            {
                "status": "completed",
                "gpus": [
                    {
                        "status": "completed",
                        "stages": [
                            {
                                "status": "completed",
                                "workers": [
                                    {"status": "completed", "returncode": 0}
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )

    completion = capture(
        queue, ("t1",), {"t1": (backend, execution)}
    )["tasks"][0]["completion_evidence"]

    assert completion["verified"] is False
    assert completion["status"] == "backend-nonzero"


def test_capture_script_supports_direct_execution(tmp_path):
    queue = tmp_path / "queue.json"
    output = tmp_path / "evidence.json"
    queue.write_text(json.dumps({"tasks": {"t1": {"id": "t1", "status": "done"}}}))
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(root / "reproducibility" / "capture_scheduler_evidence_v7.py"),
            "--queue",
            str(queue),
            "--task-id",
            "t1",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(output.read_text())["task_ids"] == ["t1"]
