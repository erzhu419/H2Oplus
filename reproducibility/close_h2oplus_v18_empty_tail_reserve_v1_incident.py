#!/usr/bin/env python3
"""Seal the zero-result v18-v1 parser-contract incident."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-incident-closeout-v1"
TASK_ID = "t83803"
STATUS_TOKEN = "1787635285637061834:1528677:126605756510208"
STATUS_PATH = "/tmp/scheduleurm_exit_t83803_48131d817f06eaeb9e2edc57.status"
REMOTE_RESULT = (
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v18_empty_tail_reserve_evaluation_v1/remote_results_v1/"
    "single_jtl110gpu"
)
EXPECTED_ERROR = (
    "the following arguments are required: --expected-activation-start-fraction, "
    "--expected-activation-end-fraction, --expected-max-onboard-capacity-fraction"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite incident closeout: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v1.json",
    )
    parser.add_argument("--expected-submission-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v1_parser_contract/incident_closeout_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission_path = args.submission.expanduser().resolve(strict=True)
    if sha256(submission_path) != args.expected_submission_sha256:
        raise ValueError("v18-v1 submission identity drifted")
    submission = json.loads(submission_path.read_text())
    if (
        submission.get("status") != "submitted"
        or submission.get("task", {}).get("task_id") != TASK_ID
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("total_remote_sumo_episodes") != 135
    ):
        raise ValueError("v18-v1 submission scope drifted")

    queue_path = Path.home() / ".claude/scheduler/queue.json"
    queue = json.loads(queue_path.read_text())
    tasks = queue if isinstance(queue, list) else queue.get("tasks", queue.get("queue", []))
    matches = [task for task in tasks if task.get("id") == TASK_ID]
    if len(matches) != 1:
        raise ValueError("expected exactly one v18-v1 scheduler task")
    task = matches[0]
    if (
        task.get("status") != "failed"
        or task.get("exit_status_path") != STATUS_PATH
        or task.get("exit_status_token") != STATUS_TOKEN
        or task.get("node") != "jtl110gpu"
        or task.get("cpu_cores") != 12
        or task.get("ram_mb") != 32768
        or task.get("est_vram_mb") != 0
    ):
        raise ValueError("v18-v1 terminal scheduler evidence drifted")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    remote_code = r'''
import hashlib
import json
from pathlib import Path

root = Path(%r).resolve(strict=True)
status_path = Path(%r)
scheduler_log = Path('/tmp/sched_t83803.log')
expected_error = %r

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

entries = []
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    entries.append({
        'path': str(path.relative_to(root)),
        'sha256': digest(path),
        'size_bytes': path.stat().st_size,
    })
result_files = sorted(root.rglob('sumo_seed*.json'))
evaluator_logs = sorted(root.rglob('sumo_seed*.log'))
error_lines = []
for path in evaluator_logs:
    lines = path.read_text(errors='replace').splitlines()
    matches = [line for line in lines if expected_error in line]
    if len(matches) != 1:
        error_lines.append({'path': str(path.relative_to(root)), 'matches': matches})
host_path = root / 'host_execution.json'
host = json.loads(host_path.read_text())
manifests = sorted(root.rglob('matrix_manifest.json'))
matrix = json.loads(manifests[0].read_text()) if len(manifests) == 1 else {}
status_text = status_path.read_text(errors='replace').strip() if status_path.exists() else None
print(json.dumps({
    'root': str(root),
    'entries': entries,
    'entry_count': len(entries),
    'total_bytes': sum(row['size_bytes'] for row in entries),
    'result_file_count': len(result_files),
    'evaluator_log_count': len(evaluator_logs),
    'unexpected_evaluator_log_errors': error_lines,
    'matrix_manifest_count': len(manifests),
    'matrix_schema': matrix.get('schema_version'),
    'matrix_job_count': len(matrix.get('jobs', [])),
    'matrix_shard_count': matrix.get('matrix_partition', {}).get('shard_count'),
    'matrix_shard_index': matrix.get('matrix_partition', {}).get('shard_index'),
    'host_execution': {
        key: host.get(key) for key in (
            'schema_version', 'status', 'study_stage', 'screen_type', 'training',
            'formal_held_out_outcomes_read', 'workers', 'expected_results',
            'completed_results', 'completed_blocks', 'active_block', 'error'
        )
    },
    'status_path': str(status_path),
    'status_text': status_text,
    'scheduler_log': {
        'path': str(scheduler_log),
        'sha256': digest(scheduler_log),
        'size_bytes': scheduler_log.stat().st_size,
    },
}, sort_keys=True))
''' % (REMOTE_RESULT, STATUS_PATH, EXPECTED_ERROR)
    command = (
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python -c "
        + shlex.quote(remote_code)
    )
    returncode, stdout, stderr = scheduler.run_on(
        "jtl110gpu", command, timeout=120, check=False
    )
    if returncode != 0:
        raise RuntimeError(f"remote incident inventory failed: {stderr.strip()}")
    observation = json.loads(stdout.strip())
    host = observation.get("host_execution", {})
    expected_status = f"1\t1787635304\t{STATUS_TOKEN}"
    if (
        observation.get("result_file_count") != 0
        or observation.get("evaluator_log_count") != 45
        or observation.get("unexpected_evaluator_log_errors") != []
        or observation.get("matrix_manifest_count") != 1
        or observation.get("matrix_schema") != "transit-evaluation-matrix-v7"
        or observation.get("matrix_job_count") != 45
        or observation.get("matrix_shard_count") != 1
        or observation.get("matrix_shard_index") != 0
        or host.get("status") != "failed"
        or host.get("training") is not False
        or host.get("formal_held_out_outcomes_read") is not False
        or host.get("workers") != 12
        or host.get("expected_results") != 135
        or host.get("completed_results") != 0
        or host.get("completed_blocks") != []
        or observation.get("status_text") != expected_status
    ):
        raise ValueError("v18-v1 remote incident evidence drifted")

    payload = {
        "schema_version": SCHEMA,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed-invalid-zero-result-incident",
        "study_stage": "development",
        "task_id": TASK_ID,
        "node": "jtl110gpu",
        "formal_held_out_outcomes_read": False,
        "scientific_numeric_outcomes_read": False,
        "completed_scientific_result_count": 0,
        "training": False,
        "local_sumo_simulation_executed": False,
        "backend_exit": {
            "returncode": 1,
            "path": STATUS_PATH,
            "token": STATUS_TOKEN,
            "observed": observation["status_text"],
        },
        "scheduler": {
            key: task.get(key)
            for key in (
                "id",
                "status",
                "signature",
                "node",
                "remote_pids",
                "submitted_at",
                "started_at",
                "finished_at",
                "cpu_cores",
                "ram_mb",
                "est_vram_mb",
                "last_block_reason",
                "exit_status_path",
                "exit_status_token",
            )
        },
        "submission": {
            "path": str(submission_path),
            "sha256": args.expected_submission_sha256,
        },
        "root_cause": {
            "class": "nested-argparse-option-collision",
            "summary": (
                "v18 reused the parent v17 expected activation and onboard option "
                "names; the outer parser consumed them before the parent parser ran"
            ),
            "observed_error": EXPECTED_ERROR,
            "affected_jobs": 45,
            "sumo_rollouts_completed": 0,
        },
        "correction_boundary": {
            "same_identity_retry_permitted": False,
            "frozen_v18_v1_source_modification_permitted": False,
            "fresh_successor_identity_required": True,
            "permitted_change": (
                "rename only the three v18-specific expected empty-tail CLI options "
                "and add an end-to-end no-SUMO nested-parser smoke test"
            ),
            "scientific_design_change_permitted": False,
        },
        "remote_incident_inventory": observation,
        "v18_v1_results_reusable": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
