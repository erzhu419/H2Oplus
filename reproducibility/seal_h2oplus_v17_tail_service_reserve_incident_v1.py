#!/usr/bin/env python3
"""Seal the v17 Python 3.8 dispatch failure without analyzing its outcomes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT = (
    REPRO
    / "h2oplus_v17_tail_service_reserve_results/evaluation_incidents/"
    "v1_python38_removeprefix"
)
REMOTE_RUN = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v17_tail_service_reserve_evaluation_v1/remote_results_v1/"
    "single_jtl110gpu"
)
SUBMISSION = REPRO / "h2oplus_v17_tail_service_reserve_evaluation_submission_v1.json"
SOURCE_FREEZE = REPRO / "h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json"
DISPATCH = REPRO / "run_h2oplus_v17_tail_service_reserve_evaluation_dispatch.py"
SUBMISSION_SHA256 = "37795ac948a19bb49667d4a54b7304be6c10952a39a79430da3b18644aeda804"
SOURCE_FREEZE_SHA256 = "30e4b23219dc01c96b7eeaf89e4a8564716a887b9e525e8bcf65164ca8026f95"
TASK_IDS = ("t83728", "t83729", "t83730", "t83731")
EXPECTED_SIGNATURE = (
    "H2Oplus/v17/development/tail-service-reserve-eval/"
    "a4bb3706a8f4/single/jtl110gpu/glibc227"
)
EXPECTED_ERROR = "AttributeError: 'str' object has no attribute 'removeprefix'"
EXPECTED_BUG_LINE = (
    'variant = algorithm.removeprefix("h2oplus-v17-tail-service-reserve-")'
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def relative_descriptor(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def queue_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve(strict=True).read_text())
    tasks = payload.get("tasks", [])
    return list(tasks.values()) if isinstance(tasks, dict) else list(tasks)


def load_scheduler(scheduler_root: Path):
    sys.path.insert(0, str(scheduler_root / "skill"))
    return importlib.import_module("scheduler")


def run_remote(scheduler, command: str, timeout: int = 120) -> str:
    rc, out, err = scheduler.run_on(
        "jtl110gpu", command, timeout=timeout, check=False
    )
    if rc != 0:
        raise RuntimeError(f"remote evidence command failed: {err or out}")
    return out


def sync_run(scheduler, remote: Path, local: Path) -> None:
    """Install a compressed route-aware mirror in a new local directory."""

    local.parent.mkdir(parents=True, exist_ok=True)
    target = scheduler._ssh_target_for_node("jtl110gpu")
    shell = scheduler._ssh_rsync_shell_for_node("jtl110gpu")
    remote_command = (
        f"tar -C {shlex.quote(str(remote.parent))} -czf - "
        f"{shlex.quote(remote.name)}"
    )
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{local.name}.sync-", dir=local.parent)
    )
    try:
        import subprocess

        command = (
            f"{shell} {shlex.quote(target)} {shlex.quote(remote_command)} | "
            f"tar -xzf - -C {shlex.quote(str(staging_parent))}"
        )
        completed = subprocess.run(
            ["bash", "-o", "pipefail", "-c", command],
            text=True,
            capture_output=True,
            check=False,
        )
        staged = staging_parent / remote.name
        if completed.returncode != 0 or not staged.is_dir():
            raise RuntimeError(
                "route-aware incident sync failed: "
                + (completed.stderr or completed.stdout)[-2000:]
            )
        os.replace(staged, local)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def validate_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {task.get("id"): task for task in tasks}
    if set(TASK_IDS) - set(by_id):
        raise ValueError("scheduler queue lacks part of the v17 retry family")
    selected = [by_id[task_id] for task_id in TASK_IDS]
    for index, task in enumerate(selected):
        expected_parent = None if index == 0 else TASK_IDS[index - 1]
        expected_retry = None if index == 0 else index
        if (
            task.get("status") != "failed"
            or task.get("exit_code") != 1
            or task.get("signature") != EXPECTED_SIGNATURE
            or task.get("node") != "jtl110gpu"
            or task.get("require_node") != "jtl110gpu"
            or task.get("cpu_declared_cores") != 12
            or task.get("est_vram_mb") != 0
            or task.get("parent_id") != expected_parent
            or task.get("retry_count") != expected_retry
            or not task.get("exit_status_path")
            or not task.get("exit_status_token")
            or not task.get("log_path")
        ):
            raise ValueError(f"v17 retry-family binding drifted: {task['id']}")
    if [selected[index].get("requeued_as") for index in range(3)] != list(
        TASK_IDS[1:]
    ) or selected[-1].get("requeued_as") is not None:
        raise ValueError("v17 automatic retry chain drifted")
    return selected


def reduced_task(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "status",
        "exit_code",
        "signature",
        "node",
        "require_node",
        "cpu_declared_cores",
        "ram_mb",
        "est_vram_mb",
        "parent_id",
        "retry_count",
        "requeued_as",
        "submitted_at",
        "started_at",
        "finished_at",
        "exit_status_path",
        "exit_status_token",
        "log_path",
        "last_block_reason",
        "failure_category",
        "extra_env",
    )
    return {key: task.get(key) for key in keys}


def collect_remote_evidence(
    scheduler, tasks: list[dict[str, Any]], root: Path
) -> list[dict[str, Any]]:
    rows = []
    evidence = root / "remote_evidence/jtl110gpu/tmp"
    evidence.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        status_path = Path(task["exit_status_path"])
        log_path = Path(task["log_path"])
        status_text = run_remote(scheduler, f"cat {shlex.quote(str(status_path))}")
        log_text = run_remote(scheduler, f"cat {shlex.quote(str(log_path))}")
        fields = status_text.strip().split()
        if (
            len(fields) != 3
            or fields[0] != "1"
            or fields[2] != task["exit_status_token"]
            or EXPECTED_ERROR not in log_text
        ):
            raise ValueError(f"v17 backend evidence drifted: {task['id']}")
        local_status = evidence / status_path.name
        local_log = evidence / log_path.name
        local_status.write_text(status_text)
        local_log.write_text(log_text)
        rows.append(
            {
                "task_id": task["id"],
                "exit_code": 1,
                "exit_epoch": int(fields[1]),
                "launch_token": fields[2],
                "status": relative_descriptor(local_status, root),
                "log": relative_descriptor(local_log, root),
                "registered_error_present": True,
            }
        )
    return rows


def validate_partition(root: Path, incident_root: Path) -> dict[str, Any]:
    host_path = root / "host_execution.json"
    host = json.loads(host_path.read_text())
    results = sorted(root.rglob("sumo_seed*.json"))
    manifests = sorted(root.glob("*/matrix_manifest.json"))
    if (
        host.get("schema_version")
        != "transit-h2oplus-v17-tail-service-reserve-evaluation-host-execution-v1"
        or host.get("status") != "failed"
        or host.get("error") != EXPECTED_ERROR
        or host.get("formal_held_out_outcomes_read") is not False
        or host.get("training") is not False
        or host.get("workers") != 12
        or host.get("completed_blocks") != []
        or len(results) != 45
        or len(manifests) != 1
        or (root / "host_summary.json").exists()
    ):
        raise ValueError("v17 failed partition contract drifted")

    manifest = json.loads(manifests[0].read_text())
    jobs = manifest.get("jobs", [])
    source_hashes = manifest.get("source_sha256", {})
    partition = manifest.get("matrix_partition", {})
    if (
        manifest.get("schema_version") != "transit-evaluation-matrix-v7"
        or len(jobs) != 45
        or len(source_hashes) != 53
        or partition.get("full_job_count") != 45
        or partition.get("selected_job_count") != 45
        or partition.get("shard_count") != 1
        or partition.get("shard_index") != 0
    ):
        raise ValueError("v17 failed matrix manifest drifted")

    signatures = set()
    for job in jobs:
        output = Path(job["output"])
        local_output = root / output.relative_to(REMOTE_RUN)
        result = json.loads(local_output.read_text())
        method = job.get("method", {})
        algorithm = str(method.get("algorithm", ""))
        prefix = "h2oplus-v17-tail-service-reserve-"
        if not algorithm.startswith(prefix):
            raise ValueError("v17 incident method prefix drifted")
        reserve = result.get("policy", {}).get("tail_service_reserve", {})
        signature = job["signature"]
        if (
            result.get("schema_version") != "transit-evaluation-v7"
            or result.get("evaluation_job", {}).get("signature") != signature
            or result.get("source_sha256") != source_hashes
            or result.get("method") != method.get("name")
            or reserve.get("schema_version")
            != "transit-h2oplus-v17-tail-service-reserve-v1"
            or reserve.get("variant") != algorithm[len(prefix) :]
            or reserve.get("v17_protocol_sha256")
            != "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790"
            or reserve.get("checkpoint_selection_by_behavior") is not False
            or reserve.get("formal_held_out_outcomes_read") is not False
            or signature in signatures
        ):
            raise ValueError(f"v17 incident metadata binding drifted: {local_output}")
        signatures.add(signature)
    if len(signatures) != 45:
        raise ValueError("v17 incident signatures are incomplete")
    return {
        "host_execution": relative_descriptor(host_path, incident_root),
        "matrix_manifest": relative_descriptor(manifests[0], incident_root),
        "physical_result_file_count": 45,
        "unique_job_signature_count": 45,
        "source_file_count": 53,
        "scientific_numeric_outcomes_inspected": False,
    }


def file_manifest(root: Path) -> dict[str, Any]:
    excluded = {
        "incident_file_manifest_v1.json",
        "incident_closeout_v1.json",
        "incident_verification_v1.json",
    }
    entries = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        entries.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "transit-h2oplus-v17-python38-incident-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "formal_held_out_outcomes_read": False,
        "scientific_numeric_outcomes_inspected": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if INCIDENT.exists():
        raise FileExistsError(f"refusing to overwrite sealed incident: {INCIDENT}")
    if sha256(SUBMISSION) != SUBMISSION_SHA256:
        raise ValueError("v17 submission identity drifted")
    if sha256(SOURCE_FREEZE) != SOURCE_FREEZE_SHA256:
        raise ValueError("v17 source-freeze identity drifted")
    source_lines = DISPATCH.read_text().splitlines()
    if len(source_lines) < 385 or source_lines[384].strip() != EXPECTED_BUG_LINE:
        raise ValueError("registered v17 Python 3.8 incompatibility is not exact")

    selected = validate_tasks(queue_tasks(args.queue))
    scheduler = load_scheduler(args.scheduler_root.expanduser().resolve(strict=True))
    staging = INCIDENT.with_name(INCIDENT.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        sync_run(scheduler, REMOTE_RUN, staging / "remote_partition")
        partition = validate_partition(staging / "remote_partition", staging)
        backend = collect_remote_evidence(scheduler, selected, staging)
        runtime_text = run_remote(
            scheduler,
            "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python -c "
            + shlex.quote(
                "import json,platform,sys; "
                "print(json.dumps({'version':platform.python_version(),"
                "'implementation':platform.python_implementation(),"
                "'soabi':sys.implementation.cache_tag},sort_keys=True))"
            ),
        )
        runtime = json.loads(runtime_text)
        process_text = run_remote(
            scheduler,
            "ps -eo pid=,args= | grep -E "
            + shlex.quote(
                "run_h2oplus_v17_tail_service_reserve_evaluation_dispatch.py|"
                "run_evaluation_matrix_v17_tail_service_reserve.py|"
                "evaluate_transit_v17_tail_service_reserve.py"
            )
            + " | grep -v grep || true",
        )
        if runtime.get("version") != "3.8.20" or process_text.strip():
            raise ValueError("v17 incident runtime/process evidence drifted")

        scheduler_snapshot = {
            "schema_version": "transit-h2oplus-v17-python38-incident-scheduler-snapshot-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": [reduced_task(task) for task in selected],
            "original_submission_count": 1,
            "automatic_retry_count": 3,
            "manual_duplicate_submission_count": 0,
            "retry_chain": list(TASK_IDS),
            "all_backend_exit_codes": [1, 1, 1, 1],
            "formal_held_out_outcomes_read": False,
        }
        scheduler_path = staging / "scheduler_terminal_snapshot_v1.json"
        atomic_json(scheduler_path, scheduler_snapshot)
        runtime_path = staging / "python_runtime_probe_v1.json"
        atomic_json(
            runtime_path,
            {
                "schema_version": "transit-h2oplus-v17-python-runtime-probe-v1",
                "node": "jtl110gpu",
                "python": runtime,
                "active_v17_processes": [],
                "sumo_executed_by_probe": False,
            },
        )
        source_bug_path = staging / "source_compatibility_diagnosis_v1.json"
        atomic_json(
            source_bug_path,
            {
                "schema_version": "transit-h2oplus-v17-python38-compatibility-diagnosis-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "failure_class": "post-block-dispatch-runtime-compatibility",
                "python_version": "3.8.20",
                "unsupported_api": "str.removeprefix (introduced in Python 3.9)",
                "source_line": 385,
                "source_text": EXPECTED_BUG_LINE,
                "registered_error": EXPECTED_ERROR,
                "dispatch_source": descriptor(DISPATCH),
                "source_freeze": descriptor(SOURCE_FREEZE),
                "submission": descriptor(SUBMISSION),
                "scientific_design_failure": False,
                "algorithm_convergence_failure": False,
                "scientific_gate_result_registered": False,
                "frozen_v17_source_modified": False,
                "formal_held_out_outcomes_read": False,
                "local_training_or_sumo_executed": False,
            },
        )
        backend_path = staging / "backend_evidence_v1.json"
        atomic_json(
            backend_path,
            {
                "schema_version": "transit-h2oplus-v17-python38-incident-backend-evidence-v1",
                "tasks": backend,
                "backend_nonzero_overrides_host_state": True,
            },
        )
        manifest_path = staging / "incident_file_manifest_v1.json"
        atomic_json(manifest_path, file_manifest(staging))
        closeout_path = staging / "incident_closeout_v1.json"
        atomic_json(
            closeout_path,
            {
                "schema_version": "transit-h2oplus-v17-python38-incident-closeout-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "sealed-invalid-runtime-lineage",
                "study_stage": "development",
                "invalid_task_ids": list(TASK_IDS),
                "original_task_id": TASK_IDS[0],
                "automatic_retry_task_ids": list(TASK_IDS[1:]),
                "manual_duplicate_submission_count": 0,
                "backend_exit_code": 1,
                "failure": EXPECTED_ERROR,
                "failure_class": "post-block-dispatch-runtime-compatibility",
                "remote_python_version": "3.8.20",
                "preserved_incident_result_count": 45,
                "preserved_matrix_manifest_count": 1,
                "incident_results_reusable": False,
                "incident_results_mergeable": False,
                "incident_results_analyzable_for_selection": False,
                "scientific_numeric_outcomes_inspected": False,
                "scientific_gate_result_registered": False,
                "algorithm_convergence_failure": False,
                "frozen_v17_source_modified": False,
                "new_successor_required": True,
                "successor_requirement": (
                    "fresh registration, fresh output root, Python 3.8-compatible "
                    "dispatch, and no reuse of the 45 incident results"
                ),
                "partition_metadata_audit": partition,
                "backend_evidence": relative_descriptor(backend_path, staging),
                "scheduler_snapshot": relative_descriptor(scheduler_path, staging),
                "runtime_probe": relative_descriptor(runtime_path, staging),
                "compatibility_diagnosis": relative_descriptor(source_bug_path, staging),
                "file_manifest": relative_descriptor(manifest_path, staging),
                "formal_held_out_outcomes_read": False,
                "local_training_or_sumo_executed": False,
                "formal_confirmation_authorized_or_submitted": False,
            },
        )
        os.replace(staging, INCIDENT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    payload = json.loads((INCIDENT / "incident_closeout_v1.json").read_text())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
