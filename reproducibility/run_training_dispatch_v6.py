#!/usr/bin/env python3
"""Validate and execute a host/GPU-locked v6/v7 training dispatch manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_training_matrix_v4 import load_plan, signature
from SimpleSAC.train_transit_v3 import sha256


DISPATCH_SCHEMA = "transit-training-dispatch-v1"
EXECUTION_SCHEMA = "transit-training-dispatch-execution-v1"
CAPACITY_AUDIT_SCHEMA = "transit-v7-gpu-capacity-audit-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def job_set_sha256(jobs: list[dict[str, Any]]) -> str:
    return signature(
        {
            "jobs": [
                {
                    "name": job["name"],
                    "method": job["method"],
                    "seed": job["seed"],
                    "signature": job["signature"],
                    "max_canonical_processes_per_gpu": job[
                        "max_canonical_processes_per_gpu"
                    ],
                }
                for job in sorted(jobs, key=lambda item: item["name"])
            ]
        }
    )


def _safe_id(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{location} must be a safe non-empty identifier")
    return value


def _resolve_plan(dispatch_path: Path, raw: Any) -> Path:
    if not isinstance(raw, dict):
        raise ValueError("dispatch plan descriptor must be an object")
    configured = raw.get("path")
    if not isinstance(configured, str) or not configured:
        raise ValueError("dispatch plan path must be a non-empty string")
    plan = Path(configured).expanduser()
    if not plan.is_absolute():
        plan = dispatch_path.parent / plan
    plan = plan.resolve()
    if not plan.is_file():
        raise FileNotFoundError(f"dispatch training plan is missing: {plan}")
    if raw.get("sha256") != sha256(plan):
        raise RuntimeError("dispatch training-plan SHA-256 does not match")
    return plan


def _resolve_artifact(
    dispatch_path: Path, raw: Any, *, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} descriptor must be an object")
    configured = raw.get("path")
    if not isinstance(configured, str) or not configured:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = dispatch_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    digest = sha256(path)
    if raw.get("sha256") != digest:
        raise RuntimeError(f"{label} SHA-256 does not match")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, payload


def _same_number(first: Any, second: Any) -> bool:
    try:
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def _validate_v7_capacity_audit(
    *,
    dispatch_path: Path,
    descriptor: Any,
    plan_payload: dict[str, Any],
    evidence: dict[str, Any],
    host: dict[str, Any],
    physical_gpu: str,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    path, payload = _resolve_artifact(
        dispatch_path, descriptor, label="GPU capacity audit"
    )
    gate = (
        plan_payload.get("registration", {})
        .get("compute_policy", {})
        .get("same_gpu_capacity_gate", {})
    )
    methods = {job["method"] for job in jobs}
    expected_processes = max(
        int(job["max_canonical_processes_per_gpu"]) for job in jobs
    )
    checks = payload.get("decision", {}).get("checks", {})
    if (
        payload.get("schema_version") != CAPACITY_AUDIT_SCHEMA
        or payload.get("decision", {}).get("passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or payload.get("host_label") != host["hostname"]
        or payload.get("observed_hostname")
        not in {host["hostname"], *host.get("hostname_aliases", [])}
        or str(payload.get("physical_gpu")) != physical_gpu
        or payload.get("cuda_visible_devices_for_workers") != physical_gpu
        or payload.get("method") not in methods
        or payload.get("method") != "rlpd"
        or payload.get("phase") != "online"
        or payload.get("processes") != expected_processes
        or payload.get("dataset_sha256") != evidence["dataset_sha256"]
        or payload.get("max_offline_rows") != 0
        or int(payload.get("online_rows", -1))
        < int(plan_payload.get("shared_arguments", {}).get("online_capacity", 0))
        or payload.get("device_cache")
        != plan_payload.get("shared_arguments", {}).get(
            "device_cache_observation_dtype"
        )
        or int(payload.get("warmup_updates", -1))
        < int(gate.get("minimum_warmup_updates_per_process", -1))
        or int(payload.get("measured_updates", -1))
        < int(gate.get("minimum_measured_updates_per_process", -1))
        or int(payload.get("minimum_free_bytes", -1))
        != int(gate.get("minimum_free_memory_bytes", -2))
        or gate.get("single_process_reference_required") is not True
        or not isinstance(payload.get("single_process_reference"), dict)
        or not _same_number(
            payload.get("minimum_aggregate_speedup"),
            gate.get("minimum_aggregate_speedup"),
        )
        or not _same_number(
            payload.get("minimum_worker_rate_fraction"),
            gate.get("minimum_worker_rate_fraction"),
        )
    ):
        raise ValueError(
            f"GPU capacity audit is stale or incompatible with "
            f"{host['hostname']} GPU {physical_gpu}"
        )

    selected = plan_payload.get("shared_arguments", {})
    if (
        not _same_number(payload.get("reward_scale"), selected.get("reward_scale"))
        or payload.get("critic_loss") != selected.get("critic_loss")
        or not _same_number(payload.get("huber_delta"), selected.get("huber_delta"))
        or not _same_number(
            payload.get("target_entropy"), selected.get("target_entropy")
        )
    ):
        raise ValueError("GPU capacity audit uses another task-adaptation freeze")

    audit_sources = payload.get("source_sha256", {})
    if (
        audit_sources.get("reproducibility/audit_gpu_capacity_v7.py")
        != gate.get("protocol", {}).get("sha256")
        or audit_sources.get("reproducibility/benchmark_training_kernel_v4.py")
        != gate.get("benchmark_protocol", {}).get("sha256")
    ):
        raise ValueError("GPU capacity audit protocol source is stale")
    benchmark_sources = payload.get("benchmark_source_sha256")
    if not isinstance(benchmark_sources, dict) or not benchmark_sources:
        raise ValueError("GPU capacity audit lacks benchmark source evidence")
    for relative, digest in benchmark_sources.items():
        expected = (
            gate.get("benchmark_protocol", {}).get("sha256")
            if relative == "reproducibility/benchmark_training_kernel_v4.py"
            else evidence.get("source_sha256", {}).get(relative)
        )
        if expected is None or digest != expected:
            raise ValueError(f"GPU capacity benchmark source is stale: {relative}")

    single_reference = payload["single_process_reference"]
    try:
        reference_rate = float(single_reference.get("updates_per_second"))
    except (TypeError, ValueError):
        reference_rate = float("nan")
    if (
        not isinstance(single_reference.get("sha256"), str)
        or len(single_reference["sha256"]) != 64
        or not math.isfinite(reference_rate)
        or reference_rate <= 0.0
        or single_reference.get("benchmark_source_sha256") != benchmark_sources
    ):
        raise ValueError("GPU capacity audit has an invalid single-process reference")

    workers = payload.get("workers", [])
    if (
        len(workers) != expected_processes
        or any(worker.get("returncode") != 0 for worker in workers)
        or any(worker.get("report_validation_failures") != [] for worker in workers)
    ):
        raise ValueError("GPU capacity audit contains an invalid worker result")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "schema_version": payload["schema_version"],
        "host_label": payload["host_label"],
        "observed_hostname": payload["observed_hostname"],
        "physical_gpu": physical_gpu,
        "gpu_identity": payload.get("gpu_identity"),
        "decision": payload["decision"],
    }


def load_dispatch(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    dispatch_path = path.expanduser().resolve()
    payload = json.loads(dispatch_path.read_text())
    if payload.get("schema_version") != DISPATCH_SCHEMA:
        raise ValueError(f"dispatch must implement {DISPATCH_SCHEMA}")
    plan_path = _resolve_plan(dispatch_path, payload.get("plan"))
    jobs, evidence = load_plan(plan_path)
    plan_payload = json.loads(plan_path.read_text())
    if payload.get("plan_id") != plan_payload.get("plan_id"):
        raise RuntimeError("dispatch plan_id does not match the training plan")
    expected_job_set = job_set_sha256(jobs)
    if payload.get("job_set_sha256") != expected_job_set:
        raise RuntimeError("dispatch job-set SHA-256 does not match the signed plan")

    jobs_by_name = {job["name"]: job for job in jobs}
    assigned: list[str] = []
    host_names: set[str] = set()
    worker_names: set[str] = set()
    normalized_hosts: list[dict[str, Any]] = []
    raw_hosts = payload.get("hosts")
    if not isinstance(raw_hosts, list) or not raw_hosts:
        raise ValueError("dispatch hosts must be a non-empty list")
    for host_index, raw_host in enumerate(raw_hosts):
        if not isinstance(raw_host, dict):
            raise ValueError(f"hosts[{host_index}] must be an object")
        hostname = _safe_id(
            raw_host.get("hostname"), location=f"hosts[{host_index}].hostname"
        )
        if hostname in host_names:
            raise ValueError(f"duplicate dispatch hostname {hostname!r}")
        host_names.add(hostname)
        aliases = raw_host.get("hostname_aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"hosts[{host_index}].hostname_aliases must be a list")
        aliases = [
            _safe_id(value, location=f"hosts[{host_index}].hostname_aliases")
            for value in aliases
        ]
        if hostname in aliases or len(aliases) != len(set(aliases)):
            raise ValueError(f"hosts[{host_index}] contains duplicate aliases")
        raw_gpus = raw_host.get("gpus")
        if not isinstance(raw_gpus, list) or not raw_gpus:
            raise ValueError(f"hosts[{host_index}].gpus must be a non-empty list")
        physical_gpus: set[str] = set()
        normalized_gpus: list[dict[str, Any]] = []
        for gpu_index, raw_gpu in enumerate(raw_gpus):
            if not isinstance(raw_gpu, dict):
                raise ValueError(
                    f"hosts[{host_index}].gpus[{gpu_index}] must be an object"
                )
            physical_gpu = str(raw_gpu.get("physical_gpu", ""))
            if not physical_gpu or not re.fullmatch(r"[A-Za-z0-9._:-]+", physical_gpu):
                raise ValueError("dispatch physical_gpu contains unsafe characters")
            if physical_gpu in physical_gpus:
                raise ValueError(
                    f"host {hostname!r} assigns physical GPU {physical_gpu!r} twice"
                )
            physical_gpus.add(physical_gpu)
            raw_stages = raw_gpu.get("stages")
            if not isinstance(raw_stages, list) or not raw_stages:
                raise ValueError(
                    f"host {hostname!r} GPU {physical_gpu!r} lacks stages"
                )
            stage_names: set[str] = set()
            normalized_stages: list[dict[str, Any]] = []
            gpu_jobs: list[dict[str, Any]] = []
            for stage_index, raw_stage in enumerate(raw_stages):
                if not isinstance(raw_stage, dict):
                    raise ValueError("dispatch stage must be an object")
                stage_name = _safe_id(
                    raw_stage.get("name"),
                    location=(
                        f"hosts[{host_index}].gpus[{gpu_index}]."
                        f"stages[{stage_index}].name"
                    ),
                )
                if stage_name in stage_names:
                    raise ValueError(
                        f"host {hostname!r} GPU {physical_gpu!r} repeats stage "
                        f"{stage_name!r}"
                    )
                stage_names.add(stage_name)
                raw_workers = raw_stage.get("workers")
                if not isinstance(raw_workers, list) or not raw_workers:
                    raise ValueError(f"stage {stage_name!r} has no workers")
                normalized_workers: list[dict[str, Any]] = []
                stage_jobs: list[dict[str, Any]] = []
                for worker_index, raw_worker in enumerate(raw_workers):
                    if not isinstance(raw_worker, dict):
                        raise ValueError("dispatch worker must be an object")
                    worker_name = _safe_id(
                        raw_worker.get("name"),
                        location=f"stage {stage_name!r} worker[{worker_index}].name",
                    )
                    if worker_name in worker_names:
                        raise ValueError(f"duplicate dispatch worker {worker_name!r}")
                    worker_names.add(worker_name)
                    job_names = raw_worker.get("jobs")
                    if not isinstance(job_names, list) or not job_names:
                        raise ValueError(f"worker {worker_name!r} has no jobs")
                    if any(not isinstance(name, str) for name in job_names):
                        raise ValueError(f"worker {worker_name!r} has an invalid job name")
                    if len(job_names) != len(set(job_names)):
                        raise ValueError(f"worker {worker_name!r} repeats a job")
                    missing = [name for name in job_names if name not in jobs_by_name]
                    if missing:
                        raise ValueError(
                            f"worker {worker_name!r} references unknown jobs: {missing}"
                        )
                    assigned.extend(job_names)
                    stage_jobs.extend(jobs_by_name[name] for name in job_names)
                    gpu_jobs.extend(jobs_by_name[name] for name in job_names)
                    normalized_workers.append(
                        {"name": worker_name, "jobs": list(job_names)}
                    )

                if len(normalized_workers) > 1:
                    if any(len(worker["jobs"]) != 1 for worker in normalized_workers):
                        raise ValueError(
                            f"parallel stage {stage_name!r} requires exactly one job "
                            "per worker"
                        )
                    methods = {job["method"] for job in stage_jobs}
                    quotas = {
                        job["max_canonical_processes_per_gpu"] for job in stage_jobs
                    }
                    if len(methods) != 1 or len(quotas) != 1:
                        raise ValueError(
                            f"parallel stage {stage_name!r} mixes methods or GPU quotas"
                        )
                    quota = next(iter(quotas))
                    if len(normalized_workers) > quota:
                        raise ValueError(
                            f"parallel stage {stage_name!r} has "
                            f"{len(normalized_workers)} workers but quota {quota}"
                        )
                normalized_stages.append(
                    {"name": stage_name, "workers": normalized_workers}
                )
            normalized_gpu: dict[str, Any] = {
                "physical_gpu": physical_gpu,
                "stages": normalized_stages,
            }
            requires_capacity = (
                plan_payload.get("schema_version") == "transit-training-plan-v7"
                and any(
                    len(stage["workers"]) > 1
                    and any(
                        jobs_by_name[name]["method"] == "rlpd"
                        and jobs_by_name[name][
                            "max_canonical_processes_per_gpu"
                        ]
                        > 1
                        for worker in stage["workers"]
                        for name in worker["jobs"]
                    )
                    for stage in normalized_stages
                )
            )
            raw_capacity = raw_gpu.get("capacity_audit")
            if requires_capacity and raw_capacity is None:
                raise ValueError(
                    f"host {hostname!r} GPU {physical_gpu!r} lacks its required "
                    "full-data capacity audit"
                )
            if raw_capacity is not None:
                normalized_gpu["capacity_audit"] = _validate_v7_capacity_audit(
                    dispatch_path=dispatch_path,
                    descriptor=raw_capacity,
                    plan_payload=plan_payload,
                    evidence=evidence,
                    host={
                        "hostname": hostname,
                        "hostname_aliases": aliases,
                    },
                    physical_gpu=physical_gpu,
                    jobs=gpu_jobs,
                )
            normalized_gpus.append(normalized_gpu)
        normalized_hosts.append(
            {
                "hostname": hostname,
                "hostname_aliases": aliases,
                "gpus": normalized_gpus,
            }
        )

    duplicates = sorted({name for name in assigned if assigned.count(name) > 1})
    missing = sorted(set(jobs_by_name) - set(assigned))
    if duplicates or missing:
        raise ValueError(
            "dispatch assignment is not an exact partition of the plan: "
            f"duplicates={duplicates}, missing={missing}"
        )
    normalized = {
        **payload,
        "path": str(dispatch_path),
        "sha256": sha256(dispatch_path),
        "plan_path": str(plan_path),
        "hosts": normalized_hosts,
    }
    return normalized, jobs, evidence


def _worker_command(plan_path: Path, job_names: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "reproducibility" / "run_training_matrix_v4.py"),
        "--plan",
        str(plan_path),
    ]
    for job_name in job_names:
        command.extend(("--job-name", job_name))
    return command


def _run_gpu_schedule(
    *,
    host: dict[str, Any],
    gpu: dict[str, Any],
    plan_path: Path,
    execution_root: Path,
) -> dict[str, Any]:
    physical_gpu = gpu["physical_gpu"]
    gpu_root = execution_root / f"gpu_{re.sub(r'[^A-Za-z0-9._-]', '_', physical_gpu)}"
    report_path = gpu_root / "execution.json"
    report: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": host["hostname"],
        "observed_hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "plan": str(plan_path),
        "stages": [],
        "status": "running",
    }
    _write_json(report_path, report)
    failed = False
    for stage in gpu["stages"]:
        stage_report: dict[str, Any] = {
            "name": stage["name"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "workers": [],
        }
        report["stages"].append(stage_report)
        _write_json(report_path, report)
        processes: list[tuple[subprocess.Popen[str], Any, dict[str, Any]]] = []
        for worker in stage["workers"]:
            command = _worker_command(plan_path, worker["jobs"])
            log_path = gpu_root / f"{worker['name']}.log"
            log_handle = log_path.open("a")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
            environment["PYTHONUNBUFFERED"] = "1"
            worker_report = {
                "name": worker["name"],
                "jobs": worker["jobs"],
                "command": command,
                "log": str(log_path),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            }
            stage_report["workers"].append(worker_report)
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            worker_report["pid"] = process.pid
            processes.append((process, log_handle, worker_report))
        _write_json(report_path, report)

        pending = list(processes)
        stage_failed = False
        try:
            while pending:
                for item in list(pending):
                    process, log_handle, worker_report = item
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    pending.remove(item)
                    log_handle.close()
                    worker_report["returncode"] = returncode
                    worker_report["completed_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    worker_report["status"] = (
                        "completed" if returncode == 0 else "failed"
                    )
                    stage_failed = stage_failed or returncode != 0
                    _write_json(report_path, report)
                if stage_failed and pending:
                    for process, _, _ in pending:
                        process.terminate()
                if pending:
                    time.sleep(2.0)
        finally:
            for process, log_handle, worker_report in pending:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                log_handle.close()
                worker_report["returncode"] = process.returncode
                worker_report["completed_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                worker_report["status"] = "failed"
            _write_json(report_path, report)

        stage_report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        stage_report["status"] = "failed" if stage_failed else "completed"
        _write_json(report_path, report)
        if stage_failed:
            failed = True
            break

    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "failed" if failed else "completed"
    _write_json(report_path, report)
    return report


def _select_host(dispatch: dict[str, Any], expected: str | None) -> dict[str, Any]:
    observed = socket.gethostname()
    candidates = {observed, observed.split(".", 1)[0]}
    matches = [
        host
        for host in dispatch["hosts"]
        if candidates.intersection(
            {host["hostname"], *host.get("hostname_aliases", [])}
        )
    ]
    if expected is not None:
        matches = [host for host in matches if host["hostname"] == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"observed host {observed!r} does not uniquely match the dispatch"
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--host", help="expected canonical hostname; never an override")
    parser.add_argument("--execution-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dispatch, jobs, evidence = load_dispatch(args.dispatch)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "dispatch": dispatch["path"],
                    "dispatch_sha256": dispatch["sha256"],
                    "plan": dispatch["plan_path"],
                    "plan_sha256": evidence["plan_sha256"],
                    "jobs": len(jobs),
                    "hosts": len(dispatch["hosts"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    host = _select_host(dispatch, args.host)
    execution_root = (
        args.execution_root.expanduser().resolve()
        if args.execution_root is not None
        else Path(evidence["output_root"]) / "dispatch_execution" / host["hostname"]
    )
    execution_root.mkdir(parents=True, exist_ok=True)
    host_report_path = execution_root / "host_execution.json"
    host_report: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dispatch": dispatch["path"],
        "dispatch_sha256": dispatch["sha256"],
        "plan": dispatch["plan_path"],
        "plan_sha256": evidence["plan_sha256"],
        "hostname": host["hostname"],
        "observed_hostname": socket.gethostname(),
        "status": "running",
        "gpus": [],
    }
    _write_json(host_report_path, host_report)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(host["gpus"])
    ) as executor:
        futures = [
            executor.submit(
                _run_gpu_schedule,
                host=host,
                gpu=gpu,
                plan_path=Path(dispatch["plan_path"]),
                execution_root=execution_root,
            )
            for gpu in host["gpus"]
        ]
        for future in concurrent.futures.as_completed(futures):
            host_report["gpus"].append(future.result())
            _write_json(host_report_path, host_report)
    failed = any(report["status"] != "completed" for report in host_report["gpus"])
    host_report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    host_report["status"] = "failed" if failed else "completed"
    _write_json(host_report_path, host_report)
    print(host_report_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
