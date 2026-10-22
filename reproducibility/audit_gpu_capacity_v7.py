#!/usr/bin/env python3
"""Gate a formal same-GPU training concurrency policy on the full v7 data."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
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

from bus_h2o.sumo_provenance import sha256
from SimpleSAC.train_transit_v3 import resolve_nvidia_smi


SCHEMA_VERSION = "transit-v7-gpu-capacity-audit-v1"
DEFAULT_MINIMUM_FREE_BYTES = 1 * 2**30
DEFAULT_MINIMUM_AGGREGATE_SPEEDUP = 1.20
DEFAULT_MINIMUM_WORKER_RATE_FRACTION = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", default="rlpd")
    parser.add_argument("--phase", choices=("offline", "online"), default="online")
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=9501)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--measured-updates", type=int, default=5_000)
    parser.add_argument("--chunk-updates", type=int, default=100)
    parser.add_argument("--online-rows", type=int, default=120_000)
    parser.add_argument("--cpu-threads-per-process", type=int, default=8)
    parser.add_argument("--max-offline-rows", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, required=True)
    parser.add_argument("--critic-loss", choices=("mse", "huber"), required=True)
    parser.add_argument("--huber-delta", type=float, required=True)
    parser.add_argument("--target-entropy", type=float, required=True)
    parser.add_argument(
        "--device-cache",
        choices=("none", "float32", "float16"),
        required=True,
    )
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=DEFAULT_MINIMUM_FREE_BYTES
    )
    parser.add_argument("--single-process-reference", type=Path, required=True)
    parser.add_argument(
        "--minimum-aggregate-speedup",
        type=float,
        default=DEFAULT_MINIMUM_AGGREGATE_SPEEDUP,
    )
    parser.add_argument(
        "--minimum-worker-rate-fraction",
        type=float,
        default=DEFAULT_MINIMUM_WORKER_RATE_FRACTION,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _safe_physical_gpu(value: str) -> str:
    if not value or any(character not in "0123456789abcdefABCDEF-_.:" for character in value):
        raise ValueError("physical GPU identifier contains unsafe characters")
    return value


def _safe_label(value: str) -> str:
    if not value or any(
        character not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_."
        for character in value
    ):
        raise ValueError("host label contains unsafe characters")
    return value


def _numeric_csv_row(raw: str, expected: int) -> list[float]:
    fields = [field.strip() for field in raw.strip().split(",")]
    if len(fields) != expected:
        raise RuntimeError(f"unexpected nvidia-smi row: {raw!r}")
    values = []
    for field in fields:
        try:
            value = float(field)
        except ValueError as error:
            raise RuntimeError(f"non-numeric nvidia-smi field: {field!r}")
        values.append(value)
    return values


def _compute_pids_for_gpu(raw: str, gpu_uuid: str) -> list[int]:
    pids = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise RuntimeError(f"unexpected nvidia-smi compute row: {line!r}")
        if fields[0] == gpu_uuid:
            pids.append(int(fields[1]))
    return sorted(pids)


def _classify_compute_pids(
    compute_pids: list[int], worker_pids: set[int]
) -> dict[str, Any]:
    """Attribute CUDA helper processes to worker process groups, failing closed."""

    registered = []
    unregistered = []
    process_groups: dict[str, int | None] = {}
    registered_worker_groups = set()
    for pid in compute_pids:
        try:
            process_group = os.getpgid(pid)
        except (PermissionError, ProcessLookupError):
            process_group = None
        process_groups[str(pid)] = process_group
        owner = pid if pid in worker_pids else process_group
        if owner in worker_pids:
            registered.append(pid)
            registered_worker_groups.add(int(owner))
        else:
            unregistered.append(pid)
    return {
        "compute_process_groups": process_groups,
        "registered_compute_pids": sorted(registered),
        "unregistered_compute_pids": sorted(unregistered),
        "registered_worker_process_groups": sorted(registered_worker_groups),
    }


def _gpu_sample(
    nvidia_smi: str, physical_gpu: str, *, gpu_uuid: str
) -> dict[str, Any]:
    gpu = subprocess.run(
        [
            nvidia_smi,
            "-i",
            physical_gpu,
            "--query-gpu=memory.total,memory.used,utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total_mib, used_mib, gpu_util, memory_util, power_watts = _numeric_csv_row(
        gpu.stdout.splitlines()[0], 5
    )
    applications = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = _compute_pids_for_gpu(applications.stdout, gpu_uuid)
    return {
        "elapsed_seconds": 0.0,
        "memory_total_bytes": int(total_mib * 2**20),
        "memory_used_bytes": int(used_mib * 2**20),
        "gpu_utilization_percent": int(gpu_util),
        "memory_utilization_percent": int(memory_util),
        "power_draw_watts": power_watts,
        "compute_pids": pids,
    }


def _gpu_identity(nvidia_smi: str, physical_gpu: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            nvidia_smi,
            "-i",
            physical_gpu,
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi GPU identity row")
    name, uuid, memory_mib, driver = fields
    return {
        "name": name,
        "uuid": uuid,
        "memory_total_bytes": int(float(memory_mib) * 2**20),
        "driver_version": driver,
    }


def _canonical_source_map(raw: dict[str, str]) -> dict[str, str]:
    result = {}
    root = ROOT.resolve()
    for raw_path, digest in raw.items():
        path = Path(raw_path).expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"benchmark source lies outside the capacity workspace: {path}"
            ) from error
        result[relative] = digest
    return dict(sorted(result.items()))


def _benchmark_command(args: argparse.Namespace, worker: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "reproducibility" / "benchmark_training_kernel_v4.py"),
        "--dataset",
        str(args.dataset.expanduser().resolve()),
        "--mdp-version",
        "v7",
        "--method",
        args.method,
        "--phase",
        args.phase,
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed_base + worker),
        "--warmup-updates",
        str(args.warmup_updates),
        "--measured-updates",
        str(args.measured_updates),
        "--chunk-updates",
        str(args.chunk_updates),
        "--online-rows",
        str(args.online_rows),
        "--cpu-threads",
        str(args.cpu_threads_per_process),
        "--max-offline-rows",
        str(args.max_offline_rows),
        "--reward-scale",
        str(args.reward_scale),
        "--critic-loss",
        args.critic_loss,
        "--huber-delta",
        str(args.huber_delta),
        "--target-entropy",
        str(args.target_entropy),
        "--device-cache",
        args.device_cache,
        "--compile-networks",
        "--compile-mode",
        "default",
        "--max-canonical-processes-per-gpu",
        str(args.processes),
        "--output",
        str(output),
    ]
    return command


def _validate_worker_report(
    report: dict[str, Any],
    *,
    args: argparse.Namespace,
    dataset_sha256: str,
) -> list[str]:
    failures = []
    expected = {
        "method": args.method,
        "mdp_version": "v7",
        "phase": args.phase,
        "dataset_sha256": dataset_sha256,
        "device_cache_mode": args.device_cache,
        "compile_networks": True,
        "compile_mode": "default",
        "max_canonical_processes_per_gpu": args.processes,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            failures.append(f"{name} does not match the capacity protocol")
    for raw_path, expected_hash in report.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            failures.append(f"benchmark source is stale: {path}")
    if not report.get("source_sha256"):
        failures.append("benchmark report lacks source hashes")
    return failures


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _load_single_process_reference(
    args: argparse.Namespace,
    *,
    dataset_sha256: str,
    gpu_identity: dict[str, Any],
) -> dict[str, Any]:
    path = args.single_process_reference.expanduser().resolve(strict=True)
    report = json.loads(path.read_text())
    expected = {
        "schema_version": "transit-training-kernel-benchmark-v3",
        "method": args.method,
        "mdp_version": "v7",
        "phase": args.phase,
        "dataset_sha256": dataset_sha256,
        "device_cache_mode": args.device_cache,
        "compile_networks": True,
        "compile_mode": "default",
        "max_canonical_processes_per_gpu": 1,
        "max_offline_rows": args.max_offline_rows,
        "allow_shared_gpu": False,
        "allow_external_gpu_processes": False,
    }
    mismatches = {
        name: {"reference": report.get(name), "expected": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    trainer = report.get("trainer_config", {})
    numeric_expected = {
        "reward_scale": args.reward_scale,
        "huber_delta": args.huber_delta,
        "target_entropy": args.target_entropy,
    }
    for name, value in numeric_expected.items():
        if not _same_number(trainer.get(name), value):
            mismatches[f"trainer_config.{name}"] = {
                "reference": trainer.get(name),
                "expected": value,
            }
    if trainer.get("critic_loss") != args.critic_loss:
        mismatches["trainer_config.critic_loss"] = {
            "reference": trainer.get("critic_loss"),
            "expected": args.critic_loss,
        }
    runtime_host = report.get("runtime", {}).get("host", {})
    if runtime_host.get("gpu") != gpu_identity["name"]:
        mismatches["runtime.host.gpu"] = {
            "reference": runtime_host.get("gpu"),
            "expected": gpu_identity["name"],
        }
    if int(report.get("warmup_updates", -1)) < args.warmup_updates:
        mismatches["warmup_updates"] = {
            "reference": report.get("warmup_updates"),
            "minimum": args.warmup_updates,
        }
    if int(report.get("measured_updates", -1)) < args.measured_updates:
        mismatches["measured_updates"] = {
            "reference": report.get("measured_updates"),
            "minimum": args.measured_updates,
        }
    rate = report.get("updates_per_second")
    try:
        rate_value = float(rate)
    except (TypeError, ValueError):
        rate_value = float("nan")
    if not math.isfinite(rate_value) or rate_value <= 0.0:
        mismatches["updates_per_second"] = {"reference": rate, "expected": ">0"}
    source_map = _canonical_source_map(report.get("source_sha256", {}))
    if not source_map:
        mismatches["source_sha256"] = {"reference": source_map, "expected": "non-empty"}
    for relative, expected_hash in source_map.items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != expected_hash:
            mismatches[f"source_sha256.{relative}"] = {
                "reference": expected_hash,
                "current": sha256(source) if source.is_file() else None,
            }
    if mismatches:
        raise ValueError(f"single-process throughput reference is stale: {mismatches}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "updates_per_second": rate_value,
        "warmup_updates": int(report["warmup_updates"]),
        "measured_updates": int(report["measured_updates"]),
        "benchmark_source_sha256": source_map,
        "runtime_host": runtime_host,
    }


def capacity_decision(
    *,
    workers: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    expected_processes: int,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    worker_pids = {int(worker["pid"]) for worker in workers}
    successful = [
        worker
        for worker in workers
        if worker.get("returncode") == 0
        and worker.get("report_validation_failures") == []
    ]
    concurrent_counts = []
    for sample in samples:
        registered_groups = sample.get("registered_worker_process_groups")
        if registered_groups is None:
            concurrent_counts.append(
                len(worker_pids.intersection(sample.get("compute_pids", [])))
            )
        else:
            concurrent_counts.append(
                len(worker_pids.intersection(registered_groups))
            )
    max_concurrent = max(concurrent_counts, default=0)
    unexpected_pids = sorted(
        {
            int(pid)
            for sample in samples
            for pid in sample.get(
                "unregistered_compute_pids",
                [
                    raw_pid
                    for raw_pid in sample.get("compute_pids", [])
                    if int(raw_pid) not in worker_pids
                ],
            )
        }
    )
    peak_used = max(
        (int(sample["memory_used_bytes"]) for sample in samples), default=0
    )
    total_values = {
        int(sample["memory_total_bytes"]) for sample in samples
    }
    stable_total = len(total_values) == 1
    total = next(iter(total_values)) if stable_total else 0
    minimum_free_observed = total - peak_used if stable_total else 0
    checks = {
        "all_workers_completed": len(successful) == expected_processes,
        "registered_processes_overlapped": max_concurrent == expected_processes,
        "no_unregistered_compute_process": not unexpected_pids,
        "stable_device_capacity": stable_total,
        "minimum_free_memory_headroom": (
            stable_total and minimum_free_observed >= minimum_free_bytes
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_processes": expected_processes,
        "successful_workers": len(successful),
        "maximum_concurrent_registered_processes": max_concurrent,
        "unexpected_compute_pids": unexpected_pids,
        "memory_total_bytes": total,
        "peak_memory_used_bytes": peak_used,
        "minimum_free_memory_observed_bytes": minimum_free_observed,
        "minimum_free_memory_required_bytes": minimum_free_bytes,
        "policy": (
            "admit the registered same-GPU process count only when every worker "
            "completes, all processes overlap, no external compute PID appears, "
            "and the full-data run retains the fixed memory headroom"
        ),
    }


def throughput_decision(
    *,
    worker_rates: list[float],
    single_process_rate: float,
    expected_processes: int,
    minimum_aggregate_speedup: float,
    minimum_worker_rate_fraction: float,
) -> dict[str, Any]:
    valid_rates = (
        len(worker_rates) == expected_processes
        and math.isfinite(single_process_rate)
        and single_process_rate > 0.0
        and all(math.isfinite(rate) and rate > 0.0 for rate in worker_rates)
    )
    if not valid_rates:
        return {
            "passed": False,
            "checks": {
                "all_worker_rates_available": False,
                "minimum_aggregate_speedup": False,
                "minimum_worker_rate_fraction": False,
            },
            "single_process_updates_per_second": single_process_rate,
            "worker_updates_per_second": worker_rates,
            "aggregate_updates_per_second": None,
            "aggregate_speedup": None,
            "minimum_observed_worker_rate_fraction": None,
        }
    aggregate = sum(worker_rates)
    aggregate_speedup = aggregate / single_process_rate
    worker_fraction = min(worker_rates) / single_process_rate
    checks = {
        "all_worker_rates_available": True,
        "minimum_aggregate_speedup": (
            aggregate_speedup >= minimum_aggregate_speedup
        ),
        "minimum_worker_rate_fraction": (
            worker_fraction >= minimum_worker_rate_fraction
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "single_process_updates_per_second": single_process_rate,
        "worker_updates_per_second": worker_rates,
        "aggregate_updates_per_second": aggregate,
        "aggregate_speedup": aggregate_speedup,
        "minimum_observed_worker_rate_fraction": worker_fraction,
    }


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main() -> int:
    args = parse_args()
    if not 2 <= args.processes <= 8:
        raise ValueError("capacity-audit process count must lie in [2, 8]")
    if (
        args.warmup_updates < 0
        or args.measured_updates <= 0
        or args.chunk_updates <= 0
        or args.online_rows <= 0
        or args.cpu_threads_per_process <= 0
        or args.max_offline_rows < 0
        or args.huber_delta <= 0.0
        or args.sample_interval <= 0.0
        or args.minimum_free_bytes < 0
        or args.minimum_aggregate_speedup <= 1.0
        or not 0.0 < args.minimum_worker_rate_fraction <= 1.0
    ):
        raise ValueError("capacity-audit sizes must be positive")
    args.physical_gpu = _safe_physical_gpu(args.physical_gpu)
    args.host_label = _safe_label(args.host_label)
    args.dataset = args.dataset.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite capacity audit {output_dir}")
    report_path = output_dir / "capacity_audit.json"
    nvidia_smi = resolve_nvidia_smi()
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for the GPU capacity audit")

    dataset_digest = sha256(args.dataset)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    environment["PYTHONUNBUFFERED"] = "1"
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    gpu_identity = _gpu_identity(nvidia_smi, args.physical_gpu)
    single_process_reference = _load_single_process_reference(
        args,
        dataset_sha256=dataset_digest,
        gpu_identity=gpu_identity,
    )
    initial = _gpu_sample(
        nvidia_smi, args.physical_gpu, gpu_uuid=gpu_identity["uuid"]
    )
    if initial["compute_pids"]:
        raise RuntimeError(
            "capacity audit requires an initially idle physical GPU; observed "
            f"compute PIDs {initial['compute_pids']}"
        )
    output_dir.mkdir(parents=True)

    processes: list[subprocess.Popen[str]] = []
    log_handles = []
    workers: list[dict[str, Any]] = []
    samples = [initial]
    try:
        for worker_index in range(args.processes):
            output = output_dir / f"worker_{worker_index}.json"
            log_path = output_dir / f"worker_{worker_index}.log"
            command = _benchmark_command(args, worker_index, output)
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(process)
            log_handles.append(log_handle)
            workers.append(
                {
                    "worker": worker_index,
                    "pid": process.pid,
                    "seed": args.seed_base + worker_index,
                    "command": command,
                    "output": str(output),
                    "log": str(log_path),
                }
            )

        worker_pids = {process.pid for process in processes}
        while any(process.poll() is None for process in processes):
            sample = _gpu_sample(
                nvidia_smi,
                args.physical_gpu,
                gpu_uuid=gpu_identity["uuid"],
            )
            sample.update(
                _classify_compute_pids(sample["compute_pids"], worker_pids)
            )
            sample["elapsed_seconds"] = time.monotonic() - started
            samples.append(sample)
            failed = any(
                process.poll() not in (None, 0) for process in processes
            )
            if failed:
                _terminate(processes)
                break
            time.sleep(args.sample_interval)
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for log_handle in log_handles:
            log_handle.close()

    final = _gpu_sample(
        nvidia_smi, args.physical_gpu, gpu_uuid=gpu_identity["uuid"]
    )
    final.update(_classify_compute_pids(final["compute_pids"], worker_pids))
    final["elapsed_seconds"] = time.monotonic() - started
    samples.append(final)
    for worker, process in zip(workers, processes):
        worker["returncode"] = process.wait()
        output = Path(worker["output"])
        failures = []
        if output.is_file():
            report = json.loads(output.read_text())
            failures = _validate_worker_report(
                report, args=args, dataset_sha256=dataset_digest
            )
            worker["updates_per_second"] = report.get("updates_per_second")
            worker["setup_cuda_memory"] = report.get("resources", {}).get(
                "cuda", {}
            ).get("setup")
            worker["training_cuda_memory"] = report.get("resources", {}).get(
                "cuda", {}
            ).get("training")
            worker["report_sha256"] = sha256(output)
            worker["source_sha256"] = _canonical_source_map(
                report.get("source_sha256", {})
            )
        else:
            failures.append("worker report is missing")
        worker["report_validation_failures"] = failures

    decision = capacity_decision(
        workers=workers,
        samples=samples,
        expected_processes=args.processes,
        minimum_free_bytes=args.minimum_free_bytes,
    )
    worker_rates = [
        float(worker["updates_per_second"])
        for worker in workers
        if worker.get("updates_per_second") is not None
    ]
    throughput = throughput_decision(
        worker_rates=worker_rates,
        single_process_rate=single_process_reference["updates_per_second"],
        expected_processes=args.processes,
        minimum_aggregate_speedup=args.minimum_aggregate_speedup,
        minimum_worker_rate_fraction=args.minimum_worker_rate_fraction,
    )
    decision["checks"].update(throughput["checks"])
    decision["passed"] = decision["passed"] and throughput["passed"]
    decision["throughput"] = throughput
    source_maps = [
        worker.get("source_sha256")
        for worker in workers
        if worker.get("source_sha256")
    ]
    benchmark_sources = (
        source_maps[0]
        if len(source_maps) == args.processes
        and all(source_map == source_maps[0] for source_map in source_maps)
        else None
    )
    if benchmark_sources is None:
        decision["checks"]["identical_worker_source_hashes"] = False
        decision["passed"] = False
    else:
        decision["checks"]["identical_worker_source_hashes"] = True
    reference_sources_match = (
        benchmark_sources
        == single_process_reference["benchmark_source_sha256"]
    )
    decision["checks"]["single_and_multi_process_source_hashes_match"] = (
        reference_sources_match
    )
    decision["passed"] = decision["passed"] and reference_sources_match
    decision["throughput_policy"] = (
        "admit concurrency only when aggregate update throughput reaches the "
        f"registered single-process rate times {args.minimum_aggregate_speedup} "
        "and every worker retains at least "
        f"{args.minimum_worker_rate_fraction} of that reference rate"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "performance-and-capacity-gate",
        "scientific_disclosure": (
            "synthetic online rows exercise the registered update path; this audit "
            "does not produce policy-effectiveness evidence"
        ),
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_digest,
        "method": args.method,
        "phase": args.phase,
        "host_label": args.host_label,
        "observed_hostname": socket.gethostname(),
        "physical_gpu": args.physical_gpu,
        "gpu_identity": gpu_identity,
        "cuda_visible_devices_for_workers": args.physical_gpu,
        "device_cache": args.device_cache,
        "reward_scale": args.reward_scale,
        "critic_loss": args.critic_loss,
        "huber_delta": args.huber_delta,
        "target_entropy": args.target_entropy,
        "max_offline_rows": args.max_offline_rows,
        "online_rows": args.online_rows,
        "processes": args.processes,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "minimum_free_bytes": args.minimum_free_bytes,
        "minimum_aggregate_speedup": args.minimum_aggregate_speedup,
        "minimum_worker_rate_fraction": args.minimum_worker_rate_fraction,
        "single_process_reference": single_process_reference,
        "workers": workers,
        "aggregate_updates_per_second": throughput[
            "aggregate_updates_per_second"
        ],
        "aggregate_speedup": throughput["aggregate_speedup"],
        "samples": samples,
        "decision": decision,
        "benchmark_source_sha256": benchmark_sources,
        "source_sha256": {
            "reproducibility/audit_gpu_capacity_v7.py": sha256(
                Path(__file__).resolve()
            ),
            "reproducibility/benchmark_training_kernel_v4.py": sha256(
                ROOT / "reproducibility" / "benchmark_training_kernel_v4.py"
            ),
        },
    }
    _write_json(report_path, payload)
    print(
        json.dumps(
            {
                "output": str(report_path),
                "passed": decision["passed"],
                "maximum_concurrent_registered_processes": decision[
                    "maximum_concurrent_registered_processes"
                ],
                "minimum_free_memory_observed_bytes": decision[
                    "minimum_free_memory_observed_bytes"
                ],
                "updates_per_second": [
                    worker.get("updates_per_second") for worker in workers
                ],
                "aggregate_updates_per_second": (
                    throughput["aggregate_updates_per_second"]
                ),
                "aggregate_speedup": throughput["aggregate_speedup"],
            },
            sort_keys=True,
        )
    )
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
