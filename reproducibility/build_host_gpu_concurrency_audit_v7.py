#!/usr/bin/env python3
"""Audit two-GPU host concurrency from frozen v7 kernel benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_gpu_capacity_v7 import (
    DEFAULT_MINIMUM_AGGREGATE_SPEEDUP,
    DEFAULT_MINIMUM_FREE_BYTES,
    DEFAULT_MINIMUM_WORKER_RATE_FRACTION,
    throughput_decision,
)


SCHEMA_VERSION = "transit-v7-host-gpu-concurrency-audit-v1"
BENCHMARK_SCHEMA = "transit-training-kernel-benchmark-v3"
REQUIRED_WARMUP_UPDATES = 100
REQUIRED_MEASURED_UPDATES = 5_000
REFERENCE_CPU_THREADS = 8
CONCURRENT_CPU_THREADS = 6


def _descriptor(
    path: Path, *, relative_to: Path | None = None
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    rendered_path = (
        os.path.relpath(resolved, relative_to.expanduser().resolve())
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": rendered_path,
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _logical_source_map(raw: dict[str, str]) -> dict[str, str]:
    """Map deployment-absolute source paths back to logical workspace paths."""

    if not isinstance(raw, dict) or not raw:
        raise ValueError("benchmark source map is empty")
    result: dict[str, str] = {}
    markers = ("SimpleSAC", "bus_h2o", "reproducibility")
    for raw_path, digest in raw.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid source digest for {raw_path}")
        parts = Path(raw_path).expanduser().parts
        candidates: list[Path] = []
        for marker in markers:
            indices = [index for index, part in enumerate(parts) if part == marker]
            candidates.extend(Path(*parts[index:]) for index in reversed(indices))
        matched = None
        for relative in candidates:
            source = ROOT / relative
            if source.is_file() and sha256(source) == digest:
                matched = relative.as_posix()
                break
        if matched is None:
            raise ValueError(f"benchmark source is missing or stale: {raw_path}")
        previous = result.get(matched)
        if previous is not None and previous != digest:
            raise ValueError(f"conflicting source identity for {matched}")
        result[matched] = digest
    return dict(sorted(result.items()))


def _read_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark report is not an object: {resolved}")
    return resolved, payload


def _validate_common(
    report: dict[str, Any],
    *,
    dataset_sha256: str,
    trainer_identity: dict[str, Any],
    source_identity: dict[str, str],
) -> None:
    expected = {
        "schema_version": BENCHMARK_SCHEMA,
        "method": "rlpd",
        "mdp_version": "v7",
        "phase": "online",
        "dataset_sha256": dataset_sha256,
        "device_cache_mode": "float32",
        "compile_networks": True,
        "compile_mode": "default",
        "max_canonical_processes_per_gpu": 1,
        "max_offline_rows": 0,
        "allow_shared_gpu": False,
        "allow_external_gpu_processes": False,
    }
    mismatches = {
        name: {"observed": report.get(name), "expected": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    trainer = report.get("trainer_config", {})
    for name, value in trainer_identity.items():
        observed = trainer.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            matches = _same_number(observed, value)
        else:
            matches = observed == value
        if not matches:
            mismatches[f"trainer_config.{name}"] = {
                "observed": observed,
                "expected": value,
            }
    if int(report.get("warmup_updates", -1)) < REQUIRED_WARMUP_UPDATES:
        mismatches["warmup_updates"] = {
            "observed": report.get("warmup_updates"),
            "minimum": REQUIRED_WARMUP_UPDATES,
        }
    if int(report.get("measured_updates", -1)) < REQUIRED_MEASURED_UPDATES:
        mismatches["measured_updates"] = {
            "observed": report.get("measured_updates"),
            "minimum": REQUIRED_MEASURED_UPDATES,
        }
    if _logical_source_map(report.get("source_sha256", {})) != source_identity:
        mismatches["source_sha256"] = "source identity differs from reference"
    if mismatches:
        raise ValueError(f"benchmark configuration mismatch: {mismatches}")


def build_audit(
    *,
    host_label: str,
    single_process_reference: Path,
    worker_results: Iterable[Path],
    physical_gpus: Iterable[str],
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    minimum_aggregate_speedup: float = DEFAULT_MINIMUM_AGGREGATE_SPEEDUP,
    minimum_worker_rate_fraction: float = DEFAULT_MINIMUM_WORKER_RATE_FRACTION,
    descriptor_base: Path | None = None,
) -> dict[str, Any]:
    worker_paths = list(worker_results)
    gpu_ids = [str(value) for value in physical_gpus]
    if len(worker_paths) != 2 or len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("host concurrency audit requires two distinct GPU workers")
    if minimum_free_bytes < 0 or minimum_aggregate_speedup <= 1.0:
        raise ValueError("host concurrency thresholds are invalid")
    if not 0.0 < minimum_worker_rate_fraction <= 1.0:
        raise ValueError("worker-rate threshold is invalid")

    reference_path, reference = _read_report(single_process_reference)
    dataset_digest = str(reference.get("dataset_sha256", ""))
    if len(dataset_digest) != 64:
        raise ValueError("single-process reference lacks a dataset identity")
    trainer_identity = {
        name: reference.get("trainer_config", {}).get(name)
        for name in ("reward_scale", "critic_loss", "huber_delta", "target_entropy")
    }
    source_identity = _logical_source_map(reference.get("source_sha256", {}))
    _validate_common(
        reference,
        dataset_sha256=dataset_digest,
        trainer_identity=trainer_identity,
        source_identity=source_identity,
    )
    reference_host = reference.get("runtime", {}).get("host", {})
    if reference_host.get("cpu_threads") != REFERENCE_CPU_THREADS:
        raise ValueError("single-process reference must use eight CPU threads")
    reference_rate = float(reference.get("updates_per_second", float("nan")))
    if not math.isfinite(reference_rate) or reference_rate <= 0.0:
        raise ValueError("single-process reference rate is invalid")

    workers = []
    worker_rates = []
    for path, gpu_id in zip(worker_paths, gpu_ids):
        resolved, report = _read_report(path)
        _validate_common(
            report,
            dataset_sha256=dataset_digest,
            trainer_identity=trainer_identity,
            source_identity=source_identity,
        )
        runtime_host = report.get("runtime", {}).get("host", {})
        if runtime_host.get("cpu_threads") != CONCURRENT_CPU_THREADS:
            raise ValueError(f"concurrent worker on GPU {gpu_id} must use six threads")
        if str(runtime_host.get("cuda_visible_devices")) != gpu_id:
            raise ValueError(f"worker does not identify physical GPU {gpu_id}")
        if runtime_host.get("gpu") != reference_host.get("gpu"):
            raise ValueError("worker GPU model differs from the reference")
        rate = float(report.get("updates_per_second", float("nan")))
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"worker rate is invalid for GPU {gpu_id}")
        cuda = report.get("resources", {}).get("cuda", {})
        peak_reserved = int(
            cuda.get("training", {}).get("peak_reserved_bytes", -1)
        )
        total_bytes = int(cuda.get("total_bytes", -1))
        memory_headroom = total_bytes - peak_reserved
        if peak_reserved <= 0 or total_bytes <= 0:
            raise ValueError(f"worker lacks CUDA memory evidence for GPU {gpu_id}")
        worker_rates.append(rate)
        workers.append(
            {
                **_descriptor(resolved, relative_to=descriptor_base),
                "physical_gpu": gpu_id,
                "updates_per_second": rate,
                "cpu_threads": runtime_host["cpu_threads"],
                "gpu_model": runtime_host["gpu"],
                "peak_reserved_bytes": peak_reserved,
                "memory_total_bytes": total_bytes,
                "memory_headroom_bytes": memory_headroom,
                "max_rss_bytes": int(
                    report.get("resources", {}).get("max_rss_bytes", 0)
                ),
            }
        )

    throughput = throughput_decision(
        worker_rates=worker_rates,
        single_process_rate=reference_rate,
        expected_processes=2,
        minimum_aggregate_speedup=minimum_aggregate_speedup,
        minimum_worker_rate_fraction=minimum_worker_rate_fraction,
    )
    checks = dict(throughput["checks"])
    checks.update(
        {
            "distinct_physical_gpus": len(set(gpu_ids)) == 2,
            "per_gpu_memory_headroom": all(
                worker["memory_headroom_bytes"] >= minimum_free_bytes
                for worker in workers
            ),
            "identical_dataset_configuration_and_source": True,
        }
    )
    passed = throughput["passed"] and all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "performance-and-host-capacity-gate",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scientific_disclosure": (
            "synthetic online rows exercise the registered update path; this "
            "audit selects only a resource schedule and provides no policy-"
            "effectiveness evidence"
        ),
        "host_label": host_label,
        "method": "rlpd",
        "phase": "online",
        "dataset_sha256": dataset_digest,
        "device_cache_mode": "float32",
        "compile_networks": True,
        "compile_mode": "default",
        "max_canonical_processes_per_gpu": 1,
        "canonical_processes_per_host": 2,
        "cpu_threads_per_process": CONCURRENT_CPU_THREADS,
        "physical_gpus": gpu_ids,
        "single_process_reference": {
            **_descriptor(reference_path, relative_to=descriptor_base),
            "updates_per_second": reference_rate,
            "cpu_threads": REFERENCE_CPU_THREADS,
        },
        "workers": workers,
        "aggregate_updates_per_second": throughput[
            "aggregate_updates_per_second"
        ],
        "aggregate_speedup": throughput["aggregate_speedup"],
        "minimum_observed_worker_rate_fraction": throughput[
            "minimum_observed_worker_rate_fraction"
        ],
        "minimum_aggregate_speedup": minimum_aggregate_speedup,
        "minimum_worker_rate_fraction": minimum_worker_rate_fraction,
        "minimum_free_memory_bytes": minimum_free_bytes,
        "decision": {"passed": passed, "checks": checks},
        "benchmark_source_sha256": source_identity,
        "source_sha256": {
            "reproducibility/build_host_gpu_concurrency_audit_v7.py": sha256(
                Path(__file__).resolve()
            ),
            "reproducibility/benchmark_training_kernel_v4.py": sha256(
                ROOT / "reproducibility" / "benchmark_training_kernel_v4.py"
            ),
        },
    }


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite host capacity audit {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--single-process-reference", type=Path, required=True)
    parser.add_argument("--worker-result", type=Path, action="append", required=True)
    parser.add_argument("--physical-gpu", action="append", required=True)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=DEFAULT_MINIMUM_FREE_BYTES
    )
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    payload = build_audit(
        host_label=args.host_label,
        single_process_reference=args.single_process_reference,
        worker_results=args.worker_result,
        physical_gpus=args.physical_gpu,
        minimum_free_bytes=args.minimum_free_bytes,
        minimum_aggregate_speedup=args.minimum_aggregate_speedup,
        minimum_worker_rate_fraction=args.minimum_worker_rate_fraction,
        descriptor_base=output.parent,
    )
    _write_new(output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "passed": payload["passed"],
                "aggregate_speedup": payload["aggregate_speedup"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
