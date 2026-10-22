from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.build_host_gpu_concurrency_audit_v7 import (
    _logical_source_map,
    build_audit,
)
from SimpleSAC.train_transit_v3 import sha256


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "reproducibility" / "benchmark_training_kernel_v4.py"


def _report(*, rate: float, cpu_threads: int, gpu: str) -> dict:
    source_path = Path("/relocated/workspace/reproducibility") / BENCHMARK.name
    return {
        "schema_version": "transit-training-kernel-benchmark-v3",
        "method": "rlpd",
        "mdp_version": "v7",
        "phase": "online",
        "dataset_sha256": "1" * 64,
        "device_cache_mode": "float32",
        "compile_networks": True,
        "compile_mode": "default",
        "max_canonical_processes_per_gpu": 1,
        "max_offline_rows": 0,
        "allow_shared_gpu": False,
        "allow_external_gpu_processes": False,
        "warmup_updates": 100,
        "measured_updates": 5_000,
        "updates_per_second": rate,
        "trainer_config": {
            "reward_scale": 0.0001,
            "critic_loss": "huber",
            "huber_delta": 1.0,
            "target_entropy": 0.0,
        },
        "runtime": {
            "host": {
                "cpu_threads": cpu_threads,
                "cuda_visible_devices": gpu,
                "gpu": "NVIDIA GeForce RTX 3080 Ti",
            }
        },
        "resources": {
            "max_rss_bytes": 8 * 2**30,
            "cuda": {
                "total_bytes": 12 * 2**30,
                "training": {"peak_reserved_bytes": 7 * 2**30},
            },
        },
        "source_sha256": {str(source_path): sha256(BENCHMARK)},
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_logical_source_map_accepts_relocated_deployment_paths():
    raw = {
        "/some/deployment/workspace/reproducibility/benchmark_training_kernel_v4.py": sha256(
            BENCHMARK
        )
    }
    assert _logical_source_map(raw) == {
        "reproducibility/benchmark_training_kernel_v4.py": sha256(BENCHMARK)
    }


def test_host_concurrency_audit_requires_distinct_gpus_and_useful_throughput(
    tmp_path,
):
    reference = _write(tmp_path / "reference.json", _report(rate=100, cpu_threads=8, gpu="0"))
    worker0 = _write(tmp_path / "worker0.json", _report(rate=94, cpu_threads=6, gpu="0"))
    worker1 = _write(tmp_path / "worker1.json", _report(rate=92, cpu_threads=6, gpu="1"))

    result = build_audit(
        host_label="test-host",
        single_process_reference=reference,
        worker_results=[worker0, worker1],
        physical_gpus=["0", "1"],
    )

    assert result["passed"] is True
    assert result["aggregate_speedup"] == pytest.approx(1.86)
    assert result["decision"]["checks"]["per_gpu_memory_headroom"] is True

    with pytest.raises(ValueError, match="distinct GPU"):
        build_audit(
            host_label="test-host",
            single_process_reference=reference,
            worker_results=[worker0, worker1],
            physical_gpus=["0", "0"],
        )


def test_host_concurrency_audit_can_emit_portable_relative_descriptors(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    output_dir = tmp_path / "audits" / "host"
    output_dir.mkdir(parents=True)
    reference = _write(
        reports / "reference.json",
        _report(rate=100, cpu_threads=8, gpu="0"),
    )
    worker0 = _write(
        reports / "worker0.json",
        _report(rate=94, cpu_threads=6, gpu="0"),
    )
    worker1 = _write(
        reports / "worker1.json",
        _report(rate=92, cpu_threads=6, gpu="1"),
    )

    result = build_audit(
        host_label="test-host",
        single_process_reference=reference,
        worker_results=[worker0, worker1],
        physical_gpus=["0", "1"],
        descriptor_base=output_dir,
    )

    descriptors = [result["single_process_reference"], *result["workers"]]
    assert all(not Path(item["path"]).is_absolute() for item in descriptors)
    assert all((output_dir / item["path"]).resolve().is_file() for item in descriptors)


def test_host_concurrency_audit_fails_closed_on_insufficient_speedup(tmp_path):
    reference = _write(tmp_path / "reference.json", _report(rate=100, cpu_threads=8, gpu="0"))
    worker0 = _write(tmp_path / "worker0.json", _report(rate=50, cpu_threads=6, gpu="0"))
    worker1 = _write(tmp_path / "worker1.json", _report(rate=50, cpu_threads=6, gpu="1"))

    result = build_audit(
        host_label="test-host",
        single_process_reference=reference,
        worker_results=[worker0, worker1],
        physical_gpus=["0", "1"],
    )

    assert result["passed"] is False
    assert result["decision"]["checks"]["minimum_aggregate_speedup"] is False
    assert result["decision"]["checks"]["minimum_worker_rate_fraction"] is False
