from __future__ import annotations

import pytest

from reproducibility.audit_gpu_capacity_v7 import (
    _classify_compute_pids,
    _compute_pids_for_gpu,
    capacity_decision,
    throughput_decision,
)


def test_compute_pid_filter_is_scoped_to_the_target_gpu_uuid():
    rows = "GPU-a, 101\nGPU-b, 202\nGPU-a, 303\n"
    assert _compute_pids_for_gpu(rows, "GPU-a") == [101, 303]
    assert _compute_pids_for_gpu(rows, "GPU-b") == [202]
    with pytest.raises(RuntimeError, match="compute row"):
        _compute_pids_for_gpu("GPU-a, not-a-pid\n", "GPU-a")


def test_compute_pid_classification_accepts_only_worker_process_groups(
    monkeypatch,
):
    process_groups = {101: 101, 201: 101, 999: 999}
    monkeypatch.setattr(
        "reproducibility.audit_gpu_capacity_v7.os.getpgid",
        lambda pid: process_groups[pid],
    )

    result = _classify_compute_pids([101, 201, 999], {101, 102})

    assert result["registered_compute_pids"] == [101, 201]
    assert result["unregistered_compute_pids"] == [999]
    assert result["registered_worker_process_groups"] == [101]


def test_capacity_decision_counts_registered_cuda_children_as_their_workers():
    result = capacity_decision(
        workers=_workers(),
        samples=[
            {
                "compute_pids": [201, 202],
                "registered_compute_pids": [201, 202],
                "unregistered_compute_pids": [],
                "registered_worker_process_groups": [101, 102],
                "memory_total_bytes": 8 * 2**30,
                "memory_used_bytes": 6 * 2**30,
            }
        ],
        expected_processes=2,
        minimum_free_bytes=1 * 2**30,
    )

    assert result["passed"] is True
    assert result["maximum_concurrent_registered_processes"] == 2
    assert result["unexpected_compute_pids"] == []


def _workers() -> list[dict]:
    return [
        {"pid": 101, "returncode": 0, "report_validation_failures": []},
        {"pid": 102, "returncode": 0, "report_validation_failures": []},
    ]


def test_capacity_decision_requires_overlap_and_memory_headroom():
    result = capacity_decision(
        workers=_workers(),
        samples=[
            {
                "compute_pids": [101, 102],
                "memory_total_bytes": 8 * 2**30,
                "memory_used_bytes": 6 * 2**30,
            }
        ],
        expected_processes=2,
        minimum_free_bytes=1 * 2**30,
    )

    assert result["passed"] is True
    assert result["maximum_concurrent_registered_processes"] == 2
    assert result["minimum_free_memory_observed_bytes"] == 2 * 2**30


def test_capacity_decision_fails_closed_on_external_pid_or_worker_failure():
    workers = _workers()
    workers[1]["returncode"] = 137
    result = capacity_decision(
        workers=workers,
        samples=[
            {
                "compute_pids": [101, 102, 999],
                "memory_total_bytes": 8 * 2**30,
                "memory_used_bytes": 15 * 2**29,
            }
        ],
        expected_processes=2,
        minimum_free_bytes=1 * 2**30,
    )

    assert result["passed"] is False
    assert result["checks"]["all_workers_completed"] is False
    assert result["checks"]["no_unregistered_compute_process"] is False
    assert result["checks"]["minimum_free_memory_headroom"] is False
    assert result["unexpected_compute_pids"] == [999]


def test_throughput_decision_requires_useful_aggregate_speedup_and_fair_workers():
    result = throughput_decision(
        worker_rates=[80.0, 85.0],
        single_process_rate=100.0,
        expected_processes=2,
        minimum_aggregate_speedup=1.2,
        minimum_worker_rate_fraction=0.55,
    )
    assert result["passed"] is True
    assert result["aggregate_speedup"] == pytest.approx(1.65)
    assert result["minimum_observed_worker_rate_fraction"] == pytest.approx(0.8)

    result = throughput_decision(
        worker_rates=[40.0, 40.0],
        single_process_rate=100.0,
        expected_processes=2,
        minimum_aggregate_speedup=1.2,
        minimum_worker_rate_fraction=0.55,
    )
    assert result["passed"] is False
    assert result["checks"]["minimum_aggregate_speedup"] is False
    assert result["checks"]["minimum_worker_rate_fraction"] is False


def test_throughput_decision_fails_closed_when_a_worker_rate_is_missing():
    result = throughput_decision(
        worker_rates=[80.0],
        single_process_rate=100.0,
        expected_processes=2,
        minimum_aggregate_speedup=1.2,
        minimum_worker_rate_fraction=0.55,
    )
    assert result["passed"] is False
    assert result["checks"]["all_worker_rates_available"] is False
