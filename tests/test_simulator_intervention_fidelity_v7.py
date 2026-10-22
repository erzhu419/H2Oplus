from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from reproducibility.audit_simulator_intervention_fidelity_v7 import (
    DEMAND_SCALES,
    POLICIES,
    SIMULATOR_SEEDS,
    compare_interventions,
    intervention_source_hashes,
)


def _metrics(policy_index: int) -> dict[str, float]:
    return {
        "mean_hold_seconds": 0.0 if policy_index == 0 else float(policy_index),
        "exact_zero_rate": 1.0 if policy_index == 0 else 0.5,
        "control_decisions": 10_000.0 + policy_index * 2.0,
        "generalized_seconds_per_person": 2_000.0 + policy_index * 5.0,
        "wait_seconds_per_person": 600.0 - policy_index * 2.0,
        "in_vehicle_seconds_per_person": 800.0 + policy_index * 9.0,
        "completion_rate": 0.98 - policy_index * 0.001,
        "trip_duration_seconds": 2_800.0 + policy_index * 8.0,
    }


def _rows() -> tuple[list[dict], list[dict]]:
    sumo = []
    simulator = []
    for scale in DEMAND_SCALES:
        for block, seed in enumerate(SIMULATOR_SEEDS):
            for policy_index, policy in enumerate(POLICIES):
                common = {
                    "demand_scale": scale,
                    "policy_name": policy["policy_name"],
                    "metrics": _metrics(policy_index),
                }
                sumo.append({**common, "realization_block_id": block})
                simulator.append({**deepcopy(common), "simulator_seed": seed})
    return sumo, simulator


def test_identical_intervention_responses_pass() -> None:
    sumo, simulator = _rows()
    report = compare_interventions(sumo, simulator)
    assert report["passed"]


def test_large_intervention_response_error_fails() -> None:
    sumo, simulator = _rows()
    for row in simulator:
        if row["policy_name"] == "random-hurdle-p0.5-uniform":
            row["metrics"]["generalized_seconds_per_person"] += 500.0
    report = compare_interventions(sumo, simulator)
    assert not report["passed"]
    metric = report["comparisons"]["1"]["random-hurdle-p0.5-uniform"][
        "metrics"
    ]["generalized_seconds_per_person"]
    assert not metric["point_passed"]


def test_intervention_source_hash_keys_are_workspace_relative() -> None:
    source = intervention_source_hashes()
    assert source
    assert all(not Path(path).is_absolute() for path in source)
    assert "reproducibility/audit_simulator_intervention_fidelity_v7.py" in source
