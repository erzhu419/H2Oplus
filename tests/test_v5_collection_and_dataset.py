from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from bus_h2o.mdp_contract_v5 import (
    DEFAULT_CONFIG_V5,
    RAW_EVENT_STATE_NAMES,
    continuous_time_discount_v5,
    transition_duration_seconds_v5,
)
from collect_policy.build_transit_v5_dataset import (
    build_dataset,
    validate_collection_matrix,
)
from reproducibility import run_collection_matrix_v5 as matrix
from reproducibility.analyze_collection_matrix_v5 import analyze
from reproducibility.audit_transit_v5_dataset import audit


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(station: int, time_seconds: float, line: int = 0) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 300 + 10 * station,
        "backward_headway_seconds": 300 - 10 * station,
        "target_forward_headway_seconds": 300,
        "target_backward_headway_seconds": 300,
        "eligible_stranded_passengers": 0,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 8,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
        "system_waiting_passengers": 100 + station,
        "system_in_vehicle_passengers": 200 + station,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def _write_shard(root: Path, name: str, policy: str, policy_seed: int) -> Path:
    path = root / f"{name}.npz"
    observations = np.stack((_event(0, 10), _event(1, 70)))
    next_observations = np.stack((_event(1, 70), _event(2, 130)))
    actions = np.asarray([[-1.0], [0.0]], np.float32)
    # Values above float32's exact-integer range guard the immutable welfare ledger
    # against precision loss before the replay buffer's explicit training cast.
    wait_seconds = np.asarray([60_000_001.0, 60_000_101.0], np.float64)
    in_vehicle_seconds = np.asarray(
        [120_000_003.0, 120_000_103.0], np.float64
    )
    rewards = -(
        DEFAULT_CONFIG_V5.waiting_time_value * wait_seconds
        + DEFAULT_CONFIG_V5.in_vehicle_time_value * in_vehicle_seconds
    ) / DEFAULT_CONFIG_V5.reward_unit_passenger_seconds
    duration = np.asarray(
        transition_duration_seconds_v5(observations, next_observations),
        np.float32,
    )
    terminals = np.asarray([0, 1], np.uint8)
    discounts = np.asarray(continuous_time_discount_v5(duration), np.float32)
    discounts[terminals.astype(bool)] = 0.0
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        discounts=discounts,
        terminals=terminals,
        delta_t_seconds=duration,
        trajectory_local_id=np.zeros(2, np.int64),
        wait_passenger_seconds=wait_seconds,
        in_vehicle_passenger_seconds=in_vehicle_seconds,
    )
    generalized_per_passenger = 100.0 if policy == "zero" else 90.0
    summary = {
        "schema_version": "transit-smdp-v5-shard-v2",
        "contract_schema": "transit-smdp-v5",
        "contract": DEFAULT_CONFIG_V5.to_dict(),
        "raw_state_names": list(RAW_EVENT_STATE_NAMES),
        "shard_sha256": _hash(path),
        "split": "train",
        "diagnostic": False,
        "training_eligible": True,
        "policy_name": policy,
        "policy_seed": policy_seed,
        "sumo_seed": 101,
        "job_signature": name,
        "scenario": {
            "manifest_sha256": "scenario-a",
            "runtime_input_sha256": {"scenario.sumocfg": "runtime-hash"},
            "demand_scale": 1.0,
        },
        "episode": {
            "configured_seconds": 24_000,
            "observed_end_time_seconds": 24_000,
            "transitions": 2,
            "completed_trip_contexts": 1,
            "horizon_closures": 0,
        },
        "passenger_welfare": {
            "accrued_generalized_seconds_per_departed_passenger": generalized_per_passenger,
        },
        "passengers": {
            "accrued_wait_seconds_per_departed_passenger": generalized_per_passenger / 3.0,
            "accrued_in_vehicle_seconds_per_departed_passenger": generalized_per_passenger / 3.0,
            "departed_passengers": 100,
            "completed_passengers": 90,
            "completion_rate": 0.9,
        },
        "actions": {
            "mean_hold_seconds": 0.0 if policy == "zero" else 5.0,
            "exact_zero_rate": 1.0 if policy == "zero" else 0.5,
        },
        "ledger_audit": {
            "legacy_sampled_trip_histories_used_as_ground_truth": False,
            "count_source_audit": {
                "max_waiting_count_error": 0,
                "max_in_vehicle_count_error": 0,
            },
        },
        "training_reward": {"max_reward_reconstruction_error": 0.0},
        "service": {
            "departed_trips": 1,
            "completed_trips": 1,
            "active_trips_at_horizon": 0,
        },
        "source_sha256": {"collector.py": "source-hash"},
        "runtime": {
            "sumo_version": "SUMO test-version",
            "wall_time_seconds": 1.0,
        },
    }
    path.with_suffix(".npz.json").write_text(json.dumps(summary))
    return path


def _write_matrix(root: Path, shards: list[tuple[Path, str, int]]) -> Path:
    jobs = []
    for path, policy, policy_seed in shards:
        jobs.append(
            {
                "block_id": 0,
                "output": str(path.resolve()),
                "signature": path.stem,
                "policy": {"policy_name": policy},
                "policy_seed": policy_seed,
                "sumo_seed": 101,
                "split": "train",
                "diagnostic": False,
                "scenario": {
                    "sha256": "scenario-a",
                    "selection_seed": 1201,
                    "runtime_input_sha256": {
                        "scenario.sumocfg": "runtime-hash"
                    },
                },
            }
        )
    manifest = root / "matrix_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "transit-smdp-v5-collection-matrix-v1",
                "study_stage": "pilot",
                "environment_design": "paired",
                "independent_environment_blocks": 1,
                "collector_sha256": "collector-hash",
                "runner_sha256": "runner-hash",
                "source_sha256": {"collector.py": "source-hash"},
                "sumo_version": "SUMO test-version",
                "jobs": jobs,
            }
        )
    )
    return manifest


def test_v5_paired_blocks_and_policy_seeds_are_selection_independent():
    scenarios = [
        {"sha256": "a"},
        {"sha256": "b"},
        {"sha256": "c"},
    ]
    blocks = matrix._blocks(scenarios, [101, 102, 103], "paired")
    assert len(blocks) == 3
    assert [(row["scenario"]["sha256"], row["sumo_seed"]) for row in blocks] == [
        ("a", 101),
        ("b", 102),
        ("c", 103),
    ]
    seed = matrix.policy_seed(7, "headway-gain-0.1", "scenario", 101)
    assert seed == matrix.policy_seed(7, matrix.POLICIES[3]["policy_name"], "scenario", 101)
    assert seed != matrix.policy_seed(7, "headway-gain-0.1", "other", 101)
    with pytest.raises(ValueError, match="same number"):
        matrix._blocks(scenarios, [101], "paired")


def test_v5_matrix_and_builder_lock_all_shards_and_true_costs(tmp_path):
    first = _write_shard(tmp_path, "zero", "zero", 1)
    second = _write_shard(tmp_path, "random", "random-hurdle-p0.5-beta1-3", 2)
    manifest = _write_matrix(
        tmp_path,
        [
            (first, "zero", 1),
            (second, "random-hurdle-p0.5-beta1-3", 2),
        ],
    )
    provenance = validate_collection_matrix(manifest, [first, second])
    assert provenance["jobs"] == 2
    assert provenance["independent_environment_blocks"] == 1

    output = tmp_path / "dataset.h5"
    summary = build_dataset(
        [first, second],
        output,
        matrix_manifest=manifest,
    )
    assert summary["rows"] == 4
    assert summary["terminal_rows"] == 2
    assert summary["collection_matrix"]["environment_design"] == "paired"
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == "transit-smdp-v5"
        assert handle["observations"].shape == (4, 15)
        assert handle["raw_observations"].shape == (4, 16)
        np.testing.assert_array_equal(
            handle["rewards"][:],
            -(handle["waiting_cost"][:] + handle["in_vehicle_cost"][:]),
        )
        np.testing.assert_array_equal(
            handle["discounts"][:][handle["terminals"][:].astype(bool)],
            np.zeros(2, np.float32),
        )
        assert json.loads(handle.attrs["collection_matrix"])["jobs"] == 2

    analysis = analyze(manifest)
    assert analysis["ranking"] == ["random-hurdle-p0.5-beta1-3", "zero"]
    comparison = analysis["paired_vs_zero"]["random-hurdle-p0.5-beta1-3"]
    assert comparison["generalized_seconds_per_departed_passenger"]["wins"] == 1
    assert comparison["generalized_seconds_per_departed_passenger"][
        "absolute_difference"
    ]["mean"] == -10.0
    audit_report = audit(output, verify_source_files=False)
    assert audit_report["passed"] is True
    assert not any(audit_report["reconstruction_max_absolute_error"].values())

    with pytest.raises(ValueError, match="missing="):
        validate_collection_matrix(manifest, [first])


def test_v5_matrix_rejects_repeated_logical_job(tmp_path):
    first = _write_shard(tmp_path, "zero-a", "zero", 1)
    second = _write_shard(tmp_path, "random-a", "random-hurdle-p0.5-beta1-3", 2)
    manifest = _write_matrix(
        tmp_path,
        [
            (first, "zero", 1),
            (second, "random-hurdle-p0.5-beta1-3", 2),
        ],
    )
    payload = json.loads(manifest.read_text())
    duplicated_environment = dict(payload["jobs"][0])
    duplicated_environment["block_id"] = 1
    duplicated_environment["output"] = str(second.resolve())
    duplicated_environment["signature"] = "random-a"
    payload["jobs"] = [payload["jobs"][0], duplicated_environment]
    payload["independent_environment_blocks"] = 2
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="repeats logical job"):
        validate_collection_matrix(manifest, [first, second])
