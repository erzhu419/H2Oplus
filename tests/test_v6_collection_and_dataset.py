from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from bus_h2o.mdp_contract_v6 import (
    DEFAULT_CONFIG_V6,
    RAW_EVENT_STATE_NAMES,
    RAW_SLOT_WIDTH,
    RAW_STATE_NAMES,
    continuous_time_discount_v6,
    pack_event_batch_v6,
    terminal_raw_state_v6,
    transition_cost_components_v6,
    transition_reward_v6,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot
from collect_policy.build_transit_v6_dataset import (
    build_dataset,
    validate_collection_matrix,
    validate_serialized_sequence,
)
from reproducibility.audit_transit_v6_dataset import audit


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(line: int, station: int, time_seconds: float) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 360.0,
        "target_forward_headway_seconds": 330.0,
        "target_backward_headway_seconds": 330.0,
        "eligible_stranded_passengers": 4.0,
        "onboard_passengers": 20.0,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 8.0,
        "forward_bus_present": 1.0,
        "backward_bus_present": 1.0,
        "bus_capacity": 50.0,
        "system_waiting_passengers": 100.0,
        "system_in_vehicle_passengers": 200.0,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def _write_shard(root: Path) -> Path:
    path = root / "zero" / "scenario" / "sumo_seed1.npz"
    path.parent.mkdir(parents=True)
    events = np.stack((_event(0, 2, 10), _event(3, 5, 10)))
    first = pack_event_batch_v6(events, 0)
    second = pack_event_batch_v6(
        events, 1, np.asarray([0.25, -1.0], np.float32)
    )
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(70.0, 90, 180, 7_001.0, 14_003.0)
    terminal = terminal_raw_state_v6(second, end)
    components = transition_cost_components_v6(start, end)
    rewards = np.asarray([0.0, transition_reward_v6(start, end)], np.float64)
    durations = np.asarray([0.0, 60.0], np.float64)
    terminals = np.asarray([0, 1], np.uint8)
    discounts = np.asarray(continuous_time_discount_v6(durations), np.float32)
    discounts[terminals.astype(bool)] = 0.0
    arrays = {
        "observations": np.stack((first, second)),
        "actions": np.asarray([[0.25], [-1.0]], np.float32),
        "rewards": rewards,
        "next_observations": np.stack((second, terminal)),
        "discounts": discounts,
        "terminals": terminals,
        "delta_t_seconds": durations,
        "trajectory_local_id": np.zeros(2, np.int64),
        "wait_passenger_seconds": np.asarray(
            [0.0, components["wait_passenger_seconds"]], np.float64
        ),
        "in_vehicle_passenger_seconds": np.asarray(
            [0.0, components["in_vehicle_passenger_seconds"]], np.float64
        ),
    }
    np.savez_compressed(path, **arrays)
    summary = {
        "schema_version": "transit-smdp-v6-shard-v1",
        "contract_schema": "transit-smdp-v6",
        "contract": DEFAULT_CONFIG_V6.to_dict(),
        "raw_state_names": list(RAW_STATE_NAMES),
        "shard_sha256": _hash(path),
        "job_signature": None,
        "split": "train",
        "diagnostic": True,
        "training_eligible": False,
        "policy_name": "zero",
        "policy_seed": 1,
        "sumo_seed": 1,
        "scenario": {
            "manifest_sha256": "scenario",
            "selection_seed": 101,
            "demand_scale": 1.0,
        },
        "episode": {
            "configured_seconds": 70,
            "observed_end_time_seconds": 70,
            "control_decisions": 2,
            "transitions": 2,
        },
        "training_reward": {
            "reward_sum": float(rewards.sum()),
            "chronological_coverage_errors": {
                "wait_passenger_seconds": 0.0,
                "in_vehicle_passenger_seconds": 0.0,
                "reward": 0.0,
            },
        },
        "ledger_audit": {
            "legacy_sampled_trip_histories_used_as_ground_truth": False,
            "count_source_audit": {
                "max_waiting_count_error": 0,
                "max_in_vehicle_count_error": 0,
            },
        },
        "service": {
            "departed_trips": 1,
            "completed_trips": 0,
            "active_trips_at_horizon": 1,
        },
    }
    path.with_suffix(".npz.json").write_text(json.dumps(summary))
    return path


def test_v6_builder_and_independent_audit_preserve_chronology(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    dataset = tmp_path / "dataset.h5"
    summary = build_dataset([shard], dataset, allow_partial=True)
    assert summary["rows"] == 2
    assert summary["trajectories"] == 1
    assert summary["terminal_rows"] == 1
    assert summary["zero_duration_rows"] == 1
    with h5py.File(dataset, "r") as handle:
        np.testing.assert_array_equal(handle["demand_realization_seed"][:], 101)
        assert handle.attrs["discount_semantics"].startswith(
            "finite-horizon undiscounted objective"
        )
    report = audit(dataset)
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["chronological_reward_sums"]["0"] == pytest.approx(
        summary["reward"]["sum"]
    )


def test_v6_validation_rejects_lost_same_time_action(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    with np.load(shard, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]).copy() for name in handle.files}
    action_position = 3 + RAW_SLOT_WIDTH - 1
    arrays["next_observations"][0, action_position] = -1.0
    with pytest.raises(ValueError, match="lost the preceding action"):
        validate_serialized_sequence(arrays, shard)


def test_v6_validation_rejects_multiple_terminal_rows(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    with np.load(shard, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]).copy() for name in handle.files}
    arrays["terminals"][0] = 1
    with pytest.raises(ValueError, match="exactly one terminal"):
        validate_serialized_sequence(arrays, shard)


def test_v6_builder_accepts_and_audits_collection_matrix_v2(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    manifest = tmp_path / "matrix.json"
    payload = {
        "schema_version": "transit-smdp-v6-collection-matrix-v2",
        "study_stage": "formal",
        "environment_design": {"demand_scales": [1.0]},
        "independent_realization_blocks": 1,
        "environment_cells": 1,
        "source_sha256": None,
        "jobs": [
            {
                "block_id": 0,
                "realization_block_id": 0,
                "output": str(shard.resolve()),
                "signature": None,
                "policy": {"policy_name": "zero"},
                "policy_seed": 1,
                "sumo_seed": 1,
                "split": "train",
                "diagnostic": True,
                "scenario": {
                    "sha256": "scenario",
                    "selection_seed": 101,
                    "demand_scale": 1.0,
                },
            }
        ],
    }
    manifest.write_text(json.dumps(payload))

    provenance = validate_collection_matrix(manifest, [shard])
    assert provenance["schema_version"] == payload["schema_version"]
    assert provenance["independent_realization_blocks"] == 1
    assert provenance["environment_cells"] == 1

    payload["environment_design"] = {"demand_scales": []}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lacks declared demand scales"):
        validate_collection_matrix(manifest, [shard])
