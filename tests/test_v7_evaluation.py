from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from bus_h2o.mdp_contract_v7 import (
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V7,
    RAW_EVENT_STATE_NAMES,
    RAW_STATE_DIM,
    RAW_STATE_NAMES,
    SCHEMA_VERSION,
    pack_event_batch_v7,
    policy_action_to_hold_seconds_v7,
    terminal_raw_state_v7,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot
from collect_policy.build_transit_v7_dataset import validate_serialized_sequence
from collect_policy.collect_transit_v7 import append_transition
from SimpleSAC.evaluate_transit_v7 import (
    build_policy,
    load_dataset_contract,
    sha256,
    transition_arrays,
)
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


def _event(line: int, station: int, time_seconds: float) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 240.0,
        "backward_headway_seconds": 420.0,
        "target_forward_headway_seconds": 300.0,
        "target_backward_headway_seconds": 300.0,
        "eligible_stranded_passengers": 4.0,
        "onboard_passengers": 18.0,
        "eligible_arrival_rate_per_second": 0.04,
        "base_dwell_seconds": 12.0,
        "forward_bus_present": 1.0,
        "backward_bus_present": 1.0,
        "bus_capacity": 50.0,
        "system_waiting_passengers": 100.0,
        "system_in_vehicle_passengers": 200.0,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def _dataset(path: Path, rows: int = 2) -> Path:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["config"] = json.dumps(DEFAULT_CONFIG_V7.to_dict())
        handle.attrs["station_max_by_line"] = json.dumps([40.0] * 12)
        handle.attrs["raw_state_dim"] = RAW_STATE_DIM
        handle.attrs["state_dim"] = CANONICAL_STATE_DIM
        handle.attrs["action_dim"] = 1
        handle.attrs["raw_state_names"] = json.dumps(list(RAW_STATE_NAMES))
        handle.attrs["state_names"] = json.dumps(list(CANONICAL_STATE_NAMES))
        handle.attrs["n_transitions"] = rows
        handle.attrs["content_signature"] = "content"
        handle.attrs["derivation_signature"] = "derivation"
        handle.create_dataset(
            "raw_observations", data=np.zeros((rows, RAW_STATE_DIM), np.float32)
        )
        handle.create_dataset(
            "raw_next_observations",
            data=np.zeros((rows, RAW_STATE_DIM), np.float32),
        )
        handle.create_dataset(
            "observations", data=np.zeros((rows, CANONICAL_STATE_DIM), np.float32)
        )
        handle.create_dataset(
            "next_observations",
            data=np.zeros((rows, CANONICAL_STATE_DIM), np.float32),
        )
        handle.create_dataset("actions", data=np.zeros((rows, 1), np.float32))
    return path


def _policy_args(**overrides) -> Namespace:
    values = {
        "zero_hold": False,
        "headway_balance_gain": None,
        "myopic_welfare": False,
        "checkpoint": None,
        "device": "cpu",
        "expected_algorithm": None,
        "expected_training_seed": None,
        "expected_training_run_signature": None,
        "expected_training_admission_sha256": None,
        "expected_training_checkpoint_sha256": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_v7_evaluation_dataset_contract_supports_parent_hash_stat_lock(tmp_path):
    dataset = _dataset(tmp_path / "dataset.h5")
    digest = sha256(dataset)
    stat = dataset.stat()

    config, station_max, actual, identity = load_dataset_contract(
        dataset,
        verified_sha256=digest,
        verified_size_bytes=stat.st_size,
        verified_mtime_ns=stat.st_mtime_ns,
    )

    assert config == DEFAULT_CONFIG_V7
    assert station_max.shape == (12,)
    assert actual == digest
    assert identity["verification"] == "parent-matrix-sha256-plus-child-stat-lock"
    assert identity["rows"] == 2
    with pytest.raises(ValueError, match="stat identity changed"):
        load_dataset_contract(
            dataset,
            verified_sha256=digest,
            verified_size_bytes=stat.st_size,
            verified_mtime_ns=stat.st_mtime_ns + 1,
        )


def test_v7_rule_policies_consume_one_serialized_current_event():
    events = np.stack((_event(0, 3, 900.0), _event(4, 8, 900.0)))
    first = pack_event_batch_v7(events, 0)
    second = pack_event_batch_v7(
        events, 1, np.asarray([0.25, -1.0], dtype=np.float32)
    )
    zero, _ = build_policy(
        _policy_args(zero_hold=True), DEFAULT_CONFIG_V7, np.full(12, 40.0), "d" * 64
    )
    headway, _ = build_policy(
        _policy_args(headway_balance_gain=0.1),
        DEFAULT_CONFIG_V7,
        np.full(12, 40.0),
        "d" * 64,
    )
    myopic, _ = build_policy(
        _policy_args(myopic_welfare=True),
        DEFAULT_CONFIG_V7,
        np.full(12, 40.0),
        "d" * 64,
    )

    np.testing.assert_array_equal(zero(first), [-1.0])
    assert headway(first).shape == (1,)
    assert headway(second).shape == (1,)
    assert myopic(second).shape == (1,)


def test_v7_formal_policy_artifact_is_compact_and_provenance_locked(tmp_path):
    class Replay:
        contract_config = DEFAULT_CONFIG_V7.to_dict()

    trainer_config = TransitTrainerConfig(
        hidden_dim=8,
        ensemble_size=2,
        batch_size=4,
    )
    trainer = TransitH2OTrainerV7(
        Replay(), config=trainer_config, device="cpu", seed=7
    )
    dataset_hash = "d" * 64
    extra = {
        "epoch": 10,
        "phase": "complete",
        "run_signature": "formal-run",
        "method": "rlpd",
        "training_seed": 3,
        "dataset_sha256": dataset_hash,
        "contract_config": DEFAULT_CONFIG_V7.to_dict(),
        "mdp_version": "v7",
        "training_checkpoint_sha256": "c" * 64,
        "training_admission": {
            "schema_version": "transit-training-admission-v7-v2",
            "sha256": "a" * 64,
        },
    }
    artifact = tmp_path / "final_actor.pt"
    trainer.save_policy_artifact(artifact, extra=extra)

    policy, metadata = build_policy(
        _policy_args(
            checkpoint=artifact,
            expected_algorithm="rlpd",
            expected_training_seed=3,
        ),
        DEFAULT_CONFIG_V7,
        np.full(12, 40.0, np.float32),
        dataset_hash,
    )
    raw_state = pack_event_batch_v7(np.stack((_event(0, 3, 900.0),)), 0)
    action = policy(raw_state)
    assert action.shape == (1,)
    assert np.isfinite(action).all()
    assert metadata["artifact_schema"] == "transit-policy-artifact-v7-v1"
    assert artifact.stat().st_size < 5_000_000

    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["extra"]["phase"] = "online_training"
    stale = tmp_path / "stale_actor.pt"
    torch.save(payload, stale)
    with pytest.raises(ValueError, match="provenance mismatch"):
        build_policy(
            _policy_args(
                checkpoint=stale,
                expected_algorithm="rlpd",
                expected_training_seed=3,
            ),
            DEFAULT_CONFIG_V7,
            np.full(12, 40.0, np.float32),
            dataset_hash,
        )


def test_v7_evaluation_transition_arrays_preserve_same_time_credit():
    events = np.stack((_event(0, 3, 10.0), _event(4, 8, 10.0)))
    first = pack_event_batch_v7(events, 0)
    first_action = np.asarray([0.25], np.float32)
    second = pack_event_batch_v7(events, 1, np.asarray([0.25, -1.0], np.float32))
    second_action = np.asarray([-1.0], np.float32)
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(70.0, 90, 180, 7_001.0, 14_003.0)
    rows = {
        name: []
        for name in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "discounts",
            "terminals",
            "delta_t_seconds",
            "trajectory_local_id",
            "wait_passenger_seconds",
            "in_vehicle_passenger_seconds",
            "event_bus_id",
            "event_line_index",
            "event_station_index",
            "event_decision_time_seconds",
            "event_base_dwell_seconds",
        )
    }
    first_pending = {
        "raw": first,
        "action": first_action,
        "passenger_time": start,
        "event_bus_id": "bus-a",
        "event_line_index": 0,
        "event_station_index": 3,
        "event_decision_time_seconds": 10.0,
        "event_base_dwell_seconds": 12.0,
    }
    append_transition(rows, first_pending, second, start, terminal=False)
    second_pending = {
        "raw": second,
        "action": second_action,
        "passenger_time": start,
        "event_bus_id": "bus-b",
        "event_line_index": 4,
        "event_station_index": 8,
        "event_decision_time_seconds": 10.0,
        "event_base_dwell_seconds": 12.0,
    }
    append_transition(
        rows,
        second_pending,
        terminal_raw_state_v7(second, end),
        end,
        terminal=True,
    )
    holds = [
        policy_action_to_hold_seconds_v7(first_action[0]),
        policy_action_to_hold_seconds_v7(second_action[0]),
    ]
    arrays, bus_ids = transition_arrays(rows, holds)

    validate_serialized_sequence(arrays, Path("synthetic-v7-evaluation"))
    assert bus_ids == ["bus-a", "bus-b"]
    assert arrays["delta_t_seconds"].tolist() == [0.0, 60.0]
    assert arrays["rewards"][0] == 0.0
    np.testing.assert_array_equal(
        arrays["next_observations"][0], arrays["observations"][1]
    )
