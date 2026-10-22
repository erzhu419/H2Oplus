import json
from argparse import Namespace

import h5py
import numpy as np
import pytest
import torch

from bus_h2o.mdp_contract_v5 import (
    DEFAULT_CONFIG_V5,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot
from SimpleSAC.evaluate_transit_v5 import (
    _close_completed_trip_transitions,
    _transition_record,
    build_policy,
    load_dataset_contract,
    sha256,
    summarize_operations_v5,
)
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v5.networks import HurdleTransitActorV5
from SimpleSAC.transit_v5.trainer import TransitH2OTrainerV5


def _dataset(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["config"] = json.dumps(DEFAULT_CONFIG_V5.to_dict())
        handle.attrs["station_max_by_line"] = json.dumps([24.0] * 12)
    return path


def _raw_event(station=3, time_seconds=900):
    values = {
        "line_id": 0,
        "station_index": station,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 240,
        "backward_headway_seconds": 420,
        "target_forward_headway_seconds": 300,
        "target_backward_headway_seconds": 300,
        "eligible_stranded_passengers": 4,
        "onboard_passengers": 18,
        "eligible_arrival_rate_per_second": 0.04,
        "base_dwell_seconds": 12,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
        "system_waiting_passengers": 100,
        "system_in_vehicle_passengers": 200,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def _args(**overrides):
    values = {
        "zero_hold": False,
        "headway_balance_gain": None,
        "myopic_welfare": False,
        "checkpoint": None,
        "device": "cpu",
    }
    values.update(overrides)
    return Namespace(**values)


def test_v5_dataset_and_rule_policy_contract(tmp_path):
    path = _dataset(tmp_path / "dataset.h5")
    config, station_max, digest = load_dataset_contract(path)
    assert config == DEFAULT_CONFIG_V5
    assert digest == sha256(path)
    raw = _raw_event()[None, :]

    zero, _ = build_policy(_args(zero_hold=True), config, station_max, digest)
    np.testing.assert_array_equal(zero(raw), np.asarray([[-1.0]], np.float32))
    headway, _ = build_policy(
        _args(headway_balance_gain=0.1), config, station_max, digest
    )
    myopic, _ = build_policy(
        _args(myopic_welfare=True), config, station_max, digest
    )
    assert headway(raw).shape == (1, 1)
    assert myopic(raw).shape == (1, 1)

    with h5py.File(path, "a") as handle:
        handle.attrs["schema_version"] = "legacy"
    with pytest.raises(ValueError, match="transit-smdp-v5"):
        load_dataset_contract(path)


def test_v5_checkpoint_policy_requires_exact_provenance(tmp_path):
    dataset = _dataset(tmp_path / "dataset.h5")
    config, station_max, digest = load_dataset_contract(dataset)
    trainer_config = TransitTrainerConfig()
    actor = HurdleTransitActorV5(
        (trainer_config.hidden_dim, trainer_config.hidden_dim),
        layer_norm=trainer_config.actor_layer_norm,
        activation=trainer_config.activation,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_version": TransitH2OTrainerV5.CHECKPOINT_VERSION,
            "config": trainer_config.__dict__,
            "actor": actor.state_dict(),
            "extra": {
                "dataset_sha256": digest,
                "contract_config": config.to_dict(),
                "mdp_version": "v5",
            },
        },
        checkpoint,
    )
    policy, metadata = build_policy(
        _args(checkpoint=checkpoint), config, station_max, digest
    )
    assert policy(_raw_event()[None, :]).shape == (1, 1)
    assert metadata["sha256"] == sha256(checkpoint)
    with pytest.raises(ValueError, match="provenance mismatch"):
        build_policy(
            _args(checkpoint=checkpoint), config, station_max, "different"
        )


def test_v5_terminal_transition_uses_exact_ledger_and_is_not_summed():
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(70.0, 110, 210, 7_001.0, 14_003.0)
    previous = {
        "line_id": "10",
        "raw": _raw_event(station=0, time_seconds=10),
        "action": np.asarray([-1.0], np.float32),
        "passenger_time": start,
    }
    transition = _transition_record(
        previous,
        _raw_event(station=1, time_seconds=70),
        end,
        DEFAULT_CONFIG_V5,
        terminal=True,
    )
    assert transition["wait_passenger_seconds"] == 6_001.0
    assert transition["in_vehicle_passenger_seconds"] == 12_003.0
    assert transition["reward"] == -(2 * 6_001 + 12_003) / 100_000
    assert transition["discount"] == 0.0
    assert transition["terminal"] == 1
    operations = summarize_operations_v5([], [transition])
    assert "cumulative_smdp_reward" not in operations
    assert "not summed" in operations["reward_aggregation_warning"]


def test_v5_completion_context_closes_at_actual_trip_end():
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(75.0, 90, 180, 7_500.0, 14_000.0)
    pending = {
        "bus-a": {
            "line_id": "7X",
            "raw": _raw_event(station=4, time_seconds=10),
            "action": np.asarray([-1.0], np.float32),
            "passenger_time": start,
        }
    }

    class Bridge:
        def consume_completed_trip_passenger_times(self):
            return {"bus-a": end, "bus-without-decision": end}

    transitions = []
    contexts, closures = _close_completed_trip_transitions(
        Bridge(), pending, transitions, DEFAULT_CONFIG_V5
    )
    assert contexts == 2
    assert closures == 1
    assert pending == {}
    assert transitions[0]["terminal"] == 1
    assert transitions[0]["delta_t_seconds"] == 65.0
    assert transitions[0]["raw_next_observation"][2] == 75.0


def test_v5_passenger_summary_uses_ledger_not_sampled_histories():
    bridge = Namespace(
        departed_passenger_times={"a": 10.0, "b": 20.0, "c": 30.0},
        completed_passenger_times={"a": 70.0},
    )
    snapshot = PassengerTimeSnapshot(100.0, 1, 1, 321.0, 654.0)
    summary = summarize_passengers_v5(bridge, snapshot)
    assert summary["legacy_sampled_trip_histories_used_for_welfare"] is False
    assert summary["accrued_wait_seconds_total"] == 321.0
    assert summary["accrued_in_vehicle_seconds_total"] == 654.0
    assert summary["state_counts_at_horizon"]["other_active"] == 0
    assert summary["completed_journey_seconds"]["mean"] == 60.0
    assert summary["unfinished_departure_seconds"]["mean"] == 25.0
    assert summary["unfinished_elapsed_seconds_at_horizon"]["mean"] == 75.0
