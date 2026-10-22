import json
from argparse import Namespace

import h5py
import numpy as np
import pytest
import torch

from bus_h2o.mdp_contract_v4 import (
    DEFAULT_CONFIG_V4,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
)
from SimpleSAC.evaluate_transit_v4 import (
    build_policy,
    load_dataset_contract,
    sha256,
)
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v4.networks import HurdleTransitActor
from SimpleSAC.transit_v4.trainer import TransitH2OTrainerV4


def _dataset(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["config"] = json.dumps(DEFAULT_CONFIG_V4.to_dict())
        handle.attrs["station_max_by_line"] = json.dumps([24.0] * 12)
    return path


def _raw_event():
    values = {
        "line_id": 0,
        "station_index": 3,
        "sim_time_seconds": 900,
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


def test_load_dataset_contract_requires_v4_and_valid_station_scales(tmp_path):
    path = _dataset(tmp_path / "dataset.h5")
    config, station_max, digest = load_dataset_contract(path)
    assert config == DEFAULT_CONFIG_V4
    np.testing.assert_array_equal(station_max, np.full(12, 24.0, np.float32))
    assert digest == sha256(path)

    with h5py.File(path, "a") as handle:
        handle.attrs["schema_version"] = "legacy"
    with pytest.raises(ValueError, match="transit-smdp-v4"):
        load_dataset_contract(path)


def test_rule_policies_share_the_hurdle_action_contract(tmp_path):
    path = _dataset(tmp_path / "dataset.h5")
    config, station_max, digest = load_dataset_contract(path)
    raw = _raw_event()[None, :]

    zero, zero_metadata = build_policy(
        _args(zero_hold=True), config, station_max, digest
    )
    np.testing.assert_array_equal(zero(raw), np.asarray([[-1.0]], np.float32))
    assert zero_metadata["type"] == "zero"

    headway, _ = build_policy(
        _args(headway_balance_gain=0.1), config, station_max, digest
    )
    action = headway(raw)
    assert action.shape == (1, 1)
    assert -1.0 < action.item() <= 1.0

    myopic, _ = build_policy(
        _args(myopic_welfare=True), config, station_max, digest
    )
    myopic_action = myopic(raw)
    assert myopic_action.shape == (1, 1)
    assert -1.0 <= myopic_action.item() <= 1.0


def test_checkpoint_policy_requires_exact_dataset_and_contract(tmp_path):
    dataset = _dataset(tmp_path / "dataset.h5")
    config, station_max, digest = load_dataset_contract(dataset)
    trainer_config = TransitTrainerConfig()
    actor = HurdleTransitActor(
        (trainer_config.hidden_dim, trainer_config.hidden_dim),
        layer_norm=trainer_config.actor_layer_norm,
        activation=trainer_config.activation,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_version": TransitH2OTrainerV4.CHECKPOINT_VERSION,
            "config": trainer_config.__dict__,
            "actor": actor.state_dict(),
            "extra": {
                "dataset_sha256": digest,
                "contract_config": config.to_dict(),
                "mdp_version": "v4",
            },
        },
        checkpoint,
    )

    policy, metadata = build_policy(
        _args(checkpoint=checkpoint), config, station_max, digest
    )
    action = policy(_raw_event()[None, :])
    assert action.shape == (1, 1)
    assert metadata["sha256"] == sha256(checkpoint)

    with pytest.raises(ValueError, match="provenance mismatch"):
        build_policy(
            _args(checkpoint=checkpoint),
            config,
            station_max,
            "different-dataset",
        )
