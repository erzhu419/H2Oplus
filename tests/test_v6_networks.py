from __future__ import annotations

import numpy as np
import torch
from dataclasses import asdict
from torch import nn

from bus_h2o.mdp_contract_v6 import (
    CANONICAL_SLOT_WIDTH,
    CANONICAL_STATE_DIM,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV6,
    pack_event_batch_v6,
    raw_state_to_canonical_v6,
)
from SimpleSAC.transit_v6.networks import (
    FEATURE_DIM,
    HurdleTransitActorV6,
    TransitDynamicsRatioV6,
    TransitQEnsembleV6,
    TransitStateEncoderV6,
    TransitValueV6,
    structured_slot_features_v6,
)
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v6.trainer import TransitH2OTrainerV6


class _V6ReplayStub:
    contract_config = asdict(TransitMDPConfigV6())


class _CountingEncoder(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(states)


def _event(line: int, station: int, time: float) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 360.0,
        "target_forward_headway_seconds": 330.0,
        "target_backward_headway_seconds": 330.0,
        "eligible_stranded_passengers": 5,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 9,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
        "system_waiting_passengers": 100,
        "system_in_vehicle_passengers": 200,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], dtype=np.float32)


def _states() -> torch.Tensor:
    config = TransitMDPConfigV6()
    station_max = np.full(12, 24.0)
    batch = np.stack((_event(1, 3, 100.0), _event(11, 4, 100.0), _event(7, 2, 100.0)))
    rows = [
        pack_event_batch_v6(batch, 0, np.asarray([-1.0, -1.0, -1.0]), config),
        pack_event_batch_v6(batch, 1, np.asarray([-1.0, -1.0, -1.0]), config),
        pack_event_batch_v6(batch, 2, np.asarray([-1.0, 0.25, -1.0]), config),
    ]
    canonical = raw_state_to_canonical_v6(np.stack(rows), station_max, config)
    return torch.as_tensor(canonical, dtype=torch.float32)


def test_v6_structured_features_disambiguate_selected_zero_action() -> None:
    decoded = structured_slot_features_v6(_states()[1:2])
    assert decoded["selected"][0, 0]
    assert decoded["current"][0, 1]
    assert decoded["future"][0, 2]
    assert decoded["local"][0, 0, -4] == -1.0
    assert decoded["local"][0, 0, -3] == 1.0
    assert decoded["local"][0, 2, -4] == -1.0
    assert decoded["local"][0, 2, -3] == 0.0


def test_v6_encoder_masks_inactive_slots() -> None:
    states = _states()[0:1]
    changed = states.clone()
    inactive_start = 3 + 3 * CANONICAL_SLOT_WIDTH
    changed[:, inactive_start + 1 : inactive_start + CANONICAL_SLOT_WIDTH] = 0.73
    encoder = TransitStateEncoderV6().eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(states), encoder(changed))
    assert encoder(states).shape == (1, FEATURE_DIM)


def test_v6_network_and_ratio_shapes_are_finite() -> None:
    states = _states()
    actions = torch.tensor([[-1.0], [0.0], [0.5]])
    actor = HurdleTransitActorV6((32, 32))
    critic = TransitQEnsembleV6(3, (32, 32))
    value = TransitValueV6((32, 32))
    ratio = TransitDynamicsRatioV6((32, 32))

    actor_actions, log_prob = actor(states, deterministic=True)
    assert actor_actions.shape == (3, 1)
    assert log_prob.shape == (3,)
    assert critic(states, actions).shape == (3, 3)
    assert value(states).shape == (3,)
    batch = {
        "observations": states,
        "actions": actions,
        "next_observations": states.roll(1, dims=0),
        "discounts": torch.ones(3),
        "delta_t_seconds": torch.tensor([0.0, 60.0, 24_000.0]),
    }
    joint_features = ratio._joint_features(batch)
    torch.testing.assert_close(
        joint_features[:, -1], torch.tensor([0.0, 60.0 / 24_000.0, 1.0])
    )
    counting_encoder = _CountingEncoder(ratio.feature_fn)
    ratio.feature_fn = counting_encoder
    joint, state_action = ratio.logits(batch)
    assert counting_encoder.calls == 1
    assert joint.shape == state_action.shape == (3,)
    assert torch.isfinite(joint).all()
    assert any(name.startswith("feature_fn.") for name in ratio.state_dict())
    assert states.shape[1] == CANONICAL_STATE_DIM


def test_v6_critic_feature_reuse_matches_repeated_state_evaluation() -> None:
    states = _states()
    critic = TransitQEnsembleV6(3, (32, 32))
    repeats = 4
    actions = torch.linspace(-1.0, 1.0, len(states) * repeats).reshape(-1, 1)
    repeated_states = states[:, None, :].expand(-1, repeats, -1).reshape(
        len(states) * repeats, -1
    )
    features = critic.encode_states(states)
    repeated_features = features[:, None, :].expand(-1, repeats, -1).reshape(
        len(states) * repeats, -1
    )
    torch.testing.assert_close(
        critic(repeated_states, actions),
        critic.forward_features(repeated_features, actions),
    )
    indices = torch.tensor([2, 0], dtype=torch.long)
    torch.testing.assert_close(
        critic.forward_subset(repeated_states, actions, indices),
        critic(repeated_states, actions).index_select(0, indices),
    )
    torch.testing.assert_close(
        critic.forward_features_subset(repeated_features, actions, indices),
        critic.forward_features(repeated_features, actions).index_select(0, indices),
    )


def test_v6_actor_repeated_components_match_direct_encoding() -> None:
    states = _states()
    actor = HurdleTransitActorV6((32, 32))
    repeats = 3
    repeated_states = states[:, None, :].expand(-1, repeats, -1).reshape(
        len(states) * repeats, -1
    )
    direct = actor.soft_action_components(
        repeated_states, deterministic_positive=True
    )
    reused = actor.repeated_soft_action_components(
        states, repeats, deterministic_positive=True
    )
    for key in direct:
        torch.testing.assert_close(direct[key], reused[key])


def test_v6_compilation_includes_all_structured_encoders() -> None:
    trainer = TransitH2OTrainerV6(
        _V6ReplayStub(),
        config=TransitTrainerConfig(
            ensemble_size=2,
            hidden_dim=16,
            batch_size=4,
            target_subset_size=2,
            offline_objective="iql",
            online_backup="h2oplus",
        ),
        seed=81,
    )
    compiled = set(trainer.compile_networks())
    assert {
        "actor_encoder",
        "critic_encoder",
        "target_critic_encoder",
        "value_encoder",
        "ratio_encoder",
    } <= compiled
    assert all("_orig_mod" not in key for key in trainer.actor.state_dict())
