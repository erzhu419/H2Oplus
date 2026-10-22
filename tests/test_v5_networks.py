from __future__ import annotations

import torch

from bus_h2o.mdp_contract_v5 import CANONICAL_STATE_DIM
from SimpleSAC.transit_v5.networks import (
    FEATURE_DIM,
    HurdleTransitActorV5,
    TransitQEnsembleV5,
    TransitValueV5,
    canonical_features_v5,
)


def _states(rows: int = 16) -> torch.Tensor:
    states = torch.rand(rows, CANONICAL_STATE_DIM)
    states[:, 0] = torch.arange(rows) % 12
    return states


def test_v5_networks_consume_broadcast_welfare_state() -> None:
    states = _states(7)
    actions = torch.linspace(-1.0, 1.0, 7)[:, None]
    features = canonical_features_v5(states)
    actor = HurdleTransitActorV5((32, 32))
    critic = TransitQEnsembleV5(3, (32, 32))
    value = TransitValueV5((32, 32))

    assert features.shape == (7, FEATURE_DIM)
    assert actor(states, deterministic=True)[0].shape == (7, 1)
    assert critic(states, actions).shape == (3, 7)
    assert value(states).shape == (7,)


def test_v5_hurdle_measure_preserves_exact_zero_atom() -> None:
    actor = HurdleTransitActorV5((32, 32))
    states = _states(2)
    actions = torch.tensor([[-1.0], [torch.nextafter(torch.tensor(-1.0), torch.tensor(0.0))]])
    likelihood = actor.action_log_prob(states, actions)
    assert torch.isfinite(likelihood).all()
    assert likelihood[0] != likelihood[1]
