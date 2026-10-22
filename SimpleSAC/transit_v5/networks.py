"""Hurdle-action networks bound to the transit-smdp-v5 state contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from bus_h2o.mdp_contract_v5 import ACTION_DIM, CANONICAL_STATE_DIM
from SimpleSAC.transit_v3.networks import EnsembleMLP, MLP
from SimpleSAC.transit_v4.networks import HurdleTransitActor


N_LINES = 12
CONTINUOUS_STATE_DIM = CANONICAL_STATE_DIM - 1
FEATURE_DIM = N_LINES + CONTINUOUS_STATE_DIM


def canonical_features_v5(states: torch.Tensor) -> torch.Tensor:
    """Replace the directional line index with a one-hot representation."""

    if states.ndim != 2 or states.shape[-1] != CANONICAL_STATE_DIM:
        raise ValueError(
            f"states must have shape (N, {CANONICAL_STATE_DIM}); "
            f"got {tuple(states.shape)}"
        )
    line_ids = torch.round(states[:, 0]).long().clamp(0, N_LINES - 1)
    line_features = F.one_hot(line_ids, num_classes=N_LINES).to(states.dtype)
    return torch.cat((line_features, states[:, 1:]), dim=-1)


class HurdleTransitActorV5(HurdleTransitActor):
    def __init__(
        self,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        nn.Module.__init__(self)
        self.network = MLP(
            FEATURE_DIM,
            3,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )
        output = self.network.network[-1]
        with torch.no_grad():
            output.bias.copy_(torch.tensor([-0.62, -1.4, -0.5]))

    def distribution_parameters(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logit, positive_mean, positive_log_std = self.network(
            canonical_features_v5(states)
        ).unbind(dim=-1)
        return gate_logit, positive_mean, positive_log_std.clamp(-5.0, 2.0)


class TransitValueV5(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.network = MLP(
            FEATURE_DIM,
            1,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(canonical_features_v5(states)).squeeze(-1)


class TransitQEnsembleV5(nn.Module):
    def __init__(
        self,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two")
        self.network = EnsembleMLP(
            ensemble_size,
            FEATURE_DIM + ACTION_DIM,
            1,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )
        self._ensemble_size = ensemble_size

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = canonical_features_v5(states)
        values = torch.cat((features, actions), dim=-1)
        return self.network(values).squeeze(-1)


__all__ = [
    "FEATURE_DIM",
    "HurdleTransitActorV5",
    "TransitQEnsembleV5",
    "TransitValueV5",
    "canonical_features_v5",
]
