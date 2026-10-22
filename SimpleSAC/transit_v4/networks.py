"""Hybrid zero-or-positive holding networks for transit-smdp-v4."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from bus_h2o.mdp_contract_v4 import ACTION_DIM, CANONICAL_STATE_DIM
from SimpleSAC.transit_v3.networks import EnsembleMLP, MLP


N_LINES = 12
CONTINUOUS_STATE_DIM = CANONICAL_STATE_DIM - 1
FEATURE_DIM = N_LINES + CONTINUOUS_STATE_DIM
ZERO_ACTION = -1.0


def canonical_features_v4(states: torch.Tensor) -> torch.Tensor:
    """Replace the directional line index with a one-hot representation."""

    if states.ndim != 2 or states.shape[-1] != CANONICAL_STATE_DIM:
        raise ValueError(
            f"states must have shape (N, {CANONICAL_STATE_DIM}); got {tuple(states.shape)}"
        )
    line_ids = torch.round(states[:, 0]).long().clamp(0, N_LINES - 1)
    line_features = F.one_hot(line_ids, num_classes=N_LINES).to(states.dtype)
    return torch.cat((line_features, states[:, 1:]), dim=-1)


class HurdleTransitActor(nn.Module):
    """A point mass at zero hold plus a logistic-normal positive branch.

    The physical zero hold is represented by policy action -1 exactly.  Positive
    holds use a continuous density on (-1, 1), so likelihoods are defined with
    respect to the mixed counting-plus-Lebesgue base measure.
    """

    is_hybrid = True

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
            3,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )
        output = self.network.network[-1]
        with torch.no_grad():
            # Initial p(hold)=0.35 and a modest positive hold conditional mode.
            output.bias.copy_(torch.tensor([-0.62, -1.4, -0.5]))

    def distribution_parameters(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logit, positive_mean, positive_log_std = self.network(
            canonical_features_v4(states)
        ).unbind(dim=-1)
        return gate_logit, positive_mean, positive_log_std.clamp(-5.0, 2.0)

    @staticmethod
    def _positive_transform(pre_sigmoid: torch.Tensor) -> torch.Tensor:
        # x in (0,1) is the positive hold fraction; action 2x-1 is in (-1,1).
        action = 2.0 * torch.sigmoid(pre_sigmoid) - 1.0
        epsilon = torch.finfo(action.dtype).eps
        return action.clamp(min=-1.0 + epsilon, max=1.0 - epsilon)

    @staticmethod
    def _positive_log_det(pre_sigmoid: torch.Tensor) -> torch.Tensor:
        # log da/dz = log(2) + log(sigmoid(z)) + log(sigmoid(-z)).
        return (
            math.log(2.0)
            + F.logsigmoid(pre_sigmoid)
            + F.logsigmoid(-pre_sigmoid)
        )

    def positive_action_and_log_prob(
        self, states: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self.distribution_parameters(states)
        distribution = Normal(mean, log_std.exp())
        pre_sigmoid = mean if deterministic else distribution.rsample()
        action = self._positive_transform(pre_sigmoid)
        conditional_log_prob = distribution.log_prob(pre_sigmoid) - self._positive_log_det(
            pre_sigmoid
        )
        return action[:, None], conditional_log_prob

    def soft_action_components(
        self, states: torch.Tensor, *, deterministic_positive: bool = False
    ) -> dict[str, torch.Tensor]:
        """Return differentiable mixture probabilities and one positive sample."""

        gate_logit, mean, log_std = self.distribution_parameters(states)
        return self.soft_action_components_from_parameters(
            gate_logit,
            mean,
            log_std,
            deterministic_positive=deterministic_positive,
        )

    def soft_action_components_from_parameters(
        self,
        gate_logit: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        deterministic_positive: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Construct hybrid samples from precomputed distribution parameters."""

        if not (gate_logit.shape == mean.shape == log_std.shape):
            raise ValueError("hybrid distribution parameter shapes must match")
        log_hold = F.logsigmoid(gate_logit)
        log_zero = F.logsigmoid(-gate_logit)
        hold_probability = log_hold.exp()
        zero_probability = log_zero.exp()
        distribution = Normal(mean, log_std.exp())
        pre_sigmoid = mean if deterministic_positive else distribution.rsample()
        positive_action = self._positive_transform(pre_sigmoid)[:, None]
        positive_conditional_log_prob = (
            distribution.log_prob(pre_sigmoid)
            - self._positive_log_det(pre_sigmoid)
        )
        return {
            "zero_probability": zero_probability,
            "hold_probability": hold_probability,
            "zero_action": torch.full_like(positive_action, ZERO_ACTION),
            "positive_action": positive_action,
            "zero_log_prob": log_zero,
            "positive_conditional_log_prob": positive_conditional_log_prob,
            "positive_log_prob": log_hold + positive_conditional_log_prob,
        }

    def repeated_soft_action_components(
        self,
        states: torch.Tensor,
        repeats: int,
        *,
        deterministic_positive: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Draw repeated actions without re-encoding identical states."""

        if repeats <= 0:
            raise ValueError("action repeat count must be positive")
        parameters = self.distribution_parameters(states)
        expanded = tuple(
            value[:, None].expand(-1, repeats).reshape(-1)
            for value in parameters
        )
        return self.soft_action_components_from_parameters(
            *expanded,
            deterministic_positive=deterministic_positive,
        )

    def repeated_actions_and_log_prob(
        self,
        states: torch.Tensor,
        repeats: int,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        components = self.repeated_soft_action_components(
            states,
            repeats,
            deterministic_positive=deterministic,
        )
        if deterministic:
            choose_hold = components["hold_probability"] >= 0.5
        else:
            choose_hold = torch.bernoulli(
                components["hold_probability"]
            ).bool()
        actions = torch.where(
            choose_hold[:, None],
            components["positive_action"],
            components["zero_action"],
        )
        log_prob = torch.where(
            choose_hold,
            components["positive_log_prob"],
            components["zero_log_prob"],
        )
        return actions, log_prob

    def forward(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        components = self.soft_action_components(
            states, deterministic_positive=deterministic
        )
        if deterministic:
            choose_hold = components["hold_probability"] >= 0.5
        else:
            choose_hold = torch.bernoulli(components["hold_probability"]).bool()
        actions = torch.where(
            choose_hold[:, None],
            components["positive_action"],
            components["zero_action"],
        )
        log_prob = torch.where(
            choose_hold,
            components["positive_log_prob"],
            components["zero_log_prob"],
        )
        return actions, log_prob

    def action_log_prob(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
        gate_logit, mean, log_std = self.distribution_parameters(states)
        log_hold = F.logsigmoid(gate_logit)
        log_zero = F.logsigmoid(-gate_logit)
        scalar_action = actions[:, 0]
        is_zero = scalar_action == ZERO_ACTION
        fraction_epsilon = torch.finfo(scalar_action.dtype).eps * 0.5
        positive_fraction = ((scalar_action + 1.0) * 0.5).clamp(
            fraction_epsilon, 1.0 - fraction_epsilon
        )
        pre_sigmoid = torch.logit(positive_fraction)
        distribution = Normal(mean, log_std.exp())
        positive_log_prob = (
            log_hold
            + distribution.log_prob(pre_sigmoid)
            - self._positive_log_det(pre_sigmoid)
        )
        return torch.where(is_zero, log_zero, positive_log_prob)


class TransitValueV4(nn.Module):
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
        return self.network(canonical_features_v4(states)).squeeze(-1)


class TransitQEnsembleV4(nn.Module):
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
        features = canonical_features_v4(states)
        values = torch.cat((features, actions), dim=-1)
        return self.network(values).squeeze(-1)


__all__ = [
    "FEATURE_DIM",
    "HurdleTransitActor",
    "TransitQEnsembleV4",
    "TransitValueV4",
    "canonical_features_v4",
]
