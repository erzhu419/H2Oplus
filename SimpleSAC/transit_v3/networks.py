"""Neural networks for the canonical transit SMDP."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from bus_h2o.mdp_contract import ACTION_DIM, CANONICAL_STATE_DIM


N_LINES = 12
CONTINUOUS_STATE_DIM = CANONICAL_STATE_DIM - 1
FEATURE_DIM = N_LINES + CONTINUOUS_STATE_DIM
BEHAVIOR_ACTION_CLIP = 0.99


def canonical_features(states: torch.Tensor) -> torch.Tensor:
    """Replace the categorical line index with a one-hot representation."""

    if states.ndim != 2 or states.shape[-1] != CANONICAL_STATE_DIM:
        raise ValueError(
            f"states must have shape (N, {CANONICAL_STATE_DIM}); got {tuple(states.shape)}"
        )
    line_ids = torch.round(states[:, 0]).long().clamp(0, N_LINES - 1)
    line_features = F.one_hot(line_ids, num_classes=N_LINES).to(states.dtype)
    return torch.cat((line_features, states[:, 1:]), dim=-1)


def _init_linear(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if activation not in {"relu", "silu"}:
            raise ValueError("activation must be 'relu' or 'silu'")
        activation_type = nn.ReLU if activation == "relu" else nn.SiLU
        layers: list[nn.Module] = []
        width = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(width, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_type())
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)
        self.apply(_init_linear)
        _init_linear(self.network[-1], gain=0.01)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TransitActor(nn.Module):
    """Squashed Gaussian holding policy."""

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
            2 * ACTION_DIM,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )
        output = self.network.network[-1]
        with torch.no_grad():
            output.bias.copy_(torch.tensor([0.0, -0.5]))

    def distribution_parameters(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(canonical_features(states)).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        return mean, log_std

    def forward(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(states)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh) - torch.log(
            1.0 - actions.square() + 1e-6
        )
        return actions, log_prob.sum(dim=-1)

    def action_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
        mean, log_std = self.distribution_parameters(states)
        # Historical controls contain exact bounds, while a tanh-normal has no
        # finite inverse there. The 0.99 convention matches the WSRL reference.
        clipped = actions.clamp(-BEHAVIOR_ACTION_CLIP, BEHAVIOR_ACTION_CLIP)
        pre_tanh = torch.atanh(clipped)
        distribution = Normal(mean, log_std.exp())
        log_prob = distribution.log_prob(pre_tanh) - torch.log(
            1.0 - clipped.square() + 1e-6
        )
        return log_prob.sum(dim=-1)


class TransitValue(nn.Module):
    """State-value network used by IQL and the H2O+ mixed backup."""

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
        return self.network(canonical_features(states)).squeeze(-1)


class EnsembleLinear(nn.Module):
    """Independent linear layers evaluated in one batched operation."""

    def __init__(self, ensemble_size: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, output_dim))
        self.reset_parameters()

    def reset_parameters(self, gain: float = math.sqrt(2.0)) -> None:
        for member_weight in self.weight:
            nn.init.orthogonal_(member_weight, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            output = torch.einsum("bi,eio->ebo", values, self.weight)
        elif values.ndim == 3 and values.shape[0] == self.ensemble_size:
            output = torch.einsum("ebi,eio->ebo", values, self.weight)
        else:
            raise ValueError("ensemble input must have shape (B,D) or (E,B,D)")
        return output + self.bias[:, None, :]

    def forward_subset(
        self, values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("ensemble indices must be a one-dimensional long tensor")
        weight = self.weight.index_select(0, indices)
        bias = self.bias.index_select(0, indices)
        if values.ndim == 2:
            output = torch.einsum("bi,eio->ebo", values, weight)
        elif values.ndim == 3 and values.shape[0] == len(indices):
            output = torch.einsum("ebi,eio->ebo", values, weight)
        else:
            raise ValueError(
                "subset ensemble input must have shape (B,D) or (S,B,D)"
            )
        return output + bias[:, None, :]


class EnsembleLayerNorm(nn.Module):
    def __init__(self, ensemble_size: int, width: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, width))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, width))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance, mean = torch.var_mean(
            values, dim=-1, keepdim=True, unbiased=False
        )
        normalized = (values - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight[:, None, :] + self.bias[:, None, :]

    def forward_subset(
        self, values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[0] != len(indices):
            raise ValueError("subset normalization expects shape (S,B,D)")
        variance, mean = torch.var_mean(
            values, dim=-1, keepdim=True, unbiased=False
        )
        normalized = (values - mean) * torch.rsqrt(variance + self.epsilon)
        weight = self.weight.index_select(0, indices)
        bias = self.bias.index_select(0, indices)
        return normalized * weight[:, None, :] + bias[:, None, :]


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if activation not in {"relu", "silu"}:
            raise ValueError("activation must be 'relu' or 'silu'")
        self.activation = activation
        widths = (input_dim, *hidden_dims, output_dim)
        self.layers = nn.ModuleList(
            EnsembleLinear(ensemble_size, source, target)
            for source, target in zip(widths[:-1], widths[1:])
        )
        self.norms = nn.ModuleList(
            (
                EnsembleLayerNorm(ensemble_size, width)
                if layer_norm
                else nn.Identity()
            )
            for width in hidden_dims
        )
        self.layers[-1].reset_parameters(gain=0.01)

    def forward(
        self,
        values: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if indices is not None:
            return self.forward_subset(values, indices)
        for layer, normalization in zip(self.layers[:-1], self.norms):
            values = normalization(layer(values))
            values = F.relu(values) if self.activation == "relu" else F.silu(values)
        return self.layers[-1](values)

    def forward_subset(
        self, values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if indices.ndim != 1 or indices.dtype != torch.long or not len(indices):
            raise ValueError("ensemble subset must contain long indices")
        for layer, normalization in zip(self.layers[:-1], self.norms):
            values = layer.forward_subset(values, indices)
            if isinstance(normalization, EnsembleLayerNorm):
                values = normalization.forward_subset(values, indices)
            else:
                values = normalization(values)
            values = F.relu(values) if self.activation == "relu" else F.silu(values)
        return self.layers[-1].forward_subset(values, indices)


class TransitQEnsemble(nn.Module):
    """Fully independent critics; no shared trunk that could collapse diversity."""

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
        features = canonical_features(states)
        values = torch.cat((features, actions), dim=-1)
        return self.network(values).squeeze(-1)

    def forward_subset(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        features = canonical_features(states)
        values = torch.cat((features, actions), dim=-1)
        return self.network(values, indices).squeeze(-1)
