"""Structured hurdle-action networks for transit-smdp-v7."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from bus_h2o.mdp_contract_v7 import (
    ACTION_DIM,
    CANONICAL_PREFIX_NAMES,
    CANONICAL_SLOT_WIDTH,
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_INDEX,
    LOCAL_CANONICAL_STATE_DIM as EVENT_STATE_DIM,
    MAX_SIMULTANEOUS_DECISIONS,
)
from SimpleSAC.transit_v3.domain_ratio import FactoredDynamicsRatio
from SimpleSAC.transit_v3.networks import EnsembleMLP, MLP
from SimpleSAC.transit_v4.networks import HurdleTransitActor


N_LINES = 12
SLOT_EMBED_DIM = 128
FEATURE_DIM = 256
SLOT_MODEL_INPUT_DIM = N_LINES + (EVENT_STATE_DIM - 1) + 4
N_POOLED_VIEWS = 5


def structured_slot_features_v7(states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode canonical slots and make autoregressive status explicit."""

    if states.ndim != 2 or states.shape[-1] != CANONICAL_STATE_DIM:
        raise ValueError(
            f"states must have shape (N, {CANONICAL_STATE_DIM}); "
            f"got {tuple(states.shape)}"
        )
    # Replay and sampler boundaries validate values; CUDA-side assertions here
    # would synchronize every actor, critic, value, and ratio forward pass.
    batch_size = states.shape[0]
    prefix_width = len(CANONICAL_PREFIX_NAMES)
    slots = states[:, prefix_width:].reshape(
        batch_size, MAX_SIMULTANEOUS_DECISIONS, CANONICAL_SLOT_WIDTH
    )
    active = slots[:, :, 0] > 0.5
    current_index = torch.round(
        states[:, CANONICAL_STATE_INDEX["current_decision_index_scaled"]]
        * max(MAX_SIMULTANEOUS_DECISIONS - 1, 1)
    ).long().clamp(0, MAX_SIMULTANEOUS_DECISIONS - 1)
    slot_index = torch.arange(
        MAX_SIMULTANEOUS_DECISIONS, device=states.device
    )[None, :]
    selected = active & (slot_index < current_index[:, None])
    current = active & (slot_index == current_index[:, None])
    future = active & (slot_index > current_index[:, None])
    event = slots[:, :, 1 : 1 + EVENT_STATE_DIM]
    line_ids = torch.round(event[:, :, 0]).long().clamp(0, N_LINES - 1)
    line_one_hot = F.one_hot(line_ids, num_classes=N_LINES).to(states.dtype)
    prior_action = slots[:, :, -1:]
    relative_position = (
        slot_index.to(states.dtype) - current_index[:, None].to(states.dtype)
    ) / max(MAX_SIMULTANEOUS_DECISIONS - 1, 1)
    local = torch.cat(
        (
            line_one_hot,
            event[:, :, 1:],
            prior_action,
            selected[:, :, None].to(states.dtype),
            current[:, :, None].to(states.dtype),
            relative_position[:, :, None],
        ),
        dim=-1,
    )
    if local.shape[-1] != SLOT_MODEL_INPUT_DIM:
        raise RuntimeError("v7 structured slot feature width drifted")
    return {
        "prefix": states[:, :prefix_width],
        "local": local,
        "active": active,
        "selected": selected,
        "current": current,
        "future": future,
        "current_index": current_index,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, :, None].to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    minimum = torch.finfo(values.dtype).min
    pooled = values.masked_fill(~mask[:, :, None], minimum).max(dim=1).values
    return torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


class TransitStateEncoderV7(nn.Module):
    """Encode current, selected, and future events with shared slot weights."""

    def __init__(
        self,
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.slot_encoder = MLP(
            SLOT_MODEL_INPUT_DIM,
            SLOT_EMBED_DIM,
            (SLOT_EMBED_DIM,),
            layer_norm=layer_norm,
            activation=activation,
        )
        fusion_input_dim = (
            len(CANONICAL_PREFIX_NAMES) + N_POOLED_VIEWS * SLOT_EMBED_DIM
        )
        self.fusion = MLP(
            fusion_input_dim,
            FEATURE_DIM,
            (FEATURE_DIM,),
            layer_norm=layer_norm,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        decoded = structured_slot_features_v7(states)
        local = decoded["local"]
        embedded = self.slot_encoder(local.reshape(-1, local.shape[-1])).reshape(
            states.shape[0], MAX_SIMULTANEOUS_DECISIONS, SLOT_EMBED_DIM
        )
        row_index = torch.arange(states.shape[0], device=states.device)
        current_embedding = embedded[row_index, decoded["current_index"]]
        pooled = torch.cat(
            (
                current_embedding,
                _masked_mean(embedded, decoded["active"]),
                _masked_max(embedded, decoded["active"]),
                _masked_mean(embedded, decoded["selected"]),
                _masked_mean(embedded, decoded["future"]),
            ),
            dim=-1,
        )
        return self.fusion(torch.cat((decoded["prefix"], pooled), dim=-1))


class HurdleTransitActorV7(HurdleTransitActor):
    def __init__(
        self,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        nn.Module.__init__(self)
        self.encoder = TransitStateEncoderV7(
            layer_norm=layer_norm, activation=activation
        )
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
        return self.distribution_parameters_from_features(
            self.encode_states(states)
        )

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def distribution_parameters_from_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logit, positive_mean, positive_log_std = self.network(
            features
        ).unbind(dim=-1)
        return gate_logit, positive_mean, positive_log_std.clamp(-5.0, 2.0)


class TransitValueV7(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        layer_norm: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.encoder = TransitStateEncoderV7(
            layer_norm=layer_norm, activation=activation
        )
        self.network = MLP(
            FEATURE_DIM,
            1,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoder(states)).squeeze(-1)


class TransitQEnsembleV7(nn.Module):
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
        self.encoder = TransitStateEncoderV7(
            layer_norm=layer_norm, activation=activation
        )
        self.network = EnsembleMLP(
            ensemble_size,
            FEATURE_DIM + ACTION_DIM,
            1,
            hidden_dims,
            layer_norm=layer_norm,
            activation=activation,
        )
        self._ensemble_size = int(ensemble_size)

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward_features(self.encode_states(states), actions)

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def forward_features(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(
                f"critic features must have shape (N, {FEATURE_DIM})"
            )
        if actions.ndim != 2 or actions.shape != (features.shape[0], ACTION_DIM):
            raise ValueError(
                f"critic actions must have shape (N, {ACTION_DIM})"
            )
        return self.network(torch.cat((features, actions), dim=-1)).squeeze(-1)

    def forward_subset(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features_subset(
            self.encode_states(states), actions, indices
        )

    def forward_features_subset(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(
                f"critic features must have shape (N, {FEATURE_DIM})"
            )
        if actions.ndim != 2 or actions.shape != (features.shape[0], ACTION_DIM):
            raise ValueError(
                f"critic actions must have shape (N, {ACTION_DIM})"
            )
        return self.network(
            torch.cat((features, actions), dim=-1), indices
        ).squeeze(-1)


class TransitDynamicsRatioV7(FactoredDynamicsRatio):
    """Use the same structured state representation in the ratio estimator."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        learning_rate: float = 1e-4,
        min_weight: float = 0.05,
        max_weight: float = 20.0,
        fused_adam: bool = False,
        horizon_seconds: float = 24_000.0,
    ) -> None:
        encoder = TransitStateEncoderV7()
        super().__init__(
            hidden_dims,
            learning_rate=learning_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            feature_dim=FEATURE_DIM,
            action_dim=ACTION_DIM,
            feature_fn=encoder,
            fused_adam=fused_adam,
            transition_feature_key="delta_t_seconds",
            transition_feature_scale=horizon_seconds,
        )


__all__ = [
    "FEATURE_DIM",
    "HurdleTransitActorV7",
    "SLOT_EMBED_DIM",
    "SLOT_MODEL_INPUT_DIM",
    "TransitDynamicsRatioV7",
    "TransitQEnsembleV7",
    "TransitStateEncoderV7",
    "TransitValueV7",
    "structured_slot_features_v7",
]
