"""Factored conditional dynamics-ratio estimator for simulator correction."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from bus_h2o.mdp_contract import ACTION_DIM

from .networks import FEATURE_DIM, MLP, canonical_features


def conditional_log_ratio(
    joint_domain_logits: torch.Tensor,
    state_action_domain_logits: torch.Tensor,
) -> torch.Tensor:
    """Return log p_real(s',dt|s,a) / p_sim(s',dt|s,a).

    Both domain classifiers are trained with equal class priors.  Their
    logits are log density ratios for the joint and marginal distributions;
    subtracting them cancels the state-action occupancy ratio.
    """

    if joint_domain_logits.shape != state_action_domain_logits.shape:
        raise ValueError("joint and state-action logits must have matching shapes")
    return joint_domain_logits - state_action_domain_logits


def _binary_auc(real_scores: torch.Tensor, sim_scores: torch.Tensor) -> torch.Tensor:
    comparisons = real_scores[:, None] - sim_scores[None, :]
    return (comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean()


class FactoredDynamicsRatio(nn.Module):
    """Estimate a target-to-simulator SMDP transition-kernel ratio."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        learning_rate: float = 1e-4,
        min_weight: float = 0.05,
        max_weight: float = 20.0,
        feature_dim: int = FEATURE_DIM,
        action_dim: int = ACTION_DIM,
        feature_fn: Callable[[torch.Tensor], torch.Tensor] = canonical_features,
        fused_adam: bool = False,
        transition_feature_key: str = "discounts",
        transition_feature_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < min_weight <= 1.0 <= max_weight:
            raise ValueError("ratio clipping must satisfy 0 < min <= 1 <= max")
        if feature_dim <= 0 or action_dim <= 0:
            raise ValueError("feature_dim and action_dim must be positive")
        if not transition_feature_key:
            raise ValueError("transition feature key cannot be empty")
        if not math.isfinite(transition_feature_scale) or transition_feature_scale <= 0:
            raise ValueError("transition feature scale must be finite and positive")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.feature_fn = feature_fn
        self.transition_feature_key = str(transition_feature_key)
        self.transition_feature_scale = float(transition_feature_scale)
        self.state_action = MLP(self.feature_dim + self.action_dim, 1, hidden_dims)
        self.joint = MLP(
            2 * self.feature_dim + self.action_dim + 1, 1, hidden_dims
        )
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._min_log_weight = math.log(self.min_weight)
        self._max_log_weight = math.log(self.max_weight)
        cuda_optimizer = bool(fused_adam)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            fused=cuda_optimizer,
            capturable=cuda_optimizer,
        )

    @staticmethod
    def _classifier_metrics(
        real_logits: torch.Tensor,
        sim_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real_probability = real_logits.sigmoid()
        sim_probability = sim_logits.sigmoid()
        accuracy = 0.5 * (
            (real_probability >= 0.5).float().mean()
            + (sim_probability < 0.5).float().mean()
        )
        brier = 0.5 * (
            (real_probability - 1.0).square().mean()
            + sim_probability.square().mean()
        )
        return {
            "auc": _binary_auc(real_logits, sim_logits),
            "accuracy": accuracy,
            "brier": brier,
            "real_logit_mean": real_logits.mean(),
            "sim_logit_mean": sim_logits.mean(),
        }

    def _state_action_features(
        self,
        batch: dict[str, torch.Tensor],
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = (
            self.feature_fn(batch["observations"])
            if observation_features is None
            else observation_features
        )
        return torch.cat(
            (features, batch["actions"]), dim=-1
        )

    def _joint_features(
        self,
        batch: dict[str, torch.Tensor],
        observation_features: torch.Tensor | None = None,
        next_observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = (
            self.feature_fn(batch["observations"])
            if observation_features is None
            else observation_features
        )
        try:
            transition_feature = batch[self.transition_feature_key]
        except KeyError as error:
            raise KeyError(
                f"ratio batch lacks {self.transition_feature_key!r}"
            ) from error
        transition_feature = transition_feature.reshape(-1, 1) / (
            self.transition_feature_scale
        )
        return torch.cat(
            (
                features,
                batch["actions"],
                (
                    self.feature_fn(batch["next_observations"])
                    if next_observation_features is None
                    else next_observation_features
                ),
                transition_feature,
            ),
            dim=-1,
        )

    def logits(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch["observations"].shape[0]
        combined_features = self.feature_fn(
            torch.cat((batch["observations"], batch["next_observations"]), dim=0)
        )
        observation_features, next_observation_features = combined_features.split(
            batch_size, dim=0
        )
        state_action_logits = self.state_action(
            self._state_action_features(batch, observation_features)
        ).squeeze(-1)
        joint_logits = self.joint(
            self._joint_features(
                batch, observation_features, next_observation_features
            )
        ).squeeze(-1)
        return joint_logits, state_action_logits

    def update(
        self,
        real_batch: dict[str, torch.Tensor],
        sim_batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> dict[str, torch.Tensor]:
        real_joint, real_sa = self.logits(real_batch)
        sim_joint, sim_sa = self.logits(sim_batch)
        real_labels = torch.ones_like(real_joint)
        sim_labels = torch.zeros_like(sim_joint)
        joint_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_joint, real_labels)
            + F.binary_cross_entropy_with_logits(sim_joint, sim_labels)
        )
        state_action_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_sa, real_labels)
            + F.binary_cross_entropy_with_logits(sim_sa, sim_labels)
        )
        loss = joint_loss + state_action_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()

        if not collect_metrics:
            return {}
        with torch.no_grad():
            joint_metrics = self._classifier_metrics(real_joint, sim_joint)
            state_action_metrics = self._classifier_metrics(real_sa, sim_sa)
        metrics = {
            "ratio/loss": loss.detach(),
            "ratio/joint_loss": joint_loss.detach(),
            "ratio/state_action_loss": state_action_loss.detach(),
        }
        metrics.update(
            {f"ratio/joint_{key}": value for key, value in joint_metrics.items()}
        )
        metrics.update(
            {
                f"ratio/state_action_{key}": value
                for key, value in state_action_metrics.items()
            }
        )
        return metrics

    @torch.no_grad()
    def weights(
        self,
        sim_batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        joint_logits, state_action_logits = self.logits(sim_batch)
        log_weights = conditional_log_ratio(joint_logits, state_action_logits)
        clipped_log_weights = log_weights.clamp(
            self._min_log_weight, self._max_log_weight
        )
        weights = clipped_log_weights.exp()
        if not collect_metrics:
            return weights, {}
        finite_log_weights = log_weights.clamp(-30.0, 30.0)
        raw_weights = finite_log_weights.exp()
        raw_ess = raw_weights.sum().square() / raw_weights.square().sum().clamp_min(1e-8)
        ess = weights.sum().square() / weights.square().sum().clamp_min(1e-8)
        low_clip_fraction = (log_weights < self._min_log_weight).float().mean()
        high_clip_fraction = (log_weights > self._max_log_weight).float().mean()
        diagnostics = {
            "ratio/weight_min": weights.min(),
            "ratio/weight_max": weights.max(),
            "ratio/weight_mean": weights.mean(),
            "ratio/ess_fraction": ess / max(weights.numel(), 1),
            "ratio/raw_ess_fraction": raw_ess / max(weights.numel(), 1),
            "ratio/clip_fraction": low_clip_fraction + high_clip_fraction,
            "ratio/low_clip_fraction": low_clip_fraction,
            "ratio/high_clip_fraction": high_clip_fraction,
            "ratio/log_weight_mean": log_weights.mean(),
            "ratio/log_weight_std": log_weights.std(unbiased=False),
        }
        return weights, diagnostics

    @torch.no_grad()
    def evaluate(
        self,
        real_batch: dict[str, torch.Tensor],
        sim_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate classifiers and weights on a fresh diagnostic minibatch."""

        real_joint, real_sa = self.logits(real_batch)
        sim_joint, sim_sa = self.logits(sim_batch)
        joint_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_joint, torch.ones_like(real_joint))
            + F.binary_cross_entropy_with_logits(sim_joint, torch.zeros_like(sim_joint))
        )
        state_action_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_sa, torch.ones_like(real_sa))
            + F.binary_cross_entropy_with_logits(sim_sa, torch.zeros_like(sim_sa))
        )
        metrics: dict[str, torch.Tensor] = {
            "ratio/validation_loss": joint_loss + state_action_loss,
            "ratio/validation_joint_loss": joint_loss,
            "ratio/validation_state_action_loss": state_action_loss,
        }
        for name, real_logits, sim_logits in (
            ("joint", real_joint, sim_joint),
            ("state_action", real_sa, sim_sa),
        ):
            metrics.update(
                {
                    f"ratio/validation_{name}_{key}": value
                    for key, value in self._classifier_metrics(
                        real_logits, sim_logits
                    ).items()
                }
            )
        _, weight_metrics = self.weights(sim_batch, collect_metrics=True)
        metrics.update(
            {
                key.replace("ratio/", "ratio/validation_", 1): value
                for key, value in weight_metrics.items()
            }
        )
        return metrics
