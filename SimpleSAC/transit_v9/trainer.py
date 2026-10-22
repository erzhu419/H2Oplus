"""Development-only H2O+ interventions for the v9 few-shot pilot.

The v7 network, replay, SMDP and checkpoint contracts remain unchanged.  This
module isolates two online-phase interventions so the registered 2x2 study can
attribute any improvement to ratio reliability, adaptation, or their
combination.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from SimpleSAC.transit_v3.domain_ratio import conditional_log_ratio
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig
from SimpleSAC.transit_v7.networks import (
    HurdleTransitActorV7,
    TransitDynamicsRatioV7,
    TransitQEnsembleV7,
    TransitValueV7,
)
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


RATIO_GATE_SCHEMA = "transit-h2oplus-v9-ratio-gate-v1"
V9_VARIANT_SCHEMA = "transit-h2oplus-v9-adaptation-variant-v1"


def offline_fraction_for_online_events(online_events: int) -> float:
    """Return the preregistered offline replay share for adaptive variants."""

    events = max(int(online_events), 0)
    knots = ((1_000, 0.50), (10_000, 0.25), (25_000, 0.10))
    if events <= knots[0][0]:
        return knots[0][1]
    for (left_events, left_value), (right_events, right_value) in zip(
        knots, knots[1:]
    ):
        if events <= right_events:
            progress = (events - left_events) / (right_events - left_events)
            return left_value + progress * (right_value - left_value)
    return knots[-1][1]


class ReliabilityGatedDynamicsRatioV9(TransitDynamicsRatioV7):
    """Apply ratio weights only after trajectory-disjoint validation passes.

    The gate uses two-success/two-failure hysteresis.  When open, the raw log
    ratio is tempered to a minimum effective sample size and normalized under
    bounded weights.  When closed, the simulator TD loss receives unit weights
    while both classifiers continue to train.
    """

    OPEN_AFTER_PASSES = 2
    CLOSE_AFTER_FAILURES = 2
    MIN_JOINT_AUC = 0.60
    MAX_JOINT_BRIER = 0.24
    MAX_STATE_ACTION_BRIER = 0.26
    MIN_RAW_ESS_FRACTION = 0.25
    MAX_HIGH_CLIP_FRACTION = 0.25
    MAX_TOTAL_CLIP_FRACTION = 0.50
    TARGET_ESS_FRACTION = 0.50
    APPLIED_MIN_WEIGHT = 0.10
    APPLIED_MAX_WEIGHT = 5.0
    _STATE_KEYS = {
        "v9_gate_open",
        "v9_pass_streak",
        "v9_failure_streak",
        "v9_validation_count",
        "v9_tempering_exponent",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("v9_gate_open", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("v9_pass_streak", torch.zeros((), dtype=torch.long))
        self.register_buffer("v9_failure_streak", torch.zeros((), dtype=torch.long))
        self.register_buffer("v9_validation_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("v9_tempering_exponent", torch.zeros(()))

    def _reset_gate(self) -> None:
        self.v9_gate_open.zero_()
        self.v9_pass_streak.zero_()
        self.v9_failure_streak.zero_()
        self.v9_validation_count.zero_()
        self.v9_tempering_exponent.zero_()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Accept a frozen v7 initializer once, but require full v9 resumes."""

        legacy = self._STATE_KEYS.isdisjoint(state_dict)
        if not legacy:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        unexpected = set(result.unexpected_keys)
        missing = set(result.missing_keys) - self._STATE_KEYS
        if strict and (missing or unexpected):
            raise RuntimeError(
                "legacy ratio state has non-v9 incompatibilities: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self._reset_gate()
        return result

    @staticmethod
    def _ess_fraction(weights: torch.Tensor) -> torch.Tensor:
        count = max(weights.numel(), 1)
        ess = weights.sum().square() / weights.square().sum().clamp_min(1e-8)
        return ess / count

    def _raw_diagnostics(
        self, log_weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raw_weights = log_weights.clamp(-30.0, 30.0).exp()
        low_clip = (log_weights < self._min_log_weight).float().mean()
        high_clip = (log_weights > self._max_log_weight).float().mean()
        return {
            "raw_ess_fraction": self._ess_fraction(raw_weights),
            "low_clip_fraction": low_clip,
            "high_clip_fraction": high_clip,
            "clip_fraction": low_clip + high_clip,
            "log_weight_mean": log_weights.mean(),
            "log_weight_std": log_weights.std(unbiased=False),
        }

    @classmethod
    def _bounded_mean_one(
        cls, log_weights: torch.Tensor, exponent: float
    ) -> torch.Tensor:
        tempered = log_weights.clamp(-30.0, 30.0) * float(exponent)
        minimum = cls.APPLIED_MIN_WEIGHT
        maximum = cls.APPLIED_MAX_WEIGHT
        lower = torch.tensor(-60.0, device=tempered.device, dtype=tempered.dtype)
        upper = torch.tensor(60.0, device=tempered.device, dtype=tempered.dtype)
        for _ in range(40):
            shift = (lower + upper) * 0.5
            mean = (tempered + shift).exp().clamp(minimum, maximum).mean()
            if bool(mean < 1.0):
                lower = shift
            else:
                upper = shift
        shift = (lower + upper) * 0.5
        return (tempered + shift).exp().clamp(minimum, maximum)

    @classmethod
    def _select_tempering_exponent(cls, log_weights: torch.Tensor) -> float:
        full = cls._bounded_mean_one(log_weights, 1.0)
        if float(cls._ess_fraction(full)) >= cls.TARGET_ESS_FRACTION:
            return 1.0
        lower, upper = 0.0, 1.0
        for _ in range(30):
            middle = (lower + upper) * 0.5
            candidate = cls._bounded_mean_one(log_weights, middle)
            if float(cls._ess_fraction(candidate)) >= cls.TARGET_ESS_FRACTION:
                lower = middle
            else:
                upper = middle
        return lower

    def _applied_weights(self, log_weights: torch.Tensor) -> torch.Tensor:
        if not bool(self.v9_gate_open.item()):
            return torch.ones_like(log_weights)
        weights = self._bounded_mean_one(
            log_weights, float(self.v9_tempering_exponent.item())
        )
        # Validation fixes the log-ratio exponent.  A mean-preserving shrinkage
        # keeps a noisier training minibatch above the same ESS floor without a
        # second iterative solve: E[(1+a(w-1))^2] = 1 + a^2 Var(w).
        variance = (weights - 1.0).square().mean()
        maximum_variance = 1.0 / self.TARGET_ESS_FRACTION - 1.0
        shrinkage = torch.sqrt(
            torch.as_tensor(
                maximum_variance,
                device=weights.device,
                dtype=weights.dtype,
            )
            / variance.clamp_min(1e-12)
        ).clamp(max=1.0)
        return 1.0 + shrinkage * (weights - 1.0)

    def _gate_passes(
        self,
        *,
        joint_metrics: dict[str, torch.Tensor],
        state_action_metrics: dict[str, torch.Tensor],
        raw: dict[str, torch.Tensor],
    ) -> bool:
        return bool(
            float(joint_metrics["auc"]) >= self.MIN_JOINT_AUC
            and float(joint_metrics["brier"]) <= self.MAX_JOINT_BRIER
            and float(state_action_metrics["brier"])
            <= self.MAX_STATE_ACTION_BRIER
            and float(raw["raw_ess_fraction"])
            >= self.MIN_RAW_ESS_FRACTION
            and float(raw["high_clip_fraction"])
            <= self.MAX_HIGH_CLIP_FRACTION
            and float(raw["clip_fraction"])
            <= self.MAX_TOTAL_CLIP_FRACTION
        )

    def _update_gate(self, passed: bool, log_weights: torch.Tensor) -> None:
        self.v9_validation_count.add_(1)
        if passed:
            self.v9_pass_streak.add_(1)
            self.v9_failure_streak.zero_()
            if int(self.v9_pass_streak.item()) >= self.OPEN_AFTER_PASSES:
                self.v9_gate_open.fill_(1)
        else:
            self.v9_failure_streak.add_(1)
            self.v9_pass_streak.zero_()
            if int(self.v9_failure_streak.item()) >= self.CLOSE_AFTER_FAILURES:
                self.v9_gate_open.zero_()
        if bool(self.v9_gate_open.item()):
            self.v9_tempering_exponent.fill_(
                self._select_tempering_exponent(log_weights)
            )
        else:
            self.v9_tempering_exponent.zero_()

    @torch.no_grad()
    def weights(
        self,
        sim_batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        joint_logits, state_action_logits = self.logits(sim_batch)
        log_weights = conditional_log_ratio(joint_logits, state_action_logits)
        raw = self._raw_diagnostics(log_weights)
        weights = self._applied_weights(log_weights)
        if not collect_metrics:
            return weights, {}
        diagnostics = {
            "ratio/weight_min": weights.min(),
            "ratio/weight_max": weights.max(),
            "ratio/weight_mean": weights.mean(),
            "ratio/ess_fraction": self._ess_fraction(weights),
            "ratio/raw_ess_fraction": raw["raw_ess_fraction"],
            "ratio/clip_fraction": raw["clip_fraction"],
            "ratio/low_clip_fraction": raw["low_clip_fraction"],
            "ratio/high_clip_fraction": raw["high_clip_fraction"],
            "ratio/log_weight_mean": raw["log_weight_mean"],
            "ratio/log_weight_std": raw["log_weight_std"],
            "ratio/v9_gate_open": self.v9_gate_open.float(),
            "ratio/v9_tempering_exponent": self.v9_tempering_exponent,
            "ratio/v9_validation_count": self.v9_validation_count.float(),
        }
        return weights, diagnostics

    @torch.no_grad()
    def evaluate(
        self,
        real_batch: dict[str, torch.Tensor],
        sim_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        real_joint, real_sa = self.logits(real_batch)
        sim_joint, sim_sa = self.logits(sim_batch)
        joint_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                real_joint, torch.ones_like(real_joint)
            )
            + F.binary_cross_entropy_with_logits(
                sim_joint, torch.zeros_like(sim_joint)
            )
        )
        state_action_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_sa, torch.ones_like(real_sa))
            + F.binary_cross_entropy_with_logits(sim_sa, torch.zeros_like(sim_sa))
        )
        joint_metrics = self._classifier_metrics(real_joint, sim_joint)
        state_action_metrics = self._classifier_metrics(real_sa, sim_sa)
        log_weights = conditional_log_ratio(sim_joint, sim_sa)
        raw = self._raw_diagnostics(log_weights)
        passed = self._gate_passes(
            joint_metrics=joint_metrics,
            state_action_metrics=state_action_metrics,
            raw=raw,
        )
        self._update_gate(passed, log_weights)
        applied = self._applied_weights(log_weights)
        metrics: dict[str, torch.Tensor] = {
            "ratio/validation_loss": joint_loss + state_action_loss,
            "ratio/validation_joint_loss": joint_loss,
            "ratio/validation_state_action_loss": state_action_loss,
        }
        for name, values in (
            ("joint", joint_metrics),
            ("state_action", state_action_metrics),
        ):
            metrics.update(
                {
                    f"ratio/validation_{name}_{key}": value
                    for key, value in values.items()
                }
            )
        metrics.update(
            {
                "ratio/validation_raw_ess_fraction": raw["raw_ess_fraction"],
                "ratio/validation_low_clip_fraction": raw["low_clip_fraction"],
                "ratio/validation_high_clip_fraction": raw["high_clip_fraction"],
                "ratio/validation_clip_fraction": raw["clip_fraction"],
                "ratio/validation_log_weight_mean": raw["log_weight_mean"],
                "ratio/validation_log_weight_std": raw["log_weight_std"],
                "ratio/validation_weight_min": applied.min(),
                "ratio/validation_weight_max": applied.max(),
                "ratio/validation_weight_mean": applied.mean(),
                "ratio/validation_ess_fraction": self._ess_fraction(applied),
                "ratio/validation_v9_gate_pass": torch.tensor(
                    float(passed), device=log_weights.device
                ),
                "ratio/validation_v9_gate_open": self.v9_gate_open.float(),
                "ratio/validation_v9_pass_streak": self.v9_pass_streak.float(),
                "ratio/validation_v9_failure_streak": (
                    self.v9_failure_streak.float()
                ),
                "ratio/validation_v9_tempering_exponent": (
                    self.v9_tempering_exponent
                ),
                "ratio/validation_v9_validation_count": (
                    self.v9_validation_count.float()
                ),
            }
        )
        return metrics


class _TransitH2OTrainerV9Base(TransitH2OTrainerV7):
    CHECKPOINT_VERSION = TransitH2OTrainerV7.CHECKPOINT_VERSION
    V9_VARIANT = "abstract"
    USE_RELIABILITY_GATE = False
    USE_ADAPTIVE_ONLINE_PHASE = False
    ACTOR_BURN_IN_UPDATES = 500

    def __init__(
        self,
        replay_buffer,
        *,
        config: TransitTrainerConfig = TransitTrainerConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if self.V9_VARIANT == "abstract":
            raise TypeError("instantiate a registered v9 intervention variant")
        horizon_seconds = float(replay_buffer.contract_config["horizon_seconds"])
        ratio_class = (
            ReliabilityGatedDynamicsRatioV9
            if self.USE_RELIABILITY_GATE
            else TransitDynamicsRatioV7
        )
        TransitH2OTrainer.__init__(
            self,
            replay_buffer,
            config=config,
            device=device,
            seed=seed,
            actor_class=HurdleTransitActorV7,
            critic_class=TransitQEnsembleV7,
            value_class=TransitValueV7,
            ratio_class=ratio_class,
            ratio_kwargs={"horizon_seconds": horizon_seconds},
        )

    def _actor_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.USE_ADAPTIVE_ONLINE_PHASE and online_batch is not None:
            return online_batch["observations"]
        return TransitH2OTrainer._actor_states(offline_batch, online_batch)

    def _temperature_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.USE_ADAPTIVE_ONLINE_PHASE and online_batch is not None:
            key = (
                "observations"
                if self.config.temperature_state_source == "actor"
                else "next_observations"
            )
            return online_batch[key]
        return super()._temperature_states(offline_batch, online_batch)

    def _scheduled_online_actor_states(
        self,
        actor_states: torch.Tensor,
        temperature_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            self.USE_ADAPTIVE_ONLINE_PHASE
            and self.online_critic_updates < self.ACTOR_BURN_IN_UPDATES
        ):
            self.online_critic_updates += 1
            self._online_actor_state_batches.clear()
            self._online_temperature_state_batches.clear()
            return None
        return super()._scheduled_online_actor_states(
            actor_states, temperature_states
        )

    def update(
        self,
        *,
        online: bool,
        online_only: bool = False,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        if not online or not self.USE_ADAPTIVE_ONLINE_PHASE:
            return super().update(
                online=online,
                online_only=online_only,
                collect_metrics=collect_metrics,
            )
        offline_fraction = offline_fraction_for_online_events(
            self.replay_buffer.online_size
        )
        online_fraction = 1.0 - offline_fraction
        original_config = self.config
        self.config = replace(
            original_config,
            online_fraction=online_fraction,
            online_update_order="critic-before-actor",
        )
        try:
            metrics = super().update(
                online=True,
                online_only=online_only,
                collect_metrics=collect_metrics,
            )
        finally:
            self.config = original_config
        if collect_metrics:
            metrics.update(
                {
                    "v9/offline_replay_fraction": offline_fraction,
                    "v9/online_replay_fraction": online_fraction,
                    "v9/actor_burn_in_updates": float(
                        self.ACTOR_BURN_IN_UPDATES
                    ),
                    "v9/actor_burn_in_complete": float(
                        self.online_critic_updates > self.ACTOR_BURN_IN_UPDATES
                    ),
                }
            )
        return metrics

    @property
    def v9_intervention(self) -> dict[str, Any]:
        return {
            "schema_version": V9_VARIANT_SCHEMA,
            "variant": self.V9_VARIANT,
            "reliability_gated_ratio": self.USE_RELIABILITY_GATE,
            "adaptive_online_phase": self.USE_ADAPTIVE_ONLINE_PHASE,
            "actor_state_source": (
                "online-only"
                if self.USE_ADAPTIVE_ONLINE_PHASE
                else "registered-v7-mixed"
            ),
            "online_update_order": (
                "critic-before-actor"
                if self.USE_ADAPTIVE_ONLINE_PHASE
                else "registered-v7-actor-before-critic"
            ),
            "actor_burn_in_updates": (
                self.ACTOR_BURN_IN_UPDATES
                if self.USE_ADAPTIVE_ONLINE_PHASE
                else 0
            ),
            "offline_replay_schedule": (
                [[1_000, 0.50], [10_000, 0.25], [25_000, 0.10]]
                if self.USE_ADAPTIVE_ONLINE_PHASE
                else [[25_000, 0.50]]
            ),
            "ratio_gate_schema": (
                RATIO_GATE_SCHEMA if self.USE_RELIABILITY_GATE else None
            ),
        }


class TransitH2ORatioGatedTrainerV9(_TransitH2OTrainerV9Base):
    V9_VARIANT = "ratio-gated"
    USE_RELIABILITY_GATE = True


class TransitH2OAdaptationTrainerV9(_TransitH2OTrainerV9Base):
    V9_VARIANT = "adaptation"
    USE_ADAPTIVE_ONLINE_PHASE = True


class TransitH2OCombinedTrainerV9(_TransitH2OTrainerV9Base):
    V9_VARIANT = "combined"
    USE_RELIABILITY_GATE = True
    USE_ADAPTIVE_ONLINE_PHASE = True


V9_TRAINER_VARIANTS = {
    "ratio-gated": TransitH2ORatioGatedTrainerV9,
    "adaptation": TransitH2OAdaptationTrainerV9,
    "combined": TransitH2OCombinedTrainerV9,
}


_V7_REGISTERED_SHARED_INITIALIZER_BRANCHES = frozenset(
    {
        "h2oplus-no-ratio",
        "h2oplus-lambda0",
        "h2oplus-lambda0p2",
        "h2oplus-lambda0p5",
        "h2oplus-lambda1",
    }
)


def authorize_registered_initializer_bridge(
    base_module: Any,
    *,
    method: str,
    validator: Any,
) -> frozenset[str]:
    """Authorize sealed v9 initializer reuse inside the wrapper process only."""

    if method != "h2oplus":
        raise ValueError("v9 initializer bridge only authorizes h2oplus")
    if getattr(base_module, "validate_v7_offline_initializer_runtime", None) is not validator:
        raise RuntimeError("v9 initializer validator was not installed before authorization")
    registered = frozenset(
        getattr(base_module, "V7_SHARED_IQL_BRANCH_METHODS", ())
    )
    if registered != _V7_REGISTERED_SHARED_INITIALIZER_BRANCHES:
        raise RuntimeError("v7 shared-initializer branch registry drifted")
    authorized = registered | {method}
    base_module.V7_SHARED_IQL_BRANCH_METHODS = set(authorized)
    return authorized


__all__ = [
    "RATIO_GATE_SCHEMA",
    "ReliabilityGatedDynamicsRatioV9",
    "TransitH2OAdaptationTrainerV9",
    "TransitH2OCombinedTrainerV9",
    "TransitH2ORatioGatedTrainerV9",
    "V9_TRAINER_VARIANTS",
    "V9_VARIANT_SCHEMA",
    "authorize_registered_initializer_bridge",
    "offline_fraction_for_online_events",
]
