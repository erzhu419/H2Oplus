"""Single-factor delayed-transition interventions for H2O+ v10.

Every v10 branch starts from the same source- and state-bound 10k-event
checkpoint. Ratio classifiers continue to train for diagnostics, but their TD
weights are fixed to one. The replay, actor-state and update-order changes are
activated only after 10k online decision events so the early H2O+ policy is
preserved exactly.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from SimpleSAC.transit_v3.domain_ratio import conditional_log_ratio
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig
from SimpleSAC.transit_v7.networks import (
    HurdleTransitActorV7,
    TransitDynamicsRatioV7,
    TransitQEnsembleV7,
    TransitValueV7,
)
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


V10_VARIANT_SCHEMA = "transit-h2oplus-v10-delayed-variant-v1"
UNIT_RATIO_SCHEMA = "transit-h2oplus-v10-unit-ratio-v1"
TRANSITION_START_EVENTS = 10_000
TRANSITION_END_EVENTS = 25_000
FINAL_OFFLINE_FRACTION = 0.25


def transition_is_active(online_events: int) -> bool:
    """Return whether a delayed branch may depart from its shared prefix."""

    return int(online_events) > TRANSITION_START_EVENTS


def delayed_offline_fraction(online_events: int) -> float:
    """Keep 50% offline replay through 10k, then ramp to 25% at 25k."""

    events = max(int(online_events), 0)
    if events <= TRANSITION_START_EVENTS:
        return 0.50
    if events >= TRANSITION_END_EVENTS:
        return FINAL_OFFLINE_FRACTION
    progress = (events - TRANSITION_START_EVENTS) / (
        TRANSITION_END_EVENTS - TRANSITION_START_EVENTS
    )
    return 0.50 + progress * (FINAL_OFFLINE_FRACTION - 0.50)


class UnitWeightDynamicsRatioV10(TransitDynamicsRatioV7):
    """Train the ratio classifiers while applying unit simulator TD weights."""

    @torch.no_grad()
    def weights(
        self,
        sim_batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        joint_logits, state_action_logits = self.logits(sim_batch)
        log_weights = conditional_log_ratio(joint_logits, state_action_logits)
        weights = torch.ones_like(log_weights)
        if not collect_metrics:
            return weights, {}
        finite = log_weights.clamp(-30.0, 30.0)
        raw_weights = finite.exp()
        raw_ess = raw_weights.sum().square() / raw_weights.square().sum().clamp_min(
            1e-8
        )
        return weights, {
            "ratio/weight_min": weights.min(),
            "ratio/weight_max": weights.max(),
            "ratio/weight_mean": weights.mean(),
            "ratio/ess_fraction": torch.ones((), device=weights.device),
            "ratio/raw_ess_fraction": raw_ess / max(weights.numel(), 1),
            "ratio/clip_fraction": torch.zeros((), device=weights.device),
            "ratio/low_clip_fraction": torch.zeros((), device=weights.device),
            "ratio/high_clip_fraction": torch.zeros((), device=weights.device),
            "ratio/log_weight_mean": log_weights.mean(),
            "ratio/log_weight_std": log_weights.std(unbiased=False),
            "ratio/v10_unit_weights": torch.ones((), device=weights.device),
        }


class _TransitH2ODelayedTrainerV10Base(TransitH2OTrainerV7):
    CHECKPOINT_VERSION = TransitH2OTrainerV7.CHECKPOINT_VERSION
    V10_VARIANT = "abstract"
    USE_DELAYED_REPLAY = False
    USE_DELAYED_ACTOR_STATES = False
    USE_DELAYED_UPDATE_ORDER = False

    def __init__(
        self,
        replay_buffer,
        *,
        config: TransitTrainerConfig = TransitTrainerConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if self.V10_VARIANT == "abstract":
            raise TypeError("instantiate a registered v10 intervention variant")
        horizon_seconds = float(replay_buffer.contract_config["horizon_seconds"])
        TransitH2OTrainer.__init__(
            self,
            replay_buffer,
            config=config,
            device=device,
            seed=seed,
            actor_class=HurdleTransitActorV7,
            critic_class=TransitQEnsembleV7,
            value_class=TransitValueV7,
            ratio_class=UnitWeightDynamicsRatioV10,
            ratio_kwargs={"horizon_seconds": horizon_seconds},
        )

    def _transition_active(self) -> bool:
        return transition_is_active(self.replay_buffer.online_size)

    def _actor_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if (
            self.USE_DELAYED_ACTOR_STATES
            and self._transition_active()
            and online_batch is not None
        ):
            return online_batch["observations"]
        return TransitH2OTrainer._actor_states(offline_batch, online_batch)

    def _temperature_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if (
            self.USE_DELAYED_ACTOR_STATES
            and self._transition_active()
            and online_batch is not None
        ):
            key = (
                "observations"
                if self.config.temperature_state_source == "actor"
                else "next_observations"
            )
            return online_batch[key]
        return super()._temperature_states(offline_batch, online_batch)

    def update(
        self,
        *,
        online: bool,
        online_only: bool = False,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        active = bool(online and self._transition_active())
        offline_fraction = (
            delayed_offline_fraction(self.replay_buffer.online_size)
            if active and self.USE_DELAYED_REPLAY
            else 1.0 - self.config.online_fraction
        )
        original_config = self.config
        if active and (
            self.USE_DELAYED_REPLAY or self.USE_DELAYED_UPDATE_ORDER
        ):
            self.config = replace(
                original_config,
                online_fraction=(
                    1.0 - offline_fraction
                    if self.USE_DELAYED_REPLAY
                    else original_config.online_fraction
                ),
                online_update_order=(
                    "critic-before-actor"
                    if self.USE_DELAYED_UPDATE_ORDER
                    else original_config.online_update_order
                ),
            )
        try:
            metrics = super().update(
                online=online,
                online_only=online_only,
                collect_metrics=collect_metrics,
            )
        finally:
            self.config = original_config
        if collect_metrics and online:
            metrics.update(
                {
                    "v10/transition_active": float(active),
                    "v10/offline_replay_fraction": float(offline_fraction),
                    "v10/online_replay_fraction": float(1.0 - offline_fraction),
                    "v10/delayed_replay_active": float(
                        active and self.USE_DELAYED_REPLAY
                    ),
                    "v10/delayed_actor_states_active": float(
                        active and self.USE_DELAYED_ACTOR_STATES
                    ),
                    "v10/delayed_update_order_active": float(
                        active and self.USE_DELAYED_UPDATE_ORDER
                    ),
                    "v10/unit_ratio_weights": 1.0,
                }
            )
        return metrics

    @property
    def v10_intervention(self) -> dict[str, Any]:
        return {
            "schema_version": V10_VARIANT_SCHEMA,
            "variant": self.V10_VARIANT,
            "unit_ratio_schema": UNIT_RATIO_SCHEMA,
            "unit_ratio_weights": True,
            "transition_start_online_events_exclusive": TRANSITION_START_EVENTS,
            "transition_end_online_events": TRANSITION_END_EVENTS,
            "delayed_replay": self.USE_DELAYED_REPLAY,
            "delayed_actor_states": self.USE_DELAYED_ACTOR_STATES,
            "delayed_update_order": self.USE_DELAYED_UPDATE_ORDER,
            "offline_replay_schedule": (
                [[10_000, 0.50], [25_000, 0.25]]
                if self.USE_DELAYED_REPLAY
                else [[25_000, 0.50]]
            ),
            "actor_state_source_after_transition": (
                "online-only" if self.USE_DELAYED_ACTOR_STATES else "v7-mixed"
            ),
            "online_update_order_after_transition": (
                "critic-before-actor"
                if self.USE_DELAYED_UPDATE_ORDER
                else "v7-actor-before-critic"
            ),
            "actor_burn_in_updates": 0,
        }


class TransitH2OSharedPrefixTrainerV10(_TransitH2ODelayedTrainerV10Base):
    V10_VARIANT = "shared-prefix"


class TransitH2OUnitControlTrainerV10(_TransitH2ODelayedTrainerV10Base):
    V10_VARIANT = "unit-control"


class TransitH2ODelayedReplayTrainerV10(_TransitH2ODelayedTrainerV10Base):
    V10_VARIANT = "delayed-replay"
    USE_DELAYED_REPLAY = True


class TransitH2ODelayedActorTrainerV10(_TransitH2ODelayedTrainerV10Base):
    V10_VARIANT = "delayed-actor"
    USE_DELAYED_ACTOR_STATES = True


class TransitH2ODelayedOrderTrainerV10(_TransitH2ODelayedTrainerV10Base):
    V10_VARIANT = "delayed-order"
    USE_DELAYED_UPDATE_ORDER = True


class TransitH2ODelayedCombinedTrainerV10(_TransitH2ODelayedTrainerV10Base):
    V10_VARIANT = "delayed-combined"
    USE_DELAYED_REPLAY = True
    USE_DELAYED_ACTOR_STATES = True
    USE_DELAYED_UPDATE_ORDER = True


V10_TRAINER_VARIANTS = {
    "shared-prefix": TransitH2OSharedPrefixTrainerV10,
    "unit-control": TransitH2OUnitControlTrainerV10,
    "delayed-replay": TransitH2ODelayedReplayTrainerV10,
    "delayed-actor": TransitH2ODelayedActorTrainerV10,
    "delayed-order": TransitH2ODelayedOrderTrainerV10,
    "delayed-combined": TransitH2ODelayedCombinedTrainerV10,
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
    """Authorize the sealed v8 initializer in this wrapper process only."""

    if method != "h2oplus":
        raise ValueError("v10 initializer bridge only authorizes h2oplus")
    if getattr(base_module, "validate_v7_offline_initializer_runtime", None) is not validator:
        raise RuntimeError("v10 initializer validator was not installed")
    registered = frozenset(
        getattr(base_module, "V7_SHARED_IQL_BRANCH_METHODS", ())
    )
    if registered != _V7_REGISTERED_SHARED_INITIALIZER_BRANCHES:
        raise RuntimeError("v7 shared-initializer branch registry drifted")
    authorized = registered | {method}
    base_module.V7_SHARED_IQL_BRANCH_METHODS = set(authorized)
    return authorized
