"""Staged actor/temperature release interventions for H2O+ v12.

All variants preserve the v11 actor-and-temperature hold through 10k online
events. Registered branches then isolate actor learning-rate release,
temperature learning-rate release, and a small offline actor-state anchor.
"""

from __future__ import annotations

from typing import Any

import torch

from SimpleSAC.transit_v11.trainer import (
    ACTOR_HOLD_CRITIC_UPDATES,
    TransitH2OActorHoldTrainerV11,
)


V12_VARIANT_SCHEMA = "transit-h2oplus-v12-staged-release-variant-v1"
RELEASE_START_CRITIC_UPDATES = ACTOR_HOLD_CRITIC_UPDATES
RELEASE_END_CRITIC_UPDATES = 10_000
MINIMUM_RELEASE_MULTIPLIER = 0.10
ACTOR_ANCHOR_OFFLINE_FRACTION = 0.20
REGISTERED_ACTOR_BATCH_SIZE = 128
REGISTERED_ACTOR_ANCHOR_OFFLINE_ROWS = round(
    REGISTERED_ACTOR_BATCH_SIZE * ACTOR_ANCHOR_OFFLINE_FRACTION
)


def release_multiplier(online_critic_updates: int) -> float:
    """Linearly restore an optimizer from 10% at 10k to full speed at 15k."""

    updates = int(online_critic_updates)
    if updates <= RELEASE_START_CRITIC_UPDATES:
        return 0.0
    if updates >= RELEASE_END_CRITIC_UPDATES:
        return 1.0
    progress = (updates - RELEASE_START_CRITIC_UPDATES) / (
        RELEASE_END_CRITIC_UPDATES - RELEASE_START_CRITIC_UPDATES
    )
    return MINIMUM_RELEASE_MULTIPLIER + progress * (
        1.0 - MINIMUM_RELEASE_MULTIPLIER
    )


def anchored_actor_states(
    offline_states: torch.Tensor,
    online_states: torch.Tensor,
    *,
    offline_fraction: float = ACTOR_ANCHOR_OFFLINE_FRACTION,
) -> torch.Tensor:
    """Build a fixed-size actor batch with a registered offline-state share."""

    if offline_states.ndim != online_states.ndim:
        raise ValueError("offline and online actor states must have equal rank")
    if offline_states.shape[1:] != online_states.shape[1:]:
        raise ValueError("offline and online actor state shapes disagree")
    if not 0.0 <= float(offline_fraction) < 1.0:
        raise ValueError("offline actor anchor fraction must be in [0, 1)")
    target = int(online_states.shape[0])
    if target <= 0:
        raise ValueError("online actor state batch must be nonempty")
    offline_count = int(round(target * float(offline_fraction)))
    online_count = target - offline_count
    if offline_states.shape[0] < offline_count:
        raise ValueError("offline actor state batch is too small for the anchor")
    if online_states.shape[0] < online_count:
        raise ValueError("online actor state batch is too small for the anchor")
    return torch.cat(
        (offline_states[:offline_count], online_states[:online_count]), dim=0
    )


class _TransitH2OStagedReleaseTrainerV12(TransitH2OActorHoldTrainerV11):
    V12_VARIANT = "abstract"
    USE_ACTOR_LR_RAMP = False
    USE_TEMPERATURE_LR_RAMP = False
    USE_ACTOR_STATE_ANCHOR = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.V12_VARIANT == "abstract":
            raise TypeError("instantiate a registered v12 intervention variant")
        super().__init__(*args, **kwargs)
        self._v12_last_actor_lr_multiplier = 0.0
        self._v12_last_temperature_lr_multiplier = 0.0
        self._v12_actor_update_executed = False
        self._v12_last_actor_anchor_offline_rows = 0
        self._v12_last_actor_batch_rows = 0

    def _actor_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if (
            self.USE_ACTOR_STATE_ANCHOR
            and not self._actor_hold_active_for_update()
            and offline_batch is not None
            and online_batch is not None
        ):
            states = anchored_actor_states(
                offline_batch["observations"],
                online_batch["observations"],
            )
            self._v12_last_actor_batch_rows = int(states.shape[0])
            self._v12_last_actor_anchor_offline_rows = int(
                round(states.shape[0] * ACTOR_ANCHOR_OFFLINE_FRACTION)
            )
            return states
        return super()._actor_states(offline_batch, online_batch)

    @staticmethod
    def _scale_optimizer_lrs(optimizer: Any, multiplier: float) -> list[float]:
        original = [float(group["lr"]) for group in optimizer.param_groups]
        for group, learning_rate in zip(optimizer.param_groups, original):
            group["lr"] = learning_rate * float(multiplier)
        return original

    @staticmethod
    def _restore_optimizer_lrs(optimizer: Any, values: list[float]) -> None:
        if len(optimizer.param_groups) != len(values):
            raise RuntimeError("optimizer parameter groups changed during v12 update")
        for group, learning_rate in zip(optimizer.param_groups, values):
            group["lr"] = learning_rate

    def _update_sac_actor(
        self,
        actor_states: torch.Tensor,
        *,
        temperature_states: torch.Tensor,
        temperature_action_samples: int = 1,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scheduled = release_multiplier(self.online_critic_updates)
        actor_multiplier = scheduled if self.USE_ACTOR_LR_RAMP else 1.0
        temperature_multiplier = (
            scheduled if self.USE_TEMPERATURE_LR_RAMP else 1.0
        )
        actor_lrs = self._scale_optimizer_lrs(
            self.actor_optimizer, actor_multiplier
        )
        temperature_lrs = self._scale_optimizer_lrs(
            self.alpha_optimizer, temperature_multiplier
        )
        self._v12_last_actor_lr_multiplier = actor_multiplier
        self._v12_last_temperature_lr_multiplier = temperature_multiplier
        self._v12_actor_update_executed = True
        try:
            return super()._update_sac_actor(
                actor_states,
                temperature_states=temperature_states,
                temperature_action_samples=temperature_action_samples,
                collect_metrics=collect_metrics,
            )
        finally:
            self._restore_optimizer_lrs(self.actor_optimizer, actor_lrs)
            self._restore_optimizer_lrs(self.alpha_optimizer, temperature_lrs)

    def update(
        self,
        *,
        online: bool,
        online_only: bool = False,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        if online:
            self._v12_actor_update_executed = False
            self._v12_last_actor_anchor_offline_rows = 0
            self._v12_last_actor_batch_rows = 0
        if online and self._actor_hold_active_for_update():
            self._v12_last_actor_lr_multiplier = 0.0
            self._v12_last_temperature_lr_multiplier = 0.0
        metrics = super().update(
            online=online,
            online_only=online_only,
            collect_metrics=collect_metrics,
        )
        if collect_metrics and online:
            metrics.update(
                {
                    "v12/actor_lr_multiplier": float(
                        self._v12_last_actor_lr_multiplier
                    ),
                    "v12/temperature_lr_multiplier": float(
                        self._v12_last_temperature_lr_multiplier
                    ),
                    "v12/actor_anchor_offline_fraction": float(
                        self._v12_last_actor_anchor_offline_rows
                        / self._v12_last_actor_batch_rows
                        if self._v12_last_actor_batch_rows
                        else 0.0
                    ),
                    "v12/staged_release_active": float(
                        self._v12_actor_update_executed
                    ),
                }
            )
        return metrics

    @property
    def v12_intervention(self) -> dict[str, Any]:
        return {
            "schema_version": V12_VARIANT_SCHEMA,
            "variant": self.V12_VARIANT,
            "warmup_events": 5_000,
            "actor_and_temperature_hold_through_online_events": 10_000,
            "release_ramp_online_events": [10_000, 15_000],
            "release_ramp_critic_updates": [
                RELEASE_START_CRITIC_UPDATES,
                RELEASE_END_CRITIC_UPDATES,
            ],
            "minimum_release_multiplier": MINIMUM_RELEASE_MULTIPLIER,
            "actor_learning_rate_ramp": self.USE_ACTOR_LR_RAMP,
            "temperature_learning_rate_ramp": self.USE_TEMPERATURE_LR_RAMP,
            "actor_state_anchor": self.USE_ACTOR_STATE_ANCHOR,
            "actor_anchor_offline_fraction": (
                ACTOR_ANCHOR_OFFLINE_FRACTION
                if self.USE_ACTOR_STATE_ANCHOR
                else 0.0
            ),
            "actor_anchor_rounding": "nearest integer rows",
            "registered_actor_batch_size": REGISTERED_ACTOR_BATCH_SIZE,
            "registered_actor_anchor_offline_rows": (
                REGISTERED_ACTOR_ANCHOR_OFFLINE_ROWS
                if self.USE_ACTOR_STATE_ANCHOR
                else 0
            ),
            "registered_actor_anchor_effective_offline_fraction": (
                REGISTERED_ACTOR_ANCHOR_OFFLINE_ROWS / REGISTERED_ACTOR_BATCH_SIZE
                if self.USE_ACTOR_STATE_ANCHOR
                else 0.0
            ),
            "actor_batch_size_preserved": True,
            "temperature_state_source_after_release": "online-only",
            "offline_critic_replay_schedule": [[0, 0.50], [25_000, 0.50]],
            "online_update_order": "v7-actor-before-critic",
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "checkpoint_selection_by_behavior": False,
        }


class TransitH2OSharedPrefixTrainerV12(_TransitH2OStagedReleaseTrainerV12):
    V12_VARIANT = "shared-prefix"


class TransitH2OAbruptControlTrainerV12(_TransitH2OStagedReleaseTrainerV12):
    V12_VARIANT = "abrupt-control"


class TransitH2OActorRampTrainerV12(_TransitH2OStagedReleaseTrainerV12):
    V12_VARIANT = "actor-ramp"
    USE_ACTOR_LR_RAMP = True


class TransitH2OTemperatureRampTrainerV12(_TransitH2OStagedReleaseTrainerV12):
    V12_VARIANT = "temperature-ramp"
    USE_TEMPERATURE_LR_RAMP = True


class TransitH2OActorAnchorTrainerV12(_TransitH2OStagedReleaseTrainerV12):
    V12_VARIANT = "actor-anchor"
    USE_ACTOR_STATE_ANCHOR = True


class TransitH2OStagedCombinedTrainerV12(_TransitH2OStagedReleaseTrainerV12):
    V12_VARIANT = "staged-combined"
    USE_ACTOR_LR_RAMP = True
    USE_TEMPERATURE_LR_RAMP = True
    USE_ACTOR_STATE_ANCHOR = True


V12_TRAINER_VARIANTS = {
    "shared-prefix": TransitH2OSharedPrefixTrainerV12,
    "abrupt-control": TransitH2OAbruptControlTrainerV12,
    "actor-ramp": TransitH2OActorRampTrainerV12,
    "temperature-ramp": TransitH2OTemperatureRampTrainerV12,
    "actor-anchor": TransitH2OActorAnchorTrainerV12,
    "staged-combined": TransitH2OStagedCombinedTrainerV12,
}
