"""Preregistered terminal actor-taper interventions for H2O+ v14.

Every variant resumes the exact sealed v12 10k state and preserves the v13
strong fixed proximal pull. The sole new factor is a linear actor learning-rate
taper after the v12 release ramp reaches full speed at 15k events. Temperature,
critic replay, update ordering, and simulator TD weights remain unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from SimpleSAC.transit_v12.trainer import release_multiplier
from SimpleSAC.transit_v13.trainer import (
    REGISTERED_POST_10K_ACTOR_UPDATES,
    TransitH2OProximalReleaseTrainerV13,
    proximal_retention,
)


V14_VARIANT_SCHEMA = "transit-h2oplus-v14-terminal-taper-variant-v1"
BASE_PROXIMAL_RATE = 5.0e-5
TERMINAL_ACTOR_LR_MULTIPLIERS = {
    "terminal-half": 0.50,
    "terminal-quarter": 0.25,
    "terminal-zero": 0.0,
}
PREDESIGNATED_CANDIDATE = "terminal-quarter"

# The actor is released when online_critic_updates reaches 5k. The v12 ramp is
# full at 10k, corresponding to 15k simulator events. The final registered
# actor update starts with online_critic_updates == 19,999.
ACTOR_RELEASE_CRITIC_UPDATES = 5_000
TERMINAL_TAPER_START_CRITIC_UPDATES = 10_000
FINAL_REGISTERED_ACTOR_UPDATE = 19_999
REGISTERED_TERMINAL_CRITIC_UPDATES = 20_000


def terminal_actor_lr_multiplier(
    online_critic_updates: int,
    terminal_multiplier: float,
) -> float:
    """Return the registered post-15k linear actor LR taper multiplier."""

    updates = int(online_critic_updates)
    terminal = float(terminal_multiplier)
    if updates < 0:
        raise ValueError("online critic updates must be nonnegative")
    if not 0.0 <= terminal <= 1.0:
        raise ValueError("terminal actor LR multiplier must be in [0, 1]")
    if updates <= TERMINAL_TAPER_START_CRITIC_UPDATES:
        return 1.0
    if updates >= FINAL_REGISTERED_ACTOR_UPDATE:
        return terminal
    progress = (
        updates - TERMINAL_TAPER_START_CRITIC_UPDATES
    ) / (
        FINAL_REGISTERED_ACTOR_UPDATE
        - TERMINAL_TAPER_START_CRITIC_UPDATES
    )
    return 1.0 + progress * (terminal - 1.0)


def integrated_actor_lr_exposure(terminal_multiplier: float) -> float:
    """Return full-rate-equivalent actor updates over the registered branch."""

    return sum(
        release_multiplier(update)
        * terminal_actor_lr_multiplier(update, terminal_multiplier)
        for update in range(
            ACTOR_RELEASE_CRITIC_UPDATES,
            REGISTERED_TERMINAL_CRITIC_UPDATES,
        )
    )


class TransitH2OTerminalTaperTrainerV14(
    TransitH2OProximalReleaseTrainerV13
):
    """Compose the v12 release ramp with a registered terminal LR taper."""

    V13_VARIANT = "v14-terminal-taper"
    PROXIMAL_RATE = BASE_PROXIMAL_RATE
    V14_VARIANT = "abstract"
    TERMINAL_ACTOR_LR_MULTIPLIER = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.V14_VARIANT == "abstract":
            raise TypeError("instantiate a registered v14 intervention variant")
        super().__init__(*args, **kwargs)
        self._v14_last_terminal_taper_multiplier = 1.0

    def _update_sac_actor(
        self,
        actor_states: torch.Tensor,
        *,
        temperature_states: torch.Tensor,
        temperature_action_samples: int = 1,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        taper = terminal_actor_lr_multiplier(
            self.online_critic_updates,
            self.TERMINAL_ACTOR_LR_MULTIPLIER,
        )
        original_actor_lrs = self._scale_optimizer_lrs(
            self.actor_optimizer, taper
        )
        self._v14_last_terminal_taper_multiplier = taper
        try:
            return super()._update_sac_actor(
                actor_states,
                temperature_states=temperature_states,
                temperature_action_samples=temperature_action_samples,
                collect_metrics=collect_metrics,
            )
        finally:
            self._restore_optimizer_lrs(
                self.actor_optimizer, original_actor_lrs
            )

    def update(
        self,
        *,
        online: bool,
        online_only: bool = False,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        metrics = super().update(
            online=online,
            online_only=online_only,
            collect_metrics=collect_metrics,
        )
        if collect_metrics and online:
            metrics.update(
                {
                    "v14/base_proximal_rate": BASE_PROXIMAL_RATE,
                    "v14/terminal_actor_lr_multiplier": float(
                        self.TERMINAL_ACTOR_LR_MULTIPLIER
                    ),
                    "v14/current_terminal_taper_multiplier": float(
                        self._v14_last_terminal_taper_multiplier
                    ),
                    "v14/full_rate_equivalent_actor_updates": float(
                        integrated_actor_lr_exposure(
                            self.TERMINAL_ACTOR_LR_MULTIPLIER
                        )
                    ),
                }
            )
        return metrics

    @property
    def v14_intervention(self) -> dict[str, Any]:
        return {
            "schema_version": V14_VARIANT_SCHEMA,
            "variant": self.V14_VARIANT,
            "warmup_events": 5_000,
            "actor_and_temperature_hold_through_online_events": 10_000,
            "actor_learning_rate_release_ramp": "0.1 to 1.0 over 10k-15k",
            "terminal_actor_lr_taper_events": [15_000, 25_000],
            "terminal_actor_lr_taper_critic_updates": [
                TERMINAL_TAPER_START_CRITIC_UPDATES,
                FINAL_REGISTERED_ACTOR_UPDATE,
            ],
            "terminal_actor_lr_multiplier": (
                self.TERMINAL_ACTOR_LR_MULTIPLIER
            ),
            "full_rate_equivalent_actor_updates": (
                integrated_actor_lr_exposure(
                    self.TERMINAL_ACTOR_LR_MULTIPLIER
                )
            ),
            "temperature_learning_rate_after_release": "full registered rate",
            "actor_state_source_after_release": "online-only",
            "actor_anchor": "exact seed-matched sealed v12 10k actor tensors",
            "proximal_update": (
                "theta <- (1-rate)*theta + rate*theta_anchor"
            ),
            "proximal_rate_per_actor_update": BASE_PROXIMAL_RATE,
            "registered_post_10k_actor_updates": (
                REGISTERED_POST_10K_ACTOR_UPDATES
            ),
            "no_gradient_anchor_retention_at_25k": proximal_retention(
                BASE_PROXIMAL_RATE,
                REGISTERED_POST_10K_ACTOR_UPDATES,
            ),
            "predesignated_candidate": (
                self.V14_VARIANT == PREDESIGNATED_CANDIDATE
            ),
            "offline_critic_replay_schedule": [[0, 0.50], [25_000, 0.50]],
            "online_update_order": "v7-actor-before-critic",
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "checkpoint_selection_by_behavior": False,
        }


class TransitH2OTerminalHalfTrainerV14(TransitH2OTerminalTaperTrainerV14):
    V14_VARIANT = "terminal-half"
    TERMINAL_ACTOR_LR_MULTIPLIER = TERMINAL_ACTOR_LR_MULTIPLIERS[V14_VARIANT]


class TransitH2OTerminalQuarterTrainerV14(TransitH2OTerminalTaperTrainerV14):
    V14_VARIANT = "terminal-quarter"
    TERMINAL_ACTOR_LR_MULTIPLIER = TERMINAL_ACTOR_LR_MULTIPLIERS[V14_VARIANT]


class TransitH2OTerminalZeroTrainerV14(TransitH2OTerminalTaperTrainerV14):
    V14_VARIANT = "terminal-zero"
    TERMINAL_ACTOR_LR_MULTIPLIER = TERMINAL_ACTOR_LR_MULTIPLIERS[V14_VARIANT]


V14_TRAINER_VARIANTS = {
    "terminal-half": TransitH2OTerminalHalfTrainerV14,
    "terminal-quarter": TransitH2OTerminalQuarterTrainerV14,
    "terminal-zero": TransitH2OTerminalZeroTrainerV14,
}
