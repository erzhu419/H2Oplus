"""Early actor-hold intervention for the H2O+ v11 development pilot.

The critic, value function and diagnostic dynamics-ratio classifiers continue
to adapt after the registered 5k collection warmup. Actor and temperature
updates remain disabled for the next 5k online critic updates, which is the
registered 10k nominal milestone even when an atomic event batch overshoots its
nominal boundary. After release, the actor trains only on online states. Replay
remains the registered 50/50 mix and simulator TD weights remain one throughout.
"""

from __future__ import annotations

from typing import Any

import torch

from SimpleSAC.transit_v3.trainer import TransitH2OTrainer
from SimpleSAC.transit_v10.trainer import TransitH2ODelayedActorTrainerV10


V11_VARIANT_SCHEMA = "transit-h2oplus-v11-actor-hold-variant-v2"
ACTOR_RELEASE_EVENTS = 10_000
ACTOR_HOLD_CRITIC_UPDATES = 5_000


def actor_hold_is_active(online_events: int) -> bool:
    """Return whether actor and temperature updates must still be held."""

    return int(online_events) <= ACTOR_RELEASE_EVENTS


class TransitH2OActorHoldTrainerV11(TransitH2ODelayedActorTrainerV10):
    """Hold the offline actor through 10k, then use online-only actor states."""

    V10_VARIANT = "actor-hold-online-only"

    def _actor_hold_active_for_update(self) -> bool:
        return self.online_critic_updates < ACTOR_HOLD_CRITIC_UPDATES

    def _actor_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self._actor_hold_active_for_update() and online_batch is not None:
            return online_batch["observations"]
        return TransitH2OTrainer._actor_states(offline_batch, online_batch)

    def _temperature_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self._actor_hold_active_for_update() and online_batch is not None:
            key = (
                "observations"
                if self.config.temperature_state_source == "actor"
                else "next_observations"
            )
            return online_batch[key]
        return TransitH2OTrainer._temperature_states(
            self, offline_batch, online_batch
        )

    def _scheduled_online_actor_states(
        self,
        actor_states: torch.Tensor,
        temperature_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._actor_hold_active_for_update():
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
        metrics = super().update(
            online=online,
            online_only=online_only,
            collect_metrics=collect_metrics,
        )
        if collect_metrics and online:
            held = self.online_critic_updates <= ACTOR_HOLD_CRITIC_UPDATES
            metrics.update(
                {
                    "v11/actor_hold_active": float(held),
                    "v11/actor_release_events": float(ACTOR_RELEASE_EVENTS),
                    "v11/actor_hold_critic_updates": float(
                        ACTOR_HOLD_CRITIC_UPDATES
                    ),
                    "v11/online_only_actor_active": float(not held),
                    "v11/unit_ratio_weights": 1.0,
                    "v11/offline_replay_fraction": 0.5,
                }
            )
        return metrics

    @property
    def v11_intervention(self) -> dict[str, Any]:
        return {
            "schema_version": V11_VARIANT_SCHEMA,
            "variant": self.V10_VARIANT,
            "warmup_events": 5_000,
            "actor_and_temperature_hold_through_online_events": (
                ACTOR_RELEASE_EVENTS
            ),
            "actor_and_temperature_hold_online_critic_updates": (
                ACTOR_HOLD_CRITIC_UPDATES
            ),
            "actor_state_source_after_release": "online-only",
            "offline_replay_schedule": [[0, 0.50], [25_000, 0.50]],
            "online_update_order": "v7-actor-before-critic",
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "checkpoint_selection_by_behavior": False,
        }


V11_TRAINER_VARIANTS = {
    "actor-hold-online-only": TransitH2OActorHoldTrainerV11,
}
