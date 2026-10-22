"""Network binding for chronological transit offline-to-online training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig

from .networks import (
    HurdleTransitActorV6,
    TransitDynamicsRatioV6,
    TransitQEnsembleV6,
    TransitValueV6,
)


class TransitH2OTrainerV6(TransitH2OTrainer):
    CHECKPOINT_VERSION = (
        "transit-offline-online-v9-source-aligned-critic-updates"
    )

    def __init__(
        self,
        replay_buffer,
        *,
        config: TransitTrainerConfig = TransitTrainerConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        episode_seconds = float(replay_buffer.contract_config["episode_seconds"])
        super().__init__(
            replay_buffer,
            config=config,
            device=device,
            seed=seed,
            actor_class=HurdleTransitActorV6,
            critic_class=TransitQEnsembleV6,
            value_class=TransitValueV6,
            ratio_class=TransitDynamicsRatioV6,
            ratio_kwargs={"episode_seconds": episode_seconds},
        )


def load_actor_checkpoint_v6(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[HurdleTransitActorV6, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != TransitH2OTrainerV6.CHECKPOINT_VERSION:
        raise ValueError("not a transit-smdp-v6 chronological-welfare checkpoint")
    config = TransitTrainerConfig(**payload["config"])
    actor = HurdleTransitActorV6(
        (config.hidden_dim, config.hidden_dim),
        layer_norm=config.actor_layer_norm,
        activation=config.activation,
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, dict(payload.get("extra", {}))


__all__ = [
    "TransitH2OTrainerV6",
    "TransitTrainerConfig",
    "load_actor_checkpoint_v6",
]
