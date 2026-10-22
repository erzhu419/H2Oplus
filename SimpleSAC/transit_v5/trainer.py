"""Network binding for true-welfare offline-to-online transit training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bus_h2o.mdp_contract_v5 import ACTION_DIM
from SimpleSAC.transit_v3.domain_ratio import FactoredDynamicsRatio
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig

from .networks import (
    FEATURE_DIM,
    HurdleTransitActorV5,
    TransitQEnsembleV5,
    TransitValueV5,
    canonical_features_v5,
)


class TransitH2OTrainerV5(TransitH2OTrainer):
    CHECKPOINT_VERSION = "transit-offline-online-v6-true-welfare-hurdle"

    def __init__(
        self,
        replay_buffer,
        *,
        config: TransitTrainerConfig = TransitTrainerConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__(
            replay_buffer,
            config=config,
            device=device,
            seed=seed,
            actor_class=HurdleTransitActorV5,
            critic_class=TransitQEnsembleV5,
            value_class=TransitValueV5,
            ratio_class=FactoredDynamicsRatio,
            ratio_kwargs={
                "feature_dim": FEATURE_DIM,
                "action_dim": ACTION_DIM,
                "feature_fn": canonical_features_v5,
            },
        )


def load_actor_checkpoint_v5(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[HurdleTransitActorV5, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != TransitH2OTrainerV5.CHECKPOINT_VERSION:
        raise ValueError("not a transit-smdp-v5 true-welfare checkpoint")
    config = TransitTrainerConfig(**payload["config"])
    actor = HurdleTransitActorV5(
        (config.hidden_dim, config.hidden_dim),
        layer_norm=config.actor_layer_norm,
        activation=config.activation,
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, dict(payload.get("extra", {}))


__all__ = [
    "TransitH2OTrainerV5",
    "TransitTrainerConfig",
    "load_actor_checkpoint_v5",
]
