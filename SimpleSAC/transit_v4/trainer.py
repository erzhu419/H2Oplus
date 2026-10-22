"""v4 network binding for the shared canonical offline-to-online trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bus_h2o.mdp_contract_v4 import ACTION_DIM
from SimpleSAC.transit_v3.domain_ratio import FactoredDynamicsRatio
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig

from .networks import (
    FEATURE_DIM,
    HurdleTransitActor,
    TransitQEnsembleV4,
    TransitValueV4,
    canonical_features_v4,
)


class TransitH2OTrainerV4(TransitH2OTrainer):
    """H2O+ trainer with the v4 state and hurdle action measure."""

    CHECKPOINT_VERSION = "transit-offline-online-v5-hurdle"

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
            actor_class=HurdleTransitActor,
            critic_class=TransitQEnsembleV4,
            value_class=TransitValueV4,
            ratio_class=FactoredDynamicsRatio,
            ratio_kwargs={
                "feature_dim": FEATURE_DIM,
                "action_dim": ACTION_DIM,
                "feature_fn": canonical_features_v4,
            },
        )


def load_actor_checkpoint_v4(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[HurdleTransitActor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != TransitH2OTrainerV4.CHECKPOINT_VERSION:
        raise ValueError("not a transit-smdp-v4 hurdle-policy checkpoint")
    config = TransitTrainerConfig(**payload["config"])
    actor = HurdleTransitActor(
        (config.hidden_dim, config.hidden_dim),
        layer_norm=config.actor_layer_norm,
        activation=config.activation,
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, dict(payload.get("extra", {}))


__all__ = [
    "TransitH2OTrainerV4",
    "TransitTrainerConfig",
    "load_actor_checkpoint_v4",
]

