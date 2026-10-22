"""Network binding for chronological transit offline-to-online training."""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig

from .networks import (
    HurdleTransitActorV7,
    TransitDynamicsRatioV7,
    TransitQEnsembleV7,
    TransitValueV7,
)


POLICY_ARTIFACT_SCHEMA = "transit-policy-artifact-v7-v1"
OFFLINE_INITIALIZER_SCHEMA = "transit-offline-initializer-v7-v1"
OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS = {
    "mixed_backup_lambda",
    "use_ratio_correction",
}


def inspect_offline_initializer_v7(path: str | Path) -> dict[str, Any]:
    """Load and structurally audit a shared IQL prefix without mutating a trainer."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("artifact_schema") != OFFLINE_INITIALIZER_SCHEMA
        or payload.get("checkpoint_version")
        != TransitH2OTrainerV7.CHECKPOINT_VERSION
    ):
        raise ValueError("not a transit-smdp-v7 offline initializer")
    if (
        int(payload.get("online_critic_updates", -1)) != 0
        or payload.get("online_actor_state_batches") != []
        or payload.get("online_temperature_state_batches") != []
        or "online_replay" in payload
    ):
        raise ValueError("offline initializer contains invalid online state")
    required_states = (
        "actor",
        "critic",
        "target_critic",
        "value",
        "ratio",
        "actor_optimizer",
        "critic_optimizer",
        "value_optimizer",
        "ratio_optimizer",
        "alpha_optimizer",
    )
    if any(payload.get(name) is None for name in required_states):
        raise ValueError("offline initializer is missing model or optimizer state")

    def tensors(value: Any):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)

    audited_states = required_states + ("log_alpha",)
    if not all(
        bool(torch.isfinite(tensor).all())
        for name in audited_states
        for tensor in tensors(payload.get(name))
    ):
        raise ValueError("offline initializer contains non-finite tensor state")
    return {
        "artifact_schema": payload["artifact_schema"],
        "checkpoint_version": payload["checkpoint_version"],
        "source_config": payload.get("source_config"),
        "offline_compatibility_config": payload.get(
            "offline_compatibility_config"
        ),
        "total_updates": int(payload.get("total_updates", -1)),
        "extra": dict(payload.get("extra", {})),
    }


class TransitH2OTrainerV7(TransitH2OTrainer):
    CHECKPOINT_VERSION = (
        "transit-offline-online-v12-smdp-v7-wsrl-source-aligned"
    )

    def __init__(
        self,
        replay_buffer,
        *,
        config: TransitTrainerConfig = TransitTrainerConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        horizon_seconds = float(replay_buffer.contract_config["horizon_seconds"])
        super().__init__(
            replay_buffer,
            config=config,
            device=device,
            seed=seed,
            actor_class=HurdleTransitActorV7,
            critic_class=TransitQEnsembleV7,
            value_class=TransitValueV7,
            ratio_class=TransitDynamicsRatioV7,
            ratio_kwargs={"horizon_seconds": horizon_seconds},
        )

    def save_policy_artifact(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any],
    ) -> None:
        """Save the deployable actor without replay or optimizer state."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema": POLICY_ARTIFACT_SCHEMA,
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "extra": dict(extra),
        }
        temporary = destination.with_suffix(destination.suffix + ".partial")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def _offline_initializer_config(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in asdict(self.config).items()
            if name not in OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS
        }

    def save_offline_initializer(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any],
    ) -> None:
        """Freeze a deterministic IQL prefix for same-seed H2O+ branches."""

        if self.config.offline_objective != "iql":
            raise ValueError("v7 shared offline initialization requires IQL")
        if self.online_critic_updates or self.replay_buffer.online_size:
            raise ValueError("offline initializer cannot contain online training state")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema": OFFLINE_INITIALIZER_SCHEMA,
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "source_config": asdict(self.config),
            "offline_compatibility_config": self._offline_initializer_config(),
            "total_updates": self.total_updates,
            "online_critic_updates": self.online_critic_updates,
            "online_actor_state_batches": [],
            "online_temperature_state_batches": [],
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "value": self.value.state_dict() if self.value is not None else None,
            "ratio": self.ratio.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": (
                self.value_optimizer.state_dict()
                if self.value_optimizer is not None
                else None
            ),
            "ratio_optimizer": self.ratio.optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "numpy_rng_state": self.rng.bit_generator.state,
            "diagnostic_rng_state": self.diagnostic_rng.bit_generator.state,
            "numpy_global_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "extra": dict(extra),
        }
        temporary = destination.with_suffix(destination.suffix + ".partial")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def load_offline_initializer(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if (
            payload.get("artifact_schema") != OFFLINE_INITIALIZER_SCHEMA
            or payload.get("checkpoint_version") != self.CHECKPOINT_VERSION
        ):
            raise ValueError("not a transit-smdp-v7 offline initializer")
        if self.config.offline_objective != "iql" or payload.get(
            "offline_compatibility_config"
        ) != self._offline_initializer_config():
            raise ValueError("offline initializer is incompatible with this branch")
        if (
            payload.get("value") is None
            or int(payload.get("online_critic_updates", -1)) != 0
            or payload.get("online_actor_state_batches") != []
            or payload.get("online_temperature_state_batches") != []
            or "online_replay" in payload
        ):
            raise ValueError("offline initializer contains invalid online state")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        if self.value is None:
            raise ValueError("offline initializer target lacks a value network")
        self.value.load_state_dict(payload["value"])
        self.ratio.load_state_dict(payload["ratio"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if self.value_optimizer is None or payload.get("value_optimizer") is None:
            raise ValueError("offline initializer lacks the value optimizer")
        self.value_optimizer.load_state_dict(payload["value_optimizer"])
        self.ratio.optimizer.load_state_dict(payload["ratio_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.total_updates = int(payload["total_updates"])
        self.online_critic_updates = 0
        self._online_actor_state_batches = []
        self._online_temperature_state_batches = []
        self.rng.bit_generator.state = payload["numpy_rng_state"]
        self.diagnostic_rng.bit_generator.state = payload[
            "diagnostic_rng_state"
        ]
        np.random.set_state(payload["numpy_global_rng_state"])
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_rng_state = payload.get("torch_cuda_rng_state")
        if cuda_rng_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state.cpu(), self.device)
        return dict(payload.get("extra", {}))


def load_actor_checkpoint_v7(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    require_policy_artifact: bool = False,
) -> tuple[HurdleTransitActorV7, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != TransitH2OTrainerV7.CHECKPOINT_VERSION:
        raise ValueError("not a transit-smdp-v7 chronological-welfare checkpoint")
    artifact_schema = payload.get("artifact_schema")
    if artifact_schema not in (None, POLICY_ARTIFACT_SCHEMA):
        raise ValueError("unsupported transit-smdp-v7 policy artifact schema")
    if require_policy_artifact and artifact_schema != POLICY_ARTIFACT_SCHEMA:
        raise ValueError(
            "formal v7 evaluation requires the compact completed-policy artifact"
        )
    config = TransitTrainerConfig(**payload["config"])
    actor = HurdleTransitActorV7(
        (config.hidden_dim, config.hidden_dim),
        layer_norm=config.actor_layer_norm,
        activation=config.activation,
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    extra = dict(payload.get("extra", {}))
    extra["policy_artifact_schema"] = artifact_schema
    return actor, extra


__all__ = [
    "OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS",
    "OFFLINE_INITIALIZER_SCHEMA",
    "POLICY_ARTIFACT_SCHEMA",
    "TransitH2OTrainerV7",
    "TransitTrainerConfig",
    "inspect_offline_initializer_v7",
    "load_actor_checkpoint_v7",
]
