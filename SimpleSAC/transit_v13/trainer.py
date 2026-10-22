"""Preregistered proximal actor-release interventions for H2O+ v13.

Every variant resumes the exact sealed v12 10k state and preserves the v12
actor learning-rate ramp, full-speed temperature update, online-only actor
states, 50/50 critic replay, and unit simulator TD weights.  The sole new
factor is a fixed per-actor-update proximal pull toward the seed-matched 10k
actor.  The immutable anchor is always reconstructed from the materialized
branch source, including after an interrupted run resumes from a later epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from SimpleSAC.transit_v12.trainer import TransitH2OActorRampTrainerV12


V13_VARIANT_SCHEMA = "transit-h2oplus-v13-proximal-release-variant-v1"
PROXIMAL_RATES = {
    "proximal-weak": 1.0e-5,
    "proximal-medium": 2.5e-5,
    "proximal-strong": 5.0e-5,
}
PREDESIGNATED_CANDIDATE = "proximal-medium"
REGISTERED_POST_10K_ACTOR_UPDATES = 15_000


def proximal_retention(rate: float, updates: int) -> float:
    """Return no-gradient retention after repeated proximal interpolation."""

    value = float(rate)
    count = int(updates)
    if not 0.0 <= value < 1.0:
        raise ValueError("proximal rate must be in [0, 1)")
    if count < 0:
        raise ValueError("proximal update count must be nonnegative")
    return math.pow(1.0 - value, count)


def _actor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


class TransitH2OProximalReleaseTrainerV13(TransitH2OActorRampTrainerV12):
    """Apply one fixed proximal dose after every released actor update."""

    V13_VARIANT = "abstract"
    PROXIMAL_RATE = 0.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.V13_VARIANT == "abstract":
            raise TypeError("instantiate a registered v13 intervention variant")
        super().__init__(*args, **kwargs)
        self._v13_actor_anchor: tuple[tuple[str, torch.Tensor], ...] = ()
        self._v13_actor_anchor_sha256 = ""
        self._v13_last_distance_before = 0.0
        self._v13_last_distance_after = 0.0
        self._v13_proximal_update_executed = False

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    def _capture_actor_anchor(
        self,
        checkpoint_path: Path,
        checkpoint_extra: dict[str, Any],
    ) -> None:
        run_dir_raw = checkpoint_extra.get("run_dir")
        source_path = None
        expected_digest = None
        if isinstance(run_dir_raw, str) and run_dir_raw:
            run_dir = Path(run_dir_raw).expanduser().resolve()
            metadata_path = run_dir / "metadata.json"
            candidate = run_dir / "v13_branch_source/latest.pt"
            if metadata_path.is_file() and candidate.is_file():
                metadata = self._read_json(metadata_path)
                binding = metadata.get("v13_branch_handoff", {})
                expected_digest = binding.get("actor_tensor_sha256")
                source_path = candidate
        if source_path is None:
            source_path = checkpoint_path.expanduser().resolve(strict=True)

        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if (
            int(payload.get("total_updates", -1)) != 55_000
            or int(payload.get("online_critic_updates", -1)) != 5_000
        ):
            raise ValueError("v13 actor anchor is not an exact 10k handoff state")
        state = payload.get("actor")
        if not isinstance(state, dict) or not state:
            raise ValueError("v13 actor anchor checkpoint lacks actor tensors")
        digest = _actor_state_sha256(state)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError("v13 actor anchor digest disagrees with branch binding")

        parameters = dict(self.actor.named_parameters())
        if set(parameters) != set(state):
            raise ValueError("v13 actor anchor parameter names drifted")
        anchors = []
        for name, parameter in parameters.items():
            anchor = state[name]
            if tuple(anchor.shape) != tuple(parameter.shape):
                raise ValueError(f"v13 actor anchor shape drifted for {name}")
            anchors.append(
                (name, anchor.detach().to(device=parameter.device, dtype=parameter.dtype).clone())
            )
        self._v13_actor_anchor = tuple(anchors)
        self._v13_actor_anchor_sha256 = digest

    def load(
        self, path: str | Path, *, load_optimizers: bool = True
    ) -> dict[str, Any]:
        checkpoint_path = Path(path).expanduser().resolve(strict=True)
        extra = super().load(checkpoint_path, load_optimizers=load_optimizers)
        self._capture_actor_anchor(checkpoint_path, extra)
        return extra

    def _apply_actor_proximal_pull(self, *, collect_metrics: bool) -> None:
        if not self._v13_actor_anchor:
            raise RuntimeError("v13 actor update attempted before anchor restoration")
        parameters = dict(self.actor.named_parameters())
        if tuple(parameters) != tuple(name for name, _ in self._v13_actor_anchor):
            raise RuntimeError("v13 actor parameters changed after anchor restoration")

        numerator_before = 0.0
        denominator = 0.0
        with torch.no_grad():
            for name, anchor in self._v13_actor_anchor:
                parameter = parameters[name]
                if collect_metrics:
                    numerator_before += float(
                        torch.sum((parameter - anchor).double().square()).item()
                    )
                    denominator += float(torch.sum(anchor.double().square()).item())
                parameter.lerp_(anchor, self.PROXIMAL_RATE)
        if collect_metrics:
            self._v13_last_distance_before = math.sqrt(
                numerator_before / max(denominator, torch.finfo(torch.float64).tiny)
            )
            self._v13_last_distance_after = (
                self._v13_last_distance_before * (1.0 - self.PROXIMAL_RATE)
            )
        self._v13_proximal_update_executed = True

    def _update_sac_actor(
        self,
        actor_states: torch.Tensor,
        *,
        temperature_states: torch.Tensor,
        temperature_action_samples: int = 1,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = super()._update_sac_actor(
            actor_states,
            temperature_states=temperature_states,
            temperature_action_samples=temperature_action_samples,
            collect_metrics=collect_metrics,
        )
        self._apply_actor_proximal_pull(collect_metrics=collect_metrics)
        return result

    def update(
        self,
        *,
        online: bool,
        online_only: bool = False,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        if online:
            self._v13_proximal_update_executed = False
        metrics = super().update(
            online=online,
            online_only=online_only,
            collect_metrics=collect_metrics,
        )
        if collect_metrics and online:
            metrics.update(
                {
                    "v13/proximal_rate": float(self.PROXIMAL_RATE),
                    "v13/proximal_update_executed": float(
                        self._v13_proximal_update_executed
                    ),
                    "v13/actor_anchor_restored": float(
                        bool(self._v13_actor_anchor)
                    ),
                    "v13/actor_relative_l2_before_pull": float(
                        self._v13_last_distance_before
                    ),
                    "v13/actor_relative_l2_after_pull": float(
                        self._v13_last_distance_after
                    ),
                }
            )
        return metrics

    @property
    def v13_intervention(self) -> dict[str, Any]:
        return {
            "schema_version": V13_VARIANT_SCHEMA,
            "variant": self.V13_VARIANT,
            "warmup_events": 5_000,
            "actor_and_temperature_hold_through_online_events": 10_000,
            "actor_learning_rate_ramp": "0.1 to 1.0 over 10k-15k",
            "temperature_learning_rate_after_release": "full registered rate",
            "actor_state_source_after_release": "online-only",
            "actor_anchor": "exact seed-matched sealed v12 10k actor tensors",
            "proximal_update": "theta <- (1-rate)*theta + rate*theta_anchor",
            "proximal_rate_per_actor_update": self.PROXIMAL_RATE,
            "registered_post_10k_actor_updates": REGISTERED_POST_10K_ACTOR_UPDATES,
            "no_gradient_anchor_retention_at_25k": proximal_retention(
                self.PROXIMAL_RATE, REGISTERED_POST_10K_ACTOR_UPDATES
            ),
            "predesignated_candidate": self.V13_VARIANT == PREDESIGNATED_CANDIDATE,
            "offline_critic_replay_schedule": [[0, 0.50], [25_000, 0.50]],
            "online_update_order": "v7-actor-before-critic",
            "unit_ratio_weights": True,
            "ratio_classifiers_continue_for_diagnostics": True,
            "checkpoint_selection_by_behavior": False,
        }


class TransitH2OProximalWeakTrainerV13(TransitH2OProximalReleaseTrainerV13):
    V13_VARIANT = "proximal-weak"
    PROXIMAL_RATE = PROXIMAL_RATES[V13_VARIANT]


class TransitH2OProximalMediumTrainerV13(TransitH2OProximalReleaseTrainerV13):
    V13_VARIANT = "proximal-medium"
    PROXIMAL_RATE = PROXIMAL_RATES[V13_VARIANT]


class TransitH2OProximalStrongTrainerV13(TransitH2OProximalReleaseTrainerV13):
    V13_VARIANT = "proximal-strong"
    PROXIMAL_RATE = PROXIMAL_RATES[V13_VARIANT]


V13_TRAINER_VARIANTS = {
    "proximal-weak": TransitH2OProximalWeakTrainerV13,
    "proximal-medium": TransitH2OProximalMediumTrainerV13,
    "proximal-strong": TransitH2OProximalStrongTrainerV13,
}
