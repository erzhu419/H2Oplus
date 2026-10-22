"""Canonical offline-to-online trainers for transit holding control."""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .domain_ratio import FactoredDynamicsRatio
from .networks import TransitActor, TransitQEnsemble, TransitValue


@dataclass(frozen=True)
class TransitTrainerConfig:
    ensemble_size: int = 5
    hidden_dim: int = 256
    batch_size: int = 512
    reward_scale: float = 1.0
    critic_loss: str = "mse"
    huber_delta: float = 1.0
    activation: str = "silu"
    actor_layer_norm: bool = True
    critic_layer_norm: bool = True
    value_layer_norm: bool = True
    online_fraction: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    ratio_learning_rate: float = 1e-4
    soft_target_rate: float = 0.005
    target_subset_size: int = 2
    target_subset_replacement: bool = False
    target_entropy: float = -1.0
    initial_temperature: float = 0.1
    backup_entropy: bool = True
    bootstrap_probability: float = 0.8
    offline_objective: str = "cql"
    online_backup: str = "sac"
    actor_q_reduction: str = "min-subset"
    online_actor_update_interval: int = 1
    online_actor_batch_mode: str = "latest"
    online_update_order: str = "critic-before-actor"
    offline_cql_alpha: float = 0.2
    online_cql_alpha: float = 0.0
    cql_n_actions: int = 10
    cql_max_target_backup: bool = False
    mixed_backup_lambda: float = 0.0
    iql_expectile: float = 0.7
    iql_advantage_temperature: float = 3.0
    iql_max_weight: float = 100.0
    # H2O clips P_real/P_sim on simulated TD errors without batch
    # self-normalization; this preserves the absolute trust in the simulator.
    ratio_min_weight: float = 1e-5
    ratio_max_weight: float = 1.0
    use_ratio_correction: bool = True
    ratio_update_interval: int = 4
    gradient_clip_norm: float | None = None
    cql_penalty_subset_size: int | None = None
    cql_penalty_subset_replacement: bool = False
    offline_update_order: str = "critic-before-actor"
    temperature_state_source: str = "actor"
    offline_temperature_action_samples: int = 1


class TransitH2OTrainer:
    """Shared implementation for H2O+ and all canonical comparison methods."""

    CHECKPOINT_VERSION = "transit-offline-online-v4"

    def __init__(
        self,
        replay_buffer,
        *,
        config: TransitTrainerConfig = TransitTrainerConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
        actor_class=TransitActor,
        critic_class=TransitQEnsemble,
        value_class=TransitValue,
        ratio_class=FactoredDynamicsRatio,
        ratio_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if config.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two")
        if not 1 <= config.target_subset_size <= config.ensemble_size:
            raise ValueError("target_subset_size must lie in [1, ensemble_size]")
        if config.cql_penalty_subset_size is not None and not (
            1 <= config.cql_penalty_subset_size <= config.ensemble_size
        ):
            raise ValueError(
                "cql_penalty_subset_size must lie in [1, ensemble_size] or be None"
            )
        if not 0.0 < config.bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must lie in (0, 1]")
        if config.ratio_update_interval <= 0:
            raise ValueError("ratio_update_interval must be positive")
        if config.offline_objective not in {"none", "cql", "iql", "bc"}:
            raise ValueError("unsupported offline objective")
        if config.online_backup not in {"sac", "h2oplus"}:
            raise ValueError("online_backup must be 'sac' or 'h2oplus'")
        if config.actor_q_reduction not in {"mean-all", "min-all", "min-subset"}:
            raise ValueError("unsupported actor Q reduction")
        if config.online_actor_update_interval <= 0:
            raise ValueError("online_actor_update_interval must be positive")
        if config.online_actor_batch_mode not in {"latest", "aggregate"}:
            raise ValueError("online_actor_batch_mode must be 'latest' or 'aggregate'")
        if config.online_update_order not in {
            "critic-before-actor",
            "actor-before-critic",
        }:
            raise ValueError(
                "online_update_order must be 'critic-before-actor' or "
                "'actor-before-critic'"
            )
        if config.offline_update_order not in {
            "critic-before-actor",
            "actor-before-critic",
        }:
            raise ValueError(
                "offline_update_order must be 'critic-before-actor' or "
                "'actor-before-critic'"
            )
        if config.temperature_state_source not in {
            "actor",
            "next-observations",
        }:
            raise ValueError(
                "temperature_state_source must be 'actor' or 'next-observations'"
            )
        if config.offline_temperature_action_samples <= 0:
            raise ValueError("offline_temperature_action_samples must be positive")
        if not 0.0 <= config.mixed_backup_lambda <= 1.0:
            raise ValueError("mixed_backup_lambda must lie in [0, 1]")
        if not 0.0 < config.iql_expectile < 1.0:
            raise ValueError("iql_expectile must lie in (0, 1)")
        if config.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if not math.isfinite(config.reward_scale) or config.reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and positive")
        if config.critic_loss not in {"mse", "huber"}:
            raise ValueError("critic_loss must be 'mse' or 'huber'")
        if not math.isfinite(config.huber_delta) or config.huber_delta <= 0.0:
            raise ValueError("huber_delta must be finite and positive")
        if not math.isfinite(config.target_entropy):
            raise ValueError("target_entropy must be finite")
        if (
            not math.isfinite(config.iql_advantage_temperature)
            or config.iql_advantage_temperature <= 0.0
        ):
            raise ValueError("iql_advantage_temperature must be finite and positive")
        if config.activation not in {"relu", "silu"}:
            raise ValueError("activation must be 'relu' or 'silu'")
        if config.gradient_clip_norm is not None and (
            not math.isfinite(config.gradient_clip_norm)
            or config.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient clip norm must be positive or None")

        self.replay_buffer = replay_buffer
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.diagnostic_rng = np.random.default_rng(
            int(seed) ^ 0x5A17C9E3
        )
        torch.random.default_generator.manual_seed(seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(seed)

        hidden_dims = (config.hidden_dim, config.hidden_dim)
        self.actor = actor_class(
            hidden_dims,
            layer_norm=config.actor_layer_norm,
            activation=config.activation,
        ).to(self.device)
        self.critic = critic_class(
            config.ensemble_size,
            hidden_dims,
            layer_norm=config.critic_layer_norm,
            activation=config.activation,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        needs_value = config.offline_objective == "iql" or config.online_backup == "h2oplus"
        self.value = (
            value_class(
                hidden_dims,
                layer_norm=config.value_layer_norm,
                activation=config.activation,
            ).to(self.device)
            if needs_value
            else None
        )
        self.ratio = ratio_class(
            (config.hidden_dim // 2, config.hidden_dim // 2),
            learning_rate=config.ratio_learning_rate,
            min_weight=config.ratio_min_weight,
            max_weight=config.ratio_max_weight,
            fused_adam=self.device.type == "cuda",
            **(ratio_kwargs or {}),
        ).to(self.device)

        cuda_optimizer = self.device.type == "cuda"
        adam_runtime = {
            "fused": cuda_optimizer,
            "capturable": cuda_optimizer,
        }
        self.optimizer_config = {
            "name": "Adam",
            "cuda_fused": cuda_optimizer,
            "cuda_capturable": cuda_optimizer,
        }
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
            **adam_runtime,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_learning_rate,
            **adam_runtime,
        )
        self.value_optimizer = (
            torch.optim.Adam(
                self.value.parameters(),
                lr=config.value_learning_rate,
                **adam_runtime,
            )
            if self.value is not None
            else None
        )
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(config.initial_temperature), device=self.device)
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=config.alpha_learning_rate,
            **adam_runtime,
        )
        self.total_updates = 0
        self.online_critic_updates = 0
        self._online_actor_state_batches: list[torch.Tensor] = []
        self._online_temperature_state_batches: list[torch.Tensor] = []
        self._last_actor_diagnostics: dict[str, torch.Tensor] = {}
        self.compiled_modules: tuple[str, ...] = ()
        self.compiler_config: dict[str, Any] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def compile_networks(self, *, mode: str = "default") -> list[str]:
        """Compile repeated kernels without changing checkpoint keys.

        CUDA Graph modes expose static output buffers.  A training update calls
        several independently compiled modules before backward, so a later
        module invocation can overwrite an earlier output that autograd still
        owns.  Keep CUDA Graph modes out of this multi-module training path.
        """

        import torch._dynamo as dynamo

        # Actor, value, and ratio heads share the same MLP.forward code object
        # but have distinct shapes. The default cache of eight variants falls
        # back to eager execution during a normal mixed update.
        dynamo_limits: dict[str, int] = {}
        for limit_name in (
            "recompile_limit",
            "cache_size_limit",
            "accumulated_cache_size_limit",
        ):
            if not hasattr(dynamo.config, limit_name):
                continue
            current_limit = int(getattr(dynamo.config, limit_name))
            configured_limit = max(current_limit, 32)
            setattr(dynamo.config, limit_name, configured_limit)
            dynamo_limits[limit_name] = configured_limit

        mode_options = torch._inductor.list_mode_options()
        if mode not in mode_options:
            raise ValueError(
                f"unsupported torch.compile mode {mode!r}; "
                f"expected one of {sorted(mode_options)}"
            )
        if (
            self.device.type == "cuda"
            and mode_options[mode].get("triton.cudagraphs", False)
        ):
            raise ValueError(
                f"torch.compile mode {mode!r} enables CUDA Graph static outputs, "
                "which are unsafe across this trainer's independently compiled "
                "actor, critic, value, and ratio modules; use 'default' or "
                "'max-autotune-no-cudagraphs'"
            )

        triton_backend_hash = None
        if self.device.type == "cuda":
            try:
                from torch.utils._triton import triton_hash_with_backend

                triton_backend_hash = triton_hash_with_backend()
            except Exception as error:
                raise RuntimeError(
                    "CUDA compilation backend preflight failed before training; "
                    f"CC={os.environ.get('CC')!r}, CPATH={os.environ.get('CPATH')!r}"
                ) from error

        modules = {
            "actor_mlp": self.actor.network,
            "critic_ensemble": self.critic.network,
            "target_critic_ensemble": self.target_critic.network,
            "ratio_state_action_mlp": self.ratio.state_action,
            "ratio_joint_mlp": self.ratio.joint,
        }
        for name, owner in (
            ("actor_encoder", self.actor),
            ("critic_encoder", self.critic),
            ("target_critic_encoder", self.target_critic),
        ):
            encoder = getattr(owner, "encoder", None)
            if isinstance(encoder, torch.nn.Module):
                modules[name] = encoder
        ratio_encoder = getattr(self.ratio, "feature_fn", None)
        if isinstance(ratio_encoder, torch.nn.Module):
            modules["ratio_encoder"] = ratio_encoder
        if self.value is not None:
            modules["value_mlp"] = self.value.network
            value_encoder = getattr(self.value, "encoder", None)
            if isinstance(value_encoder, torch.nn.Module):
                modules["value_encoder"] = value_encoder
        unavailable = [name for name, module in modules.items() if not hasattr(module, "compile")]
        if unavailable:
            raise RuntimeError(
                "installed PyTorch cannot compile required modules: "
                + ", ".join(unavailable)
            )
        for module in modules.values():
            module.compile(mode=mode)
        self.compiled_modules = tuple(sorted(modules))
        self.compiler_config = {
            "mode": mode,
            "mode_options": dict(mode_options[mode]),
            "cuda_graphs_enabled": bool(
                mode_options[mode].get("triton.cudagraphs", False)
            ),
            "dynamo_limits": dynamo_limits,
            "triton_backend_hash": triton_backend_hash,
        }
        return list(self.compiled_modules)

    @staticmethod
    def _repeat_features(
        features: torch.Tensor,
        repeats: int,
    ) -> torch.Tensor:
        if repeats <= 0:
            raise ValueError("feature repeat count must be positive")
        return features[:, None, :].expand(-1, repeats, -1).reshape(
            features.shape[0] * repeats, -1
        )

    @staticmethod
    def _expanded_features(
        critic,
        states: torch.Tensor,
        repeats: int,
    ) -> torch.Tensor | None:
        if not hasattr(critic, "encode_states"):
            return None
        features = critic.encode_states(states)
        return TransitH2OTrainer._repeat_features(features, repeats)

    @staticmethod
    def _critic_values(
        critic,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is not None:
            if subset is not None and hasattr(critic, "forward_features_subset"):
                return critic.forward_features_subset(features, actions, subset)
            values = critic.forward_features(features, actions)
        else:
            if subset is not None and hasattr(critic, "forward_subset"):
                return critic.forward_subset(states, actions, subset)
            values = critic(states, actions)
        return values if subset is None else values.index_select(0, subset)

    def act(self, observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observations_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            actions, _ = self.actor(observations_tensor, deterministic=deterministic)
        return actions.cpu().numpy()

    def _sample_ensemble_subset(
        self, size: int, *, replacement: bool
    ) -> torch.Tensor:
        if replacement:
            return torch.randint(
                self.config.ensemble_size,
                (size,),
                device=self.device,
            )
        return torch.randperm(
            self.config.ensemble_size, device=self.device
        )[:size]

    def _sample_target_subset(self) -> torch.Tensor:
        return self._sample_ensemble_subset(
            self.config.target_subset_size,
            replacement=self.config.target_subset_replacement,
        )

    def _sample_cql_penalty_subset(self) -> torch.Tensor | None:
        size = self.config.cql_penalty_subset_size
        if size is None:
            return None
        return self._sample_ensemble_subset(
            size,
            replacement=self.config.cql_penalty_subset_replacement,
        )

    @torch.no_grad()
    def _sac_bootstrap(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if getattr(self.actor, "is_hybrid", False):
            components = self.actor.soft_action_components(batch["next_observations"])
            batch_size = batch["next_observations"].shape[0]
            combined_actions = torch.cat(
                (components["zero_action"], components["positive_action"]),
                dim=0,
            )
            chosen = self._sample_target_subset()
            if hasattr(self.target_critic, "forward_features"):
                features = self.target_critic.encode_states(
                    batch["next_observations"]
                )
                repeated_features = torch.cat((features, features), dim=0)
                combined_q = (
                    self.target_critic.forward_features_subset(
                        repeated_features, combined_actions, chosen
                    )
                    if hasattr(self.target_critic, "forward_features_subset")
                    else self.target_critic.forward_features(
                        repeated_features, combined_actions
                    )[chosen]
                )
            else:
                repeated_states = torch.cat(
                    (batch["next_observations"], batch["next_observations"]),
                    dim=0,
                )
                combined_q = (
                    self.target_critic.forward_subset(
                        repeated_states, combined_actions, chosen
                    )
                    if hasattr(self.target_critic, "forward_subset")
                    else self.target_critic(repeated_states, combined_actions)[chosen]
                )
            zero_q, positive_q = combined_q.split(batch_size, dim=1)
            zero_value = zero_q.min(dim=0).values
            positive_value = positive_q.min(dim=0).values
            if self.config.backup_entropy:
                zero_value = (
                    zero_value
                    - self.alpha.detach() * components["zero_log_prob"]
                )
                positive_value = (
                    positive_value
                    - self.alpha.detach() * components["positive_log_prob"]
                )
            return (
                components["zero_probability"] * zero_value
                + components["hold_probability"] * positive_value
            )
        next_actions, next_log_prob = self.actor(batch["next_observations"])
        chosen = self._sample_target_subset()
        selected_next_q = (
            self.target_critic.forward_subset(
                batch["next_observations"], next_actions, chosen
            )
            if hasattr(self.target_critic, "forward_subset")
            else self.target_critic(batch["next_observations"], next_actions)[chosen]
        )
        bootstrap = selected_next_q.min(dim=0).values
        if self.config.backup_entropy:
            bootstrap = bootstrap - self.alpha.detach() * next_log_prob
        return bootstrap

    @torch.no_grad()
    def _cql_max_bootstrap(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """CQL max-target backup over policy actions.

        The registered WSRL baseline uses this CQL initialization variant.
        Each state receives ``cql_n_actions`` independent policy proposals
        before the best conservative target is selected.
        """

        batch_size = batch["next_observations"].shape[0]
        n_actions = self.config.cql_n_actions
        repeated_next = batch["next_observations"][:, None, :].expand(
            -1, n_actions, -1
        ).reshape(batch_size * n_actions, -1)
        if hasattr(self.actor, "repeated_actions_and_log_prob"):
            next_actions, next_log_prob = self.actor.repeated_actions_and_log_prob(
                batch["next_observations"], n_actions
            )
        else:
            next_actions, next_log_prob = self.actor(repeated_next)
        repeated_features = self._expanded_features(
            self.target_critic, batch["next_observations"], n_actions
        )
        chosen = self._sample_target_subset()
        if repeated_features is not None:
            selected_next_q = (
                self.target_critic.forward_features_subset(
                    repeated_features, next_actions, chosen
                )
                if hasattr(self.target_critic, "forward_features_subset")
                else self.target_critic.forward_features(
                    repeated_features, next_actions
                )[chosen]
            )
        else:
            selected_next_q = (
                self.target_critic.forward_subset(
                    repeated_next, next_actions, chosen
                )
                if hasattr(self.target_critic, "forward_subset")
                else self.target_critic(repeated_next, next_actions)[chosen]
            )
        candidates = selected_next_q.reshape(
            self.config.target_subset_size, batch_size, n_actions
        ).min(dim=0).values
        if self.config.backup_entropy:
            candidates = candidates - self.alpha.detach() * next_log_prob.reshape(
                batch_size, n_actions
            )
        return candidates.max(dim=-1).values

    @torch.no_grad()
    def _target(
        self, batch: dict[str, torch.Tensor], *, backup: str
    ) -> torch.Tensor:
        if backup == "iql":
            if self.value is None:
                raise RuntimeError("IQL backup requires a state-value network")
            bootstrap = self.value(batch["next_observations"])
        elif backup == "sac":
            bootstrap = self._sac_bootstrap(batch)
        elif backup == "cql":
            bootstrap = (
                self._cql_max_bootstrap(batch)
                if self.config.cql_max_target_backup
                else self._sac_bootstrap(batch)
            )
        elif backup == "h2oplus":
            if self.value is None:
                raise RuntimeError("H2O+ backup requires a state-value network")
            mixing = self.config.mixed_backup_lambda
            bootstrap = mixing * self.value(batch["next_observations"])
            bootstrap = bootstrap + (1.0 - mixing) * self._sac_bootstrap(batch)
        else:
            raise ValueError(f"unsupported backup {backup!r}")
        effective_discount = batch["discounts"].reshape(-1) * (
            1.0 - batch["terminals"].reshape(-1)
        )
        return (
            self.config.reward_scale * batch["rewards"].reshape(-1)
            + effective_discount * bootstrap
        )

    def _elementwise_td_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        errors = predictions - targets.unsqueeze(0)
        if self.config.critic_loss == "mse":
            return errors.square()
        absolute = errors.abs()
        delta = self.config.huber_delta
        return torch.where(
            absolute <= delta,
            0.5 * errors.square(),
            delta * (absolute - 0.5 * delta),
        )

    def _ensemble_td_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        elementwise_loss = self._elementwise_td_loss(predictions, targets)
        if weights is None:
            weights = torch.ones_like(targets)
        weighted_error = elementwise_loss * weights.unsqueeze(0)
        if self.config.bootstrap_probability < 1.0:
            mask = (
                torch.rand_like(weighted_error) < self.config.bootstrap_probability
            ).to(weighted_error.dtype)
            denominator = (mask * weights.unsqueeze(0)).sum(dim=1).clamp_min(1.0)
            return ((weighted_error * mask).sum(dim=1) / denominator).mean()
        return weighted_error.mean()

    def _td_diagnostics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        prefix: str,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        absolute_error = (predictions.detach() - targets.detach().unsqueeze(0)).abs()
        per_row_absolute_error = absolute_error.mean(dim=0)
        terminal = batch["terminals"].reshape(-1).detach() > 0.5
        nonterminal = ~terminal
        zero = torch.zeros((), device=predictions.device)
        terminal_absolute = (
            per_row_absolute_error[terminal].mean() if terminal.any() else zero
        )
        nonterminal_absolute = (
            per_row_absolute_error[nonterminal].mean() if nonterminal.any() else zero
        )
        elementwise_loss = self._elementwise_td_loss(
            predictions.detach(), targets.detach()
        )
        if weights is not None:
            elementwise_loss = elementwise_loss * weights.detach().unsqueeze(0)
        total_loss = elementwise_loss.sum()
        terminal_loss = (
            elementwise_loss[:, terminal].sum() if terminal.any() else zero
        )
        terminal_fraction = torch.where(
            total_loss > 0.0, terminal_loss / total_loss, zero
        )
        return {
            f"train/{prefix}_terminal_rows": terminal.sum(),
            f"train/{prefix}_terminal_abs_td": terminal_absolute,
            f"train/{prefix}_nonterminal_abs_td": nonterminal_absolute,
            f"train/{prefix}_terminal_loss_fraction": terminal_fraction,
            f"train/{prefix}_max_abs_td": absolute_error.max(),
        }

    def _cql_penalty(
        self, batch: dict[str, torch.Tensor], data_q: torch.Tensor
    ) -> torch.Tensor:
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        batch_size = observations.shape[0]
        n_actions = self.config.cql_n_actions
        repeated_observations = observations[:, None, :].expand(
            -1, n_actions, -1
        ).reshape(batch_size * n_actions, -1)
        repeated_next = next_observations[:, None, :].expand(
            -1, n_actions, -1
        ).reshape(batch_size * n_actions, -1)
        random_actions = torch.empty(
            batch_size * n_actions, 1, device=self.device
        ).uniform_(-1.0, 1.0)
        penalty_subset = self._sample_cql_penalty_subset()
        penalty_ensemble_size = (
            self.config.ensemble_size
            if penalty_subset is None
            else self.config.cql_penalty_subset_size
        )
        assert penalty_ensemble_size is not None
        penalty_data_q = (
            data_q
            if penalty_subset is None
            else data_q.index_select(0, penalty_subset)
        )
        if getattr(self.actor, "is_hybrid", False):
            with torch.no_grad():
                if hasattr(self.actor, "repeated_soft_action_components"):
                    current_components = self.actor.repeated_soft_action_components(
                        observations, n_actions
                    )
                    next_components = self.actor.repeated_soft_action_components(
                        next_observations, n_actions
                    )
                else:
                    current_components = self.actor.soft_action_components(
                        repeated_observations
                    )
                    next_components = self.actor.soft_action_components(repeated_next)
            candidate_actions = torch.cat(
                (
                    random_actions,
                    current_components["positive_action"],
                    next_components["positive_action"],
                ),
                dim=0,
            )
            if hasattr(self.critic, "forward_features"):
                base_features = self.critic.encode_states(observations)
                repeated_features = self._repeat_features(
                    base_features, n_actions
                )
                candidate_q = self._critic_values(
                    self.critic,
                    torch.cat((repeated_observations,) * 3, dim=0),
                    candidate_actions,
                    features=torch.cat((repeated_features,) * 3, dim=0),
                    subset=penalty_subset,
                )
                zero_q = self._critic_values(
                    self.critic,
                    observations,
                    torch.full_like(batch["actions"], -1.0),
                    features=base_features,
                    subset=penalty_subset,
                ).unsqueeze(-1)
            else:
                candidate_q = self._critic_values(
                    self.critic,
                    torch.cat((repeated_observations,) * 3, dim=0),
                    candidate_actions,
                    subset=penalty_subset,
                )
                zero_q = self._critic_values(
                    self.critic,
                    observations,
                    torch.full_like(batch["actions"], -1.0),
                    subset=penalty_subset,
                ).unsqueeze(-1)
            candidate_q = candidate_q.reshape(
                penalty_ensemble_size, 3, batch_size, n_actions
            )
            random_q, current_q, next_q = candidate_q.unbind(dim=1)
            log_partition = self._hybrid_cql_log_partition(
                zero_q.squeeze(-1),
                (
                    random_q + math.log(2.0),
                    current_q
                    - current_components["positive_conditional_log_prob"].reshape(
                        1, batch_size, n_actions
                    ),
                    next_q
                    - next_components["positive_conditional_log_prob"].reshape(
                        1, batch_size, n_actions
                    ),
                ),
            )
            return (log_partition - penalty_data_q).mean()
        with torch.no_grad():
            current_actions, current_log_prob = self.actor(repeated_observations)
            next_actions, next_log_prob = self.actor(repeated_next)

        candidate_actions = torch.cat(
            (random_actions, current_actions, next_actions), dim=0
        )
        repeated_features = self._expanded_features(
            self.critic, observations, n_actions
        )
        candidate_q = self._critic_values(
            self.critic,
            torch.cat((repeated_observations,) * 3, dim=0),
            candidate_actions,
            features=(
                torch.cat((repeated_features,) * 3, dim=0)
                if repeated_features is not None
                else None
            ),
            subset=penalty_subset,
        ).reshape(penalty_ensemble_size, 3, batch_size, n_actions)
        random_q, current_q, next_q = candidate_q.unbind(dim=1)
        random_log_density = -math.log(2.0)
        candidates = torch.cat(
            (
                random_q - random_log_density,
                current_q - current_log_prob.reshape(1, batch_size, n_actions),
                next_q - next_log_prob.reshape(1, batch_size, n_actions),
            ),
            dim=-1,
        )
        return (torch.logsumexp(candidates, dim=-1) - penalty_data_q).mean()

    @staticmethod
    def _hybrid_cql_log_partition(
        zero_q: torch.Tensor,
        continuous_importance_terms: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Estimate the counting-plus-Lebesgue action partition.

        Each continuous proposal estimates the same integral and is averaged
        before the exact zero-action atom is added.  Normalizing by proposal
        sample count is essential because the atom has unit base-measure mass.
        """

        if not continuous_importance_terms:
            raise ValueError("at least one continuous CQL proposal is required")
        log_integrals = []
        for terms in continuous_importance_terms:
            if terms.ndim != 3 or terms.shape[:2] != zero_q.shape:
                raise ValueError("hybrid CQL proposal tensors have invalid shapes")
            if terms.shape[-1] <= 0:
                raise ValueError("hybrid CQL proposals cannot be empty")
            log_integrals.append(
                torch.logsumexp(terms, dim=-1) - math.log(terms.shape[-1])
            )
        continuous_log_integral = torch.logsumexp(
            torch.stack(log_integrals, dim=-1), dim=-1
        ) - math.log(len(log_integrals))
        return torch.logaddexp(zero_q, continuous_log_integral)

    @staticmethod
    def _float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
        """Move all CUDA scalars in one synchronization point."""

        tensor_keys = [key for key, value in metrics.items() if torch.is_tensor(value)]
        tensor_values = [metrics[key].detach().reshape(()) for key in tensor_keys]
        converted = dict(metrics)
        if tensor_values:
            host_values = torch.stack(tensor_values).cpu().tolist()
            converted.update(zip(tensor_keys, host_values))
        return {key: float(value) for key, value in converted.items()}

    def _set_critic_grad(self, enabled: bool) -> None:
        for parameter in self.critic.parameters():
            parameter.requires_grad_(enabled)

    def _soft_update_target(self) -> None:
        rate = self.config.soft_target_rate
        with torch.no_grad():
            sources = tuple(self.critic.parameters())
            targets = tuple(self.target_critic.parameters())
            torch._foreach_lerp_(targets, sources, rate)

    def _gradient_norm(
        self,
        parameters,
        *,
        collect_metrics: bool,
    ) -> torch.Tensor:
        parameters = tuple(parameters)
        if self.config.gradient_clip_norm is not None:
            return torch.nn.utils.clip_grad_norm_(
                parameters, self.config.gradient_clip_norm
            )
        if not collect_metrics:
            return torch.zeros((), device=self.device)
        gradients = [
            parameter.grad.detach()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            return torch.zeros((), device=self.device)
        norms = torch.stack(
            [torch.linalg.vector_norm(gradient) for gradient in gradients]
        )
        return torch.linalg.vector_norm(norms)

    def _iql_value_terms(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.value is None:
            raise RuntimeError("IQL value update requires a state-value network")
        with torch.no_grad():
            target_q = self.target_critic(
                batch["observations"], batch["actions"]
            ).min(dim=0).values
        value = self.value(batch["observations"])
        advantage = target_q - value
        expectile = self.config.iql_expectile
        weights = torch.where(
            advantage >= 0.0,
            torch.full_like(advantage, expectile),
            torch.full_like(advantage, 1.0 - expectile),
        )
        value_loss = (weights * advantage.square()).mean()
        actor_weights = torch.exp(
            advantage.detach() * self.config.iql_advantage_temperature
        ).clamp_max(self.config.iql_max_weight)
        return value_loss, advantage.detach(), actor_weights

    def _update_value(
        self, value_loss: torch.Tensor, *, collect_metrics: bool = True
    ) -> torch.Tensor:
        if self.value is None or self.value_optimizer is None:
            raise RuntimeError("value optimizer is unavailable")
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_grad_norm = self._gradient_norm(
            self.value.parameters(), collect_metrics=collect_metrics
        )
        self.value_optimizer.step()
        return value_grad_norm

    def _reduce_actor_q(self, actor_q: torch.Tensor) -> torch.Tensor:
        if self.config.actor_q_reduction == "mean-all":
            return actor_q.mean(dim=0)
        if self.config.actor_q_reduction == "min-all":
            return actor_q.min(dim=0).values
        selected = torch.randperm(
            self.config.ensemble_size, device=self.device
        )[: self.config.target_subset_size]
        return actor_q[selected].min(dim=0).values

    def _temperature_log_probability(
        self, states: torch.Tensor, action_samples: int
    ) -> torch.Tensor:
        if getattr(self.actor, "is_hybrid", False):
            if action_samples == 1:
                components = self.actor.soft_action_components(states)
            elif hasattr(self.actor, "repeated_soft_action_components"):
                components = self.actor.repeated_soft_action_components(
                    states, action_samples
                )
            else:
                repeated_states = states[:, None, :].expand(
                    -1, action_samples, -1
                ).reshape(states.shape[0] * action_samples, -1)
                components = self.actor.soft_action_components(repeated_states)
            return (
                components["zero_probability"] * components["zero_log_prob"]
                + components["hold_probability"]
                * components["positive_log_prob"]
            )
        if action_samples == 1:
            _, log_probability = self.actor(states)
            return log_probability
        repeated_states = states[:, None, :].expand(
            -1, action_samples, -1
        ).reshape(states.shape[0] * action_samples, -1)
        _, log_probability = self.actor(repeated_states)
        return log_probability

    def _update_sac_actor(
        self,
        actor_states: torch.Tensor,
        *,
        temperature_states: torch.Tensor | None = None,
        temperature_action_samples: int = 1,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if temperature_states is None:
            temperature_states = actor_states
        self._last_actor_diagnostics = {}
        self._set_critic_grad(False)
        try:
            if getattr(self.actor, "is_hybrid", False):
                components = self.actor.soft_action_components(actor_states)
                batch_size = actor_states.shape[0]
                combined_actions = torch.cat(
                    (components["zero_action"], components["positive_action"]),
                    dim=0,
                )
                if hasattr(self.critic, "forward_features"):
                    features = self.critic.encode_states(actor_states)
                    combined_q = self.critic.forward_features(
                        torch.cat((features, features), dim=0), combined_actions
                    )
                else:
                    combined_q = self.critic(
                        torch.cat((actor_states, actor_states), dim=0),
                        combined_actions,
                    )
                reduced_q = self._reduce_actor_q(combined_q)
                zero_q, positive_q = reduced_q.split(batch_size, dim=0)
                zero_loss = (
                    self.alpha.detach() * components["zero_log_prob"] - zero_q
                )
                positive_loss = (
                    self.alpha.detach() * components["positive_log_prob"]
                    - positive_q
                )
                actor_loss = (
                    components["zero_probability"] * zero_loss
                    + components["hold_probability"] * positive_loss
                ).mean()
                actor_log_prob = (
                    components["zero_probability"] * components["zero_log_prob"]
                    + components["hold_probability"]
                    * components["positive_log_prob"]
                )
                hold_log_prob = (
                    components["positive_log_prob"]
                    - components["positive_conditional_log_prob"]
                )
                gate_entropy = -(
                    components["zero_probability"]
                    * components["zero_log_prob"]
                    + components["hold_probability"] * hold_log_prob
                )
                weighted_positive_entropy = -(
                    components["hold_probability"]
                    * components["positive_conditional_log_prob"]
                )
                self._last_actor_diagnostics = {
                    "train/hybrid_entropy": (-actor_log_prob).mean().detach(),
                    "train/hybrid_gate_entropy": gate_entropy.mean().detach(),
                    "train/hybrid_weighted_positive_entropy": (
                        weighted_positive_entropy.mean().detach()
                    ),
                    "train/hold_probability": (
                        components["hold_probability"].mean().detach()
                    ),
                    "train/positive_action_mean": (
                        components["positive_action"].mean().detach()
                    ),
                }
            else:
                actor_actions, actor_log_prob = self.actor(actor_states)
                actor_q = self.critic(actor_states, actor_actions)
                reduced_q = self._reduce_actor_q(actor_q)
                actor_loss = (
                    self.alpha.detach() * actor_log_prob - reduced_q
                ).mean()
                self._last_actor_diagnostics = {
                    "train/policy_entropy": (-actor_log_prob).mean().detach()
                }
            if (
                self.config.temperature_state_source == "actor"
                and temperature_action_samples == 1
            ):
                temperature_log_prob = actor_log_prob.detach()
            else:
                with torch.no_grad():
                    temperature_log_prob = self._temperature_log_probability(
                        temperature_states, temperature_action_samples
                    )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = self._gradient_norm(
                self.actor.parameters(), collect_metrics=collect_metrics
            )
            self.actor_optimizer.step()
        finally:
            self._set_critic_grad(True)

        alpha_loss = -(
            self.log_alpha
            * (temperature_log_prob + self.config.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return actor_loss, alpha_loss, actor_grad_norm

    def _update_iql_actor(
        self,
        batch: dict[str, torch.Tensor],
        actor_weights: torch.Tensor,
        *,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_actor_diagnostics = {}
        log_probability = self.actor.action_log_prob(
            batch["observations"], batch["actions"]
        )
        actor_loss = -(actor_weights * log_probability).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._gradient_norm(
            self.actor.parameters(), collect_metrics=collect_metrics
        )
        self.actor_optimizer.step()
        return actor_loss, actor_grad_norm

    @staticmethod
    def _batch_states(
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
        *,
        key: str,
    ) -> torch.Tensor:
        batches = [
            batch[key]
            for batch in (offline_batch, online_batch)
            if batch is not None
        ]
        if not batches:
            raise RuntimeError("an update requires at least one replay partition")
        return batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)

    @staticmethod
    def _actor_states(
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        return TransitH2OTrainer._batch_states(
            offline_batch, online_batch, key="observations"
        )

    def _temperature_states(
        self,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        key = (
            "observations"
            if self.config.temperature_state_source == "actor"
            else "next_observations"
        )
        return self._batch_states(offline_batch, online_batch, key=key)

    @staticmethod
    def _concatenate_batches(
        first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if first.keys() != second.keys():
            raise ValueError("replay partitions expose different tensor fields")
        return {
            key: torch.cat((first[key], second[key]), dim=0)
            for key in first
        }

    def _scheduled_online_actor_states(
        self,
        actor_states: torch.Tensor,
        temperature_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        detached_actor = actor_states.detach()
        detached_temperature = temperature_states.detach()
        if self.config.online_actor_batch_mode == "latest":
            self._online_actor_state_batches = [detached_actor]
            self._online_temperature_state_batches = [detached_temperature]
        else:
            self._online_actor_state_batches.append(detached_actor)
            self._online_temperature_state_batches.append(detached_temperature)
        self.online_critic_updates += 1
        if self.online_critic_updates % self.config.online_actor_update_interval:
            return None
        if self.config.online_actor_batch_mode == "aggregate":
            selected_actor = torch.cat(self._online_actor_state_batches, dim=0)
            selected_temperature = torch.cat(
                self._online_temperature_state_batches, dim=0
            )
        else:
            selected_actor = self._online_actor_state_batches[-1]
            selected_temperature = self._online_temperature_state_batches[-1]
        self._online_actor_state_batches.clear()
        self._online_temperature_state_batches.clear()
        return selected_actor, selected_temperature

    def _update_actor_for_batch(
        self,
        *,
        online: bool,
        offline_batch: dict[str, torch.Tensor] | None,
        online_batch: dict[str, torch.Tensor] | None,
        iql_actor_weights: torch.Tensor | None,
        collect_metrics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if not online and self.config.offline_objective == "iql":
            assert offline_batch is not None and iql_actor_weights is not None
            actor_loss, actor_grad_norm = self._update_iql_actor(
                offline_batch,
                iql_actor_weights,
                collect_metrics=collect_metrics,
            )
            return (
                actor_loss,
                torch.zeros((), device=self.device),
                actor_grad_norm,
                1.0,
            )

        actor_states = self._actor_states(offline_batch, online_batch)
        temperature_states = self._temperature_states(
            offline_batch, online_batch
        )
        if online:
            scheduled = self._scheduled_online_actor_states(
                actor_states, temperature_states
            )
            if scheduled is not None:
                actor_states, temperature_states = scheduled
        else:
            scheduled = (actor_states, temperature_states)
        if scheduled is None:
            self._last_actor_diagnostics = {}
            zero = torch.zeros((), device=self.device)
            return zero, zero, zero, 0.0
        actor_loss, alpha_loss, actor_grad_norm = self._update_sac_actor(
            actor_states,
            temperature_states=temperature_states,
            temperature_action_samples=(
                1 if online else self.config.offline_temperature_action_samples
            ),
            collect_metrics=collect_metrics,
        )
        return actor_loss, alpha_loss, actor_grad_norm, 1.0

    def update(
        self,
        *,
        online: bool,
        online_only: bool = False,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        if online_only and not online:
            raise ValueError("online_only requires online=True")
        if online_only and self.config.use_ratio_correction:
            raise ValueError("dynamics-ratio correction requires an offline reference batch")
        if not online and self.config.offline_objective in {"none", "bc"}:
            raise ValueError(
                f"offline update is incompatible with {self.config.offline_objective!r}"
            )
        if online_only:
            online_fraction = 1.0
            offline_batch = None
            online_batch = self.replay_buffer.sample(
                self.config.batch_size, scope="online", rng=self.rng
            )
        else:
            online_fraction = self.config.online_fraction if online else 0.0
            offline_batch, online_batch = self.replay_buffer.sample_mixed(
                self.config.batch_size, online_fraction, rng=self.rng
            )

        metrics: dict[str, torch.Tensor | float] = {}
        online_weights = None
        if online_batch is not None and self.config.use_ratio_correction:
            assert offline_batch is not None
            if self.total_updates % self.config.ratio_update_interval == 0:
                ratio_batch_size = max(self.config.batch_size // 2, 1)
                ratio_real_batch = self.replay_buffer.sample(
                    ratio_batch_size,
                    scope="offline",
                    partition="ratio_train",
                    rng=self.rng,
                )
                ratio_sim_batch = self.replay_buffer.sample(
                    ratio_batch_size,
                    scope="online",
                    partition="ratio_train",
                    rng=self.rng,
                )
                metrics.update(
                    self.ratio.update(
                        ratio_real_batch,
                        ratio_sim_batch,
                        collect_metrics=collect_metrics,
                    )
                )
            online_weights, weight_metrics = self.ratio.weights(
                online_batch, collect_metrics=collect_metrics
            )
            metrics.update(weight_metrics)

        backup = (
            self.config.online_backup
            if online
            else "iql"
            if self.config.offline_objective == "iql"
            else "cql"
            if self.config.offline_objective == "cql"
            else "sac"
        )
        combined_batch = None
        combined_target = None
        if offline_batch is not None and online_batch is not None:
            combined_batch = self._concatenate_batches(
                offline_batch, online_batch
            )
            combined_target = self._target(combined_batch, backup=backup)
            offline_size = offline_batch["observations"].shape[0]
            offline_target = combined_target[:offline_size]
            online_target = combined_target[offline_size:]
        else:
            offline_target = (
                self._target(offline_batch, backup=backup)
                if offline_batch is not None
                else None
            )
            online_target = (
                self._target(online_batch, backup=backup)
                if online_batch is not None
                else None
            )

        value_loss = torch.zeros((), device=self.device)
        value_grad_norm = torch.zeros((), device=self.device)
        value_advantage = torch.zeros((), device=self.device)
        iql_actor_weights = None
        if self.value is not None and offline_batch is not None:
            value_loss, advantages, iql_actor_weights = self._iql_value_terms(
                offline_batch
            )
            value_advantage = advantages.mean()
            value_grad_norm = self._update_value(
                value_loss, collect_metrics=collect_metrics
            )

        combined_q = (
            self.critic(combined_batch["observations"], combined_batch["actions"])
            if combined_batch is not None
            else None
        )
        if offline_batch is None:
            offline_q = None
            offline_td_loss = torch.zeros((), device=self.device)
        else:
            offline_q = (
                combined_q[:, : offline_batch["observations"].shape[0]]
                if combined_q is not None
                else self.critic(
                    offline_batch["observations"], offline_batch["actions"]
                )
            )
            assert offline_target is not None
            offline_td_loss = self._ensemble_td_loss(offline_q, offline_target)
        if online_batch is None:
            online_q = None
            td_loss = offline_td_loss
            sim_td_loss = torch.zeros((), device=self.device)
        else:
            online_q = (
                combined_q[:, offline_batch["observations"].shape[0] :]
                if combined_q is not None and offline_batch is not None
                else self.critic(
                    online_batch["observations"], online_batch["actions"]
                )
            )
            assert online_target is not None
            sim_td_loss = self._ensemble_td_loss(
                online_q, online_target, online_weights
            )
            td_loss = (
                (1.0 - online_fraction) * offline_td_loss
                + online_fraction * sim_td_loss
            )

        cql_alpha = (
            self.config.online_cql_alpha if online else self.config.offline_cql_alpha
        )
        cql_penalty = (
            self._cql_penalty(offline_batch, offline_q)
            if cql_alpha > 0 and offline_batch is not None and offline_q is not None
            else torch.zeros((), device=self.device)
        )
        critic_loss = td_loss + cql_alpha * cql_penalty
        actor_before_critic = (
            online
            and self.config.online_update_order == "actor-before-critic"
        ) or (
            not online
            and self.config.offline_update_order == "actor-before-critic"
        )
        if actor_before_critic:
            actor_loss, alpha_loss, actor_grad_norm, actor_updated = (
                self._update_actor_for_batch(
                    online=online,
                    offline_batch=offline_batch,
                    online_batch=online_batch,
                    iql_actor_weights=iql_actor_weights,
                    collect_metrics=collect_metrics,
                )
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = self._gradient_norm(
            self.critic.parameters(), collect_metrics=collect_metrics
        )
        self.critic_optimizer.step()

        if not actor_before_critic:
            actor_loss, alpha_loss, actor_grad_norm, actor_updated = (
                self._update_actor_for_batch(
                    online=online,
                    offline_batch=offline_batch,
                    online_batch=online_batch,
                    iql_actor_weights=iql_actor_weights,
                    collect_metrics=collect_metrics,
                )
            )

        self._soft_update_target()
        self.total_updates += 1

        if not collect_metrics:
            return {}
        if offline_q is not None and offline_target is not None:
            metrics.update(
                self._td_diagnostics(
                    offline_q,
                    offline_target,
                    offline_batch,
                    prefix="offline",
                )
            )
        if online_q is not None and online_target is not None:
            metrics.update(
                self._td_diagnostics(
                    online_q,
                    online_target,
                    online_batch,
                    prefix="sim",
                    weights=online_weights,
                )
            )
        q_data = offline_q if offline_q is not None else online_q
        assert q_data is not None
        metrics.update(
            {
                "train/update": self.total_updates,
                "train/online_critic_update": self.online_critic_updates,
                "train/critic_loss": critic_loss.detach(),
                "train/offline_td_loss": offline_td_loss.detach(),
                "train/sim_td_loss": sim_td_loss.detach(),
                "train/cql_penalty": cql_penalty.detach(),
                "train/value_loss": value_loss.detach(),
                "train/value_advantage": value_advantage.detach(),
                "train/actor_updated": actor_updated,
                "train/actor_loss": actor_loss.detach(),
                "train/alpha_loss": alpha_loss.detach(),
                "train/alpha": self.alpha.detach(),
                "train/q_data_mean": q_data.detach().mean(),
                "train/critic_grad_norm": critic_grad_norm.detach(),
                "train/value_grad_norm": value_grad_norm.detach(),
                "train/actor_grad_norm": actor_grad_norm.detach(),
            }
        )
        if actor_updated:
            metrics.update(self._last_actor_diagnostics)
        return self._float_metrics(metrics)

    def behavior_clone_step(self, *, collect_metrics: bool = True) -> dict[str, float]:
        """Maximum-likelihood behavior cloning on the immutable offline data."""

        batch = self.replay_buffer.sample(
            self.config.batch_size, scope="offline", rng=self.rng
        )
        log_probability = self.actor.action_log_prob(
            batch["observations"], batch["actions"]
        )
        actor_loss = -log_probability.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._gradient_norm(
            self.actor.parameters(), collect_metrics=collect_metrics
        )
        self.actor_optimizer.step()
        self.total_updates += 1
        if not collect_metrics:
            return {}
        return self._float_metrics(
            {
                "train/update": self.total_updates,
                "train/actor_loss": actor_loss.detach(),
                "train/actor_grad_norm": actor_grad_norm.detach(),
                "train/action_log_probability": log_probability.detach().mean(),
            }
        )

    def pretrain_ratio(self, steps: int) -> dict[str, float]:
        if steps <= 0:
            raise ValueError("ratio pretraining steps must be positive")
        if self.replay_buffer.online_size == 0:
            raise RuntimeError("ratio pretraining requires simulator transitions")
        latest_training_metrics: dict[str, torch.Tensor] = {}
        for step in range(steps):
            real_batch = self.replay_buffer.sample(
                self.config.batch_size,
                scope="offline",
                partition="ratio_train",
                rng=self.rng,
            )
            sim_batch = self.replay_buffer.sample(
                self.config.batch_size,
                scope="online",
                partition="ratio_train",
                rng=self.rng,
            )
            latest_training_metrics = self.ratio.update(
                real_batch,
                sim_batch,
                collect_metrics=step == steps - 1,
            )
        if not self._ratio_validation_available():
            latest_training_metrics["ratio/validation_available"] = 0.0
            return self._float_metrics(latest_training_metrics)
        validation_real = self.replay_buffer.sample(
            self.config.batch_size,
            scope="offline",
            partition="ratio_validation",
            rng=self.diagnostic_rng,
        )
        validation_sim = self.replay_buffer.sample(
            self.config.batch_size,
            scope="online",
            partition="ratio_validation",
            rng=self.diagnostic_rng,
        )
        metrics = self.ratio.evaluate(validation_real, validation_sim)
        metrics["ratio/validation_available"] = 1.0
        return self._float_metrics(metrics)

    def _ratio_validation_available(self) -> bool:
        partition_size = getattr(self.replay_buffer, "partition_size", None)
        if partition_size is None:
            return True
        return bool(
            partition_size("offline", "ratio_validation")
            and partition_size("online", "ratio_validation")
        )

    def evaluate_ratio(self) -> dict[str, float]:
        """Evaluate the dynamics ratio on trajectory-disjoint holdouts."""

        if self.replay_buffer.online_size == 0:
            raise RuntimeError("ratio validation requires simulator transitions")
        if not self._ratio_validation_available():
            return {"ratio/validation_available": 0.0}
        validation_real = self.replay_buffer.sample(
            self.config.batch_size,
            scope="offline",
            partition="ratio_validation",
            rng=self.diagnostic_rng,
        )
        validation_sim = self.replay_buffer.sample(
            self.config.batch_size,
            scope="online",
            partition="ratio_validation",
            rng=self.diagnostic_rng,
        )
        metrics = self.ratio.evaluate(validation_real, validation_sim)
        metrics["ratio/validation_available"] = 1.0
        return self._float_metrics(metrics)

    def save(
        self,
        path: str | Path,
        extra: dict[str, Any] | None = None,
        *,
        include_online_replay: bool = True,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "total_updates": self.total_updates,
            "online_critic_updates": self.online_critic_updates,
            "online_actor_state_batches": [
                states.detach().cpu() for states in self._online_actor_state_batches
            ],
            "online_temperature_state_batches": [
                states.detach().cpu()
                for states in self._online_temperature_state_batches
            ],
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
            "extra": extra or {},
        }
        if include_online_replay:
            payload["online_replay"] = self.replay_buffer.online_state_dict()
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError(
                "checkpoint does not implement the transit-offline-online-v4 contract"
            )
        if payload.get("config") != asdict(self.config):
            raise ValueError("checkpoint trainer configuration does not match this run")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        if self.value is not None:
            if payload.get("value") is None:
                raise ValueError("checkpoint lacks the configured state-value network")
            self.value.load_state_dict(payload["value"])
        elif payload.get("value") is not None:
            raise ValueError("checkpoint unexpectedly contains a state-value network")
        self.ratio.load_state_dict(payload["ratio"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.total_updates = int(payload["total_updates"])
        self.online_critic_updates = int(payload.get("online_critic_updates", 0))
        self._online_actor_state_batches = [
            states.to(self.device) for states in payload.get("online_actor_state_batches", [])
        ]
        self._online_temperature_state_batches = [
            states.to(self.device)
            for states in payload.get("online_temperature_state_batches", [])
        ]
        if len(self._online_actor_state_batches) != len(
            self._online_temperature_state_batches
        ):
            raise ValueError("checkpoint has inconsistent delayed actor state batches")
        self.rng.bit_generator.state = payload["numpy_rng_state"]
        if "diagnostic_rng_state" in payload:
            self.diagnostic_rng.bit_generator.state = payload[
                "diagnostic_rng_state"
            ]
        if "numpy_global_rng_state" in payload:
            np.random.set_state(payload["numpy_global_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_rng_state = payload.get("torch_cuda_rng_state")
        if cuda_rng_state is None:
            legacy_states = payload.get("torch_cuda_rng_state_all")
            if legacy_states:
                logical_index = self.device.index
                if logical_index is None and self.device.type == "cuda":
                    logical_index = torch.cuda.current_device()
                if logical_index is not None and logical_index < len(legacy_states):
                    cuda_rng_state = legacy_states[logical_index]
                elif len(legacy_states) == 1:
                    cuda_rng_state = legacy_states[0]
        if cuda_rng_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state.cpu(), self.device)
        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            if self.value_optimizer is not None:
                if payload.get("value_optimizer") is None:
                    raise ValueError("checkpoint lacks the configured value optimizer")
                self.value_optimizer.load_state_dict(payload["value_optimizer"])
            self.ratio.optimizer.load_state_dict(payload["ratio_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        if "online_replay" in payload:
            self.replay_buffer.load_online_state_dict(payload["online_replay"])
        return dict(payload.get("extra", {}))


def load_actor_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    actor_class=TransitActor,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only the deployable policy and its immutable run metadata."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != TransitH2OTrainer.CHECKPOINT_VERSION:
        raise ValueError("not a canonical transit-offline-online-v4 checkpoint")
    config = TransitTrainerConfig(**payload["config"])
    actor = actor_class(
        (config.hidden_dim, config.hidden_dim),
        layer_norm=config.actor_layer_norm,
        activation=config.activation,
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, dict(payload.get("extra", {}))
