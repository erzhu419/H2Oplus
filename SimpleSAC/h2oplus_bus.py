"""
h2oplus_bus.py
==============
Phase 3: H2O+ algorithm adapted for bus sim-to-real.

Key differences from original h2oplus.py:
    1. Uses ZOnlyDiscriminator (z_t, z_t1) instead of d_sa + d_sas.
    2. Buffer sampling returns z_t/z_t1 directly (from BusMixedReplayBuffer).
    3. Action dim is 1 (hold time), obs dim is 15 (SUMO observation).
    4. Dynamics ratio w is computed from z-only discriminator.

Algorithm structure preserved from original:
    - Blended Bellman backup (exploit + explore targets)
    - V-function quantile regression
    - Automatic entropy tuning
    - Soft target network updates
    - Policy loss = alpha * log_pi - Q
"""

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_collections import ConfigDict
from torch import nn

from model import Scalar, soft_target_update

import os, sys
_BUS_H2O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bus_h2o")
if _BUS_H2O not in sys.path:
    sys.path.insert(0, _BUS_H2O)
from common.data_utils import ZOnlyDiscriminator, TransitionDiscriminator, compute_z_importance_weight  # noqa


class H2OPlusBus:
    """H2O+ algorithm for bus sim-to-real, using z-only discriminator."""

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.batch_size = 2048
        config.batch_sim_ratio = 0.5
        config.device = "cuda"
        config.discount = 0.80                    # match SUMO baseline (was 0.99 → Q-value explosion)
        config.quantile = 0.5
        config.exploit_coeff = 0.5
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_policy_entropy = True
        config.target_entropy = -0.21             # match SUMO baseline: -2+ln(30)+ln(0.2) (was 0.0 → alpha explosion)
        config.max_alpha = 0.6                    # clamp alpha (match SUMO baseline, was uncapped → 3.4+)
        config.init_log_alpha = -2.3              # init alpha≈0.1 (match SUMO baseline, was 0.0 → alpha=1.0)
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.vf_lr = 3e-4
        config.discriminator_lr = 1e-4          # slower disc training (was 3e-4)
        config.disc_train_interval = 5          # train disc every N gradient steps (not every step)
        config.noise_std_discriminator = 0.2    # 20% Gaussian noise on real z (was 5%)
        config.noise_std_sim = 0.1              # 10% noise on sim z too (bidirectional)
        config.label_smooth_real = 0.8          # symmetric smoothing → balanced disc
        config.label_smooth_sim = 0.2           # symmetric smoothing (was 0.1 → biased disc)
        config.optimizer_type = "adam"
        config.soft_target_update_rate = 1e-2
        config.target_update_period = 1
        config.temperature = 1
        config.in_sample = True
        config.use_quantile_regression = True
        config.sparse_alpha = 1.0
        config.use_td_target_ratio = True
        config.use_real_blended_target = True
        config.use_sim_blended_target = True
        config.clip_dynamics_ratio_min = 0.1     # raise floor to prevent near-zero weights
        config.clip_dynamics_ratio_max = 5.0     # was 1.0 (allow good sim transitions more weight)
        config.disc_warmup_steps = 5000          # don't apply IS weighting until disc stabilizes
        config.disc_gp_lambda = 10.0             # gradient penalty on discriminator (was 5.0; raised to stabilize)
        config.disc_spectral_norm = True         # apply spectral norm to TransitionDiscriminator layers
        config.weight_reg = 0.01             # Q-network weight regularization (aligned with sac_ensemble_SUMO)
        config.use_gradient_clip = True      # gradient clipping on Q networks
        config.gradient_clip_max_norm = 1.0
        config.reward_scale = 10.0            # per-batch reward normalization scale (matches legacy SAC)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(
        self,
        config,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        vf,
        replay_buffer,
        discriminator=None,
    ):
        self.config = H2OPlusBus.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf
        self.replay_buffer = replay_buffer
        self.quantile = self.config.quantile
        self.exploit_coeff = self.config.exploit_coeff

        # JTT: Priority index for targeted snapshot reset (injected by main)
        self.priority_index = None

        # Z-only discriminator
        if discriminator is None:
            self.discriminator = ZOnlyDiscriminator(context_dim=30)
        else:
            self.discriminator = discriminator

        # Normalisation stats from offline data
        self.mean, self.std = self.replay_buffer.get_mean_std()

        # Global reward normalization stats (fixed from offline data)
        self.reward_mean, self.reward_std = self.replay_buffer.get_reward_stats()

        # Optimizers
        optimizer_class = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.config.qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(), self.config.vf_lr
        )
        self.discriminator_optimizer = optimizer_class(
            self.discriminator.parameters(), self.config.discriminator_lr
        )
        self.disc_criterion = nn.BCEWithLogitsLoss()

        # Entropy tuning
        if self.config.use_automatic_entropy_tuning:
            init_log_alpha = getattr(self.config, 'init_log_alpha', -2.3)
            self.log_alpha = Scalar(init_log_alpha)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(), lr=self.config.policy_lr
            )
        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    # ------------------------------------------------------------------
    # Q-network weight regularization (matches legacy SAC reg_norm)
    # ------------------------------------------------------------------

    def _compute_reg_norm(self, q_net):
        """Compute L2 parameter norm of all Linear layers in a Q-network.

        Legacy formula: sqrt(sum(||W||^2) + sum(||b||^2 except last layer))
        This is subtracted from target Q for pessimism / added to online Q
        for policy to prevent over-pessimism.

        NOTE: wrapped in no_grad to avoid keeping leaf tensors in the
        computation graph across successive train() calls (memory leak fix).
        """
        with torch.no_grad():
            weight_norms = []
            bias_norms = []
            for module in q_net.modules():
                if isinstance(module, nn.Linear):
                    weight_norms.append(torch.norm(module.weight) ** 2)
                    bias_norms.append(torch.norm(module.bias) ** 2)
            if not weight_norms:
                return torch.tensor(0.0, device=self.config.device)
            # Legacy: exclude last bias from bias sum
            reg = torch.sqrt(
                torch.sum(torch.stack(weight_norms))
                + torch.sum(torch.stack(bias_norms[:-1]))
            )
        return reg.detach()

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train(self, batch_size, pretrain_steps=0):
        self._total_steps += 1

        is_pretrain = (self._total_steps <= pretrain_steps)

        if is_pretrain:
            # Pretrain: use full batch for offline data only
            real_batch_size = batch_size
            sim_batch_size = 0
        else:
            real_batch_size = int(batch_size * (1 - self.config.batch_sim_ratio))
            sim_batch_size = int(batch_size * self.config.batch_sim_ratio)

        real_batch = self.replay_buffer.sample(real_batch_size, scope="real")
        if sim_batch_size > 0 and self.replay_buffer.has_online_data():
            sim_batch = self.replay_buffer.sample(sim_batch_size, scope="sim")
        else:
            sim_batch = None

        # Unpack real transitions
        real_observations = real_batch["observations"]
        real_actions = real_batch["actions"]
        real_rewards = real_batch["rewards"].squeeze()
        real_next_observations = real_batch["next_observations"]
        real_dones = real_batch["dones"].squeeze()
        real_z_t = real_batch["z_t"]
        real_z_t1 = real_batch["z_t1"]

        # Global reward normalization (consistent across batches, prevents Q drift)
        real_rewards = self.config.reward_scale * (
            real_rewards - self.reward_mean
        ) / self.reward_std

        if is_pretrain or sim_batch is None:
            # ── Pre-training phase: offline only ──────────────────────
            real_q = torch.min(
                self.qf1(real_observations, real_actions),
                self.qf2(real_observations, real_actions),
            )
            real_v = self.vf(real_observations)
            exp_weight = torch.exp(
                (real_q - real_v.squeeze()) * self.config.temperature
            )
            exp_weight = torch.clamp(exp_weight, max=100)
            real_log_prob = self.policy.log_prob(real_observations, real_actions)
            policy_loss = -(exp_weight.detach() * real_log_prob).mean()

            real_q1_pred = self.qf1(real_observations, real_actions)
            real_q2_pred = self.qf2(real_observations, real_actions)

            real_next_vf_pred = self.vf(real_next_observations).squeeze()
            real_exploit_td_target = (
                real_rewards
                + (1.0 - real_dones) * self.config.discount * real_next_vf_pred
            )

            real_qf1_loss = F.mse_loss(
                real_q1_pred, real_exploit_td_target.detach()
            )
            real_qf2_loss = F.mse_loss(
                real_q2_pred, real_exploit_td_target.detach()
            )
            qf_loss = real_qf1_loss + real_qf2_loss

            metrics = dict(
                training_phase="pretrain",
                mean_real_rewards=real_rewards.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                total_steps=self.total_steps,
                jtt_updated=False,         # JTT never runs in pretrain
                jtt_td_error_mean=0.0,
                jtt_q_disagree_mean=0.0,
                jtt_disc_drift_mean=0.0,
            )
        else:
            # ── Main H2O+ phase: real + sim ───────────────────────────
            sim_observations = sim_batch["observations"]
            sim_actions = sim_batch["actions"]
            sim_rewards = sim_batch["rewards"].squeeze()
            sim_next_observations = sim_batch["next_observations"]
            sim_dones = sim_batch["dones"].squeeze()
            sim_z_t = sim_batch["z_t"]
            sim_z_t1 = sim_batch["z_t1"]

            # Global reward normalization for sim data (same scale as real)
            sim_rewards = self.config.reward_scale * (
                sim_rewards - self.reward_mean
            ) / self.reward_std

            # Train discriminator (every N steps to prevent overfitting)
            if self._total_steps % self.config.disc_train_interval == 0:
                disc_loss = self._train_discriminator(
                    real_z_t, real_z_t1, sim_z_t, sim_z_t1,
                    real_obs=real_observations, real_actions=real_actions,
                    real_next_obs=real_next_observations,
                    sim_obs=sim_observations, sim_actions=sim_actions,
                    sim_next_obs=sim_next_observations,
                )
                self._last_disc_loss = disc_loss
            else:
                disc_loss = getattr(self, '_last_disc_loss', 0.0)

            # Mixed obs for policy
            df_observations = torch.cat(
                [real_observations, sim_observations], dim=0
            )
            df_new_actions, df_log_pi = self.policy(df_observations)
            df_log_pi = df_log_pi.squeeze(-1)  # [N,1] → [N]

            # Entropy tuning
            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -(
                    self.log_alpha()
                    * (df_log_pi + self.config.target_entropy).detach()
                ).mean()
                alpha_raw = self.log_alpha().exp() * self.config.alpha_multiplier
                max_alpha = getattr(self.config, 'max_alpha', 0.6)
                alpha = torch.min(alpha_raw, torch.tensor(max_alpha, device=alpha_raw.device))
            else:
                alpha_loss = df_observations.new_tensor(0.0)
                alpha = df_observations.new_tensor(self.config.alpha_multiplier)

            # ── Policy loss ───────────────────────────────────────────
            # Add reg_norm to Q values for policy (prevents over-pessimism)
            online_reg1 = self._compute_reg_norm(self.qf1)
            online_reg2 = self._compute_reg_norm(self.qf2)
            q_new_actions = torch.min(
                self.qf1(df_observations, df_new_actions) + self.config.weight_reg * online_reg1,
                self.qf2(df_observations, df_new_actions) + self.config.weight_reg * online_reg2,
            )
            policy_loss = (alpha * df_log_pi - q_new_actions).mean()

            # ── Q function loss ───────────────────────────────────────
            real_q1_pred = self.qf1(real_observations, real_actions)
            real_q2_pred = self.qf2(real_observations, real_actions)
            sim_q1_pred = self.qf1(sim_observations, sim_actions)
            sim_q2_pred = self.qf2(sim_observations, sim_actions)

            # Exploration targets
            real_new_next_actions, real_next_log_pi = self.policy(
                real_next_observations
            )
            real_next_log_pi = real_next_log_pi.squeeze(-1)  # [N,1] → [N]
            real_target_q_values = torch.min(
                self.target_qf1(real_next_observations, real_new_next_actions),
                self.target_qf2(real_next_observations, real_new_next_actions),
            )
            sim_new_next_actions, sim_next_log_pi = self.policy(
                sim_next_observations
            )
            sim_next_log_pi = sim_next_log_pi.squeeze(-1)  # [N,1] → [N]
            sim_target_q_values = torch.min(
                self.target_qf1(sim_next_observations, sim_new_next_actions),
                self.target_qf2(sim_next_observations, sim_new_next_actions),
            )

            if self.config.backup_policy_entropy:
                real_target_q_values -= alpha * real_next_log_pi
                sim_target_q_values -= alpha * sim_next_log_pi

            # ── Q-network weight regularization (pessimism on targets) ─
            reg_norm1 = self._compute_reg_norm(self.target_qf1)
            reg_norm2 = self._compute_reg_norm(self.target_qf2)
            real_target_q_values = real_target_q_values - self.config.weight_reg * torch.min(reg_norm1, reg_norm2)
            sim_target_q_values = sim_target_q_values - self.config.weight_reg * torch.min(reg_norm1, reg_norm2)

            real_explore_td_target = (
                real_rewards
                + (1.0 - real_dones) * self.config.discount * real_target_q_values
            )
            sim_explore_td_target = (
                sim_rewards
                + (1.0 - sim_dones) * self.config.discount * sim_target_q_values
            )

            # Exploitation targets
            real_next_vf_pred = self.vf(real_next_observations).squeeze()
            real_exploit_td_target = (
                real_rewards
                + (1.0 - real_dones) * self.config.discount * real_next_vf_pred
            )
            sim_next_vf_pred = self.vf(sim_next_observations).squeeze()
            sim_exploit_td_target = (
                sim_rewards
                + (1.0 - sim_dones) * self.config.discount * sim_next_vf_pred
            )

            # Blended targets
            if self.config.use_real_blended_target:
                real_td_target = (
                    self.exploit_coeff * real_exploit_td_target
                    + (1 - self.exploit_coeff) * real_explore_td_target
                )
            else:
                real_td_target = real_exploit_td_target

            if self.config.use_sim_blended_target:
                sim_td_target = (
                    self.exploit_coeff * sim_exploit_td_target
                    + (1 - self.exploit_coeff) * sim_explore_td_target
                )
            else:
                sim_td_target = sim_explore_td_target

            real_qf1_loss = F.mse_loss(real_q1_pred, real_td_target.detach())
            real_qf2_loss = F.mse_loss(real_q2_pred, real_td_target.detach())

            # ── Dynamics ratio (importance weight from z-only discriminator) ──
            # Warmup: don't apply IS weighting until discriminator has enough
            # data to be meaningful (disc_warmup_steps gradient steps).
            disc_warmup = getattr(self.config, 'disc_warmup_steps', 5000)
            if self.config.use_td_target_ratio and self._total_steps > disc_warmup:
                raw_w = compute_z_importance_weight(
                    self.discriminator, sim_z_t, sim_z_t1,
                    obs=sim_observations, action=sim_actions,
                    next_obs=sim_next_observations,
                ).squeeze()
                sqrt_real_sim_ratio = torch.clamp(
                    raw_w,
                    self.config.clip_dynamics_ratio_min,
                    self.config.clip_dynamics_ratio_max,
                ).sqrt()
            else:
                # During warmup: equal weight for all sim transitions
                sqrt_real_sim_ratio = torch.ones(
                    sim_observations.shape[0],
                    device=self.config.device,
                )

            sim_qf1_loss = F.mse_loss(
                sqrt_real_sim_ratio * sim_q1_pred,
                sqrt_real_sim_ratio * sim_td_target.detach(),
            )
            sim_qf2_loss = F.mse_loss(
                sqrt_real_sim_ratio * sim_q2_pred,
                sqrt_real_sim_ratio * sim_td_target.detach(),
            )

            qf_loss = real_qf1_loss + sim_qf1_loss + real_qf2_loss + sim_qf2_loss

            # ── JTT: Extract fragility signals (per-sample, real batch only) ──
            jtt_updated = False
            jtt_td_err_mean = 0.0
            jtt_q_dis_mean = 0.0
            jtt_dd_mean = 0.0

            if self.priority_index is not None and "_indices" in real_batch:
                with torch.no_grad():
                    # 1. TD Error (absolute Bellman residual, avg of twin Q)
                    jtt_td_1 = torch.abs(
                        real_q1_pred.squeeze() - real_td_target
                    ).cpu().numpy()
                    jtt_td_2 = torch.abs(
                        real_q2_pred.squeeze() - real_td_target
                    ).cpu().numpy()
                    jtt_td_error = 0.5 * (jtt_td_1 + jtt_td_2)

                    # 2. Q Disagreement (epistemic uncertainty)
                    jtt_q_disagree = torch.abs(
                        real_q1_pred.squeeze() - real_q2_pred.squeeze()
                    ).cpu().numpy()

                    # 3. Discriminator Drift (how far real dynamics are from
                    #    what the discriminator expects — high w means "real",
                    #    so drift = 1 - w/w_max)
                    jtt_w_real = compute_z_importance_weight(
                        self.discriminator, real_z_t, real_z_t1,
                        obs=real_observations, action=real_actions,
                        next_obs=real_next_observations,
                    ).squeeze()
                    jtt_disc_drift = (
                        1.0 - jtt_w_real.clamp(0, 10) / 10.0
                    ).cpu().numpy()

                self.priority_index.update(
                    real_batch["_indices"].cpu().numpy(),
                    jtt_td_error,
                    jtt_q_disagree,
                    jtt_disc_drift,
                )
                jtt_updated = True
                jtt_td_err_mean = float(jtt_td_error.mean())
                jtt_q_dis_mean = float(jtt_q_disagree.mean())
                jtt_dd_mean = float(jtt_disc_drift.mean())

            metrics = dict(
                training_phase="main",
                exploit_coeff=self.exploit_coeff,
                sqrt_IS_ratio=sqrt_real_sim_ratio.mean().item(),
                mean_real_rewards=real_rewards.mean().item(),
                mean_sim_rewards=sim_rewards.mean().item(),
                log_pi=df_log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                sim_qf1_loss=sim_qf1_loss.item(),
                sim_qf2_loss=sim_qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                disc_loss=disc_loss,
                total_steps=self.total_steps,
                # ── JTT diagnostics ──
                jtt_updated=jtt_updated,
                jtt_td_error_mean=jtt_td_err_mean,
                jtt_q_disagree_mean=jtt_q_dis_mean,
                jtt_disc_drift_mean=jtt_dd_mean,
                jtt_priority_index_attached=(self.priority_index is not None),
                jtt_indices_in_batch=("_indices" in real_batch),
            )

            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # ── V function loss (always from real data) ───────────────────
        real_vf_pred = self.vf(real_observations).squeeze()
        real_qf_target_pred = torch.min(
            self.target_qf1(real_observations, real_actions),
            self.target_qf2(real_observations, real_actions),
        )
        real_vf_error = real_qf_target_pred - real_vf_pred

        if self.config.use_quantile_regression:
            vf_sign = (real_vf_error < 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (
                1 - self.quantile
            )
            vf_loss = (vf_weight * (real_vf_error ** 2)).mean()
        else:
            sparse_term = real_vf_error / (2 * self.config.sparse_alpha) + 1.0
            vf_sign = (sparse_term > 0).float()
            vf_loss = (
                vf_sign * (sparse_term ** 2) + real_vf_pred / self.config.sparse_alpha
            ).mean()

        # ── Backward passes ───────────────────────────────────────────
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.config.use_gradient_clip:
            nn.utils.clip_grad_norm_(self.qf1.parameters(), self.config.gradient_clip_max_norm)
            nn.utils.clip_grad_norm_(self.qf2.parameters(), self.config.gradient_clip_max_norm)
        self.qf_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(self.config.soft_target_update_rate)

        metrics.update(
            dict(
                vf_loss=vf_loss.item(),
                vf_pred=real_vf_pred.mean().item(),
                vf_error=real_vf_error.mean().item(),
            )
        )
        return metrics

    # ------------------------------------------------------------------
    # Discriminator training
    # ------------------------------------------------------------------

    def _disc_forward(self, obs, action, next_obs, z_t, z_t1):
        """Dispatch discriminator forward based on type."""
        if isinstance(self.discriminator, TransitionDiscriminator):
            return self.discriminator(obs, action, next_obs, z_t, z_t1)
        else:
            return self.discriminator(z_t, z_t1)

    def _train_discriminator(
        self,
        real_z_t, real_z_t1, sim_z_t, sim_z_t1,
        real_obs=None, real_actions=None, real_next_obs=None,
        sim_obs=None, sim_actions=None, sim_next_obs=None,
    ):
        """
        Train discriminator with label smoothing, bidirectional noise,
        and gradient penalty. Supports both ZOnlyDiscriminator and
        TransitionDiscriminator (when obs/action/next_obs are provided).
        """
        use_transition = isinstance(self.discriminator, TransitionDiscriminator)

        # Noise injection on z features
        if self.config.noise_std_discriminator > 0:
            real_z_t_noisy = real_z_t + torch.randn_like(real_z_t) * self.config.noise_std_discriminator
            real_z_t1_noisy = real_z_t1 + torch.randn_like(real_z_t1) * self.config.noise_std_discriminator
        else:
            real_z_t_noisy = real_z_t
            real_z_t1_noisy = real_z_t1

        noise_std_sim = getattr(self.config, 'noise_std_sim', 0.0)
        if noise_std_sim > 0:
            sim_z_t_noisy = sim_z_t + torch.randn_like(sim_z_t) * noise_std_sim
            sim_z_t1_noisy = sim_z_t1 + torch.randn_like(sim_z_t1) * noise_std_sim
        else:
            sim_z_t_noisy = sim_z_t
            sim_z_t1_noisy = sim_z_t1

        # Forward pass
        real_logits = self._disc_forward(
            real_obs, real_actions, real_next_obs,
            real_z_t_noisy, real_z_t1_noisy,
        )
        real_labels = torch.full_like(real_logits, self.config.label_smooth_real)
        loss_real = self.disc_criterion(real_logits, real_labels)

        sim_logits = self._disc_forward(
            sim_obs, sim_actions, sim_next_obs,
            sim_z_t_noisy, sim_z_t1_noisy,
        )
        sim_labels = torch.full_like(sim_logits, self.config.label_smooth_sim)
        loss_sim = self.disc_criterion(sim_logits, sim_labels)

        # Gradient penalty (simplified: on z-features only to avoid complexity)
        gp_lambda = getattr(self.config, 'disc_gp_lambda', 5.0)
        if gp_lambda > 0:
            bs = min(real_z_t_noisy.shape[0], sim_z_t_noisy.shape[0])
            alpha = torch.rand(bs, 1, device=real_z_t.device)
            interp_z_t = (alpha * real_z_t_noisy[:bs] + (1 - alpha) * sim_z_t_noisy[:bs]).requires_grad_(True)
            interp_z_t1 = (alpha * real_z_t1_noisy[:bs] + (1 - alpha) * sim_z_t1_noisy[:bs]).requires_grad_(True)
            if use_transition:
                interp_obs = alpha * real_obs[:bs] + (1 - alpha) * sim_obs[:bs]
                interp_act = alpha * real_actions[:bs] + (1 - alpha) * sim_actions[:bs]
                interp_nobs = alpha * real_next_obs[:bs] + (1 - alpha) * sim_next_obs[:bs]
                interp_logits = self.discriminator(interp_obs, interp_act, interp_nobs, interp_z_t, interp_z_t1)
            else:
                interp_logits = self.discriminator(interp_z_t, interp_z_t1)
            grad_t = torch.autograd.grad(interp_logits.sum(), interp_z_t, create_graph=True)[0]
            grad_t1 = torch.autograd.grad(interp_logits.sum(), interp_z_t1, create_graph=True)[0]
            grad_norm = (grad_t.norm(2, dim=1) + grad_t1.norm(2, dim=1)) / 2.0
            gp_loss = gp_lambda * ((grad_norm - 1.0) ** 2).mean()
        else:
            gp_loss = 0.0

        total_loss = loss_real + loss_sim + gp_loss
        self.discriminator_optimizer.zero_grad()
        total_loss.backward()
        self.discriminator_optimizer.step()

        return total_loss.item()

    def discriminator_evaluate(self):
        """Evaluate discriminator accuracy on held-out samples."""
        real_batch = self.replay_buffer.sample(self.config.batch_size, scope="real")
        sim_batch = self.replay_buffer.sample(self.config.batch_size, scope="sim")

        with torch.no_grad():
            real_logits = self._disc_forward(
                real_batch["observations"], real_batch["actions"],
                real_batch["next_observations"],
                real_batch["z_t"], real_batch["z_t1"],
            )
            sim_logits = self._disc_forward(
                sim_batch["observations"], sim_batch["actions"],
                sim_batch["next_observations"],
                sim_batch["z_t"], sim_batch["z_t1"],
            )
            real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean().item()
            sim_acc = (torch.sigmoid(sim_logits) < 0.5).float().mean().item()

        return real_acc, sim_acc

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.discriminator,
        ]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.in_sample:
            modules.append(self.vf)
        return modules

    @property
    def total_steps(self):
        return self._total_steps
