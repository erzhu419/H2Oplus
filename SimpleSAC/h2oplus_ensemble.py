"""
h2oplus_ensemble.py
===================
H2O+ with RE-SAC-style ensemble Q-networks.

Replaces the original H2O+ architecture:
  OLD: twin-Q (qf1, qf2) + V-function + AWR pretrain
  NEW: ensemble-Q (E=5) + no V-function + mean-std pessimism

Key changes from h2oplus_bus.py:
  1. Single EnsembleCritic (E members) instead of separate qf1/qf2
  2. Independent targets per ensemble member
  3. Policy: mean + β*std pessimism (RE-SAC style)
  4. No V-function, no quantile regression
  5. CQL penalty (optional)
  6. Can load from ensemble offline RL checkpoint as pretrain

Used by h2o+_bus_main.py when --use_ensemble is set.
"""

import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributions import Normal

import os, sys
_BUS_H2O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bus_h2o")
if _BUS_H2O not in sys.path:
    sys.path.insert(0, _BUS_H2O)
from common.data_utils import TransitionDiscriminator, compute_z_importance_weight

from model import Scalar, soft_target_update


class H2OPlusEnsemble:
    """H2O+ with ensemble Q-networks and RE-SAC pessimism."""

    @staticmethod
    def get_default_config(updates=None):
        from ml_collections import ConfigDict
        config = ConfigDict()
        config.batch_size = 2048
        config.batch_sim_ratio = 0.5
        config.device = "cpu"
        config.discount = 0.80
        config.ensemble_size = 5
        config.beta = -2.0              # pessimism: mean + β*std (negative = pessimistic)
        config.beta_ood = 0.01          # OOD Q-std penalty in critic loss
        config.beta_bc = 0.005          # behavior cloning regularization
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.target_entropy = -0.21
        config.max_alpha = 0.6
        config.init_log_alpha = -2.3
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = "adam"
        config.soft_target_update_rate = 1e-2
        config.target_update_period = 1
        config.critic_actor_ratio = 3   # more critic updates per actor update
        config.use_gradient_clip = True
        config.gradient_clip_max_norm = 1.0
        config.reward_scale = 10.0
        # Discriminator
        config.discriminator_lr = 1e-4
        config.disc_train_interval = 5
        config.noise_std_discriminator = 0.2
        config.noise_std_sim = 0.1
        config.label_smooth_real = 0.8
        config.label_smooth_sim = 0.2
        config.disc_gp_lambda = 10.0
        config.disc_warmup_steps = 5000
        config.clip_dynamics_ratio_min = 0.1
        config.clip_dynamics_ratio_max = 5.0
        config.use_td_target_ratio = True
        # CQL (optional)
        config.use_cql = False
        config.cql_alpha = 5.0
        config.cql_n_actions = 10

        if updates is not None:
            config.update(updates)
        return config

    def __init__(self, config, policy, qf, target_qf, replay_buffer, discriminator=None):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
        self.replay_buffer = replay_buffer
        self.E = self.qf.E

        self.discriminator = discriminator
        self.priority_index = None  # JTT support

        # Reward normalization stats
        self.reward_mean, self.reward_std = self.replay_buffer.get_reward_stats()

        # Optimizers
        self.qf_optimizer = optim.Adam(self.qf.parameters(), self.config.qf_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), self.config.policy_lr)

        if discriminator is not None:
            self.disc_optimizer = optim.Adam(discriminator.parameters(), self.config.discriminator_lr)
        self.disc_criterion = nn.BCEWithLogitsLoss()

        # Entropy tuning
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(self.config.init_log_alpha)
            self.alpha_optimizer = optim.Adam(self.log_alpha.parameters(), lr=self.config.policy_lr)
        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    @property
    def total_steps(self):
        return self._total_steps

    def train(self, batch_size, pretrain_steps=0):
        self._total_steps += 1
        is_pretrain = (self._total_steps <= pretrain_steps)

        if is_pretrain:
            real_batch_size = batch_size
            sim_batch_size = 0
        else:
            real_batch_size = int(batch_size * (1 - self.config.batch_sim_ratio))
            sim_batch_size = int(batch_size * self.config.batch_sim_ratio)

        real_batch = self.replay_buffer.sample(real_batch_size, scope="real")
        sim_batch = None
        if sim_batch_size > 0 and self.replay_buffer.has_online_data():
            sim_batch = self.replay_buffer.sample(sim_batch_size, scope="sim")

        # Unpack real
        S = real_batch["observations"]
        A = real_batch["actions"]
        R = real_batch["rewards"].squeeze()
        S2 = real_batch["next_observations"]
        D = real_batch.get("terminals", real_batch.get("dones", torch.zeros_like(R))).squeeze()
        real_z_t = real_batch["z_t"]
        real_z_t1 = real_batch["z_t1"]

        # Reward normalization
        R = self.config.reward_scale * (R - self.reward_mean) / (self.reward_std + 1e-6)

        alpha = min(self.config.max_alpha, self.log_alpha().exp().item()) if self.log_alpha else 0.1

        if sim_batch is not None:
            sim_S = sim_batch["observations"]
            sim_A = sim_batch["actions"]
            sim_R = sim_batch["rewards"].squeeze()
            sim_S2 = sim_batch["next_observations"]
            sim_D = sim_batch.get("terminals", sim_batch.get("dones", torch.zeros_like(sim_R))).squeeze()
            sim_z_t = sim_batch["z_t"]
            sim_z_t1 = sim_batch["z_t1"]
            sim_R = self.config.reward_scale * (sim_R - self.reward_mean) / (self.reward_std + 1e-6)

            # Train discriminator
            if self._total_steps % self.config.disc_train_interval == 0 and self.discriminator is not None:
                disc_loss = self._train_discriminator(
                    real_z_t, real_z_t1, sim_z_t, sim_z_t1,
                    S, A, S2, sim_S, sim_A, sim_S2,
                )
                self._last_disc_loss = disc_loss
            disc_loss_val = getattr(self, '_last_disc_loss', 0.0)

            # IS weights for sim data
            if self.config.use_td_target_ratio and self._total_steps > self.config.disc_warmup_steps and self.discriminator is not None:
                raw_w = compute_z_importance_weight(
                    self.discriminator, sim_z_t, sim_z_t1,
                    obs=sim_S, action=sim_A, next_obs=sim_S2,
                ).squeeze()
                sqrt_w = torch.clamp(raw_w, self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max).sqrt()
            else:
                sqrt_w = torch.ones(sim_S.shape[0], device=self.config.device)
        else:
            disc_loss_val = 0.0

        # ── Critic update ──────────────────────────────────────────
        with torch.no_grad():
            a2, lp2 = self.policy(S2, deterministic=False)
            q_next = self.target_qf(S2, a2)  # (E, B)
            lp2_e = lp2.squeeze(-1).unsqueeze(0).expand(self.E, -1)
            td_target_real = R.unsqueeze(0) + (1 - D.unsqueeze(0)) * self.config.discount * (q_next - alpha * lp2_e)

        q_pred_real = self.qf(S, A)  # (E, B)
        qf_loss = F.mse_loss(q_pred_real, td_target_real)

        # OOD penalty
        ood_loss = q_pred_real.std(0).mean()
        total_q_loss = qf_loss + self.config.beta_ood * ood_loss

        # Sim Q-loss (if available)
        sim_qf_loss = torch.tensor(0.0)
        if sim_batch is not None:
            with torch.no_grad():
                sim_a2, sim_lp2 = self.policy(sim_S2, deterministic=False)
                sim_q_next = self.target_qf(sim_S2, sim_a2)
                sim_lp2_e = sim_lp2.squeeze(-1).unsqueeze(0).expand(self.E, -1)
                td_target_sim = sim_R.unsqueeze(0) + (1 - sim_D.unsqueeze(0)) * self.config.discount * (sim_q_next - alpha * sim_lp2_e)

            sim_q_pred = self.qf(sim_S, sim_A)
            # IS-weighted sim loss
            w_expanded = sqrt_w.unsqueeze(0).expand(self.E, -1)
            sim_qf_loss = F.mse_loss(w_expanded * sim_q_pred, w_expanded * td_target_sim)
            total_q_loss = total_q_loss + sim_qf_loss

        # CQL penalty (optional)
        cql_loss = torch.tensor(0.0)
        if self.config.use_cql:
            B = S.shape[0]
            n_act = self.config.cql_n_actions
            rand_a = torch.FloatTensor(B * n_act, A.shape[1]).uniform_(-1, 1).to(self.config.device)
            S_rep = S.unsqueeze(1).repeat(1, n_act, 1).view(B * n_act, -1)
            q_rand = self.qf(S_rep, rand_a).view(self.E, B, n_act)
            logsumexp_q = torch.logsumexp(q_rand, dim=2)
            cql_loss = (logsumexp_q - q_pred_real).mean()
            total_q_loss = total_q_loss + self.config.cql_alpha * cql_loss

        self.qf_optimizer.zero_grad()
        total_q_loss.backward()
        if self.config.use_gradient_clip:
            nn.utils.clip_grad_norm_(self.qf.parameters(), self.config.gradient_clip_max_norm)
        self.qf_optimizer.step()

        # ── Policy update (every critic_actor_ratio steps) ─────────
        pi_loss_val = 0.0
        bc_loss_val = 0.0
        all_S = torch.cat([S, sim_S], 0) if sim_batch is not None else S
        all_A = torch.cat([A, sim_A], 0) if sim_batch is not None else A

        if self._total_steps % self.config.critic_actor_ratio == 0:
            a_new, lp_new = self.policy(all_S, deterministic=False)
            lp_new = lp_new.squeeze(-1)
            q_ens = self.qf(all_S, a_new)  # (E, B)
            q_pessimistic = q_ens.mean(0) + self.config.beta * q_ens.std(0)
            pi_loss = (alpha * lp_new - q_pessimistic).mean()

            bc_loss = F.mse_loss(a_new, all_A.detach())
            total_pi_loss = pi_loss + self.config.beta_bc * bc_loss

            self.policy_optimizer.zero_grad()
            total_pi_loss.backward()
            self.policy_optimizer.step()
            pi_loss_val = pi_loss.item()
            bc_loss_val = bc_loss.item()

        # ── Alpha update ───────────────────────────────────────────
        if self.config.use_automatic_entropy_tuning:
            _, lp = self.policy(all_S, deterministic=False)
            lp = lp.squeeze(-1)
            al_loss = -(self.log_alpha() * (lp + self.config.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            al_loss.backward()
            self.alpha_optimizer.step()

        # ── Target network update ──────────────────────────────────
        if self._total_steps % self.config.target_update_period == 0:
            self.update_target_network(self.config.soft_target_update_rate)

        metrics = dict(
            training_phase="pretrain" if is_pretrain else "main",
            policy_loss=pi_loss_val,
            real_qf1_loss=qf_loss.item(),  # named qf1 for compatibility with logging
            sim_qf1_loss=sim_qf_loss.item(),
            alpha=alpha,
            alpha_loss=al_loss.item() if self.config.use_automatic_entropy_tuning else 0,
            disc_loss=disc_loss_val,
            sqrt_IS_ratio=sqrt_w.mean().item() if sim_batch is not None else 1.0,
            log_pi=lp_new.mean().item() if self._total_steps % self.config.critic_actor_ratio == 0 else 0,
            total_steps=self.total_steps,
            mean_real_rewards=R.mean().item(),
            q_mean=q_pred_real.mean().item(),
            q_std=q_pred_real.std(0).mean().item(),
            cql_loss=cql_loss.item() if self.config.use_cql else 0,
        )
        return metrics

    def _train_discriminator(self, real_z_t, real_z_t1, sim_z_t, sim_z_t1,
                              real_obs, real_act, real_nobs, sim_obs, sim_act, sim_nobs):
        use_transition = isinstance(self.discriminator, TransitionDiscriminator)

        # Noise
        if self.config.noise_std_discriminator > 0:
            real_z_t_n = real_z_t + torch.randn_like(real_z_t) * self.config.noise_std_discriminator
            real_z_t1_n = real_z_t1 + torch.randn_like(real_z_t1) * self.config.noise_std_discriminator
        else:
            real_z_t_n, real_z_t1_n = real_z_t, real_z_t1

        if self.config.noise_std_sim > 0:
            sim_z_t_n = sim_z_t + torch.randn_like(sim_z_t) * self.config.noise_std_sim
            sim_z_t1_n = sim_z_t1 + torch.randn_like(sim_z_t1) * self.config.noise_std_sim
        else:
            sim_z_t_n, sim_z_t1_n = sim_z_t, sim_z_t1

        if use_transition:
            real_logits = self.discriminator(real_obs, real_act, real_nobs, real_z_t_n, real_z_t1_n)
            sim_logits = self.discriminator(sim_obs, sim_act, sim_nobs, sim_z_t_n, sim_z_t1_n)
        else:
            real_logits = self.discriminator(real_z_t_n, real_z_t1_n)
            sim_logits = self.discriminator(sim_z_t_n, sim_z_t1_n)

        loss_real = self.disc_criterion(real_logits, torch.full_like(real_logits, self.config.label_smooth_real))
        loss_sim = self.disc_criterion(sim_logits, torch.full_like(sim_logits, self.config.label_smooth_sim))

        total_loss = loss_real + loss_sim
        self.disc_optimizer.zero_grad()
        total_loss.backward()
        self.disc_optimizer.step()
        return total_loss.item()

    def discriminator_evaluate(self):
        real_batch = self.replay_buffer.sample(self.config.batch_size, scope="real")
        sim_batch = self.replay_buffer.sample(self.config.batch_size, scope="sim")
        with torch.no_grad():
            if isinstance(self.discriminator, TransitionDiscriminator):
                real_l = self.discriminator(real_batch["observations"], real_batch["actions"],
                                            real_batch["next_observations"], real_batch["z_t"], real_batch["z_t1"])
                sim_l = self.discriminator(sim_batch["observations"], sim_batch["actions"],
                                           sim_batch["next_observations"], sim_batch["z_t"], sim_batch["z_t1"])
            else:
                real_l = self.discriminator(real_batch["z_t"], real_batch["z_t1"])
                sim_l = self.discriminator(sim_batch["z_t"], sim_batch["z_t1"])
            real_acc = (torch.sigmoid(real_l) > 0.5).float().mean().item()
            sim_acc = (torch.sigmoid(sim_l) < 0.5).float().mean().item()
        return real_acc, sim_acc

    def update_target_network(self, tau):
        soft_target_update(self.qf, self.target_qf, tau)

    def torch_to_device(self, device):
        for m in [self.policy, self.qf, self.target_qf]:
            m.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)
        if self.log_alpha is not None:
            self.log_alpha.to(device)

    def save_checkpoint(self, path, epoch, variant=None):
        torch.save({
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'qf_state_dict': self.qf.state_dict(),
            'target_qf_state_dict': self.target_qf.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict() if self.discriminator else None,
            'log_alpha': self.log_alpha.state_dict() if self.log_alpha else None,
            'variant': variant,
        }, path)
