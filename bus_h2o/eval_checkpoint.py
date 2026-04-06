"""
eval_checkpoint.py
==================
Evaluate the checkpoint_episode_39 SAC policy and zero-hold baseline
on BusSimEnv (single 7X line).

Reports cumulative reward for each policy over one full episode.

Usage:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python eval_checkpoint.py
"""

import os
import sys
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── Path setup ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

_H2O_ROOT = os.path.dirname(SCRIPT_DIR)
_COLLECT = os.path.join(_H2O_ROOT, "collect_policy")
sys.path.insert(0, _COLLECT)

# ── Device ──────────────────────────────────────────────────────────
device = torch.device("cpu")

# ── Env ─────────────────────────────────────────────────────────────
from envs.bus_sim_env import BusSimEnv
from sim_core.sim import env_bus
env_bus._DATA_CACHE.clear()

# ── Checkpoint paths ────────────────────────────────────────────────
CHECKPOINT_PREFIX = os.path.join(_COLLECT, "checkpoint_episode_39")


# ══════════════════════════════════════════════════════════════════════
# Network architecture (mirrors collect_worker.py exactly)
# ══════════════════════════════════════════════════════════════════════

class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols, layer_norm=False, dropout=0.0):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)
        self.cardinalities = {}
        modules = {}
        total_dim = 0
        for col in self.cat_cols:
            card = max(cat_code_dict[col].values()) + 1
            self.cardinalities[col] = card
            dim = min(32, max(2, int(round(card ** 0.5)) + 1)) if card > 1 else 1
            modules[col] = nn.Embedding(card, dim)
            total_dim += dim
        self.embeddings = nn.ModuleDict(modules)
        self.output_dim = total_dim
        self.layer_norm = nn.LayerNorm(total_dim) if layer_norm and total_dim > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1: cat_tensor = cat_tensor.unsqueeze(0)
        parts = []
        for idx, col in enumerate(self.cat_cols):
            indices = torch.clamp(cat_tensor[:, idx].long(), 0, self.cardinalities[col] - 1)
            parts.append(self.embeddings[col](indices))
        embed = torch.cat(parts, dim=1) if parts else torch.empty(cat_tensor.size(0), 0)
        if self.layer_norm: embed = self.layer_norm(embed)
        if self.dropout: embed = self.dropout(embed)
        return embed

    def clone(self): return copy.deepcopy(self)


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        cat_t = state[:, :len(self.embedding_layer.cat_cols)]
        num_t = state[:, len(self.embedding_layer.cat_cols):]
        emb = self.embedding_layer(cat_t.long())
        x = torch.cat([emb, num_t], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.mean_linear(x), torch.clamp(self.log_std_linear(x), -20, 2)

    def get_action(self, state, deterministic=True):
        if state.dim() == 1: state = state.unsqueeze(0)
        mean, log_std = self.forward(state.float())
        if deterministic:
            a0 = torch.tanh(mean)
        else:
            a0 = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        return a0.detach().cpu().numpy()[0]


# ══════════════════════════════════════════════════════════════════════
# Load checkpoint policy + normalizer
# ══════════════════════════════════════════════════════════════════════

def load_sac_policy():
    from normalization import Normalization, RunningMeanStd

    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},
        'station_id':  {0: 0},
        'time_period': {0: 0},
        'direction':   {0: 0, 1: 1},
    }
    num_cont_features = 10
    action_dim = 2

    emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
    state_dim = emb.output_dim + num_cont_features + action_dim
    hidden_dim = 48

    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, emb.clone()).to(device)
    policy.load_state_dict(torch.load(CHECKPOINT_PREFIX + "_policy", map_location=device, weights_only=True))
    policy.eval()

    norm_data = torch.load(CHECKPOINT_PREFIX + "_norm", map_location=device, weights_only=False)
    num_cat = len(cat_cols)
    num_num = num_cont_features + action_dim
    running_ms = RunningMeanStd(shape=(num_num,))
    if isinstance(norm_data, dict):
        running_ms.mean = norm_data.get('mean', running_ms.mean)
        running_ms.var = norm_data.get('var', running_ms.var)
        running_ms.count = norm_data.get('count', running_ms.count)
    else:
        running_ms = norm_data.running_ms if hasattr(norm_data, 'running_ms') else running_ms
    state_norm = Normalization(num_categorical=num_cat, num_numerical=num_num, running_ms=running_ms)

    return policy, state_norm


# ══════════════════════════════════════════════════════════════════════
# Obs processing helper (mirrors train_sim.py)
# ══════════════════════════════════════════════════════════════════════

def obs_to_vec(obs_list):
    return np.array(obs_list, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# Run one full episode
# ══════════════════════════════════════════════════════════════════════

def run_episode(env, policy_fn, policy_name="policy"):
    """
    Run one full episode on BusSimEnv.

    policy_fn: callable(obs_15d, bus_id, last_action_dict) -> action_value (float, hold time)

    Returns (cumulative_reward, n_decisions, wall_time).
    """
    t0 = time.time()
    env.reset()
    state_dict, _, _ = env.initialize_state()

    action_dict = {k: None for k in range(env.max_agent_num)}
    pending = {}  # bus_id -> (obs_vec, action_arr)
    last_action = {}  # bus_id -> np.array (for SAC policy that needs last_action)

    done = False
    ep_reward = 0.0
    ep_decisions = 0

    # Seed initial actions
    for bus_id, obs_list in state_dict.items():
        if not obs_list:
            continue
        sv = obs_to_vec(obs_list[-1])
        a_val = policy_fn(sv, bus_id, last_action)
        action_dict[bus_id] = a_val
        pending[bus_id] = (sv, a_val)

    while not done:
        cur_state, reward_dict, done = env.step_to_event(action_dict)

        for k in action_dict:
            action_dict[k] = None

        for bus_id, obs_list in cur_state.items():
            if not obs_list:
                continue
            sv_new = obs_to_vec(obs_list[-1])
            r_raw = float(reward_dict.get(bus_id, 0.0))

            if bus_id in pending:
                sv_old, a_old = pending[bus_id]
                if int(sv_old[2]) != int(sv_new[2]):
                    # Real transition: station changed
                    pending.pop(bus_id)
                    ep_reward += r_raw
                    ep_decisions += 1

                    a_val = policy_fn(sv_new, bus_id, last_action)
                    action_dict[bus_id] = a_val
                    pending[bus_id] = (sv_new, a_val)
            else:
                a_val = policy_fn(sv_new, bus_id, last_action)
                action_dict[bus_id] = a_val
                pending[bus_id] = (sv_new, a_val)

    wall_time = time.time() - t0
    return ep_reward, ep_decisions, wall_time


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Checkpoint Evaluation: SAC Policy vs Zero-Hold Baseline")
    print("=" * 70)

    env = BusSimEnv(path=SCRIPT_DIR)
    print(f"BusSimEnv created: max_agent_num={env.max_agent_num}, "
          f"stations={len(env.stations)}")

    # ── Load SAC policy ──────────────────────────────────────────────
    print("\nLoading SAC checkpoint policy...")
    policy_net, state_norm = load_sac_policy()
    print("  Policy loaded successfully.")

    _sac_last_action = {}  # persistent across calls within an episode

    def sac_policy_fn(obs_15d, bus_id, last_action_dict):
        """SAC policy: obs_15d (15-dim) + last_action (2-dim) → hold time."""
        prev_a = _sac_last_action.get(bus_id, np.zeros(2, dtype=np.float32))
        state_vec = np.concatenate([obs_15d, prev_a])  # 17-dim
        state_vec_norm = state_norm(np.array(state_vec), update=False)
        a = policy_net.get_action(torch.from_numpy(state_vec_norm).float(), deterministic=True)
        # a is in [-1, 1] from tanh → we need to map to [0, 60]
        # The checkpoint uses use_1d_mapping=True: scale=1.0, bias=0.0
        # So a is in [-1, 1]. We need to interpret this.
        # Looking at train_sim.py: ACTION_SCALE=30, ACTION_BIAS=30
        # action = 30 * tanh + 30 → [0, 60]
        hold_time = float(np.clip(30.0 * a[0] + 30.0, 0.0, 60.0))
        _sac_last_action[bus_id] = np.array(a, dtype=np.float32)
        return hold_time

    def zero_policy_fn(obs_15d, bus_id, last_action_dict):
        """Zero-hold baseline: always hold 0 seconds."""
        return 0.0

    # ── Run SAC policy ───────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Running SAC checkpoint policy (1 episode)...")
    _sac_last_action.clear()
    sac_reward, sac_decisions, sac_time = run_episode(env, sac_policy_fn, "sac")
    print(f"  SAC Policy Result:")
    print(f"    Cumulative Reward: {sac_reward:,.1f}")
    print(f"    Decisions:         {sac_decisions}")
    print(f"    Wall Time:         {sac_time:.1f}s")

    # ── Run zero-hold baseline ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("Running zero-hold baseline (1 episode)...")
    zero_reward, zero_decisions, zero_time = run_episode(env, zero_policy_fn, "zero")
    print(f"  Zero-Hold Baseline Result:")
    print(f"    Cumulative Reward: {zero_reward:,.1f}")
    print(f"    Decisions:         {zero_decisions}")
    print(f"    Wall Time:         {zero_time:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  SAC Policy:       {sac_reward:>12,.1f}  ({sac_decisions} decisions, {sac_time:.1f}s)")
    print(f"  Zero-Hold:        {zero_reward:>12,.1f}  ({zero_decisions} decisions, {zero_time:.1f}s)")
    if zero_reward != 0:
        improvement = (sac_reward - zero_reward) / abs(zero_reward) * 100
        print(f"  Improvement:      {improvement:>+11.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
