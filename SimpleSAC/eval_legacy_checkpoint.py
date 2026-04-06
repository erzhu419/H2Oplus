"""
eval_legacy_checkpoint.py
=========================
Evaluate legacy SAC checkpoint (ep39) on the H2O+ sim_core MultiLineEnv.

Controls ALL 12 lines with the same policy (matching SUMO eval_ep39.py).
Uses the ORIGINAL action mapping (scale=30, bias=30), NOT residual control.
"""

import os
import sys
import time
import numpy as np
import torch
from collections import defaultdict

# ── Path setup ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
_LEGACY = os.path.join(os.path.dirname(_H2O_ROOT), "LSTM-RL-legacy", "ensemble_version")
_LEGACY_ROOT = os.path.join(os.path.dirname(_H2O_ROOT), "LSTM-RL-legacy")

sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)
sys.path.insert(0, _LEGACY)
sys.path.insert(0, _LEGACY_ROOT)

from sim_core.sim import MultiLineEnv
from normalization import Normalization, RewardScaling, RunningMeanStd

import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Model definitions (matching checkpoint architecture)
# ═══════════════════════════════════════════════════════════════════

class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols, emb_dims=None, layer_norm=True, dropout=0.05):
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for col in cat_cols:
            n_cats = max(cat_code_dict[col].values()) + 1
            if emb_dims and col in emb_dims:
                d = emb_dims[col]
            else:
                d = min(int(n_cats ** 0.5) + 1, 50)
            self.embeddings[col] = nn.Embedding(n_cats, d, padding_idx=None)
            total_dim += d
        self.output_dim = total_dim
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x_cat):
        embs = []
        for i, col in enumerate(self.cat_cols):
            idx = x_cat[:, i].long().clamp(0, self.embeddings[col].num_embeddings - 1)
            embs.append(self.embeddings[col](idx))
        out = torch.cat(embs, dim=-1)
        if self.layer_norm:
            out = self.layer_norm(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer,
                 action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_cat = len(embedding_layer.cat_cols)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        cat_tensor = state[:, :self.num_cat]
        num_tensor = state[:, self.num_cat:]
        embedding = self.embedding_layer(cat_tensor.long())
        x = torch.cat([embedding, num_tensor], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        """Original action mapping: hold=[0,60], speed=[0.8,1.2]"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.float()
        mean, log_std = self.forward(state)
        std = log_std.exp()
        scale = torch.tensor([30.0, 0.2])
        bias = torch.tensor([30.0, 1.0])
        if deterministic:
            action_0 = torch.tanh(mean)
        else:
            z = torch.randn_like(mean)
            action_0 = torch.tanh(mean + std * z)
        action = scale * action_0 + bias
        return action.detach().numpy()[0]


def _infer_emb_dims_from_checkpoint(state_dict, cat_cols):
    emb_dims = {}
    for col in cat_cols:
        key = f"embedding_layer.embeddings.{col}.weight"
        if key in state_dict:
            emb_dims[col] = state_dict[key].shape[1]
    return emb_dims


# ═══════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════

def main():
    checkpoint_path = os.path.join(_LEGACY, "best model", "checkpoint_episode_39")
    sim_env_path = os.path.join(_BUS_H2O, "calibrated_env")
    action_dim = 2
    hidden_dim = 48
    DETERMINISTIC = False

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sim env: {sim_env_path}")
    print(f"DETERMINISTIC: {DETERMINISTIC}")
    print(f"Mode: ALL 12 LINES (matching SUMO eval_ep39.py)")

    # ── Create environment (MultiLineEnv: all 12 lines) ──
    print("\nCreating MultiLineEnv (all lines)...")
    env = MultiLineEnv(path=sim_env_path, debug=False)
    print(f"  Lines: {list(env.line_map.keys())}")
    for lid, le in env.line_map.items():
        print(f"    {lid}: {len(le.timetables)} trips, max_agent={le.max_agent_num}")

    # ── Build model ──
    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},
        'station_id':  {i: i for i in range(1)},
        'time_period': {i: i for i in range(1)},
        'direction':   {0: 0, 1: 1},
    }
    policy_sd = torch.load(checkpoint_path + "_policy", weights_only=True, map_location="cpu")
    emb_dims = _infer_emb_dims_from_checkpoint(policy_sd, cat_cols)
    print(f"\nEmbedding dims: {emb_dims}")

    embedding = EmbeddingLayer(cat_code_dict, cat_cols, emb_dims=emb_dims, layer_norm=True, dropout=0.05)
    num_cont = 15 - len(cat_cols) + action_dim  # 12
    state_dim = embedding.output_dim + num_cont
    print(f"state_dim={state_dim} (emb={embedding.output_dim} + cont={num_cont})")

    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding)
    policy.load_state_dict(policy_sd)
    policy.eval()
    print("Loaded policy checkpoint ✓")

    norm_path = checkpoint_path + "_norm"
    if os.path.exists(norm_path):
        state_norm = torch.load(norm_path, weights_only=False, map_location="cpu")
        print("Loaded state normalization ✓")
    else:
        running_ms = RunningMeanStd(shape=(num_cont,))
        state_norm = Normalization(num_categorical=len(cat_cols), num_numerical=num_cont, running_ms=running_ms)
        print("WARNING: No norm file found, using default")

    # ── Run evaluation ──
    print(f"\n{'='*60}")
    print("Starting eval on MultiLineEnv (ALL 12 LINES)")
    print(f"{'='*60}")
    t0 = time.time()

    env.reset()
    # Initialize: step until at least one line has bus obs
    actions = {lid: {i: 0.0 for i in range(le.max_agent_num)}
               for lid, le in env.line_map.items()}
    for _ in range(10000):
        state, reward, done = env.step(actions)
        if done:
            break
        has_obs = False
        for lid, bus_dict in state.items():
            for bid, obs_list in bus_dict.items():
                if obs_list:
                    has_obs = True
                    break
            if has_obs:
                break
        if has_obs:
            break

    # Per-bus last_action tracking: {line_id: {bus_id: np.array}}
    last_action = defaultdict(lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32)))
    # Pending transitions: {(line_id, bus_id): {"station_idx": int}}
    pending = {}

    episode_reward = 0
    episode_steps = 0
    per_line_reward = defaultdict(float)
    per_line_steps = defaultdict(int)
    hold_values = []
    speed_values = []

    for ev_idx in range(200000):  # safety limit
        if done:
            break

        # Build action dict for ALL lines
        action_dict = {lid: {i: None for i in range(le.max_agent_num)}
                       for lid, le in env.line_map.items()}

        # Process all lines
        for lid, bus_dict in state.items():
            le = env.line_map[lid]
            for bus_id, obs_list in bus_dict.items():
                if not obs_list:
                    continue
                # Extract last obs vector
                inner = obs_list[-1]
                if isinstance(inner, list) and inner:
                    if isinstance(inner[0], list):
                        inner = inner[-1]
                    obs_vec = np.array(inner, dtype=np.float32)
                else:
                    continue

                station_idx = int(obs_vec[2]) if len(obs_vec) > 2 else -1
                reward_val = le.reward.get(bus_id, 0.0)

                # Settle pending transition
                key = (lid, bus_id)
                if key in pending:
                    prev = pending.pop(key)
                    if station_idx != prev["station_idx"]:
                        episode_reward += reward_val
                        episode_steps += 1
                        per_line_reward[lid] += reward_val
                        per_line_steps[lid] += 1

                # Augment obs with last action
                prev_a = last_action[lid][bus_id]
                obs_aug = np.concatenate([obs_vec, prev_a])

                # Normalize & get action
                obs_normed = state_norm(obs_aug)
                with torch.no_grad():
                    action = policy.get_action(
                        torch.from_numpy(obs_normed).float(),
                        deterministic=DETERMINISTIC
                    )

                hold_val = float(action[0])
                speed_val = float(action[1])
                hold_values.append(hold_val)
                speed_values.append(speed_val)

                action_dict[lid][bus_id] = [hold_val, speed_val]
                last_action[lid][bus_id] = action.copy()
                pending[key] = {"station_idx": station_idx}

        # Step to next decision event (any line)
        state, reward, done = env.step_to_event(action_dict)

    duration = time.time() - t0
    hold_arr = np.array(hold_values) if hold_values else np.array([0.0])
    speed_arr = np.array(speed_values) if speed_values else np.array([0.0])

    print(f"\n{'='*60}")
    print(f"RESULTS - MultiLineEnv ALL 12 LINES (Original Mapping)")
    print(f"{'='*60}")
    print(f"Total episode reward:  {episode_reward:.0f}")
    print(f"Total decision steps:  {episode_steps}")
    print(f"Duration:              {duration:.1f}s")
    print(f"Hold:  mean={hold_arr.mean():.2f}, std={hold_arr.std():.2f}, range=[{hold_arr.min():.2f}, {hold_arr.max():.2f}]")
    print(f"Speed: mean={speed_arr.mean():.2f}, std={speed_arr.std():.2f}, range=[{speed_arr.min():.2f}, {speed_arr.max():.2f}]")
    print(f"\n--- Per-line breakdown ---")
    for lid in sorted(per_line_reward.keys()):
        print(f"  {lid:6s}: reward={per_line_reward[lid]:10.0f}, steps={per_line_steps[lid]:5d}")
    print(f"  {'TOTAL':6s}: reward={episode_reward:10.0f}, steps={episode_steps:5d}")
    print(f"\nReference (SUMO, original mapping): ~-691K (12 lines, ~12000 steps)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
