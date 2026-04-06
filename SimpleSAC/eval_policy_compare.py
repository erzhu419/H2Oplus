"""
eval_policy_compare.py
======================
Compare multiple policies on the SIM env to verify that
SUMO-good policy = SIM-good policy.

Tests:
  1. ep39 (best SUMO policy)  — should be best on SIM too
  2. ep0  (untrained policy)  — should be worst on SIM
  3. zero-hold baseline       — reference

If ep0 beats ep39 on SIM, the environments are misaligned.
"""

import os, sys, time, csv
import numpy as np
import torch
from collections import defaultdict

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


# ── Model (same as eval_legacy_checkpoint.py) ──

class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols, emb_dims=None, layer_norm=True, dropout=0.05):
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for col in cat_cols:
            n_cats = max(cat_code_dict[col].values()) + 1
            d = emb_dims.get(col, min(int(n_cats ** 0.5) + 1, 50)) if emb_dims else min(int(n_cats ** 0.5) + 1, 50)
            self.embeddings[col] = nn.Embedding(n_cats, d)
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
        if self.layer_norm: out = self.layer_norm(out)
        if self.dropout: out = self.dropout(out)
        return out


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_cat = len(embedding_layer.cat_cols)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        cat_t = state[:, :self.num_cat]
        num_t = state[:, self.num_cat:]
        emb = self.embedding_layer(cat_t.long())
        x = torch.cat([emb, num_t], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(-20, 2)
        return mean, log_std

    def get_action(self, state, deterministic=True):
        if state.dim() == 1: state = state.unsqueeze(0)
        mean, log_std = self.forward(state.float())
        scale = torch.tensor([30.0, 0.2])
        bias = torch.tensor([30.0, 1.0])
        if deterministic:
            action_0 = torch.tanh(mean)
        else:
            action_0 = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        return (scale * action_0 + bias).detach().numpy()[0]


def load_policy(checkpoint_prefix, cat_cols, cat_code_dict, action_dim=2, hidden_dim=48):
    """Load a legacy SAC policy checkpoint."""
    policy_sd = torch.load(checkpoint_prefix + "_policy", weights_only=True, map_location="cpu")
    emb_dims = {}
    for col in cat_cols:
        key = f"embedding_layer.embeddings.{col}.weight"
        if key in policy_sd:
            emb_dims[col] = policy_sd[key].shape[1]
    emb = EmbeddingLayer(cat_code_dict, cat_cols, emb_dims=emb_dims)
    num_cont = 15 - len(cat_cols) + action_dim
    state_dim = emb.output_dim + num_cont
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, emb)
    policy.load_state_dict(policy_sd)
    policy.eval()

    norm_path = checkpoint_prefix + "_norm"
    if os.path.exists(norm_path):
        state_norm = torch.load(norm_path, weights_only=False, map_location="cpu")
    else:
        running_ms = RunningMeanStd(shape=(num_cont,))
        state_norm = Normalization(num_categorical=len(cat_cols), num_numerical=num_cont, running_ms=running_ms)
    return policy, state_norm


def _map_bang_bang_speed(raw_speed):
    """Discretize speed to 5 tiers (matching SUMO eval)."""
    if raw_speed > 1.12: return 1.2
    if raw_speed > 1.04: return 1.1
    if raw_speed > 0.96: return 1.0
    if raw_speed > 0.88: return 0.9
    return 0.8


def run_eval(env, policy, state_norm, action_dim, csv_path=None, label="policy"):
    """Run one episode on MultiLineEnv and return total reward + per-bus action log."""
    env.reset()
    # Initialize
    actions = {lid: {i: 0.0 for i in range(le.max_agent_num)}
               for lid, le in env.line_map.items()}
    state, reward, done = None, None, False
    for _ in range(10000):
        state, reward, done = env.step(actions)
        if done: break
        if any(obs for bus_dict in state.values() for obs in bus_dict.values() if obs):
            break

    last_action = defaultdict(lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32)))
    pending = {}
    ep_reward = 0.0
    ep_steps = 0
    per_line_reward = defaultdict(float)
    per_line_steps = defaultdict(int)
    action_log = []  # detailed per-decision log

    for ev_idx in range(200000):
        if done: break
        action_dict = {lid: {i: None for i in range(le.max_agent_num)}
                       for lid, le in env.line_map.items()}

        for lid, bus_dict in state.items():
            le = env.line_map[lid]
            for bus_id, obs_list in bus_dict.items():
                if not obs_list: continue
                inner = obs_list[-1]
                if isinstance(inner, list) and inner:
                    if isinstance(inner[0], list): inner = inner[-1]
                    obs_vec = np.array(inner, dtype=np.float32)
                else: continue

                station_idx = int(obs_vec[2]) if len(obs_vec) > 2 else -1
                reward_val = le.reward.get(bus_id, 0.0)

                key = (lid, bus_id)
                if key in pending:
                    prev = pending.pop(key)
                    if station_idx != prev["station_idx"]:
                        ep_reward += reward_val
                        ep_steps += 1
                        per_line_reward[lid] += reward_val
                        per_line_steps[lid] += 1

                prev_a = last_action[lid][bus_id]
                obs_aug = np.concatenate([obs_vec, prev_a])

                if policy is not None:
                    obs_normed = state_norm(obs_aug)
                    with torch.no_grad():
                        action = policy.get_action(
                            torch.from_numpy(obs_normed).float(), deterministic=True
                        )
                    hold_val = float(np.clip(action[0], 0, 60))
                    speed_raw = float(action[1])
                    speed_val = _map_bang_bang_speed(speed_raw)
                else:
                    # Zero-hold baseline
                    hold_val = 0.0
                    speed_val = 1.0
                    action = np.array([0.0, 1.0], dtype=np.float32)

                action_dict[lid][bus_id] = [hold_val, speed_val]
                last_action[lid][bus_id] = action.copy()
                pending[key] = {"station_idx": station_idx}

                # Log action details
                fwd_hw = float(obs_vec[5]) if len(obs_vec) > 5 else 0
                bwd_hw = float(obs_vec[6]) if len(obs_vec) > 6 else 0
                target = float(obs_vec[8]) if len(obs_vec) > 8 else 360
                action_log.append({
                    "event": ev_idx, "line": lid, "bus_id": bus_id,
                    "station": station_idx, "fwd_hw": fwd_hw, "bwd_hw": bwd_hw,
                    "target_hw": target, "hold": hold_val, "speed": speed_val,
                    "reward": reward_val,
                })

        state, reward, done = env.step_to_event(action_dict)

    if csv_path and action_log:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=action_log[0].keys())
            writer.writeheader()
            writer.writerows(action_log)

    return ep_reward, ep_steps, per_line_reward, per_line_steps


def main():
    sim_env_path = os.path.join(_BUS_H2O, "calibrated_env")
    model_dir = os.path.join(_LEGACY, "model",
                              "sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long")
    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},
        'station_id':  {i: i for i in range(1)},
        'time_period': {i: i for i in range(1)},
        'direction':   {0: 0, 1: 1},
    }

    output_dir = os.path.join(_H2O_ROOT, "experiment_output", "policy_compare")
    os.makedirs(output_dir, exist_ok=True)

    env = MultiLineEnv(path=sim_env_path, debug=False)
    print(f"MultiLineEnv: {len(env.line_map)} lines")

    results = {}

    # ── 1. ep39 (best SUMO policy) ──
    print("\n[1/3] Evaluating ep39 (best SUMO policy)...")
    ckpt39 = os.path.join(model_dir, "checkpoint_episode_39")
    if os.path.exists(ckpt39 + "_policy"):
        policy39, norm39 = load_policy(ckpt39, cat_cols, cat_code_dict)
        t0 = time.time()
        r39, s39, lr39, ls39 = run_eval(
            env, policy39, norm39, 2,
            csv_path=os.path.join(output_dir, "ep39_actions.csv"), label="ep39"
        )
        print(f"  ep39: reward={r39:.0f}, steps={s39}, time={time.time()-t0:.1f}s")
        results["ep39"] = r39
    else:
        print(f"  ep39 checkpoint not found at {ckpt39}")

    # ── 2. ep0 (untrained policy) ──
    print("\n[2/3] Evaluating ep0 (early/untrained policy)...")
    ckpt0 = os.path.join(model_dir, "checkpoint_episode_0")
    if os.path.exists(ckpt0 + "_policy"):
        policy0, norm0 = load_policy(ckpt0, cat_cols, cat_code_dict)
        t0 = time.time()
        r0, s0, lr0, ls0 = run_eval(
            env, policy0, norm0, 2,
            csv_path=os.path.join(output_dir, "ep0_actions.csv"), label="ep0"
        )
        print(f"  ep0:  reward={r0:.0f}, steps={s0}, time={time.time()-t0:.1f}s")
        results["ep0"] = r0
    else:
        print(f"  ep0 checkpoint not found at {ckpt0}")

    # ── 3. Zero-hold baseline ──
    print("\n[3/3] Evaluating zero-hold baseline...")
    t0 = time.time()
    rz, sz, lrz, lsz = run_eval(
        env, None, None, 2,
        csv_path=os.path.join(output_dir, "zero_actions.csv"), label="zero"
    )
    print(f"  zero: reward={rz:.0f}, steps={sz}, time={time.time()-t0:.1f}s")
    results["zero"] = rz

    # ── Summary ──
    print(f"\n{'='*60}")
    print("POLICY COMPARISON ON SIM (MultiLineSimEnv)")
    print(f"{'='*60}")
    for name, rew in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "← best" if rew == max(results.values()) else ""
        print(f"  {name:10s}: {rew:12.0f}  {marker}")

    if "ep39" in results and "ep0" in results:
        if results["ep39"] > results["ep0"]:
            print("\n✓ GOOD: ep39 > ep0 on SIM (SUMO-good = SIM-good)")
        else:
            print("\n✗ BAD:  ep39 <= ep0 on SIM (SUMO-good ≠ SIM-good — env misalignment!)")

    print(f"\nAction logs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
