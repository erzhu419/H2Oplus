"""
eval_sumo_checkpoint.py
=======================
Evaluate checkpoint_episode_39 SAC policy and zero-hold baseline
on the SUMO env (SumoRLBridge), matching collect_worker.py setup.

Reports cumulative reward for each policy over one full episode (18000 steps).

Usage:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/collect_policy
    python eval_sumo_checkpoint.py
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

# ── Path setup (mirrors collect_worker.py exactly) ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUS_H2O = os.path.join(os.path.dirname(_HERE), "bus_h2o")
sys.path.insert(0, _BUS_H2O)

SUMO_DIR = os.path.normpath(os.path.join(
    _BUS_H2O, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
_CASE_DIR = os.path.join(_BUS_H2O, "sumo_env", "case")
if os.path.isdir(_CASE_DIR):
    sys.path.insert(0, _CASE_DIR)
sys.path.insert(0, _HERE)

from sumo_env.rl_bridge import SumoRLBridge
from common.data_utils import (build_edge_linear_map, extract_structured_context, set_route_length)
import xml.etree.ElementTree as _ET

EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
LINE_ID = "7X"
CHECKPOINT_PREFIX = os.path.join(_HERE, "checkpoint_episode_39")
SCHEDULE_XML = os.path.join(SUMO_DIR, "initialize_obj", "save_obj_bus.add.xml")

device = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# Stable SUMO indices (from collect_worker.py)
# ══════════════════════════════════════════════════════════════════════

def _build_sumo_indices(schedule_xml):
    from collections import defaultdict as _dd
    tree = _ET.parse(schedule_xml)
    root = tree.getroot()
    line_deps = _dd(list)
    for elem in root.findall(".//bus_obj"):
        lid = elem.get("belong_line_id_s")
        bid = elem.get("bus_id_s")
        st = float(elem.get("start_time_n", "0"))
        if lid and bid:
            line_deps[lid].append((st, bid))
    for entries in line_deps.values():
        entries.sort(key=lambda p: p[0])
    line_idx = {lid: i for i, lid in enumerate(sorted(line_deps.keys()))}
    bus_idx = {}
    counter = 0
    for lid, deps in line_deps.items():
        for _, bid in deps:
            if bid not in bus_idx:
                bus_idx[bid] = counter
                counter += 1
    return line_idx, bus_idx


_SUMO_LINE_INDEX, _SUMO_BUS_INDEX = {}, {}
if os.path.exists(SCHEDULE_XML):
    _SUMO_LINE_INDEX, _SUMO_BUS_INDEX = _build_sumo_indices(SCHEDULE_XML)
    print(f"Loaded SUMO indices: {len(_SUMO_LINE_INDEX)} lines, {len(_SUMO_BUS_INDEX)} trips")


# ══════════════════════════════════════════════════════════════════════
# Obs / Reward (from collect_worker.py)
# ══════════════════════════════════════════════════════════════════════

_station_index = {}
_time_period_index = {}
_line_headway = {}


def _reset_indices():
    _station_index.clear()
    _time_period_index.clear()


def event_to_obs(ev, headway_fallback=360.0):
    line_id = ev.line_id
    line_idx = _SUMO_LINE_INDEX.get(line_id, 0)
    bus_idx = _SUMO_BUS_INDEX.get(str(ev.bus_id), 0)

    sk = (line_id, ev.stop_id)
    if sk not in _station_index:
        _station_index[sk] = ev.stop_idx if ev.stop_idx is not None and ev.stop_idx >= 0 else len(_station_index)
    station_idx = _station_index[sk]

    tp = int(ev.sim_time // 3600)
    if tp not in _time_period_index:
        _time_period_index[tp] = len(_time_period_index)
    tp_idx = _time_period_index[tp]

    target_hw = _line_headway.get(line_id, headway_fallback)
    dyn_target = getattr(ev, 'target_forward_headway', target_hw)
    gap = (dyn_target - ev.forward_headway) if getattr(ev, 'forward_bus_present', True) else 0.0

    return np.array([
        float(line_idx), float(bus_idx), float(station_idx),
        float(tp_idx), float(int(ev.direction)),
        float(ev.forward_headway), float(ev.backward_headway),
        float(ev.waiting_passengers), float(target_hw),
        float(ev.base_stop_duration), float(ev.sim_time), float(gap),
        float(ev.co_line_forward_headway), float(ev.co_line_backward_headway),
        float(ev.segment_mean_speed),
    ], dtype=np.float32)


def compute_reward(ev, headway_fallback=360.0):
    def hr(hw, t): return -abs(hw - t)
    t_f = getattr(ev, 'target_forward_headway', headway_fallback)
    t_b = getattr(ev, 'target_backward_headway', headway_fallback)
    fp = getattr(ev, 'forward_bus_present', True)
    bp = getattr(ev, 'backward_bus_present', True)
    rf = hr(ev.forward_headway, t_f) if fp else None
    rb = hr(ev.backward_headway, t_b) if bp else None
    if rf is not None and rb is not None:
        fd = abs(ev.forward_headway - t_f)
        bd = abs(ev.backward_headway - t_b)
        w = fd / (fd + bd + 1e-6)
        R = t_f / max(t_b, 1e-6)
        sb = -abs(ev.forward_headway - R * ev.backward_headway) * 0.5 / ((1 + R) / 2)
        reward = rf * w + rb * (1 - w) + sb
    elif rf is not None:
        reward = rf
    elif rb is not None:
        reward = rb
    else:
        return -50.0
    f_pen = (20.0 * np.tanh((abs(ev.forward_headway - t_f) - 0.5 * t_f) / 30.0)
             if fp and t_f > 0 else 0.0)
    b_pen = (20.0 * np.tanh((abs(ev.backward_headway - t_b) - 0.5 * t_b) / 30.0)
             if bp and t_b > 0 else 0.0)
    reward -= max(0.0, f_pen + b_pen)
    return reward


# ══════════════════════════════════════════════════════════════════════
# Network architecture (from collect_worker.py)
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
# SAC Policy Loader
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
# Episode runner (mirrors collect_worker.py run_episode)
# ══════════════════════════════════════════════════════════════════════

def run_episode(bridge, all_edge_maps, line_route_lengths, policy_fn, policy_name="policy"):
    """Run one episode on SUMO, return (cumulative_reward, n_decisions, wall_time)."""
    t0 = time.time()
    _reset_indices()
    bridge.reset()
    _line_headway.update(bridge.line_headways)

    pending = {}
    last_action = {}
    cumulative_reward = 0.0
    n_decisions = 0

    for _ in range(100000):
        events, done, departed = bridge.fetch_events()
        for bus_id in departed:
            pending.pop(bus_id, None)
        if done:
            break
        if not events:
            continue

        for ev in events:
            bus_id = ev.bus_id
            obs = event_to_obs(ev)
            station_idx = int(obs[2])
            reward = compute_reward(ev)

            # Settle pending
            if bus_id in pending:
                prev = pending.pop(bus_id)
                if station_idx != prev["station_idx"]:
                    cumulative_reward += reward
                    n_decisions += 1

            # New action: policy returns raw tanh [-1,1] x 2
            raw_action = policy_fn(ev, obs, bus_id, last_action)

            # ── Original mapping (matches training, NOT residual control) ──
            hold = float(np.clip(30.0 * raw_action[0] + 30.0, 0.0, 60.0))
            a_speed = float(raw_action[1])
            if a_speed > 0.6:    speed = 1.2
            elif a_speed > 0.2:  speed = 1.1
            elif a_speed > -0.2: speed = 1.0
            elif a_speed > -0.6: speed = 0.9
            else:                speed = 0.8

            bridge.apply_action(ev, [hold, speed])  # 2D action

            pending[bus_id] = {
                "station_idx": station_idx,
            }
            last_action[bus_id] = raw_action.copy()  # store raw tanh output for last_action

    wall_time = time.time() - t0
    return cumulative_reward, n_decisions, wall_time


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SUMO Checkpoint Evaluation: SAC Policy vs Zero-Hold Baseline")
    print("=" * 70)

    # Build edge maps
    all_edge_maps = {}
    line_route_lengths = {}
    if os.path.exists(EDGE_XML):
        tree = _ET.parse(EDGE_XML)
        root = tree.getroot()
        for bl in root.findall("busline"):
            lid = bl.get("id")
            all_edge_maps[lid] = build_edge_linear_map(EDGE_XML, lid)
            total_len = sum(float(e.get("length", 0)) for e in bl.findall("element"))
            line_route_lengths[lid] = total_len
        print(f"Built edge_maps for {len(all_edge_maps)} lines")

    route_len = line_route_lengths.get(LINE_ID, 13119.0)
    set_route_length(route_len)

    # Bridge
    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=18000)

    # ── Load SAC policy ──────────────────────────────────────────────
    print("\nLoading SAC checkpoint policy...")
    policy_net, state_norm = load_sac_policy()
    print("  Policy loaded successfully.")

    def sac_policy_fn(ev, obs, bus_id, last_action_dict):
        """Returns raw tanh output [-1,1] x 2. Mapping done in run_episode."""
        prev_a = last_action_dict.get(bus_id, np.zeros(2, dtype=np.float32))
        state_vec = np.concatenate([obs, prev_a])
        state_vec = state_norm(np.array(state_vec), update=False)
        a = policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=True)
        return np.array(a, dtype=np.float32)

    def zero_policy_fn(ev, obs, bus_id, last_action_dict):
        """Zero hold = raw action -1 maps to hold=0, speed maps to 1.0"""
        return np.array([-1.0, 0.0], dtype=np.float32)  # hold→0, speed→1.0

    # ── Run SAC policy ───────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Running SAC checkpoint policy (1 episode, 18000 steps)...")
    sac_reward, sac_decisions, sac_time = run_episode(
        bridge, all_edge_maps, line_route_lengths, sac_policy_fn, "sac"
    )
    print(f"  SAC Policy Result:")
    print(f"    Cumulative Reward: {sac_reward:,.1f}")
    print(f"    Decisions:         {sac_decisions}")
    print(f"    Wall Time:         {sac_time:.1f}s")

    # ── Run zero-hold baseline ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("Running zero-hold baseline (1 episode, 18000 steps)...")
    zero_reward, zero_decisions, zero_time = run_episode(
        bridge, all_edge_maps, line_route_lengths, zero_policy_fn, "zero"
    )
    print(f"  Zero-Hold Baseline Result:")
    print(f"    Cumulative Reward: {zero_reward:,.1f}")
    print(f"    Decisions:         {zero_decisions}")
    print(f"    Wall Time:         {zero_time:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY (SUMO env, 18000 steps)")
    print("=" * 70)
    print(f"  SAC Policy:       {sac_reward:>14,.1f}  ({sac_decisions} decisions, {sac_time:.1f}s)")
    print(f"  Zero-Hold:        {zero_reward:>14,.1f}  ({zero_decisions} decisions, {zero_time:.1f}s)")
    if zero_reward != 0:
        improvement = (sac_reward - zero_reward) / abs(zero_reward) * 100
        print(f"  Improvement:      {improvement:>+13.1f}%")
    print("=" * 70)

    bridge.close()


if __name__ == "__main__":
    main()
