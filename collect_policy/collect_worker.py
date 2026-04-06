"""
collect_worker.py  — 多策略 SUMO offline 数据采集 worker
========================================================
接受 --policy 和 --seed 参数，在单个 SUMO 会话中运行 N 个 episode，
输出 d4rl 标准格式 HDF5 (observations/actions/rewards/next_observations/terminals)
加上 H2O+ 额外字段 (z_t/z_t1/sim_time/snapshot_T1)。

snapshot_T1:
  每条 Transition 在 T1 时刻（decision event 触发时）捕获的全保真系统快照，
  通过 pickle 序列化后存为变长字节 dataset。
  Phase 3 buffer reset 时用 pickle.loads() 反序列化，
  传入 BusSimEnv.restore_full_system_snapshot(snapshot)。

策略:
  zero              无动作 (hold=0)
  random            随机 hold ∈ [0, 60]
  heuristic_best    ac0be9 gap-based deterministic
  heuristic_weak    gap-based stochastic
  sac               加载 checkpoint 的 SAC 策略

Usage:
    python collect_worker.py --policy zero --seed 42 --n_episodes 10
"""

import argparse
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

# ── path setup ─────────────────────────────────────────────────────────
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

# Add normalization.py path
sys.path.insert(0, _HERE)

from sumo_env.rl_bridge import SumoRLBridge                          # noqa
from sumo_env.sumo_snapshot import bridge_to_snapshot                 # noqa
from common.data_utils import (build_edge_linear_map,                 # noqa
                               extract_structured_context,
                               set_route_length)
import pickle
import zlib
import ctypes

EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
LINE_ID  = "7X"
POLICY_DIR = _HERE
CHECKPOINT_PREFIX = os.path.join(POLICY_DIR, "checkpoint_episode_39")
SCHEDULE_XML = os.path.join(SUMO_DIR, "initialize_obj", "save_obj_bus.add.xml")

device = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# Stable SUMO trip/line indices — matches rl_env._initialize_indices()
# ══════════════════════════════════════════════════════════════════════

def _build_sumo_indices(schedule_xml):
    """Pre-build stable bus/line index from SUMO schedule XML.

    Exactly replicates rl_env._load_schedule() + _initialize_indices():
    - line_index: alphabetically sorted line_id → int
    - bus_index:  trip_id string → sequential int (per-line in XML order,
                  within each line sorted by start_time)

    Returns (line_index, bus_index) dicts.
    """
    import xml.etree.ElementTree as _ET
    from collections import defaultdict as _dd

    tree = _ET.parse(schedule_xml)
    root = tree.getroot()

    line_deps = _dd(list)
    for elem in root.findall(".//bus_obj"):
        lid = elem.get("belong_line_id_s")
        bid = elem.get("bus_id_s")
        st  = float(elem.get("start_time_n", "0"))
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


if os.path.exists(SCHEDULE_XML):
    _SUMO_LINE_INDEX, _SUMO_BUS_INDEX = _build_sumo_indices(SCHEDULE_XML)
    print(f"[collect_worker] Loaded stable SUMO indices: "
          f"{len(_SUMO_LINE_INDEX)} lines, {len(_SUMO_BUS_INDEX)} trips", flush=True)
else:
    _SUMO_LINE_INDEX, _SUMO_BUS_INDEX = {}, {}
    print(f"[collect_worker] WARNING: schedule XML not found at {SCHEDULE_XML}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Obs / Reward — 从 collect_data_sumo.py 复制过来的完全一致逻辑
# ══════════════════════════════════════════════════════════════════════
# NOTE: line_index and bus_index are now pre-built from the schedule XML
# (_SUMO_LINE_INDEX, _SUMO_BUS_INDEX) and never reset, ensuring stable
# embeddings across all episodes. Only station/time_period indices are
# per-episode (their ordering is deterministic anyway).
_station_index = {}
_time_period_index = {}
_line_headway = {}


def _reset_indices():
    _station_index.clear(); _time_period_index.clear()


def event_to_obs(ev, headway_fallback=360.0):
    line_id = ev.line_id
    # Stable pre-built indices (never reset, match rl_env / checkpoint)
    line_idx = _SUMO_LINE_INDEX.get(line_id, 0)
    bus_idx  = _SUMO_BUS_INDEX.get(str(ev.bus_id), 0)

    sk = (line_id, ev.stop_id)
    if sk not in _station_index:
        _station_index[sk] = ev.stop_idx if ev.stop_idx is not None and ev.stop_idx >= 0 else len(_station_index)
    station_idx = _station_index[sk]

    tp = int(ev.sim_time // 3600)
    if tp not in _time_period_index: _time_period_index[tp] = len(_time_period_index)
    tp_idx = _time_period_index[tp]

    target_hw = _line_headway.get(line_id, headway_fallback)
    # gap uses DYNAMIC per-pair target (matches rl_env / checkpoint training)
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
    """Reward matching rl_env._compute_reward_linear at commit c9f82e8."""
    def hr(hw, t): return -abs(hw - t)
    t_f = getattr(ev, 'target_forward_headway', headway_fallback)
    t_b = getattr(ev, 'target_backward_headway', headway_fallback)
    fp = getattr(ev, 'forward_bus_present', True)
    bp = getattr(ev, 'backward_bus_present', True)
    rf = hr(ev.forward_headway, t_f) if fp else None
    rb = hr(ev.backward_headway, t_b) if bp else None
    if rf is not None and rb is not None:
        fd = abs(ev.forward_headway - t_f); bd = abs(ev.backward_headway - t_b)
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
    # Smooth tanh penalty for large deviations (matches rl_env training)
    f_pen = (20.0 * np.tanh((abs(ev.forward_headway - t_f) - 0.5 * t_f) / 30.0)
             if fp and t_f > 0 else 0.0)
    b_pen = (20.0 * np.tanh((abs(ev.backward_headway - t_b) - 0.5 * t_b) / 30.0)
             if bp and t_b > 0 else 0.0)
    reward -= max(0.0, f_pen + b_pen)
    return reward


# ══════════════════════════════════════════════════════════════════════
# SAC Policy Loader (mirror sac_ensemble_SUMO_linear_penalty.py arch)
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
        # use_1d_mapping=True: scale=1.0, bias=0.0
        if deterministic:
            a0 = torch.tanh(mean)
        else:
            a0 = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        return a0.detach().cpu().numpy()[0]


def load_sac_policy():
    """Load the SAC policy from checkpoint."""
    from normalization import Normalization, RunningMeanStd

    # Match the exact architecture from the checkpoint
    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},  # Based on torch.Size([389, 21])
        'station_id':  {0: 0},                     # Based on torch.Size([1, 1])
        'time_period': {0: 0},                     # Based on torch.Size([1, 1])
        'direction':   {0: 0, 1: 1},               # Based on torch.Size([2, 2])
    }
    num_cont_features = 10  # obs[5:15] = 10 continuous features
    action_dim = 2          # Based on mean_linear.weight torch.Size([2, 48])

    emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
    state_dim = emb.output_dim + num_cont_features + action_dim  # +action_dim for last_action
    hidden_dim = 48

    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, emb.clone()).to(device)
    policy.load_state_dict(torch.load(CHECKPOINT_PREFIX + "_policy", map_location=device, weights_only=True))
    policy.eval()

    # Load normalizer
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
# Policy callables
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# Original action mapping (matches ensemble checkpoint training)
# ══════════════════════════════════════════════════════════════════════

def _map_raw_to_env(action_raw):
    """Convert raw tanh [-1,1]×2 → env [hold_time, speed_ratio].

    Matches sac_ensemble_SUMO_linear_penalty.py PolicyNetwork.get_action():
        hold  = 30 * tanh(mean) + 30   → [0, 60]s
        speed = 0.2 * tanh(mean) + 1.0 → [0.8, 1.2]

    NOTE: Previously used --use_residual_control mapping (hold = (a+1)*60,
    range [0,120]) which was INCORRECT — caused 2x hold time inflation.
    """
    a_hold = float(action_raw[0])
    a_speed = float(action_raw[1])

    # Original mapping: hold = 30*tanh + 30 → [0, 60]
    hold = float(np.clip(30.0 * a_hold + 30.0, 0.0, 60.0))

    # Bang-Bang deterministic speed mapping (5 tiers)
    if a_speed > 0.6:
        speed = 1.2
    elif a_speed > 0.2:
        speed = 1.1
    elif a_speed > -0.2:
        speed = 1.0
    elif a_speed > -0.6:
        speed = 0.9
    else:
        speed = 0.8

    return hold, speed


def _hold_to_tanh(hold_val):
    """Convert hold time [0, 60] → raw tanh [-1, 1] (inverse of original mapping)."""
    return float(np.clip((hold_val - 30.0) / 30.0, -1.0, 1.0))


def make_policy_fn(policy_name, rng):
    """Return (action_fn, needs_obs) tuple."""

    if policy_name == "zero":
        # Zero hold → tanh = -1.0, speed tanh = 0.0 (maps to 1.0)
        return lambda ev, obs: np.array([-1.0, 0.0], dtype=np.float32), False

    elif policy_name == "random":
        def _rand_fn(ev, obs):
            hold = float(rng.uniform(0, 60))
            return np.array([_hold_to_tanh(hold), 0.0], dtype=np.float32)
        return _rand_fn, False

    elif policy_name == "heuristic_best":
        def fn(ev, obs):
            if getattr(ev, 'forward_bus_present', True) and ev.target_forward_headway > 0:
                if ev.forward_headway > ev.target_forward_headway:
                    return np.array([-1.0, 0.0], dtype=np.float32)  # behind schedule: no hold
                else:
                    gap = ev.target_forward_headway - ev.forward_headway
                    hold = min(60.0, gap)
                    return np.array([_hold_to_tanh(hold), 0.0], dtype=np.float32)
            return np.array([-1.0, 0.0], dtype=np.float32)
        return fn, False

    elif policy_name == "heuristic_weak":
        def fn(ev, obs):
            if getattr(ev, 'forward_bus_present', True) and ev.target_forward_headway > 0:
                if ev.forward_headway > ev.target_forward_headway:
                    return np.array([-1.0, 0.0], dtype=np.float32)
                else:
                    hold = float(rng.uniform(0, 60))
                    return np.array([_hold_to_tanh(hold), 0.0], dtype=np.float32)
            return np.array([-1.0, 0.0], dtype=np.float32)
        return fn, False

    elif policy_name == "sac":
        policy_net, state_norm = load_sac_policy()
        last_action = {}  # bus_id → last action (raw tanh)

        def fn(ev, obs):
            bus_id = ev.bus_id
            prev_a = last_action.get(bus_id, np.zeros(2, dtype=np.float32))
            # Append last_action to obs (matching train_sim: state = cat(obs, last_a))
            state_vec = np.concatenate([obs, prev_a])
            state_vec = state_norm(np.array(state_vec), update=False)
            a = policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=True)
            # a is raw tanh in [-1, 1] for both dims
            action_2d = np.array(a, dtype=np.float32)
            last_action[bus_id] = action_2d.copy()
            return action_2d
        return fn, True

    else:
        raise ValueError(f"Unknown policy: {policy_name}")


# ══════════════════════════════════════════════════════════════════════
# Main collection loop
# ══════════════════════════════════════════════════════════════════════

def run_episode(bridge, all_edge_maps, line_route_lengths, policy_fn, needs_obs, rng,
                flush_fn=None, flush_interval=500):
    """Run one episode, streaming transitions via flush_fn to avoid OOM.

    Snapshots are zlib-compressed in memory (~5-10x smaller) and decompressed
    only during flush.  Every *flush_interval* transitions the buffer is
    written to HDF5 via *flush_fn* and freed.

    Returns:
        int: total transitions produced in this episode.
    """
    _reset_indices()
    bridge.reset()
    _line_headway.update(bridge.line_headways)

    pending = {}
    last_action = {}  # bus_id → np.array([hold, speed])
    buffer = []
    total_flushed = 0
    event_count = 0

    for _ in range(100000):  # safety
        events, done, departed = bridge.fetch_events()
        for bus_id in departed:
            pending.pop(bus_id, None)
        if done: break
        if not events: continue

        # 轻量快照: 提取 z 特征 (30维)
        snap = bridge_to_snapshot(
            bridge,
            all_edge_maps=all_edge_maps,
            line_route_lengths=line_route_lengths,
        )
        z_now = extract_structured_context(snap)

        # 同一 tick 内共享一次快照压缩 (全12线)
        _snap_cache = None  # reset per tick

        for ev in events:
            bus_id = ev.bus_id
            obs = event_to_obs(ev)
            station_idx = int(obs[2])
            reward = compute_reward(ev)

            # ── 结算上一站 ────────────────────────────────────────────
            if bus_id in pending:
                prev = pending.pop(bus_id)
                if station_idx != prev["station_idx"]:
                    buffer.append({
                        "obs":          prev["obs_aug"],
                        "action":       prev["action"],
                        "reward":       np.array([reward], np.float32),
                        "next_obs":     np.concatenate([obs, last_action.get(bus_id, np.zeros(2, np.float32))]),
                        "z_t":          prev["z_t"],
                        "z_t1":         z_now.copy(),
                        "terminal":     np.array([0.0], np.float32),
                        "sim_time":     prev["sim_time"],
                        "raw_snapshot": prev["raw_snapshot"],
                    })
                    # Batch flush to avoid OOM
                    if flush_fn is not None and len(buffer) >= flush_interval:
                        flush_fn(buffer)
                        total_flushed += len(buffer)
                        buffer.clear()

            # ── 新开单 ─────────────────────────────────────────────────
            action = policy_fn(ev, obs)
            hold_env, speed_env = _map_raw_to_env(action)
            bridge.apply_action(ev, hold_env)

            # 同一 tick 内多辆车共享一次快照压缩（全12线, 节省开销）
            if _snap_cache is None:
                _snap_cache = zlib.compress(
                    pickle.dumps(snap, protocol=4), level=1)

            prev_a = last_action.get(bus_id, np.zeros(2, dtype=np.float32))
            obs_aug = np.concatenate([obs, prev_a])  # 15 + 2 = 17

            pending[bus_id] = {
                "obs_aug":      obs_aug.copy(),
                "action":       action.copy(),
                "raw_snapshot":  _snap_cache,
                "z_t":          z_now.copy(),
                "sim_time":     ev.sim_time,
                "station_idx":  station_idx,
            }
            last_action[bus_id] = action.copy()
            event_count += 1

    # Flush remaining buffer
    if buffer:
        if flush_fn is not None:
            flush_fn(buffer)
        total_flushed += len(buffer)
        buffer.clear()

    # Release snapshot references held by unsettled pending entries
    pending.clear()

    return total_flushed


def main(args):
    import gc
    import h5py
    import xml.etree.ElementTree as _ET

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ── Build per-line edge_maps and route_lengths for ALL 12 lines ──
    all_edge_maps = {}    # {line_id: {edge_id: cumulative_dist}}
    line_route_lengths = {}  # {line_id: total_route_length_m}
    if os.path.exists(EDGE_XML):
        tree = _ET.parse(EDGE_XML)
        root = tree.getroot()
        for bl in root.findall("busline"):
            lid = bl.get("id")
            all_edge_maps[lid] = build_edge_linear_map(EDGE_XML, lid)
            total_len = sum(float(e.get("length", 0)) for e in bl.findall("element"))
            line_route_lengths[lid] = total_len
        print(f"[collect_worker] Built edge_maps for {len(all_edge_maps)} lines: "
              f"{sorted(all_edge_maps.keys())}", flush=True)
        print(f"[collect_worker] Route lengths: "
              f"{{{', '.join(f'{k}: {v:.0f}m' for k, v in sorted(line_route_lengths.items()))}}}",
              flush=True)
    else:
        print(f"[collect_worker] WARNING: edge XML not found at {EDGE_XML}", flush=True)

    # Set global ROUTE_LENGTH to 7X (for backward compat fallback)
    route_len = line_route_lengths.get(LINE_ID, 13119.0)
    set_route_length(route_len)

    # Policy
    policy_fn, needs_obs = make_policy_fn(args.policy, rng)

    # Bridge
    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=args.max_steps)

    # ── Streaming HDF5 write with intra-episode batch flush ──
    out_path = args.out or f"datasets/sumo_{args.policy}_seed{args.seed}.h5"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    OBS_DIM = 17  # 15 obs + 2 last_action
    Z_DIM = 30
    ACT_DIM = 2   # hold_time + speed_ratio
    CHUNK_SIZE = 4096  # HDF5 chunk size for resizable datasets
    FLUSH_INTERVAL = 500  # batch-flush every N transitions

    total_transitions = 0

    # Limit h5py chunk cache to 16 MB to prevent unbounded cache growth
    with h5py.File(out_path, "w", rdcc_nbytes=16 * 1024 * 1024, rdcc_nslots=521) as f:
        # Pre-create resizable datasets (maxshape=None means unlimited)
        ds_obs      = f.create_dataset("observations",      shape=(0, OBS_DIM), maxshape=(None, OBS_DIM), chunks=(CHUNK_SIZE, OBS_DIM), dtype=np.float32, compression="gzip")
        ds_act      = f.create_dataset("actions",           shape=(0, ACT_DIM), maxshape=(None, ACT_DIM), chunks=(CHUNK_SIZE, ACT_DIM), dtype=np.float32, compression="gzip")
        ds_rew      = f.create_dataset("rewards",           shape=(0,),         maxshape=(None,),         chunks=(CHUNK_SIZE,),         dtype=np.float32, compression="gzip")
        ds_next_obs = f.create_dataset("next_observations", shape=(0, OBS_DIM), maxshape=(None, OBS_DIM), chunks=(CHUNK_SIZE, OBS_DIM), dtype=np.float32, compression="gzip")
        ds_term     = f.create_dataset("terminals",         shape=(0,),         maxshape=(None,),         chunks=(CHUNK_SIZE,),         dtype=np.float32, compression="gzip")
        ds_timeout  = f.create_dataset("timeouts",          shape=(0,),         maxshape=(None,),         chunks=(CHUNK_SIZE,),         dtype=np.float32, compression="gzip")
        ds_zt       = f.create_dataset("z_t",               shape=(0, Z_DIM),   maxshape=(None, Z_DIM),   chunks=(CHUNK_SIZE, Z_DIM),   dtype=np.float32, compression="gzip")
        ds_zt1      = f.create_dataset("z_t1",              shape=(0, Z_DIM),   maxshape=(None, Z_DIM),   chunks=(CHUNK_SIZE, Z_DIM),   dtype=np.float32, compression="gzip")
        ds_simtime  = f.create_dataset("sim_time",          shape=(0,),         maxshape=(None,),         chunks=(CHUNK_SIZE,),         dtype=np.float64, compression="gzip")

        # Variable-length byte datasets for snapshots
        vlen_dt = h5py.vlen_dtype(np.dtype('uint8'))
        ds_raw_snap = f.create_dataset("raw_snapshot",      shape=(0,), maxshape=(None,), dtype=vlen_dt)

        def _flush_batch(batch):
            """Write a batch of transitions to HDF5, decompress snapshots one-at-a-time."""
            nonlocal total_transitions
            n = len(batch)
            if n == 0:
                return
            offset = total_transitions
            new_size = offset + n

            # Resize all datasets
            for ds in (ds_obs, ds_act, ds_rew, ds_next_obs, ds_term,
                       ds_timeout, ds_zt, ds_zt1, ds_simtime, ds_raw_snap):
                ds.resize(new_size, axis=0)

            # Write numeric arrays (batch-vectorized)
            ds_obs[offset:new_size]      = np.stack([t["obs"]      for t in batch])
            ds_act[offset:new_size]      = np.stack([t["action"]   for t in batch])
            ds_rew[offset:new_size]      = np.concatenate([t["reward"]   for t in batch])
            ds_next_obs[offset:new_size] = np.stack([t["next_obs"] for t in batch])
            ds_term[offset:new_size]     = np.concatenate([t["terminal"] for t in batch])
            ds_timeout[offset:new_size]  = np.zeros(n, dtype=np.float32)
            ds_zt[offset:new_size]       = np.stack([t["z_t"]  for t in batch])
            ds_zt1[offset:new_size]      = np.stack([t["z_t1"] for t in batch])
            ds_simtime[offset:new_size]  = np.array([t["sim_time"] for t in batch])

            # Decompress snapshots one-at-a-time to keep peak memory low
            for i, t in enumerate(batch):
                raw_compressed = t["raw_snapshot"]
                raw_bytes = zlib.decompress(raw_compressed)
                ds_raw_snap[offset + i] = np.frombuffer(raw_bytes, dtype=np.uint8)
                del raw_bytes

            total_transitions = new_size

        for ep in range(args.n_episodes):
            t0 = time.time()
            before_count = total_transitions

            run_episode(bridge, all_edge_maps, line_route_lengths,
                        policy_fn, needs_obs, rng,
                        flush_fn=_flush_batch, flush_interval=FLUSH_INTERVAL)

            ep_n = total_transitions - before_count

            # Aggressive memory release: GC + return freed pages to OS
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            print(f"  [{args.policy}|seed={args.seed}] ep {ep+1}/{args.n_episodes}: "
                  f"{ep_n} transitions ({time.time()-t0:.1f}s), "
                  f"total={total_transitions} [flushed to HDF5]", flush=True)

        # Write final metadata
        f.attrs["policy"]        = args.policy
        f.attrs["seed"]          = args.seed
        f.attrs["n_episodes"]    = args.n_episodes
        f.attrs["n_transitions"] = total_transitions
        f.attrs["source"]        = "real"
        f.attrs["snapshot_fmt"]  = "pickle4_vlen"
        f.attrs["z_version"]     = "per_line_edge_map_v2"

    bridge.close()

    if total_transitions == 0:
        print(f"[{args.policy}|seed={args.seed}] No transitions collected!")
        return

    print(f"[{args.policy}|seed={args.seed}] Saved {total_transitions} transitions → {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True,
                        choices=["zero", "random", "heuristic_best", "heuristic_weak", "sac"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=18000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\n!!! FATAL [{args.policy}|seed={args.seed}]: {e}")
        import traceback; traceback.print_exc()
