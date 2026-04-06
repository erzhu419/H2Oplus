"""
eval_offline_on_sumo.py
=======================
Evaluate the H2O+ offline-trained policy on SUMO (SumoRLBridge).
Compares: offline_RL policy vs ep39 (reference) vs zero-hold baseline.

Usage:
    cd H2Oplus/SimpleSAC
    conda run -n LSTM-RL python eval_offline_on_sumo.py
"""

import os, sys, time, copy, math, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as _ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
_COLLECT = os.path.join(_H2O_ROOT, "collect_policy")
_LEGACY = os.path.join(os.path.dirname(_H2O_ROOT), "LSTM-RL-legacy", "ensemble_version")
_LEGACY_ROOT = os.path.join(os.path.dirname(_H2O_ROOT), "LSTM-RL-legacy")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)
sys.path.insert(0, _LEGACY)
sys.path.insert(0, _LEGACY_ROOT)

SUMO_DIR = os.path.normpath(os.path.join(
    _BUS_H2O, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))

from sumo_env.rl_bridge import SumoRLBridge
from common.data_utils import build_edge_linear_map, set_route_length
from normalization import Normalization, RunningMeanStd

EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
SCHEDULE_XML = os.path.join(SUMO_DIR, "initialize_obj", "save_obj_bus.add.xml")
device = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════
# SUMO indices (same as collect_worker.py)
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# Obs / Reward
# ══════════════════════════════════════════════════════════════════

_station_index = {}
_time_period_index = {}
_line_headway = {}

def _reset_indices():
    _station_index.clear()
    _time_period_index.clear()

def event_to_obs(ev, line_idx_map, bus_idx_map, headway_fallback=360.0):
    line_idx = line_idx_map.get(ev.line_id, 0)
    bus_idx = bus_idx_map.get(str(ev.bus_id), 0)
    sk = (ev.line_id, ev.stop_id)
    if sk not in _station_index:
        _station_index[sk] = ev.stop_idx if ev.stop_idx is not None and ev.stop_idx >= 0 else len(_station_index)
    station_idx = _station_index[sk]
    tp = int(ev.sim_time // 3600)
    if tp not in _time_period_index:
        _time_period_index[tp] = len(_time_period_index)
    tp_idx = _time_period_index[tp]
    target_hw = _line_headway.get(ev.line_id, headway_fallback)
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
        fd, bd = abs(ev.forward_headway - t_f), abs(ev.backward_headway - t_b)
        w = fd / (fd + bd + 1e-6)
        R = t_f / max(t_b, 1e-6)
        sb = -abs(ev.forward_headway - R * ev.backward_headway) * 0.5 / ((1 + R) / 2)
        reward = rf * w + rb * (1 - w) + sb
    elif rf is not None: reward = rf
    elif rb is not None: reward = rb
    else: return -50.0
    f_pen = 20.0 * np.tanh((abs(ev.forward_headway - t_f) - 0.5 * t_f) / 30.0) if fp and t_f > 0 else 0.0
    b_pen = 20.0 * np.tanh((abs(ev.backward_headway - t_b) - 0.5 * t_b) / 30.0) if bp and t_b > 0 else 0.0
    reward -= max(0.0, f_pen + b_pen)
    return reward


# ══════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════

def run_episode(bridge, line_idx_map, bus_idx_map, policy_fn, name="policy"):
    t0 = time.time()
    _reset_indices()
    bridge.reset()
    _line_headway.update(bridge.line_headways)

    pending, last_action = {}, {}
    cum_reward, n_dec = 0.0, 0

    for _ in range(100000):
        events, done, departed = bridge.fetch_events()
        for bid in departed:
            pending.pop(bid, None)
        if done: break
        if not events: continue

        for ev in events:
            bid = ev.bus_id
            obs = event_to_obs(ev, line_idx_map, bus_idx_map)
            si = int(obs[2])
            rew = compute_reward(ev)

            if bid in pending:
                prev = pending.pop(bid)
                if si != prev["si"]:
                    cum_reward += rew
                    n_dec += 1

            raw_action = policy_fn(ev, obs, bid, last_action)
            hold = float(np.clip(30.0 * raw_action[0] + 30.0, 0.0, 60.0))
            a_sp = float(raw_action[1])
            speed = 1.2 if a_sp > 0.6 else 1.1 if a_sp > 0.2 else 1.0 if a_sp > -0.2 else 0.9 if a_sp > -0.6 else 0.8

            bridge.apply_action(ev, [hold, speed])
            pending[bid] = {"si": si}
            last_action[bid] = raw_action.copy()

    return cum_reward, n_dec, time.time() - t0


# ══════════════════════════════════════════════════════════════════
# Policy loaders
# ══════════════════════════════════════════════════════════════════

def load_offline_rl_policy(ckpt_path=None):
    """Load offline RL policy. Auto-detects AWR vs Ensemble format."""
    from model import EmbeddingLayer, BusEmbeddingPolicy

    # Try ensemble first, then AWR
    ensemble_path = os.path.join(_H2O_ROOT, "experiment_output", "offline_ensemble", "offline_ensemble_final.pt")
    awr_path = os.path.join(_H2O_ROOT, "experiment_output", "offline_only", "offline_final.pt")

    if ckpt_path is None:
        ckpt_path = ensemble_path if os.path.exists(ensemble_path) else awr_path

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},
        'station_id':  {i: i for i in range(1)},
        'time_period': {i: i for i in range(1)},
        'direction':   {0: 0, 1: 1},
    }

    N_CAT = len(cat_cols)
    emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
    state_dim = emb.output_dim + (17 - N_CAT)

    # Detect format: ensemble PolicyNet has 'net.0.weight' shape, BusEmbeddingPolicy has 'linear1.weight'
    policy_sd = ckpt['policy']
    is_ensemble = any('net.' in k for k in policy_sd.keys())

    if is_ensemble:
        from train_offline_ensemble import PolicyNet
        hidden = ckpt.get('args', {}).get('hidden_dim', 48)
        policy = PolicyNet(state_dim, hidden, copy.deepcopy(emb))
        policy.load_state_dict(policy_sd)
        print(f"  Loaded ENSEMBLE offline policy from {ckpt_path} (step {ckpt.get('step', '?')})")
    else:
        policy = BusEmbeddingPolicy(
            num_inputs=state_dim, num_actions=2,
            hidden_size=48, embedding_layer=emb.clone(), action_range=1.0,
        )
        policy.load_state_dict(policy_sd)
        print(f"  Loaded AWR offline policy from {ckpt_path} (step {ckpt.get('step', '?')})")

    policy.eval()
    return policy


def load_ep39_policy():
    """Load legacy ep39 checkpoint."""
    from eval_legacy_checkpoint import EmbeddingLayer as LEL, PolicyNetwork, _infer_emb_dims_from_checkpoint

    ckpt_prefix = os.path.join(_LEGACY, "best model", "checkpoint_episode_39")
    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},
        'station_id':  {i: i for i in range(1)},
        'time_period': {i: i for i in range(1)},
        'direction':   {0: 0, 1: 1},
    }
    policy_sd = torch.load(ckpt_prefix + "_policy", weights_only=True, map_location=device)
    emb_dims = _infer_emb_dims_from_checkpoint(policy_sd, cat_cols)
    emb = LEL(cat_code_dict, cat_cols, emb_dims=emb_dims, layer_norm=True, dropout=0.05)
    state_dim = emb.output_dim + 12  # 10 cont + 2 last_action
    policy = PolicyNetwork(state_dim, 2, 48, emb)
    policy.load_state_dict(policy_sd)
    policy.eval()

    norm = torch.load(ckpt_prefix + "_norm", weights_only=False, map_location=device)
    print(f"  Loaded ep39 policy from {ckpt_prefix}")
    return policy, norm


def main():
    print("=" * 70)
    print("SUMO Evaluation: Offline RL vs ep39 vs Zero-Hold")
    print("=" * 70)

    # Setup
    if os.path.exists(SCHEDULE_XML):
        line_idx_map, bus_idx_map = _build_sumo_indices(SCHEDULE_XML)
        print(f"SUMO indices: {len(line_idx_map)} lines, {len(bus_idx_map)} trips")
    else:
        line_idx_map, bus_idx_map = {}, {}

    if os.path.exists(EDGE_XML):
        em = build_edge_linear_map(EDGE_XML, "7X")
        set_route_length(max(em.values()) if em else 13119.0)

    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=18000)

    results = {}

    # ── 1. Offline RL policy ──
    print("\n[1/3] Offline RL policy...")
    offline_policy = load_offline_rl_policy()
    def offline_fn(ev, obs, bid, last_act):
        prev_a = last_act.get(bid, np.zeros(2, dtype=np.float32))
        obs_aug = np.concatenate([obs, prev_a])  # 15 + 2 = 17
        with torch.no_grad():
            if hasattr(offline_policy, 'get_action'):
                # Ensemble PolicyNet: get_action returns tanh(mean) as numpy
                return offline_policy.get_action(obs_aug, deterministic=True)
            else:
                # BusEmbeddingPolicy: forward(s, deterministic=True) returns (tanh_action, log_prob)
                s = torch.FloatTensor(obs_aug).unsqueeze(0)
                action, _ = offline_policy(s, deterministic=True)
                return action.cpu().numpy()[0]
    r, n, t = run_episode(bridge, line_idx_map, bus_idx_map, offline_fn, "offline_rl")
    print(f"  offline_rl: reward={r:.0f}, decisions={n}, time={t:.1f}s")
    results["offline_rl"] = r

    # ── 2. ep39 (reference best) ──
    print("\n[2/3] ep39 (SUMO best)...")
    ep39_policy, ep39_norm = load_ep39_policy()
    def ep39_fn(ev, obs, bid, last_act):
        prev_a = last_act.get(bid, np.zeros(2, dtype=np.float32))
        obs_aug = np.concatenate([obs, prev_a])
        obs_normed = ep39_norm(obs_aug, update=False)
        with torch.no_grad():
            action = ep39_policy.get_action(torch.FloatTensor(obs_normed), deterministic=True)
        # ep39 returns mapped action [0,60] x [0.8,1.2], need to convert BACK to raw tanh
        raw = np.array([(action[0] - 30.0) / 30.0, (action[1] - 1.0) / 0.2], dtype=np.float32)
        return np.clip(raw, -1.0, 1.0)
    r, n, t = run_episode(bridge, line_idx_map, bus_idx_map, ep39_fn, "ep39")
    print(f"  ep39: reward={r:.0f}, decisions={n}, time={t:.1f}s")
    results["ep39"] = r

    # ── 3. Zero-hold baseline ──
    print("\n[3/3] Zero-hold baseline...")
    def zero_fn(ev, obs, bid, last_act):
        return np.array([-1.0, 0.0], dtype=np.float32)  # hold=0, speed=1.0
    r, n, t = run_episode(bridge, line_idx_map, bus_idx_map, zero_fn, "zero")
    print(f"  zero: reward={r:.0f}, decisions={n}, time={t:.1f}s")
    results["zero"] = r

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMO EVALUATION RESULTS")
    print(f"{'='*70}")
    for name, rew in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " ← best" if rew == max(results.values()) else ""
        print(f"  {name:12s}: {rew:12.0f}{marker}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
