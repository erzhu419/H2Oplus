"""
quick_disc_test.py — Quick Discriminator Alignment Verification
================================================================
Collects z samples from SUMO (real) and MultiLineSimEnv (sim),
trains a small binary classifier, and reports accuracy.

If accuracy ≈ 50%: alignment is good (can't distinguish).
If accuracy ≈ 100%: still misaligned.

Usage:
    python quick_disc_test.py
"""

import os
import sys
import time
import pickle
import numpy as np

# ── Path setup ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUS_H2O = os.path.join(_HERE, "bus_h2o")
sys.path.insert(0, _BUS_H2O)

SUMO_DIR = os.path.normpath(os.path.join(
    _HERE, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
sys.path.insert(0, os.path.join(SUMO_DIR, "initialize_obj"))
_CASE_DIR = os.path.join(_BUS_H2O, "sumo_env", "case")
if os.path.isdir(_CASE_DIR):
    sys.path.insert(0, _CASE_DIR)

import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim

from sumo_env.rl_bridge import SumoRLBridge
from sumo_env.sumo_snapshot import bridge_to_snapshot
from common.data_utils import (
    build_edge_linear_map,
    extract_structured_context,
    set_route_length,
    ZOnlyDiscriminator,
)
from envs.bus_sim_env import MultiLineSimEnv


EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
SIM_ENV_DIR = os.path.join(_BUS_H2O, "calibrated_env")
LINE_IDS = ['7X', '7S', '102X', '102S', '122X', '122S',
            '311X', '311S', '406X', '406S', '705X', '705S']


def build_all_edge_maps():
    """Build per-line edge_maps and route_lengths for all 12 lines."""
    all_edge_maps = {}
    line_route_lengths = {}
    tree = ET.parse(EDGE_XML)
    root = tree.getroot()
    for bl in root.findall("busline"):
        lid = bl.get("id")
        all_edge_maps[lid] = build_edge_linear_map(EDGE_XML, lid)
        total_len = sum(float(e.get("length", 0)) for e in bl.findall("element"))
        line_route_lengths[lid] = total_len
    return all_edge_maps, line_route_lengths


def collect_sumo_z(max_time=4000, sample_every_sec=10):
    """Run SUMO for max_time seconds, sample z every sample_every_sec seconds.
    
    Uses zero-hold policy (no control intervention).
    """
    print(f"\n{'='*60}")
    print(f"  SUMO Collection: {max_time}s, sample every {sample_every_sec}s")
    print(f"{'='*60}")

    all_edge_maps, line_route_lengths = build_all_edge_maps()
    set_route_length(line_route_lengths.get('7X', 13298.0))

    print(f"  Edge maps: {len(all_edge_maps)} lines")
    print(f"  Route lengths: {len(line_route_lengths)} lines")

    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=max_time + 500)
    bridge.reset()

    z_samples = []
    t0 = time.time()
    last_sample_time = 0

    for iteration in range(200000):  # safety cap
        events, done, departed = bridge.fetch_events()

        if done:
            break

        cur_time = bridge.current_time
        if cur_time > max_time:
            break

        # Apply zero-hold to all events
        for ev in events:
            bridge.apply_action(ev, 0.0)  # zero hold

        # Sample z based on sim time
        if cur_time - last_sample_time >= sample_every_sec:
            snap = bridge_to_snapshot(
                bridge,
                all_edge_maps=all_edge_maps,
                line_route_lengths=line_route_lengths,
            )
            z = extract_structured_context(snap)
            z_samples.append(z)
            last_sample_time = cur_time

            if len(z_samples) % 50 == 0:
                n_buses = len(snap.get('all_buses', []))
                n_lines = len(set(b.get('line_id', '?')
                                  for b in snap.get('all_buses', [])))
                print(f"  t={cur_time:.0f}s: {n_buses} buses from "
                      f"{n_lines} lines, z_density={z[10:20]}")

    bridge.close()
    elapsed = time.time() - t0
    print(f"  SUMO done: {len(z_samples)} z samples in {elapsed:.1f}s, "
          f"final_time={cur_time:.0f}s")
    return np.array(z_samples, dtype=np.float32)


def collect_sim_z(n_resets=3, max_time_per_reset=4000, sample_every_sec=10):
    """Run MultiLineSimEnv with zero-hold, sample z every sample_every_sec seconds.
    
    Runs multiple resets, each for max_time_per_reset seconds of sim time.
    This ensures Sim covers the same temporal range as SUMO (0-4000s).
    """
    print(f"\n{'='*60}")
    print(f"  Sim Collection: {n_resets} resets × {max_time_per_reset}s, "
          f"sample every {sample_every_sec}s")
    print(f"{'='*60}")

    _, line_route_lengths = build_all_edge_maps()
    set_route_length(line_route_lengths.get('7X', 13298.0))

    env = MultiLineSimEnv(SIM_ENV_DIR)

    z_samples = []
    t0 = time.time()

    for r in range(n_resets):
        env.reset()

        sample_count = 0
        last_sample_time = 0
        for step in range(max_time_per_reset * 2):  # safety: 2x steps
            zero_actions = {lid: {} for lid in env.line_map}
            try:
                state, reward, done = env.step(zero_actions)
            except Exception:
                break
            if done:
                break

            cur_time = env.current_time
            if cur_time > max_time_per_reset:
                break

            # Sample z based on sim time
            if cur_time - last_sample_time >= sample_every_sec:
                snap = env.capture_full_system_snapshot()
                z = extract_structured_context(snap)
                z_samples.append(z)
                sample_count += 1
                last_sample_time = cur_time

        n_buses = sum(1 for lid, le in env.line_map.items()
                      for b in le.bus_all if b.on_route)
        n_lines = sum(1 for lid, le in env.line_map.items()
                      if any(b.on_route for b in le.bus_all))
        print(f"  reset {r+1}/{n_resets}: {sample_count} samples, "
              f"final_time={env.current_time:.0f}s, "
              f"{n_buses} buses from {n_lines} lines active")

    elapsed = time.time() - t0
    print(f"  Sim done: {len(z_samples)} z samples in {elapsed:.1f}s")
    return np.array(z_samples, dtype=np.float32)


def train_discriminator(z_real, z_sim, n_epochs=200, lr=3e-4):
    """Train a small discriminator and report accuracy."""
    print(f"\n{'='*60}")
    print(f"  Discriminator Test")
    print(f"{'='*60}")
    print(f"  Real (SUMO): {z_real.shape}")
    print(f"  Sim:         {z_sim.shape}")

    # Compare distributions
    print(f"\n  Feature distributions (mean ± std):")
    channels = ['speed', 'density', 'waiting']
    for ch_idx, ch_name in enumerate(channels):
        s = ch_idx * 10
        e = s + 10
        r_mean = z_real[:, s:e].mean()
        r_std = z_real[:, s:e].std()
        s_mean = z_sim[:, s:e].mean()
        s_std = z_sim[:, s:e].std()
        diff = abs(r_mean - s_mean)
        print(f"    {ch_name:>8s}: SUMO={r_mean:.4f}±{r_std:.4f}  "
              f"Sim={s_mean:.4f}±{s_std:.4f}  Δ={diff:.4f}")

    # Per-segment density comparison
    print(f"\n  Per-segment density (mean):")
    print(f"  {'seg':>4s} | {'SUMO':>8s} | {'Sim':>8s} | {'Δ':>6s}")
    print(f"  {'-'*35}")
    for i in range(10):
        rv = z_real[:, 10+i].mean()
        sv = z_sim[:, 10+i].mean()
        d = abs(rv - sv)
        bar_r = '▓' * int(rv * 10)
        bar_s = '░' * int(sv * 10)
        print(f"  {i:>4d} | {rv:8.4f} | {sv:8.4f} | {d:6.4f}  {bar_r}|{bar_s}")

    # Build dataset
    n_real = len(z_real)
    n_sim = len(z_sim)
    n_min = min(n_real, n_sim)

    # Balance classes
    idx_r = np.random.choice(n_real, n_min, replace=n_real < n_min)
    idx_s = np.random.choice(n_sim, n_min, replace=n_sim < n_min)

    X = np.concatenate([z_real[idx_r], z_sim[idx_s]], axis=0)
    y = np.concatenate([np.ones(n_min), np.zeros(n_min)]).astype(np.float32)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    # Simple 2-layer discriminator
    device = torch.device('cpu')
    input_dim = X_train.shape[1]
    disc = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    ).to(device)

    optimizer = optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.from_numpy(X_train).to(device)
    y_tr = torch.from_numpy(y_train).unsqueeze(1).to(device)
    X_te = torch.from_numpy(X_test).to(device)
    y_te = torch.from_numpy(y_test).unsqueeze(1).to(device)

    best_test_acc = 0
    for epoch in range(n_epochs):
        disc.train()
        logits = disc(X_tr)
        loss = criterion(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            disc.eval()
            with torch.no_grad():
                train_pred = (torch.sigmoid(disc(X_tr)) > 0.5).float()
                train_acc = (train_pred == y_tr).float().mean().item()
                test_pred = (torch.sigmoid(disc(X_te)) > 0.5).float()
                test_acc = (test_pred == y_te).float().mean().item()
                best_test_acc = max(best_test_acc, test_acc)
            print(f"  epoch {epoch+1:3d}: loss={loss.item():.4f}, "
                  f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    # Final evaluation
    disc.eval()
    with torch.no_grad():
        test_logits = disc(X_te)
        test_pred = (torch.sigmoid(test_logits) > 0.5).float()
        final_acc = (test_pred == y_te).float().mean().item()

        # Per-class accuracy
        real_mask = y_te.squeeze() == 1
        sim_mask = y_te.squeeze() == 0
        acc_real = (test_pred[real_mask] == y_te[real_mask]).float().mean().item() if real_mask.any() else 0
        acc_sim = (test_pred[sim_mask] == y_te[sim_mask]).float().mean().item() if sim_mask.any() else 0

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Test Accuracy:       {final_acc:.3f}")
    print(f"  Best Test Accuracy:  {best_test_acc:.3f}")
    print(f"  Acc (SUMO as real):  {acc_real:.3f}")
    print(f"  Acc (Sim as fake):   {acc_sim:.3f}")
    print()
    if best_test_acc < 0.60:
        print(f"  ✅ EXCELLENT: Discriminator cannot distinguish SUMO from Sim!")
        print(f"     z distributions are well aligned.")
    elif best_test_acc < 0.70:
        print(f"  ✅ GOOD: Discriminator has slight edge, but distributions are close.")
        print(f"     Acceptable for H2O+ training.")
    elif best_test_acc < 0.85:
        print(f"  ⚠️  MODERATE: Discriminator can partially distinguish.")
        print(f"     May need further tuning.")
    else:
        print(f"  ❌ POOR: Discriminator easily distinguishes SUMO from Sim!")
        print(f"     Alignment still needs work.")
    print(f"{'='*60}\n")

    return final_acc, best_test_acc


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Collect from SUMO ──
    z_sumo = collect_sumo_z(max_time=4000, sample_every_sec=10)

    # ── Collect from Sim ──
    z_sim = collect_sim_z(n_resets=3, max_time_per_reset=4000, sample_every_sec=10)

    # ── Test 1: full 30-dim z ──
    print("\n" + "█"*60)
    print("  TEST 1: Full 30-dim z (speed + density + waiting)")
    print("█"*60)
    if len(z_sumo) > 10 and len(z_sim) > 10:
        train_discriminator(z_sumo, z_sim, n_epochs=200)

    # ── Test 2: 20-dim z (speed + density only, drop waiting) ──
    print("\n" + "█"*60)
    print("  TEST 2: 20-dim z (speed + density only, zero waiting)")
    print("█"*60)
    z_sumo_20 = z_sumo[:, :20].copy()
    z_sim_20 = z_sim[:, :20].copy()
    if len(z_sumo_20) > 10 and len(z_sim_20) > 10:
        train_discriminator(z_sumo_20, z_sim_20, n_epochs=200)
