#!/usr/bin/env python3
"""
test_disc_divergence.py
========================
Discriminator divergence test for H2O+.

Goal: From a shared starting snapshot, observe how the discriminator's
      ability to distinguish SUMO vs Sim grows over successive bus‑arrival
      events (i.e. over time since reset).

Method:
    Phase 1 — Train discriminator:
        Collect (z_t, z_t1) pairs from both SUMO offline data and SimBus
        at steady state (sim_time 5000-15000s). Train ZOnlyDiscriminator.

    Phase 2 — Divergence tracking:
        For each of N_TRIALS starting snapshots:
          a) SUMO side:  Read subsequent z pairs from offline data
                         chronologically after the snapshot time.
          b) Sim side:   Inject snapshot into MultiLineSimEnv, rollout
                         with zero-hold policy, extract z at each bus
                         arrival (decision event).
          c) Score each event's (z_t, z_t1) with the trained discriminator.
          d) Track D(z_t, z_t1) over event‑steps since reset.

Usage:
    cd H2Oplus/SimpleSAC
    python test_disc_divergence.py [--n_trials 10] [--max_events 100]
"""

import argparse
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from common.data_utils import (
    ZOnlyDiscriminator,
    extract_structured_context,
    set_route_length,
    build_edge_linear_map,
)
from snapshot_store import SnapshotStore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_file",
                   default=os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5"))
    p.add_argument("--manifest_file",
                   default=os.path.join(_BUS_H2O, "datasets_v2", "file_manifest.json"))
    p.add_argument("--sim_env_path",
                   default=os.path.join(_BUS_H2O, "calibrated_env"))
    p.add_argument("--edge_xml",
                   default=os.path.join(_BUS_H2O, "network_data",
                                        "a_sorted_busline_edge.xml"))
    p.add_argument("--line_id", default="7X")
    # Phase 1: discriminator training
    p.add_argument("--disc_train_pairs", type=int, default=300,
                   help="Number of z pairs per source for disc training")
    p.add_argument("--disc_train_steps", type=int, default=3000)
    # Phase 2: divergence tracking
    p.add_argument("--n_trials", type=int, default=5,
                   help="Number of snapshot starting points to test")
    p.add_argument("--max_events", type=int, default=80,
                   help="Max bus arrival events per trial")
    p.add_argument("--sumo_lookahead", type=int, default=200,
                   help="Number of SUMO transitions to look ahead from snapshot")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =====================================================================
# Phase 1: Train discriminator
# =====================================================================

def train_discriminator(merged_file, sim_env_path, args):
    """Train a ZOnlyDiscriminator on steady-state SUMO vs Sim z pairs."""
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    N = args.disc_train_pairs

    # ── Collect SUMO z pairs (mid-episode, steady state) ──
    print("[Phase1] Loading SUMO z pairs from offline data...")
    with h5py.File(merged_file, "r") as f:
        sim_times = np.array(f["sim_time"])
        mask = (sim_times > 5000) & (sim_times < 15000)
        valid_idx = np.where(mask)[0]
        chosen = rng.choice(valid_idx, size=N, replace=False)
        real_zt = np.array(f["z_t"])[chosen]
        real_zt1 = np.array(f["z_t1"])[chosen]
    print(f"  SUMO: {N} pairs, z_t mean={real_zt.mean():.3f}")

    # ── Collect Sim z pairs (steady-state rollouts) ──
    print("[Phase1] Collecting Sim z pairs (steady-state rollouts)...")
    from envs.bus_sim_env import MultiLineSimEnv
    env = MultiLineSimEnv(path=sim_env_path, debug=False)

    sim_zt_list, sim_zt1_list = [], []
    n_eps = 5
    per_ep = (N + n_eps - 1) // n_eps

    for ep in range(n_eps):
        env.reset()
        # Warm up to ~5000s
        for _ in range(50000):
            full_a = {lid: {k: 0.0 for k in range(env.line_map[lid].max_agent_num)}
                      for lid in env.line_map}
            _, _, done = type(env).__bases__[0].step(env, full_a)
            if done or env.current_time > 5000:
                break

        snap_prev = env.capture_full_system_snapshot()
        z_prev = extract_structured_context(snap_prev)
        collected = 0

        for _ in range(50000):
            full_a = {lid: {k: 0.0 for k in range(env.line_map[lid].max_agent_num)}
                      for lid in env.line_map}
            _, _, done = type(env).__bases__[0].step(env, full_a)
            if done:
                break
            if rng.random() < 0.01:
                snap_now = env.capture_full_system_snapshot()
                z_now = extract_structured_context(snap_now)
                sim_zt_list.append(z_prev.copy())
                sim_zt1_list.append(z_now.copy())
                z_prev = z_now
                collected += 1
                if collected >= per_ep:
                    break
        print(f"  Sim ep {ep+1}: {collected} pairs, t={env.current_time:.0f}s")

    sim_zt = np.stack(sim_zt_list[:N])
    sim_zt1 = np.stack(sim_zt1_list[:N])
    n_use = min(len(real_zt), len(sim_zt))
    print(f"  Using {n_use} pairs per source")

    # ── Train ──
    disc = ZOnlyDiscriminator(context_dim=30).to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=3e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    r_zt = torch.FloatTensor(real_zt[:n_use]).to(device)
    r_zt1 = torch.FloatTensor(real_zt1[:n_use]).to(device)
    s_zt = torch.FloatTensor(sim_zt[:n_use]).to(device)
    s_zt1 = torch.FloatTensor(sim_zt1[:n_use]).to(device)

    n_train = int(0.8 * n_use)
    perm = torch.randperm(n_use)
    tr_idx, ev_idx = perm[:n_train], perm[n_train:]

    print(f"\n[Phase1] Training discriminator ({args.disc_train_steps} steps)...")
    for step in range(1, args.disc_train_steps + 1):
        disc.train()
        bi = tr_idx[torch.randint(0, n_train, (64,))]
        rl = disc(r_zt[bi], r_zt1[bi])
        sl = disc(s_zt[bi], s_zt1[bi])
        loss = (criterion(rl, torch.full_like(rl, 0.9)) +
                criterion(sl, torch.full_like(sl, 0.1)))
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 0 or step == 1:
            disc.eval()
            with torch.no_grad():
                er = disc(r_zt[ev_idx], r_zt1[ev_idx])
                es = disc(s_zt[ev_idx], s_zt1[ev_idx])
                ra = (torch.sigmoid(er) > 0.5).float().mean().item()
                sa = (torch.sigmoid(es) < 0.5).float().mean().item()
            print(f"  step={step:5d} loss={loss.item():.4f} "
                  f"R_acc={ra:.1%} S_acc={sa:.1%}")

    disc.eval()
    print("[Phase1] Discriminator trained.\n")
    return disc, env


# =====================================================================
# Phase 2: Divergence tracking
# =====================================================================

def run_divergence_trials(disc, env, merged_file, manifest_file, args):
    """
    For each trial snapshot:
      - SUMO: read subsequent z_t, z_t1 pairs from offline data
      - Sim:  inject snapshot, rollout, extract z at each decision event
      - Score each with disc
    """
    rng = np.random.RandomState(args.seed + 100)
    device = torch.device(args.device)

    with open(manifest_file) as mf:
        file_manifest = json.load(mf)
    snap_store = SnapshotStore(os.path.dirname(manifest_file), file_manifest)

    # Load metadata from merged file
    with h5py.File(merged_file, "r") as f:
        all_sim_times = np.array(f["sim_time"])
        all_sfids = np.array(f["snap_file_id"])
        all_srids = np.array(f["snap_row_id"])
        all_zt = np.array(f["z_t"])
        all_zt1 = np.array(f["z_t1"])

    # Pick starting points: mid-episode (sim_time 4000-8000s)
    mask = (all_sim_times > 4000) & (all_sim_times < 8000)
    valid = np.where(mask)[0]
    trial_indices = rng.choice(valid, size=args.n_trials, replace=False)

    all_sumo_scores = []   # [trial][event_step] = D logit
    all_sim_scores = []

    for trial_i, start_idx in enumerate(trial_indices):
        fid = int(all_sfids[start_idx])
        rid = int(all_srids[start_idx])
        snap_t0 = all_sim_times[start_idx]

        print(f"\n{'='*60}")
        print(f"Trial {trial_i+1}/{args.n_trials}: "
              f"idx={start_idx}, file={file_manifest[fid][0]}, "
              f"t0={snap_t0:.0f}s")

        # ── SUMO side: subsequent z pairs from offline data ──
        # Find transitions in the same source file after sim_time t0
        file_mask = (all_sfids == fid) & (all_sim_times >= snap_t0)
        follow_indices = np.where(file_mask)[0]
        # Sort by sim_time
        sorted_by_time = follow_indices[np.argsort(all_sim_times[follow_indices])]
        sumo_indices = sorted_by_time[:args.sumo_lookahead]

        sumo_scores_trial = []
        for si in sumo_indices:
            zt_t = torch.FloatTensor(all_zt[si:si+1]).to(device)
            zt1_t = torch.FloatTensor(all_zt1[si:si+1]).to(device)
            with torch.no_grad():
                logit = disc(zt_t, zt1_t).item()
            sumo_scores_trial.append({
                "event_step": len(sumo_scores_trial),
                "sim_time": float(all_sim_times[si]),
                "logit": logit,
                "prob_real": float(torch.sigmoid(torch.tensor(logit)).item()),
            })

        print(f"  SUMO: {len(sumo_scores_trial)} events")
        all_sumo_scores.append(sumo_scores_trial)

        # ── Sim side: inject snapshot, rollout ──
        snap_bytes = snap_store.get(fid, rid)
        snap_dict = pickle.loads(snap_bytes)

        # Inject and rollout
        env.reset(snapshot=snap_dict)
        z_prev = extract_structured_context(snap_dict)

        sim_scores_trial = []
        for event_step in range(args.max_events):
            # Step until a 7X bus arrives at a station (decision event)
            action_dict = {k: 0.0 for k in range(env.max_agent_num)}
            try:
                state, reward, done = env.step_to_event(action_dict)
            except Exception as e:
                print(f"  Sim error at event {event_step}: {e}")
                break
            if done:
                break

            # Capture z after event
            snap_now = env.capture_full_system_snapshot()
            z_now = extract_structured_context(snap_now)

            zt_t = torch.FloatTensor(z_prev[None]).to(device)
            zt1_t = torch.FloatTensor(z_now[None]).to(device)
            with torch.no_grad():
                logit = disc(zt_t, zt1_t).item()

            sim_scores_trial.append({
                "event_step": event_step,
                "sim_time": float(env.current_time),
                "logit": logit,
                "prob_real": float(torch.sigmoid(torch.tensor(logit)).item()),
            })
            z_prev = z_now

        print(f"  Sim:  {len(sim_scores_trial)} events, "
              f"final_t={env.current_time:.0f}s")
        all_sim_scores.append(sim_scores_trial)

        # ── Print per-trial summary ──
        if sumo_scores_trial and sim_scores_trial:
            n_show = min(15, len(sumo_scores_trial), len(sim_scores_trial))
            print(f"\n  {'Step':>4s}  {'SUMO_t':>7s}  {'SUMO_D':>7s}  "
                  f"{'SUMO_P':>7s}  {'Sim_t':>7s}  {'Sim_D':>7s}  {'Sim_P':>7s}")
            print(f"  {'-'*55}")
            for i in range(n_show):
                s = sumo_scores_trial[i]
                m = sim_scores_trial[i]
                print(f"  {i:4d}  {s['sim_time']:7.0f}  {s['logit']:7.3f}  "
                      f"{s['prob_real']:7.3f}  {m['sim_time']:7.0f}  "
                      f"{m['logit']:7.3f}  {m['prob_real']:7.3f}")

    snap_store.close()

    # ── Global summary ──
    print(f"\n{'='*60}")
    print("GLOBAL SUMMARY")
    print(f"{'='*60}")

    # Average D score at each event step across trials
    max_steps = max(
        max((len(t) for t in all_sumo_scores), default=0),
        max((len(t) for t in all_sim_scores), default=0),
    )
    print(f"\n  {'Step':>4s}  {'SUMO_D_mean':>11s}  {'SUMO_P_mean':>11s}  "
          f"{'Sim_D_mean':>10s}  {'Sim_P_mean':>10s}  {'Gap':>6s}")
    print(f"  {'-'*60}")

    for step in range(min(max_steps, 20)):
        sumo_logits = [t[step]["logit"] for t in all_sumo_scores if step < len(t)]
        sim_logits = [t[step]["logit"] for t in all_sim_scores if step < len(t)]
        if sumo_logits and sim_logits:
            s_mean = np.mean(sumo_logits)
            m_mean = np.mean(sim_logits)
            s_prob = float(torch.sigmoid(torch.tensor(s_mean)).item())
            m_prob = float(torch.sigmoid(torch.tensor(m_mean)).item())
            gap = s_prob - m_prob
            print(f"  {step:4d}  {s_mean:11.3f}  {s_prob:11.3f}  "
                  f"{m_mean:10.3f}  {m_prob:10.3f}  {gap:+6.3f}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Route length
    if os.path.exists(args.edge_xml):
        edge_map = build_edge_linear_map(args.edge_xml, args.line_id)
        route_length = max(edge_map.values()) if edge_map else 13119.0
    else:
        route_length = 13119.0
    set_route_length(route_length)
    print(f"Route length: {route_length:.1f} m")

    # Phase 1: Train discriminator + create sim env
    t0 = time.time()
    disc, env = train_discriminator(args.merged_file, args.sim_env_path, args)
    print(f"Phase 1 took {time.time()-t0:.1f}s")

    # Phase 2: Divergence tracking
    t1 = time.time()
    run_divergence_trials(disc, env, args.merged_file, args.manifest_file, args)
    print(f"\nPhase 2 took {time.time()-t1:.1f}s")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
