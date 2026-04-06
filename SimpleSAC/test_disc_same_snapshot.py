#!/usr/bin/env python3
"""
test_disc_same_snapshot.py
==========================
Discriminator sanity test for H2O+.

Goal: Verify that when the SAME snapshot is used as the starting state
      in both SUMO (offline, "real") and BusSimEnv ("sim"), the
      ZOnlyDiscriminator can learn to distinguish the two z-feature
      distributions over training steps.

Method:
    Phase 1 — Collect z pairs:
        a) Sample N offline transitions from merged_all_v2.h5.
           Each gives (z_t, z_t1) from SUMO.       → "real" set
        b) For each sampled snapshot, inject it into BusSimEnv via
           god-mode reset, rollout H steps with a random policy,
           and extract (z_t, z_t1) from the sim.    → "sim" set
    Phase 2 — Train discriminator:
        Train ZOnlyDiscriminator on the collected real/sim z pairs
        for T steps, logging accuracy every E steps.

Expected: discriminator accuracy should increase from ~50% to >70%
          if the z features genuinely differ between SUMO and sim.

Usage:
    cd H2Oplus/SimpleSAC
    python test_disc_same_snapshot.py [--n_snapshots 50] [--train_steps 2000]
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
    parser = argparse.ArgumentParser(description="Discriminator sanity test")
    parser.add_argument("--merged_file",
                        default=os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5"))
    parser.add_argument("--manifest_file",
                        default=os.path.join(_BUS_H2O, "datasets_v2", "file_manifest.json"))
    parser.add_argument("--sim_env_path",
                        default=os.path.join(_BUS_H2O, "calibrated_env"))
    parser.add_argument("--edge_xml",
                        default=os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml"))
    parser.add_argument("--line_id", default="7X")
    parser.add_argument("--n_snapshots", type=int, default=100,
                        help="Number of snapshots to use for z collection")
    parser.add_argument("--sim_rollout_steps", type=int, default=20,
                        help="Sim env steps after snapshot injection")
    parser.add_argument("--train_steps", type=int, default=3000,
                        help="Number of discriminator training steps")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate accuracy every N steps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_real_z(merged_file, manifest_file, n_snapshots, seed=42):
    """
    Sample offline (SUMO) z pairs from the merged file.
    Returns: dict with 'z_t' and 'z_t1' as numpy arrays [N, 30].
    """
    rng = np.random.RandomState(seed)

    print(f"[collect_real_z] Loading merged file: {merged_file}")
    with h5py.File(merged_file, "r") as f:
        total_n = f["rewards"].shape[0]
        # Sample random indices
        indices = rng.choice(total_n, size=n_snapshots, replace=False)
        indices.sort()

        z_t = np.array(f["z_t"])[indices]
        z_t1 = np.array(f["z_t1"])[indices]

    print(f"[collect_real_z] Collected {n_snapshots} real z pairs "
          f"(shape: z_t={z_t.shape}, z_t1={z_t1.shape})")
    return {"z_t": z_t, "z_t1": z_t1, "indices": indices}


def collect_sim_z(merged_file, manifest_file, sim_env_path,
                  real_indices, n_snapshots, sim_rollout_steps, seed=42):
    """
    For each snapshot index, inject into BusSimEnv and rollout to get sim z pairs.
    Returns: dict with 'z_t' and 'z_t1' as numpy arrays [M, 30].
    """
    from envs.bus_sim_env import MultiLineSimEnv

    rng = np.random.RandomState(seed)

    # Load snapshot store
    with open(manifest_file) as mf:
        file_manifest = json.load(mf)

    archive_dir = os.path.dirname(manifest_file)
    snap_store = SnapshotStore(
        archive_dir=archive_dir,
        file_manifest=file_manifest,
        cache_size=256,
        snapshot_key="snapshot_T1",
    )

    # Load snap indices from merged file
    with h5py.File(merged_file, "r") as f:
        snap_file_ids = np.array(f["snap_file_id"])
        snap_row_ids = np.array(f["snap_row_id"])

    # Create sim env
    print(f"[collect_sim_z] Creating MultiLineSimEnv from {sim_env_path}")
    sim_env = MultiLineSimEnv(path=sim_env_path, debug=False)

    sim_z_t_list = []
    sim_z_t1_list = []

    n_success = 0
    n_fail = 0

    for i, idx in enumerate(real_indices[:n_snapshots]):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n_snapshots}] Processing snapshot at buffer idx={idx}...")

        # Load the snapshot from the original HDF5 via lazy store
        file_id = int(snap_file_ids[idx])
        row_id = int(snap_row_ids[idx])

        try:
            snap_bytes = snap_store.get(file_id, row_id)
            snapshot_dict = pickle.loads(snap_bytes)
        except Exception as e:
            print(f"  WARNING: Failed to load snapshot at idx={idx}: {e}")
            n_fail += 1
            continue

        # Extract z_t from the snapshot BEFORE sim rollout
        z_t_init = extract_structured_context(snapshot_dict)

        # Inject into sim env
        try:
            sim_env.reset(snapshot=snapshot_dict)
        except Exception as e:
            print(f"  WARNING: Failed to reset sim env with snapshot at idx={idx}: {e}")
            n_fail += 1
            continue

        # Rollout with random actions for some steps
        z_after = None
        for step in range(sim_rollout_steps):
            # Random actions for all buses
            action_dict = {}
            for bus_id in range(sim_env.max_agent_num):
                hold = float(rng.uniform(0, 60))
                speed = float(rng.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
                action_dict[bus_id] = [hold, speed]

            try:
                state, reward, done = sim_env.step_to_event(action_dict)
            except Exception:
                break

            if done:
                break

            # Capture snapshot and extract z
            try:
                snap_after = sim_env.capture_full_system_snapshot()
                z_after = extract_structured_context(snap_after)
            except Exception:
                pass

        if z_after is not None:
            sim_z_t_list.append(z_t_init)
            sim_z_t1_list.append(z_after)
            n_success += 1
        else:
            n_fail += 1

    snap_store.close()

    if sim_z_t_list:
        sim_z_t = np.stack(sim_z_t_list)
        sim_z_t1 = np.stack(sim_z_t1_list)
    else:
        sim_z_t = np.zeros((0, 30), dtype=np.float32)
        sim_z_t1 = np.zeros((0, 30), dtype=np.float32)

    print(f"[collect_sim_z] Collected {n_success} sim z pairs "
          f"(failed: {n_fail}, shape: {sim_z_t.shape})")
    return {"z_t": sim_z_t, "z_t1": sim_z_t1}


def train_discriminator(real_data, sim_data, args):
    """
    Train ZOnlyDiscriminator and track accuracy over training.
    """
    device = torch.device(args.device)

    # Prepare tensors
    real_z_t = torch.FloatTensor(real_data["z_t"]).to(device)
    real_z_t1 = torch.FloatTensor(real_data["z_t1"]).to(device)
    sim_z_t = torch.FloatTensor(sim_data["z_t"]).to(device)
    sim_z_t1 = torch.FloatTensor(sim_data["z_t1"]).to(device)

    n_real = real_z_t.shape[0]
    n_sim = sim_z_t.shape[0]
    n_total = min(n_real, n_sim)

    if n_total < 10:
        print("ERROR: Not enough data pairs to train discriminator!")
        return

    # Use matching sizes
    real_z_t = real_z_t[:n_total]
    real_z_t1 = real_z_t1[:n_total]
    sim_z_t = sim_z_t[:n_total]
    sim_z_t1 = sim_z_t1[:n_total]

    print(f"\n{'='*60}")
    print(f"Training discriminator on {n_total} real + {n_total} sim z pairs")
    print(f"{'='*60}")

    # Split into train/eval (80/20)
    n_train = int(0.8 * n_total)
    n_eval = n_total - n_train
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    # Create discriminator
    disc = ZOnlyDiscriminator(context_dim=30).to(device)
    optimizer = torch.optim.Adam(disc.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Label smoothing
    LABEL_REAL = 0.9
    LABEL_SIM = 0.1

    print(f"\n{'Step':>6s}  {'Loss':>8s}  {'R_acc':>6s}  {'S_acc':>6s}  "
          f"{'Avg':>6s}  {'R_logit':>8s}  {'S_logit':>8s}")
    print("-" * 60)

    history = []

    for step in range(1, args.train_steps + 1):
        disc.train()

        # Sample mini-batch from train set
        batch_idx = train_idx[
            torch.randint(0, n_train, (args.batch_size,))
        ]

        # Real batch
        r_zt = real_z_t[batch_idx]
        r_zt1 = real_z_t1[batch_idx]
        r_logits = disc(r_zt, r_zt1)
        r_labels = torch.full_like(r_logits, LABEL_REAL)
        loss_real = criterion(r_logits, r_labels)

        # Sim batch
        s_zt = sim_z_t[batch_idx]
        s_zt1 = sim_z_t1[batch_idx]
        s_logits = disc(s_zt, s_zt1)
        s_labels = torch.full_like(s_logits, LABEL_SIM)
        loss_sim = criterion(s_logits, s_labels)

        loss = loss_real + loss_sim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        if step % args.eval_every == 0 or step == 1:
            disc.eval()
            with torch.no_grad():
                # Eval on held-out set
                e_real_logits = disc(real_z_t[eval_idx], real_z_t1[eval_idx])
                e_sim_logits = disc(sim_z_t[eval_idx], sim_z_t1[eval_idx])

                real_acc = (torch.sigmoid(e_real_logits) > 0.5).float().mean().item()
                sim_acc = (torch.sigmoid(e_sim_logits) < 0.5).float().mean().item()
                avg_acc = (real_acc + sim_acc) / 2

                r_mean_logit = e_real_logits.mean().item()
                s_mean_logit = e_sim_logits.mean().item()

            print(f"{step:6d}  {loss.item():8.4f}  {real_acc:6.1%}  {sim_acc:6.1%}  "
                  f"{avg_acc:6.1%}  {r_mean_logit:8.3f}  {s_mean_logit:8.3f}")

            history.append({
                "step": step,
                "loss": loss.item(),
                "real_acc": real_acc,
                "sim_acc": sim_acc,
                "avg_acc": avg_acc,
                "real_logit": r_mean_logit,
                "sim_logit": s_mean_logit,
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if history:
        first = history[0]
        last = history[-1]
        print(f"  Initial accuracy:  {first['avg_acc']:.1%} "
              f"(real={first['real_acc']:.1%}, sim={first['sim_acc']:.1%})")
        print(f"  Final accuracy:    {last['avg_acc']:.1%} "
              f"(real={last['real_acc']:.1%}, sim={last['sim_acc']:.1%})")
        print(f"  Accuracy improved: {last['avg_acc'] > first['avg_acc'] + 0.05}")

        # Z-feature statistics
        print(f"\n  Real z stats:")
        print(f"    z_t  mean={real_z_t.mean().item():.4f}, "
              f"std={real_z_t.std().item():.4f}, "
              f"min={real_z_t.min().item():.4f}, max={real_z_t.max().item():.4f}")
        print(f"    z_t1 mean={real_z_t1.mean().item():.4f}, "
              f"std={real_z_t1.std().item():.4f}")
        print(f"  Sim z stats:")
        print(f"    z_t  mean={sim_z_t.mean().item():.4f}, "
              f"std={sim_z_t.std().item():.4f}, "
              f"min={sim_z_t.min().item():.4f}, max={sim_z_t.max().item():.4f}")
        print(f"    z_t1 mean={sim_z_t1.mean().item():.4f}, "
              f"std={sim_z_t1.std().item():.4f}")

        # Compute L2 distance between real and sim z distributions
        z_diff_t = (real_z_t - sim_z_t).norm(dim=1).mean().item()
        z_diff_t1 = (real_z_t1 - sim_z_t1).norm(dim=1).mean().item()
        print(f"\n  Mean L2 distance (real vs sim):")
        print(f"    z_t:  {z_diff_t:.4f}")
        print(f"    z_t1: {z_diff_t1:.4f}")

    return history


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set route length
    if os.path.exists(args.edge_xml):
        edge_map = build_edge_linear_map(args.edge_xml, args.line_id)
        route_length = max(edge_map.values()) if edge_map else 13119.0
    else:
        route_length = 13119.0
    set_route_length(route_length)
    print(f"[main] Route length set to {route_length:.1f} m")

    # Phase 1a: Collect real (SUMO) z pairs
    t0 = time.time()
    real_data = collect_real_z(
        args.merged_file, args.manifest_file,
        args.n_snapshots, seed=args.seed,
    )

    # Phase 1b: Collect sim z pairs (snapshot injection + rollout)
    sim_data = collect_sim_z(
        args.merged_file, args.manifest_file, args.sim_env_path,
        real_data["indices"], args.n_snapshots, args.sim_rollout_steps,
        seed=args.seed,
    )
    t_collect = time.time() - t0
    print(f"\n[main] Data collection took {t_collect:.1f}s")

    # Phase 2: Train discriminator
    t1 = time.time()
    history = train_discriminator(real_data, sim_data, args)
    t_train = time.time() - t1
    print(f"\n[main] Training took {t_train:.1f}s")
    print(f"[main] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
