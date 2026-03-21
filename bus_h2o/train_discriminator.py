"""
train_discriminator.py
======================
Phase 3 — Train ZOnlyDiscriminator to separate SUMO (real) from BusSimEnv (sim).

Pipeline:
  1. Load SUMO offline data (z_t, z_t+1) from datasets/sumo_offline.h5
  2. Collect BusSimEnv (sim) z_t, z_t+1 by rolling out with zero-hold policy
  3. Train ZOnlyDiscriminator: real → label 0.9, sim → label 0.1
  4. Compute per-transition importance weight w = D/(1-D) for real transitions
  5. Plot: training loss, w over time, real vs sim z-space PCA projection
  6. Save trained model → models/discriminator.pt

Usage
-----
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o

    # Train on pre-collected SUMO data:
    python train_discriminator.py \\
        --real_data    datasets/sumo_offline.h5 \\
        --n_sim_events 500 \\
        --epochs       200 \\
        --out_model    models/discriminator.pt

    # Smoke-test without SUMO data (generates mock real data):
    python train_discriminator.py --mock --n_sim_events 200 --epochs 50
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common.data_utils import (
    ZOnlyDiscriminator,
    compute_z_importance_weight,
    extract_structured_context,
    build_edge_linear_map,
    set_route_length,
)
from envs.bus_sim_env import BusSimEnv
from sumo_env.sumo_snapshot import make_mock_snapshot

EDGE_XML  = os.path.join(_HERE, "network_data", "a_sorted_busline_edge.xml")
CALIB_DIR = os.path.join(_HERE, "calibrated_env")


# ── sim rollout helpers ──────────────────────────────────────────────────

def collect_sim_transitions(env: BusSimEnv, n_events: int) -> np.ndarray:
    """Roll BusSimEnv with zero-hold policy, collect z_t pairs. Returns (N, 60) array."""
    env.reset()
    actions = {k: 0.0 for k in range(env.max_agent_num)}
    pairs = []
    prev_z = None

    for _ in range(n_events * 500):   # max steps budget
        obs, rew, done, info = env.step(actions)
        has_event = any(len(v) > 0 for v in obs.values())
        if has_event:
            snap = info.get("snapshot")
            if snap is not None:
                z = extract_structured_context(snap)
                if prev_z is not None:
                    pairs.append(np.concatenate([prev_z, z]))   # (60,)
                    if len(pairs) >= n_events:
                        break
                prev_z = z
        if done:
            env.reset()
            prev_z = None

    return np.stack(pairs, axis=0).astype(np.float32) if pairs else np.zeros((1, 60), np.float32)


def collect_mock_real_transitions(n: int, route_len: float) -> np.ndarray:
    """Generate synthetic 'real' pairs for smoke testing without SUMO."""
    pairs = []
    for i in range(n):
        s1 = make_mock_snapshot(sim_time=float(i * 30), route_length=route_len)
        s2 = make_mock_snapshot(sim_time=float(i * 30 + 2), route_length=route_len)
        z1 = extract_structured_context(s1)
        z2 = extract_structured_context(s2)
        pairs.append(np.concatenate([z1, z2]))
    return np.stack(pairs, axis=0).astype(np.float32)


# ── training ─────────────────────────────────────────────────────────────

def train(real_pairs: np.ndarray, sim_pairs: np.ndarray,
          epochs: int, lr: float, batch_size: int,
          device: str) -> tuple[ZOnlyDiscriminator, list[float], list[float]]:
    """Train ZOnlyDiscriminator, return (model, loss_history, w_history)."""
    D = ZOnlyDiscriminator(context_dim=30, hidden_dim=256, n_hidden=2).to(device)
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    R = torch.tensor(real_pairs, dtype=torch.float32, device=device)
    S = torch.tensor(sim_pairs,  dtype=torch.float32, device=device)

    n_real = len(R); n_sim = len(S)
    loss_hist = []
    w_hist    = []

    for ep in range(epochs):
        # ── mini-batch sampling ──────────────────────────────────────
        idx_r = torch.randint(0, n_real, (batch_size,), device=device)
        idx_s = torch.randint(0, n_sim,  (batch_size,), device=device)

        r_batch = R[idx_r]    # (B, 60)
        s_batch = S[idx_s]    # (B, 60)

        # Add 5% Gaussian noise to real samples (augmentation)
        r_noisy = r_batch + 0.05 * torch.randn_like(r_batch)

        zt_r  = r_noisy[:, :30];  zt1_r = r_noisy[:, 30:]
        zt_s  = s_batch[:, :30];  zt1_s = s_batch[:, 30:]

        lbl_r = torch.full((batch_size, 1), 0.9, device=device)
        lbl_s = torch.full((batch_size, 1), 0.1, device=device)

        logit_r = D(zt_r,  zt1_r)
        logit_s = D(zt_s,  zt1_s)

        loss = crit(logit_r, lbl_r) + crit(logit_s, lbl_s)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt.step()

        loss_hist.append(loss.item())

        # Compute mean w on real data every 10 epochs
        if ep % 10 == 0:
            w = compute_z_importance_weight(D, R[:, :30], R[:, 30:])
            w_hist.append(float(w.mean().item()))
            print(f"  epoch {ep:4d} | loss={loss.item():.4f} | w_real_mean={w_hist[-1]:.3f}")

    return D, loss_hist, w_hist


# ── main ─────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Set route length from edge map
    edge_map = build_edge_linear_map(EDGE_XML, "7X")
    route_len = max(edge_map.values()) if edge_map else 13119.0
    set_route_length(route_len)

    # ── 1. Load / generate real transitions ─────────────────────────
    if args.mock:
        print(f"\n[1] Generating {args.n_real_events} mock real transitions ...")
        real_pairs = collect_mock_real_transitions(args.n_real_events, route_len)
    else:
        try:
            import h5py
        except ImportError:
            print("ERROR: h5py required. pip install h5py"); sys.exit(1)

        print(f"\n[1] Loading real transitions from {args.real_data} ...")
        with h5py.File(args.real_data, "r") as f:
            z_t  = f["z_t"][:]   # (N, 30)
            z_t1 = f["z_t1"][:]  # (N, 30)
        real_pairs = np.concatenate([z_t, z_t1], axis=1).astype(np.float32)

    print(f"    Real pairs: {real_pairs.shape}")

    # ── 2. Collect sim transitions ────────────────────────────────────
    print(f"\n[2] Collecting {args.n_sim_events} sim transitions from BusSimEnv ...")
    sim_env = BusSimEnv(path=CALIB_DIR)
    sim_pairs = collect_sim_transitions(sim_env, args.n_sim_events)
    print(f"    Sim pairs:  {sim_pairs.shape}")

    # ── 3. Train discriminator ────────────────────────────────────────
    print(f"\n[3] Training ZOnlyDiscriminator for {args.epochs} epochs ...")
    D, loss_hist, w_hist = train(
        real_pairs, sim_pairs,
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        device     = device,
    )

    # ── 4. Importance weights on real data ───────────────────────────
    R = torch.tensor(real_pairs, device=device)
    w_real = compute_z_importance_weight(D, R[:, :30], R[:, 30:]).cpu().numpy().squeeze()
    print(f"\n    w_real: mean={w_real.mean():.3f}, median={np.median(w_real):.3f}, "
          f"p10={np.percentile(w_real,10):.3f}, p90={np.percentile(w_real,90):.3f}")

    S = torch.tensor(sim_pairs, device=device)
    w_sim  = compute_z_importance_weight(D, S[:, :30], S[:, 30:]).cpu().numpy().squeeze()
    print(f"    w_sim:  mean={w_sim.mean():.3f}, median={np.median(w_sim):.3f}")

    # ── 5. Save model ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_model)), exist_ok=True)
    torch.save({
        "state_dict"  : D.state_dict(),
        "context_dim" : 30,
        "hidden_dim"  : 256,
        "n_hidden"    : 2,
        "route_len"   : route_len,
    }, args.out_model)
    print(f"\n    Model saved → {args.out_model}")

    # ── 6. Plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(loss_hist, "b-", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Discriminator training loss"); axes[0].grid(True, alpha=0.3)

    # w_real over training
    axes[1].plot(range(0, args.epochs, 10), w_hist, "g-o", lw=2, ms=4)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean w (real)")
    axes[1].set_title("Importance weight (real) over training"); axes[1].grid(True, alpha=0.3)

    # w distribution
    n_plot = min(len(w_real), len(w_sim), 300)
    axes[2].hist(w_real[:n_plot],  bins=30, alpha=0.6, color="steelblue", label="real (SUMO)")
    axes[2].hist(w_sim[:n_plot],   bins=30, alpha=0.6, color="tomato",    label="sim (BusSimEnv)")
    axes[2].set_xlabel("Importance weight w"); axes[2].set_ylabel("Count")
    axes[2].set_title("w distribution: real vs sim"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(_HERE, "discriminator_plot.png")
    plt.savefig(plot_path, dpi=120)
    print(f"    Plot saved → {plot_path}")

    # ── 7. Assertion ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    w_real_med = float(np.median(w_real))
    w_sim_med  = float(np.median(w_sim))
    passed = w_real_med > w_sim_med
    print(f"  w_real median = {w_real_med:.3f}")
    print(f"  w_sim  median = {w_sim_med:.3f}")
    print(f"  Assertion (w_real > w_sim): {'PASS ✅' if passed else 'FAIL ⚠️'}")
    if not passed and not args.mock:
        print("  Hint: need more SUMO data or more epochs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train H2O+ ZOnlyDiscriminator")
    parser.add_argument("--real_data",    type=str,   default="datasets/sumo_offline.h5",
                        help="Path to SUMO offline HDF5 (ignored with --mock)")
    parser.add_argument("--n_sim_events", type=int,   default=500,
                        help="Number of BusSimEnv events to collect (default: 500)")
    parser.add_argument("--n_real_events",type=int,   default=500,
                        help="Number of mock real events (only with --mock)")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--out_model",    type=str,   default="models/discriminator.pt")
    parser.add_argument("--mock",         action="store_true",
                        help="Use mock (randomly generated) real data — no SUMO needed")
    args = parser.parse_args()
    main(args)
